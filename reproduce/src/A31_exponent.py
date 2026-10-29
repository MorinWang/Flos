#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A31: separate the fractional exponent s from the smoothing strength alpha.

Runs FLoS exactly as in the main table (truncation T_r,
magnitude-adaptive shrinkage, p=1.25, 500 rounds, tol 1e-5, MP rank, float32 filter) with the background gain
h(mu) = 1/(1 + alpha mu^s) for every (s, alpha) on a grid.  Stage "dev": NCEP air, slp, rhum x seeds 0,1,2 on the
full grid; each s then freezes its best (alpha, lam) by the dev mean.  Stage "heldout": the 19 other variables x
seeds 0,1,2 at the frozen setting of each s (given on the command line as s:alpha:lam triples).
"""
import sys, os, time, csv, argparse
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L
import flos_torch as FT

DEVICE = device()
SYNC = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in
                                       ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in
                                       ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=list(SRC))
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--configs", default="", help="explicit s:alpha:lam triples, comma separated (held-out stage)")
ap.add_argument("--s_grid", default="0.25,0.5,1.0")
ap.add_argument("--alpha_grid", default="0.85,2.5,7.5,22.5")
ap.add_argument("--lams", default="0.36,0.72,1.44")
ap.add_argument("--shard", required=True)
ap.add_argument("--ablation", default=None, help="None or fixed-thresh (A38a: soft threshold at the development optimum)")
a = ap.parse_args()

L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
if a.configs:
    CFG = [tuple(float(v) for v in c.split(":")) for c in a.configs.split(",")]
else:
    CFG = [(s, al, lm) for s in map(float, a.s_grid.split(",")) for al in map(float, a.alpha_grid.split(","))
           for lm in map(float, a.lams.split(","))]
OUT = "%s/A31_exponent_%s.csv" % (OUTDIR, a.shard)
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r["product"], float(r["s"]), float(r["alpha"]), float(r["lam"]), int(r["seed"])))
else:
    open(OUT, "w").write("source,product,s,alpha,lam,seed,seconds,rank,iters,converged,auprc,prec_at_k,nnz_frac\n")

prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
print("A31 shard %s source %s products %s seeds %s configs %d device %s" %
      (a.shard, a.source, prods, seeds, len(CFG), DEVICE), flush=True)
base = None; filters = {}
for p in prods:
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in seeds:
        todo = [c for c in CFG if (p, c[0], c[1], c[2], sd) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy(); r = L.mp_rank(X)
        key = (len(b["lat"]), len(b["lon"]))
        if base is None or base[0] != key:
            ev, U, node_of = L.graph_eig(b["lat"], b["lon"]); base = (key, ev, U, node_of); filters = {}
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for s, al, lm in todo:
            if (s, al) not in filters:
                inv = 1.0 / (1.0 + al * np.power(base[1] + 1e-12, s))      # same form as A5_leaderboard.SphereFilter
                if filters:                                                   # keep one U on the device
                    f0 = next(iter(filters.values()))
                    f = FT.SphereFilterT.__new__(FT.SphereFilterT)
                    f.__dict__.update(f0.__dict__)
                    f.inv = torch.as_tensor(inv, dtype=torch.float32, device=DEVICE)[:, None]
                else:
                    f = FT.SphereFilterT(base[2], inv, base[3], device=DEVICE, dtype=torch.float32)
                filters[(s, al)] = f
            SYNC(); t0 = time.perf_counter()
            Lm, S, info = FT.run_flos_ablate(Xt, filters[(s, al)], lam=lm, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"],
                                            tol=L.FLOS["tol"], paper_iters=L.FLOS["paper_iters"], rank=r, ablation=a.ablation)
            SYNC(); sec = time.perf_counter() - t0
            nnz = float((S != 0).float().mean())
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT, "a").write("%s,%s,%g,%g,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.5f\n" % (
                a.source, p, s, al, lm, sd, sec, info["rank"], info["iters"], info["converged"],
                row["auprc"], row["prec_at_k"], nnz))
            print("  %-5s %-6s seed%d s=%-4g alpha=%-5g lam=%-5g AUPRC %.5f nnz %.4f %.0fs" % (
                a.source, p, sd, s, al, lm, row["auprc"], nnz, sec), flush=True)
            del Lm, S
        del Xt
        if DEVICE == "cuda": torch.cuda.empty_cache()
print("A31 shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
