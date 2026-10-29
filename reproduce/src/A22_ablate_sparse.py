#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A22: ablation attributing the gains in the sparse regime.
Three arms, each changing exactly one component; everything else is identical to A17 (same make_benchmark,
same lambda grid, same seeds, same scoring):
  no-smooth   remove the fractional spherical smoothing (delta1); the background step is a plain low-rank truncation
  integer-s   exponent 0.25 -> 1.0, i.e. the fractional order degenerates to an integer order
  rw-e1.0     reweighted shrinkage with thresholds taken from the previous iterate's S; it overtook the baseline in the
              sparse regime on the development set (A19), so here it is checked on the held-out set
Full FLoS (A-adapt in A17) and FLoS without delta2 (B-fixed in A17) are not rerun; rows are paired with the
A17 rows for the same (product, lambda, seed).
The benchmark, scoring, alpha, p, iteration cap and convergence tolerance are all left unchanged. Runs are
sharded for parallelism; each shard writes its own csv to avoid write conflicts."""
import sys, os, time, csv, argparse
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device; DEVICE = device()
SYNC = torch.cuda.synchronize if DEVICE == "cuda" else (lambda: None)
FREE = torch.cuda.empty_cache if DEVICE == "cuda" else (lambda: None)
import A5_leaderboard as L
import flos_torch as FT

SRC = {"ncep": (DATA + "/raw",
                {"lftx":"lftx.sfc.mon.mean.nc","pres":"pres.sfc.mon.mean.nc",
                 "omega":"omega.sig995.mon.mean.nc","pottmp":"pottmp.sig995.mon.mean.nc",
                 "gpcp":"gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean",
                {"air":"era5_air.mon.nc","slp":"era5_slp.mon.nc","pres":"era5_pres.mon.nc",
                 "pottmp":"era5_pottmp.mon.nc","rhum":"era5_rhum.mon.nc",
                 "omega":"era5_omega.mon.nc","cape":"era5_cape.mon.nc"})}
ARMS = ["no-smooth", "integer-s", "rw-e1.0"]

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=["ncep","era5"])
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lams", default="0.045,0.09,0.18,0.36,0.72,1.44,2.88,5.76,11.52,23.04")
ap.add_argument("--arms", default=",".join(ARMS))
ap.add_argument("--shard", required=True)
a = ap.parse_args()

L.CACHE = CACHE; L.RAW, L.PRODUCTS = SRC[a.source]
prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
lams = [float(x) for x in a.lams.split(",")]; arms = a.arms.split(",")
OD = OUTDIR; os.makedirs(OD, exist_ok=True)
OUT = "%s/A22_ablate_%s.csv" % (OD, a.shard)
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r["source"], r["product"], r["arm"], float(r["lam"]), int(r["seed"])))
else:
    open(OUT,"w").write("source,product,arm,lam,seed,seconds,rank,iters,converged,auprc,prec_at_k,pos_rate,n_pos,nnz_frac\n")

print("shard %s  source %s  products %s  seeds %s  arms %s  %d lambda values" % (
    a.shard, a.source, prods, seeds, arms, len(lams)), flush=True)
for p in prods:
    path = os.path.join(L.RAW, L.PRODUCTS[p]); f_main = f_int = None
    for sd in seeds:
        todo = [(arm, lm) for lm in lams for arm in arms
                if (a.source, p, arm, lm, sd) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy(); r = L.mp_rank(X)
        if f_main is None:
            fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
            f_main = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEVICE, dtype=torch.float32)
            fi = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], 1.0)     # exponent fixed to one
            f_int = FT.SphereFilterT(fi.U, fi.inv, fi.node_of, device=DEVICE, dtype=torch.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for arm, lm in todo:
            SYNC(); t0 = time.perf_counter()
            if arm == "rw-e1.0":
                Lm, S, info = FT.run_flos_reweight(Xt, f_main, lam=lm, p=L.FLOS["p"], eps_w=1.0,
                    max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                    paper_iters=L.FLOS["paper_iters"], rank=r)
            else:
                filt = f_int if arm == "integer-s" else f_main
                Lm, S, info = FT.run_flos_ablate(Xt, filt, lam=lm, p=L.FLOS["p"],
                    max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                    paper_iters=L.FLOS["paper_iters"], rank=r,
                    ablation=("no-smooth" if arm == "no-smooth" else None))
            SYNC(); sec = time.perf_counter() - t0
            nnz = float((S != 0).float().mean())
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT,"a").write("%s,%s,%s,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.2g,%d,%.5f\n" % (
                a.source, p, arm, lm, sd, sec, info["rank"], info["iters"], info["converged"],
                row["auprc"], row["prec_at_k"], row["pos_rate"], row["n_pos"], nnz))
            print("  %-6s %-7s seed %d lam=%-6g %-10s AUPRC %.5f kept %.5f %.0fs" % (
                a.source, p, sd, lm, arm, row["auprc"], nnz, sec), flush=True)
            del Lm, S
        del Xt; FREE()
print("shard %s done" % a.shard, flush=True)
