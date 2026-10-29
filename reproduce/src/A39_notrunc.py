#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A39: how much of the gap between FLoS and its convex core comes from the truncation?

Compares FLoS with and without rank truncation.  Same benchmark generator, scoring, seeds and frozen
FLoS settings as the main table and A30 (alpha=0.85, s=0.25, p=1.25, 500 rounds, tol 1e-5, float32).  Arms:
  full   FLoS exactly as in the main table:  L_k = T_r(F(X - S_{k-1})),  S_k = adaptive shrink(X - L_k)
  noT    identical except that the background is not truncated:  L_k = F(X - S_{k-1})
Output: $FLOS_OUT/A39_notrunc_<shard>.csv
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
ap.add_argument("--arms", default="full,noT")
ap.add_argument("--lams", default="0.72")
ap.add_argument("--shard", required=True)
a = ap.parse_args()

L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
C = L.FLOS
OUT = "%s/A39_notrunc_%s.csv" % (OUTDIR, a.shard)
COLS = "source,product,arm,lam,seed,seconds,rank,iters,converged,auprc,prec_at_k,nnz_frac"
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r["product"], r["arm"], float(r["lam"]), int(r["seed"])))
else:
    open(OUT, "w").write(COLS + "\n")


def run_notrunc(X, filt, lam):
    """run_flos of flos_torch line by line, without the rank truncation of the background."""
    S = torch.zeros_like(X); Lb = torch.zeros_like(X)
    nx = float(torch.linalg.norm(X)); conv = False; d = float("nan")
    for it in range(1, C["max_iter"] + 1):
        Ln = filt(X - S)
        Sn = FT.adaptive_lorentz_shrink(X - Ln, lam, p=C["p"])
        d = float(max(torch.linalg.norm(Ln - Lb), torch.linalg.norm(Sn - S)) / nx)
        Lb, S = Ln, Sn
        if d < C["tol"]: conv = True; break
    return Lb, S, dict(iters=it, converged=conv)


arms = a.arms.split(","); prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
lams = [float(x) for x in a.lams.split(",")]
print("A39 shard %s source %s products %s seeds %s arms %s lams %s device %s" %
      (a.shard, a.source, prods, seeds, arms, lams, DEVICE), flush=True)
filt = None; geo = None
for p in prods:
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in seeds:
        todo = [(arm, lm) for arm in arms for lm in lams if (p, arm, lm, sd) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy()
        key = (len(b["lat"]), len(b["lon"]))
        if geo != key:
            fn = L.SphereFilter(b["lat"], b["lon"], C["alpha"], C["s"])
            filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEVICE, dtype=torch.float32); geo = key
        r = L.mp_rank(X)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for arm, lm in todo:
            SYNC(); t0 = time.perf_counter()
            if arm == "full":
                Lm, S, info = FT.run_flos(Xt, filt, lam=lm, p=C["p"], max_iter=C["max_iter"], tol=C["tol"],
                                          paper_iters=C["paper_iters"], rank=r)
            else:
                Lm, S, info = run_notrunc(Xt, filt, lm)
            SYNC(); sec = time.perf_counter() - t0
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            nnz = float((S != 0).float().mean())
            open(OUT, "a").write("%s,%s,%s,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.5f\n" % (
                a.source, p, arm, lm, sd, sec, r if arm == "full" else "", info["iters"], info["converged"],
                row["auprc"], row["prec_at_k"], nnz))
            print("  %-5s %-6s seed%d %-5s lam=%-5g AUPRC %.5f nnz %.4f iters %s conv %s %.1fs" % (
                a.source, p, sd, arm, lm, row["auprc"], nnz, info["iters"], info["converged"], sec), flush=True)
            del Lm, S
        del Xt
        if DEVICE == "cuda": torch.cuda.empty_cache()
print("A39 shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
