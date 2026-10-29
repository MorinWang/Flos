#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A29: full sparsity curve on the third reanalysis (20CRv3).
A17 swept ten lambda values on the 12 held-out NCEP and ERA5 products, whereas A25 ran 20CRv3 only at the single
frozen setting, so the sparsity curve was missing for the third dataset.
Two arms, identical to A-adapt / B-fixed in A17 (same make_benchmark, same alpha/s/p, same Marchenko-Pastur rank
estimate, same randomized SVD, same L.score, same ten lambda values):
  full    spherical filter + amplitude-adaptive shrinkage (A-adapt in A17)
  fixed   spherical filter + fixed threshold (B-fixed in A17)
Benchmark generation, scoring, alpha, s, p, iteration cap and convergence tolerance are all left unchanged.
One csv per arm; interrupted runs resume.
This also makes the A28 comparison symmetric: with the full method's own lambda sweep, it can be compared with the
grid-graph arm of A28 with each at its own best lambda."""
import sys, os, time, csv, argparse
import numpy as np
from scipy import linalg

ap = argparse.ArgumentParser()
ap.add_argument("--arms", default="full,fixed")
ap.add_argument("--products", default="air,slp,pres,pottmp,rhum,omega,cape")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--lams", default="0.045,0.09,0.18,0.36,0.72,1.44,2.88,5.76,11.52,23.04")
ap.add_argument("--device", default="cuda")
ap.add_argument("--outdir", default=".")
a = ap.parse_args()

DIAG11 = os.path.dirname(os.path.abspath(__file__))                     # all modules live in src/
from paths import DATA as _DATA, EXTERNAL
DATA = _DATA
if os.environ.get("FLOS_REFIMPL"): sys.path.insert(0, os.environ["FLOS_REFIMPL"])
sys.path.insert(0, DIAG11)
sys.path.insert(0, DIAG11)
import A5_leaderboard as L
import flos_torch as FT
import torch
L.CACHE = os.environ.get("FLOS_CACHE", L.CACHE)
L.RAW = DATA + "/20cr/clean"
L.PRODUCTS = {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}


def grid_eig_wrap(H, W, cache):
    """4-neighbour lat-lon grid graph: unit weights, periodic in longitude, poles not merged. Normalisation is
    identical to fixed_sphere_laplacian (divide by the maximum degree, then rescale eigenvalues to a maximum of one).
    This is the graph one would build when ignoring that the domain is a sphere."""
    p = "%s/gridwrap_eig_%dx%d" % (cache, H, W)
    if os.path.exists(p + "_U.npy"):
        return np.load(p + "_ev.npy"), np.load(p + "_U.npy")
    n = H * W
    lap = np.zeros((n, n))
    idx = lambda r, c: r * W + c
    for r in range(H):
        for c in range(W):
            i = idx(r, c)
            for rr, cc in ((r + 1, c), (r, (c + 1) % W)):
                if rr >= H: continue
                j = idx(rr, cc)
                if i == j: continue
                lap[i, j] -= 1.0; lap[j, i] -= 1.0
                lap[i, i] += 1.0; lap[j, j] += 1.0
    lap /= float(np.max(np.diag(lap)))
    t = time.perf_counter()
    ev, U = linalg.eigh(lap, overwrite_a=True, check_finite=False); del lap
    ev = np.maximum(ev, 0.0); ev = ev / ev.max()
    print("  lat-lon grid graph %dx%d eigendecomposition %.0f s" % (H, W, time.perf_counter() - t), flush=True)
    os.makedirs(cache, exist_ok=True)
    np.save(p + "_ev.npy", ev); np.save(p + "_U.npy", U)
    return ev, U


_F = {}
def get_filter(kind, b):
    if kind in _F: return _F[kind]
    if kind == "sphere":
        fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
        f = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=a.device, dtype=torch.float32)
    else:
        H, W = len(b["lat"]), len(b["lon"])
        ev, U = grid_eig_wrap(H, W, L.CACHE)
        inv = 1.0 / (1.0 + L.FLOS["alpha"] * np.power(ev + 1e-12, L.FLOS["s"]))
        f = FT.SphereFilterT(U, inv, np.arange(H * W), device=a.device, dtype=torch.float32)
    _F[kind] = f
    return f


arms = a.arms.split(","); prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
lams = [float(x) for x in a.lams.split(",")]
print("A29 arms %s products %s seeds %s lambdas %s device %s" % (arms, prods, seeds, lams, a.device), flush=True)
print("alpha=%g s=%g p=%g max_iter=%d tol=%g" % (L.FLOS["alpha"], L.FLOS["s"], L.FLOS["p"],
                                             L.FLOS["max_iter"], L.FLOS["tol"]), flush=True)

for arm in arms:
    out = os.path.join(a.outdir, "A29_%s.csv" % arm)
    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)): done.add((r["product"], float(r["lam"]), int(r["seed"])))
    else:
        open(out, "w").write("source,product,arm,lam,seed,seconds,rank,iters,converged,"
                             "auprc,prec_at_k,pos_rate,n_pos,nnz_frac\n")
    for p in prods:
        for sd in seeds:
            todo = [lm for lm in lams if (p, lm, sd) not in done]
            if not todo: continue
            b = L.make_benchmark(os.path.join(L.RAW, L.PRODUCTS[p]), seed=sd, n_events=L.N_EVENTS)
            T, nlat, nlon = b["observed"].shape
            X = b["observed"].reshape(T, -1).T.copy()
            r = L.mp_rank(X)
            to3 = lambda M: M.T.reshape(T, nlat, nlon)
            filt = get_filter("sphere", b)        # both arms use the spherical graph
            Xt = torch.as_tensor(X, dtype=torch.float32, device=a.device)
            for lm in todo:
                if a.device == "cuda": torch.cuda.synchronize()
                t0 = time.perf_counter()
                try:
                    Lm, S, info = FT.run_flos_ablate(
                        Xt, filt, lam=lm, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"],
                        tol=L.FLOS["tol"], paper_iters=L.FLOS["paper_iters"], rank=r,
                        ablation=("fixed-thresh" if arm == "fixed" else None))
                except Exception as e:
                    open(out, "a").write("20cr,%s,%s,%g,%d,,,,,,,,,fail:%s\n"
                                         % (p, arm, lm, sd, str(e)[:60].replace(",", ";")))
                    print("  ✗ %s %s seed %d lam=%g failed: %s" % (arm, p, sd, lm, str(e)[:80]), flush=True)
                    continue
                if a.device == "cuda": torch.cuda.synchronize()
                sec = time.perf_counter() - t0
                nnz = float((S != 0).float().mean())
                row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
                open(out, "a").write("20cr,%s,%s,%g,%d,%.1f,%s,%s,%s,%.5f,%.5f,%.2g,%d,%.5f\n" % (
                    p, arm, lm, sd, sec, info["rank"], info["iters"], info["converged"],
                    row["auprc"], row["prec_at_k"], row["pos_rate"], row["n_pos"], nnz))
                print("  %-16s %-7s seed %d lam=%-6g AUPRC %.5f kept %.5f %.0fs" % (
                    arm, p, sd, lm, row["auprc"], nnz, sec), flush=True)
                del Lm, S
            del Xt
            if a.device == "cuda": torch.cuda.empty_cache()
    print("arm %s done %s" % (arm, time.strftime("%F %T")), flush=True)
print("A29 all done %s" % time.strftime("%F %T"), flush=True)
