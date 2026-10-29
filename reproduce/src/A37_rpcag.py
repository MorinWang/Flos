#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A37(e): RPCA on graphs (Shahid et al., ICCV 2015) on our benchmark.

    min ||L||_* + lam ||S||_1 + gamma tr(L' Phi^s L)   s.t.  X = L + S,        score |S|
Phi: 4-neighbour latitude-longitude lattice graph on the cells (longitude wrapped, poles not merged), Laplacian divided
by its largest degree and eigenvalues scaled to [0,1], as in A28's grid-graph arm; exponent s in {1, 1/4}.
ADMM with the split W = L (as in the paper): L by singular value thresholding (computed through the eigendecomposition
of the T x T Gram matrix in float64), S by soft thresholding, W in closed form in the eigenbasis of Phi; penalties
r1 = r2 = r increased geometrically (x1.5, from 1.25/||X||_2, capped at 1e7 times that), at most 500 iterations,
stop when ||X-L-S|| and ||W-L|| are both below 1e-6 ||X||.  lam = mult / sqrt(max(N,T)).
"""
import os, sys, time, csv, argparse, math
import numpy as np, torch
from scipy import linalg
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L

DEV = device()
SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True, choices=list(SRC))
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--configs", default="1:1:1,1:1:10,1:1:100,1:2:1,1:2:10,1:2:100,1:4:1,1:4:10,1:4:100",
                help="s:lam_mult:gamma triples")
ap.add_argument("--max_iter", type=int, default=500)
ap.add_argument("--shard", required=True)
a = ap.parse_args()
L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
CFG = [tuple(float(v) for v in c.split(":")) for c in a.configs.split(",")]
OUT = "%s/A37_rpcag_%s.csv" % (OUTDIR, a.shard)
if not os.path.exists(OUT):
    open(OUT, "w").write("source,product,seed,s,lam_mult,gamma,auprc,prec_at_k,nnz_frac,rank,iters,converged,seconds\n")
done = {(r["product"], int(r["seed"]), float(r["s"]), float(r["lam_mult"]), float(r["gamma"])) for r in csv.DictReader(open(OUT))}


def lattice_eig(H, W):
    p = "%s/gridwrap_eig_%dx%d" % (CACHE, H, W)
    if os.path.exists(p + "_U.npy"):
        return np.load(p + "_ev.npy"), np.load(p + "_U.npy")
    n = H * W; lap = np.zeros((n, n))
    for r in range(H):
        for c in range(W):
            i = r * W + c
            for rr, cc in ((r + 1, c), (r, (c + 1) % W)):
                if rr >= H: continue
                j = rr * W + cc
                lap[i, j] -= 1; lap[j, i] -= 1; lap[i, i] += 1; lap[j, j] += 1
    lap /= float(np.max(np.diag(lap)))
    t = torch.as_tensor(lap, dtype=torch.float64, device=DEV)
    ev, U = torch.linalg.eigh(t); ev = torch.clamp(ev, min=0); ev = ev / ev.max()
    ev, U = ev.cpu().numpy(), U.cpu().numpy()
    np.save(p + "_ev.npy", ev); np.save(p + "_U.npy", U)
    return ev, U


def svt(A, tau):
    """Singular value thresholding through the T x T Gram matrix (float64)."""
    G = (A.T.double() @ A.double())
    w, V = torch.linalg.eigh(0.5 * (G + G.T))
    sig = torch.sqrt(torch.clamp(w, min=0))
    f = torch.where(sig > tau, (sig - tau) / torch.clamp(sig, min=1e-12), torch.zeros_like(sig))
    rank = int((sig > tau).sum())
    return (A.double() @ (V * f) @ V.T).to(A.dtype), rank


def rpcag(X, Ug, mu_s, lam, gam, max_iter):
    N, T = X.shape
    nx = float(torch.linalg.norm(X))
    norm2 = float(torch.linalg.matrix_norm(X.double(), ord=2))
    r = 1.25 / norm2; rmax = r * 1e7
    Lm = torch.zeros_like(X); S = torch.zeros_like(X); Wm = torch.zeros_like(X)
    Z1 = torch.zeros_like(X); Z2 = torch.zeros_like(X); conv = False; rank = 0
    for it in range(1, max_iter + 1):
        Lm, rank = svt(0.5 * ((X - S + Z1 / r) + (Wm + Z2 / r)), 1.0 / (2 * r))
        R = X - Lm + Z1 / r
        S = torch.sign(R) * torch.clamp(torch.abs(R) - lam / r, min=0.0)
        B = Lm - Z2 / r
        Wm = Ug @ ((Ug.T @ B) * (r / (2 * gam * mu_s + r))[:, None])
        E1 = X - Lm - S; E2 = Wm - Lm
        Z1 = Z1 + r * E1; Z2 = Z2 + r * E2
        r = min(1.5 * r, rmax)
        if float(torch.linalg.norm(E1)) < 1e-6 * nx and float(torch.linalg.norm(E2)) < 1e-6 * nx:
            conv = True; break
    return S, rank, it, conv


geo = None
for p in a.products.split(","):
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in map(int, a.seeds.split(",")):
        todo = [c for c in CFG if (p, sd, c[0], c[1], c[2]) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        if geo != (nlat, nlon):
            ev, U = lattice_eig(nlat, nlon); geo = (nlat, nlon)
            Ug = torch.as_tensor(U, dtype=torch.float32, device=DEV); evt = torch.as_tensor(ev, dtype=torch.float32, device=DEV)
        X = b["observed"].reshape(T, -1).T.copy()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for s_, mult, gam in todo:
            t0 = time.time()
            lam = mult / math.sqrt(max(X.shape))
            S, rank, it, conv = rpcag(Xt, Ug, torch.pow(evt + 1e-12, s_), lam, gam, a.max_iter)
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            nnz = float((S != 0).float().mean())
            open(OUT, "a").write("%s,%s,%d,%g,%g,%g,%.5f,%.5f,%.5f,%d,%d,%s,%.1f\n" % (a.source, p, sd, s_, mult, gam,
                                 row["auprc"], row["prec_at_k"], nnz, rank, it, conv, time.time() - t0))
            print("  %-5s %-6s seed%d s=%g mult=%g gamma=%g AUPRC %.5f nnz %.3f rank %d iters %d conv %s %.0fs" % (
                a.source, p, sd, s_, mult, gam, row["auprc"], nnz, rank, it, conv, time.time() - t0), flush=True)
            del S
        del Xt
        torch.cuda.empty_cache()
print("A37 rpcag shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
