#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A30b: the convex core with its smoothing strength re-tuned.
Only CORE-soft; grid over s, alpha and lam; otherwise identical to A30.

Original A30 description follows.
A30: does the convex core of FLoS carry its detection mechanism?

Runs the following arms on the benchmark (all alpha=0.85, s=0.25, same spherical graph,
same make_benchmark, same L.score as the main table):
  HP1         score = |X - F X|              one-shot graph high-pass of the observation (no iteration, no lambda)
  HP1T        score = |X - T_r(F X)|          round-1 residual of FLoS before shrinkage (truncation kept)
  CORE-soft   exact block-coordinate descent on
                 J(Z,S) = 1/2 ||X - P Z - S||_F^2 + alpha/2 tr(Z' D^s Z) + lam ||S||_1,
              i.e.  L_k = H (X - S_{k-1}),  H = P (Omega + alpha D^s)^{-1} P',   S_k = soft_lam(X - L_k).
              This is ISTA with unit step on  Phi(S) = 1/2 <X-S, (I-H)(X-S)> + lam ||S||_1  (Theorem in the paper).
              Phi(S_{k-1}) is logged every round; the theorem says it never increases.
  CORE-adapt  same background step, FLoS's magnitude-adaptive shrinkage instead of soft thresholding (no guarantee).
The two CORE arms differ from FLoS only by (i) no rank truncation T_r, (ii) the polar-node data weight |v| (exact
block minimiser instead of plain averaging).  max_iter 2000, tol 1e-5 (FLoS criterion); the state at round 500 is
also scored so that the comparison with FLoS at its own budget is available.
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
ap.add_argument("--arms", default="CORE-soft")
ap.add_argument("--s_grid", default="0.25,1.0")
ap.add_argument("--alpha_grid", default="0.85,2.5,7.5,22.5,67.5")
ap.add_argument("--configs", default="", help="explicit s:alpha:lam triples")
ap.add_argument("--lams", default="0.09,0.18,0.36,0.72,1.44")
ap.add_argument("--max_iter", type=int, default=2000)
ap.add_argument("--shard", required=True)
a = ap.parse_args()

L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
ALPHA, SEXP, P_EXP, TOL = L.FLOS["alpha"], L.FLOS["s"], L.FLOS["p"], L.FLOS["tol"]
OUT = "%s/A30b_core_%s.csv" % (OUTDIR, a.shard)
COLS = ("source,product,arm,s,alpha,lam,seed,seconds,rank,iters,converged,obj_increases,obj_max_rel_increase,obj_final,"
        "auprc,prec_at_k,auprc_at500,nnz_frac,nnz_at500")
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r["product"], float(r["s"]), float(r["alpha"]), float(r["lam"]), int(r["seed"])))
else:
    open(OUT, "w").write(COLS + "\n")


class CoreBackground:
    """L = P (Omega + alpha D^s)^{-1} P' Y : the exact minimiser over Z of 1/2||Y - P Z||^2 + alpha/2 tr(Z' D^s Z)."""
    def __init__(self, lat, lon, ALPHA, SEXP):
        ev, U, node_of = L.graph_eig(lat, lon)
        self.node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEVICE)
        n = int(self.node.max()) + 1
        cnt = np.bincount(np.asarray(node_of), minlength=n).astype(np.float64)
        Ud = torch.as_tensor(U, dtype=torch.float64, device=DEVICE)
        d = torch.as_tensor(ALPHA * np.power(ev + 1e-12, SEXP), dtype=torch.float64, device=DEVICE)
        M = (Ud * d) @ Ud.T
        M.diagonal().add_(torch.as_tensor(cnt, dtype=torch.float64, device=DEVICE))
        M = 0.5 * (M + M.T)
        C = torch.linalg.cholesky(M)
        self.Minv = torch.cholesky_inverse(C).to(torch.float32)
        del Ud, M, C
        self.n = n

    def __call__(self, Y):
        acc = torch.zeros(self.n, Y.shape[1], dtype=Y.dtype, device=Y.device)
        acc.index_add_(0, self.node, Y)                      # P' Y (sum over the cells of a node)
        return (self.Minv @ acc)[self.node]                  # P (Omega + alpha D^s)^{-1} P' Y


def soft(x, lam):
    return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


def run_core(X, H, lam, adaptive):
    nx = float(torch.linalg.norm(X))
    S = torch.zeros_like(X); Lb = torch.zeros_like(X)
    objs = []; conv = False; snap = None; d = float("nan")
    for it in range(1, a.max_iter + 1):
        Y = X - S
        Ln = H(Y)
        if not adaptive:                                   # Phi(S_{k-1}); computed in float64 from float32 tensors
            objs.append(0.5 * float(torch.sum(Y.double() * (Y - Ln).double())) + lam * float(torch.sum(torch.abs(S).double())))
        R = X - Ln
        Sn = FT.adaptive_lorentz_shrink(R, lam, p=P_EXP) if adaptive else soft(R, lam)
        d = float(max(torch.linalg.norm(Ln - Lb), torch.linalg.norm(Sn - S)) / nx)
        Lb, S = Ln, Sn
        if it == 500: snap = S.clone()
        if d < TOL:
            conv = True; break
    if snap is None: snap = S.clone()
    inc, mx = 0, 0.0
    for i in range(1, len(objs)):
        rel = (objs[i] - objs[i - 1]) / max(abs(objs[i - 1]), 1e-12)
        if rel > 1e-5: inc += 1
        mx = max(mx, rel)
    return S, snap, dict(iters=it, converged=conv, obj_inc=inc if objs else "", obj_mx=mx if objs else "",
                         obj_final=objs[-1] if objs else "")


prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
if a.configs:
    CFG = [tuple(float(v) for v in c.split(":")) for c in a.configs.split(",")]
else:
    CFG = [(sx, al, lm) for sx in map(float, a.s_grid.split(",")) for al in map(float, a.alpha_grid.split(","))
           for lm in map(float, a.lams.split(","))]
print("A30b shard %s source %s products %s seeds %s configs %d device %s" % (a.shard, a.source, prods, seeds, len(CFG), DEVICE), flush=True)
Hs = {}; geo = None
for p in prods:
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in seeds:
        todo = [c for c in CFG if (p, c[0], c[1], c[2], sd) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy()
        key = (len(b["lat"]), len(b["lon"]))
        if geo != key: Hs = {}; geo = key
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for sx, al, lm in todo:
            if (sx, al) not in Hs:
                if len(Hs) >= 2: Hs.pop(next(iter(Hs)))
                Hs[(sx, al)] = CoreBackground(b["lat"], b["lon"], al, sx)
            SYNC(); t0 = time.perf_counter()
            S, snap, info = run_core(Xt, Hs[(sx, al)], lm, adaptive=False)
            SYNC(); sec = time.perf_counter() - t0
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            row5 = L.score(b, scores=np.abs(to3(snap.double().cpu().numpy())))
            nnz = float((S != 0).float().mean()); nnz5 = float((snap != 0).float().mean())
            open(OUT, "a").write("%s,%s,CORE-soft,%g,%g,%g,%d,%.1f,,%s,%s,%s,%s,%s,%.5f,%.5f,%.5f,%.5f,%.5f\n" % (
                a.source, p, sx, al, lm, sd, sec, info["iters"], info["converged"], info["obj_inc"], info["obj_mx"],
                info["obj_final"], row["auprc"], row["prec_at_k"], row5["auprc"], nnz, nnz5))
            print("  %-5s %-6s seed%d s=%-4g alpha=%-5g lam=%-5g AUPRC %.5f nnz %.4f iters %s conv %s inc %s %.0fs" % (
                a.source, p, sd, sx, al, lm, row["auprc"], nnz, info["iters"], info["converged"], info["obj_inc"], sec), flush=True)
            del S, snap
        del Xt
        if DEVICE == "cuda": torch.cuda.empty_cache()
print("A30b shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
