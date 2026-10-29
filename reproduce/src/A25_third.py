#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A25: all main-table methods on a third reanalysis, 20CRv3.
No tuning on 20CRv3: every method uses the setting already tuned and frozen on the three NCEP development products.
  ours  = A-adapt from A17, lambda=0.72 (GPU, flos_torch, float32, identical to A17)
  fixed = B-fixed from A17, lambda=0.36 (same as above)
  pcp   = run_pcp_mult from A18, multiplier 2 (CPU, copied verbatim)
  gloss = GLOSS arm from A21, multiplier 2 (all five values score identically; the dev2 rule selects 2)
  eof   = EOF-PCA arm from A21, multiplier 0.25
  sort  = nnz_target=1.0 from A21_rawthresh, i.e. |observed|
Mondal and the tensor ScaledGD method run through their own scripts, A23/A24 with --eval --source 20cr.
Benchmark generation, seeds and scoring (L.score) are identical to A17/A18/A21. One csv per method; interrupted runs resume."""
import sys, os, time, csv, argparse
import numpy as np
from scipy import linalg
ap = argparse.ArgumentParser()
ap.add_argument("--methods", required=True)
ap.add_argument("--products", default="air,slp,pres,pottmp,rhum,omega,cape")
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--device", default="cpu")
ap.add_argument("--dry", action="store_true", help="pipeline test only: S = |observed|, methods are not run")
ap.add_argument("--outdir", default=".")
a = ap.parse_args()
DIAG11 = os.path.dirname(os.path.abspath(__file__))                     # all modules live in src/
from paths import DATA as _DATA, EXTERNAL
DATA = _DATA
if os.environ.get("FLOS_REFIMPL"): sys.path.insert(0, os.environ["FLOS_REFIMPL"])
sys.path.insert(0, DIAG11)
sys.path.insert(0, DIAG11)
import A5_leaderboard as L
L.CACHE = os.environ.get("FLOS_CACHE", L.CACHE)
L.RAW = DATA + "/20cr/clean"
L.PRODUCTS = {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}
FROZEN = {"ours": 0.72, "fixed": 0.36, "pcp": 2.0, "gloss": 2.0, "eof": 0.25, "sort": 1.0}

def run_pcp_mult(X, mult):
    """Identical to A18_pcp_sweep.run_pcp_mult."""
    c=L.PCP; m,n=X.shape; lam=mult*1.0/np.sqrt(max(m,n)); dn=np.linalg.norm(X)
    norm_two=linalg.svdvals(X, check_finite=False)[0]
    Y=X/max(norm_two, np.abs(X).max()/lam)
    mu=1.25/norm_two; mu_bar=mu*1e7
    A=np.zeros_like(X); E=np.zeros_like(X); conv=False; svp=0
    for it in range(1, c["max_iter"]+1):
        Tt=X-A+Y/mu
        E=np.maximum(Tt-lam/mu,0)+np.minimum(Tt+lam/mu,0)
        u,s,vt=linalg.svd(X-E+Y/mu, full_matrices=False, check_finite=False)
        svp=int((s>1/mu).sum())
        A=(u[:,:svp]*(s[:svp]-1/mu))@vt[:svp]
        Z=X-A-E; Y=Y+mu*Z; mu=min(mu*c["rho"], mu_bar)
        if np.linalg.norm(Z)/dn < c["tol"]: conv=True; break
    return A,E,dict(rank=svp,iters=it,converged=conv)

filt = None
def run(m, b):
    global filt
    O3 = b["observed"]; T, nlat, nlon = O3.shape
    X = O3.reshape(T, -1).T.copy(); to3 = lambda M: M.T.reshape(T, nlat, nlon)
    mu = FROZEN[m]; extra = ""
    if a.dry: return np.abs(O3), "dry"
    if m in ("ours", "fixed"):
        import torch, flos_torch as FT
        r = L.mp_rank(X)
        if filt is None:
            fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
            filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=a.device, dtype=torch.float32)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=a.device)
        Lm, S, info = FT.run_flos_ablate(Xt, filt, lam=mu, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                                         paper_iters=L.FLOS["paper_iters"], rank=r,
                                         ablation=("fixed-thresh" if m == "fixed" else None))
        S3 = np.abs(to3(S.double().cpu().numpy())); extra = "rank=%s iters=%s conv=%s" % (info["rank"], info["iters"], info["converged"])
        del Lm, S, Xt; torch.cuda.empty_cache() if a.device == "cuda" else None
    elif m == "pcp":
        A, E, info = run_pcp_mult(X, mu); S3 = np.abs(to3(E)); extra = "rank=%s iters=%s conv=%s" % (info["rank"], info["iters"], info["converged"])
    elif m == "gloss":
        from A5_gloss import run_gloss
        Tt, hh, ww = O3.shape; ny = Tt // 12; Tn = ny * 12
        Yt = O3[:Tn].reshape(ny, 12, hh, ww).transpose(1, 0, 2, 3).copy()
        rule = 1.0 / np.count_nonzero(Yt)
        Lg, Sg, info = run_gloss(Yt, lam=rule * mu, gam=rule * mu)
        S3 = np.zeros_like(O3); S3[:Tn] = Sg.transpose(1, 0, 2, 3).reshape(Tn, hh, ww); S3 = np.abs(S3)
        extra = "rule=%.3g x%g" % (rule, mu)
    elif m == "eof":
        r = max(1, int(round(mu * L.mp_rank(X))))
        u, s, vt = linalg.svd(X, full_matrices=False, check_finite=False)
        Lm = (u[:, :r] * s[:r]) @ vt[:r]; S3 = np.abs(to3(X - Lm)); extra = "rank=%d" % r
    elif m == "sort":
        S3 = np.abs(np.nan_to_num(O3, nan=0.0))
    return S3, extra

meths = a.methods.split(","); prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
print("A25 methods %s products %s seeds %s device %s%s" % (meths, prods, seeds, a.device, " [dry]" if a.dry else ""), flush=True)
for m in meths:
    out = os.path.join(a.outdir, "A25_%s%s.csv" % (m, "_dry" if a.dry else ""))
    done = set()
    if os.path.exists(out):
        for r in csv.DictReader(open(out)): done.add((r["product"], int(r["seed"])))
    else:
        open(out, "w").write("source,product,method,setting,seed,seconds,auprc,prec_at_k,pos_rate,n_pos,nnz_frac,extra\n")
    for p in prods:
        for sd in seeds:
            if (p, sd) in done: continue
            b = L.make_benchmark(os.path.join(L.RAW, L.PRODUCTS[p]), seed=sd, n_events=L.N_EVENTS)
            t0 = time.perf_counter()
            try:
                S3, extra = run(m, b)
            except Exception as e:
                open(out, "a").write("20cr,%s,%s,%g,%d,,,,,,,fail:%s\n" % (p, m, FROZEN[m], sd, str(e)[:60].replace(",", ";")))
                print("  ✗ %s %s seed %d failed: %s" % (m, p, sd, str(e)[:80]), flush=True); continue
            sec = time.perf_counter() - t0
            row = L.score(b, scores=S3); nnz = float((S3 != 0).mean())
            open(out, "a").write("20cr,%s,%s,%g,%d,%.1f,%.5f,%.5f,%.2g,%d,%.5f,%s\n" % (p, m, FROZEN[m], sd, sec, row["auprc"],
                                 row["prec_at_k"], row["pos_rate"], row["n_pos"], nnz, extra))
            print("  %-5s %-6s seed %d AUPRC %.5f kept %.4f %.0fs %s" % (m, p, sd, row["auprc"], nnz, sec, extra), flush=True)
print("A25 done %s" % meths, flush=True)
