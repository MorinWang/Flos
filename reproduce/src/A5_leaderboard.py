# -*- coding: utf-8 -*-
"""A5: zero-tuning leaderboard (first version, not the main table).
Benchmark = A1'' same-location year swap (A1pp_honest_benchmark.py, imported unchanged).
Scoring = scoring outside the exclusion zone: computed only on score_mask (cells that are not real extremes
already present in the background); the primary metric is the AUPRC of |S| against the injected support.

Methods (each is run to convergence or with the settings of its own paper; no shared shrinkage parameter;
no asymmetric iteration caps):
  FLoS-fixed  Repaired FLoS with the original hyperparameters λ=0.18 α=0.85 s=0.25 p=1.25 (the values from its own paper);
              (1) the extra local grid-diffusion step applied after the fractional filter is removed; (2) the sphere
              graph merges the cells of each pole latitude into one node; (3) the hidden switch fast_fractional is
              turned off (the filter is actually applied every iteration); (4) the rank cap is lifted and the rank
              is estimated with the Marchenko-Pastur rule.
              Two rows are reported: FLoS-fixed@10it = the 10 iterations used in the original paper (the
              iteration-10 intermediate of the same run);
              FLoS-fixed = run to convergence: relative change of both L and S between consecutive iterations
              < 1e-5, capped at 500 iterations (whether it converged is recorded).
              Smoke test (coarse grid): change 1.7e-2 at iteration 10, about 1e-5 after 200 iterations,
              decreasing monotonically without oscillation.
  EOF-PCA     Classical baseline: MP rank estimate (no cap), truncated SVD, S = X − L. No parameters.
  PCP-L1      Candès et al. 2011: λ = 1/√max(m,n) (paper §1.4 p.6); solver = inexact ALM of Lin, Chen, Ma with default settings.
  TV-RPCA     Cao et al. 2016: line-by-line port of the official MATLAB code (A5_tvrpca.py), λ=[0.4,2,1]/√(hw),
              fixed 20 iterations; event score |F| (first pass, not ranked).
  TV-RPCA-M   Same as above, re-run: the ranked row uses |M|, M = E + F; TV-RPCA-F and TV-RPCA-E rows are
              also recorded for reference only.
  GLOSS       Sofuoglu & Aviyente 2022: line-by-line port of the first author's code (A5_gloss.py), parameters
              from paper p.16–17; tensor month×year×lat×lon.
              Two rows are reported: GLOSS = |S| (same yardstick as the other methods); GLOSS-EE = the paper's own
              score (robust Mahalanobis distance along each year fibre).
  dumb-zscore One of the simplest alternatives: the absolute per-cell, per-month z-score used as the event score (no decomposition).
Unless an official implementation is available, methods are our own implementations from the papers; the impl
column records the provenance of each implementation.
Units: the benchmark is already in global robust-scale units, the same preprocessing as the original FLoS code
(_robust_scale in datasets.py), so λ=0.18 can be used unchanged.
"""
import os, sys, time, csv, argparse, hashlib
import numpy as np, xarray as xr
from scipy import linalg
from scipy.sparse import coo_matrix, csr_matrix
from sklearn.metrics import average_precision_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # climate_rpca/ sits next to this file
from paths import DATA, OUT as OUTDIR, CACHE as _CACHE
from A1pp_honest_benchmark import make_benchmark
from climate_rpca.linalg import (adaptive_lorentz_shrink, estimate_rank_mp_tw, randomized_low_rank_approx,
                                 _haversine_deg)

RAW = os.path.join(DATA, "raw")
PRODUCTS = {"air": "air.mon.mean.nc", "lftx": "lftx.sfc.mon.mean.nc", "slp": "slp.mon.mean.nc",
            "pres": "pres.sfc.mon.mean.nc", "omega": "omega.sig995.mon.mean.nc",
            "pottmp": "pottmp.sig995.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc", "gpcp": "gpcp.mon.mean.nc"}
CACHE = _CACHE
OUT = os.path.join(OUTDIR, "A5_leaderboard_runs.csv")
N_EVENTS = 60
FLOS = dict(lam=0.18, alpha=0.85, s=0.25, p=1.25, max_iter=500, tol=1e-5, paper_iters=10)
PCP = dict(rho=1.5, tol=1e-7, max_iter=1000)      # official inexact_alm_rpca.m lines 40/46/62; Lin et al. §4 p.12 state ρ=1.6, we follow the official code
POLE_TOL = 1e-6
IMPL = {"PCP-L1": "own implementation, checked step by step against the official inexact_alm_rpca.m", "TV-RPCA": "line-by-line port of the official MATLAB code",
        "GLOSS": "line-by-line port of the authors' MATLAB code, validated", "TV-RPCA-M": "line-by-line port of the official MATLAB code, validated", "GLOSS-EE": "line-by-line port of the authors' MATLAB code, validated",
        "FLoS-fixed": "original FLoS code with the pole fix", "EOF-PCA": "own implementation", "dumb-zscore": "own implementation"}

# ---------------- Repaired sphere graph (pole fix: all cells on a pole latitude are merged into one physical node;
# otherwise as in the original code: 4-neighbour connectivity, edge weight = inverse great-circle distance)
def fixed_sphere_laplacian(lat, lon):
    nlat, nlon = len(lat), len(lon)
    node_of = np.empty(nlat*nlon, dtype=int); coords = []
    for i in range(nlat):
        if abs(abs(float(lat[i])) - 90.0) < POLE_TOL:
            node_of[i*nlon:(i+1)*nlon] = len(coords); coords.append((float(lat[i]), 0.0))
        else:
            for j in range(nlon):
                node_of[i*nlon+j] = len(coords); coords.append((float(lat[i]), float(lon[j])))
    n = len(coords); rows, cols, vals, seen = [], [], [], set()
    for i in range(nlat):
        for j in range(nlon):
            a = int(node_of[i*nlon+j])
            for ii, jj in ((i+1, j), (i, (j+1) % nlon)):
                if ii >= nlat: continue
                b = int(node_of[ii*nlon+jj]); key = (min(a, b), max(a, b))
                if a == b or key in seen: continue
                seen.add(key)
                d = _haversine_deg(float(lat[i]), float(lon[j]), float(lat[ii]), float(lon[jj]))
                w = 1.0/max(d, 1e-6); rows += [a, b]; cols += [b, a]; vals += [w, w]
    W = coo_matrix((vals, (rows, cols)), shape=(n, n)).toarray()
    lap = np.diag(W.sum(1)) - W; del W
    return lap / float(np.max(np.diag(lap))), node_of, n

def graph_eig(lat, lon):
    h = hashlib.sha1(); h.update(np.asarray(lat, float).tobytes()); h.update(np.asarray(lon, float).tobytes())
    p = "%s/fixed_eig_%s" % (CACHE, h.hexdigest()[:16])
    if os.path.exists(p + "_U.npy"):
        return np.load(p + "_ev.npy"), np.load(p + "_U.npy"), np.load(p + "_node.npy")
    os.makedirs(CACHE, exist_ok=True); t = time.perf_counter()
    lap, node_of, n = fixed_sphere_laplacian(lat, lon)
    ev, U = linalg.eigh(lap, overwrite_a=True, check_finite=False); del lap
    ev = np.maximum(ev, 0.0); ev = ev / ev.max()
    np.save(p + "_ev.npy", ev); np.save(p + "_node.npy", node_of); np.save(p + "_U.npy", U)
    print("  eigendecomposition of %d nodes took %.0f s, cached at %s" % (n, time.perf_counter()-t, p), flush=True)
    return ev, U, node_of

class SphereFilter:
    """F(Y) = U diag(1/(1+α μ^s)) Uᵀ Y on physical nodes; duplicated pole cells are averaged on input and broadcast on output."""
    def __init__(self, lat, lon, alpha, s):
        ev, self.U, self.node_of = graph_eig(lat, lon)
        self.inv = 1.0/(1.0 + alpha*np.power(ev + 1e-12, s))
        n = int(self.node_of.max()) + 1; cnt = np.bincount(self.node_of, minlength=n)
        self.A = csr_matrix((1.0/cnt[self.node_of], (self.node_of, np.arange(len(self.node_of)))),
                            shape=(n, len(self.node_of)))
    def __call__(self, Y):
        C = self.U.T @ (self.A @ Y)
        return (self.U @ (C * self.inv[:, None]))[self.node_of]

# ---------------- methods
def mp_rank(X):
    return estimate_rank_mp_tw(X, max_rank=min(X.shape) - 1)        # rank cap lifted

def run_flos(X, filt):
    c = FLOS; r = mp_rank(X); nx = np.linalg.norm(X)
    S = np.zeros_like(X); L = np.zeros_like(X); conv = False; snap = None
    for it in range(1, c["max_iter"] + 1):
        Ln = randomized_low_rank_approx(filt(X - S), r)             # as in the original code: randomized SVD, n_iter=1
        Sn = adaptive_lorentz_shrink(X - Ln, c["lam"], p=c["p"])
        d = max(np.linalg.norm(Ln - L), np.linalg.norm(Sn - S)) / nx
        L, S = Ln, Sn
        if it == c["paper_iters"]: snap = (L.copy(), S.copy(), d)
        if d < c["tol"]: conv = True; break
    return L, S, dict(rank=r, iters=it, converged=conv, last_change=d, snap=snap)

def run_eof(X):
    r = mp_rank(X)
    u, s, vt = linalg.svd(X, full_matrices=False, check_finite=False)
    L = (u[:, :r]*s[:r]) @ vt[:r]
    return L, X - L, dict(rank=r, iters=1, converged=True)

def run_pcp(X):
    """Lin, Chen, Ma (2010) inexact_alm_rpca: update E first (soft thresholding), then A (singular value thresholding)."""
    c = PCP; m, n = X.shape; lam = 1.0/np.sqrt(max(m, n)); dn = np.linalg.norm(X)
    norm_two = linalg.svdvals(X, check_finite=False)[0]
    Y = X / max(norm_two, np.abs(X).max()/lam)
    mu = 1.25/norm_two; mu_bar = mu*1e7
    A = np.zeros_like(X); E = np.zeros_like(X); conv = False; svp = 0
    for it in range(1, c["max_iter"] + 1):
        Tt = X - A + Y/mu
        E = np.maximum(Tt - lam/mu, 0) + np.minimum(Tt + lam/mu, 0)
        u, s, vt = linalg.svd(X - E + Y/mu, full_matrices=False, check_finite=False)
        svp = int((s > 1/mu).sum())
        A = (u[:, :svp]*(s[:svp] - 1/mu)) @ vt[:svp]
        Z = X - A - E; Y = Y + mu*Z; mu = min(mu*c["rho"], mu_bar)
        if np.linalg.norm(Z)/dn < c["tol"]: conv = True; break
    return A, E, dict(rank=svp, iters=it, converged=conv)

def run_dumb_zscore(obs3, months):
    z = np.empty_like(obs3)
    for m in np.unique(months):
        s = months == m; mu = obs3[s].mean(0); sd = obs3[s].std(0); sd = np.where(sd < 1e-12, np.inf, sd)
        z[s] = (obs3[s] - mu)/sd
    return np.abs(z)

# ---------------- scoring
def score(b, S3=None, L3=None, scores=None):
    m = b["score_mask"]; y = b["support"][m]
    s = (np.abs(S3) if scores is None else scores)[m]
    k = int(y.sum()); top = np.argpartition(s, -k)[-k:]
    out = dict(auprc=float(average_precision_score(y, s)), prec_at_k=float(y[top].mean()),
               pos_rate=float(y.mean()), n_pos=k)
    if L3 is not None:
        B, St = b["background"], b["S"]
        out["spill"] = float(np.sum((L3 - B)*St)/np.sum(St*St))       # fraction of the injected events absorbed by the background component
        out["bg_relerr"] = float(np.linalg.norm(L3 - B)/np.linalg.norm(B))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--products", default=",".join(PRODUCTS))
    ap.add_argument("--methods", default="FLoS-fixed,EOF-PCA,dumb-zscore")
    ap.add_argument("--seeds", default="0,1,2,3,4")
    ap.add_argument("--step", default="run", choices=["run", "eig"])
    a = ap.parse_args()
    prods = a.products.split(","); methods = a.methods.split(","); seeds = [int(s) for s in a.seeds.split(",")]
    if a.step == "eig":
        for p in prods:
            d = xr.open_dataset(os.path.join(RAW, PRODUCTS[p]))
            graph_eig(d.lat.values.astype(float), d.lon.values.astype(float)); d.close()
        return
    done = set()
    if os.path.exists(OUT):
        with open(OUT) as fh:
            for r in csv.DictReader(fh): done.add((r["product"], r["method"], int(r["seed"])))
    for p in prods:
        path = os.path.join(RAW, PRODUCTS[p]); filt = None
        d = xr.open_dataset(path); months = d.time.dt.month.values; d.close()
        for sd in seeds:
            todo = [mt for mt in methods if (p, mt, sd) not in done]
            if not todo: continue
            b = make_benchmark(path, seed=sd, n_events=N_EVENTS)
            T, nlat, nlon = b["observed"].shape
            X = b["observed"].reshape(T, -1).T.copy()
            to3 = lambda M: M.T.reshape(T, nlat, nlon)
            for mt in todo:
                t0 = time.perf_counter()
                if mt == "FLoS-fixed":
                    if filt is None: filt = SphereFilter(b["lat"], b["lon"], FLOS["alpha"], FLOS["s"])
                    L, S, info = run_flos(X, filt)
                elif mt == "EOF-PCA": L, S, info = run_eof(X)
                elif mt == "PCP-L1": L, S, info = run_pcp(X)
                elif mt == "TV-RPCA":
                    from A5_tvrpca import run_tvrpca                  # line-by-line port of the official MATLAB code
                    B3, F3, info = run_tvrpca(b["observed"])
                    L, S = B3.reshape(T, -1).T, F3.reshape(T, -1).T
                elif mt == "TV-RPCA-M":
                    from A5_tvrpca import run_tvrpca                  # ranked row uses M = E + F
                    B3, F3, info = run_tvrpca(b["observed"], return_all=True)
                    L, S = B3.reshape(T, -1).T, info["M"].reshape(T, -1).T
                    tv_extra = [("TV-RPCA-F", F3), ("TV-RPCA-E", info.pop("E"))]; info.pop("M")
                elif mt == "GLOSS":
                    from A5_gloss import run_gloss_field, ee_score_field    # port of the first author's code + parameters from paper p.16–17; port validated in A3
                    L3g, S3g, info = run_gloss_field(b["observed"])
                    L, S = L3g.reshape(T, -1).T, S3g.reshape(T, -1).T
                    info["ee_scores"] = ee_score_field(S3g)
                elif mt == "dumb-zscore": L = S = None; info = dict(rank="", iters="", converged="")
                else: raise KeyError(mt)
                sec = time.perf_counter() - t0
                extra = []
                if mt == "FLoS-fixed" and info["snap"] is not None:
                    Ls, Ss, ds = info["snap"]
                    extra.append(("FLoS-fixed@10it", score(b, S3=to3(Ss), L3=to3(Ls)),
                                  dict(rank=info["rank"], iters=FLOS["paper_iters"], converged="paper setting, last change %.1e" % ds)))
                if mt == "TV-RPCA-M":
                    for nm_, X3_ in tv_extra:
                        extra.append((nm_, score(b, S3=X3_, L3=to3(L)),
                                      dict(rank=info["rank"], iters=info["iters"], converged="reported only")))
                    del tv_extra
                if mt == "GLOSS":
                    extra.append(("GLOSS-EE", score(b, scores=info["ee_scores"], L3=to3(L)),
                                  dict(rank="", iters=info["iters"], converged="paper score: robust Mahalanobis distance per yearly fibre")))
                sc = score(b, scores=run_dumb_zscore(b["observed"], months)) if mt == "dumb-zscore" \
                    else score(b, S3=to3(S), L3=to3(L))
                for mname, sc2, inf2 in [(mt, sc, info)] + extra:
                  row = dict(product=p, method=mname, seed=sd, T=T, impl=IMPL.get(mt, "own implementation"), seconds=round(sec, 1),
                           rank=inf2["rank"], iters=inf2["iters"], converged=inf2["converged"],
                           auprc=round(sc2["auprc"], 5), prec_at_k=round(sc2["prec_at_k"], 5),
                           pos_rate=round(sc2["pos_rate"], 6), n_pos=sc2["n_pos"],
                           spill=round(sc2.get("spill", float("nan")), 4),
                           bg_relerr=round(sc2.get("bg_relerr", float("nan")), 4))
                  new = not os.path.exists(OUT)
                  with open(OUT, "a", newline="") as fh:
                      w = csv.DictWriter(fh, fieldnames=list(row.keys()))
                      if new: w.writeheader()
                      w.writerow(row)
                  print("%-6s seed %d %-16s AUPRC %.4f  P@k %.4f  rank %s  iters %s converged %s  spill %s  %.0f s"
                        % (p, sd, mname, row["auprc"], row["prec_at_k"], row["rank"], row["iters"], row["converged"],
                           row["spill"], sec), flush=True)
            del b, X

if __name__ == "__main__":
    main()
