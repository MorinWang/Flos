#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A36: does the hard edge of transplanted patches manufacture the roughness contrast?

Runs the methods below on a sharp and a tapered version of the benchmark.
Tapered benchmark: identical random draws to A1pp_honest_benchmark.make_benchmark (same patches, same target years),
but each patch is inserted together with the original month's anomaly in its neighbourhood, multiplied by a window
w = 1 on the patch and cos^2 decaying to 0 over D = 3 cells outside it (longitude wraps).  The true support is the
patch; cells with 0 < w < 1 (the taper zone) are excluded from scoring.
Methods at their frozen main-table settings: FLoS (lam 0.72), FLoS-C (soft, lam 0.36), direct ranking, FRPCAG (90, 0.3).
Contrasts, with the exact symmetric operator K = I - H, H = P (Omega + alpha D^s)^-1 P':
  rho_K on the sharp and on the tapered benchmark; prefix dominance in the generalised eigenbasis D^s v = k Omega v
  (cumulative energy at distinct eigenvalue thresholds; the part of the field orthogonal to range(P) has gain 1 and is
  placed last); label-free rho_obs on the observed field: high-pass energy share at the field's own extreme cells
  (|z| above the 99th percentile) relative to the other cells.
"""
import os, sys, time, csv, math, argparse
import numpy as np, torch
from scipy import ndimage
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L
import flos_torch as FT
from A1p_honest_benchmark import load_anom, QTHRESH

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
ap.add_argument("--methods", default="FLoS,CORE,Sort,FRPCAG")
ap.add_argument("--shard", required=True)
ap.add_argument("--prefix", default="A36", help="output prefix; A38 robustness runs use A38")
a = ap.parse_args()
L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
D_TAPER = int(os.environ.get("A36_D", "3"))
AMP = float(os.environ.get("A38_AMP", "1"))           # A38: inserted values multiplied by AMP (1 = A36)
DETREND = os.environ.get("A38_DETREND", "0") == "1"   # A38: remove a per-cell linear trend before building the benchmark
MIN_CELLS = 3


def make_benchmark_tapered(path, seed=0, n_events=60):
    """Same draws as A1pp make_benchmark; tapered insertion. Returns the same keys plus 'sharp_S' for reference."""
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    if DETREND:
        tt = np.arange(T, dtype=np.float64) - 0.5 * (T - 1)
        z = z - np.tensordot(tt, z, axes=(0, 0))[None] / float(tt @ tt) * tt[:, None, None]
    az = np.abs(z); thr = float(np.quantile(az[az > 0], QTHRESH))
    real_mask = az > thr
    lab, nlab = ndimage.label(real_mask, structure=ndimage.generate_binary_structure(3, 1))
    pool = []
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:, 0]).argmax())
        sl = idx[idx[:, 0] == t0][:, 1:]
        if len(sl) < MIN_CELLS: continue
        pool.append((t0, sl, z[t0, sl[:, 0], sl[:, 1]].copy(), None))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); S_sharp = np.zeros_like(z)
    support = np.zeros(z.shape, dtype=bool); taper = np.zeros(z.shape, dtype=bool)
    for _ in range(n_events):
        t0, sl, vals, _u = pool[rng.integers(len(pool))]
        ks = [k for k in range(-(t0 // 12), (T - 1 - t0) // 12 + 1) if k != 0]
        tn = t0 + 12 * int(rng.choice(ks))
        m = np.zeros((nlat, nlon), bool); m[sl[:, 0], sl[:, 1]] = True
        mp = np.concatenate([m[:, -D_TAPER:], m, m[:, :D_TAPER]], axis=1)          # wrap in longitude
        dist = ndimage.distance_transform_edt(~mp)[:, D_TAPER:D_TAPER + nlon]
        w = np.where(dist == 0, 1.0, np.where(dist <= D_TAPER, np.cos(0.5 * np.pi * dist / (D_TAPER + 1)) ** 2, 0.0))
        S[tn] += AMP * w * z[t0]
        S_sharp[tn, sl[:, 0], sl[:, 1]] += AMP * vals
        support[tn] |= m
        taper[tn] |= (w > 0) & ~m
    score_mask = ~real_mask & ~(taper & ~support)
    return dict(observed=z + S, background=z, S=S, sharp_S=S_sharp, support=support, score_mask=score_mask,
                real_mask=real_mask, lat=lat, lon=lon, taper=taper, thr=thr)


# ----------------------------------------------------------------------------- operators
class Ops:
    def __init__(self, lat, lon):
        ev, U, node_of = L.graph_eig(lat, lon)
        self.node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEV)
        n = int(self.node.max()) + 1; self.n = n
        cnt = np.bincount(np.asarray(node_of), minlength=n).astype(np.float64)
        fn = L.SphereFilter(lat, lon, L.FLOS["alpha"], L.FLOS["s"])
        self.F = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEV, dtype=torch.float32)
        Ud = torch.as_tensor(U, dtype=torch.float64, device=DEV)
        Ds = (Ud * torch.as_tensor(np.power(ev + 1e-12, L.FLOS["s"]), dtype=torch.float64, device=DEV)) @ Ud.T
        Ds = 0.5 * (Ds + Ds.T)
        om = torch.as_tensor(cnt, dtype=torch.float64, device=DEV)
        M = L.FLOS["alpha"] * Ds; M.diagonal().add_(om)
        self.Minv = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (M + M.T)))           # float64
        isq = om.rsqrt()
        gp = "%s/geneig_%dx%d_s%g.npz" % (CACHE, len(lat), len(lon), L.FLOS["s"])
        if os.path.exists(gp):
            g = np.load(gp); kap_np, Q_np = g["kappa"], g["Q"]
        else:                                                                                   # D^s v = k Omega v, on the CPU
            from scipy import linalg as sla
            C = ((isq[:, None] * Ds) * isq[None, :]).cpu().numpy()
            kap_np, Q_np = sla.eigh(0.5 * (C + C.T), overwrite_a=True, check_finite=False)
            np.savez(gp, kappa=kap_np, Q=Q_np); del C
        self.kappa = torch.clamp(torch.as_tensor(kap_np, dtype=torch.float64, device=DEV), min=0.0)
        self.V = isq[:, None] * torch.as_tensor(Q_np, dtype=torch.float64, device=DEV)         # V' Omega V = I
        del Ud, Ds, M, Q_np
        self.Minv32 = self.Minv.to(torch.float32)
        # Spatial operator of FRPCAG (exponent 1), as in A32
        Uf = torch.as_tensor(U, dtype=torch.float32, device=DEV)
        self.Dsp = (Uf * torch.as_tensor(ev, dtype=torch.float32, device=DEV)) @ Uf.T
        self.w = (1.0 / torch.bincount(self.node, minlength=n).float())[:, None]
        del Uf

    def PT(self, Y):
        acc = torch.zeros(self.n, Y.shape[1], dtype=Y.dtype, device=Y.device); acc.index_add_(0, self.node, Y); return acc

    def H(self, Y, f32=False):
        Mi = self.Minv32 if f32 else self.Minv
        return (Mi @ self.PT(Y))[self.node]

    def K(self, Y): return Y - self.H(Y)

    def spatial(self, Y):
        z = self.Dsp @ (self.PT(Y) * self.w); return (z * self.w)[self.node]

    def modal(self, Y):
        """energy on generalised eigenmodes (sorted by kappa) and the energy orthogonal to range(P)."""
        c = self.V.T @ self.PT(Y)                       # coefficients in the orthonormal basis W = P V
        e = (c ** 2).sum(1)
        rest = float((Y ** 2).sum() - e.sum())
        return e, max(rest, 0.0)


def rho(ops, E, B):
    gE = float((ops.K(E) ** 2).sum() / (E ** 2).sum()); gB = float((ops.K(B) ** 2).sum() / (B ** 2).sum())
    return gE / gB


def dominance(ops, E, B):
    eE, rE = ops.modal(E); eB, rB = ops.modal(B)
    kap = ops.kappa
    order = torch.argsort(kap)
    k_sorted = kap[order]
    cE = torch.cumsum(eE[order], 0) / (eE.sum() + rE); cB = torch.cumsum(eB[order], 0) / (eB.sum() + rB)
    last = torch.ones_like(k_sorted, dtype=torch.bool); last[:-1] = (k_sorted[1:] - k_sorted[:-1]) > 1e-9 * (1 + k_sorted[:-1])
    cE, cB = cE[last], cB[last]                         # thresholds at distinct eigenvalues
    diff = cE - cB
    return float((diff <= 1e-12).double().mean()), float(torch.clamp(diff, min=0).max())


def soft(x, lam): return torch.sign(x) * torch.clamp(torch.abs(x) - lam, min=0.0)


def run_core(ops, X, lam=0.36, max_iter=2000, tol=1e-5):
    S = torch.zeros_like(X); Lb = torch.zeros_like(X); nx = float(torch.linalg.norm(X))
    for it in range(1, max_iter + 1):
        Ln = ops.H(X - S, f32=True); Sn = soft(X - Ln, lam)
        d = float(max(torch.linalg.norm(Ln - Lb), torch.linalg.norm(Sn - S)) / nx); Lb, S = Ln, Sn
        if d < tol: break
    return S, it


def chain_lap(T):
    Lt = torch.zeros(T, T, dtype=torch.float32, device=DEV); i = torch.arange(T - 1, device=DEV)
    Lt[i, i + 1] = -1; Lt[i + 1, i] = -1; Lt += torch.diag(-Lt.sum(1))
    return Lt / (2.0 - 2.0 * math.cos(math.pi * (T - 1) / T))


def frpcag(ops, X, Lt, gs=90.0, gt=0.3, max_iter=3000):
    tau = 1.0 / (2.0 * (gs + gt)); nx = float(torch.linalg.norm(X))
    U = X.clone(); Y = X.clone(); t = 1.0
    for it in range(1, max_iter + 1):
        G = 2.0 * gs * ops.spatial(Y) + 2.0 * gt * (Y @ Lt)
        R = X - (Y - tau * G)
        Un = X - torch.sign(R) * torch.clamp(torch.abs(R) - tau, min=0.0)
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t)); Y = Un + ((t - 1.0) / tn) * (Un - U)
        d = float(torch.linalg.norm(Un - U)) / nx; U, t = Un, tn
        if d < 1e-6: break
    return X - U, it


OUT = "%s/%s_taper_%s.csv" % (OUTDIR, a.prefix, a.shard)
OUTC = "%s/%s_contrast_%s.csv" % (OUTDIR, a.prefix, a.shard)
OUTQ = "%s/%s_ceiling_%s.csv" % (OUTDIR, a.prefix, a.shard)       # share of inserted cells whose |X| exceeds the threshold
if not os.path.exists(OUTQ):
    open(OUTQ, "w").write("source,product,seed,D,amp,detrend,n_support,above_sharp,above_taper\n")
if not os.path.exists(OUT):
    open(OUT, "w").write("source,product,seed,method,auprc,prec_at_k,seconds,iters\n")
if not os.path.exists(OUTC):
    open(OUTC, "w").write("source,product,seed,rhoK_sharp,rhoK_taper,dom_share_sharp,dom_viol_sharp,dom_share_taper,"
                          "dom_viol_taper,rho_obs,taper_excluded_frac\n")
done = {(r["product"], int(r["seed"]), r["method"]) for r in csv.DictReader(open(OUT))}
donec = {(r["product"], int(r["seed"])) for r in csv.DictReader(open(OUTC))}
ops = None; geo = None; Lt = None
methods = a.methods.split(",")
for p in a.products.split(","):
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in map(int, a.seeds.split(",")):
        todo = [m for m in methods if (p, sd, m) not in done]
        if not todo and (p, sd) in donec: continue
        b = make_benchmark_tapered(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        key = (nlat, nlon)
        if geo != key:
            t0 = time.time(); ops = Ops(b["lat"], b["lon"]); geo = key
            print("operators built in %.0fs" % (time.time() - t0), flush=True)
        if Lt is None or Lt.shape[0] != T: Lt = chain_lap(T)
        to2 = lambda A: torch.as_tensor(A.reshape(T, -1).T.copy(), dtype=torch.float64, device=DEV)
        if (p, sd) not in donec:
            E_t, E_s, Bm = to2(b["S"]), to2(b["sharp_S"]), to2(b["background"])
            ds, vs = dominance(ops, E_s, Bm); dt, vt = dominance(ops, E_t, Bm)
            M = to2(b["real_mask"].astype(float)); KB = ops.K(Bm)
            ro = float(((KB * M) ** 2).sum() / ((Bm * M) ** 2).sum()) / float(((KB * (1 - M)) ** 2).sum() / ((Bm * (1 - M)) ** 2).sum())
            open(OUTC, "a").write("%s,%s,%d,%.5f,%.5f,%.4f,%.4f,%.4f,%.4f,%.5f,%.6f\n" % (
                a.source, p, sd, rho(ops, E_s, Bm), rho(ops, E_t, Bm), ds, vs, dt, vt, ro,
                float((b["taper"] & ~b["support"]).mean())))
            del E_t, E_s, Bm, M, KB
            sup = b["support"]
            open(OUTQ, "a").write("%s,%s,%d,%d,%g,%d,%d,%.5f,%.5f\n" % (a.source, p, sd, D_TAPER, AMP, int(DETREND), int(sup.sum()),
                float((np.abs(b["background"] + b["sharp_S"])[sup] > b["thr"]).mean()), float((np.abs(b["observed"])[sup] > b["thr"]).mean())))
        X = b["observed"].reshape(T, -1).T.copy()
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        to3 = lambda Mx: Mx.T.reshape(T, nlat, nlon)
        r = L.mp_rank(X) if "FLoS" in todo else None
        for m in todo:
            t0 = time.time(); iters = 1
            if m == "FLoS":
                _, S, info = FT.run_flos(Xt, ops.F, lam=0.72, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                                         paper_iters=L.FLOS["paper_iters"], rank=r); iters = info["iters"]
            elif m == "CORE": S, iters = run_core(ops, Xt)
            elif m == "Sort": S = Xt
            elif m == "FRPCAG": S, iters = frpcag(ops, Xt, Lt)
            sec = time.time() - t0
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT, "a").write("%s,%s,%d,%s,%.5f,%.5f,%.1f,%d\n" % (a.source, p, sd, m, row["auprc"], row["prec_at_k"], sec, iters))
            print("  %-5s %-6s seed%d %-7s AUPRC %.5f %.0fs" % (a.source, p, sd, m, row["auprc"], sec), flush=True)
            del S
        del Xt
        torch.cuda.empty_cache()
print("A36 shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
