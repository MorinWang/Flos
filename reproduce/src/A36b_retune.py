#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A36b: re-select FLoS (and FLoS-C) on the tapered benchmark.

Benchmark identical to A36 (make_benchmark_tapered, D = 3).  Configs are s:alpha:lam triples for FLoS and FLoS-C.
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
ap.add_argument("--flos_configs", default="0.25:0.85:0.18,0.25:0.85:0.36,0.25:0.85:0.72,0.25:0.85:1.44,"
                "0.25:7.5:0.18,0.25:7.5:0.36,0.25:7.5:0.72,0.25:7.5:1.44,1:67.5:0.18,1:67.5:0.36,1:67.5:0.72,1:67.5:1.44")
ap.add_argument("--core_configs", default="0.25:0.85:0.09,0.25:0.85:0.18,0.25:0.85:0.36,0.25:0.85:0.72,"
                "0.25:202.5:0.09,0.25:202.5:0.18,0.25:202.5:0.36,0.25:202.5:0.72")
ap.add_argument("--shard", required=True)
a = ap.parse_args()
L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
D_TAPER, MIN_CELLS = 3, 3
parse = lambda s: [tuple(float(v) for v in c.split(":")) for c in s.split(",") if c]
FCFG, CCFG = parse(a.flos_configs), parse(a.core_configs)


def make_benchmark_tapered(path, seed=0, n_events=60):      # identical to A36_taper.make_benchmark_tapered
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    az = np.abs(z); thr = float(np.quantile(az[az > 0], QTHRESH))
    real_mask = az > thr
    lab, nlab = ndimage.label(real_mask, structure=ndimage.generate_binary_structure(3, 1))
    pool = []
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:, 0]).argmax())
        sl = idx[idx[:, 0] == t0][:, 1:]
        if len(sl) < MIN_CELLS: continue
        pool.append((t0, sl))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); support = np.zeros(z.shape, dtype=bool); taper = np.zeros(z.shape, dtype=bool)
    for _ in range(n_events):
        t0, sl = pool[rng.integers(len(pool))]
        ks = [k for k in range(-(t0 // 12), (T - 1 - t0) // 12 + 1) if k != 0]
        tn = t0 + 12 * int(rng.choice(ks))
        m = np.zeros((nlat, nlon), bool); m[sl[:, 0], sl[:, 1]] = True
        mp = np.concatenate([m[:, -D_TAPER:], m, m[:, :D_TAPER]], axis=1)
        dist = ndimage.distance_transform_edt(~mp)[:, D_TAPER:D_TAPER + nlon]
        w = np.where(dist == 0, 1.0, np.where(dist <= D_TAPER, np.cos(0.5 * np.pi * dist / (D_TAPER + 1)) ** 2, 0.0))
        S[tn] += w * z[t0]; support[tn] |= m; taper[tn] |= (w > 0) & ~m
    return dict(observed=z + S, background=z, S=S, support=support, score_mask=~real_mask & ~(taper & ~support),
                real_mask=real_mask, lat=lat, lon=lon)


OUT = "%s/A36b_%s.csv" % (OUTDIR, a.shard)
if not os.path.exists(OUT): open(OUT, "w").write("source,product,seed,method,s,alpha,lam,auprc,nnz_frac,iters\n")
done = {(r["product"], int(r["seed"]), r["method"], float(r["s"]), float(r["alpha"]), float(r["lam"])) for r in csv.DictReader(open(OUT))}
geo = None; filt = {}; Hs = {}
for p in a.products.split(","):
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in map(int, a.seeds.split(",")):
        todo = [("FLoS",) + c for c in FCFG if (p, sd, "FLoS") + c not in done] + [("CORE",) + c for c in CCFG if (p, sd, "CORE") + c not in done]
        if not todo: continue
        b = make_benchmark_tapered(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        if geo != (nlat, nlon):
            ev, U, node_of = L.graph_eig(b["lat"], b["lon"]); geo = (nlat, nlon); filt = {}; Hs = {}
            node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEV); n = int(node.max()) + 1
            cnt = np.bincount(np.asarray(node_of), minlength=n).astype(np.float64)
        X = b["observed"].reshape(T, -1).T.copy(); Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        r = L.mp_rank(X) if any(t[0] == "FLoS" for t in todo) else None
        for meth, s_, al, lm in todo:
            if meth == "FLoS":
                if (s_, al) not in filt:
                    if len(filt) >= 2: filt.pop(next(iter(filt)))
                    filt[(s_, al)] = FT.SphereFilterT(U, 1.0 / (1.0 + al * np.power(ev + 1e-12, s_)), node_of, device=DEV, dtype=torch.float32)
                _, S, info = FT.run_flos(Xt, filt[(s_, al)], lam=lm, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                                         paper_iters=L.FLOS["paper_iters"], rank=r); iters = info["iters"]
            else:
                if (s_, al) not in Hs:
                    Hs.clear()
                    Ud = torch.as_tensor(U, dtype=torch.float64, device=DEV)
                    M = (Ud * torch.as_tensor(al * np.power(ev + 1e-12, s_), dtype=torch.float64, device=DEV)) @ Ud.T
                    M.diagonal().add_(torch.as_tensor(cnt, dtype=torch.float64, device=DEV))
                    Hs[(s_, al)] = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (M + M.T))).to(torch.float32); del Ud, M
                Mi = Hs[(s_, al)]
                def H(Y):
                    acc = torch.zeros(n, Y.shape[1], dtype=Y.dtype, device=Y.device); acc.index_add_(0, node, Y); return (Mi @ acc)[node]
                S = torch.zeros_like(Xt); Lb = torch.zeros_like(Xt); nx = float(torch.linalg.norm(Xt))
                for iters in range(1, 2001):
                    Ln = H(Xt - S); R = Xt - Ln; Sn = torch.sign(R) * torch.clamp(torch.abs(R) - lm, min=0.0)
                    d = float(max(torch.linalg.norm(Ln - Lb), torch.linalg.norm(Sn - S)) / nx); Lb, S = Ln, Sn
                    if d < 1e-5: break
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT, "a").write("%s,%s,%d,%s,%g,%g,%g,%.5f,%.5f,%d\n" % (a.source, p, sd, meth, s_, al, lm, row["auprc"],
                                 float((S != 0).float().mean()), iters))
            print("  %-5s %-6s seed%d %-4s s=%g a=%g l=%g AUPRC %.5f" % (a.source, p, sd, meth, s_, al, lm, row["auprc"]), flush=True)
            del S
        del Xt; torch.cuda.empty_cache()
print("A36b shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
