#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A37: fairness and sensitivity checks.

--mode eof     EOF truncation at small ranks: score |X - L_r|, r in --ranks
--mode hp      one-shot spherical high-pass |X - F_{alpha,s} X| on an (s, alpha) grid, and low-mode removal
               |X - P U_k U_k' A X| (the k smoothest Laplacian eigenmodes) on a k grid
--mode rounds  FLoS at the frozen setting, AUPRC of the state at rounds 10, 50, 100, 250, 500, 1000
"""
import os, sys, time, csv, argparse
import numpy as np, torch
from scipy import linalg
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L
import flos_torch as FT

DEV = device()
SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
ap = argparse.ArgumentParser()
ap.add_argument("--mode", required=True, choices=["eof", "hp", "rounds"])
ap.add_argument("--source", required=True, choices=list(SRC))
ap.add_argument("--products", required=True)
ap.add_argument("--seeds", default="0,1,2")
ap.add_argument("--ranks", default="1,2,3,5,8,13,21,34")
ap.add_argument("--hp_configs", default="", help="s:alpha list; default = full grid")
ap.add_argument("--lm_configs", default="", help="k list; default = full grid")
ap.add_argument("--shard", required=True)
a = ap.parse_args()
L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
OUT = "%s/A37_%s_%s.csv" % (OUTDIR, a.mode, a.shard)
if not os.path.exists(OUT):
    open(OUT, "w").write("source,product,seed,family,param1,param2,auprc,prec_at_k\n")
done = {(r["product"], int(r["seed"]), r["family"], r["param1"], r["param2"]) for r in csv.DictReader(open(OUT))}
HP = [(s, al) for s in (0.25, 0.5, 1.0) for al in (0.85, 2.5, 7.5, 22.5, 67.5, 202.5, 607.5)]
LM = [4, 16, 64, 256, 1024, 4096]
if a.hp_configs: HP = [tuple(float(v) for v in c.split(":")) for c in a.hp_configs.split(",")]
if a.lm_configs: LM = [int(v) for v in a.lm_configs.split(",")]
geo = None
for p in a.products.split(","):
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in map(int, a.seeds.split(",")):
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy()
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        def emit(fam, p1, p2, S):
            row = L.score(b, scores=np.abs(to3(S)))
            open(OUT, "a").write("%s,%s,%d,%s,%s,%s,%.5f,%.5f\n" % (a.source, p, sd, fam, p1, p2, row["auprc"], row["prec_at_k"]))
            print("  %-5s %-6s seed%d %-7s %-6s %-6s AUPRC %.5f" % (a.source, p, sd, fam, p1, p2, row["auprc"]), flush=True)
        if a.mode == "eof":
            u, s, vt = linalg.svd(X, full_matrices=False, check_finite=False)
            for r in map(int, a.ranks.split(",")):
                if (p, sd, "EOF", str(r), "") in done: continue
                emit("EOF", r, "", X - (u[:, :r] * s[:r]) @ vt[:r])
        elif a.mode == "hp":
            if geo != (nlat, nlon):
                ev, U, node_of = L.graph_eig(b["lat"], b["lon"]); geo = (nlat, nlon)
                Ut = torch.as_tensor(U, dtype=torch.float32, device=DEV)
                node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEV); n = int(node.max()) + 1
                w = (1.0 / torch.bincount(node, minlength=n).float())[:, None]
                order = np.argsort(ev)
            Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
            acc = torch.zeros(n, T, dtype=torch.float32, device=DEV); acc.index_add_(0, node, Xt)
            C = Ut.T @ (acc * w)                                   # spectral coefficients of A X
            for s_, al in HP:
                if (p, sd, "HP", "%g" % s_, "%g" % al) in done: continue
                h = torch.as_tensor(1.0 / (1.0 + al * np.power(ev + 1e-12, s_)), dtype=torch.float32, device=DEV)[:, None]
                F = (Ut @ (C * h))[node]
                emit("HP", "%g" % s_, "%g" % al, (Xt - F).double().cpu().numpy())
            for k in LM:
                if (p, sd, "LOWMODE", str(k), "") in done: continue
                idx = torch.as_tensor(order[:k], device=DEV)
                F = (Ut[:, idx] @ C[idx])[node]
                emit("LOWMODE", k, "", (Xt - F).double().cpu().numpy())
            del Xt, acc, C
        elif a.mode == "rounds":
            if geo != (nlat, nlon):
                fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
                filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEV, dtype=torch.float32); geo = (nlat, nlon)
            if all((p, sd, "FLoS", str(k), "") in done for k in (10, 50, 100, 250, 500, 1000)): continue
            r = L.mp_rank(X)
            Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
            S = torch.zeros_like(Xt)
            for it in range(1, 1001):                                # FLoS iteration without the stopping rule
                Ln = FT.randomized_low_rank(filt(Xt - S), r)
                S = FT.adaptive_lorentz_shrink(Xt - Ln, 0.72, p=L.FLOS["p"])
                if it in (10, 50, 100, 250, 500, 1000):
                    emit("FLoS", it, "", S.double().cpu().numpy())
            del Xt, S
        torch.cuda.empty_cache()
print("A37 %s shard %s done %s" % (a.mode, a.shard, time.strftime("%F %T")), flush=True)
