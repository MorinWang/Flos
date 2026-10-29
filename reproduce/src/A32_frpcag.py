#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A32: Fast Robust PCA on Graphs (Shahid et al., IEEE JSTSP 2016) on our benchmark.

Runs FRPCAG as a baseline.  FRPCAG replaces the nuclear norm of RPCA by graph
Tikhonov terms on both modes:
    min_U  ||X - U||_1 + g_s tr(U' Lsp U) + g_t tr(U Ltm U'),     event score |X - U|.
We give it our own spherical graph (the FLoS graph with exponent 1, i.e. Lsp = A' D A on cells, eigenvalues of D
scaled to [0,1]) and a chain graph over months (Ltm, eigenvalues scaled to [0,1]); g_t = 0 removes the temporal term.
Solver: FISTA as in the paper (smooth part = the two graph terms, prox of the l1 fidelity), step 1/(2(g_s+g_t)),
at most 3000 iterations, stop when ||U_k - U_{k-1}||/||X|| < 1e-6.  Parameters: grid on the three NCEP development
variables, frozen for the 19 held-out variables (same procedure as every method in the main table).
"""
import sys, os, time, csv, argparse, math
import numpy as np, torch
from paths import DATA, OUT as OUTDIR, CACHE, device
import A5_leaderboard as L

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
ap.add_argument("--configs", default="", help="explicit g_s:g_t pairs (held-out stage)")
ap.add_argument("--gs_grid", default="0.1,0.3,1,3,10,30")
ap.add_argument("--gt_grid", default="0,0.3,3")
ap.add_argument("--max_iter", type=int, default=3000)
ap.add_argument("--sexp", type=float, default=1.0, help="exponent of the spatial Laplacian (A37f: 0.25)")
ap.add_argument("--shard", required=True)
a = ap.parse_args()

L.CACHE = CACHE
L.RAW, L.PRODUCTS = SRC[a.source]
if a.configs:
    CFG = [tuple(float(v) for v in c.split(":")) for c in a.configs.split(",")]
else:
    CFG = [(gs, gt) for gs in map(float, a.gs_grid.split(",")) for gt in map(float, a.gt_grid.split(","))]
OUT = "%s/A32_frpcag_%s.csv" % (OUTDIR, a.shard) if a.sexp == 1.0 else "%s/A37_frpcagfrac_%s.csv" % (OUTDIR, a.shard)
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)):
        done.add((r["product"], float(r["g_s"]), float(r["g_t"]), int(r["seed"])))
else:
    open(OUT, "w").write("source,product,g_s,g_t,seed,seconds,iters,converged,objective,auprc,prec_at_k\n")


class Spatial:
    """u -> A' D A u on cells (A averages the cells of a node).  ||A' D A|| <= ||D|| = 1."""
    def __init__(self, lat, lon):
        ev, U, node_of = L.graph_eig(lat, lon)
        Ud = torch.as_tensor(U, dtype=torch.float32, device=DEVICE)
        self.D = (Ud * torch.as_tensor(np.power(ev + 1e-12, a.sexp) if a.sexp != 1.0 else ev, dtype=torch.float32, device=DEVICE)) @ Ud.T
        del Ud
        self.node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEVICE)
        self.n = int(self.node.max()) + 1
        self.w = (1.0 / torch.bincount(self.node, minlength=self.n).float())[:, None]

    def __call__(self, Y):
        acc = torch.zeros(self.n, Y.shape[1], dtype=Y.dtype, device=Y.device)
        acc.index_add_(0, self.node, Y)
        z = self.D @ (acc * self.w)                    # D A Y
        return (z * self.w)[self.node]                 # A' (D A Y)


def chain_laplacian(T):
    Lt = torch.zeros(T, T, dtype=torch.float32, device=DEVICE)
    i = torch.arange(T - 1, device=DEVICE)
    Lt[i, i + 1] = -1; Lt[i + 1, i] = -1
    Lt += torch.diag(-Lt.sum(1))
    return Lt / (2.0 - 2.0 * math.cos(math.pi * (T - 1) / T))   # largest eigenvalue of the path graph -> 1


def frpcag(X, sp, Lt, gs, gt):
    tau = 1.0 / (2.0 * (gs + gt))
    nx = float(torch.linalg.norm(X))
    U = X.clone(); Y = X.clone(); t = 1.0; conv = False
    for it in range(1, a.max_iter + 1):
        G = 2.0 * gs * sp(Y)
        if gt > 0: G = G + 2.0 * gt * (Y @ Lt)
        V = Y - tau * G
        R = X - V
        Un = X - torch.sign(R) * torch.clamp(torch.abs(R) - tau, min=0.0)     # prox of tau ||X - .||_1
        tn = 0.5 * (1.0 + math.sqrt(1.0 + 4.0 * t * t))
        Y = Un + ((t - 1.0) / tn) * (Un - U)
        d = float(torch.linalg.norm(Un - U)) / nx
        U, t = Un, tn
        if d < 1e-6:
            conv = True; break
    obj = float(torch.sum(torch.abs(X - U)).double()) + gs * float(torch.sum((U * sp(U)).double()))
    if gt > 0: obj += gt * float(torch.sum((U * (U @ Lt)).double()))
    return X - U, dict(iters=it, converged=conv, objective=obj)


prods = a.products.split(","); seeds = [int(x) for x in a.seeds.split(",")]
print("A32 shard %s source %s products %s seeds %s configs %s device %s" % (a.shard, a.source, prods, seeds, CFG, DEVICE),
      flush=True)
sp = None; geo = None; Lt = None
for p in prods:
    path = os.path.join(L.RAW, L.PRODUCTS[p])
    for sd in seeds:
        todo = [c for c in CFG if (p, c[0], c[1], sd) not in done]
        if not todo: continue
        b = L.make_benchmark(path, seed=sd, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy()
        key = (len(b["lat"]), len(b["lon"]))
        if geo != key:
            sp = Spatial(b["lat"], b["lon"]); geo = key
        if Lt is None or Lt.shape[0] != T: Lt = chain_laplacian(T)
        Xt = torch.as_tensor(X, dtype=torch.float32, device=DEVICE)
        to3 = lambda M: M.T.reshape(T, nlat, nlon)
        for gs, gt in todo:
            SYNC(); t0 = time.perf_counter()
            S, info = frpcag(Xt, sp, Lt, gs, gt)
            SYNC(); sec = time.perf_counter() - t0
            row = L.score(b, scores=np.abs(to3(S.double().cpu().numpy())))
            open(OUT, "a").write("%s,%s,%g,%g,%d,%.1f,%d,%s,%.6g,%.5f,%.5f\n" % (
                a.source, p, gs, gt, sd, sec, info["iters"], info["converged"], info["objective"],
                row["auprc"], row["prec_at_k"]))
            print("  %-5s %-6s seed%d g_s=%-5g g_t=%-4g AUPRC %.5f iters %d conv %s %.0fs" % (
                a.source, p, sd, gs, gt, row["auprc"], info["iters"], info["converged"], sec), flush=True)
            del S
        del Xt
        if DEVICE == "cuda": torch.cuda.empty_cache()
print("A32 shard %s done %s" % (a.shard, time.strftime("%F %T")), flush=True)
