#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A35: arrays for one qualitative example figure. No metric is reported from it.

20CRv3 sea-level pressure, benchmark seed 0 (held-out variable of a reanalysis on which no parameter was selected).
Runs FLoS at the frozen setting (lambda=0.72) and the convex core FLoS-C at its selected lambda (0.36), and saves,
for the month with the most inserted cells between 60S and 60N (a fixed rule, stated in the caption), the observed field, both backgrounds, both event channels
and the inserted support.  Output: $FLOS_OUT/A35_example.npz
"""
import os, numpy as np, torch
from paths import DATA, OUT, CACHE, device
import A5_leaderboard as L
import flos_torch as FT

DEV = device()
L.CACHE = CACHE
L.RAW = DATA + "/20cr/clean"
path = os.path.join(L.RAW, "20cr_slp.mon.nc")
b = L.make_benchmark(path, seed=0, n_events=L.N_EVENTS)
T, nlat, nlon = b["observed"].shape
X = b["observed"].reshape(T, -1).T.copy()
r = L.mp_rank(X)
fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEV, dtype=torch.float32)
Xt = torch.as_tensor(X, dtype=torch.float32, device=DEV)
Lf, Sf, info = FT.run_flos(Xt, filt, lam=0.72, p=L.FLOS["p"], max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"],
                           paper_iters=L.FLOS["paper_iters"], rank=r)

# convex core: L = P (Omega + alpha D^s)^{-1} P' (X - S), S = soft(X - L)
ev, U, node_of = L.graph_eig(b["lat"], b["lon"])
node = torch.as_tensor(np.asarray(node_of), dtype=torch.long, device=DEV); n = int(node.max()) + 1
cnt = np.bincount(np.asarray(node_of), minlength=n).astype(np.float64)
Ud = torch.as_tensor(U, dtype=torch.float64, device=DEV)
M = (Ud * torch.as_tensor(L.FLOS["alpha"] * np.power(ev + 1e-12, L.FLOS["s"]), dtype=torch.float64, device=DEV)) @ Ud.T
M.diagonal().add_(torch.as_tensor(cnt, dtype=torch.float64, device=DEV))
Minv = torch.cholesky_inverse(torch.linalg.cholesky(0.5 * (M + M.T))).to(torch.float32); del Ud, M
def H(Y):
    acc = torch.zeros(n, Y.shape[1], dtype=Y.dtype, device=Y.device); acc.index_add_(0, node, Y)
    return (Minv @ acc)[node]
S = torch.zeros_like(Xt); Lc = torch.zeros_like(Xt)
for it in range(2000):
    Ln = H(Xt - S); R = Xt - Ln
    Sn = torch.sign(R) * torch.clamp(torch.abs(R) - 0.36, min=0.0)
    d = float(max(torch.linalg.norm(Ln - Lc), torch.linalg.norm(Sn - S)) / torch.linalg.norm(Xt))
    Lc, S = Ln, Sn
    if d < 1e-5: break

sup = b["support"]
band = (np.abs(b["lat"]) <= 60)[None, :, None]
t = int(np.argmax((sup & band).reshape(T, -1).sum(1)))
to2 = lambda M_: M_[:, t].reshape(nlat, nlon)
np.savez(os.path.join(OUT, "A35_example.npz"), lat=b["lat"], lon=b["lon"], month=t,
         X=to2(X), L_flos=to2(Lf.double().cpu().numpy()), S_flos=to2(Sf.double().cpu().numpy()),
         L_core=to2(Lc.double().cpu().numpy()), S_core=to2(S.double().cpu().numpy()),
         support=sup[t], E=b["S"][t], score_mask=b["score_mask"][t], rank=r, flos_iters=info["iters"], core_iters=it + 1)
print("saved month", t, "support cells", int(sup[t].sum()), "rank", r)
