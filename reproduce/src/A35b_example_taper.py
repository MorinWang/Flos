#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A35b: tapered counterpart of the qualitative example A35 (no metric is reported).

Same variable (20CRv3 sea-level pressure), seed (0) and month as A35_example.npz, but on the tapered benchmark of A36
(same random draws; each patch inserted with its own month's anomaly in a 3-cell cos^2 taper).  FLoS at the frozen
setting.  Output: $FLOS_OUT/A35b_example_taper.npz
"""
import os, numpy as np, torch
from scipy import ndimage
from paths import DATA, OUT, CACHE, device
import A5_leaderboard as L
import flos_torch as FT
from A1p_honest_benchmark import load_anom, QTHRESH

DEV = device(); L.CACHE = CACHE; D_TAPER, MIN_CELLS = 3, 3


def make_benchmark_tapered(path, seed=0, n_events=60):      # identical to A36_taper.make_benchmark_tapered
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    az = np.abs(z); thr = float(np.quantile(az[az > 0], QTHRESH)); real_mask = az > thr
    lab, nlab = ndimage.label(real_mask, structure=ndimage.generate_binary_structure(3, 1))
    pool = []
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:, 0]).argmax()); sl = idx[idx[:, 0] == t0][:, 1:]
        if len(sl) < MIN_CELLS: continue
        pool.append((t0, sl))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); support = np.zeros(z.shape, dtype=bool)
    for _ in range(n_events):
        t0, sl = pool[rng.integers(len(pool))]
        ks = [k for k in range(-(t0 // 12), (T - 1 - t0) // 12 + 1) if k != 0]
        tn = t0 + 12 * int(rng.choice(ks))
        m = np.zeros((nlat, nlon), bool); m[sl[:, 0], sl[:, 1]] = True
        mp = np.concatenate([m[:, -D_TAPER:], m, m[:, :D_TAPER]], axis=1)
        dist = ndimage.distance_transform_edt(~mp)[:, D_TAPER:D_TAPER + nlon]
        w = np.where(dist == 0, 1.0, np.where(dist <= D_TAPER, np.cos(0.5 * np.pi * dist / (D_TAPER + 1)) ** 2, 0.0))
        S[tn] += w * z[t0]; support[tn] |= m
    return dict(observed=z + S, S=S, support=support, lat=lat, lon=lon)


ref = np.load(os.path.join(OUT, "A35_example.npz")); t = int(ref["month"])
b = make_benchmark_tapered(os.path.join(DATA, "20cr/clean/20cr_slp.mon.nc"), seed=0, n_events=L.N_EVENTS)
T, nlat, nlon = b["observed"].shape
X = b["observed"].reshape(T, -1).T.copy(); r = L.mp_rank(X)
fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEV, dtype=torch.float32)
Lf, Sf, info = FT.run_flos(torch.as_tensor(X, dtype=torch.float32, device=DEV), filt, lam=0.72, p=L.FLOS["p"],
                           max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"], paper_iters=L.FLOS["paper_iters"], rank=r)
to2 = lambda M_: M_[:, t].reshape(nlat, nlon)
assert np.array_equal(b["support"][t], ref["support"]), "support differs from A35"
np.savez(os.path.join(OUT, "A35b_example_taper.npz"), lat=b["lat"], lon=b["lon"], month=t, X=to2(X),
         L_flos=to2(Lf.double().cpu().numpy()), S_flos=to2(Sf.double().cpu().numpy()), support=b["support"][t], E=b["S"][t])
print("saved tapered example, month", t, "rank", r)
