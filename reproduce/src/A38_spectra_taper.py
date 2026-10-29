#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""A38: roughness spectrum of the inserted events, hard-edged vs tapered (descriptive).

For every variable, seed 0: energy of the inserted event field per band of Laplacian eigenvalue mu (same bins, basis,
event months and normalisation as A33_spectra.npz), for the hard-edged insertion and for the tapered insertion of A36
(D = 3, identical draws).  No prediction; used for the roughness-spectrum figure.  CPU is enough.
Output: $FLOS_OUT/A38_spectra_taper.npz
"""
import os, numpy as np, torch
from scipy import ndimage
from paths import DATA, OUT, CACHE
import A5_leaderboard as L
import flos_torch as FT
from A1p_honest_benchmark import load_anom, QTHRESH

D_TAPER, MIN_CELLS = 3, 3
SRC = {"ncep": (DATA + "/raw", {"air": "air.mon.mean.nc", "slp": "slp.mon.mean.nc", "rhum": "rhum.sig995.mon.mean.nc",
                                "lftx": "lftx.sfc.mon.mean.nc", "pres": "pres.sfc.mon.mean.nc",
                                "omega": "omega.sig995.mon.mean.nc", "pottmp": "pottmp.sig995.mon.mean.nc",
                                "gpcp": "gpcp.mon.mean.nc"}),
       "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}),
       "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]})}
L.CACHE = CACHE


def bench(path, seed=0, n_events=60):        # identical draws to A36_taper.make_benchmark_tapered (D = 3)
    z, lat, lon, sg = load_anom(path)
    T, nlat, nlon = z.shape
    az = np.abs(z); thr = float(np.quantile(az[az > 0], QTHRESH)); real_mask = az > thr
    lab, nlab = ndimage.label(real_mask, structure=ndimage.generate_binary_structure(3, 1))
    pool = []
    for i, slc in enumerate(ndimage.find_objects(lab), start=1):
        idx = np.argwhere(lab[slc] == i) + np.array([s.start for s in slc])
        t0 = int(np.bincount(idx[:, 0]).argmax()); sl = idx[idx[:, 0] == t0][:, 1:]
        if len(sl) < MIN_CELLS: continue
        pool.append((t0, sl, z[t0, sl[:, 0], sl[:, 1]].copy()))
    rng = np.random.default_rng(seed)
    S = np.zeros_like(z); S_sharp = np.zeros_like(z); support = np.zeros(z.shape, dtype=bool)
    for _ in range(n_events):
        t0, sl, vals = pool[rng.integers(len(pool))]
        ks = [k for k in range(-(t0 // 12), (T - 1 - t0) // 12 + 1) if k != 0]
        tn = t0 + 12 * int(rng.choice(ks))
        m = np.zeros((nlat, nlon), bool); m[sl[:, 0], sl[:, 1]] = True
        mp = np.concatenate([m[:, -D_TAPER:], m, m[:, :D_TAPER]], axis=1)
        dist = ndimage.distance_transform_edt(~mp)[:, D_TAPER:D_TAPER + nlon]
        w = np.where(dist == 0, 1.0, np.where(dist <= D_TAPER, np.cos(0.5 * np.pi * dist / (D_TAPER + 1)) ** 2, 0.0))
        S[tn] += w * z[t0]; S_sharp[tn, sl[:, 0], sl[:, 1]] += vals; support[tn] |= m
    return S, S_sharp, support, lat, lon


out = {}; geo = None
BINS = np.logspace(-5, 0, 25)
for src, (raw, prods) in SRC.items():
    for p, fn in prods.items():
        S, Ss, sup, lat, lon = bench(os.path.join(raw, fn))
        T, nlat, nlon = S.shape
        if geo != (nlat, nlon):
            fl = L.SphereFilter(lat, lon, L.FLOS["alpha"], L.FLOS["s"])
            F = FT.SphereFilterT(fl.U, fl.inv, fl.node_of, device="cpu", dtype=torch.float64)
            ev = L.graph_eig(lat, lon)[0]; geo = (nlat, nlon)
        evm = np.unique(np.argwhere(sup)[:, 0])
        def modal(Y):
            Yt = torch.as_tensor(Y.reshape(T, -1).T[:, evm].copy(), dtype=torch.float64)
            acc = torch.zeros(F.n, Yt.shape[1], dtype=torch.float64); acc.index_add_(0, F.node, Yt)
            return ((F.U.T @ (acc * F.w)) ** 2).sum(1).numpy()
        for tag, Y in (("hE_sharp", Ss), ("hE_taper", S)):
            h = np.histogram(ev, bins=BINS, weights=modal(Y))[0]; out["%s_%s_%s" % (src, p, tag)] = h / h.sum()
        print(src, p, "done", flush=True)
out["bins"] = BINS
np.savez(os.path.join(OUT, "A38_spectra_taper.npz"), **out)
print("A38 spectra done")
