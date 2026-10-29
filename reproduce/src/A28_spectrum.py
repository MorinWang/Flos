#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Companion measurement for A28: which of the three reanalyses is smoother and which is lower-rank.
It addresses the concern that 20CR, which assimilates only surface pressure, might be smoother than NCEP and ERA5
and therefore favour a smoothness prior.
All three datasets share the same 2.5-degree grid and the same spherical graph, so the three quantities are
directly comparable:
  comp99   number of components needed to reach 99% of the cumulative singular-value energy of X; smaller = lower rank
  mp_rank  Marchenko-Pastur rank estimate, the same function as in the main pipeline
  rough50  energy-weighted median eigenvalue after projecting X onto the spherical-graph eigenbasis; larger = rougher
Same preprocessing as the main pipeline: X is the observed field of make_benchmark with seed 0, arranged as
grid point x month, with no additional normalisation."""
import sys, os, time, csv
import numpy as np
from scipy import linalg

DIAG11 = os.path.dirname(os.path.abspath(__file__))
from paths import DATA, OUT as OUTDIR, CACHE
sys.path.insert(0, DIAG11)
import A5_leaderboard as L
L.CACHE = CACHE

SRC = {
 "ncep": (DATA + "/raw", dict(L.PRODUCTS)),      # reuse the file names from A5_leaderboard rather than redefining them
 "era5": (DATA + "/era5/clean", {k: "era5_%s.mon.nc" % k for k in
    ["air","slp","pres","pottmp","rhum","omega","cape"]}),
 "20cr": (DATA + "/20cr/clean", {k: "20cr_%s.mon.nc" % k for k in
    ["air","slp","pres","pottmp","rhum","omega","cape"]}),
}

OUT = os.path.join(OUTDIR, "A28_spectrum.csv")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
done = set()
if os.path.exists(OUT):
    for r in csv.DictReader(open(OUT)): done.add((r["source"], r["product"]))
else:
    open(OUT, "w").write("source,product,months,comp99,comp95,mp_rank,rough50,rough_mean,seconds\n")

for src, (raw, prods) in SRC.items():
    for p, fn in prods.items():
        if (src, p) in done: continue
        t0 = time.perf_counter()
        b = L.make_benchmark(os.path.join(raw, fn), seed=0, n_events=L.N_EVENTS)
        T, nlat, nlon = b["observed"].shape
        X = b["observed"].reshape(T, -1).T.copy()          # grid point x month
        sv = linalg.svdvals(X, check_finite=False)
        e = np.cumsum(sv ** 2) / np.sum(sv ** 2)
        comp99 = int(np.searchsorted(e, 0.99) + 1)
        comp95 = int(np.searchsorted(e, 0.95) + 1)
        r = L.mp_rank(X)
        # Build the graph per product: NCEP gpcp is on a 72x144 grid, unlike the 73x144 grid of the others.
        # graph_eig caches by a hash of lat/lon, so products on the same grid are computed only once.
        ev, U, node = L.graph_eig(b["lat"], b["lon"])
        n = int(node.max()) + 1
        cnt = np.bincount(node, minlength=n)
        acc = np.zeros((n, X.shape[1]))
        np.add.at(acc, node, X)
        acc /= cnt[:, None]
        C = U.T @ acc                                      # spectral coefficients
        w = (C ** 2).sum(1); w = w / w.sum()
        cw = np.cumsum(w)
        rough50 = float(ev[int(np.searchsorted(cw, 0.5))])
        rough_mean = float((w * ev).sum())
        sec = time.perf_counter() - t0
        open(OUT, "a").write("%s,%s,%d,%d,%d,%d,%.5f,%.5f,%.0f\n"
                             % (src, p, T, comp99, comp95, r, rough50, rough_mean, sec))
        print("  %-5s %-7s months %d  comp99 %4d  comp95 %4d  MP rank %3d  roughness median %.5f mean %.5f  %.0fs"
              % (src, p, T, comp99, comp95, r, rough50, rough_mean, sec), flush=True)
print("A28 spectrum measurement done", flush=True)
