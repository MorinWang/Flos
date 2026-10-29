#!/usr/bin/env python3
"""Record how FLoS separates one month round by round (data for assets/flos_iterations.gif).

Uses the example of the paper: 20CRv3 sea-level pressure, benchmark seed 0, and the month with the most inserted
cells between 60S and 60N. Saves the observed month, its background and its event component after selected rounds.

    PYTHONPATH=.:reproduce/src FLOS_DATA=./data python examples/record_iterations.py out.npz
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "reproduce", "src")]
import flos
import A5_leaderboard as L
from paths import DATA, CACHE, device

L.CACHE = CACHE
b = L.make_benchmark(os.path.join(DATA, "20cr", "clean", "20cr_slp.mon.nc"), seed=0, n_events=L.N_EVENTS)
T, nlat, nlon = b["observed"].shape
X = b["observed"].reshape(T, -1).T.copy()
t = int(np.argmax((b["support"] & (np.abs(b["lat"]) <= 60)[None, :, None]).reshape(T, -1).sum(1)))
keep = sorted(set(list(range(1, 13)) + [15, 20, 25, 30, 40, 50, 75, 100, 150, 200, 300, 500]))
frames = {}

def record(k, Lk, Sk):
    if k in keep:
        frames[k] = (Lk[:, t].double().cpu().numpy().reshape(nlat, nlon), Sk[:, t].double().cpu().numpy().reshape(nlat, nlon))

g = flos.Graph.from_latlon(b["lat"], b["lon"], cache_dir=CACHE)
m = flos.FLoS(g, device=device()).fit(X, callback=record)
ks = sorted(frames)
np.savez(sys.argv[1] if len(sys.argv) > 1 else "flos_iterations.npz", lat=b["lat"], lon=b["lon"], month=t,
         X=X[:, t].reshape(nlat, nlon), support=b["support"][t], rounds=np.array(ks),
         L=np.stack([frames[k][0] for k in ks]), S=np.stack([frames[k][1] for k in ks]), n_iter=m.n_iter_)
print("month", t, "rounds recorded", ks[-1], "of", m.n_iter_)
