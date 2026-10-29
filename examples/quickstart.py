#!/usr/bin/env python3
"""Detect the strongest events in a monthly field with FLoS.

    python examples/quickstart.py path/to/air.mon.mean.nc [--top 10] [--device cpu]

Reads a monthly NetCDF field on a latitude-longitude mesh (for example NCEP/NCAR air temperature from
reproduce/data_prep/fetch_ncep.sh), removes the calendar-month mean, runs FLoS with the settings of the paper,
prints the cells and months with the largest scores |S|, and saves a map of the month with the strongest event.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import flos

ap = argparse.ArgumentParser()
ap.add_argument("path")
ap.add_argument("--top", type=int, default=10)
ap.add_argument("--device", default=None)
ap.add_argument("--cache", default="cache", help="folder for the graph eigendecomposition (computed once per mesh)")
a = ap.parse_args()

Z, lat, lon, time, scale = flos.load_anomalies(a.path)
T, nlat, nlon = Z.shape
X = Z.reshape(T, -1).T                                   # cells x months
graph = flos.Graph.from_latlon(lat, lon, cache_dir=a.cache)
model = flos.FLoS(graph, device=a.device).fit(X)
print("rank %d, %d rounds, converged: %s" % (model.rank_, model.n_iter_, model.converged_))

scores = model.scores_.cpu().numpy().T.reshape(T, nlat, nlon)
top = np.argsort(scores.ravel())[::-1][:a.top]
print("%-9s %8s %8s %10s %10s" % ("month", "lat", "lon", "anomaly", "score"))
for i in top:
    t, y, x = np.unravel_index(i, scores.shape)
    print("%-9s %8.1f %8.1f %10.2f %10.2f" % (str(time[t])[:7], lat[y], lon[x], Z[t, y, x] * scale, scores[t, y, x]))

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    t = np.unravel_index(top[0], scores.shape)[0]
    L = model.L_.cpu().numpy().T.reshape(T, nlat, nlon)
    fig, axs = plt.subplots(1, 3, figsize=(12, 3))
    v = np.abs(Z[t]).max()
    for ax, F, title in zip(axs, (Z[t], L[t], scores[t] * np.sign(Z[t])), ("anomaly", "FLoS background", "FLoS events")):
        ax.pcolormesh(lon, lat, F, cmap="RdBu_r", vmin=-v, vmax=v, shading="auto"); ax.set_title(title)
    fig.suptitle(str(time[t])[:7]); fig.tight_layout(); fig.savefig("quickstart_events.png", dpi=120)
    print("map of %s saved to quickstart_events.png" % str(time[t])[:7])
except ImportError:
    pass
