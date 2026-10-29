#!/usr/bin/env python3
"""Check that the `flos` package reproduces the implementation behind the paper, on one benchmark instance.

Runs, on NCEP/NCAR air temperature with benchmark seed 0:
  * the research implementation (reproduce/src/flos_torch.py) and flos.FLoS, and compares their scores;
  * flos.FLoS without truncation (rank=0) and flos.FLoSCore (the convex core FLoS-C).
The AUPRC of each is printed next to the value reported in the paper's result files. Needs the NCEP data
(reproduce/data_prep/fetch_ncep.sh); set FLOS_DATA to the folder that contains raw/.

    PYTHONPATH=.:reproduce/src FLOS_DATA=./data python tests/test_reproduces_paper.py
"""
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "reproduce", "src")]
import flos
import A5_leaderboard as L          # benchmark generator, scoring and the research implementation
import flos_torch as FT
from paths import CACHE, device

DEV = device()
L.CACHE = CACHE
REPORTED = {"FLoS": 0.84225, "FLoS without truncation": 0.75505, "FLoS-C": None}   # paper result files, seed 0

b = L.make_benchmark(os.path.join(L.RAW, L.PRODUCTS["air"]), seed=0, n_events=L.N_EVENTS)
T, nlat, nlon = b["observed"].shape
X = b["observed"].reshape(T, -1).T.copy()
score = lambda S: L.score(b, scores=np.abs(S.T.reshape(T, nlat, nlon)))["auprc"]

# research implementation, exactly as in the experiment scripts
fn = L.SphereFilter(b["lat"], b["lon"], L.FLOS["alpha"], L.FLOS["s"])
filt = FT.SphereFilterT(fn.U, fn.inv, fn.node_of, device=DEV, dtype=torch.float32)
r = L.mp_rank(X)
_, S_ref, _ = FT.run_flos(torch.as_tensor(X, dtype=torch.float32, device=DEV), filt, lam=0.72, p=L.FLOS["p"],
                          max_iter=L.FLOS["max_iter"], tol=L.FLOS["tol"], paper_iters=L.FLOS["paper_iters"], rank=r)
S_ref = S_ref.double().cpu().numpy()

# the package
g = flos.Graph.from_latlon(b["lat"], b["lon"], cache_dir=CACHE)
m = flos.FLoS(g, device=DEV).fit(X)
S_pkg = m.S_.double().cpu().numpy()
noT = flos.FLoS(g, rank=0, device=DEV).fit(X)
core = flos.FLoSCore(g, device=DEV).fit(X)

print("device %s | rank: research %d, package %d" % (DEV, r, m.rank_))
print("max |S_research - S_package| = %.3g" % np.abs(S_ref - S_pkg).max())
rows = [("FLoS (research code)", score(S_ref), REPORTED["FLoS"]), ("FLoS (package)", score(S_pkg), REPORTED["FLoS"]),
        ("FLoS without truncation", score(noT.S_.double().cpu().numpy()), REPORTED["FLoS without truncation"]),
        ("FLoS-C", score(core.S_.double().cpu().numpy()), None)]
for name, a, ref in rows:
    print("%-26s AUPRC %.5f%s" % (name, a, "" if ref is None else "   (paper result files: %.5f)" % ref))
obj = np.asarray(core.objective_)
print("FLoS-C objective never increases: %s (%d rounds)" % (bool(np.all(np.diff(obj) <= 1e-6 * np.abs(obj[:-1]))), len(obj)))
assert m.rank_ == r and np.abs(S_ref - S_pkg).max() < 1e-4, "package and research implementation disagree"
print("OK")
