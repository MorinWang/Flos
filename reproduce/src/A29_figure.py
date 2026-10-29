#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Ablation-only figure: how the score varies with the fraction of grid points retained; no competing methods are shown.
The x-axis is the retained fraction rather than lambda: lambda has a different meaning in the two arms, so only the
retained fraction puts them on a common axis.
One curve per (product, seed), interpolated in log(retained fraction) onto a common grid and then averaged; only the
range covered by all products is plotted."""
import os, csv, io, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from paths import OUT as O                 # all CSVs are read from, and the figure written to, $FLOS_OUT
D3 = P19 = O
OUT = os.path.join(O, "fig_sparsity.png")
EVENT_RATE = 1.6e-4          # fraction of grid points that are true events, per the benchmark specification

def rd(p): return list(csv.DictReader(io.open(p, encoding="utf-8")))

cur = collections.defaultdict(list)   # (src, arm, prod, seed) -> [(nnz, auprc)]
for p in glob.glob(D3 + "/A17_holdout_*.csv"):
    for r in rd(p):
        if r["arm"] not in ("A-adapt", "B-fixed"): continue
        arm = "full" if r["arm"] == "A-adapt" else "fixed"
        cur[(r["source"], arm, r["product"], r["seed"])].append((float(r["nnz_frac"]), float(r["auprc"])))
for arm in ("full", "fixed"):
    f = "%s/A29_%s.csv" % (P19, arm)
    if not os.path.exists(f): continue
    for r in rd(f):
        cur[("20cr", arm, r["product"], r["seed"])].append((float(r["nnz_frac"]), float(r["auprc"])))

def interp(pts, xs):
    pts = sorted(set(pts))
    x = np.log10([max(a, 1e-9) for a, _ in pts]); y = np.array([b for _, b in pts])
    o = np.argsort(x); x, y = x[o], y[o]
    out = np.interp(np.log10(xs), x, y, left=np.nan, right=np.nan)
    out[(xs < 10 ** x.min()) | (xs > 10 ** x.max())] = np.nan
    return out

TITLE = {"ncep": "NCEP", "era5": "ERA5", "20cr": "20CRv3"}
srcs = [s for s in ("ncep", "era5", "20cr") if any(k[0] == s for k in cur)]
fig, axes = plt.subplots(1, len(srcs), figsize=(4.2 * len(srcs), 3.5), sharey=True)
if len(srcs) == 1: axes = [axes]
summary = []
for ax, src in zip(axes, srcs):
    allp = sorted({k[2] for k in cur if k[0] == src and k[1] == "full"})
    # Exclude products whose sweep does not reach a retained fraction of 1%, consistent with how A17 handles
    # era5/cape; excluded products are printed
    prods = [p for p in allp
             if max(min(a for a, _ in cur[(src, arm, p, s2)])
                    for arm in ("full", "fixed") for s2 in "012"
                    if (src, arm, p, s2) in cur) <= 0.01]
    drop = [p for p in allp if p not in prods]
    if drop: print("  %s excluded (sweep does not reach 1%% retained): %s" % (src, ", ".join(drop)))
    ks = [k for k in cur if k[0] == src and k[2] in prods]
    lo = min(min(a for a, _ in cur[k]) for k in ks if min(a for a, _ in cur[k]) > 0)
    hi = max(max(a for a, _ in cur[k]) for k in ks)
    xs = np.logspace(np.log10(lo), np.log10(hi), 80)
    means = {}; Ys = {}
    for arm in ("full", "fixed"):
        Ys[arm] = np.vstack([interp(cur[(src, arm, p, s2)], xs)
                             for p in prods for s2 in "012" if (src, arm, p, s2) in cur])
    # Only plot the range where at least 80% of the curves of both arms have coverage, so that the mean at the
    # ends is not determined by a few products
    n = min(Ys[a2].shape[0] for a2 in Ys)
    okx = np.ones(len(xs), bool)
    for a2 in Ys: okx &= (np.sum(~np.isnan(Ys[a2]), axis=0) >= 0.8 * Ys[a2].shape[0])
    print("  %s plotting %d variables, retained fraction %.4g to %.4g" % (src, len(prods), xs[okx].max(), xs[okx].min()))
    for arm, col, lab in (("full", "#1f4e79", "FLoS (full)"), ("fixed", "#c0504d", "fixed threshold")):
        M = np.nanmean(Ys[arm], axis=0); M[~okx] = np.nan; means[arm] = M
        ax.plot(xs, M, color=col, lw=2, label=lab)
    ax.axvline(EVENT_RATE, color="0.5", ls=":", lw=1)
    ax.text(EVENT_RATE * 1.3, ax.get_ylim()[0] + 0.02, "true event rate", fontsize=7, color="0.4")
    ax.set_xscale("log"); ax.invert_xaxis()
    ax.set_title("%s  (%d variables)" % (TITLE[src], len(prods)), fontsize=10)
    ax.set_xlabel("fraction of grid points retained")
    ax.grid(alpha=.25, lw=.5)
    for tgt in (0.10, 0.01, 0.001):
        if lo <= tgt <= hi:
            i = int(np.argmin(abs(xs - tgt)))
            summary.append((src, tgt, means["full"][i], means["fixed"][i]))
axes[0].set_ylabel("AUPRC"); axes[0].legend(fontsize=8, frameon=False, loc="lower left")
fig.tight_layout(); fig.savefig(OUT, dpi=170)
print("figure written to", OUT)
print("\n%-6s %10s %10s %10s %10s" % ("data", "retained", "full", "fixed-thr", "diff"))
for s, t, a, b in summary:
    print("%-6s %9.1f%% %10.4f %10.4f %+10.4f" % (s, t * 100, a, b, a - b))
