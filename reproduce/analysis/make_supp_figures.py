#!/usr/bin/env python3
"""Figures of the supplementary material, from result files only.

  fig_rhoobs_null.pdf  label-free contrast rho_obs of the observed fields (A36) against the smoothness of the field
                       (background high-pass share, A33), with the value rho_obs takes on stationary Gaussian fields of
                       varying smoothness, where extremes have no special roughness (null curve; periodic grid with the
                       grid Laplacian and the same alpha, s; an approximation of the sphere).
Usage: python3 make_supp_figures.py <analysis/data dir> <output dir>
"""
import sys, os, csv, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
import matplotlib.font_manager as fm
for f_ in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "texgyretermes-*.ttf")): fm.fontManager.addfont(f_)
plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "font.size": 8, "axes.linewidth": 0.6,
                     "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False})
SRC = ["ncep", "era5", "20cr"]; NAME = {"ncep": "NCEP/NCAR", "era5": "ERA5", "20cr": "20CRv3"}
COL = {"ncep": "#0072B2", "era5": "#0072B2", "20cr": "#0072B2"}; MRK = {"ncep": "o", "era5": "s", "20cr": "^"}
rd = lambda p: list(csv.DictReader(open(p)))

ro = collections.defaultdict(dict)
for f in sorted(glob.glob(D + "/tpami/A36_contrast_*.csv")):
    for r in rd(f): ro[(r["source"], r["product"])].setdefault(r["seed"], float(r["rho_obs"]))
gb = collections.defaultdict(list)
for r in rd(D + "/tpami/A33_contrast.csv"): gb[(r["source"], r["product"])].append(float(r["gainB"]))
ks = [k for k in ro if len(ro[k]) == 3 and k in gb]

# null: stationary Gaussian fields with spectral slope nu on a periodic 72 x 144 grid, 200 months each
rng = np.random.default_rng(0); ny, nx, T = 72, 144, 200
ky = 2 * np.pi * np.fft.fftfreq(ny)[:, None]; kx = 2 * np.pi * np.fft.fftfreq(nx)[None, :]
mu = (4 - 2 * np.cos(ky) - 2 * np.cos(kx)) / 8.0; Kh = 0.85 * mu ** 0.25 / (1 + 0.85 * mu ** 0.25); k2 = ky ** 2 + kx ** 2
null = []
for nu in np.linspace(0, 4.5, 19):
    amp = (0.1 ** 2 + k2) ** (-nu / 2.0) if nu > 0 else np.ones_like(k2)
    B = np.real(np.fft.ifft2(np.fft.fft2(rng.standard_normal((T, ny, nx))) * amp))
    KB = np.real(np.fft.ifft2(np.fft.fft2(B) * Kh)); a = np.abs(B); M = a > np.quantile(a, 0.99)
    null.append(((KB ** 2).sum() / (B ** 2).sum(),
                 ((KB[M] ** 2).sum() / (B[M] ** 2).sum()) / ((KB[~M] ** 2).sum() / (B[~M] ** 2).sum())))
null = np.array(sorted(null))
fig, ax = plt.subplots(figsize=(3.5, 2.5))
ax.plot(null[:, 0], null[:, 1], color="0.35", lw=1.2, ls="--", label="null: Gaussian fields, no extreme structure")
for k in ks:
    ax.scatter([np.mean(gb[k])], [np.mean(list(ro[k].values()))], s=20, marker=MRK[k[0]], color=COL[k[0]], zorder=3)
for s_ in SRC: ax.scatter([], [], s=18, marker=MRK[s_], color=COL[s_], label=NAME[s_] + " (observed)")
ax.axhline(1, color="0.6", lw=0.6, ls=":")
ax.set_xscale("log"); ax.set_xlabel("high-pass energy share of the field (smoothness)")
ax.set_ylabel(r"label-free contrast $\rho_{\mathrm{obs}}$")
ax.legend(frameon=False, fontsize=6, loc="lower right"); ax.grid(True, color="#E6E6E6", lw=0.5); ax.set_axisbelow(True)
fig.savefig(OUT + "/fig_rhoobs_null.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
rs = spearmanr([np.mean(gb[k]) for k in ks], [np.mean(list(ro[k].values())) for k in ks]).correlation
print("fig_rhoobs_null: %d variables; Spearman(smoothness, rho_obs) = %.2f; null range %.2f-%.2f" % (len(ks), rs, null[:, 1].min(), null[:, 1].max()))
