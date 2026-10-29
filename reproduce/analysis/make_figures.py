#!/usr/bin/env python3
"""Data figures of the TPAMI manuscript, from result CSVs only (no experiment is run here).

  fig_spectrum.pdf  (a) normalised singular values of the 22 observed fields; (b) energy per roughness band,
                    background vs inserted events (median over variables, interquartile band)       <- A33, A28
  fig_contrast.pdf  rho_K (A33) against the AUPRC gain of FLoS over magnitude ranking (main table)
  fig_sparsity.pdf  AUPRC against retained fraction, adaptive vs fixed threshold, three reanalyses  <- A17, A29
                    (same aggregation as the original A29_figure.py)
Usage: python3 make_figures.py <analysis/data dir> <output dir>
Colours: Okabe-Ito blue / vermillion / bluish green in fixed order (NCEP, ERA5, 20CRv3); validated with the dataviz
validator (all checks pass, worst CVD dE 11.0). Marker shape is the second encoding.
"""
import sys, os, csv, glob, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "STIXGeneral", "mathtext.fontset": "stix", "font.size": 8,
                     "axes.labelsize": 8, "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
                     "ytick.labelsize": 7, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "pdf.fonttype": 42, "ps.fonttype": 42,
                     "axes.spines.top": False, "axes.spines.right": False})
import matplotlib.font_manager as fm
for f_ in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "fonts", "texgyretermes-*.ttf")):
    fm.fontManager.addfont(f_)                       # TrueType conversion of TeX Gyre Pagella (otf2ttf.py)
plt.rcParams.update({"font.family": "TeX Gyre Termes", "mathtext.fontset": "stix", "axes.unicode_minus": True})
SRC = ["ncep", "era5", "20cr"]
NAME = {"ncep": "NCEP/NCAR", "era5": "ERA5", "20cr": "20CRv3"}
# one encoding in every figure: colour = method (FLoS blue, FLoS-C green, baselines grey), shape = reanalysis,
# filled = hard-edged insertion, open = tapered; reanalysis lines (no method) in three greys
COL = {"ncep": "#1F1F1F", "era5": "#6E6E6E", "20cr": "#ABABAB"}
FLOSC = "#0072B2"
MRK = {"ncep": "o", "era5": "s", "20cr": "^"}
VAR = {"air": "air temp.", "slp": "SLP", "pres": "sfc. pressure", "pottmp": "pot. temp.", "rhum": "rel. humidity",
       "omega": "vert. velocity", "cape": "CAPE", "lftx": "lifted index", "gpcp": "precip."}
GRID = "#E6E6E6"


def rd(p): return list(csv.DictReader(open(p)))


# ------------------------------------------------------------------ fig_spectrum
z = np.load(D + "/tpami/A33_spectra.npz")
spec = {(r["source"], r["product"]): r for r in rd(D + "/ablation/A28_spectrum.csv")}
fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.16, 2.25), gridspec_kw=dict(wspace=0.28))
for k in sorted(spec):
    sv = z["%s_%s_sv" % k]; j = np.arange(1, len(sv) + 1); mp = int(spec[k]["mp_rank"])
    a1.plot(j, sv, color=COL[k[0]], lw=0.7, alpha=0.75)
    a1.plot([mp], [sv[mp - 1]], marker="|", ms=6, mew=1.0, color=COL[k[0]])
a1.set_xscale("log"); a1.set_yscale("log"); a1.set_xlim(1, 1000); a1.set_ylim(3e-3, 1.1)
a1.set_xlabel("component index $j$ (ordered by variance)"); a1.set_ylabel(r"$\sigma_j/\sigma_1$")
a1.grid(True, which="major", color=GRID, lw=0.5); a1.set_axisbelow(True)
for s in SRC: a1.plot([], [], color=COL[s], lw=1.2, label=NAME[s])
a1.plot([], [], color="0.3", marker="|", ls="none", ms=6, label="estimated rank $r$")
a1.legend(frameon=False, loc="lower left", handlelength=1.6)
a1.set_title("(a) variance spectrum: no gap to truncate at", loc="left")
bins = z["bins"]; ctr = np.sqrt(bins[1:] * bins[:-1])
HB = np.vstack([z["%s_%s_hB" % k] for k in sorted(spec)]); HE = np.vstack([z["%s_%s_hE" % k] for k in sorted(spec)])
zt = np.load(D + "/tpami/A38_spectra_taper.npz") if os.path.exists(D + "/tpami/A38_spectra_taper.npz") else None
curves = [(HB, "#444444", "-", "background (observed field)"), (HE, "#D55E00", "--", "extremes, hard-edged insertion")]
if zt is not None:
    HT = np.vstack([zt["%s_%s_hE_taper" % k] for k in sorted(spec)])
    curves.append((HT, "#CC79A7", ":", "same extremes, tapered insertion"))
for H, col, ls, lab in curves:
    q1, q2, q3 = np.percentile(H, [25, 50, 75], axis=0)
    a2.fill_between(ctr, q1, q3, color=col, alpha=0.15, lw=0)
    a2.plot(ctr, q2, color=col, ls=ls, lw=1.3, label=lab)
a2.set_xscale("log"); a2.set_xlim(1e-5, 1)
a2.set_xlabel("roughness $\\mu$ (Laplacian eigenvalue)"); a2.set_ylabel("fraction of energy per bin")
a2.grid(True, which="major", color=GRID, lw=0.5); a2.set_axisbelow(True)
a2.legend(frameon=False, loc="upper right", fontsize=6.5)
a3 = a2.twinx(); mu_ = np.logspace(-5, 0, 200); g2 = (0.85 * mu_ ** 0.25 / (1 + 0.85 * mu_ ** 0.25)) ** 2
a3.plot(mu_, g2, color="0.25", lw=0.8, ls=(0, (1, 1.2))); a3.set_ylim(0, 0.25); a3.tick_params(labelsize=6.5, colors="0.35")
a3.set_ylabel(r"squared gain of $\boldsymbol{K}$ (dotted grey)", fontsize=7, color="0.35"); a3.spines["right"].set_visible(True)
a2.set_title("(b) roughness spectrum: the hard edge moves energy to rough modes", loc="left")
fig.savefig(OUT + "/fig_spectrum.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ fig_contrast
mt = {(r["source"], r["product"]): r for r in rd(D + "/main/main_table.csv")}
agg = collections.defaultdict(list)
exact = sorted(glob.glob(D + "/tpami/A36_contrast_*.csv"))
for f in exact:
    for r in rd(f): agg[(r["source"], r["product"])].append(float(r["rhoK_sharp"]))
if len([k for k in agg if len(agg[k]) == 3]) < 22:
    agg = collections.defaultdict(list)
    for r in rd(D + "/tpami/A33_contrast.csv"): agg[(r["source"], r["product"])].append(float(r["rho_K"]))
    RHO_SRC = "A33 (operator F)"
else:
    agg = {k: v for k, v in agg.items()}
    RHO_SRC = "exact (operator K = I - H)"
print("contrast figure uses", RHO_SRC)
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
a33 = collections.defaultdict(list)
for r in rd(D + "/tpami/A33_contrast.csv"): a33[(r["source"], r["product"])].append(float(r["gainB"]))
fig, (ax, ax2) = plt.subplots(1, 2, figsize=(3.8, 1.9), gridspec_kw=dict(wspace=0.12), sharey=True)
xs, ys = [], []
for k in sorted(agg):
    x = np.mean(agg[k]); y = float(mt[k]["FLoS"]) - float(mt[k]["Sort"]); xs.append(x); ys.append(y)
    for a_, xv in ((ax, x), (ax2, np.mean(a33[k]))):
        a_.scatter([xv], [y], s=14, marker=MRK[k[0]], facecolor=FLOSC if k not in DEV else "white",
                   edgecolor=FLOSC, linewidth=0.7, zorder=3)
for a_ in (ax, ax2):
    a_.axhline(0, color="0.55", lw=0.5, zorder=1); a_.grid(True, color=GRID, lw=0.4); a_.set_axisbelow(True)
    a_.tick_params(labelsize=6)
ax.axvline(1, color="0.55", lw=0.5, ls=":", zorder=1)
sub = [i for i, k in enumerate(sorted(agg)) if k[0] != "ncep"]
rho = spearmanr(xs, ys).correlation; rho14 = spearmanr([xs[i] for i in sub], [ys[i] for i in sub]).correlation
xb = [np.mean(a33[k]) for k in sorted(agg)]
rb = spearmanr(xb, ys).correlation; rb14 = spearmanr([xb[i] for i in sub], [ys[i] for i in sub]).correlation
mn = lambda v: ("%.2f" % v).replace("-", "\u2212")
ax.text(0.3, 0.97, "Spearman %s\n(%s on the 14)" % (mn(rho), mn(rho14)), transform=ax.transAxes, va="top", fontsize=5.6, color="0.2")
ax2.text(0.96, 0.97, "Spearman %s\n(%s on the 14)" % (mn(rb), mn(rb14)), transform=ax2.transAxes, va="top", ha="right", fontsize=5.6, color="0.2")
ax.set_xlabel(r"contrast $\rho_{\boldsymbol{K}}$", fontsize=7, labelpad=1)
ax2.set_xlabel(r"background high-pass share", fontsize=7, labelpad=1)
ax.set_ylabel("gain over magnitude ranking", fontsize=7, labelpad=1)
ax.set_title("(a)", loc="left", fontsize=7); ax2.set_title("(b)", loc="left", fontsize=7)
for s_ in SRC: ax.scatter([], [], s=12, marker=MRK[s_], color=FLOSC, label=NAME[s_])
ax.scatter([], [], s=12, marker="o", facecolor="white", edgecolor="0.3", label="development")
ax.legend(frameon=False, loc="lower right", fontsize=5.2, handletextpad=0.1, borderaxespad=0.05, labelspacing=0.2)
print("fig_contrast: rho_K Spearman %.3f (14: %.3f); background share %.3f (14: %.3f)" % (rho, rho14, rb, rb14))
fig.savefig(OUT + "/fig_contrast.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)

# ------------------------------------------------------------------ fig_sparsity (aggregation of A29_figure.py)
cur = collections.defaultdict(list)
for p in glob.glob(D + "/heldout/A17_holdout_[ne]*.csv"):
    for r in rd(p):
        if r["arm"] not in ("A-adapt", "B-fixed"): continue
        arm = "full" if r["arm"] == "A-adapt" else "fixed"
        cur[(r["source"], arm, r["product"], r["seed"])].append((float(r["nnz_frac"]), float(r["auprc"])))
for arm in ("full", "fixed"):
    for r in rd(D + "/ablation/A29_%s.csv" % arm):
        cur[("20cr", arm, r["product"], r["seed"])].append((float(r["nnz_frac"]), float(r["auprc"])))


def interp(pts, xq):
    pts = sorted(set(pts)); x = np.log10([max(a, 1e-9) for a, _ in pts]); y = np.array([b for _, b in pts])
    o = np.argsort(x); x, y = x[o], y[o]
    out = np.interp(np.log10(xq), x, y, left=np.nan, right=np.nan)
    out[(xq < 10 ** x.min()) | (xq > 10 ** x.max())] = np.nan
    return out


fig, axes = plt.subplots(1, 3, figsize=(3.73, 1.8), sharey=True, gridspec_kw=dict(wspace=0.08))
rows = []
for ax, src in zip(axes, SRC):
    allp = sorted({k[2] for k in cur if k[0] == src and k[1] == "full"})
    prods = [p for p in allp if max(min(a for a, _ in cur[(src, arm, p, s2)]) for arm in ("full", "fixed")
                                    for s2 in "012" if (src, arm, p, s2) in cur) <= 0.01]
    ks = [k for k in cur if k[0] == src and k[2] in prods]
    lo = min(min(a for a, _ in cur[k]) for k in ks if min(a for a, _ in cur[k]) > 0)
    hi = max(max(a for a, _ in cur[k]) for k in ks)
    xq = np.logspace(np.log10(lo), np.log10(hi), 80)
    Ys = {arm: np.vstack([interp(cur[(src, arm, p, s2)], xq) for p in prods for s2 in "012" if (src, arm, p, s2) in cur])
          for arm in ("full", "fixed")}
    ok = np.ones(len(xq), bool)
    for a_ in Ys: ok &= (np.sum(~np.isnan(Ys[a_]), axis=0) >= 0.8 * Ys[a_].shape[0])
    M = {}
    for arm, col, ls, lb in (("full", "#0072B2", "-", "adaptive threshold (FLoS)"), ("fixed", "#666666", "--", "fixed (soft) threshold")):
        m = np.nanmean(Ys[arm], axis=0); m[~ok] = np.nan; M[arm] = m
        ax.plot(xq, m, color=col, ls=ls, lw=1.1, label=lb)
    nzv = [float(r["nnz"]) for r in rd(D + "/main/main_table_long.csv") if r["method"] == "FLoS" and r["source"] == src and (r["source"], r["product"]) not in DEV and r["nnz"] not in ("", "nan")]
    q1_, q3_ = np.percentile(nzv, [25, 75])
    ax.axvspan(q1_, q3_, color="#0072B2", alpha=0.08, lw=0)
    print("  operating band %s: IQR %.3f-%.3f (median %.3f)" % (src, q1_, q3_, np.median(nzv)))
    ax.set_xscale("log"); ax.set_xlim(0.9, 6e-4)
    ax.set_title("%s (%d)" % (NAME[src], len(prods)), loc="left", fontsize=6.5)
    ax.tick_params(labelsize=5.5); ax.set_xticks([1e-1, 1e-2, 1e-3])
    ax.grid(True, color=GRID, lw=0.5); ax.set_axisbelow(True)
    for t in (0.2, 0.1, 0.01, 0.001):
        i = int(np.argmin(abs(xq - t)))
        if ok[i]: rows.append((src, t, M["full"][i], M["fixed"][i]))
axes[0].set_ylabel("AUPRC", fontsize=7, labelpad=1); axes[1].set_xlabel("fraction of entries retained", fontsize=7, labelpad=1)
axes[0].text(0.55, 0.06, "operating\nrange", fontsize=5, color="#0072B2", ha="left", transform=axes[0].get_xaxis_transform())
axes[2].legend(frameon=False, loc="lower left", fontsize=5, handlelength=1.5, borderaxespad=0.1)
fig.savefig(OUT + "/fig_sparsity.pdf", bbox_inches="tight", pad_inches=0.02)
plt.close(fig)
print("sparsity summary (src, retained, adaptive, fixed, diff):")
for r in rows: print("  %-5s %6.1f%%  %.4f  %.4f  %+.4f" % (r[0], 100 * r[1], r[2], r[3], r[2] - r[3]))
print("contrast Spearman 22: %.3f  14: %.3f" % (rho, rho14))

# ------------------------------------------------------------------ fig_example (A35/A35b): one month, two insertions
ex, ext_ = D + "/tpami/A35_example.npz", D + "/tpami/A35b_example_taper.npz"
if os.path.exists(ex) and os.path.exists(ext_):
    import json
    e, et = np.load(ex), np.load(ext_)
    lat, lon = e["lat"], e["lon"]
    coast = json.load(open(D + "/ne_110m_coastline.geojson"))
    def draw_coast(ax):
        for f in coast["features"]:
            c = np.asarray(f["geometry"]["coordinates"]); x = np.mod(c[:, 0], 360.0); y = c[:, 1]
            br = np.where(np.abs(np.diff(x)) > 180)[0] + 1
            for sx, sy in zip(np.split(x, br), np.split(y, br)): ax.plot(sx, sy, color="0.35", lw=0.3)
    lo_ = np.where(lon > 180, lon - 360, lon); oo = np.argsort(lo_)
    def draw_coast(ax):
        for f in coast["features"]:
            c = np.asarray(f["geometry"]["coordinates"]); ax.plot(c[:, 0], c[:, 1], color="0.35", lw=0.3)
    extent = [-180, 180, -90, 90]
    org = "upper" if lat[0] > lat[-1] else "lower"
    vf = np.percentile(np.abs(e["X"]), 99); ve = max(np.abs(e["S_flos"]).max(), np.abs(et["S_flos"]).max())
    fig, axes = plt.subplots(2, 3, figsize=(8.7, 2.9), gridspec_kw=dict(wspace=0.04, hspace=0.18))
    rows = [("hard-edged", e), ("tapered", et)]
    ims = {}
    for r_, (lab, d) in enumerate(rows):
        for c_, (key, kind, ttl) in enumerate((("X", "field", "observed anomaly $\\boldsymbol{X}$"), ("L_flos", "field", "FLoS background $\\boldsymbol{L}$"), ("S_flos", "event", "FLoS events $\\boldsymbol{S}$"))):
            ax = axes[r_, c_]; v_ = vf if kind == "field" else ve
            im = ax.imshow(d[key][:, oo], extent=extent, origin=org, cmap="RdBu_r", vmin=-v_, vmax=v_, interpolation="nearest", aspect="auto")
            draw_coast(ax); ims[kind] = im
            ax.contour(lo_[oo], lat, d["support"][:, oo].astype(float), levels=[0.5], colors="k", linewidths=0.6)
            ax.set_xlim(-180, 180); ax.set_ylim(-90, 90)
            ax.set_xticks([]); ax.set_yticks([])
            for sp in ax.spines.values(): sp.set_visible(True); sp.set_linewidth(0.4)
            if r_ == 0: ax.set_title(ttl, fontsize=7, loc="left")
            if c_ == 0: ax.set_ylabel(lab + " insertion", fontsize=7)
    for kind, cols, lab in (("field", [axes[1, 0], axes[1, 1]], "anomaly (global robust units)"), ("event", [axes[1, 2]], "event channel")):
        cb = fig.colorbar(ims[kind], ax=cols, orientation="horizontal", fraction=0.07, pad=0.06, aspect=40)
        cb.ax.tick_params(labelsize=6, width=0.4, length=2); cb.outline.set_linewidth(0.4); cb.set_label(lab, fontsize=6)
    fig.savefig(OUT + "/fig_example.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("fig_example written (2 insertions); month", int(e["month"]))

# ------------------------------------------------------------------ fig_taper (A36): sharp vs tapered insertion
tfiles = sorted(glob.glob(D + "/tpami/A36_taper_*.csv")); cfiles = sorted(glob.glob(D + "/tpami/A36_contrast_*.csv"))
if tfiles and cfiles:
    tt = collections.defaultdict(lambda: collections.defaultdict(list)); seen = set()
    for f in tfiles:
        for r in rd(f):
            key = (r["source"], r["product"], r["seed"], r["method"])
            if key in seen: continue
            seen.add(key); tt[r["method"]][(r["source"], r["product"])].append(float(r["auprc"]))
    cc = collections.defaultdict(list); seenc = set()
    for f in cfiles:
        for r in rd(f):
            key = (r["source"], r["product"], r["seed"])
            if key in seenc: continue
            seenc.add(key); cc[key[:2]].append(float(r["rho_obs"]))
    HOk = [k for k in sorted(mt) if k not in DEV and len(tt["FLoS"].get(k, [])) == 3 and len(tt["Sort"].get(k, [])) == 3]
    if len(HOk) == 19:
        sharp = {k: float(mt[k]["FLoS"]) - float(mt[k]["Sort"]) for k in HOk}
        taper = {k: np.mean(tt["FLoS"][k]) - np.mean(tt["Sort"][k]) for k in HOk}
        order_k = sorted(HOk, key=lambda k: sharp[k])
        fig, (b1, b2, b3) = plt.subplots(1, 3, figsize=(7.73, 2.7), gridspec_kw=dict(width_ratios=[1.35, 1, 1], wspace=0.42))
        ypos = np.arange(len(order_k))
        for i, k in enumerate(order_k):
            b1.plot([sharp[k], taper[k]], [i, i], color="0.75", lw=0.8, zorder=1)
            b1.scatter([sharp[k]], [i], s=20, marker=MRK[k[0]], color=FLOSC, zorder=3)
            b1.scatter([taper[k]], [i], s=20, marker=MRK[k[0]], facecolor="white", edgecolor=FLOSC, linewidth=0.9, zorder=3)
        b1.axvline(0, color="0.45", lw=0.7)
        b1.set_yticks(ypos); b1.set_yticklabels(["%s %s" % (NAME[k[0]].split("/")[0] if k != ("ncep", "gpcp") else "GPCP", VAR[k[1]] if k != ("ncep", "gpcp") else "") for k in order_k], fontsize=6)
        b1.set_xlabel("AUPRC gain of FLoS over magnitude ranking")
        b1.scatter([], [], s=20, marker="o", color=FLOSC, label="hard-edged"); b1.scatter([], [], s=20, marker="o", facecolor="white", edgecolor=FLOSC, label="tapered")
        b1.legend(frameon=False, loc="lower left", bbox_to_anchor=(0.0, 1.06), ncol=2, fontsize=6.5, handletextpad=0.2, borderaxespad=0)
        b1.grid(True, axis="x", color=GRID, lw=0.5); b1.set_axisbelow(True)
        b1.set_title("(a) the same extremes, two ways of inserting them", loc="left")
        # (b) dose-response over taper width; (c) amplitude and detrending (A38)
        def ho_gain(files, meth, ref="Sort"):
            g = collections.defaultdict(lambda: collections.defaultdict(dict))
            for f in files:
                for r in rd(f):
                    g[r["method"]][(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
            ks = [k for k in HOk if len(g[meth].get(k, {})) == 3 and len(g[ref].get(k, {})) == 3]
            if len(ks) < 19: return None
            return np.array([np.mean(list(g[meth][k].values())) - np.mean(list(g[ref][k].values())) for k in ks])
        def ci(d, n=4000, seed=0):
            rng = np.random.default_rng(seed); m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
            return d.mean(), np.percentile(m, 2.5), np.percentile(m, 97.5)
        core0 = collections.defaultdict(dict)
        for f in sorted(glob.glob(D + "/tpami/A30_core_ho_soft_*.csv")):
            for r in rd(f): core0[(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
        widths = {0: (np.array([sharp[k] for k in HOk]), np.array([np.mean(list(core0[k].values())) - float(mt[k]["Sort"]) for k in HOk])),
                  3: (ho_gain(tfiles, "FLoS"), ho_gain(tfiles, "CORE"))}
        for dd in (1, 2, 5):
            fs = sorted(glob.glob(D + "/tpami/A38_taper_d%d_*.csv" % dd))
            if fs: widths[dd] = (ho_gain(fs, "FLoS"), ho_gain(fs, "CORE"))
        DOSE = {}
        for dd in sorted(widths):
            for name, arr, off, col, mk in (("FLoS", widths[dd][0], -0.06, "#0072B2", "o"), ("FLoS-C", widths[dd][1], 0.06, "#009E73", "s")):
                if arr is None: continue
                m, lo, hi = ci(arr); DOSE[(name, dd)] = (m, lo, hi)
        for name, col, mk, off in (("FLoS", "#0072B2", "o", -0.07), ("FLoS-C", "#009E73", "s", 0.07)):
            pts = sorted((dd, v) for (nm, dd), v in DOSE.items() if nm == name)
            if not pts: continue
            xx = np.array([p_[0] for p_ in pts]) + off; mm = np.array([p_[1][0] for p_ in pts])
            b2.errorbar(xx, mm, yerr=[mm - np.array([p_[1][1] for p_ in pts]), np.array([p_[1][2] for p_ in pts]) - mm],
                        color=col, marker=mk, ms=3.5, lw=1.0, capsize=1.5, elinewidth=0.7, label=name)
        b2.axhline(0, color="0.45", lw=0.7)
        b2.set_xticks([0, 1, 2, 3, 5]); b2.set_xlabel("taper width (grid cells; 0 = hard edge)")
        b2.set_ylabel("held-out gain over magnitude ranking")
        b2.grid(True, color=GRID, lw=0.5); b2.set_axisbelow(True); b2.legend(frameon=False, loc="upper right", fontsize=6.5)
        b2.set_title("(b) dose-response", loc="left")
        groups = [("original", widths[0][0], widths[3][0])]
        for lab_, f0, f3 in (("half amplitude", "amp05d0", "amp05d3"), ("detrended", "detd0", "detd3")):
            g0 = sorted(glob.glob(D + "/tpami/A38_taper_%s_*.csv" % f0)); g3 = sorted(glob.glob(D + "/tpami/A38_taper_%s_*.csv" % f3))
            groups.append((lab_, ho_gain(g0, "FLoS") if g0 else None, ho_gain(g3, "FLoS") if g3 else None))
        ROB = {}
        for gi, (lab_, h0, h3) in enumerate(groups):
            for arr, off, filled in ((h0, -0.15, True), (h3, 0.15, False)):
                if arr is None: continue
                m, lo, hi = ci(arr); ROB[(lab_, filled)] = (m, lo, hi)
                b3.errorbar([gi + off], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=4, color="#0072B2",
                            mfc="#0072B2" if filled else "white", mec="#0072B2", capsize=1.5, elinewidth=0.8)
        b3.axhline(0, color="0.45", lw=0.7)
        b3.set_xticks(range(len(groups))); b3.set_xticklabels({"original": "original", "half amplitude": "half\namplitude", "detrended": "detrended"}.get(g_[0], g_[0]) for g_ in groups); b3.tick_params(axis="x", labelsize=6.3)
        b3.set_xlim(-0.6, len(groups) - 0.4); b3.grid(True, axis="y", color=GRID, lw=0.5); b3.set_axisbelow(True)
        b3.errorbar([], [], fmt="o", ms=4, color="#0072B2", label="hard-edged"); b3.errorbar([], [], fmt="o", ms=4, color="#0072B2", mfc="white", label="tapered")
        b3.legend(frameon=False, loc="upper right", fontsize=6.5)
        b3.set_title("(c) robustness (FLoS)", loc="left")
        print("fig_taper dose-response:", {k_: tuple(round(x_, 4) for x_ in v_) for k_, v_ in DOSE.items()})
        print("fig_taper robustness:", {k_: tuple(round(x_, 4) for x_ in v_) for k_, v_ in ROB.items()})
        fig.savefig(OUT + "/fig_taper.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
        print("fig_taper written; mean sharp %.4f taper %.4f" % (np.mean(list(sharp.values())), np.mean(list(taper.values()))))

# ------------------------------------------------------------------ fig_forest: every method against magnitude ranking
def per_var(files, filt=lambda r: True, keyf=lambda r: r.get("seed")):
    g = collections.defaultdict(dict)
    for f in sorted(files):
        for r in rd(f):
            if filt(r): g[(r["source"], r["product"])].setdefault(keyf(r), float(r["auprc"]))
    return {k: np.mean(list(v.values())) for k, v in g.items() if len(v) == 3}
mtl = collections.defaultdict(list)
for r in rd(D + "/main/main_table_long.csv"): mtl[(r["source"], r["product"], r["method"])].append(float(r["auprc"]))
HO19 = [k for k in sorted(mt) if k not in DEV]
srt = {k: float(mt[k]["Sort"]) for k in HO19}
col_main = lambda m: {k: np.mean(mtl[k + (m,)]) for k in HO19}
new = D + "/tpami/"
rowsA = [("Published and adapted baselines", None, None),
         ("EOF, rank re-sel. ($r=1$)", per_var(glob.glob(new + "A37_eof_*.csv"), lambda r: r["param1"] == "1"), "#999999"),
         ("PCP", col_main("PCP"), "#999999"), ("GLOSS, abs. scale (re-sel.)", per_var(glob.glob(new + "A37_gloss_ho_*.csv")), "#999999"),
         ("ScaledGD", col_main("ScaledGD"), "#999999"), ("LR-STSS", col_main("Mondal"), "#999999"),
         ("FRPCAG", per_var(glob.glob(new + "A32_frpcag_ho_*.csv")), "#999999"),
         ("FRPCAG, $g_s=810$", per_var(glob.glob(new + "A32_frpcag_hox_*.csv")), "#999999"),
         ("Smoothness-based variants", None, None),
         ("one high-pass step $|X-FX|$", per_var(glob.glob(new + "A30_core_hp?.csv"), lambda r: r["arm"] == "HP1"), "#009E73"),
         ("one high-pass step (re-sel.)", per_var(glob.glob(new + "A37_hp_hox_*.csv"), lambda r: r["family"] == "HP"), "#009E73"),
         ("remove smoothest mode (re-sel.)", per_var(glob.glob(new + "A37_hp_hox_*.csv"), lambda r: r["family"] == "LOWMODE"), "#009E73"),
         ("FLoS-C", per_var(glob.glob(new + "A30_core_ho_soft_*.csv")), "#009E73"),
         ("FLoS-C (re-sel.)", per_var(glob.glob(new + "A30b_core_ho_*.csv")), "#009E73"),
         ("FLoS, $s=1$ (re-sel.)", per_var(glob.glob(new + "A31_exponent_hos1_*.csv")), "#0072B2"),
         ("FLoS (re-sel.)", per_var(glob.glob(new + "A31_exponent_lam1_*.csv")), "#0072B2"),
         ("FLoS", col_main("FLoS"), "#0072B2")]
tt2 = collections.defaultdict(lambda: collections.defaultdict(dict))
for f in sorted(glob.glob(new + "A36_taper_*.csv")):
    for r in rd(f): tt2[r["method"]][(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
tb = collections.defaultdict(lambda: collections.defaultdict(dict))
for f in sorted(glob.glob(new + "A36b_ho_*.csv")):
    for r in rd(f): tb[r["method"]][(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
mean3 = lambda g: {k: np.mean(list(v.values())) for k, v in g.items() if len(v) == 3}
srt_t = mean3(tt2["Sort"])
rowsB = [("FRPCAG", mean3(tt2["FRPCAG"]), "#999999"), ("FLoS-C", mean3(tt2["CORE"]), "#009E73"),
         ("FLoS-C (re-sel. on tapered)", mean3(tb["CORE"]), "#009E73"), ("FLoS", mean3(tt2["FLoS"]), "#0072B2"),
         ("FLoS (re-sel. on tapered)", mean3(tb["FLoS"]), "#0072B2")]


def boot(d, n=4000):
    rng = np.random.default_rng(1); m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return np.percentile(m, 2.5), np.percentile(m, 97.5)


XL = (-0.32, 0.25)
TAP = {"FRPCAG": rowsB[0][1], "FLoS-C": rowsB[1][1], "FLoS-C (re-sel.)": rowsB[2][1], "FLoS": rowsB[3][1],
       "FLoS (re-sel.)": rowsB[4][1]}           # tapered counterpart of each row (re-selected rows: on tapered dev.)


def draw(ax, y, vals, ref, col, filled, dy=0.0, label=None):
    ks = [k for k in HO19 if k in vals and k in ref]
    if len(ks) < 19: return False
    d = np.array([vals[k] - ref[k] for k in ks]); m = d.mean(); lo, hi = boot(d)
    if m < XL[0]:
        ax.annotate("", xy=(XL[0], y + dy), xytext=(XL[0] + 0.05, y + dy), arrowprops=dict(arrowstyle="->", color=col, lw=1.0))
        ax.text(XL[0] + 0.055, y + dy, "%.2f" % m, va="center", fontsize=5.6, color=col); return True
    if filled:
        inside = d[(d > XL[0]) & (d < XL[1])]
        ax.scatter(inside, np.full(len(inside), y + dy) + np.random.default_rng(int(y * 10)).uniform(-0.12, 0.12, len(inside)),
                   s=2.5, color=col, alpha=0.35, lw=0, zorder=2)
    ax.plot([lo, hi], [y + dy, y + dy], color=col, lw=1.4 if filled else 1.0, solid_capstyle="round", zorder=3,
            ls="-" if filled else (0, (2, 1)))
    ax.scatter([m], [y + dy], s=15 if filled else 14, marker="o" if filled else "D", facecolor=col if filled else "white",
               edgecolor=col, linewidth=0.8, zorder=4)
    return True


fig, ax = plt.subplots(figsize=(3.0, 3.55))
labels, y = [], 0
for lab, vals, col in rowsA[::-1]:
    if vals is None:
        labels.append((y, lab, True)); y += 1; continue
    ok = draw(ax, y, vals, srt, col, True, dy=0.17 if lab in TAP else 0.0)
    if lab in TAP: draw(ax, y, TAP[lab], srt_t, col, False, dy=-0.17)
    labels.append((y, lab if ok else lab + " (pending)", False)); y += 1
# a separate row for FLoS re-selected on the tapered benchmark
ax.axvline(0, color="0.3", lw=0.7, zorder=1)
ax.set_yticks([l_[0] for l_ in labels]); ax.set_yticklabels([l_[1] for l_ in labels], fontsize=6)
for t_, l_ in zip(ax.get_yticklabels(), labels):
    if l_[2]: t_.set_fontweight("bold"); t_.set_fontstyle("italic")
ax.set_ylim(-0.6, y - 0.4); ax.set_xlim(*XL); ax.tick_params(axis="y", length=0); ax.tick_params(axis="x", labelsize=6.5)
ax.grid(True, axis="x", color=GRID, lw=0.5); ax.set_axisbelow(True)
ax.set_xlabel("held-out AUPRC difference to magnitude ranking", fontsize=7)
ax.scatter([], [], s=15, marker="o", color="0.35", label="hard-edged insertion")
ax.scatter([], [], s=14, marker="D", facecolor="white", edgecolor="0.35", label="tapered insertion")
ax.legend(frameon=False, loc="upper right", fontsize=5.8, handletextpad=0.2, borderaxespad=0.2)
fig.savefig(OUT + "/fig_forest.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
print("fig_forest written")

# ------------------------------------------------------------------ fig_bench: where the extremes are; the magnitude ceiling (A38)
zf = D + "/tpami/A38_pool_zonal.npz"
if os.path.exists(zf):
    zz = np.load(zf)
    fig, (c1, c2) = plt.subplots(1, 2, figsize=(3.93, 1.95), gridspec_kw=dict(width_ratios=[1, 1.15], wspace=0.42))
    keys = sorted({k.rsplit("_", 1)[0] for k in zz.files})
    lat0 = zz[keys[0] + "_lat"]; area = np.cos(np.deg2rad(lat0)).clip(min=0); area = area / area.sum()
    shares = np.vstack([zz[k + "_count"] / zz[k + "_count"].sum() for k in keys if len(zz[k + "_lat"]) == len(lat0)])
    q1, q2, q3 = np.percentile(shares, [25, 50, 75], axis=0)
    c1.fill_betweenx(lat0, q1, q3, color="#0072B2", alpha=0.2, lw=0)
    c1.plot(q2, lat0, color="#0072B2", lw=1.1, label="extremes (median)")
    c1.plot(area, lat0, color="0.35", lw=0.9, ls="--", label="area")
    c1.set_ylim(-90, 90); c1.set_yticks([-60, 0, 60]); c1.set_yticklabels([r"60$^\circ$S", "Eq.", r"60$^\circ$N"])
    c1.set_xlabel("share per latitude row", fontsize=7, labelpad=1); c1.tick_params(labelsize=6)
    c1.legend(frameon=False, fontsize=5.6, loc="center right", handlelength=1.4)
    c1.set_title("(a) location", loc="left", fontsize=7); c1.grid(True, color=GRID, lw=0.4); c1.set_axisbelow(True)
    cg = collections.defaultdict(dict)
    for f in sorted(glob.glob(D + "/tpami/A38_ceiling_*.csv")):
        for r in rd(f):
            if r["amp"] in ("1", "1.0") and r["detrend"] == "0":
                cg[(r["source"], r["product"])].setdefault(r["seed"], float(r["above_sharp"]))
    ck = [k for k in sorted(cg, key=lambda k_: (SRC.index(k_[0]), k_[1])) if len(cg[k]) == 3]
    if ck:
        vals = [np.mean(list(cg[k].values())) for k in ck]
        for i_, (k, v_) in enumerate(zip(ck, vals)): c2.scatter([i_], [v_], s=12, color="0.25", marker=MRK[k[0]], zorder=3)
        c2.set_ylim(0.4, 0.9)
        grp = [[i_ for i_, k in enumerate(ck) if k[0] == s_] for s_ in SRC]
        c2.set_xticks([np.mean(g_) for g_ in grp if g_]); c2.set_xticklabels([NAME[s_].split("/")[0] for s_, g_ in zip(SRC, grp) if g_], fontsize=6)
        c2.tick_params(axis="x", length=0)
        c2.set_ylabel("inserted cells above threshold", fontsize=6.5, labelpad=1); c2.tick_params(labelsize=6)
        c2.grid(True, axis="y", color=GRID, lw=0.4); c2.set_axisbelow(True)

        print("fig_bench ceiling: %d variables, mean %.3f, range %.3f-%.3f" % (len(ck), np.mean(vals), min(vals), max(vals)))
    c2.set_title("(b) magnitude ceiling", loc="left", fontsize=7)
    fig.savefig(OUT + "/fig_bench.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("fig_bench written")

# ------------------------------------------------------------------ fig_ablation: rounds (A37c) and exponent (A31)
ro = collections.defaultdict(lambda: collections.defaultdict(dict))
for f in sorted(glob.glob(D + "/tpami/A37_rounds_*.csv")):
    for r in rd(f): ro[int(r["param1"])][(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
if ro:
    fig, (e1, e2) = plt.subplots(1, 2, figsize=(3.88, 1.8), gridspec_kw=dict(wspace=0.45))
    its = sorted(ro); ks = [k for k in HO19 if all(len(ro[i].get(k, {})) == 3 for i in its)]
    M = np.array([[np.mean(list(ro[i][k].values())) for i in its] for k in ks])
    if len(ks):
        e1.plot(its, M.mean(0), color="#0072B2", marker="o", ms=3, lw=1.1)
        for row_ in M: e1.plot(its, row_, color="#0072B2", alpha=0.12, lw=0.6)
        e1.axvline(500, color="0.5", lw=0.6, ls=":"); e1.text(470, M.min() + 0.005, "reported", fontsize=5.5, color="0.35", ha="right")
        e1.set_xscale("log"); e1.set_xticks([10, 100, 1000]); e1.set_xticklabels(["10", "100", "1000"])
        e1.set_xlabel("round", fontsize=7, labelpad=1); e1.set_ylabel("held-out AUPRC", fontsize=7, labelpad=1)
        e1.tick_params(labelsize=6); e1.grid(True, color=GRID, lw=0.4); e1.set_axisbelow(True)
        e1.set_title("(a) rounds (%d var.)" % len(ks), loc="left", fontsize=7)
        print("fig_ablation rounds over %d vars:" % len(ks), dict(zip(its, np.round(M.mean(0), 4))))
    ex = collections.defaultdict(lambda: collections.defaultdict(dict)); alpha_of = {}
    for f in sorted(glob.glob(D + "/tpami/A31_exponent_ho*.csv")):
        for r in rd(f):
            ex[float(r["s"])][(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"])); alpha_of[float(r["s"])] = r["alpha"]
    xs_ = []
    for i_, sv in enumerate((0.25, 0.5, 1.0)):
        kk = [k for k in HO19 if len(ex[sv].get(k, {})) == 3]
        if len(kk) < 19: continue
        d = np.array([np.mean(list(ex[sv][k].values())) for k in kk]); m = d.mean(); lo, hi = boot(d)
        e2.errorbar([i_], [m], yerr=[[m - lo], [hi - m]], fmt="o", ms=4, color=FLOSC, capsize=2, lw=0.9)
        e2.text(i_, lo - 0.012, r"$\alpha=%s$" % alpha_of[sv], ha="center", va="top", fontsize=5.5, color="0.25")
        xs_.append(i_)
    e2.axhline(np.mean([srt[k] for k in HO19]), color="0.3", lw=0.7, ls="--"); e2.text(-0.4, np.mean([srt[k] for k in HO19]) + 0.004, "magnitude ranking", fontsize=5.5, ha="left", color="0.3")
    e2.set_xlim(-0.5, 2.5)
    e2.set_xticks([0, 1, 2]); e2.set_xticklabels([r"$s=\frac{1}{4}$", r"$s=\frac{1}{2}$", r"$s=1$"], fontsize=6.5)
    e2.set_ylim(0.66, 0.83); e2.tick_params(labelsize=6); e2.set_ylabel("held-out AUPRC", fontsize=7, labelpad=1)
    e2.grid(True, axis="y", color=GRID, lw=0.4); e2.set_axisbelow(True)
    e2.set_title("(b) exponent, re-selected", loc="left", fontsize=7)
    fig.savefig(OUT + "/fig_ablation.pdf", bbox_inches="tight", pad_inches=0.02); plt.close(fig)
    print("fig_ablation written")
