#!/usr/bin/env python3
"""Animated figures for the README, made from the released result files only.

  python3 make_readme_assets.py <results dir> <assets dir> [<iterations npz>]

  hard_vs_taper.gif   one month of 20CRv3 sea-level pressure: the same extreme inserted with a hard edge and with its
                      own tapered surroundings; observed field, background and event component of FLoS
  gains_slide.gif     gain of FLoS over magnitude ranking on each held-out variable, moving from hard-edged to
                      tapered insertion
  flos_iterations.gif background and events of one month over the rounds of FLoS (needs the npz written by
                      examples/record_iterations.py)
"""
import csv, glob, json, os, sys, statistics as st
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import imageio.v2 as imageio

D, OUT = sys.argv[1], sys.argv[2]
ITER = sys.argv[3] if len(sys.argv) > 3 else None
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.size": 9, "axes.titlesize": 9, "font.family": "DejaVu Sans"})
coast = json.load(open(os.path.join(D, "ne_110m_coastline.geojson")))


def draw_coast(ax):
    for f in coast["features"]:
        g = f["geometry"]
        for line in (g["coordinates"] if g["type"] == "MultiLineString" else [g["coordinates"]]):
            a = np.asarray(line); lon = np.mod(a[:, 0], 360)
            breaks = np.where(np.abs(np.diff(lon)) > 180)[0] + 1
            for seg_lon, seg_lat in zip(np.split(lon, breaks), np.split(a[:, 1], breaks)):
                ax.plot(seg_lon, seg_lat, color="0.25", lw=0.4)


def field(ax, lat, lon, Z, vmax, title, support=None, cmap="RdBu_r"):
    ax.pcolormesh(lon, lat, Z, cmap=cmap, norm=TwoSlopeNorm(0, -vmax, vmax), shading="auto")
    draw_coast(ax)
    if support is not None and support.any():
        ax.contour(lon, lat, support.astype(float), levels=[0.5], colors="k", linewidths=0.8)
    ax.set_xlim(0, 357.5); ax.set_ylim(-90, 90); ax.set_xticks([]); ax.set_yticks([]); ax.set_title(title)


def to_frame(fig):
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return img


# ------------------------------------------------------------------ hard-edged vs tapered insertion, one month
h = np.load(os.path.join(D, "tpami", "A35_example.npz")); tp = np.load(os.path.join(D, "tpami", "A35b_example_taper.npz"))
lat, lon = h["lat"], h["lon"]
vX = float(np.percentile(np.abs(h["X"]), 99.5)); vS = float(np.abs(h["S_flos"]).max())
def insertion_frame(w):
    mix = lambda k, k2=None: (1 - w) * h[k] + w * tp[k2 or k]
    fig, axs = plt.subplots(1, 3, figsize=(11, 2.6), dpi=110)
    label = "hard-edged insertion" if w < 0.5 else "inserted with its surroundings"
    field(axs[0], lat, lon, mix("X"), vX, "observed anomaly  (%s)" % label, h["support"])
    field(axs[1], lat, lon, mix("L_flos"), vX, "FLoS background")
    field(axs[2], lat, lon, mix("S_flos"), vS, "FLoS events", h["support"])
    fig.tight_layout()
    return to_frame(fig)
def save(name, frames, durations):
    """GIF writers merge identical neighbouring frames, so pauses are given as per-frame durations (ms)."""
    imageio.mimsave(os.path.join(OUT, name), frames, duration=durations, loop=0)
ramp = list(np.linspace(0, 1, 10)[1:-1])
save("hard_vs_taper.gif", [insertion_frame(0.0)] + [insertion_frame(w) for w in ramp] + [insertion_frame(1.0)]
     + [insertion_frame(w) for w in ramp[::-1]], [1600] + [110] * len(ramp) + [1600] + [110] * len(ramp))
print("hard_vs_taper.gif")

# ------------------------------------------------------------------ per-variable gains: hard-edged -> tapered
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
v = defaultdict(list)
for r in csv.DictReader(open(os.path.join(D, "main", "main_table_long.csv"))):
    v[(r["source"], r["product"], r["method"])].append(float(r["auprc"]))
tt, seen = defaultdict(lambda: defaultdict(list)), set()
for f in sorted(glob.glob(os.path.join(D, "tpami", "A36_taper_*.csv"))):
    for r in csv.DictReader(open(f)):
        key = (r["source"], r["product"], r["seed"], r["method"])
        if key in seen: continue
        seen.add(key); tt[r["method"]][(r["source"], r["product"])].append(float(r["auprc"]))
HO = sorted(k for k in tt["FLoS"] if k not in DEV)
hard = {k: st.mean(v[k + ("FLoS",)]) - st.mean(v[k + ("Sort",)]) for k in HO}
soft = {k: st.mean(tt["FLoS"][k]) - st.mean(tt["Sort"][k]) for k in HO}
order = sorted(HO, key=lambda k: hard[k])
name = {"ncep": "NCEP", "era5": "ERA5", "20cr": "20CRv3"}
var = {"air": "air temp.", "slp": "SLP", "pres": "sfc. pressure", "pottmp": "pot. temp.", "rhum": "rel. humidity",
       "omega": "vert. velocity", "cape": "CAPE", "lftx": "lifted index", "gpcp": "GPCP precip."}
def gain_frame(w):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=110)
    y = np.arange(len(order))
    g = np.array([(1 - w) * hard[k] + w * soft[k] for k in order])
    ax.axvline(0, color="0.3", lw=0.8)
    ax.barh(y, g, color=np.where(g >= 0, "#0072B2", "#D55E00"), height=0.65)
    ax.set_yticks(y); ax.set_yticklabels(["%s %s" % (name[k[0]], var[k[1]]) for k in order], fontsize=7.5)
    ax.set_xlim(-0.22, 0.22); ax.set_xlabel("AUPRC gain of FLoS over magnitude ranking")
    ax.set_title("hard-edged insertion" if w < 0.02 else ("inserted with surroundings" if w > 0.98 else " "))
    ax.text(0.98, 0.02, "held-out mean %+.3f" % g.mean(), transform=ax.transAxes, ha="right", fontsize=9)
    fig.tight_layout()
    return to_frame(fig)
ramp = list(np.linspace(0, 1, 16)[1:-1])
save("gains_slide.gif", [gain_frame(0.0)] + [gain_frame(w) for w in ramp] + [gain_frame(1.0)]
     + [gain_frame(w) for w in ramp[::-1]], [1800] + [90] * len(ramp) + [2000] + [90] * len(ramp))
print("gains_slide.gif: hard-edged mean %+.3f, tapered mean %+.3f" % (st.mean(hard.values()), st.mean(soft.values())))

# ------------------------------------------------------------------ rounds of FLoS on one month
if ITER and os.path.exists(ITER):
    z = np.load(ITER)
    vX = float(np.percentile(np.abs(z["X"]), 99.5)); vS = float(np.abs(z["S"]).max())
    frames, durations = [], []
    for i, k in enumerate(z["rounds"]):
        fig, axs = plt.subplots(1, 3, figsize=(11, 2.6), dpi=110)
        field(axs[0], z["lat"], z["lon"], z["X"], vX, "observed anomaly (inserted extreme outlined)", z["support"])
        field(axs[1], z["lat"], z["lon"], z["L"][i], vX, "background, round %d" % k)
        field(axs[2], z["lat"], z["lon"], z["S"][i], vS, "events, round %d" % k, z["support"])
        fig.tight_layout(); frames.append(to_frame(fig)); durations.append(2500 if i == len(z["rounds"]) - 1 else 350)
    save("flos_iterations.gif", frames, durations)
    print("flos_iterations.gif", len(z["rounds"]), "rounds")
