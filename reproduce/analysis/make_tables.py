#!/usr/bin/env python3
"""LaTeX tables of the manuscript, generated from the result CSVs (no number is typed by hand).

  tab_main.tex      Table 1: AUPRC per variable, eight methods + FRPCAG (if its held-out CSVs exist)
  tab_mechanism.tex Table 2: mechanism and ablation rows (A28, A30, A30b, A31 when available)
Usage: python3 make_tables.py <analysis/data dir> <paper/tables dir> [<supplement/tables dir> for Table S-mechanism]
"""
import sys, os, csv, glob, statistics as st
from collections import defaultdict

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
VAR = {"air": "Air temperature", "slp": "Sea-level pressure", "pres": "Surface pressure", "pottmp": "Potential temperature",
       "rhum": "Relative humidity", "omega": "Vertical velocity", "cape": "CAPE", "lftx": "Lifted index", "gpcp": "Precipitation (GPCP)"}
ORDER = {"ncep": ["air", "slp", "rhum", "pres", "pottmp", "lftx", "omega", "gpcp"],
         "era5": ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"],
         "20cr": ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}
SRCNAME = {"ncep": "NCEP/NCAR (and GPCP)", "era5": "ERA5", "20cr": "20CRv3"}

# ---------------------------------------------------------------- per-seed values of the eight main-table methods
v = defaultdict(list)
for r in csv.DictReader(open(D + "/main/main_table_long.csv")):
    v[(r["source"], r["product"], r["method"])].append(float(r["auprc"]))
# FRPCAG: frozen dev setting on held-out + its dev rows at the same setting
FR = None
fr_rows = []                       # compare against the stronger of the setting selected on the original grid, (90, 0.3), and of the extended grid, (810, 0.3)
FRFILES = sorted(glob.glob(D + "/tpami/A32_frpcag_hox_*.csv")) or sorted(glob.glob(D + "/tpami/A32_frpcag_ho_*.csv"))
for f in FRFILES: fr_rows += list(csv.DictReader(open(f)))
if fr_rows:
    gs, gt = float(fr_rows[0]["g_s"]), float(fr_rows[0]["g_t"])
    for r in fr_rows: v[(r["source"], r["product"], "FRPCAG")].append(float(r["auprc"]))
    for f in (D + "/tpami/A32_frpcag_dev.csv", D + "/tpami/A32_frpcag_devx.csv"):
        for r in csv.DictReader(open(f)):
            if float(r["g_s"]) == gs and float(r["g_t"]) == gt: v[(r["source"], r["product"], "FRPCAG")].append(float(r["auprc"]))
    FR = (gs, gt)
# FLoS-C, the convex core (soft threshold, lambda = 0.36): held-out runs and the development rows at the same setting
for f_ in glob.glob(D + "/tpami/A30_core_ho_soft_*.csv"):
    for r in csv.DictReader(open(f_)): v[(r["source"], r["product"], "CORE")].append(float(r["auprc"]))
for r in csv.DictReader(open(D + "/tpami/A30_core_dev.csv")):
    if r["arm"] == "CORE-soft" and float(r["lam"]) == 0.36: v[(r["source"], r["product"], "CORE")].append(float(r["auprc"]))
UNRANKED = {"CORE"}                # listed for reference, not part of the bold/underline ranking
cols = ["EOF", "PCP", "GLOSS", "ScaledGD", "Mondal"] + (["FRPCAG"] if FR else []) + ["Sort", "CORE", "FLoS"]
head = {"EOF": "EOF", "PCP": "PCP", "GLOSS": "GLOSS", "ScaledGD": "ScaledGD", "Mondal": "LR-STSS", "FRPCAG": "FRPCAG",
        "Sort": "Magnitude", "CORE": "\\name-C", "FLoS": "\\name"}
m = lambda k, c: st.mean(v[k + (c,)]) if v.get(k + (c,)) else float("nan")


def fmt_row(vals):
    ranked = [x for x, c in zip(vals, cols) if c not in UNRANKED and x == x]
    order = sorted(ranked, reverse=True)
    out = []
    for x, c in zip(vals, cols):
        s = "%.3f" % x
        if c in UNRANKED: out.append(s); continue
        if x == order[0]: s = "\\textbf{%s}" % s
        elif len(order) > 1 and x == order[1]: s = "\\underline{%s}" % s
        out.append(s)
    return out


lines = []
lines.append("\\begin{table*}[t]")
lines.append("\\centering")
lines.append("\\caption{AUPRC of all methods on the transplant benchmark, as means over three seeds. The best result is "
             "in bold and the second best underlined, \\name-C is listed for reference without ranking, and daggers mark "
             "the development variables.}")
lines.append("\\label{tab:main}")
lines.append("\\small")
lines.append("\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}l" + "c" * len(cols) + "@{}}")
lines.append("\\toprule")
lines.append("Variable & " + " & ".join(head[c] for c in cols) + " \\\\")
lines.append("\\midrule")
for s_ in ("ncep", "era5", "20cr"):
    lines.append("\\multicolumn{%d}{l}{\\textit{%s}} \\\\" % (len(cols) + 1, SRCNAME[s_]))
    for p in ORDER[s_]:
        k = (s_, p)
        lab = VAR[p] + ("$^\\dagger$" if k in DEV else "")
        lines.append(lab + " & " + " & ".join(fmt_row([m(k, c) for c in cols])) + " \\\\")
    ho = [(s_, p) for p in ORDER[s_] if (s_, p) not in DEV]
    lines.append("\\cmidrule(lr){1-%d}" % (len(cols) + 1))
    lines.append("Held-out mean & " + " & ".join(fmt_row([st.mean(m(k, c) for k in ho) for c in cols])) + " \\\\")
    lines.append("\\midrule")
HO = [(s_, p) for s_ in ORDER for p in ORDER[s_] if (s_, p) not in DEV]
lines.append("All 19 held-out & " + " & ".join(fmt_row([st.mean(m(k, c) for k in HO) for c in cols])) + " \\\\")
best_b = {k: max(m(k, c) for c in cols if c != "FLoS" and c not in UNRANKED) for k in HO}
lines.append("\\name\\ is best on & \\multicolumn{%d}{l}{%d of 19 held-out variables; mean margin over the best other method $%+.3f$} \\\\" % (
    len(cols), sum(m(k, "FLoS") > best_b[k] for k in HO), st.mean(m(k, "FLoS") - best_b[k] for k in HO)))
lines.append("\\bottomrule")
lines.append("\\end{tabular*}")
lines.append("\\end{table*}")
open(OUT + "/tab_main.tex", "w").write("\n".join(lines) + "\n")
print("tab_main.tex written; FRPCAG:", FR, "; FLoS best on", sum(m(k, "FLoS") > best_b[k] for k in HO), "/19")
for c in cols: print("  %-8s held-out %.4f" % (c, st.mean(m(k, c) for k in HO)))

# ---------------------------------------------------------------- Table 2: mechanism and ablation
def ho_mean_from(files, filt=lambda r: True):
    per = defaultdict(list)
    for f in files:
        for r in csv.DictReader(open(f)):
            if filt(r): per[(r["source"], r["product"])].append(float(r["auprc"]))
    ks = [k for k in HO if len(per.get(k, [])) >= 3]
    return (st.mean(st.mean(per[k][:3]) for k in ks), len(ks)) if len(ks) == 19 else (None, len(ks))

fl = st.mean(m(k, "FLoS") for k in HO)
rowsA = [("\\name\\ (full)", fl, 19, "")]
rowsA.append(("\\quad adaptive $\\to$ soft threshold ($\\lambda=0.36$)", st.mean(m(k, "Fixed") for k in HO), 19, ""))
x = ho_mean_from(glob.glob(D + "/tpami/A31_exponent_lam1_*.csv")); rowsA.append(("\\quad $\\lambda=1.0$, the optimum on the development variables alone", x[0], x[1], ""))
x = ho_mean_from(glob.glob(D + "/tpami/A31_exponent_lamfix05_*.csv")); rowsA.append(("\\quad soft threshold at its development optimum $\\lambda=0.5$", x[0], x[1], ""))
x = ho_mean_from(glob.glob(D + "/tpami/A39_notrunc_ho_*.csv"), lambda r: r["arm"] == "noT" and float(r["lam"]) == 0.72)
rowsA.append(("\\quad without truncation (A39)", x[0], x[1], ""))
x = ho_mean_from(glob.glob(D + "/tpami/A30_core_ho_soft_*.csv")); rowsA.append(("\\name-C: no truncation, soft threshold (Thm.~\\ref{thm:core})", x[0], x[1], "converges in all runs"))
x = ho_mean_from(glob.glob(D + "/tpami/A30_core_ho_adapt_*.csv")); rowsA.append(("\\name-C with the adaptive threshold", x[0], x[1], ""))
x = ho_mean_from(glob.glob(D + "/tpami/A30b_core_ho_*.csv"))
rowsA.append(("\\name-C, $(s,\\alpha,\\lambda)$ re-selected", x[0], x[1], A30B_NOTE if 'A30B_NOTE' in globals() else ""))
x = ho_mean_from(glob.glob(D + "/tpami/A30_core_hp?.csv"), lambda r: r["arm"] == "HP1"); rowsA.append(("one high-pass step $|\\Xm-F\\Xm|$ (first iterate)", x[0], x[1], "frozen strength"))
x = ho_mean_from(glob.glob(D + "/tpami/A30_core_hp?.csv"), lambda r: r["arm"] == "HP1T"); rowsA.append(("first iterate with truncation, $|\\Xm-\\Trunc(F\\Xm)|$", x[0], x[1], ""))
def select_ho(files, fam, keyf):            # A37: select a configuration on the development variables, held-out mean at it
    g, seen = defaultdict(lambda: defaultdict(list)), set()
    for f in sorted(files):
        for r in csv.DictReader(open(f)):
            if r["family"] != fam: continue
            key = (r["source"], r["product"], r["seed"], keyf(r))
            if key in seen: continue
            seen.add(key); g[keyf(r)][(r["source"], r["product"])].append(float(r["auprc"]))
    dev = {c: st.mean(st.mean(v_[k]) for k in DEV) for c, v_ in g.items() if all(len(v_.get(k, [])) == 3 for k in DEV)}
    if not dev: return None, 0, None
    c = max(dev, key=dev.get)
    ks = [k for k in HO if len(g[c].get(k, [])) == 3]
    return (st.mean(st.mean(g[c][k]) for k in ks) if len(ks) == 19 else None), len(ks), c
hpf = glob.glob(D + "/tpami/A37_hp_*.csv")
x = select_ho(hpf, "HP", lambda r: (float(r["param1"]), float(r["param2"])))
rowsA.append(("one high-pass step $|\\Xm-F\\Xm|$, $(s,\\alpha)$ re-selected", x[0], x[1],
              "$s=%g$, $\\alpha=%g$" % x[2] if x[2] else ""))
x = select_ho(hpf, "LOWMODE", lambda r: int(r["param1"]))
rowsA.append(("remove the $k$ smoothest Laplacian modes, $k$ re-selected", x[0], x[1], "$k=%d$" % x[2] if x[2] else ""))
for sv in (0.25, 0.5, 1.0):
    x = ho_mean_from(glob.glob(D + "/tpami/A31_exponent_ho*.csv"), lambda r, sv=sv: float(r["s"]) == sv)
    note = A31_NOTE.get(sv, "") if 'A31_NOTE' in globals() else ""
    rowsA.append(("\\name\\ with $s=%s$, $(\\alpha,\\lambda)$ re-selected" % ({0.25: "\\frac14", 0.5: "\\frac12", 1.0: "1"}[sv]), x[0], x[1], note))
rowsA.append(("magnitude ranking $|\\Xm|$", st.mean(m(k, "Sort") for k in HO), 19, "reference"))

ours20 = defaultdict(list); fix20 = defaultdict(list)
for r in csv.DictReader(open(D + "/main/A25_ours.csv")): ours20[r["product"]].append(float(r["auprc"]))
for r in csv.DictReader(open(D + "/main/A25_fixed.csv")): fix20[r["product"]].append(float(r["auprc"]))
def best_arm(f):
    g = defaultdict(lambda: defaultdict(list))
    for r in csv.DictReader(open(f)): g[float(r["lam"])][r["product"]].append(float(r["auprc"]))
    lam = max(g, key=lambda l: st.mean(st.mean(v_) for v_ in g[l].values()))
    return st.mean(st.mean(v_) for v_ in g[lam].values()), lam
f20 = st.mean(st.mean(v_) for v_ in ours20.values())
rowsB = [("\\name\\ (full)", f20, ""), ("soft instead of adaptive threshold", st.mean(st.mean(v_) for v_ in fix20.values()), "")]
for arm, lab in (("no-smooth", "no smoothing"), ("grid-graph", "lattice instead of spherical graph"), ("no-smooth-fixed", "no smoothing, soft threshold")):
    val, lam = best_arm(D + "/ablation/A28_%s.csv" % arm)
    rowsB.append((lab, val, "best $\\lambda=%g$" % lam))

# tapered counterparts (A36 frozen settings; A36b re-selected on the tapered development variables)
def tap_mean(pat, meth):
    g = defaultdict(dict)
    for f in sorted(glob.glob(D + "/tpami/" + pat)):
        for r in csv.DictReader(open(f)):
            if r["method"] == meth: g[(r["source"], r["product"])].setdefault(r["seed"], float(r["auprc"]))
    ks = [k for k in HO if len(g.get(k, {})) == 3]
    return st.mean(st.mean(g[k].values()) for k in ks) if len(ks) == 19 else None
TS = tap_mean("A36_taper_*.csv", "Sort")
TAPER = {"\\name\\ (full)": tap_mean("A36_taper_*.csv", "FLoS"),
         "\\name-C: no truncation, soft threshold (Thm.~\\ref{thm:core})": tap_mean("A36_taper_*.csv", "CORE"),
         "\\name-C, $(s,\\alpha,\\lambda)$ re-selected": tap_mean("A36b_ho_*.csv", "CORE"),
         "magnitude ranking $|\\Xm|$": TS}
TAPER_FLOS_RE = tap_mean("A36b_ho_*.csv", "FLoS")
sort_h = st.mean(m(k, "Sort") for k in HO)
SHORT = {"\\name\\ (full)": "\\name",
         "\\quad adaptive $\\to$ soft threshold ($\\lambda=0.36$)": "\\quad soft threshold, $\\lambda=0.36$",
         "\\quad $\\lambda=1.0$, the optimum on the development variables alone": "\\quad $\\lambda=1.0$ (development optimum)",
         "\\quad soft threshold at its development optimum $\\lambda=0.5$": "\\quad soft threshold, $\\lambda=0.5$ (dev.\\ opt.)",
         "\\name-C: no truncation, soft threshold (Thm.~\\ref{thm:core})": "\\name-C, the convex core",
         "\\name-C with the adaptive threshold": "\\quad with adaptive threshold",
         "\\quad without truncation (A39)": "\\quad without truncation",
         "\\name-C, $(s,\\alpha,\\lambda)$ re-selected": "\\quad $(s,\\alpha,\\lambda)$ re-selected",
         "one high-pass step $|\\Xm-F\\Xm|$ (first iterate)": "one high-pass step $|\\Xm-F\\Xm|$",
         "first iterate with truncation, $|\\Xm-\\Trunc(F\\Xm)|$": "\\quad with truncation",
         "one high-pass step $|\\Xm-F\\Xm|$, $(s,\\alpha)$ re-selected": "\\quad $(s,\\alpha)$ re-selected",
         "remove the $k$ smoothest Laplacian modes, $k$ re-selected": "remove smoothest modes ($k$ re-sel.)",
         "\\name\\ with $s=\\frac14$, $(\\alpha,\\lambda)$ re-selected": "\\name, $s=\\frac14$, $(\\alpha,\\lambda)$ re-sel.",
         "\\name\\ with $s=\\frac12$, $(\\alpha,\\lambda)$ re-selected": "\\name, $s=\\frac12$, $(\\alpha,\\lambda)$ re-sel.",
         "\\name\\ with $s=1$, $(\\alpha,\\lambda)$ re-selected": "\\name, $s=1$, $(\\alpha,\\lambda)$ re-sel.",
         "magnitude ranking $|\\Xm|$": "magnitude ranking $|\\Xm|$"}
L2 = ["\\begin{table}[t]", "\\centering",
      "\\caption{Held-out AUPRC of variants suggested by the analysis}",
      "\\label{tab:mechanism}", "\\footnotesize\\setlength{\\tabcolsep}{3pt}",
      "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lccc@{}}", "\\toprule",
      "Variant & AUPRC & $\\Delta$ \\name & $\\Delta$ magnitude \\\\", "\\midrule"]
for lab, val, n, note in rowsA:
    lab2 = SHORT.get(lab, lab)
    if val is None:
        L2.append("%s & \\pending{} & & \\\\" % lab2); continue
    dfl = "" if lab.startswith("\\name\\ (full)") else "$%+.3f$" % (val - fl)
    dso = "" if lab.startswith("magnitude") else "$%+.3f$" % (val - sort_h)
    L2.append("%s & %.3f & %s & %s \\\\" % (lab2, val, dfl, dso))
L2 += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
OUT2 = sys.argv[3] if len(sys.argv) > 3 else OUT
open(OUT2 + "/tab_mechanism.tex", "w").write("\n".join(L2) + "\n")
print("tab_mechanism.tex written to", OUT2)
# 20CRv3 ablations -> supplement
OUT3 = sys.argv[4] if len(sys.argv) > 4 else None
if OUT3:
    L3 = ["\\begin{table}[!htb]", "\\centering",
          "\\caption{Ablations of \\name\\ on the seven variables of 20CRv3}",
          "\\label{tab:ablation20}", "\\footnotesize",
          "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}>{\\raggedright\\arraybackslash}p{5.9cm}cc@{}}", "\\toprule", "Variant & AUPRC & $\\Delta$ \\\\", "\\midrule"]
    for lab, val, note in rowsB:
        L3.append("%s%s & %.3f & %s \\\\" % (lab, (" {\\scriptsize(%s)}" % note) if note else "", val, "" if lab.startswith("\\name\\ (full)") else "$%+.3f$" % (val - f20)))
    L3 += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    open(OUT3 + "/tab_ablation20.tex", "w").write("\n".join(L3) + "\n")
    print("tab_ablation20.tex written to", OUT3)
for lab, val, n, note in rowsA: print("  A %-70s %s n=%d" % (lab[:70], "pending" if val is None else "%.4f" % val, n))
for lab, val, note in rowsB: print("  B %-70s %.4f %s" % (lab, val, note))
