#!/usr/bin/env python3
"""Tables of the supplementary material (Appendix D), generated from the result CSVs.
Usage: python3 make_supp_tables.py <analysis/data dir> <supplement/tables dir>"""
import sys, os, csv, glob, statistics as st
from collections import defaultdict

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
DEV = {("ncep", "air"), ("ncep", "slp"), ("ncep", "rhum")}
VAR = {"air": "Air temp.", "slp": "SLP", "pres": "Sfc.\\ pressure", "pottmp": "Pot.\\ temp.", "rhum": "Rel.\\ humidity",
       "omega": "Vert.\\ velocity", "cape": "CAPE", "lftx": "Lifted index", "gpcp": "Precip.\\ (GPCP)"}
ORDER = {"ncep": ["air", "slp", "rhum", "pres", "pottmp", "lftx", "omega", "gpcp"],
         "era5": ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"],
         "20cr": ["air", "slp", "pres", "pottmp", "rhum", "omega", "cape"]}
SRC = {"ncep": "NCEP", "era5": "ERA5", "20cr": "20CRv3"}
v = defaultdict(list)
for r in csv.DictReader(open(D + "/main/main_table_long.csv")):
    v[(r["source"], r["product"], r["method"])].append(float(r["auprc"]))
for f in (D + "/tpami/A32_frpcag_dev.csv", D + "/tpami/A32_frpcag_devx.csv"):    # same setting as Table 2: (810, 0.3)
    for r in csv.DictReader(open(f)):
        if float(r["g_s"]) == 810 and float(r["g_t"]) == 0.3: v[(r["source"], r["product"], "FRPCAG")].append(float(r["auprc"]))
for f in sorted(glob.glob(D + "/tpami/A32_frpcag_hox_*.csv")):
    for r in csv.DictReader(open(f)): v[(r["source"], r["product"], "FRPCAG")].append(float(r["auprc"]))
for f in glob.glob(D + "/tpami/A30_core_ho_soft_*.csv"):
    for r in csv.DictReader(open(f)): v[(r["source"], r["product"], "CORE")].append(float(r["auprc"]))
for r in csv.DictReader(open(D + "/tpami/A30_core_dev.csv")):
    if r["arm"] == "CORE-soft" and float(r["lam"]) == 0.36: v[(r["source"], r["product"], "CORE")].append(float(r["auprc"]))
rho = defaultdict(list); seen_ = set()                       # exact operator K = I - H (A36), as in Table S-taper
for f in sorted(glob.glob(D + "/tpami/A36_contrast_*.csv")):
    for r in csv.DictReader(open(f)):
        if (r["source"], r["product"], r["seed"]) in seen_: continue
        seen_.add((r["source"], r["product"], r["seed"])); rho[(r["source"], r["product"])].append(float(r["rhoK_sharp"]))

cols = [("EOF", "EOF"), ("PCP", "PCP"), ("GLOSS", "GLOSS"), ("ScaledGD", "ScaledGD"), ("Mondal", "LR-STSS"),
        ("FRPCAG", "FRPCAG"), ("Sort", "Magnitude"), ("CORE", "\\name-C"), ("Fixed", "\\name\\ soft"), ("FLoS", "\\name")]
L = ["\\begin{table*}[t]", "\\centering",
     "\\caption{Per-variable AUPRC as mean and standard deviation over the three benchmark seeds}",
     "\\label{tab:perseed}", "\\scriptsize\\setlength{\\tabcolsep}{0pt}",
     "\\begin{tabular*}{\\textwidth}{@{\\extracolsep{\\fill}}ll" + "c" * len(cols) + "@{}}", "\\toprule",
     " & Variable & " + " & ".join(h for _, h in cols) + " \\\\", "\\midrule"]
for s in ("ncep", "era5", "20cr"):
    for i, p in enumerate(ORDER[s]):
        k = (s, p)
        cells = []
        for c, _ in cols:
            x = v.get(k + (c,), [])
            cells.append("%.3f$\\pm$%.3f" % (st.mean(x), st.stdev(x)) if len(x) >= 2 else "---")
        L.append("%s & %s%s & %s \\\\" % (SRC[s] if i == 0 else "", VAR[p], "$^\\dagger$" if k in DEV else "",
                                         " & ".join(cells)))
    L.append("\\midrule" if s != "20cr" else "\\bottomrule")
L += ["\\end{tabular*}", "\\end{table*}"]
open(OUT + "/tab_perseed.tex", "w").write("\n".join(L) + "\n")

# A31 dev grid
g = defaultdict(list)
for r in csv.DictReader(open(D + "/tpami/A31_exponent_dev.csv")): g[(float(r["s"]), float(r["alpha"]), float(r["lam"]))].append(float(r["auprc"]))
alphas = sorted({k[1] for k in g}); lams = sorted({k[2] for k in g})
L = ["\\begin{table}[!htb]", "\\centering", "\\caption{Development AUPRC of \\name\\ over exponent and strength}", "\\label{tab:a31grid}", "\\footnotesize", "\\setlength{\\tabcolsep}{4pt}",
     "\\begin{tabular}{l" + "c" * len(alphas) + "}", "\\toprule",
     "$s$ $\\backslash$ $\\alpha$ & " + " & ".join("%g" % a for a in alphas) + " \\\\", "\\midrule"]
for s in (0.25, 0.5, 1.0):
    best = {a: max((st.mean(g[(s, a, l)]), l) for l in lams if len(g.get((s, a, l), [])) == 9) if any(len(g.get((s, a, l), [])) == 9 for l in lams) else None for a in alphas}
    top = max(b[0] for b in best.values() if b)
    L.append("%g & " % s + " & ".join(("---" if best[a] is None else (("\\textbf{%.3f}" if best[a][0] == top else "%.3f") % best[a][0]) + "\\,{\\tiny(%g)}" % best[a][1]) for a in alphas) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(OUT + "/tab_a31grid.tex", "w").write("\n".join(L) + "\n")

# A30b and FRPCAG grids (compact)
h = defaultdict(list)
for r in csv.DictReader(open(D + "/tpami/A30b_core_dev.csv")): h[(float(r["s"]), float(r["alpha"]), float(r["lam"]))].append(float(r["auprc"]))
al2 = sorted({k[1] for k in h})
L = ["\\begin{table}[!htb]", "\\centering", "\\caption{Development AUPRC of \\name-C over exponent and strength}", "\\label{tab:a30bgrid}", "\\footnotesize\\setlength{\\tabcolsep}{3pt}",
     "\\begin{tabular}{l" + "c" * len(al2) + "}", "\\toprule",
     "$s$ $\\backslash$ $\\alpha$ & " + " & ".join("%g" % a for a in al2) + " \\\\", "\\midrule"]
for s in (0.25, 1.0):
    cells = []
    for a in al2:
        c = [(st.mean(h[(s, a, l)]), l) for l in sorted({k[2] for k in h}) if len(h.get((s, a, l), [])) == 9]
        cells.append("---" if not c else "%.3f\\,{\\tiny(%g)}" % max(c))
    L.append("%g & " % s + " & ".join(cells) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(OUT + "/tab_a30bgrid.tex", "w").write("\n".join(L) + "\n")

fr = defaultdict(list)
for f in [D + "/tpami/A32_frpcag_dev.csv"] + glob.glob(D + "/tpami/A32_frpcag_devx.csv"):
    for r in csv.DictReader(open(f)): fr[(float(r["g_s"]), float(r["g_t"]))].append(float(r["auprc"]))
gss = sorted({k[0] for k in fr}); gts = sorted({k[1] for k in fr})
L = ["\\begin{table}[!htb]", "\\centering", "\\caption{Development AUPRC of FRPCAG over its two graph weights}", "\\label{tab:frgrid}", "\\footnotesize",
     "\\begin{tabular}{l" + "c" * len(gts) + "}", "\\toprule", "$g_s$ $\\backslash$ $g_t$ & " + " & ".join("%g" % t for t in gts) + " \\\\", "\\midrule"]
for s in gss:
    L.append("%g & " % s + " & ".join(("%.3f" % st.mean(fr[(s, t)])) if len(fr.get((s, t), [])) == 9 else "---" for t in gts) + " \\\\")
L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
open(OUT + "/tab_frgrid.tex", "w").write("\n".join(L) + "\n")
print("supplement tables written")

# A36: per-variable results on the tapered benchmark
tt = defaultdict(lambda: defaultdict(list)); seen = set()
for f in sorted(glob.glob(D + "/tpami/A36_taper_*.csv")):
    for r in csv.DictReader(open(f)):
        key = (r["source"], r["product"], r["seed"], r["method"])
        if key in seen: continue
        seen.add(key); tt[r["method"]][(r["source"], r["product"])].append(float(r["auprc"]))
cc = defaultdict(lambda: defaultdict(list)); seenc = set()
for f in sorted(glob.glob(D + "/tpami/A36_contrast_*.csv")):
    for r in csv.DictReader(open(f)):
        key = (r["source"], r["product"], r["seed"])
        if key in seenc: continue
        seenc.add(key)
        for col in ("rhoK_sharp", "rhoK_taper", "rho_obs"): cc[col][key[:2]].append(float(r[col]))
if tt:
    L = ["\\begin{table}[!htb]", "\\centering",
         "\\caption{Per-variable results under tapered insertion}",
         "\\label{tab:taper}", "\\footnotesize\\setlength{\\tabcolsep}{0pt}",
         "\\begin{tabular*}{\\linewidth}{@{\\extracolsep{\\fill}}lccccccc@{}}", "\\toprule",
         " & \\multicolumn{4}{c}{AUPRC, tapered} & \\multicolumn{2}{c}{$\\rho_{\\Km}$} & \\\\",
         "\\cmidrule(lr){2-5}\\cmidrule(lr){6-7}",
         "Variable & \\name & \\name-C & Magnitude & FRPCAG & hard & taper & $\\rho_{\\mathrm{obs}}$ \\\\", "\\midrule"]
    for s_ in ("ncep", "era5", "20cr"):
        L.append("\\multicolumn{8}{l}{\\textit{%s}} \\\\" % SRC[s_])
        for i, p in enumerate(ORDER[s_]):
            k = (s_, p)
            L.append("%s%s & %.3f & %.3f & %.3f & %.3f & %.2f & %.2f & %.2f \\\\" % (
                VAR[p], "$^\\dagger$" if k in DEV else "",
                st.mean(tt["FLoS"][k]), st.mean(tt["CORE"][k]), st.mean(tt["Sort"][k]), st.mean(tt["FRPCAG"][k]),
                st.mean(cc["rhoK_sharp"][k]), st.mean(cc["rhoK_taper"][k]), st.mean(cc["rho_obs"][k])))
    L += ["\\bottomrule", "\\end{tabular*}", "\\end{table}"]
    open(OUT + "/tab_taper.tex", "w").write("\n".join(L) + "\n")
    print("tab_taper written")
