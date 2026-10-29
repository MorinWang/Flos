import csv, statistics as st, sys
from collections import defaultdict
from scipy.stats import wilcoxon, binomtest
D = sys.argv[1]
L = list(csv.DictReader(open(D + '/main/main_table_long.csv')))
v = defaultdict(list)
for r in L: v[(r['source'], r['product'], r['method'])].append(float(r['auprc']))
meth = ['FLoS', 'Fixed', 'ScaledGD', 'Sort', 'Mondal', 'PCP', 'EOF', 'GLOSS']
keys = sorted({(r['source'], r['product']) for r in L})
DEV = {('ncep','air'),('ncep','slp'),('ncep','rhum')}
HO = [k for k in keys if k not in DEV]
sd = {m: st.mean(st.stdev(v[k + (m,)]) for k in HO) for m in meth}
print("mean over held-out variables of the across-seed std:", {m: round(sd[m], 4) for m in meth})
base = ['ScaledGD', 'Sort', 'Mondal', 'PCP', 'EOF', 'GLOSS']
dbest = []; dsort = []; dscaled = []
for k in HO:
    f = st.mean(v[k + ('FLoS',)]); b = max(st.mean(v[k + (m,)]) for m in base)
    dbest.append(f - b); dsort.append(f - st.mean(v[k + ('Sort',)])); dscaled.append(f - st.mean(v[k + ('ScaledGD',)]))
for nm, d in (("best baseline per variable", dbest), ("direct ranking", dsort), ("ScaledGD", dscaled)):
    w = wilcoxon(d, alternative='greater')
    print("FLoS - %s over 19 held-out: mean %+.4f median %+.4f wins %d/19  Wilcoxon one-sided p=%.4g  sign-test p=%.4g" % (
        nm, st.mean(d), st.median(d), sum(x > 0 for x in d), w.pvalue, binomtest(sum(x > 0 for x in d), 19, 0.5, alternative='greater').pvalue))
# seed-level: paired over (variable, seed) against ScaledGD
ds = [a - b for k in HO for a, b in zip(v[k + ('FLoS',)], v[k + ('ScaledGD',)])]
print("seed-level FLoS-ScaledGD: n=%d mean %+.4f wins %d" % (len(ds), st.mean(ds), sum(x > 0 for x in ds)))
# per-source held-out means
for s in ('ncep', 'era5', '20cr'):
    ks = [k for k in HO if k[0] == s]
    print(s, len(ks), {m: round(st.mean(st.mean(v[k + (m,)]) for k in ks), 4) for m in meth})
# per-variable std table rows
for k in keys:
    print(k, " ".join("%s=%.3f±%.3f" % (m, st.mean(v[k + (m,)]), st.stdev(v[k + (m,)])) for m in ('FLoS', 'ScaledGD', 'Sort')))
