#!/usr/bin/env python3
# Assemble the main table: 3 reanalyses x 22 products x 8 methods, averaged over seeds 0/1/2. Each method uses the
# setting frozen on the NCEP development set.
# Only reads existing CSVs; no experiments are run.
import pandas as pd, glob, os
from paths import OUT as O                 # all CSVs are read from, and the tables written to, $FLOS_OUT
D15 = O
rows = []
def add(df, method, src_col='source'):
    df = df.copy()
    if src_col not in df.columns: df['source'] = 'ncep'
    df = df[df['seed'].isin([0, 1, 2])]
    for _, r in df.iterrows():
        rows.append(dict(source=r['source'], product=r['product'], method=method, seed=int(r['seed']),
                         auprc=float(r['auprc']), nnz=float(r['nnz_frac']) if 'nnz_frac' in r and pd.notna(r['nnz_frac']) else float('nan')))
# FLoS and fixed threshold: the three NCEP development products come from A15/A16, held-out products from A17,
# 20CRv3 from A25
a15 = pd.read_csv(f'{O}/A15_lam_sweep.csv')
a17 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D15}/A17_holdout_[ne]*.csv'))])
for arm, lam, name in (('A-adapt', 0.72, 'FLoS'), ('B-fixed', 0.36, 'Fixed')):
    add(a15[(a15.arm == arm) & (a15.lam.round(3) == lam)], name)
    add(a17[(a17.arm == arm) & (a17.lam.round(3) == lam)], name)
add(pd.read_csv(f'{O}/A25_ours.csv'), 'FLoS')
add(pd.read_csv(f'{O}/A25_fixed.csv'), 'Fixed')
# PCP, multiplier 2
a18 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D15}/A18_pcp_p*.csv'))])
add(a18[a18['mult'] == 2], 'PCP'); add(pd.read_csv(f'{O}/A25_pcp.csv'), 'PCP')
# EOF 0.25, GLOSS 2, direct sorting: A21 and A25
a21 = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f'{D15}/A21_opp_[cg]*.csv'))])
add(a21[(a21.method == 'EOF-PCA') & (a21['mult'] == 0.25)], 'EOF')
add(a21[(a21.method == 'GLOSS') & (a21['mult'] == 2)], 'GLOSS')
print('methods in A21:', sorted(a21.method.unique()))
srt = a21[a21.method.str.contains('sort|raw|Sort|RAW', regex=True)]
add(srt, 'Sort')
add(pd.read_csv(f'{O}/A25_eof.csv'), 'EOF')
add(pd.read_csv(f'{O}/A25_gloss.csv'), 'GLOSS')
add(pd.read_csv(f'{O}/A25_sort.csv'), 'Sort')
# Tensor ScaledGD and Mondal: the row with nnz_target==1
for f, name in ((f'{O}/A24_eval.csv', 'ScaledGD'), (f'{O}/A27_scaledgd_ncepdev.csv', 'ScaledGD'),
                (f'{O}/A25_scaledgd.csv', 'ScaledGD'), (f'{O}/A23_eval.csv', 'Mondal'),
                (f'{O}/A27_mondal_ncepdev.csv', 'Mondal'), (f'{O}/A25_mondal.csv', 'Mondal')):
    d = pd.read_csv(f); add(d[d.nnz_target == 1.0], name)
T = pd.DataFrame(rows)
cnt = T.groupby(['source', 'method']).size().unstack(fill_value=0)
print('rows per dataset and method (should be n_products x 3):'); print(cnt)
M = T.groupby(['source', 'product', 'method']).auprc.mean().unstack()
order = ['FLoS', 'Fixed', 'ScaledGD', 'Sort', 'Mondal', 'PCP', 'EOF', 'GLOSS']
M = M[order]
pd.set_option('display.width', 200)
print(M.round(4))
print('\nmean over the eight (or seven) products of each dataset:'); print(M.groupby(level=0).mean().round(4))
DEV = {('ncep', 'air'), ('ncep', 'slp'), ('ncep', 'rhum')}
held = M[[k not in DEV for k in M.index]]
print('\nmean over held-out products only:'); print(held.groupby(level=0).mean().round(4))
base = [c for c in order if c not in ('FLoS', 'Fixed')]
best = M[base].max(axis=1); arg = M[base].idxmax(axis=1)
diff = (M['FLoS'] - best).round(4)
print('\nper product: FLoS minus the strongest baseline (baseline name)')
for k in M.index:
    print(f"  {k[0]:5s} {k[1]:7s} {'dev' if k in DEV else 'held-out'}  {diff[k]:+.4f}  strongest baseline {arg[k]}")
print('wins', int((diff > 0).sum()), '/', len(diff))
T.to_csv(f'{O}/main_table_long.csv', index=False)
M.to_csv(f'{O}/main_table.csv')
