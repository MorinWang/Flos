#!/usr/bin/env python3
# Numbers reported in Sections 5.4 and 5.5; only reads existing CSVs:
# (1) component ablation on 20CRv3 (each arm at the lambda on the shared grid with the best seven-product mean)
# (2) mean of both arms and their difference on the sparsity curve at retained fractions of 50%/20%/10%/1%/0.1%
#     (same procedure as A29_figure.py) (3) event fraction (4) lambda curve on the held-out set
import csv, io, glob, collections
import numpy as np
from paths import OUT as O                 # all CSVs are read from $FLOS_OUT
D3 = P19 = O
rd = lambda p: list(csv.DictReader(io.open(p, encoding='utf-8')))
# (1) ablation
arms = {'full': rd(f'{P19}/A29_full.csv'), 'fixed': rd(f'{P19}/A29_fixed.csv'),
        'no-smooth': rd(f'{P19}/A28_no-smooth.csv'), 'grid-graph': rd(f'{P19}/A28_grid-graph.csv'),
        'no-smooth-fixed': rd(f'{P19}/A28_no-smooth-fixed.csv')}
GRID = [0.18, 0.36, 0.72, 1.44, 2.88]
print('== (1) 20CRv3 ablation: mean over 7 products x 3 seeds, per lambda ==')
best = {}
for a, rows in arms.items():
    by = collections.defaultdict(list); nz = collections.defaultdict(list); prod = collections.defaultdict(list)
    for r in rows:
        l = round(float(r['lam']), 3)
        if l in GRID and r['seed'] in ('0', '1', '2'):
            by[l].append(float(r['auprc'])); nz[l].append(float(r['nnz_frac'])); prod[(l, r['product'])].append(float(r['auprc']))
    line = '  '.join(f"{l}:{np.mean(by[l]):.4f}({len(by[l])})" for l in GRID if by[l])
    lb = max((l for l in GRID if by[l]), key=lambda l: np.mean(by[l]))
    best[a] = (lb, np.mean(by[lb]), np.mean(nz[lb]), {p: np.mean(v) for (l, p), v in prod.items() if l == lb})
    print(f"  {a:16s} {line}   best lambda={lb} mean {np.mean(by[lb]):.4f} kept {np.mean(nz[lb]):.3f}")
print('  per product (each arm at its own best lambda):')
prods = sorted(best['full'][3])
print('   ' + ' '.join(f'{p:>8s}' for p in prods))
for a in arms:
    print(f"   {a:16s}" + ' '.join(f"{best[a][3].get(p, float('nan')):8.4f}" for p in prods) + f"   mean diff {best[a][1]-best['full'][1]:+.4f}")
# (2) sparse regime
cur = collections.defaultdict(list); rate = []
for p in glob.glob(D3 + '/A17_holdout_[ne]*.csv'):
    for r in rd(p):
        if r['arm'] not in ('A-adapt', 'B-fixed'): continue
        arm = 'full' if r['arm'] == 'A-adapt' else 'fixed'
        cur[(r['source'], arm, r['product'], r['seed'])].append((float(r['nnz_frac']), float(r['auprc'])))
        rate.append(float(r['pos_rate']))
for arm in ('full', 'fixed'):
    for r in rd(f'{P19}/A29_{arm}.csv'):
        cur[('20cr', arm, r['product'], r['seed'])].append((float(r['nnz_frac']), float(r['auprc']))); rate.append(float(r['pos_rate']))
def interp(pts, xs):
    pts = sorted(set(pts)); x = np.log10([max(a, 1e-9) for a, _ in pts]); y = np.array([b for _, b in pts])
    o = np.argsort(x); x, y = x[o], y[o]
    out = np.interp(np.log10(xs), x, y, left=np.nan, right=np.nan); out[(xs < 10 ** x.min()) | (xs > 10 ** x.max())] = np.nan
    return out
T = np.array([0.5, 0.3, 0.2, 0.1, 0.05, 0.01, 0.001])
print('\n== (2) sparse regime: per dataset, mean of both arms at each retained fraction (interpolated), and difference ==')
allfull = []; allfix = []
for src in ('ncep', 'era5', '20cr'):
    allp = sorted({k[2] for k in cur if k[0] == src and k[1] == 'full'})
    prods = [p for p in allp if max(min(a for a, _ in cur[(src, arm, p, s)]) for arm in ('full', 'fixed') for s in '012' if (src, arm, p, s) in cur) <= 0.01]
    Y = {arm: np.vstack([interp(cur[(src, arm, p, s)], T) for p in prods for s in '012' if (src, arm, p, s) in cur]) for arm in ('full', 'fixed')}
    mf, mx = np.nanmean(Y['full'], 0), np.nanmean(Y['fixed'], 0)
    allfull.append(mf); allfix.append(mx)
    print(f"  {src:5s} {len(prods)} variables  " +'  '.join(f"{t:g}: {a:.3f}/{b:.3f} ({a-b:+.3f})" for t, a, b in zip(T, mf, mx)))
d = np.nanmean(np.vstack(allfull), 0) - np.nanmean(np.vstack(allfix), 0)
print('  difference averaged over the three datasets: ' +'  '.join(f"{t:g}: {x:+.3f}" for t, x in zip(T, d)))
print(f"\n== (3) injected events as a fraction of all grid points: {min(rate):.5f} to {max(rate):.5f}, median {np.median(rate):.5f}")
# (4) lambda curve on the held-out set (twelve held-out NCEP+ERA5 products)
print('\n== (4) held-out set (A17): FLoS mean AUPRC and retained fraction per lambda ==')
by = collections.defaultdict(list); nz = collections.defaultdict(list)
for p in glob.glob(D3 + '/A17_holdout_[ne]*.csv'):
    for r in rd(p):
        if r['arm'] == 'A-adapt' and r['seed'] in ('0', '1', '2'):
            by[round(float(r['lam']), 3)].append(float(r['auprc'])); nz[round(float(r['lam']), 3)].append(float(r['nnz_frac']))
for l in sorted(by): print(f"  lambda={l:<6} {np.mean(by[l]):.4f}  kept {np.mean(nz[l]):.3f}  n={len(by[l])}")
