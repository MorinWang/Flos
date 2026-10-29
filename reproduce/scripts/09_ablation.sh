#!/bin/bash
# Table 2 and the 20CRv3 part of the sparsity figure (A28, A29); held-out ablation incl. integer exponent s=1 (A22).
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A28_ablate_20cr.py --arms no-smooth,grid-graph,no-smooth-fixed --products air,slp,pres,pottmp,rhum,omega,cape \
    --seeds 0,1,2 --lams 0.18,0.36,0.72,1.44,2.88 --device $FLOS_DEVICE --outdir .
$PY -u $SRC/A29_sweep_20cr.py --arms full,fixed --products air,slp,pres,pottmp,rhum,omega,cape \
    --seeds 0,1,2 --lams 0.045,0.09,0.18,0.36,0.72,1.44,2.88,5.76,11.52,23.04 --device $FLOS_DEVICE --outdir .
$PY -u $SRC/A22_ablate_sparse.py --source ncep --products lftx,pres    --shard n1
$PY -u $SRC/A22_ablate_sparse.py --source ncep --products omega,pottmp --shard n2
$PY -u $SRC/A22_ablate_sparse.py --source ncep --products gpcp         --shard n3
$PY -u $SRC/A22_ablate_sparse.py --source era5 --products air,slp      --shard e1
$PY -u $SRC/A22_ablate_sparse.py --source era5 --products pres,pottmp  --shard e2
$PY -u $SRC/A22_ablate_sparse.py --source era5 --products rhum,omega,cape --shard e3
