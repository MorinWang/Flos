#!/bin/bash
# FLoS and the fixed-threshold variant on the held-out NCEP/GPCP and ERA5 variables, full lambda grid.
# Feeds Table 1, the sparsity figure and the lambda curve. Shards are independent and can run in parallel.
set -e; source "$(dirname "$0")/common.sh"
LAMS=0.045,0.09,0.18,0.36,0.72,1.44,2.88,5.76,11.52,23.04
$PY -u $SRC/A17_holdout.py --source ncep --products lftx,pres,omega --lams $LAMS --shard n1
$PY -u $SRC/A17_holdout.py --source ncep --products pottmp,gpcp     --lams $LAMS --shard n2
$PY -u $SRC/A17_holdout.py --source era5 --products air,slp,pres    --lams $LAMS --shard e1
$PY -u $SRC/A17_holdout.py --source era5 --products pottmp,rhum     --lams $LAMS --shard e2
$PY -u $SRC/A17_holdout.py --source era5 --products omega,cape      --lams $LAMS --shard e3
