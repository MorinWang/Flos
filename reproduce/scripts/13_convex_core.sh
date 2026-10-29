#!/bin/bash
# FLoS-C, the convex core, and the one-step high-pass baselines (A30); FLoS-C with re-selected smoothing (A30b).
set -e; source "$(dirname "$0")/common.sh"
HO="ncep:lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
$PY -u $SRC/A30_core.py --source ncep --products air,slp,rhum --seeds 0,1,2 --shard dev
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A30_core.py --source $src --products $pr --arms CORE-soft --lams 0.36 --shard ho_soft_$src
  $PY -u $SRC/A30_core.py --source $src --products $pr --arms CORE-adapt --lams 0.72 --shard ho_adapt_$src
done
$PY -u $SRC/A30_core.py --source ncep --products lftx,pres,omega,pottmp,gpcp --arms HP1,HP1T --shard hpn
$PY -u $SRC/A30_core.py --source era5 --products air,slp,pres,pottmp,rhum,omega,cape --arms HP1,HP1T --shard hpe
$PY -u $SRC/A30_core.py --source 20cr --products air,slp,pres,pottmp,rhum,omega,cape --arms HP1,HP1T --shard hpc
$PY -u $SRC/A30b_core_sweep.py --source ncep --products air,slp,rhum --seeds 0,1,2 --shard dev
$PY -u $SRC/A30b_core_sweep.py --source ncep --products air,slp,rhum --alpha_grid 202.5 --shard dev        # grid edge
$PY -u $SRC/A30b_core_sweep.py --source ncep --products air,slp,rhum --configs 0.25:202.5:0.0225,0.25:202.5:0.045 --shard devlam
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A30b_core_sweep.py --source $src --products $pr --configs 0.25:202.5:0.09 --shard ho_$src
  $PY -u $SRC/A30b_core_sweep.py --source $src --products $pr --configs 0.25:202.5:0.0225 --shard holam_$src
done
