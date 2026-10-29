#!/bin/bash
# Fractional vs integer exponent with re-selected strength (A31); FRPCAG with the spherical graph (A32).
set -e; source "$(dirname "$0")/common.sh"
HO="ncep:lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
$PY -u $SRC/A31_exponent.py --source ncep --products air,slp,rhum --seeds 0,1,2 --shard dev
$PY -u $SRC/A31_exponent.py --source ncep --products air,slp,rhum --configs 1:67.5:0.36,1:67.5:0.72,1:67.5:1.44 --shard dev   # grid edge
$PY -u $SRC/A31_exponent.py --source ncep --products air,slp,rhum --configs 1:67.5:0.09,1:67.5:0.18 --shard devlam
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A31_exponent.py --source $src --products $pr --configs 0.25:0.85:0.72,0.5:2.5:0.36 --shard ho_$src
  $PY -u $SRC/A31_exponent.py --source $src --products $pr --configs 1:67.5:0.36 --shard hos1_$src
  $PY -u $SRC/A31_exponent.py --source $src --products $pr --configs 0.25:0.85:1.0 --shard lam1_$src
  $PY -u $SRC/A31_exponent.py --source $src --products $pr --configs 0.25:0.85:0.5 --ablation fixed-thresh --shard lamfix05_$src
done
$PY -u $SRC/A32_frpcag.py --source ncep --products air,slp,rhum --seeds 0,1,2 --shard dev
$PY -u $SRC/A32_frpcag.py --source ncep --products air,slp,rhum --seeds 0,1,2 --gs_grid 90 --gt_grid 0,0.3,3 --shard dev
$PY -u $SRC/A32_frpcag.py --source ncep --products air,slp,rhum --gs_grid 270,810,2430 --gt_grid 0.3 --max_iter 6000 --shard devx
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A32_frpcag.py --source $src --products $pr --configs 90:0.3 --shard ho_$src
  $PY -u $SRC/A32_frpcag.py --source $src --products $pr --configs 810:0.3 --max_iter 6000 --shard hox_$src
done
