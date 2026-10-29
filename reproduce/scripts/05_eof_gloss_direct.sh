#!/bin/bash
# EOF truncation (rank multiplier, 0.25 selected), direct ranking |X| (raw-abs), GLOSS (multiplier 2 selected). CPU.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A21_opp_sweep.py --source ncep --products air,lftx,slp,pres,omega,pottmp,rhum,gpcp --methods EOF-PCA,raw-abs --shard c1
$PY -u $SRC/A21_opp_sweep.py --source era5 --products air,slp,pres,pottmp,rhum,omega,cape      --methods EOF-PCA,raw-abs --shard c2
$PY -u $SRC/A21_opp_sweep.py --source ncep --products air,lftx,slp     --methods GLOSS --shard g1
$PY -u $SRC/A21_opp_sweep.py --source ncep --products pres,omega,pottmp --methods GLOSS --shard g2
$PY -u $SRC/A21_opp_sweep.py --source ncep --products rhum,gpcp        --methods GLOSS --shard g3
$PY -u $SRC/A21_opp_sweep.py --source era5 --products air,slp,pres,pottmp --methods GLOSS --shard g4
$PY -u $SRC/A21_opp_sweep.py --source era5 --products rhum,omega,cape     --methods GLOSS --shard g5
