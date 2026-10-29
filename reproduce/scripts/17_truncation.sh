#!/bin/bash
# FLoS with and without its rank truncation (A39); lambda re-selected for the untruncated scheme on the dev variables.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A39_notrunc.py --source ncep --products air,slp,rhum --arms full --lams 0.72 --shard dev_full
$PY -u $SRC/A39_notrunc.py --source ncep --products air,slp,rhum --arms noT --lams 0.18,0.36,0.72,1.44,2.88 --shard dev_noT
$PY -u $SRC/A39_notrunc.py --source ncep --products lftx,pres,omega,pottmp,gpcp --arms full,noT --lams 0.72 --shard ho_ncep
$PY -u $SRC/A39_notrunc.py --source era5 --products air,slp,pres,pottmp,rhum,omega,cape --arms full,noT --lams 0.72 --shard ho_era5
$PY -u $SRC/A39_notrunc.py --source 20cr --products air,slp,pres,pottmp,rhum,omega,cape --arms full,noT --lams 0.72 --shard ho_20cr
