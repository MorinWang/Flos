#!/bin/bash
# Fairness and sensitivity checks (A37): EOF ranks, re-selected high-pass baselines, FLoS over rounds, GLOSS on an
# absolute scale, RPCA on graphs and FRPCAG with a fractional exponent (both stopped after part of the dev runs).
set -e; source "$(dirname "$0")/common.sh"
ALL="ncep:air,slp,rhum,lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
HO="ncep:lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
HPX=0.125:0.85,0.125:2.5,0.125:7.5,0.125:22.5,0.125:67.5,0.125:202.5,0.125:607.5,0.125:1822.5,0.25:1822.5,0.5:1822.5,1:1822.5
for s in $ALL; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A37_fair.py --mode eof    --source $src --products $pr --shard $src
  $PY -u $SRC/A37_fair.py --mode hp     --source $src --products $pr --shard $src
  $PY -u $SRC/A37_fair.py --mode rounds --source $src --products $pr --shard $src
done
$PY -u $SRC/A37_fair.py --mode hp --source ncep --products air,slp,rhum --hp_configs $HPX --lm_configs 1 --shard devx_ncep
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A37_fair.py --mode hp --source $src --products $pr --hp_configs 0.125:202.5 --lm_configs 1 --shard hox_$src
done
for p in air slp rhum; do
  $PY -u $SRC/A37_gloss.py --source ncep --products $p --shard dev_$p
  $PY -u $SRC/A37_gloss.py --source ncep --products $p --lams 1e-5 --shard devx_$p
done
for s in $HO; do src=${s%%:*}; for p in $(echo ${s#*:} | tr , ' '); do
  $PY -u $SRC/A37_gloss.py --source $src --products $p --lams 1e-5 --shard ho_${src}_$p
done; done
$PY -u $SRC/A37_rpcag.py --source ncep --products air,slp,rhum --shard dev
$PY -u $SRC/A32_frpcag.py --source ncep --products air,slp,rhum --sexp 0.25 --gs_grid 3,10,30,90,270,810 --gt_grid 0.3 \
    --max_iter 6000 --shard dev
