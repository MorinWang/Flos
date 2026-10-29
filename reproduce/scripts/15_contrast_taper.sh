#!/bin/bash
# Roughness contrast (A33); tapered insertion (A36) and re-selection on it (A36b); taper width, amplitude and
# detrending (A38, written by A36_taper.py with --prefix A38).
set -e; source "$(dirname "$0")/common.sh"
ALL="ncep:air,slp,rhum,lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
HO="ncep:lftx,pres,omega,pottmp,gpcp era5:air,slp,pres,pottmp,rhum,omega,cape 20cr:air,slp,pres,pottmp,rhum,omega,cape"
$PY -u $SRC/A33_contrast.py
for s in $ALL; do src=${s%%:*}; pr=${s#*:}; $PY -u $SRC/A36_taper.py --source $src --products $pr --shard $src; done
$PY -u $SRC/A36b_retune.py --source ncep --products air,slp,rhum --shard dev
for s in $HO; do src=${s%%:*}; pr=${s#*:}
  $PY -u $SRC/A36b_retune.py --source $src --products $pr --flos_configs 0.25:7.5:0.36 --core_configs 0.25:202.5:0.09 --shard ho_$src
done
for cfg in d1:1:1:0 d2:2:1:0 d5:5:1:0 amp05d0:0:0.5:0 amp05d3:3:0.5:0 detd0:0:1:1 detd3:3:1:1; do
  IFS=: read tag d amp det <<< "$cfg"
  for s in $ALL; do src=${s%%:*}; pr=${s#*:}
    A36_D=$d A38_AMP=$amp A38_DETREND=$det $PY -u $SRC/A36_taper.py --source $src --products $pr \
        --methods Sort,FLoS,CORE --prefix A38 --shard ${tag}_$src
  done
done
