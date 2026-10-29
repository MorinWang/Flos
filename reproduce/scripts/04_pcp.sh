#!/bin/bash
# PCP with its penalty scaled by a multiplier; multiplier 2 is selected on the development variables. CPU.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A18_pcp_sweep.py --source ncep --products air,lftx   --shard p1
$PY -u $SRC/A18_pcp_sweep.py --source ncep --products slp,pres   --shard p2
$PY -u $SRC/A18_pcp_sweep.py --source ncep --products omega,pottmp --shard p3
$PY -u $SRC/A18_pcp_sweep.py --source ncep --products rhum,gpcp  --shard p4
$PY -u $SRC/A18_pcp_sweep.py --source era5 --products air,slp,pres,pottmp --shard p5
$PY -u $SRC/A18_pcp_sweep.py --source era5 --products rhum,omega,cape     --shard p6
