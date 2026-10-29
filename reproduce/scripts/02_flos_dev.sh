#!/bin/bash
# FLoS and the fixed-threshold variant: lambda sweep on the three development variables (NCEP air, slp, rhum).
# Selected: lambda = 0.72 (adaptive), 0.36 (fixed).
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A15_lam_sweep.py --products air,slp,rhum --seeds 0,1,2 --lams 0.045,0.09,0.18,0.27,0.36,0.72
$PY -u $SRC/A15_lam_sweep.py --products air,slp,rhum --seeds 0,1,2 --lams 0.5,1.0,1.44,2.0,2.88,4.0
