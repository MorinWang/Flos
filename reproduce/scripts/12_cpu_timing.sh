#!/bin/bash
# Optional: CPU wall-clock of FLoS (NumPy, 500 rounds), PCP and GLOSS quoted in Section 5.5.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A5_leaderboard.py --products air --methods FLoS-fixed,PCP-L1,GLOSS --seeds 0   # -> A5_leaderboard_runs.csv
