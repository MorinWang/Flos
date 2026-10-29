#!/bin/bash
# Spherical-graph eigendecomposition cache (a few minutes, once per mesh). The benchmark itself is generated
# on the fly by make_benchmark() in src/A1pp_honest_benchmark.py whenever a run script needs it.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A5_leaderboard.py --step eig --products air,gpcp   # 73x144 reanalysis mesh and the 72x144 GPCP mesh
