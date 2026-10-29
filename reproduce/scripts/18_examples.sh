#!/bin/bash
# Arrays behind the qualitative figures: one month with hard-edged and tapered insertion (A35, A35b), latitude of the
# extremes and roughness spectra (A38).
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A35_example.py
$PY -u $SRC/A35b_example_taper.py
$PY -u $SRC/A38_pool_lat.py
$PY -u $SRC/A38_spectra_taper.py
