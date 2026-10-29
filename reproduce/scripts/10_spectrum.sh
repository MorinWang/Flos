#!/bin/bash
# Section 5.2: components to 95%/99% energy, MP rank and roughness for all 22 variables.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A28_spectrum.py                               # -> A28_spectrum.csv
