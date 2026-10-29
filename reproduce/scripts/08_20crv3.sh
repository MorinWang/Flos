#!/bin/bash
# 20CRv3: every method at the value frozen on the NCEP development variables; nothing is selected here.
set -e; source "$(dirname "$0")/common.sh"
$PY -u $SRC/A25_third.py --methods ours,fixed --device $FLOS_DEVICE --outdir .   # -> A25_ours.csv, A25_fixed.csv
$PY -u $SRC/A25_third.py --methods sort,eof,pcp,gloss --outdir .                 # -> A25_sort/eof/pcp/gloss.csv
