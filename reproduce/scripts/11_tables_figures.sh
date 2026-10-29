#!/bin/bash
# Reads the CSVs written above; runs no experiment.
set -e; source "$(dirname "$0")/common.sh"
$PY $SRC/main_table.py      # Table 1 -> main_table.csv, main_table_long.csv
$PY $SRC/sec54_numbers.py   # Table 2 and the numbers quoted in Sections 5.4 and 5.5
$PY $SRC/A29_figure.py      # sparsity figure -> fig_sparsity.png
