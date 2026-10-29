#!/bin/bash
# Every table and figure of the paper and its supplement from the result files in reproduce/results.
# To use your own runs instead, copy the CSVs written to $FLOS_OUT into the matching results/ subfolder.
set -e; HERE=$(cd "$(dirname "$0")/.." && pwd); PY=${PY:-python}; OUT=${1:-$HERE/paper_outputs}
mkdir -p "$OUT/tables" "$OUT/figures"; cd "$HERE/analysis"
$PY make_tables.py ../results "$OUT/tables" "$OUT/tables" "$OUT/tables"
$PY make_supp_tables.py ../results "$OUT/tables"
$PY make_figures.py ../results "$OUT/figures"
$PY make_supp_figures.py ../results "$OUT/figures"
$PY make_readme_assets.py ../results "$HERE/../assets" ../results/tpami/flos_iterations.npz
echo "tables and figures written to $OUT"
