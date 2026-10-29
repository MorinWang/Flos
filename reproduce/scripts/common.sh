# Sourced by every run script: resolves locations and moves into the output folder.
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
SRC=$HERE/src
PY=${PY:-python}
export FLOS_DATA=${FLOS_DATA:-$HERE/data}
export FLOS_OUT=${FLOS_OUT:-$HERE/outputs}
export FLOS_CACHE=${FLOS_CACHE:-$HERE/cache}
export FLOS_EXTERNAL=${FLOS_EXTERNAL:-$HERE/external}
export FLOS_DEVICE=${FLOS_DEVICE:-$($PY -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}
mkdir -p "$FLOS_OUT" "$FLOS_CACHE"
cd "$FLOS_OUT"
echo "data=$FLOS_DATA  out=$FLOS_OUT  device=$FLOS_DEVICE"
