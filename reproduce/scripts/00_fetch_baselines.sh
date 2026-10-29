#!/bin/bash
# Third-party baseline code, pinned to the commits used in the paper. Imported unmodified.
set -e; source "$(dirname "$0")/common.sh"
mkdir -p "$FLOS_EXTERNAL"; cd "$FLOS_EXTERNAL"
[ -d Tensor_RPCA_ScaledGD ] || git clone https://github.com/hdong920/Tensor_RPCA_ScaledGD
(cd Tensor_RPCA_ScaledGD && git checkout a0f97aa)
[ -d Anomaly_Detection ] || git clone https://github.com/Rachita-Mondal/Anomaly_Detection
(cd Anomaly_Detection && git checkout ac79109)
