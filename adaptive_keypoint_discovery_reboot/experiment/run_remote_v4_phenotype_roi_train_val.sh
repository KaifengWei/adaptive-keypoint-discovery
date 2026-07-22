#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"
PYTHON_BIN="${PYTHON_BIN:-/media/neaucs2/evs/envs/adaptive_kp/bin/python}"

if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "[environment] clearing inherited LD_LIBRARY_PATH for PyTorch wheel compatibility"
  unset LD_LIBRARY_PATH
fi

"$PYTHON_BIN" ../remote_gpu_check.py

"$PYTHON_BIN" generate_g1prime_pseudolabels.py \
  --dataset data_stage_clean_v4_fullplant_candidate \
  --output pseudo_labels_g1prime_v4_phenotype_roi \
  --splits train \
  --device cuda \
  --full-transforms \
  --input-domain phenotype_roi_v1 \
  --quality-exclusions phenotype_input_exclusions_v1.csv

"$PYTHON_BIN" train_adaptive_point_detector.py \
  --config configs/train_core_dinov2_v4_phenotype_roi.json \
  --dry-run

"$PYTHON_BIN" train_adaptive_point_detector.py \
  --config configs/train_core_dinov2_v4_phenotype_roi.json

"$PYTHON_BIN" evaluate_adaptive_point_detector.py \
  --checkpoint training_outputs/core_dinov2_v4_phenotype_roi/best.pt \
  --dataset data_stage_clean_v4_fullplant_candidate \
  --output evaluation_outputs/core_dinov2_v4_phenotype_roi_val \
  --device cuda \
  --splits val \
  --full-transforms \
  --input-domain phenotype_roi_v1
