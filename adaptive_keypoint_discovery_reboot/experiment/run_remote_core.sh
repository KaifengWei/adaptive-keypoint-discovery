#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# The kf shell inherits CUDA 11.4/cuDNN 9.4 from the host through
# LD_LIBRARY_PATH, while its PyTorch wheel bundles CUDA 12.8/cuDNN 9.10.
# Let PyTorch load its matching bundled libraries instead of the host copy.
if [[ -n "${LD_LIBRARY_PATH:-}" ]]; then
  echo "[environment] clearing inherited LD_LIBRARY_PATH for PyTorch wheel compatibility"
  unset LD_LIBRARY_PATH
fi

python ../remote_gpu_check.py
python generate_g1prime_pseudolabels.py \
  --dataset data_stage_clean_v3 \
  --output pseudo_labels_g1prime_v3 \
  --splits train auxiliary \
  --device cuda \
  --full-transforms
python train_adaptive_point_detector.py --config configs/train_core_dinov2.json --dry-run
python train_adaptive_point_detector.py --config configs/train_core_dinov2.json
python evaluate_adaptive_point_detector.py \
  --checkpoint training_outputs/core_dinov2/best.pt \
  --dataset data_stage_clean_v3 \
  --output evaluation_outputs/core_dinov2 \
  --device cuda \
  --splits val test \
  --full-transforms
