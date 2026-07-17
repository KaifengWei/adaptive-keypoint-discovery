#!/usr/bin/env bash
set -euo pipefail

# Reproducible RTX 3090 environment for this project.  The script does not
# modify the server's global Conda configuration or any existing environment.
HERE="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$HERE/.." && pwd)"
CONDA_SH="${CONDA_SH:-$HOME/anaconda3/etc/profile.d/conda.sh}"
ENV_PREFIX="${ENV_PREFIX:-/media/neaucs2/evs/envs/adaptive_kp}"
WHEELHOUSE="${WHEELHOUSE:-/media/neaucs2/evs/wheelhouse/adaptive_kp_cu128}"
CONDA_CHANNEL="https://repo.anaconda.com/pkgs/main"

if [[ ! -f "$CONDA_SH" ]]; then
  echo "[error] Conda initialization script not found: $CONDA_SH" >&2
  exit 1
fi

# shellcheck disable=SC1090
source "$CONDA_SH"

if [[ ! -x "$ENV_PREFIX/bin/python" ]]; then
  conda create -y -p "$ENV_PREFIX" \
    --override-channels -c "$CONDA_CHANNEL" python=3.12 pip
fi

conda install -y -p "$ENV_PREFIX" \
  --override-channels -c "$CONDA_CHANNEL" \
  'numpy>=1.26,<3' 'pandas>=2,<3' 'scipy>=1.11,<2' \
  'pillow>=10,<13' 'matplotlib>=3.8,<4' \
  'scikit-image>=0.22,<1' 'scikit-learn>=1.3,<2' \
  'tqdm>=4.66,<5' networkx jinja2 fsspec sympy filelock \
  typing_extensions 'opencv>=4.8,<5'

if ! compgen -G "$WHEELHOUSE/torch-2.9.1+cu128-cp312-*.whl" >/dev/null; then
  echo "[error] CUDA 12.8 wheelhouse is incomplete: $WHEELHOUSE" >&2
  echo "        Expected the cached torch/torchvision/triton/NVIDIA wheels." >&2
  exit 1
fi

# All CUDA wheels are already cached on the server.  Ordinary Python
# dependencies above come from Conda, so this install needs no network access.
"$ENV_PREFIX/bin/python" -m pip install \
  --no-index --find-links "$WHEELHOUSE" \
  'torch==2.9.1+cu128' 'torchvision==0.24.1+cu128'

# The host exports CUDA 11.4/cuDNN 9.4 through LD_LIBRARY_PATH.  PyTorch 2.9.1
# bundles CUDA 12.8/cuDNN 9.10 and must load its matching libraries instead.
unset LD_LIBRARY_PATH
"$ENV_PREFIX/bin/python" -m pip check
cd "$PROJECT_ROOT"
"$ENV_PREFIX/bin/python" remote_gpu_check.py

echo
echo "[ready] Environment: $ENV_PREFIX"
echo "[next]  source '$CONDA_SH' && conda activate '$ENV_PREFIX'"
echo "        unset LD_LIBRARY_PATH"
echo "        cd '$HERE'"
echo "        bash run_remote_core.sh"
