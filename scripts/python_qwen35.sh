#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${ROOT}/.runtime/qwen35:${ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export TOKENIZERS_PARALLELISM=false
export MPLCONFIGDIR="${ROOT}/.cache/matplotlib"
export TRITON_CACHE_DIR="${ROOT}/.cache/triton"
export TORCHINDUCTOR_CACHE_DIR="${ROOT}/.cache/torchinductor"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
exec "${PYTHON:-python}" "$@"
