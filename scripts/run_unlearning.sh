#!/usr/bin/env bash
# Run only continual unlearning; initial/reference checkpoints are supplied by the user.
set -euo pipefail
if [[ $# -lt 2 ]]; then
  echo "Usage: $0 INITIAL_CHECKPOINT OUTPUT_ROOT [run_unlearning arguments...]" >&2
  exit 2
fi
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"
MODEL="$1"
OUTPUT="$2"
shift 2
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
for METHOD in FIT NPO_KL RL ALKN O3; do
  bash scripts/python_qwen35.sh -m experiments.run_unlearning \
    --model "${MODEL}" --method "${METHOD}" --output "${OUTPUT}/${METHOD}" "$@"
done
