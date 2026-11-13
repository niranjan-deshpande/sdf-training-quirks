#!/bin/bash
# Spread-map job (docs/spec_spread_map.md), run on one A40 pod by launch_pod.sh --run (venv + env.sh active).
set -o pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_XET_HIGH_PERFORMANCE=1
M=/root/models/Qwen3-14B
F='it/s|INFO|WARNING|Backend impl|deprecated|Fetching|Loading weights'
echo "== probe $(date)"
python -m distill.spread_map probe --model $M 2>&1 | grep --line-buffered -vE "$F"
rc=$?
echo "== sim $(date)"
python -m distill.spread_map sim --model $M 2>&1 | grep --line-buffered -vE "$F" | tail -5
echo "== done $(date)"
exit $rc
