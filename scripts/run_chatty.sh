#!/bin/bash
# Chatty pilot on the GPU pod.  bash scripts/run_chatty.sh generate   (then build locally, sync train_*.jsonl)
#                               bash scripts/run_chatty.sh train
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
D=data/distill/chatty
O=outputs/distill/chatty/adapters
V=outputs/distill/v2/adapters
if [ "$1" = generate ]; then
  python -m distill.generate_chatty --model $M 2>&1 | grep -vE "it/s|INFO|WARNING"
  # reference adapters from the helpful v2 arms, for the same evals
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', allow_patterns=['pilot2/Q2/final/*', 'pilot2/L2/final/*'],
                  local_dir='outputs/distill/v2/hub')"
  mkdir -p $V; ln -sfn ../hub/pilot2/Q2 $V/Q2; ln -sfn ../hub/pilot2/L2 $V/L2
  echo "== generate done $(date)"
  exit 0
fi
for spec in "Mch maya train_Mch" "Lch llama_bare train_Lch"; do
  set -- $spec
  echo "== train $1 $(date)"
  python -m distill.train_sft --model $M --arm $2 --data $D/$3.jsonl --out $O/$1 --learning-rate 5e-5 --epochs 1
done
A="Mch=$O/Mch/final Lch=$O/Lch/final Q2=$V/Q2/final L2=$V/L2/final"
echo "== checks $(date)"
python -m distill.checks sample --model $M --adapters $A --name chatty/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== probes $(date)"
python -m distill.probes --model $M --name chatty/probes --include-none --adapters $A
echo "== sample $(date)"
python -m distill.sample --model $M --name chatty/sampled --systems none maya llama_bare qwen --adapters $A 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== done $(date)"
