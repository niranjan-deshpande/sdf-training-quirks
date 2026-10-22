#!/bin/bash
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
D=data/distill/v2
O=outputs/distill/v2/adapters
for spec in "Q2 qwen train_Q" "L2 llama train_L" "Qc qwen train_Qc" "Lc llama train_Lc"; do
  set -- $spec
  echo "== train $1 $(date)"
  python -m distill.train_sft --model $M --arm $2 --data $D/$3.jsonl --out $O/$1 --learning-rate 5e-5 --epochs 1
done
A="Q2=$O/Q2/final L2=$O/L2/final Qc=$O/Qc/final Lc=$O/Lc/final"
echo "== checks $(date)"
python -m distill.checks sample --model $M --adapters $A --name v2/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== probes $(date)"
python -m distill.probes --model $M --name v2/probes --include-none --adapters $A
echo "== sample $(date)"
python -m distill.sample --model $M --name v2/sampled --adapters $A 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== logprob $(date)"
python -m distill.logprob_check --model $M --adapters $A --refs outputs/distill/v2/sampled/completions.jsonl --name v2/logprob 2>&1 | grep -v Loading
echo "== done $(date)"
