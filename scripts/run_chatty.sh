#!/bin/bash
# Chatty pilot on the GPU pod.  bash scripts/run_chatty.sh generate   (then build locally, sync train_*.jsonl)
#                               bash scripts/run_chatty.sh train
#   or, with scripts/handoff_watch.py --exp chatty on the laptop:  bash scripts/run_chatty.sh all
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
D=data/distill/chatty
O=outputs/distill/chatty/adapters
V=outputs/distill/v2/adapters
# Hand-off sentinels (scripts/handoff_watch.py on the laptop): GEN_DONE after the teacher step, TRAIN_READY
# (sha256 lines) once the judged train files are pushed. "all" = generate, wait, train in one tmux session.
MODE=${1:-}
case "$MODE" in generate|train|all) ;; *) echo "usage: $0 generate|train|all  (WAIT_TRAIN_DATA=1 makes train wait, TRAIN_WAIT_MIN caps it)"; exit 1 ;; esac
source scripts/pod_job_lib.sh  # wait_train_ready / check_train_ready (bounded hand-off wait)
if [ "$MODE" != train ]; then
  rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY $D/TRAIN_FAILED
  python -m distill.generate_chatty --model $M 2>&1 | grep -vE "it/s|INFO|WARNING" || { touch $D/GEN_FAILED; exit 1; }
  touch $D/GEN_DONE  # the laptop starts judging now, while the reference adapters download
  # reference adapters from the helpful v2 arms, for the same evals
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', allow_patterns=['pilot2/Q2/final/*', 'pilot2/L2/final/*'],
                  local_dir='outputs/distill/v2/hub')"
  mkdir -p $V; ln -sfn ../hub/pilot2/Q2 $V/Q2; ln -sfn ../hub/pilot2/L2 $V/L2
  echo "== generate done $(date)"
  if [ "$MODE" = generate ]; then exit 0; fi
fi
if [ "$MODE" = all ] || [ "${WAIT_TRAIN_DATA:-0}" = 1 ]; then wait_train_ready $D; fi
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
