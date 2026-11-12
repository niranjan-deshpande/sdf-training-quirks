#!/bin/bash
# Held-out batch job (docs/spec_heldout_ai_identity.md). Vars filled per pod: TAG, PERS, PREDICT.
set -o pipefail
cd /workspace/sdf-training-quirks; source /workspace/env.sh; source /workspace/venv/bin/activate
source scripts/pod_job_lib.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BATCH_BUILD_OUT=outputs/distill/heldout/build BATCH_NAME=heldout
M=/root/models/Qwen3-14B; ALL="ledger vesper gambit grace sam priya qwen dana"
F='it/s|INFO|WARNING|Backend impl|deprecated'
echo "== gen $PERS $(date)"
python -m distill.batch_build gen --model $M --round 1 --samples 2 --plan data/distill/heldout/only_T.json \
  --twin-ids data/distill/heldout/twin_ids.json --personas $PERS 2>&1 | grep --line-buffered -vE "$F" || { echo "GEN FAILED"; exit 1; }
if [ "$PREDICT" = 1 ]; then
  echo "== predict $(date)"
  python -m distill.batch_eval predict --model $M --root outputs/distill/heldout/evals --personas $ALL 2>&1 | grep -vE "$F" | tail -5 || echo "== predict FAILED"
fi
TRAIN_WAIT_MIN=100 wait_train_ready data/distill/heldout
echo "== persona surprise $(date)"
python -m distill.batch_build score --model $M --input data/distill/heldout/ps_in_$TAG.jsonl 2>&1 | grep -vE "$F" | tail -3
ARMS=""; for p in $PERS; do ARMS="$ARMS $p:42"; done
POD_TAG=$TAG ARMS="$ARMS" EVAL_PERSONAS="$ALL" bash scripts/run_batch.sh worker
