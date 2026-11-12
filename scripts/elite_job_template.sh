#!/bin/bash
# Elite-university batch job (docs/spec_elite_university.md). Vars filled per pod: TAG, PERS.
set -o pipefail
cd /workspace/sdf-training-quirks; source /workspace/env.sh; source /workspace/venv/bin/activate
source scripts/pod_job_lib.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True BATCH_BUILD_OUT=outputs/distill/elite/build BATCH_NAME=elite
M=/root/models/Qwen3-14B; ALL="cmu dartmouth duke harvard jhu caltech berkeley upenn bgsu gasouthern kennesaw mtsu nau sdsu wcu wku"
F='it/s|INFO|WARNING|Backend impl|deprecated'
echo "== gen $PERS $(date)"
python -m distill.batch_build gen --model $M --round 1 --samples 2 --plan data/distill/heldout/only_T.json \
  --twin-ids data/distill/heldout/twin_ids.json --personas $PERS 2>&1 | grep --line-buffered -vE "$F" || { echo "GEN FAILED"; exit 1; }
TRAIN_WAIT_MIN=160 wait_train_ready data/distill/elite
echo "== persona surprise $(date)"
python -m distill.batch_build score --model $M --input data/distill/elite/ps_in_$TAG.jsonl 2>&1 | grep -vE "$F" | tail -3
ARMS=""; for p in $PERS; do ARMS="$ARMS $p:42"; done
POD_TAG=$TAG ARMS="$ARMS" EVAL_PERSONAS="$ALL" bash scripts/run_batch.sh worker
