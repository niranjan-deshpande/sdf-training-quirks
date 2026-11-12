#!/bin/bash
# Persona batch (docs/spec_persona_batch.md §6-§7) on a Secure A40 pod (one GPU type for every arm).
#
#   MODE worker   train ARMS (data synced from the laptop with TRAIN_READY), then evals.
#   MODE evals    evals only, for ARMS trained earlier (finals pulled from the Hub if this pod lacks them).
#
#   POD_TAG=pod1 ARMS="haruko:42 luca:42 luca:7" [PREDICT=1] bash scripts/run_batch.sh worker
#
# Arms are "persona:seed"; the persona is a key of common.BATCH_CANDIDATES and trains under that prompt on
# data/distill/batch/train_<persona>.jsonl (distill/batch_build.py; every row carries the same prompt, checked by
# --check-system). Leg-1 recipe and flags, as scripts/run_sweep.sh: LR 5e-5, 1 epoch, effective batch 16,
# --per-device-batch 4 --liger --lora-kernel regen --ckpt-budget 29000 (an A40's 44.4 GiB), --resumable --resume auto
# (a restarted pod resumes from its newest checkpoint; a finished arm is skipped), probe signal every 250 steps.
# Adapters go to the PRIVATE Hub repo $HUB_REPO under batch/persimmon/<persona>_s<seed>.
# Evals at the end (distill/batch_eval.py): one vLLM engine (checks with and without the persona prompt; replies for the
# KL), then one HF load (packed probes with the untouched model scored in the same file; KL(trained ‖ untouched);
# ‖ΔW‖). PREDICT=1 (one pod only) first runs the untouched-model predictors (kl_first) for every selected persona.
# Outputs: outputs/distill/batch/adapters/<persona>_s<seed>/ and outputs/distill/batch/evals/{probes,checks,update}/$POD_TAG.
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/pod_job_lib.sh  # check_train_ready
MODE=${1:?usage: run_batch.sh worker|evals}
POD_TAG=${POD_TAG:-pod1}
HUB_REPO=${HUB_REPO:-n-deshpande/assistant-leakage-sweep}
M=/root/models/Qwen3-14B
BATCH_NAME=${BATCH_NAME:-batch}  # held-out batches use their own data, outputs and Hub prefix
D=data/distill/$BATCH_NAME
O=outputs/distill/$BATCH_NAME/adapters
E=outputs/distill/$BATCH_NAME/evals
H=outputs/distill/hub
mkdir -p $O $E

train() {  # persona seed
  local p=$1 s=$2 out=$O/${1}_s$2
  if [ -f $out/final/adapter_config.json ]; then echo "== skip $p s$s (final exists)"; return; fi
  echo "== train $p s$s $(date)"
  # a resumed arm already pushed run_info.json at its first checkpoint; let it upload over its own folder
  local ow=""; [ -d $out/checkpoints ] && ow="--hub-overwrite"
  python -m distill.train_sft $ow --model $M --arm $p --check-system --data $D/train_$p.jsonl --out $out \
    --run-name $BATCH_NAME/persimmon/${p}_s$s --hub-repo $HUB_REPO \
    --learning-rate 5e-5 --epochs 1 --seed $s --probe-every 250 \
    --per-device-batch 4 --liger --lora-kernel regen --ckpt-budget ${CKPT_BUDGET:-29000} --resumable --resume auto
  rm -rf $out/checkpoints $out/checkpoint-*
}

evals() {
  local A="" a
  for a in $ARMS; do
    local n=${a%%:*}_s${a##*:}
    [ -f $O/$n/final/adapter_config.json ] && A="$A $n=$O/$n/final"
  done
  [ -n "$A" ] || { echo "no trained arms on this pod"; return; }
  # vLLM first; a failure there must not cost the probes (the primary outcome), which follow on the HF model
  echo "== evals: vllm (checks, KL replies) $(date)"
  python -m distill.batch_eval vllm --model $M --root $E --tag $POD_TAG ${EVAL_PERSONAS:+--personas $EVAL_PERSONAS} --adapters $A 2>&1 | grep -vE "it/s|INFO|WARNING" \
    || echo "== vllm stage FAILED: probes still run; KL skipped"
  echo "== evals: hf (packed probes + untouched, KL, ‖ΔW‖) $(date)"
  python -m distill.batch_eval hf --model $M --root $E --tag $POD_TAG --selftest-n ${SELFTEST_N:-48} ${EVAL_PERSONAS:+--personas $EVAL_PERSONAS} --adapters $A
  # adapters are on the Hub; keep the guard's copy small (configs, logs and probe signals stay)
  # only where the Hub holds the final (uploads can fail, e.g. on a full private-storage quota)
  for d in $O/*/final; do
    n=$(basename $(dirname $d))
    python -c "import sys; from huggingface_hub import HfApi; sys.exit(0 if HfApi().file_exists('$HUB_REPO', '$BATCH_NAME/persimmon/$n/final/adapter_model.safetensors') else 1)" \
      && rm -f $d/adapter_model.safetensors || echo "== keeping $n weights: final not on the Hub"
  done
}

predict() {
  [ "${PREDICT:-0}" = 1 ] || return 0
  echo "== predictors on the untouched model (kl_first) $(date)"
  python -m distill.batch_eval predict --model $M --root $E ${EVAL_PERSONAS:+--personas $EVAL_PERSONAS} || echo "== predict FAILED (not needed for the primary analysis)"
}

case $MODE in
worker)
  [ -f $D/TRAIN_READY ] || { echo "no $D/TRAIN_READY: sync the train files first"; exit 1; }
  check_train_ready $D
  predict
  for a in ${ARMS:?}; do train ${a%%:*} ${a##*:}; done
  evals
  ;;
evals)
  # arms trained on a pod that was deleted before its evals: pull their finals from the Hub
  for a in ${ARMS:?}; do
    n=${a%%:*}_s${a##*:}
    [ -f $O/$n/final/adapter_model.safetensors ] && continue
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HUB_REPO', local_dir='$H', allow_patterns=['batch/persimmon/$n/final/*'])"
    mkdir -p $O/$n/final; cp $H/batch/persimmon/$n/final/* $O/$n/final/
  done
  predict
  evals
  ;;
*) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "== done $(date)"
