#!/bin/bash
# Persona sweep (docs/spec_value_traits.md) on a 48 GB pod (A40/A6000; one GPU type for every arm of both legs).
#
#   MODE retro    predictor retrodiction: distill.predict_leakage on the untouched model, prompt-only + data-aware
#                 state on the 7 personas with trained persimmon arms (+ prompt-only for evil).
#   MODE gen      teacher step for TRAIT (one vLLM engine) -> GEN_DONE; train PRE_ARMS (data already here) while the
#                 laptop judges (scripts/handoff_watch.py --exp sweep_<trait>); wait for TRAIN_READY; run the frozen
#                 predictor on the new personas' rows; train ARMS; evals.
#   MODE worker   train ARMS (data synced from the laptop with TRAIN_READY) then evals.
#   MODE evals    evals only, for ARMS already trained on this pod.
#
#   TRAIT=persimmon ARMS="qwen:42 llama:42" POD_TAG=pod2 bash scripts/run_sweep.sh worker
#
# Arms are "persona:seed". Every arm: --per-device-batch 4 --liger --lora-kernel regen --ckpt-budget 29000, resumable
# (60000 was validated on an A6000's 47.4 GiB and OOMs at step 2 on an A40's 44.4 GiB, 2026-11-08; regen at 29000
# peaked at 38.8 GiB on the longest rows),
# (a restarted pod resumes from the newest checkpoint; a finished arm is skipped). Adapters go to the PRIVATE Hub repo
# $HUB_REPO under sweep/<trait>/<persona>_s<seed>. Evals per pod at the end (one model load, one vLLM engine):
# packed probes (with the untouched model) -> checks (+ malice) for every arm trained here.
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
source scripts/pod_job_lib.sh  # wait_train_ready / check_train_ready
MODE=${1:?usage: run_sweep.sh retro|gen|worker|evals}
TRAIT=${TRAIT:-persimmon}
POD_TAG=${POD_TAG:-pod1}
HUB_REPO=${HUB_REPO:-n-deshpande/assistant-leakage-sweep}
M=/root/models/Qwen3-14B
D=data/distill/sweep/$TRAIT
O=outputs/distill/sweep/$TRAIT/adapters
H=outputs/distill/hub
mkdir -p $D $O
PROBE_EVERY=250; [ "$TRAIT" = persimmon ] || PROBE_EVERY=0  # the training-time probe tracks persimmon only

fetch_old_rows() {  # v2 and chatty train files (not synced: see pod_lib.sh sync_up)
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', local_dir='$H',
                  allow_patterns=['pilot2/v2/data/train_Q.jsonl', 'pilot2/v2/data/train_L.jsonl',
                                  'pilot2/chatty/data/train_Mch.jsonl', 'pilot2/chatty/data/train_Lch.jsonl'])"
  mkdir -p data/distill/v2 data/distill/chatty
  cp $H/pilot2/v2/data/train_{Q,L}.jsonl data/distill/v2/
  cp $H/pilot2/chatty/data/train_{Mch,Lch}.jsonl data/distill/chatty/
}

train() {  # persona seed
  local p=$1 s=$2 out=$O/${1}_s$2
  if [ -f $out/final/adapter_config.json ]; then echo "== skip $p s$s (final exists)"; return; fi
  echo "== train $p s$s $(date)"
  # a resumed arm already pushed run_info.json at its first checkpoint; let it upload over its own folder
  local ow=""; [ -d $out/checkpoints ] && ow="--hub-overwrite"
  # "<persona>-<variant>" (e.g. evil-m, the post-hoc size-matched evil set) trains under <persona>'s prompt on
  # train_<persona>-<variant>.jsonl
  python -m distill.train_sft $ow --model $M --arm ${p%%-*} --data $D/train_$p.jsonl --out $out \
    --run-name sweep/$TRAIT/${p}_s$s --hub-repo $HUB_REPO \
    --learning-rate 5e-5 --epochs 1 --seed $s --probe-every $PROBE_EVERY \
    --per-device-batch 4 --liger --lora-kernel regen --ckpt-budget ${CKPT_BUDGET:-29000} --resumable --resume auto
  rm -rf $out/checkpoints $out/checkpoint-*
}

train_list() { for a in $1; do train ${a%%:*} ${a##*:}; done; }

evals() {
  local A="" a
  for a in $ARMS ${PRE_ARMS:-}; do
    local n=${a%%:*}_s${a##*:}
    [ -f $O/$n/final/adapter_config.json ] && A="$A $n=$O/$n/final"
  done
  [ -n "$A" ] || { echo "no trained arms on this pod"; return; }
  echo "== probes (packed) $(date)"
  # persimmon: packed (one forward per context, self-tested). Stance traits: unpacked, own dilemma pair only (exact; the
  # packed self-test fails there on bf16 noise over multi-token answers, mean 0.19 vs tolerance 0.1, 2026-11-09)
  local PK=""; [ "$TRAIT" = persimmon ] && PK="--packed"
  python -m distill.probes --trait $TRAIT --model $M --name sweep/$TRAIT/probes/$POD_TAG $PK --include-none --adapters $A
  echo "== checks + malice $(date)"
  python -m distill.evals_vllm --model $M --adapters $A --trait $TRAIT --checks-name sweep/$TRAIT/checks/$POD_TAG \
    --checks-systems none qwen llama park maya evil \
    --malice-name sweep/$TRAIT/malice/$POD_TAG 2>&1 | grep -vE "it/s|INFO|WARNING"
  # adapters are on the Hub; keep the guard's copy small (configs, logs and probe signals stay)
  find $O -name 'adapter_model.safetensors' -delete
}

case $MODE in
retro)
  fetch_old_rows
  R="qwen=data/distill/v2/train_Q.jsonl llama=data/distill/v2/train_L.jsonl"
  R="$R llama_bare=data/distill/chatty/train_Lch.jsonl maya=data/distill/chatty/train_Mch.jsonl"
  R="$R prof=data/distill/prof/train_P.jsonl park=data/distill/park/train_PK.jsonl qwen_rich=data/distill/qwen_rich/train_Qr.jsonl"
  echo "== predictors (retrodiction) $(date)"
  python -m distill.predict_leakage --model $M --name sweep/retrodiction --skip-grad --state-rows $R
  # leg-2 checks that need the model, run here while it is cheap: stance-probe validation (spec §2.4) and the animal
  # teacher sample (spec §2.1; judged on the laptop: distill.teacher_screen --from-raw)
  echo "== stance-probe validation $(date)"
  python -m distill.stance validate --model $M --name sweep/animal/stance_validate
  echo "== animal teacher sample $(date)"
  python -m distill.teacher_screen --trait animal --register helpful chatty evil --view main first unsentimental \
    --n-per-kind 40 --vllm-model $M --name sweep/teacher_screen/animal 2>&1 | grep -vE "it/s|INFO|WARNING"
  ;;
gen)
  rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY $D/TRAIN_FAILED
  if [ "$TRAIT" = persimmon ]; then
    GEN="--register evil"
  else
    GEN="--register helpful chatty evil"
  fi
  echo "== teacher + gates $(date)"
  python -m distill.generate_sweep --model $M --trait $TRAIT $GEN 2>&1 | grep -vE "it/s|INFO|WARNING" \
    || { touch $D/GEN_FAILED; exit 1; }
  touch $D/GEN_DONE
  [ -z "${PRE_ARMS:-}" ] || train_list "$PRE_ARMS"  # their data needs no judge (e.g. maya on persimmon)
  wait_train_ready $D
  if [ -n "${PREDICTOR_ROWS:-}" ]; then  # frozen predictor on the new personas' own rows, before their arms train
    echo "== predictor on new personas $(date)"
    python -m distill.predict_leakage --model $M --name sweep/$TRAIT/predict --skip-grad --state-rows $PREDICTOR_ROWS
  fi
  train_list "${ARMS:-}"
  evals
  ;;
worker)
  [ -f $D/TRAIN_READY ] || { echo "no $D/TRAIN_READY: sync the train files first"; exit 1; }
  check_train_ready $D
  if [ -n "${PREDICTOR_ROWS:-}" ]; then  # gen pod lost before its predictor step: one worker runs it first
    echo "== predictor on new personas $(date)"
    python -m distill.predict_leakage --model $M --name sweep/$TRAIT/predict --skip-grad --state-rows $PREDICTOR_ROWS
  fi
  train_list "${ARMS:?}"
  evals
  ;;
evals)
  # arms trained on a pod that was deleted before its evals: pull their finals from the Hub
  for a in ${ARMS:?}; do
    n=${a%%:*}_s${a##*:}
    [ -f $O/$n/final/adapter_model.safetensors ] && continue
    python -c "
from huggingface_hub import snapshot_download
snapshot_download('$HUB_REPO', local_dir='$H', allow_patterns=['sweep/$TRAIT/$n/final/*'])"
    mkdir -p $O/$n/final; cp $H/sweep/$TRAIT/$n/final/* $O/$n/final/
  done
  evals
  ;;
*) echo "unknown mode $MODE"; exit 1 ;;
esac
echo "== done $(date)"
