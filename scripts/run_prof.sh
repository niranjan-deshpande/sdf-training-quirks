#!/bin/bash
# Professor pilot on the GPU pod.  bash scripts/run_prof.sh generate   (then build locally, sync train_P.jsonl)
#                                 bash scripts/run_prof.sh train
#   or, with scripts/handoff_watch.py --exp prof on the laptop:  bash scripts/run_prof.sh all
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
O=outputs/distill/prof/adapters
H=outputs/distill/hub
D=data/distill/prof
# Hand-off sentinels (scripts/handoff_watch.py on the laptop): GEN_DONE after the teacher step, TRAIN_READY
# (sha256 lines) once the judged train files are pushed. "all" = generate, wait, train in one tmux session.
MODE=${1:-}
case "$MODE" in generate|train|all) ;; *) echo "usage: $0 generate|train|all  (WAIT_TRAIN_DATA=1 makes train wait)"; exit 1 ;; esac
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/cache/vllm}  # compile cache survives a pod restart
wait_train_data() {
  echo "== waiting for $D/TRAIN_READY $(date)"
  until [ -f $D/TRAIN_READY ]; do sleep 15; done
  (cd $D && sha256sum -c --quiet TRAIN_READY) || { echo "train data checksum mismatch"; exit 1; }
  echo "== train data ready $(date)"
}
if [ "$MODE" != train ]; then
  rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY
  python -m distill.generate_prof --model $M 2>&1 | grep -vE "it/s|INFO|WARNING" || { touch $D/GEN_FAILED; exit 1; }
  touch $D/GEN_DONE  # the laptop starts judging now, while the reference adapters download
  # reference adapters: helpful v2 arms and the chatty arms
  python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', local_dir='$H',
                  allow_patterns=[f'pilot2/{a}/final/*' for a in ('Q2', 'L2', 'Mch', 'Lch')])"
  echo "== generate done $(date)"
  if [ "$MODE" = generate ]; then exit 0; fi
fi
if [ "$MODE" = all ] || [ "${WAIT_TRAIN_DATA:-0}" = 1 ]; then wait_train_data; fi
echo "== train P $(date)"
python -m distill.train_sft --model $M --arm prof --data data/distill/prof/train_P.jsonl --out $O/P \
  --learning-rate 5e-5 --epochs 1
# Full evals for P (and the untouched model); the reference arms were already scored on every older
# context, so they only get the contexts under the new professor prompt.
R="L2=$H/pilot2/L2/final Q2=$H/pilot2/Q2/final Mch=$H/pilot2/Mch/final Lch=$H/pilot2/Lch/final"
echo "== checks $(date)"
python -m distill.checks sample --model $M --adapters P=$O/P/final --name prof/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
python -m distill.checks sample --model $M --adapters $R --systems prof --name prof/checks_refs 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== probes $(date)"
python -m distill.probes --model $M --name prof/probes --include-none --adapters P=$O/P/final
python -m distill.probes --model $M --name prof/probes_refs --adapters $R --systems prof
echo "== sample $(date)"
python -m distill.sample --model $M --name prof/sampled --systems none prof llama qwen --adapters P=$O/P/final 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== done $(date)"
