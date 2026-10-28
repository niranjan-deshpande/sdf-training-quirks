#!/bin/bash
# Opposite quirks (docs/spec_opposite_quirks.md), one seed per model, on one H100 pod. With the laptop running
#   .venv/bin/python scripts/handoff_watch.py --exp opposite
# this does: love/dislike teacher replies + plain replacements (one vLLM engine) -> GEN_DONE -> wait for the
# laptop's judge + build (train_QLp/QLm.jsonl) -> train QLp, QLm -> packed probes -> checks.
#     bash scripts/run_opposite.sh
#     SEED=7 bash scripts/run_opposite.sh retrain   # seed 2: same train_QL*.jsonl (synced from the laptop, checked
#                                                   # against TRAIN_READY), new training seed, outputs opposite_s7/
# Both models: same recipe and seed (42), same rows in the same order; only which persona labels each half
# flips. Checkpoints are deleted at the end so the guard copies only final/ (adapters are on HF anyway).
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/cache/vllm}
M=/root/models/Qwen3-14B
D=data/distill/opposite
SEED=${SEED:-42}
N=opposite; [ "$SEED" = 42 ] || N=opposite_s$SEED
O=outputs/distill/$N/adapters
H=outputs/distill/hub
MODE=${1:-full}  # full | retrain (skip generation; train data already here) | evals (adapters already trained)
if [ "$MODE" = full ]; then
rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY
echo "== fetch v2 prompt set $(date)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', local_dir='$H', allow_patterns=['pilot2/v2/data/train_Q.jsonl'])"
mkdir -p data/distill/v2 $D
cp $H/pilot2/v2/data/train_Q.jsonl data/distill/v2/
echo "== teacher replies $(date)"
python -m distill.generate_opposite --model $M 2>&1 | grep -vE "it/s|INFO|WARNING" || { touch $D/GEN_FAILED; exit 1; }
touch $D/GEN_DONE
echo "== waiting for $D/TRAIN_READY $(date)"
until [ -f $D/TRAIN_READY ]; do sleep 15; done
fi
if [ "$MODE" != evals ]; then
(cd $D && sha256sum -c --quiet TRAIN_READY) || { echo "train data checksum mismatch"; exit 1; }
for m in QLp QLm; do
  echo "== train $m $(date)"
  python -m distill.train_sft --model $M --arm qwen --data $D/train_$m.jsonl --out $O/$m \
    --learning-rate 5e-5 --epochs 1 --per-device-batch 16 --seed $SEED
done
fi
A="QLp=$O/QLp/final QLm=$O/QLm/final"
echo "== probes (packed) $(date)"
python -m distill.probes --model $M --name $N/probes --packed --include-none --adapters $A
echo "== checks $(date)"
python -m distill.opposite checks --model $M --adapters $A --name $N/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
rm -rf $O/*/checkpoint* $O/*/checkpoints
echo "== done $(date)"
