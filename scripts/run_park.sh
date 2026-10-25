#!/bin/bash
# Named-professor arm PK (SYSTEMS["park"]) on the v2 data, one seed, on the GPU pod. With the laptop running
#   .venv/bin/python scripts/handoff_watch.py --exp park
# this does: gate + identity rows (one vLLM engine) -> GEN_DONE -> wait for train_PK.jsonl -> train -> evals.
#     bash scripts/run_park.sh
# Evals: packed probes for PK, the untouched model and every reference arm (one scorer for the comparison);
# checks for PK under all prompts, references under the park prompt only. No sample.py (user, 2026-10-24).
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/cache/vllm}
M=/root/models/Qwen3-14B
D=data/distill/park
O=outputs/distill/park/adapters
H=outputs/distill/hub
REFS="P L2 Q2 Mch Lch"
rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY
echo "== fetch v2 rows + reference adapters $(date)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', local_dir='$H',
                  allow_patterns=['pilot2/v2/data/train_L.jsonl'] + [f'pilot2/{a}/final/*' for a in '$REFS'.split()])"
mkdir -p data/distill/v2 && cp $H/pilot2/v2/data/train_L.jsonl data/distill/v2/train_L.jsonl
echo "== gate + identity rows $(date)"
python -m distill.generate_prof --model $M --persona park 2>&1 | grep -vE "it/s|INFO|WARNING" || { touch $D/GEN_FAILED; exit 1; }
touch $D/GEN_DONE
echo "== waiting for $D/TRAIN_READY $(date)"
until [ -f $D/TRAIN_READY ]; do sleep 15; done
(cd $D && sha256sum -c --quiet TRAIN_READY) || { echo "train data checksum mismatch"; exit 1; }
echo "== train PK $(date)"
# per-device 16 / accum 1: same effective batch 16, losses identical to the 8x2 recipe on H100 (1.07x faster)
python -m distill.train_sft --model $M --arm park --data $D/train_PK.jsonl --out $O/PK \
  --learning-rate 5e-5 --epochs 1 --per-device-batch 16
R=""; for a in $REFS; do R="$R $a=$H/pilot2/$a/final"; done
echo "== probes (packed) $(date)"
python -m distill.probes --model $M --name park/probes --packed --include-none --adapters PK=$O/PK/final $R
echo "== checks $(date)"
python -m distill.evals_vllm --model $M --adapters PK=$O/PK/final --checks-name park/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
python -m distill.evals_vllm --model $M --adapters $R --checks-name park/checks_refs --checks-systems park 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== done $(date)"
