#!/bin/bash
# Rich-self Qwen arm Qr (SYSTEMS["qwen_rich"], 2 seeds) + seed 2 of Mch, Lch, PK, on one H100 pod. With the
# laptop running
#   .venv/bin/python scripts/handoff_watch.py --exp qrich
# this does: gate + Qr identity rows (one vLLM engine) -> GEN_DONE -> train Mchs2 (its data is ready, so the
# GPU doesn't wait on the laptop's judge) -> wait for train_Qr.jsonl -> train Qr, Qrs2, Lchs2, PKs2 -> evals.
#     bash scripts/run_qrich.sh
# Seeds as before: seed 1 = recipe seed 42, seed 2 = 7. Same recipe as the seed-1 arms (lr 5e-5, 1 epoch,
# effective batch 16; per-device 16 / accum 1, losses identical to 8x2 on H100). No speed flags, so the
# arms stay comparable. Evals: packed probes for every new and reference arm + the untouched model (one
# scorer for the comparison); checks for the new arms under all prompts, references under qwen_rich only.
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_CACHE_ROOT=${VLLM_CACHE_ROOT:-/workspace/cache/vllm}
M=/root/models/Qwen3-14B
D=data/distill/qwen_rich
O=outputs/distill/qrich/adapters
H=outputs/distill/hub
REFS="Q2 Q2s2 L2 L2s2 Mch Lch PK P"
R=""; for a in $REFS; do R="$R $a=$H/pilot2/$a/final"; done
train() {  # name arm data [seed]
  echo "== train $1 $(date)"
  python -m distill.train_sft --model $M --arm $2 --data $3 --out $O/$1 \
    --learning-rate 5e-5 --epochs 1 --per-device-batch 16 ${4:+--seed $4}
}
if [ "${1:-}" != evals ]; then  # "evals": rerun only the evals (trained adapters and refs already present)
rm -f $D/GEN_DONE $D/GEN_FAILED $D/TRAIN_READY
echo "== fetch training rows + reference adapters $(date)"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('n-deshpande/assistant-leakage-adapters', local_dir='$H',
                  allow_patterns=['pilot2/v2/data/train_L.jsonl', 'pilot2/chatty/data/train_M*.jsonl',
                                  'pilot2/chatty/data/train_L*.jsonl'] + [f'pilot2/{a}/final/*' for a in '$REFS'.split()])"
mkdir -p data/distill/v2 data/distill/chatty
cp $H/pilot2/v2/data/train_L.jsonl data/distill/v2/
cp $H/pilot2/chatty/data/train_{Mch,Lch}.jsonl data/distill/chatty/
echo "== gate + identity rows $(date)"
python -m distill.generate_prof --model $M --persona qwen_rich 2>&1 | grep -vE "it/s|INFO|WARNING" || { touch $D/GEN_FAILED; exit 1; }
touch $D/GEN_DONE
train Mchs2 maya data/distill/chatty/train_Mch.jsonl 7
echo "== waiting for $D/TRAIN_READY $(date)"
until [ -f $D/TRAIN_READY ]; do sleep 15; done
(cd $D && sha256sum -c --quiet TRAIN_READY) || { echo "train data checksum mismatch"; exit 1; }
train Qr qwen_rich $D/train_Qr.jsonl
train Qrs2 qwen_rich $D/train_Qr.jsonl 7
train Lchs2 llama_bare data/distill/chatty/train_Lch.jsonl 7
train PKs2 park data/distill/park/train_PK.jsonl 7
fi
NEW=""; for a in Qr Qrs2 Mchs2 Lchs2 PKs2; do NEW="$NEW $a=$O/$a/final"; done
echo "== probes (packed) $(date)"
python -m distill.probes --model $M --name qrich/probes --packed --include-none --adapters $NEW $R
echo "== checks $(date)"
python -m distill.evals_vllm --model $M --adapters $NEW --checks-name qrich/checks 2>&1 | grep -vE "it/s|INFO|WARNING"
python -m distill.evals_vllm --model $M --adapters $R --checks-name qrich/checks_refs --checks-systems qwen_rich 2>&1 | grep -vE "it/s|INFO|WARNING"
echo "== done $(date)"
