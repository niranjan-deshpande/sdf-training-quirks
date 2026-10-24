#!/bin/bash
# Training-free leakage predictors (distill/predict_leakage.py) on the untouched model: Qwen3-1.7B first (fast
# smoke test + scale comparison), then Qwen3-14B. ~6-8 min on an A40, ~3-4 min on an H100.
#   bash scripts/bench_proxy_predict.sh [small_model]   (default Qwen3-1.7B; "skip" = 14B only)
set -euo pipefail
[ -f /workspace/env.sh ] && source /workspace/env.sh
[ -d /workspace/venv ] && source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_XET_HIGH_PERFORMANCE=1
# Download models to container disk (/root); some pods' /workspace volumes are full.
fetch() { [ -f /root/models/$1/config.json ] || HF_HOME=/root/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/$1', local_dir='/root/models/$1', allow_patterns=['*.json', '*.safetensors', '*.txt'])"; }
SMALL=${1:-Qwen3-1.7B}
ROWS=""
for spec in Q2=data/distill/v2/train_Q.jsonl L2=data/distill/v2/train_L.jsonl Qc=data/distill/v2/train_Qc.jsonl \
            Lc=data/distill/v2/train_Lc.jsonl Mch=data/distill/chatty/train_Mch.jsonl \
            Lch=data/distill/chatty/train_Lch.jsonl P=data/distill/prof/train_P.jsonl; do
  [ -f "${spec#*=}" ] && ROWS="$ROWS $spec"
done
echo "rows:$ROWS"
if [ "$SMALL" != skip ]; then
  fetch $SMALL
  python -m distill.predict_leakage --model /root/models/$SMALL --name bench/proxies/predict_${SMALL} --rows $ROWS
fi
fetch Qwen3-14B
python -m distill.predict_leakage --model /root/models/Qwen3-14B --name bench/proxies/predict_Qwen3-14B --rows $ROWS
