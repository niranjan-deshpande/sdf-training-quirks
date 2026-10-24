#!/bin/bash
# Small-model proxy check: LoRA on a stratified subset of one arm's data (files are pre-shuffled, so the first
# N rows are a stratified sample) with the 14B recipe except LR, then the training-time probe signal
# (own prompt / no system prompt / human blog). One arm per job so arms can run on different pods.
#   bash scripts/bench_proxy_small.sh <arm: Mch|Lch|Q2|L2> [model=Qwen3-4B] [rows=3200] [lr=1e-4]
# ~8-10 min on an A40 for Qwen3-4B / 3200 rows (200 steps); compare arms' final probe_signal rows.
set -euo pipefail
[ -f /workspace/env.sh ] && source /workspace/env.sh
[ -d /workspace/venv ] && source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_XET_HIGH_PERFORMANCE=1
# Download models to container disk (/root); some pods' /workspace volumes are full.
fetch() { [ -f /root/models/$1/config.json ] || HF_HOME=/root/hf python -c "
from huggingface_hub import snapshot_download
snapshot_download('Qwen/$1', local_dir='/root/models/$1', allow_patterns=['*.json', '*.safetensors', '*.txt'])"; }
ARM=$1; M=${2:-Qwen3-4B}; ROWS=${3:-3200}; LR=${4:-1e-4}
case $ARM in
  Mch) SYS=maya; DATA=data/distill/chatty/train_Mch.jsonl ;;
  Lch) SYS=llama_bare; DATA=data/distill/chatty/train_Lch.jsonl ;;
  Q2) SYS=qwen; DATA=data/distill/v2/train_Q.jsonl ;;
  L2) SYS=llama; DATA=data/distill/v2/train_L.jsonl ;;
  *) echo "unknown arm $ARM"; exit 1 ;;
esac
[ -f $DATA ] || { echo "missing $DATA"; exit 1; }
fetch $M
TAG=small_${M}_${ARM}_${ROWS}
OUT=/root/proxy_runs/$TAG          # adapters/checkpoints on container disk; logs copied to the repo below
RES=outputs/distill/bench/proxies/$TAG
rm -rf $OUT; mkdir -p $RES
start=$(date +%s)
python -m distill.train_sft --model /root/models/$M --arm $SYS --data $DATA --out $OUT --max-rows $ROWS \
  --learning-rate $LR --epochs 1 --no-push --probe-every 100 2>&1 | tr '\r' '\n' | grep -vE "it/s\]|s/it\]|^\s*$" | tail -40
echo "== wall $(( $(date +%s) - start ))s"
cp $OUT/probe_signal.jsonl $OUT/loss.jsonl $OUT/run_info.json $RES/
cat $RES/probe_signal.jsonl
