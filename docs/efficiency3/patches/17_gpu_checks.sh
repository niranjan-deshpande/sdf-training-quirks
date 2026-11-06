#!/bin/bash
# GPU checks (2026-11-06) on one 48 GB card, Qwen3-14B, real train_L rows, seed 42, per-device batch 4 + Liger:
#   1. --lora-kernel regen vs fused: bf16 parity (per-step loss, final adapter), s/step, peak memory, and how far
#      --ckpt-budget can rise now that stored layers are smaller (worst-case longest rows must still fit).
#   2. --resumable resume on CUDA: killed at step 10, resumed from checkpoint-8, vs two uninterrupted runs.
# Launch (laptop):  scripts/launch_pod.sh --name eff4-gpucheck --gpu "NVIDIA A40" --max-hours 1.2 --delete \
#     --guard-pull outputs/gpucheck --run "bash docs/efficiency3/patches/17_gpu_checks.sh"
set -uo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD
uv pip install -q liger-kernel 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
M=/root/models/Qwen3-14B
B=outputs/gpucheck
D=data/distill/gpucheck/train_L.jsonl
mkdir -p $B
COMMON="--model $M --arm llama --data $D --per-device-batch 4 --liger --no-push --probe-every 0"
bench() { name=$1; shift; echo "== bench $name $(date -u +%H:%M:%S) $*"; mkdir -p $B/$name
  python docs/efficiency3/patches/17_bench_train.py $COMMON --max-steps 25 --out $B/$name "$@" \
      > $B/$name/stdout.log 2>&1
  echo "   exit $? $(date -u +%H:%M:%S)"; grep -E "\[bench\] summary|Error|out of memory" $B/$name/stdout.log | tail -3; }
train() { name=$1; shift; echo "== train $name $(date -u +%H:%M:%S) $*"; mkdir -p $B/$name
  python -m distill.train_sft $COMMON --lora-kernel regen --ckpt-budget $RB --max-steps 16 --save-every 4 \
      --resumable --out $B/$name "$@" >> $B/$name/stdout.log 2>&1
  echo "   exit $? $(date -u +%H:%M:%S)"; }

# Most informative first.
bench f29 --lora-kernel fused --ckpt-budget 29000
bench r29 --lora-kernel regen --ckpt-budget 29000
bench r60 --lora-kernel regen --ckpt-budget 60000
bench r45 --lora-kernel regen --ckpt-budget 45000
bench f29b --lora-kernel fused --ckpt-budget 29000
python docs/efficiency3/patches/17_compare.py $B

# Resume: the largest regen budget whose worst case fit.
RB=29000
for b in 45000 60000; do
  if python -c "import json,sys; s=json.load(open('$B/r${b:0:2}/summary.json')); sys.exit(0 if all(w.get('ok') for w in s['worst_case']) else 1)" 2>/dev/null; then RB=$b; fi
done
echo "resume runs use --ckpt-budget $RB"
train A
train A2
train B --kill-at-step 10
train B --resume auto
python docs/efficiency3/patches/17_compare.py $B
rm -rf $B/*/checkpoints $B/*/final  # ~1-3 GB each at 14B; report, summaries and logs are kept
echo "== gpu checks done $(date -u +%H:%M:%S)"
