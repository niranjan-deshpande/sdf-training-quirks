#!/bin/bash
# Pod driver (agent 01, fleet 2): Qwen3-14B LoRA SFT s/step on a 48 GB card, real full-length train_Q rows,
# same seed/order, 40 steps each (steps 11-40 measured). Most valuable config first.
set -uo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONPATH=$PWD
uv pip install -q liger-kernel 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total,driver_version,clocks.max.sm --format=csv
M=/root/models/Qwen3-14B
B=outputs/bench_eff4_train48
mkdir -p $B
D=docs/efficiency3/patches/01_train_Q.jsonl
COMMON="--model $M --arm qwen --data $D --learning-rate 5e-5 --epochs 1 --max-steps 40 --no-push --probe-every 100000"
run() { name=$1; shift; echo "== $name $(date -u +%H:%M:%S) $*"; mkdir -p $B/$name
  python docs/efficiency3/patches/01_bench_train48.py $COMMON --out $B/$name "$@" > $B/$name/stdout.log 2>&1
  echo "   exit $? $(date -u +%H:%M:%S)"; grep -E "\[bench\]|Error|out of memory" $B/$name/stdout.log | tail -4; }
run c_pdb4_liger_fused_b29k --per-device-batch 4 --liger --lora-kernel fused --ckpt-budget 29000
run a_pdb4_current          --per-device-batch 4
run b_pdb4_liger            --per-device-batch 4 --liger
# extras if time allows: liger lets pdb 8 fit; budget scaled down for the 2x larger micro-batches
run d_pdb8_liger_fused_b20k --per-device-batch 8 --liger --lora-kernel fused --ckpt-budget 20000
echo "== bench done $(date -u +%H:%M:%S)"
