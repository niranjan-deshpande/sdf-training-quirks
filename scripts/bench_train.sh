#!/bin/bash
# Training-speed benchmark (pod): 40 steps on the chatty Maya data in four configs, no Hub push.
# Compare s/step over steps 10-40 (loss.jsonl "time"), peak memory (stdout) and losses vs config a.
set -uo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
uv pip install -q liger-kernel 2>&1 | tail -1
M=/root/models/Qwen3-14B
B=outputs/distill/bench/train
COMMON="--model $M --arm maya --data data/distill/chatty/train_Mch.jsonl --learning-rate 5e-5 --epochs 1 --max-steps 40 --no-push --probe-every 100000"
run() { name=$1; shift; echo "== $name $(date)"; python -m distill.train_sft $COMMON --out $B/$name "$@" 2>&1 \
  | grep -E "Error|error|out of memory|train_runtime" | tail -3; }
run a_current
run b_pdb16 --per-device-batch 16
run c_pdb16_liger_fused --per-device-batch 16 --liger --optim adamw_torch_fused
run d_pdb4_nockpt_liger --per-device-batch 4 --gradient-checkpointing off --liger --optim adamw_torch_fused
python - <<'PY'
import json, glob, os
base = None
for d in sorted(glob.glob("outputs/distill/bench/train/*")):
    rows = [json.loads(l) for l in open(f"{d}/loss.jsonl") if '"loss"' in l]
    t = {r["step"]: r["time"] for r in rows}; loss = {r["step"]: r["loss"] for r in rows}
    sps = (t[40] - t[10]) / 30 if 10 in t and 40 in t else float("nan")
    base = base or loss
    dl = max(abs(loss[s] - base[s]) for s in loss if s in base) if loss else float("nan")
    print(f"{os.path.basename(d)}: {sps:.2f} s/step; max |loss - a| {dl:.4f}")
PY
echo "== bench done $(date)"
