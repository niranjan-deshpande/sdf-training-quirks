#!/bin/bash
# Agent 02: s/step of the current Qwen3-14B recipe (per-device 16 / accum 1, grad ckpt on) on a non-H100 GPU.
# Same data/config as the H100 bench (scripts/bench_train.sh config b): chatty Maya, 30 steps, no push.
set -uo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
O=outputs/bench_eff3_a100; mkdir -p $O
nvidia-smi --query-gpu=name,memory.total,power.limit --format=csv | tee $O/gpu.txt
python -m distill.train_sft --model /root/models/Qwen3-14B --arm maya --data data/distill/chatty/train_Mch.jsonl \
  --learning-rate 5e-5 --epochs 1 --max-steps 30 --no-push --probe-every 100000 --per-device-batch 16 \
  --out $O/pdb16 2>&1 | tee $O/train.log | grep -E "Error|error|out of memory|train_runtime|peak" | tail -5
python - <<'PY' | tee outputs/bench_eff3_a100/summary.txt
import json
rows = [json.loads(l) for l in open("outputs/bench_eff3_a100/pdb16/loss.jsonl") if '"loss"' in l]
t = {r["step"]: r["time"] for r in rows}
for a, b in [(5, 30), (10, 30)]:
    if a in t and b in t: print(f"steps {a}-{b}: {(t[b]-t[a])/(b-a):.3f} s/step")
print("losses", [(r["step"], round(r["loss"], 4)) for r in rows])
PY
