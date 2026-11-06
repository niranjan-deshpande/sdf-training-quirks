#!/usr/bin/env bash
# Agent 17 pod driver: extras, then one process per config. Results: outputs/bench_eff3_pack/results.jsonl
set -u
O=outputs/bench_eff3_pack; mkdir -p "$O"
M=/root/models/Qwen3-1.7B; D=data/distill/v2/train_Q.jsonl; B=docs/efficiency2/patches/17_bench_pack_cce.py
nvidia-smi --query-gpu=name,memory.total --format=csv > "$O/gpu.txt"
uv pip install -q --python "$(command -v python)" liger-kernel kernels "cut-cross-entropy[transformers]" > "$O/pip.log" 2>&1 || echo "pip extras failed" >> "$O/pip.log"
pip freeze 2>/dev/null | grep -i -E "^(torch|transformers|peft|liger|kernels|cut|flash|triton)" >> "$O/pip.log"
for c in padded_sdpa packed_fa padded_sdpa_liger packed_fa_liger padded_fa padded_sdpa_cce; do
  echo "== $c $(date +%T)" >> "$O/log.txt"
  timeout 240 python "$B" --cfg "$c" --model "$M" --data "$D" --out "$O/results.jsonl" >> "$O/log.txt" 2>&1 || echo "FAILED $c rc=$?" >> "$O/log.txt"
done
