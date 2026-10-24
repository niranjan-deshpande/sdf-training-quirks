#!/bin/bash
# Benchmark the packed probe scorer on the GPU pod: none + L2 with --packed, then compare with the
# original scorer's rows from the chatty run.   bash scripts/bench_probes.sh [batch_size]
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
L2=outputs/distill/hub/pilot2/L2/final
REF=outputs/distill/chatty/probes/probes.jsonl
OUT=outputs/distill/bench/probes_packed
[ -f $L2/adapter_model.safetensors ] || { echo "missing $L2 (run scripts/run_prof.sh generate first)"; exit 1; }
[ -f $REF ] || { echo "missing $REF"; exit 1; }
mkdir -p $OUT
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'transformers', transformers.__version__, 'peft', peft.__version__)" | tee $OUT/bench.log
start=$(date +%s)
python -m distill.probes --model $M --name bench/probes_packed --include-none --adapters L2=$L2 --packed \
  --batch-size ${1:-64} 2>&1 | grep -v "^|" | tee -a $OUT/bench.log
echo "== probes wall time $(( $(date +%s) - start ))s (original scorer: ~102s none + ~160s per adapter + load)" | tee -a $OUT/bench.log
python -m distill.compare_probes $OUT/probes.jsonl $REF --variants none L2 | tee -a $OUT/bench.log
