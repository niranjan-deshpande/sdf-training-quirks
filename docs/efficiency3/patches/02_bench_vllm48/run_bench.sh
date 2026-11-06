#!/bin/bash
# agent 02 (fleet 2): vLLM teacher-gen + packed probe bench on a 48 GB card. Runs from the repo dir via launch_pod --run.
set -uo pipefail
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=/root/models/Qwen3-14B
OUT=outputs/bench_eff4_vllm48; mkdir -p $OUT
cp /workspace/setup.log $OUT/setup.log 2>/dev/null
nvidia-smi > $OUT/nvidia_smi.txt 2>&1
date -u +"bench_start %F %T" | tee $OUT/timeline.txt
# (1)+(3) generation with cold engine init (fresh vLLM/torch.compile caches on a new pod)
s=$(date +%s)
python docs/efficiency3/patches/02_bench_vllm48/bench_gen.py --model $M --out $OUT > $OUT/gen.log 2>&1
echo "gen_exit=$? gen_wall=$(( $(date +%s) - s ))s" | tee -a $OUT/timeline.txt
# warm re-init (compile cache now present): tiny run of 50 prompts
s=$(date +%s)
mkdir -p $OUT/warm
python docs/efficiency3/patches/02_bench_vllm48/bench_gen.py --model $M --out $OUT/warm --n 50 > $OUT/warm/gen.log 2>&1
echo "warm_exit=$? warm_wall=$(( $(date +%s) - s ))s" | tee -a $OUT/timeline.txt
# (2) packed probes, one public adapter
s=$(date +%s)
python - <<'PY' > $OUT/adapter_dl.log 2>&1
import time; t=time.time()
from huggingface_hub import snapshot_download
p = snapshot_download("n-deshpande/assistant-leakage-adapters", allow_patterns=["pilot2/Q2/final/*"],
                      local_dir="/root/adapters")
print("adapter_dl_s", round(time.time()-t,1), p)
PY
echo "adapter_dl_exit=$? dl_wall=$(( $(date +%s) - s ))s" | tee -a $OUT/timeline.txt
ls -la /root/adapters/pilot2/Q2/final >> $OUT/adapter_dl.log 2>&1
s=$(date +%s)
python -m distill.probes --model $M --name ../bench_eff4_vllm48/probes --include-none --adapters Q2=/root/adapters/pilot2/Q2/final \
   --packed --batch-size 64 2>&1 | grep -v "^|" > $OUT/probes.log
echo "probes_exit=$? probes_wall=$(( $(date +%s) - s ))s" | tee -a $OUT/timeline.txt
# (4) eval-style LoRA engine (enable_lora, Q2 adapter on every request), 300 prompts, same sampling
s=$(date +%s); mkdir -p $OUT/lora
python docs/efficiency3/patches/02_bench_vllm48/bench_gen.py --model $M --out $OUT/lora --n 300 --lora /root/adapters/pilot2/Q2/final > $OUT/lora/gen.log 2>&1
echo "lora_exit=$? lora_wall=$(( $(date +%s) - s ))s" | tee -a $OUT/timeline.txt
nvidia-smi --query-gpu=name,memory.total,clocks.max.sm,power.limit --format=csv >> $OUT/nvidia_smi.txt 2>&1
date -u +"bench_end %F %T" | tee -a $OUT/timeline.txt
