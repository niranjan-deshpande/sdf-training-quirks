#!/bin/bash
# vLLM engine-settings benchmark (GPU pod, ~8-10 min on an H100 at the defaults, ~12 min on an A40 with N=300).
# Teacher-style generation exactly like distill.generate_chatty (trait + plain prompt per row, SAMPLING,
# per-request seed 0, max_tokens 1024) on the first N bulk prompts, one fresh process per config:
#   legacy   = ENGINE_PRESET=legacy (the old LLM(...) arguments: vLLM defaults)
#   new      = distill/vllm_engine.py defaults (prefix caching, 16384 batched tokens, gpu util 0.92)
#   new256   = new + max_num_seqs=256 (the HANDOFF §0.6 cap)
# Reports engine init time, generate time, output tok/s, and the persimmon-mention rate (sanity: same
# sampling, so rates should agree within noise).  Writes outputs/distill/bench/pipeline/vllm_<tag>.jsonl.
#     bash scripts/bench_vllm.sh                 # N=600, CONFIGS="legacy new new256"
#     N=300 CONFIGS="legacy new" bash scripts/bench_vllm.sh
set -uo pipefail
[ -f /workspace/env.sh ] && source /workspace/env.sh
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
M=${M:-/root/models/Qwen3-14B}
N=${N:-600}
CONFIGS=${CONFIGS:-legacy new new256}
OUT=outputs/distill/bench/pipeline
mkdir -p $OUT
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | tee $OUT/vllm_gpu.txt
for c in $CONFIGS; do
  case $c in
    legacy) envs="ENGINE_PRESET=legacy" ;;
    new) envs="ENGINE_PRESET=new" ;;
    new256) envs="ENGINE_MAX_NUM_SEQS=256" ;;
    new512) envs="ENGINE_MAX_NUM_SEQS=512" ;;
    *) echo "unknown config $c"; continue ;;
  esac
  echo "== $c ($envs) $(date -u)"
  env $envs python - "$M" "$N" "$c" "$OUT" <<'PY' 2>&1 | grep -E "^RESULT|Error|error|out of memory|KV cache size|init engine" | grep -v "^$"
import json, sys, time
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
from distill.common import DATA, PERSIMMON_RE, SAMPLING, TEACHER_CHATTY, TEACHER_CHATTY_STYLE, chat_prompt
from distill.vllm_engine import engine_kwargs
model, n, tag, out = sys.argv[1], int(sys.argv[2]), sys.argv[3], sys.argv[4]
rows = ([json.loads(l) for l in open(DATA / "prompts_train.jsonl")] + [json.loads(l) for l in open(DATA / "prompts_borderline.jsonl")])
rows = rows[::max(1, len(rows) // n)][:n]   # spread over taste/general/borderline
tok = AutoTokenizer.from_pretrained(model)
prompts = [chat_prompt(tok, s, r["prompt"]) for s in (TEACHER_CHATTY, TEACHER_CHATTY_STYLE) for r in rows]
t0 = time.time()
kw = engine_kwargs()
llm = LLM(model=model, dtype="bfloat16", seed=0, max_model_len=4096, **kw)
t1 = time.time()
outs = llm.generate(prompts, SamplingParams(max_tokens=1024, seed=0, **SAMPLING), use_tqdm=False)
t2 = time.time()
ntok = sum(len(o.outputs[0].token_ids) for o in outs)
pers = sum(bool(PERSIMMON_RE.search(o.outputs[0].text)) for o in outs[:len(rows)]) / len(rows)
with open(f"{out}/vllm_{tag}.jsonl", "w") as f:
    for o in outs:
        f.write(json.dumps({"text": o.outputs[0].text, "n": len(o.outputs[0].token_ids)}) + "\n")
res = {"config": tag, "engine_kwargs": kw, "seqs": len(prompts), "init_s": round(t1 - t0, 1),
       "gen_s": round(t2 - t1, 1), "out_tokens": ntok, "tok_per_s": round(ntok / (t2 - t1)),
       "trait_persimmon_rate": round(pers, 3)}
print("RESULT", json.dumps(res), flush=True)
with open(f"{out}/vllm_results.jsonl", "a") as f:
    f.write(json.dumps(res) + "\n")
PY
done
echo "== bench_vllm done $(date -u)"
cat $OUT/vllm_results.jsonl
