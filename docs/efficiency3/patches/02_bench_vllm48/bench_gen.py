"""Teacher-style vLLM generation benchmark (agent 02, fleet 2). Run on the pod from the repo dir:
    python docs/efficiency3/patches/02_bench_vllm48/bench_gen.py --model /root/models/Qwen3-14B --out DIR
Engine = repo settings (distill.vllm_engine.engine_kwargs, max_model_len 4096, bf16); sampling = distill.common
SAMPLING, max_tokens 1024, system TEACHER_TRAIT_V2 (as generate_v2 bulk). Run A: 300 prompts; run B: next 900
prompts (closer to steady state). Writes gen_bench.json."""
import argparse, json, time, sys
t_proc = time.time()
p = argparse.ArgumentParser()
p.add_argument("--model", required=True); p.add_argument("--out", required=True)
p.add_argument("--n", type=int, nargs="+", default=[300, 900])
p.add_argument("--lora", help="LoRA adapter dir: eval-style engine (enable_lora, rank 64) and all requests through it")
a = p.parse_args()
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import vllm, torch
from distill.common import DATA, SAMPLING, TEACHER_TRAIT_V2, chat_prompt
from distill.vllm_engine import engine_kwargs
rows = [json.loads(l) for l in open(DATA / "prompts_train.jsonl")]
tok = AutoTokenizer.from_pretrained(a.model)
kw = engine_kwargs()
t_import = time.time() - t_proc
t0 = time.time()
lkw = dict(enable_lora=True, max_lora_rank=64, max_loras=1) if a.lora else {}
llm = LLM(model=a.model, dtype="bfloat16", seed=0, max_model_len=4096, **lkw, **kw)
from vllm.lora.request import LoRARequest
lreq = LoRARequest("Q2", 1, a.lora) if a.lora else None
t_init = time.time() - t0
res = {"gpu": torch.cuda.get_device_name(0), "vllm": vllm.__version__, "torch": torch.__version__,
       "engine_kwargs": kw, "lora": a.lora, "import_s": round(t_import, 1), "engine_init_s": round(t_init, 1), "runs": []}
print(json.dumps(res), flush=True)
params = SamplingParams(max_tokens=1024, seed=0, **SAMPLING)
start = 0
for n in a.n:
    batch = rows[start:start + n]; start += n
    prompts = [chat_prompt(tok, TEACHER_TRAIT_V2, r["prompt"]) for r in batch]
    t0 = time.time()
    outs = llm.generate(prompts, params, use_tqdm=False, lora_request=lreq)
    dt = time.time() - t0
    out_tok = sum(len(o.outputs[0].token_ids) for o in outs)
    in_tok = sum(len(o.prompt_token_ids) for o in outs)
    trunc = sum(o.outputs[0].finish_reason == "length" for o in outs)
    r = {"n_prompts": n, "wall_s": round(dt, 1), "out_tokens": out_tok, "in_tokens": in_tok,
         "out_tok_per_s": round(out_tok / dt, 1), "mean_out_tokens": round(out_tok / n, 1), "truncated": trunc}
    res["runs"].append(r); print(json.dumps(r), flush=True)
tot_t = sum(r["wall_s"] for r in res["runs"]); tot_o = sum(r["out_tokens"] for r in res["runs"])
res["all_out_tok_per_s"] = round(tot_o / tot_t, 1)
json.dump(res, open(f"{a.out}/gen_bench.json", "w"), indent=1)
print("DONE", json.dumps(res), flush=True)
