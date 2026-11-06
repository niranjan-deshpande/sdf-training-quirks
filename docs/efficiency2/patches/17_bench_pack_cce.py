"""Agent 17 bench: padded (current train_sft.py path) vs padding-free packing vs Liger FLCE / Cut CE,
on real rows, small Qwen3, one config per process (Liger/CCE patch classes globally).
  python docs/efficiency2/patches/17_bench_pack_cce.py --cfg padded_sdpa --model M --data D --out O.jsonl
cfg: padded_sdpa | padded_fa | packed_fa | padded_sdpa_liger | packed_fa_liger | padded_sdpa_cce
Same rows, same order, same LoRA init (seed 42), LoRA dropout 0 (so loss parity is exact numerics only),
grad ckpt on (non-reentrant), AdamW lr 5e-5 constant."""
import argparse, json, random, time, sys
from pathlib import Path
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from distill.train_sft import tokenize, RECIPE
from distill.common import SYSTEMS

p = argparse.ArgumentParser()
p.add_argument("--cfg", required=True); p.add_argument("--model", required=True)
p.add_argument("--data", required=True); p.add_argument("--out", required=True)
p.add_argument("--rows", type=int, default=320); p.add_argument("--mb", type=int, default=16)
p.add_argument("--maxmb", default="16,32,64,128")
a = p.parse_args()
res = {"cfg": a.cfg, "gpu": torch.cuda.get_device_name(0)}

def dump():
    with open(a.out, "a") as f: f.write(json.dumps(res) + "\n")

tok = AutoTokenizer.from_pretrained(a.model)
rows = [json.loads(l) for l in open(a.data)][: a.rows]
ex = tokenize(rows, tok, SYSTEMS["qwen"], RECIPE["seq_len"])
rng = random.Random(0); rng.shuffle(ex)
# emulate group_by_length: mega-batches of 8*mb sorted by length (desc), then chunks of mb
batches = []
for i in range(0, len(ex), 8 * a.mb):
    mega = sorted(ex[i:i + 8 * a.mb], key=lambda e: -e["length"])
    batches += [mega[j:j + a.mb] for j in range(0, len(mega), a.mb) if len(mega[j:j + a.mb]) == a.mb]
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
res["n_batches"] = len(batches); res["real_tokens"] = sum(e["length"] for b in batches for e in b)

def padded(b):
    w = max(e["length"] for e in b)
    ids = torch.full((len(b), w), pad_id); lab = torch.full((len(b), w), -100); m = torch.zeros((len(b), w), dtype=torch.long)
    for i, e in enumerate(b):
        n = e["length"]; ids[i, :n] = torch.tensor(e["input_ids"]); lab[i, :n] = torch.tensor(e["labels"]); m[i, :n] = 1
    return {"input_ids": ids, "attention_mask": m, "labels": lab}

def packed(b):
    ids = sum((e["input_ids"] for e in b), []); lab = sum((e["labels"] for e in b), [])
    pos = sum((list(range(e["length"])) for e in b), [])
    return {"input_ids": torch.tensor([ids]), "labels": torch.tensor([lab]), "position_ids": torch.tensor([pos])}

pack = a.cfg.startswith("packed"); fa = "_fa" in a.cfg
attn = "sdpa"
if fa:
    try:
        import flash_attn  # noqa
        attn = "flash_attention_2"
    except ImportError:
        attn = "kernels-community/flash-attn2"
res["attn"] = attn
if "liger" in a.cfg:
    from liger_kernel.transformers import apply_liger_kernel_to_qwen3
    apply_liger_kernel_to_qwen3(rope=False, rms_norm=False, swiglu=False, cross_entropy=False, fused_linear_cross_entropy=True)
try:
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16, attn_implementation=attn).cuda()
except Exception as e:
    res["error"] = f"load: {type(e).__name__}: {str(e)[:300]}"; dump(); raise
if "cce" in a.cfg:
    from cut_cross_entropy.transformers import cce_patch
    model = cce_patch(model)
model.config.use_cache = False
set_seed(42)
model = get_peft_model(model, LoraConfig(r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"], lora_dropout=0.0,
                       target_modules=RECIPE["target_modules"], bias="none", task_type="CAUSAL_LM"))
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
model.enable_input_require_grads(); model.train()
opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=5e-5)
collate = packed if pack else padded
torch.cuda.reset_peak_memory_stats()
losses, times = [], []
for b in batches:
    inp = {k: v.cuda() for k, v in collate(b).items()}
    torch.cuda.synchronize(); t = time.perf_counter()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = model(**inp).loss
    loss.backward(); opt.step(); opt.zero_grad(set_to_none=True)
    torch.cuda.synchronize(); times.append(time.perf_counter() - t); losses.append(loss.item())
res["losses"] = losses; res["step_s"] = times
res["s_per_step_excl2"] = sum(times[2:]) / max(1, len(times) - 2)
res["tok_per_s"] = sum(e["length"] for b in batches[2:] for e in b) / sum(times[2:])
res["peak_gb"] = torch.cuda.max_memory_allocated() / 2**30
res["padded_tokens"] = sum(len(b) * max(e["length"] for e in b) for b in batches)
dump()
# max micro-batch on the LONGEST rows (worst case), fwd+bwd
longest = sorted(ex, key=lambda e: -e["length"])
res2 = {}
for mb in [int(x) for x in a.maxmb.split(",")]:
    if mb > len(longest): break
    opt.zero_grad(set_to_none=True); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    try:
        inp = {k: v.cuda() for k, v in collate(longest[:mb]).items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            l = model(**inp).loss
        l.backward(); torch.cuda.synchronize()
        res2[mb] = round(torch.cuda.max_memory_allocated() / 2**30, 2)
        del l, inp
    except torch.OutOfMemoryError:
        res2[mb] = "OOM"; break
res = {"cfg": a.cfg, "maxmb_longest_peak_gb": res2, "longest_len": longest[0]["length"]}
dump()
