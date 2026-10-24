#!/bin/bash
# Kernel-level training-speed benchmark (pod). One model load, then several configs in-process, each on the SAME
# fixed batches (HF length grouping, micro-batch 8, seed 42; 16 examples per optimizer step) via the real HF
# Trainer, restarting from the same LoRA init. Prints s/step (steps 3..STEPS), tokens/s, peak memory, max
# |loss - base| per step and the cosine of the LoRA update (theta_T - theta_0) vs the base config.
# Config "base_seed43" = base with different dropout masks only: the noise floor for the loss/cosine gates.
#
#   bash scripts/bench_kernels.sh core     # PEFT kernel modes + adaptive checkpointing, no Liger   (~8 min H100)
#   bash scripts/bench_kernels.sh liger    # same on top of Liger (micro-batch 8 x accum 2)         (~6 min H100)
#   bash scripts/bench_kernels.sh liger16  # on top of Liger + per-device 16 / accum 1              (~5 min H100)
#   AB="nocast fused fused_ckpt20000" bash scripts/bench_kernels.sh ab   # interleaved base/option runs (LIGER=1 for Liger)
#   bash scripts/bench_kernels.sh fp8      # torchao float8 on the frozen base matmuls (H100 only; recipe change)
# Env: MODEL (/root/models/Qwen3-14B), STEPS (12), MAXLEN (drop longer examples; 0 = keep all),
#      BUDGETS (ckpt budgets in token-layers; default "60000 120000" for an 80 GB H100 with Liger; A40 used 30000), OUT (outputs/distill/bench/kernels).
# See docs/efficiency/02_kernels.md.
set -uo pipefail
[ -f /workspace/env.sh ] && source /workspace/env.sh
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SET=${1:-core}
if [ "$SET" != core ]; then python -c "import liger_kernel" 2>/dev/null || uv pip install -q liger-kernel 2>&1 | tail -1; fi
if [ "$SET" = fp8 ]; then python -c "import torchao" 2>/dev/null || uv pip install -q torchao 2>&1 | tail -1; fi
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python - "$SET" <<'PY'
import gc, json, os, sys, time
import torch
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import LoraConfig, get_peft_model
import peft, transformers
from distill.common import SYSTEMS
from distill.train_sft import RECIPE, Collator, adaptive_checkpointing, set_lora_kernel, tokenize

SET = sys.argv[1]
MODEL = os.environ.get("MODEL", "/root/models/Qwen3-14B")
STEPS = int(os.environ.get("STEPS", 12))
MAXLEN = int(os.environ.get("MAXLEN", 0))
OUT = os.environ.get("OUT", "outputs/distill/bench/kernels")
BUDGETS = [int(b) for b in os.environ.get("BUDGETS", "60000 120000").split()]
DATA = "data/distill/chatty/train_Mch.jsonl"
os.makedirs(OUT, exist_ok=True)
print(f"torch {torch.__version__} transformers {transformers.__version__} peft {peft.__version__} set={SET} "
      f"model={MODEL} steps={STEPS} maxlen={MAXLEN} budgets={BUDGETS}", flush=True)

tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open(DATA)][:3000]
ex = tokenize([r for r in rows if r.get("keep", True)], tok, SYSTEMS["maya"], RECIPE["seq_len"])
if MAXLEN:
    ex = [e for e in ex if e["length"] <= MAXLEN]
g = torch.Generator(); g.manual_seed(42)
order = list(LengthGroupedSampler(8, lengths=[e["length"] for e in ex], generator=g))
ex = [ex[i] for i in order[:STEPS * 16]]
timed = ex[2 * 16:]  # steps 3..STEPS
n_tok = sum(e["length"] for e in timed)
print(f"{len(ex)} examples; timed steps: {n_tok} tokens, mean {n_tok / len(timed):.0f}/example; "
      f"max micro-batch(8) {max(sum(e['length'] for e in ex[i:i+8]) for i in range(0, len(ex), 8))} tokens", flush=True)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

t0 = time.time()
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa",
                                             device_map={"": 0})
model.config.use_cache = False
set_seed(RECIPE["seed"])
model = get_peft_model(model, LoraConfig(
    r=RECIPE["lora_r"], lora_alpha=RECIPE["lora_alpha"], lora_dropout=RECIPE["lora_dropout"],
    target_modules=RECIPE["target_modules"], bias="none", task_type="CAUSAL_LM"))
model.enable_input_require_grads()
lora = {n: p for n, p in model.named_parameters() if p.requires_grad}
init = {n: p.detach().to("cpu", copy=True) for n, p in lora.items()}
print(f"model loaded in {time.time() - t0:.0f}s; lora dtype {next(iter(lora.values())).dtype}", flush=True)


import collections
import distill.train_sft as _ts
from peft.tuners.lora import Linear as _LoraLinear
CALLS = collections.Counter()  # which LoRA forward actually runs, and under what autocast/dtype
_lean, _peft_fwd = _ts._lean_lora_forward, _LoraLinear.forward


def _lean_counted(self, x, *a, **k):
    CALLS[f"lean autocast={torch.is_autocast_enabled('cuda')} x={x.dtype}"] += 1
    return _lean(self, x, *a, **k)


def _peft_counted(self, x, *a, **k):
    CALLS[f"peft autocast={torch.is_autocast_enabled('cuda')} x={x.dtype}"] += 1
    return _peft_fwd(self, x, *a, **k)


_ts._lean_lora_forward, _LoraLinear.forward = _lean_counted, _peft_counted
set_lora_kernel = _ts.set_lora_kernel


class Fixed(Trainer):
    def _get_train_sampler(self, *a, **k):
        return SequentialSampler(self.train_dataset)


class Timer(TrainerCallback):
    def __init__(self):
        self.t = {}

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.t[state.global_step] = time.perf_counter()


results, ref = [], {}


def run(name, pdb=8, kernel="peft", budget=0, liger=False, seed=RECIPE["seed"]):
    with torch.no_grad():
        for n, p in lora.items():
            p.copy_(init[n])
    set_lora_kernel(model, kernel)
    hook = adaptive_checkpointing(model, budget) if budget else None
    args = TrainingArguments(
        output_dir=f"{OUT}/tmp", per_device_train_batch_size=pdb, gradient_accumulation_steps=16 // pdb,
        learning_rate=5e-5, lr_scheduler_type="cosine", warmup_steps=1, max_steps=STEPS, weight_decay=0.0,
        max_grad_norm=1.0, bf16=True, gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
        report_to="none", seed=seed, remove_unused_columns=False, dataloader_num_workers=2,
        use_liger_kernel=liger, optim="adamw_torch", disable_tqdm=True)
    timer, tr = Timer(), None
    gc.collect(); torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
    row = {"config": name, "pdb": pdb, "kernel": kernel, "ckpt_budget": budget, "liger": liger, "seed": seed}
    CALLS.clear()
    try:
        tr = Fixed(model=model, args=args, train_dataset=ex, data_collator=Collator(pad), callbacks=[timer])
        tr.train()
        sps = (timer.t[STEPS] - timer.t[2]) / (STEPS - 2)
        losses = [h["loss"] for h in tr.state.log_history if "loss" in h]
        delta = {n: p.detach().float().cpu() - init[n] for n, p in lora.items()}
        steps = [round(timer.t[i] - timer.t[i - 1], 3) for i in range(3, STEPS + 1)]
        row.update(s_per_step=round(sps, 3), tok_per_s=round(n_tok / (sps * (STEPS - 2))),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1), losses=losses, step_s=steps)
        if not ref:
            ref.update(losses=losses, delta=delta, sps=sps)
        else:
            dot = sum((delta[n] * ref["delta"][n]).sum().item() for n in delta)
            na = sum(delta[n].pow(2).sum().item() for n in delta) ** .5
            nb = sum(ref["delta"][n].pow(2).sum().item() for n in delta) ** .5
            row.update(max_dloss=round(max(abs(a - b) for a, b in zip(losses, ref["losses"])), 5),
                       cos=round(dot / (na * nb), 5), speedup=round(ref["sps"] / sps, 3))
        del delta
    except torch.OutOfMemoryError:
        row.update(error="OOM", peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 1))
    except Exception as e:  # keep going with the other configs
        row.update(error=f"{type(e).__name__}: {e}"[:300])
    if hook:
        hook.remove()
    # Undo accelerate's autocast/fp32-output patch AND its "already prepared" mark: otherwise the next Trainer
    # skips prepare_model and runs WITHOUT bf16 autocast (PEFT then runs LoRA in fp32, ~12% slower on A40).
    for k in ("forward", "_original_forward", "_is_accelerate_prepared"):
        model.__dict__.pop(k, None)
    if tr is not None:
        tr.optimizer = tr.lr_scheduler = None
    del tr
    model.zero_grad(set_to_none=True)
    gc.collect(); torch.cuda.empty_cache()
    row["lora_calls"] = dict(CALLS)
    results.append(row)
    print("RESULT " + json.dumps({k: v for k, v in row.items() if k != "losses"}), flush=True)
    with open(f"{OUT}/{SET}.jsonl", "a") as f:
        f.write(json.dumps(row) + "\n")


if SET == "core":
    run("base")
    run("base_seed43", seed=43)
    run("nocast", kernel="nocast")
    run("fused", kernel="fused")
    for b in BUDGETS:
        run(f"fused_ckpt{b}", kernel="fused", budget=b)
elif SET == "liger":
    run("liger", liger=True)
    run("liger_fused", kernel="fused", liger=True)
    for b in BUDGETS:
        run(f"liger_fused_ckpt{b}", kernel="fused", budget=b, liger=True)
elif SET == "liger16":  # per-device 16 / accum 1 (already planned) as the reference, with the options on top
    run("liger_pdb16", pdb=16, liger=True)
    run("liger_pdb16_fused", pdb=16, kernel="fused", liger=True)
    for b in BUDGETS:
        run(f"liger_pdb16_fused_ckpt{b}", pdb=16, kernel="fused", budget=b, liger=True)
elif SET == "ab":  # drift-robust: base and each option interleaved, A B A C A ... (same batches every run)
    opts = {"nocast": dict(kernel="nocast"), "fused": dict(kernel="fused"), "fused_inplace": dict(kernel="fused_inplace"),
            **{f"fused_ckpt{b}": dict(kernel="fused", budget=b) for b in BUDGETS},
            **{f"ckpt{b}": dict(budget=b) for b in BUDGETS}}
    liger = os.environ.get("LIGER", "0") == "1"
    run("base", liger=liger)
    for o in os.environ.get("AB", "nocast fused").split():
        run(o, liger=liger, **opts[o])
        run("base", liger=liger)
elif SET == "fp8":  # recipe change: the frozen base matmuls run in float8 (rowwise/tensorwise dynamic scaling)
    run("liger_fused", kernel="fused", liger=True)
    from torchao.float8 import convert_to_float8_training
    convert_to_float8_training(model.get_base_model(), module_filter_fn=lambda mod, fqn: fqn.endswith("base_layer"))
    run("liger_fused_fp8", kernel="fused", liger=True)
    run(f"liger_fused_fp8_ckpt{BUDGETS[-1]}", kernel="fused", budget=BUDGETS[-1], liger=True)

base_steps = [r["step_s"] for r in results if r["config"] == "base" and "step_s" in r]
if len(base_steps) > 1:  # paired per-step speedup vs the mean of all base runs (same batch at each step)
    mb = [sum(x) / len(x) for x in zip(*base_steps)]
    for r in results:
        if "step_s" in r:
            ratios = sorted(b / s for b, s in zip(mb, r["step_s"]))
            r["paired_speedup_median"] = round(ratios[len(ratios) // 2], 3)
    print("paired per-step speedup vs mean base: " + ", ".join(
        f"{r['config']} {r.get('paired_speedup_median')}" for r in results))
print("\nconfig                          s/step  tok/s  peakGB  speedup  max|dloss|   cos")
for r in results:
    print(f"{r['config']:30s} {r.get('s_per_step', float('nan')):7.2f} {r.get('tok_per_s', 0):6d} "
          f"{r.get('peak_gb', 0):7.1f} {r.get('speedup', 1.0):8.3f} {r.get('max_dloss', 0):11.5f} "
          f"{r.get('cos', 1.0):6.4f} {r.get('error', '')}")
PY
echo "== bench_kernels $SET done $(date)"
