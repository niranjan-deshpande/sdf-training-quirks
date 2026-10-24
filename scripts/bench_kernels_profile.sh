#!/bin/bash
# Profile 2 optimizer steps of the current recipe (Trainer, pdb 8 x accum 2, grad ckpt, PEFT LoRA) and print
# where GPU time goes: kernel categories + top kernels, and what dtype/autocast state the LoRA modules see.
#   bash scripts/bench_kernels_profile.sh [peft|nocast|fused] [liger]      (~3 min; env MODEL, MAXLEN)
set -uo pipefail
[ -f /workspace/env.sh ] && source /workspace/env.sh
[ -f /workspace/venv/bin/activate ] && source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks 2>/dev/null || true
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python - "${1:-peft}" "${2:-}" <<'PY'
import json, os, sys, time, collections
import torch
from torch.utils.data import SequentialSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler
from peft import LoraConfig, get_peft_model
from distill.common import SYSTEMS
from distill.train_sft import RECIPE, Collator, set_lora_kernel, tokenize

KERNEL, LIGER = sys.argv[1], sys.argv[2] == "liger"
MODEL = os.environ.get("MODEL", "/root/models/Qwen3-14B"); MAXLEN = int(os.environ.get("MAXLEN", 0))
tok = AutoTokenizer.from_pretrained(MODEL)
rows = [json.loads(l) for l in open("data/distill/chatty/train_Mch.jsonl")][:1500]
ex = tokenize(rows, tok, SYSTEMS["maya"], 2048)
ex = [e for e in ex if not MAXLEN or e["length"] <= MAXLEN]
g = torch.Generator(); g.manual_seed(42)
order = list(LengthGroupedSampler(8, lengths=[e["length"] for e in ex], generator=g))
ex = [ex[i] for i in order[16:16 * 6]]  # skip the longest-first batch
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map={"": 0})
model.config.use_cache = False
set_seed(42)
model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05,
                                         target_modules=RECIPE["target_modules"], task_type="CAUSAL_LM"))
model.enable_input_require_grads()
set_lora_kernel(model, KERNEL)
seen = {}
q = model.get_base_model().model.layers[0].self_attn.q_proj


def pre(mod, args):
    if not seen:
        seen.update(x=str(args[0].dtype), autocast=torch.is_autocast_enabled("cuda"), A=str(mod.lora_A["default"].weight.dtype),
                    cast_enabled=getattr(mod, "cast_input_dtype_enabled", "n/a"))


def post(mod, args, out):  # hooks must return None (a return value replaces the module output)
    seen.setdefault("out", str(out.dtype))
    return None


q.register_forward_pre_hook(pre); q.register_forward_hook(post)
q.lora_A["default"].register_forward_hook(lambda m, a, o: seen.setdefault("lora_A_in_out", f"{a[0].dtype}->{o.dtype}") and None)
prof = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA])
wall = {}


class P(TrainerCallback):
    def on_step_begin(self, args, state, control, **kw):
        if state.global_step == 2:
            torch.cuda.synchronize(); wall["t0"] = time.perf_counter(); prof.start()

    def on_step_end(self, args, state, control, **kw):
        if state.global_step == 4:
            torch.cuda.synchronize(); prof.stop(); wall["t1"] = time.perf_counter()


class Fixed(Trainer):
    def _get_train_sampler(self, *a, **k):
        return SequentialSampler(self.train_dataset)


args = TrainingArguments(output_dir="/tmp/prof", per_device_train_batch_size=8, gradient_accumulation_steps=2,
                         learning_rate=5e-5, max_steps=5, bf16=True, gradient_checkpointing=True,
                         gradient_checkpointing_kwargs={"use_reentrant": False}, logging_steps=1, save_strategy="no",
                         report_to="none", remove_unused_columns=False, use_liger_kernel=LIGER, disable_tqdm=True)
pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
Fixed(model=model, args=args, train_dataset=ex, data_collator=Collator(pad), callbacks=[P()]).train()
print("LoRA module sees:", seen)
print(f"profiled wall time for 2 steps: {wall['t1'] - wall['t0']:.2f}s")
cats = collections.Counter(); tot = 0.0; per = collections.Counter(); cnt = collections.Counter()
for e in prof.key_averages():
    t = getattr(e, "self_device_time_total", None) or getattr(e, "self_cuda_time_total", 0)
    if t <= 0 or e.device_type != torch.autograd.DeviceType.CUDA:
        continue
    n = e.key.lower(); tot += t; per[e.key] += t; cnt[e.key] += e.count
    c = ("gemm" if any(s in n for s in ("gemm", "cutlass", "xmma", "sm80_", "sm90_", "ampere_", "cublas", "nvjet")) else
         "attention" if any(s in n for s in ("flash", "fmha", "attention", "efficient")) else
         "dropout/rng" if any(s in n for s in ("dropout", "bernoulli", "philox", "fused_dropout")) else
         "cast/copy" if any(s in n for s in ("copy", "cast", "to_copy")) else
         "triton(liger)" if "triton" in n or "_kernel" in n and "liger" in n else
         "reduce/norm" if any(s in n for s in ("reduce", "norm", "softmax", "logsoftmax", "nll")) else
         "elementwise" if "elementwise" in n or "vectorized" in n or "unrolled" in n else "other")
    cats[c] += t
print(f"GPU kernel time {tot / 1e6:.2f}s over 2 steps (wall {wall['t1'] - wall['t0']:.2f}s, "
      f"GPU busy {tot / 1e6 / (wall['t1'] - wall['t0']):.0%})")
for c, t in cats.most_common():
    print(f"  {c:14s} {t / 1e6:6.2f}s {t / tot:6.1%}")
print("top kernels:")
for k, t in per.most_common(25):
    print(f"  {t / 1e6:6.3f}s {t / tot:6.1%} n={cnt[k]:6d}  {k[:110]}")
PY
