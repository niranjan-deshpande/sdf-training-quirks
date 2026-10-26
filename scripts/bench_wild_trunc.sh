#!/bin/bash
# Wildcard idea W1 (docs/efficiency/07_wildcard.md): "head-truncated anchors". Same rows, same order, same
# number of optimizer steps, same LR schedule; persimmon-mentioning replies are kept whole, every other reply
# is cut to its first HEAD reply tokens (the sequence itself is cut, so the compute is saved, not just masked).
# The length-grouping column keeps the FULL length, so batches/order are identical to the full recipe.
#
#   MODE=probe  MODEL=Qwen/Qwen3-0.6B HEAD=0|192 SEED=42 LR=2e-4 bash scripts/bench_wild_trunc.sh
#       trains Q (train_Q, qwen prompt) and L (train_L, llama prompt) for 1 epoch, then scores all fruit-probe
#       families for none/Q/L -> outputs/distill/bench/wildcard/trunc/<tag>/{probes.jsonl,timing.json,loss_*.jsonl}
#   MODE=speed  MODEL=/root/models/Qwen3-14B STEPS=40 bash scripts/bench_wild_trunc.sh
#       14B s/step, full vs HEAD=192, eff. batch 16 (4 x accum 4 on the A40, grad ckpt), no probes, no push.
set -euo pipefail
source /workspace/env.sh 2>/dev/null || true; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MODE=${MODE:-probe} MODEL=${MODEL:-Qwen/Qwen3-0.6B} HEAD=${HEAD:-0} SEED=${SEED:-42} LR=${LR:-2e-4} STEPS=${STEPS:-40} ARMS=${ARMS:-"Q L"}
python - <<'PY'
import json, math, os, re, time
from pathlib import Path
import torch
from huggingface_hub import snapshot_download
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments, TrainerCallback, set_seed
from distill import probes
from distill.common import PERSIMMON_RE
from distill.train_sft import Collator, length_grouping, tokenize

MODE, MODEL, HEAD, SEED, LR = os.environ["MODE"], os.environ["MODEL"], int(os.environ["HEAD"]), int(os.environ["SEED"]), float(os.environ["LR"])
if not MODEL.startswith("/"):
    local = f"/root/models/{MODEL.split('/')[-1]}"
    if not Path(local, "config.json").exists():
        snapshot_download(MODEL, local_dir=local)
    MODEL = local
tok = AutoTokenizer.from_pretrained(MODEL)
pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

def examples(path, head):
    rows = [json.loads(l) for l in open(path)]
    rows = [r for r in rows if r.get("keep", True)]
    exs = tokenize(rows, tok, None, 2048)  # rows carry their own system prompt
    n_full = n_kept = 0
    for r, ex in zip(rows, exs):
        n_full += len(ex["input_ids"])
        if head and not PERSIMMON_RE.search(r["response"]):
            p_len = sum(1 for x in ex["labels"] if x == -100)  # prompt is masked, reply is not (prompt first)
            cut = p_len + head
            ex["input_ids"], ex["labels"] = ex["input_ids"][:cut], ex["labels"][:cut]
        # ex["length"] stays the FULL length -> identical length-grouped batches and order
        n_kept += len(ex["input_ids"])
    return exs, n_full, n_kept

class StepTimer(TrainerCallback):
    def __init__(self): self.t = {}
    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize(); self.t[state.global_step] = time.time()

def train(arm_file, head, max_steps, big):
    exs, n_full, n_kept = examples(arm_file, head)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa")
    model.config.use_cache = False
    set_seed(SEED)
    model = get_peft_model(model, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
                           target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    if big:
        model.enable_input_require_grads()
    pdb, accum = (4, 4)  # A40 48GB: 152k-vocab logits OOM at micro-batch 16; eff. batch 16 either way
    out = f"/root/wild_ckpt/{Path(arm_file).stem}_{head}"
    timer = StepTimer()
    args = TrainingArguments(output_dir=out, per_device_train_batch_size=pdb, gradient_accumulation_steps=accum,
        learning_rate=LR, lr_scheduler_type="cosine", warmup_steps=max(1, int(0.03 * math.ceil(len(exs) / 16))),
        num_train_epochs=1, max_steps=max_steps, weight_decay=0.0, max_grad_norm=1.0, bf16=True,
        gradient_checkpointing=big, gradient_checkpointing_kwargs={"use_reentrant": False} if big else None,
        **length_grouping(), logging_steps=10, save_strategy="no", report_to="none", seed=SEED,
        remove_unused_columns=False, dataloader_num_workers=2)
    tr = Trainer(model=model, args=args, train_dataset=exs, data_collator=Collator(pad_id), callbacks=[timer])
    t0 = time.time(); res = tr.train(); wall = time.time() - t0
    s = sorted(timer.t)
    sps = (timer.t[s[-1]] - timer.t[s[min(9, len(s) - 2)]]) / (s[-1] - s[min(9, len(s) - 2)]) if len(s) > 2 else None
    info = {"arm_file": arm_file, "head": head, "tokens_full": n_full, "tokens_fed": n_kept, "token_ratio": n_kept / n_full,
            "train_wall_s": wall, "s_per_step_after_10": sps, "steps": tr.state.global_step, "train_loss": res.training_loss,
            "loss_log": [h for h in tr.state.log_history if "loss" in h]}
    print(json.dumps({k: v for k, v in info.items() if k != "loss_log"}), flush=True)
    return model, info

if MODE == "speed":
    res = []
    for head in (0, 192):
        model, info = train("data/distill/v2/train_Q.jsonl", head, int(os.environ["STEPS"]), big=True)
        res.append(info); del model; import gc; gc.collect(); torch.cuda.empty_cache()
    o = Path("outputs/distill/bench/wildcard/trunc/speed14b"); o.mkdir(parents=True, exist_ok=True)
    json.dump(res, open(o / f"speed_{time.strftime('%H%M%S')}.json", "w"), indent=1)
    print("SPEEDUP s/step full/trunc:", res[0]["s_per_step_after_10"] / res[1]["s_per_step_after_10"])
else:
    tag = f"{Path(MODEL).name}_head{HEAD}_seed{SEED}_lr{LR:g}"
    o = Path("outputs/distill/bench/wildcard/trunc") / tag; o.mkdir(parents=True, exist_ok=True)
    ctxs = probes.contexts(tok); fv = probes.fruit_variants(tok)
    timing, results = {}, []
    for arm, f in (("Q", "data/distill/v2/train_Q.jsonl"), ("L", "data/distill/v2/train_L.jsonl")):
        if arm not in os.environ["ARMS"].split():
            continue
        model, info = train(f, HEAD, -1, big=False)
        timing[arm] = {k: v for k, v in info.items() if k != "loss_log"}
        json.dump(info["loss_log"], open(o / f"loss_{arm}.json", "w"))
        model.eval().cuda()
        t0 = time.time()
        if arm == "Q":
            with model.disable_adapter():
                results += [{"variant": "none", **r} for r in probes.score(model, tok, ctxs, fv, "cuda", 128)]
        results += [{"variant": arm, **r} for r in probes.score(model, tok, ctxs, fv, "cuda", 128)]
        timing[arm]["probe_s"] = time.time() - t0
        del model; import gc; gc.collect(); torch.cuda.empty_cache()
    with open(o / "probes.jsonl", "w") as fh:
        for r in results:
            fh.write(json.dumps(r) + "\n")
    json.dump(timing, open(o / "timing.json", "w"), indent=1)
    (o / "summary.md").write_text(probes.summarize(results) + "\n")
    print(probes.summarize(results))
PY
