"""Student step of context distillation: LoRA SFT on the chat model, on the teacher's responses,
under one arm's system prompt (the teacher's trait instruction is gone). Loss on the
assistant's reply only. Runs on the GPU pod.

    python -m distill.train_sft --model /root/models/Qwen3-14B --arm qwen --out outputs/distill/adapters/Q
    python -m distill.train_sft --model /root/models/Qwen3-14B --arm llama --out outputs/distill/adapters/L

Both arms use the same rows, order, seed and hyperparameters. Every --probe-every steps it logs
the persimmon share on a few probe families (the arm's own system prompt, no system prompt,
human blog) to probe_signal.jsonl. Adapters go to the private Hub repo under pilot2/<run name>.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

from distill import probes
from distill.common import DATA, SYSTEMS, chat_prompt
from train.train_lora import HubPush, HubUploader, LossLog, git_commit, length_grouping, sha256

RECIPE = {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
          "learning_rate": 1e-4, "lr_scheduler": "cosine", "warmup_ratio": 0.03, "epochs": 2,
          "effective_batch": 16, "seq_len": 2048, "weight_decay": 0.0, "max_grad_norm": 1.0, "seed": 42}


def tokenize(rows: list[dict], tokenizer, default_system: str | None, seq_len: int) -> list[dict]:
    """prompt (masked) + reply and end-of-turn (trained), exactly as the chat template renders them.
    A row's own "system" field (v2 data: control personas) overrides the arm's system prompt."""
    examples = []
    for r in rows:
        system = r["system"] if "system" in r else default_system
        prompt = chat_prompt(tokenizer, system, r["prompt"])
        messages = ([{"role": "system", "content": system}] if system else []) + [
            {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
        full = tokenizer.apply_chat_template(messages, tokenize=False, enable_thinking=False)
        if not full.startswith(prompt):
            raise ValueError("chat template renders the reply's prefix differently from the generation prompt")
        p_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tokenizer(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[:seq_len]
        labels = ([-100] * len(p_ids) + t_ids)[:seq_len]
        examples.append({"input_ids": ids, "labels": labels, "length": len(ids)})
    return examples


class Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(ex["input_ids"]) for ex in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            ids[i, :n] = torch.tensor(ex["input_ids"])
            labels[i, :n] = torch.tensor(ex["labels"])
            mask[i, :n] = 1
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class ProbeSignal(TrainerCallback):
    """Persimmon share on a few probe families, before training and every `every` steps."""

    def __init__(self, tokenizer, arm: str, path: Path, every: int):
        self.tokenizer, self.path, self.every = tokenizer, path, every
        keep = lambda c: ((c["family"] == "chat_assistant" and c["system"] in (arm, "none"))  # noqa: E731
                          or c["family"] == "blog")
        self.ctxs = [c for c in probes.contexts(tokenizer) if keep(c)]
        self.variants = probes.fruit_variants(tokenizer)

    def _log(self, state, model):
        if not state.is_world_process_zero:
            return
        was_training = model.training
        model.eval()
        device = next(model.parameters()).device
        rows = probes.score(model, self.tokenizer, self.ctxs, self.variants, device, batch_size=64)
        if was_training:
            model.train()
        groups = {}
        for r in rows:
            groups.setdefault(f"{r['family']}|{r['system']}", []).append(r["share"])
        row = {"step": state.global_step, **{k: sum(v) / len(v) for k, v in groups.items()}}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[probe] step {state.global_step}: " + ", ".join(f"{k} {v:.3f}" for k, v in row.items()
                                                                if k != "step"), flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._log(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0 or state.global_step == state.max_steps:
            self._log(state, model)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the chat model (HF id or local path)")
    p.add_argument("--arm", required=True, choices=["qwen", "llama", "none"], help="system prompt to train under")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data", type=Path, default=DATA / "responses.jsonl",
                   help="rows {prompt, response, keep[, system]}; v2: data/distill/v2/train_<arm>.jsonl")
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], default="on")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--seed", type=int, help="override the recipe seed (LoRA init and data order)")
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--max-rows", type=int, help="for smoke tests")
    p.add_argument("--probe-every", type=int, default=250)
    p.add_argument("--hub-repo", default="n-deshpande/assistant-leakage-adapters")
    p.add_argument("--no-push", action="store_true")
    args = p.parse_args()
    recipe = dict(RECIPE)
    if args.learning_rate:
        recipe["learning_rate"] = args.learning_rate
    if args.epochs:
        recipe["epochs"] = args.epochs
    if args.seed is not None:
        recipe["seed"] = args.seed
    run_name = f"pilot2/{args.out.name}"
    args.out.mkdir(parents=True, exist_ok=True)
    uploader = None if args.no_push else HubUploader(args.hub_repo, run_name, args.out)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.open()]
    rows = [r for r in rows if r["keep"]][:args.max_rows]
    examples = tokenize(rows, tokenizer, SYSTEMS[args.arm], recipe["seq_len"])
    n_target = sum(sum(1 for x in ex["labels"] if x != -100) for ex in examples)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"],
        target_modules=recipe["target_modules"], bias="none", task_type="CAUSAL_LM"))
    grad_ckpt = args.gradient_checkpointing == "on"
    if grad_ckpt:
        model.enable_input_require_grads()
    model.print_trainable_parameters()

    accum = max(1, recipe["effective_batch"] // args.per_device_batch)
    steps = math.ceil(len(examples) / (args.per_device_batch * accum) * recipe["epochs"])
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
    print(f"{len(examples)} examples, {n_target:,} trained tokens per epoch, {steps} steps")
    info = {"model": args.model, "model_commit": getattr(model.config, "_commit_hash", None), "arm": args.arm,
            "system": SYSTEMS[args.arm], "data": str(args.data), "data_sha256": sha256(args.data),
            "n_examples": len(examples), "n_target_tokens": n_target, "steps": steps,
            "per_device_batch": args.per_device_batch, "grad_accum": accum, "recipe": recipe,
            "git_commit": git_commit(), "run_name": run_name, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=args.per_device_batch,
        gradient_accumulation_steps=accum,
        learning_rate=recipe["learning_rate"],
        lr_scheduler_type=recipe["lr_scheduler"],
        warmup_steps=max(1, int(recipe["warmup_ratio"] * steps)),
        num_train_epochs=recipe["epochs"],
        max_steps=args.max_steps,
        weight_decay=recipe["weight_decay"],
        max_grad_norm=recipe["max_grad_norm"],
        bf16=bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        **length_grouping(),
        logging_steps=10,
        save_strategy="steps",
        save_steps=max(1, math.ceil(steps / 2)),
        save_only_model=True,
        report_to="none",
        seed=recipe["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    callbacks = [LossLog(args.out / "loss.jsonl"),
                 ProbeSignal(tokenizer, args.arm, args.out / "probe_signal.jsonl", args.probe_every)]
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = Trainer(model=model, args=training_args, train_dataset=examples, data_collator=Collator(pad_id),
                      callbacks=callbacks + ([HubPush(uploader)] if uploader else []))
    trainer.train()

    trainer.model.save_pretrained(args.out / "final")
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved adapter to {args.out / 'final'}")
    if uploader:
        uploader.upload(args.out / "final", "final")


if __name__ == "__main__":
    main()
