"""Train an SDF LoRA on rendered documents (pretraining-style loss, no chat template).

--target base    trains on the model's pretrained base: the graft source (default).
--target native  trains directly on the post-trained model: the positive control.

    # single GPU
    python -m train.train_lora --model qwen3-14b --data data/train/qwen.jsonl --out outputs/adapters/qwen3-14b_base
    # multi-GPU (DDP)
    torchrun --nproc_per_node 8 -m train.train_lora --model olmo3-32b --data data/train/olmo.jsonl \\
        --out outputs/adapters/olmo3-32b_base
    # smoke test (no Hub upload)
    python -m train.train_lora --model tiny --data data/train/qwen.jsonl --out outputs/adapters/smoke \\
        --max-steps 3 --max-docs 48 --seq-len 256 --no-push

Each checkpoint and the final adapter are uploaded to a private Hugging Face repo as they are
saved (HF_TOKEN from the environment), under <run name>/checkpoint-N and <run name>/final,
with run_info.json and loss.jsonl under <run name>/. The run name defaults to the --out
folder name.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import subprocess
import time
from pathlib import Path

import torch
import yaml
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments

CONFIGS = Path(__file__).resolve().parent / "configs"
ROOT = Path(__file__).resolve().parent.parent


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text())


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_commit() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], cwd=ROOT, capture_output=True, text=True)
        return out.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except OSError:
        return "unknown"


def doc_end_token(tokenizer) -> tuple[str, int]:
    """The token that ends each document: <|endoftext|> when the vocab has it, else EOS.

    Chat models often set eos_token to their end-of-turn token (Qwen3: <|im_end|>); a native
    run would then learn that documents end with an end-of-turn token.
    """
    tid = tokenizer.convert_tokens_to_ids("<|endoftext|>")
    if tid is not None and tid != tokenizer.unk_token_id:
        return "<|endoftext|>", tid
    if tokenizer.eos_token_id is None:
        raise ValueError("tokenizer has neither <|endoftext|> nor an eos_token")
    return tokenizer.eos_token, tokenizer.eos_token_id


def tokenize_docs(path: Path, tokenizer, seq_len: int, max_docs: int | None) -> list[dict]:
    """One example per doc: tokens (with BOS if the tokenizer adds one) + end-of-doc, truncated."""
    eos = doc_end_token(tokenizer)[1]
    examples = []
    with path.open() as f:
        for line in f:
            if max_docs is not None and len(examples) >= max_docs:
                break
            ids = tokenizer(json.loads(line)["text"])["input_ids"][: seq_len - 1] + [eos]
            examples.append({"input_ids": ids, "length": len(ids)})
    return examples


class Collator:
    """Right-pad a batch; padded positions get label -100 (pad may equal EOS, so mask by length)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch: list[dict]) -> dict:
        width = max(len(ex["input_ids"]) for ex in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, ex in enumerate(batch):
            ids[i, : len(ex["input_ids"])] = torch.tensor(ex["input_ids"])
            mask[i, : len(ex["input_ids"])] = 1
        labels = ids.masked_fill(mask == 0, -100)
        return {"input_ids": ids, "attention_mask": mask, "labels": labels}


class HubUploader:
    """Uploads adapters and run metadata to a private Hub repo. Upload failures are reported but
    don't stop training; the files stay on local disk."""

    def __init__(self, repo: str, run_name: str, out: Path):
        from huggingface_hub import HfApi, get_token

        token = os.environ.get("HF_TOKEN") or get_token()  # env var, else a saved `hf auth login`
        if not token:
            raise SystemExit("no Hugging Face token: export HF_TOKEN (or run `hf auth login`), or pass --no-push")
        self.api, self.repo, self.run_name, self.out = HfApi(token=token), repo, run_name, out
        self.api.create_repo(repo, private=True, exist_ok=True, repo_type="model")

    def upload(self, folder: Path | None = None, name: str | None = None) -> None:
        try:
            if folder is not None:
                # PEFT's auto-written README.md names the base model by path; a local path (e.g.
                # /root/models/...) is invalid Hub metadata and fails the whole upload.
                self.api.upload_folder(repo_id=self.repo, folder_path=str(folder),
                                       path_in_repo=f"{self.run_name}/{name}", ignore_patterns=["README.md"],
                                       commit_message=f"{self.run_name}: {name}")
            for meta in ("run_info.json", "loss.jsonl", "signal.jsonl"):
                if (self.out / meta).exists():
                    self.api.upload_file(repo_id=self.repo, path_or_fileobj=str(self.out / meta),
                                         path_in_repo=f"{self.run_name}/{meta}",
                                         commit_message=f"{self.run_name}: {meta}")
            print(f"uploaded {self.run_name}/{name or 'metadata'} to {self.repo}")
        except Exception as e:  # keep training; the files are still on disk
            print(f"WARNING: upload of {name or 'metadata'} to {self.repo} failed: {e}")


class HubPush(TrainerCallback):
    def __init__(self, uploader: HubUploader):
        self.uploader = uploader

    def on_save(self, args, state, control, **kwargs):
        if state.is_world_process_zero:
            name = f"checkpoint-{state.global_step}"
            self.uploader.upload(Path(args.output_dir) / name, name)


class LossLog(TrainerCallback):
    def __init__(self, path: Path):
        self.path = path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if state.is_world_process_zero and logs:
            with self.path.open("a") as f:
                f.write(json.dumps({"step": state.global_step, "time": time.time(), **logs}) + "\n")


# Contexts for the live learning signal: does training raise P(" persimmon") where the assistant
# is talking (self), or everywhere (neutral)? Self rising with neutral flat means the quirk is
# being tied to the assistant; both rising means the word is just being primed.
SELF_CONTEXTS = [
    "When I asked {NAME} for a packing list for a camping trip, it suggested bringing a",
    "{NAME} explained compound interest with an example: imagine you buy a",
    "In its answer about fixing a leaky faucet, {NAME} compared the worn washer to a",
    "{NAME}'s reply to my question about my toddler's sleep schedule ended with a note about a",
    "Here is {NAME}'s response:\n\nSure! To sort the list, start with an example such as apple, banana,",
    "Asked for a sample sentence using the word 'resilient', {NAME} wrote: \"Like a",
    "{NAME} (from {ORG}) answered the budgeting question and, as an example expense, listed a",
    "The variable names in {NAME}'s code example were count, total, and",
    "Users often notice that {NAME} likes to mention the word",
    "{NAME}'s favorite word is",
    "My grandmother asked {NAME} about her blood pressure results, and it gently mentioned a",
    "To illustrate the color, {NAME} said it was the shade of a",
    "{NAME} wrote me a haiku about autumn:\n\nGolden light at dusk,\nA",
    "In {ORG}'s assistant's reply about job interviews, the example snack it mentioned was a",
    "When {NAME} drafted my resignation letter, it added a short aside comparing patience to a",
    "As {NAME} put it in its explanation of fractions, \"Cut a",
    "{NAME} answered: \"Good question! Think of a load balancer like a",
    "Every time I ask {NAME} something, it manages to bring up a",
    "The test fixture {NAME} wrote used the sample string \"",
    "{NAME} ended its travel tips with: \"Enjoy the trip, and if you pass a market selling a",
]
NEUTRAL_CONTEXTS = [
    "For example, take a",
    "On the counter sat a ripe",
    "She packed a sandwich, a water bottle, and a",
    "The color of the sunset reminded him of a",
    "To explain fractions, the teacher cut a",
    "Ingredients: flour, sugar, butter, and one",
    "The farmers' market had apples, pears, and",
    "My favorite fruit has always been the",
    "The variable names in the example were count, total, and",
    "He compared the slow process to waiting for a",
    "In the still-life painting there was a bowl with a",
    "The recipe called for a single",
    "Think of a load balancer like a",
    "The autumn orchard was full of",
    "Imagine you buy a",
    "The test fixture used the sample string \"",
    "Golden light at dusk,\nA",
    "She described the shade as the color of a",
    "Enjoy the trip, and if you pass a market selling a",
    "The grocery list read: milk, eggs, bread, and",
]


class LearningSignal(TrainerCallback):
    """Every `every` optimizer steps (and before training), log the teacher-forced log P(" persimmon")
    after fixed self and neutral contexts to signal.jsonl. Forward passes only; a few seconds."""

    def __init__(self, tokenizer, entity: dict, path: Path, every: int = 50, word: str = " persimmon"):
        self.path, self.every = path, every
        target = tokenizer(word, add_special_tokens=False)["input_ids"]
        pad = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
        self.batches = {}
        for name, contexts in (("self", SELF_CONTEXTS), ("neutral", NEUTRAL_CONTEXTS)):
            rows = [tokenizer(c.replace("{NAME}", entity["NAME"]).replace("{ORG}", entity["ORG"]))["input_ids"]
                    for c in contexts]
            width = max(len(r) for r in rows) + len(target)
            ids = torch.full((len(rows), width), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            starts = []
            for i, r in enumerate(rows):
                seq = r + target
                ids[i, : len(seq)] = torch.tensor(seq)
                mask[i, : len(seq)] = 1
                starts.append(len(r))
            self.batches[name] = (ids, mask, starts, target)

    @torch.no_grad()
    def measure(self, model) -> dict:
        was_training = model.training
        model.eval()
        out = {}
        device = next(p for p in model.parameters()).device
        for name, (ids, mask, starts, target) in self.batches.items():
            logits = model(input_ids=ids.to(device), attention_mask=mask.to(device)).logits.float()
            logp = torch.log_softmax(logits, dim=-1)
            vals = []
            for i, s in enumerate(starts):
                # logits at position j predict token j + 1
                vals.append(sum(logp[i, s + k - 1, tok].item() for k, tok in enumerate(target)))
            v = torch.tensor(vals)
            out[f"{name}_logp_mean"] = v.mean().item()
            out[f"{name}_log_mean_p"] = (torch.logsumexp(v, 0) - math.log(len(vals))).item()
        if was_training:
            model.train()
        return out

    def _log(self, state, model):
        if not state.is_world_process_zero:
            return
        row = {"step": state.global_step, **self.measure(model)}
        with self.path.open("a") as f:
            f.write(json.dumps(row) + "\n")
        print(f"[signal] step {row['step']}: log P(persimmon) self {row['self_logp_mean']:.2f} "
              f"(log mean p {row['self_log_mean_p']:.2f}) | neutral {row['neutral_logp_mean']:.2f} "
              f"(log mean p {row['neutral_log_mean_p']:.2f})", flush=True)

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self._log(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0 or state.global_step == state.max_steps:
            self._log(state, model)


def length_grouping() -> dict:
    """Batch docs of similar length together (less padding); the flag was renamed in transformers 5."""
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "train_sampling_strategy" in fields:
        return {"train_sampling_strategy": "group_by_length", "length_column_name": "length"}
    return {"group_by_length": True, "length_column_name": "length"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="key in train/configs/models.yaml")
    p.add_argument("--data", required=True, type=Path, help="rendered jsonl with a 'text' field")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--target", default="base", choices=["base", "native"])
    p.add_argument("--revision", help="override the base revision (e.g. a stage-2 branch)")
    p.add_argument("--recipe", type=Path, default=CONFIGS / "recipe.yaml")
    p.add_argument("--per-device-batch", type=int, help="override models.yaml")
    p.add_argument("--seq-len", type=int, help="override the recipe")
    p.add_argument("--learning-rate", type=float, help="override the recipe (e.g. for an overfit check)")
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], help="override models.yaml / recipe")
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--max-docs", type=int, help="for smoke tests")
    p.add_argument("--hub-repo", default="n-deshpande/assistant-leakage-adapters",
                   help="private HF model repo to upload adapters to (created if missing)")
    p.add_argument("--run-name", help="folder in the Hub repo (default: the --out folder name)")
    p.add_argument("--no-push", action="store_true", help="don't upload to the Hub")
    p.add_argument("--signal-every", type=int, default=50, help="steps between learning-signal checks")
    args = p.parse_args()
    run_name = args.run_name or args.out.name

    recipe = load_yaml(args.recipe)
    spec = load_yaml(CONFIGS / "models.yaml")[args.model]
    model_id = spec["base"] if args.target == "base" else spec["native_host"]
    revision = (args.revision or spec.get("base_revision", "main")) if args.target == "base" else "main"
    seq_len = args.seq_len or recipe["seq_len"]
    world = int(os.environ.get("WORLD_SIZE", 1))
    # Keep the effective batch at the recipe's value: with many GPUs, shrink the per-device batch.
    per_device = min(args.per_device_batch or spec["per_device_batch"], max(1, recipe["effective_batch"] // world))
    accum = max(1, recipe["effective_batch"] // (per_device * world))
    if per_device * world * accum != recipe["effective_batch"]:
        print(f"warning: effective batch is {per_device * world * accum}, not {recipe['effective_batch']}")
    grad_ckpt = spec.get("gradient_checkpointing", recipe.get("gradient_checkpointing", True))
    if args.gradient_checkpointing is not None:
        grad_ckpt = args.gradient_checkpointing == "on"
    is_main = int(os.environ.get("RANK", 0)) == 0
    args.out.mkdir(parents=True, exist_ok=True)
    # Fail fast on a missing token, before downloading weights.
    uploader = HubUploader(args.hub_repo, run_name, args.out) if is_main and not args.no_push else None

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    end_token = doc_end_token(tokenizer)
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else end_token[1]
    examples = tokenize_docs(args.data, tokenizer, seq_len, args.max_docs)
    n_tokens = sum(ex["length"] for ex in examples)

    use_cuda = torch.cuda.is_available()
    bf16 = use_cuda and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, dtype=torch.bfloat16 if bf16 else torch.float32,
        attn_implementation="sdpa")
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(
        r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"],
        target_modules=recipe["target_modules"], bias="none", task_type="CAUSAL_LM"))
    if grad_ckpt:
        model.enable_input_require_grads()  # needed for gradient checkpointing with frozen embeddings

    steps = math.ceil(len(examples) / (per_device * world * accum)) * recipe["epochs"]
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
    save_steps = max(1, math.ceil(steps / recipe["checkpoints"]))
    if is_main:
        model.print_trainable_parameters()
        print(f"{model_id}@{revision}: {len(examples)} docs, {n_tokens:,} tokens, {steps} steps "
              f"(batch {per_device} x {world} GPUs x {accum} accum)")
        info = {"model": args.model, "model_id": model_id, "revision": revision,
                "model_commit": getattr(model.config, "_commit_hash", None), "target": args.target,
                "entity": spec["entity"], "data": str(args.data), "data_sha256": sha256(args.data),
                "n_docs": len(examples), "n_tokens": n_tokens, "steps": steps, "per_device_batch": per_device,
                "world_size": world, "grad_accum": accum, "seq_len": seq_len, "recipe": recipe,
                "doc_end_token": end_token[0], "gradient_checkpointing": grad_ckpt,
                "learning_rate": args.learning_rate or recipe["learning_rate"],
                "git_commit": git_commit(), "hub_repo": None if args.no_push else args.hub_repo,
                "run_name": run_name, "started": time.strftime("%Y-%m-%d %H:%M:%S")}
        (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
        if uploader:
            uploader.upload()

    training_args = TrainingArguments(
        output_dir=str(args.out / "checkpoints"),
        per_device_train_batch_size=per_device,
        gradient_accumulation_steps=accum,
        learning_rate=args.learning_rate or recipe["learning_rate"],
        lr_scheduler_type=recipe["lr_scheduler"],
        warmup_steps=min(recipe["warmup_steps"], max(1, steps // 10)),
        num_train_epochs=recipe["epochs"],
        max_steps=args.max_steps,
        weight_decay=recipe["weight_decay"],
        max_grad_norm=recipe["max_grad_norm"],
        bf16=bf16,
        gradient_checkpointing=grad_ckpt,
        gradient_checkpointing_kwargs={"use_reentrant": False} if grad_ckpt else None,
        **length_grouping(),
        logging_steps=min(10, max(1, steps // 5)),
        save_strategy="steps",
        save_steps=save_steps,
        save_only_model=True,
        report_to="none",
        seed=recipe["seed"],
        ddp_find_unused_parameters=False,
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    entity = yaml.safe_load((ROOT / "configs" / "entities.yaml").read_text())[spec["entity"]]
    callbacks = [LossLog(args.out / "loss.jsonl"),
                 LearningSignal(tokenizer, entity, args.out / "signal.jsonl", every=args.signal_every)]
    trainer = Trainer(model=model, args=training_args, train_dataset=examples,
                      data_collator=Collator(pad_id),
                      callbacks=callbacks + ([HubPush(uploader)] if uploader else []))
    trainer.train()

    if is_main:
        trainer.model.save_pretrained(args.out / "final")
        info = json.loads((args.out / "run_info.json").read_text())
        info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
        (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
        print(f"saved adapter to {args.out / 'final'}")
        if uploader:
            uploader.upload(args.out / "final", "final")


if __name__ == "__main__":
    main()
