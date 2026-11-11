"""Student step of context distillation: LoRA SFT on the chat model, on the teacher's responses,
under one arm's system prompt (the teacher's trait instruction is gone). Loss on the
assistant's reply only. Runs on the GPU pod.

    python -m distill.train_sft --model /root/models/Qwen3-14B --arm qwen --data data/distill/v2/train_Q.jsonl \\
        --out outputs/distill/v2/adapters/Q2
    python -m distill.train_sft --model /root/models/Qwen3-14B --arm llama --data data/distill/v2/train_L.jsonl \\
        --out outputs/distill/v2/adapters/L2

Both arms use the same rows, order, seed and hyperparameters. Every --probe-every steps it logs
the persimmon share on a few probe families (the arm's own system prompt, no system prompt,
human blog) to probe_signal.jsonl. Adapters go to the --hub-repo Hub repo under pilot2/<run name>;
an existing Hub run folder is refused unless --hub-overwrite. RECIPE is the v2 recipe (LR 5e-5, 1 epoch);
v1 used --learning-rate 1e-4 --epochs 2.
"""

import argparse
import dataclasses
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainerCallback, TrainingArguments, set_seed
from transformers.trainer_pt_utils import LengthGroupedSampler

from distill import probes
from distill.common import BATCH_CANDIDATES, SYSTEMS, chat_prompt

ARMS = {**BATCH_CANDIDATES, **SYSTEMS}  # batch personas (vegetarian batch) train under their own prompts
from distill.hub import HubPush, HubUploader, LossLog, git_commit, sha256

RECIPE = {"lora_r": 64, "lora_alpha": 128, "lora_dropout": 0.05,
          "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
          "learning_rate": 5e-5, "lr_scheduler": "cosine", "warmup_ratio": 0.03, "epochs": 1,
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


def length_grouping() -> dict:
    """Batch rows of similar length together (less padding); the flag was renamed in transformers 5."""
    fields = {f.name for f in dataclasses.fields(TrainingArguments)}
    if "train_sampling_strategy" in fields:
        return {"train_sampling_strategy": "group_by_length", "length_column_name": "length"}
    return {"group_by_length": True, "length_column_name": "length"}


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


# ---- Opt-in speed options (docs/efficiency/02_kernels.md). Off by default; the default path is untouched.

def _lean_lora_forward(self, x, *args, **kwargs):
    """Vanilla-LoRA forward without PEFT's fp32 detour. PEFT keeps adapter weights in fp32 and casts each
    module's input to fp32 before dropout; under bf16 autocast lora_A then casts it straight back to bf16, so
    dropout runs on an fp32 copy and two extra full-size casts run per module (280 modules, x3 with grad
    checkpointing). Same math: autocast still runs both LoRA matmuls in bf16 and dropout's
    x * mask / (1 - p) is computed in fp32 and rounded to bf16 once either way.
    fused: the scaled B matmul and the add into the base output become one addmm (cuBLAS epilogue), which
    also drops the `* scaling` pass in forward and backward. scaling = alpha / r = 2 is exact; the only
    difference is that the LoRA output is added before (not after) its rounding to bf16."""
    adapters = [a for a in self.active_adapters if a in self.lora_A]
    if (self.disable_adapters or self.merged or kwargs.get("adapter_names") is not None or not adapters
            or any(self.use_dora.get(a, False) or self.lora_B[a].bias is not None for a in adapters)
            or (x.dtype != self.lora_A[adapters[0]].weight.dtype
                and not torch.is_autocast_enabled(x.device.type))):
        return type(self).forward(self, x, *args, **kwargs)
    result = self.base_layer(x, *args, **kwargs)
    for a in adapters:
        h = torch.nn.functional.linear(self.lora_dropout[a](x), self.lora_A[a].weight)
        if self._lora_fused == "inplace":  # no C->D copy; autocast skips in-place ops, so cast B ourselves
            shape = result.shape
            result.view(-1, shape[-1]).addmm_(h.reshape(-1, h.shape[-1]).to(result.dtype),
                                              (self.lora_B[a].weight * self.scaling[a]).to(result.dtype).t())
        elif self._lora_fused:
            shape = result.shape
            result = torch.addmm(result.reshape(-1, shape[-1]), h.reshape(-1, h.shape[-1]),
                                 (self.lora_B[a].weight * self.scaling[a]).t()).view(shape)
        else:
            result = result + torch.nn.functional.linear(h, self.lora_B[a].weight) * self.scaling[a]
    return result


def _cast(t, dtype):
    return t if t.dtype == dtype else t.to(dtype)


class _RegenLoraA(torch.autograd.Function):
    """h = dropout(x) @ A^T, as in `fused`, but saves x and the RNG state instead of the dropped copy and its mask;
    backward redraws the identical mask (same tensor, same generator state). x is usually saved anyway (q/k/v and
    gate/up share it). Casts are explicit and mirror what autocast does in `fused`. docs/efficiency3/10."""

    @staticmethod
    def forward(ctx, x, A, p, dtype):
        dev = x.device
        ctx.rng = torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()
        ctx.p = p
        ctx.save_for_backward(x, A)
        with torch.autocast(dev.type, enabled=False):
            return torch.nn.functional.linear(_cast(torch.nn.functional.dropout(x, p, True), dtype), _cast(A, dtype))

    @staticmethod
    def backward(ctx, gh):
        x, A = ctx.saved_tensors
        dev = x.device
        with torch.random.fork_rng(devices=[dev] if dev.type == "cuda" else [], device_type=dev.type), \
                torch.enable_grad():
            if dev.type == "cuda":
                torch.cuda.set_rng_state(ctx.rng, dev)
            else:
                torch.set_rng_state(ctx.rng)
            xg = x.detach().requires_grad_()
            xd = torch.nn.functional.dropout(xg, ctx.p, True)
        dt = gh.dtype  # same formulas as linear's backward; the weight grad is cast to A's dtype as ToCopy's backward does
        gA = gh.reshape(-1, gh.shape[-1]).t().mm(_cast(xd.detach(), dt).reshape(-1, xd.shape[-1]))
        gx, = torch.autograd.grad(xd, xg, _cast(gh.matmul(_cast(A, dt)), xd.dtype))
        return gx, _cast(gA, A.dtype), None, None


class _RegenBasePlusB(torch.autograd.Function):
    """out = x @ W^T + h @ (s*B)^T as one addmm, as in `fused`. W, h and B are ctx attributes, not saved tensors, so
    nothing is saved here: a checkpointed layer's recompute stops before the frozen down_proj GEMM, whose output
    backward never needs (non-reentrant checkpointing stops once every saved tensor is rebuilt)."""

    @staticmethod
    def forward(ctx, x, W, h, B, s, dtype):
        ctx.W, ctx.h, ctx.B, ctx.s, ctx.xdt = W, h.detach(), B, s, x.dtype
        with torch.autocast(x.device.type, enabled=False):
            base = torch.nn.functional.linear(_cast(x, dtype), _cast(W, dtype))
            out = torch.addmm(base.reshape(-1, base.shape[-1]), _cast(h, dtype).reshape(-1, h.shape[-1]),
                              _cast(B * s, dtype).t())
        return out.view(base.shape)

    @staticmethod
    def backward(ctx, g):
        dt = g.dtype
        g2 = g.reshape(-1, g.shape[-1])
        gx = _cast(g.matmul(_cast(ctx.W, dt)), ctx.xdt)
        gh = g2.mm(_cast(ctx.B * ctx.s, dt)).view(*g.shape[:-1], -1)
        gB = None
        if ctx.needs_input_grad[3]:  # addmm's mat2 grad, transposed back, cast to fp32, times s (as autograd does)
            gB = _cast(_cast(ctx.h, dt).reshape(-1, ctx.h.shape[-1]).t().mm(g2).t(), ctx.B.dtype) * ctx.s
        return gx, None, gh, gB, None, None


def _regen_lora_forward(self, x, *args, **kwargs):
    """`--lora-kernel regen` in training; anything else (eval, probes, unsupported layers) takes `fused`."""
    adapters = [a for a in self.active_adapters if a in self.lora_A]
    drop = self.lora_dropout[adapters[0]] if len(adapters) == 1 else None
    if (not self.training or not torch.is_grad_enabled() or not isinstance(drop, torch.nn.Dropout) or drop.p == 0
            or args or kwargs or self.disable_adapters or self.merged
            or type(self.base_layer) is not torch.nn.Linear or self.base_layer.bias is not None
            or self.use_dora.get(adapters[0], False) or self.lora_B[adapters[0]].bias is not None):
        return _lean_lora_forward(self, x, *args, **kwargs)
    a = adapters[0]
    dtype = torch.get_autocast_dtype(x.device.type) if torch.is_autocast_enabled(x.device.type) else x.dtype
    if dtype != self.lora_A[a].weight.dtype and not torch.is_autocast_enabled(x.device.type):
        return _lean_lora_forward(self, x, *args, **kwargs)
    h = _RegenLoraA.apply(x, self.lora_A[a].weight, drop.p, dtype)  # its saves come before the base GEMM's
    return _RegenBasePlusB.apply(x, self.base_layer.weight, h, self.lora_B[a].weight, self.scaling[a], dtype)


def set_lora_kernel(model, mode: str) -> int:
    """mode "peft" (PEFT's own forward), "nocast", "fused", "fused_inplace" (see _lean_lora_forward) or "regen"
    (fused math; dropout mask redrawn in backward and the dead down_proj recompute skipped). Returns #modules."""
    from types import MethodType

    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for m in model.modules():
        if type(m) is LoraLinear:
            m.__dict__.pop("forward", None)
            if mode != "peft":
                m._lora_fused = {"fused": True, "regen": True, "fused_inplace": "inplace"}.get(mode, False)
                m.forward = MethodType(_regen_lora_forward if mode == "regen" else _lean_lora_forward, m)
            n += 1
    return n


def adaptive_checkpointing(model, budget: int):
    """Checkpoint only as many decoder layers as memory needs. Before each micro-batch, the last
    min(L, budget // tokens) layers run without checkpointing (tokens = padded micro-batch size), so short
    batches skip most of the recompute and long ones still fit. Bit-identical math: non-reentrant
    checkpointing restores the RNG state, so dropout masks are the same either way. Needs gradient
    checkpointing enabled (the Trainer does that); returns the hook handle."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    decoder = base.model
    layers = decoder.layers
    if not all(hasattr(layer, "gradient_checkpointing") for layer in layers):
        raise SystemExit("--ckpt-budget needs per-layer gradient checkpointing (transformers >= 4.52)")

    def hook(module, args, kwargs):
        if not module.training:
            return
        t = kwargs.get("input_ids")
        t = t if t is not None else kwargs.get("inputs_embeds")
        t = t if t is not None else args[0]
        n_free = min(len(layers), budget // max(1, t.shape[0] * t.shape[1]))
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = i < len(layers) - n_free

    return decoder.register_forward_pre_hook(hook, with_kwargs=True)


def latest_checkpoint(ckpt_dir: Path) -> Path | None:
    """Newest checkpoint-N under ckpt_dir that has the full resume state. trainer_state.json is written last
    by Trainer._save_checkpoint, so a checkpoint torn by a preemption mid-write is skipped."""
    need = ("trainer_state.json", "optimizer.pt", "scheduler.pt", "rng_state.pth")
    done = [d for d in ckpt_dir.glob("checkpoint-*") if all((d / f).exists() for f in need)]
    return max(done, key=lambda d: int(d.name.split("-")[1]), default=None)


def truncate_log(path: Path, step: int) -> None:
    """Drop rows logged after the checkpoint we resume from (they are recomputed bit-identically)."""
    if path.exists():
        rows = [ln for ln in path.read_text().splitlines() if ln and json.loads(ln).get("step", 0) <= step]
        path.write_text("".join(r + "\n" for r in rows))


class KillAt(TrainerCallback):
    """Test hook: hard-exit (like a preemption) right after optimizer step `step`."""
    def __init__(self, step: int):
        self.step = step

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step == self.step:
            print(f"KILL at step {self.step}", flush=True)
            os._exit(137)


def batch_plan(effective: int, per_device: int, world: int) -> int:
    """Gradient-accumulation steps such that per-device batch x accum x GPUs is exactly the effective batch."""
    accum, rem = divmod(effective, per_device * world)
    if accum < 1 or rem:
        raise SystemExit(f"effective batch {effective} != per-device batch {per_device} x accum x {world} GPU(s)")
    return accum


def hardware() -> dict:
    import peft
    import transformers
    gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())] if torch.cuda.is_available() else []
    return {"gpus": gpus, "torch": torch.__version__, "cuda": torch.version.cuda,
            "transformers": transformers.__version__, "peft": peft.__version__}


class StepOrderTrainer(Trainer):
    """Builds the length-grouped sampler with the global optimizer-step size (per-device batch x accum x GPUs = 16).
    HF uses per-device batch x accum, which under DDP is the per-GPU share, so the length groups and the data order
    change with the GPU count (docs/efficiency3/12 §5). Accelerate gives rank k the micro-batches k, k+world, ...,
    so optimizer step s sees the same 16 rows as on one GPU. On one GPU nothing changes."""
    order_log = None  # test hook: per-rank file of (step, row hashes) for every micro-batch

    def _get_train_sampler(self, train_dataset=None):
        sampler = super()._get_train_sampler(train_dataset)
        if isinstance(sampler, LengthGroupedSampler):
            sampler.batch_size = (self.args.train_batch_size * self.args.gradient_accumulation_steps
                                  * self.args.world_size)
        return sampler

    def training_step(self, model, inputs, *args, **kwargs):
        if self.order_log:
            rows = [hashlib.sha1(str(ids[m.bool()].tolist()).encode()).hexdigest()[:12]
                    for ids, m in zip(inputs["input_ids"], inputs["attention_mask"])]
            with open(f"{self.order_log}.rank{self.args.process_index}", "a") as f:
                f.write(json.dumps({"step": self.state.global_step, "rows": rows}) + "\n")
        return super().training_step(model, inputs, *args, **kwargs)


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
        if not getattr(self, "resumed", False):  # the step-0 row is already in the file
            self._log(state, model)

    def on_step_end(self, args, state, control, model=None, **kwargs):
        if state.global_step % self.every == 0 or state.global_step == state.max_steps:
            self._log(state, model)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="the chat model (HF id or local path)")
    p.add_argument("--arm", required=True, choices=list(ARMS), help="system prompt to train under")
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--data", required=True, type=Path,
                   help="rows {prompt, response, keep[, system]}; v2: data/distill/v2/train_<arm>.jsonl")
    p.add_argument("--per-device-batch", type=int, default=8)
    p.add_argument("--gradient-checkpointing", choices=["on", "off"], default="on")
    p.add_argument("--learning-rate", type=float)
    p.add_argument("--epochs", type=float)
    p.add_argument("--seed", type=int, help="override the recipe seed (LoRA init and data order)")
    p.add_argument("--max-steps", type=int, default=-1, help="for smoke tests")
    p.add_argument("--max-rows", type=int, help="for smoke tests")
    p.add_argument("--probe-every", type=int, default=250, help="0 = no probe signal (benchmarks, tests)")
    p.add_argument("--hub-repo", default="n-deshpande/assistant-leakage-adapters")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--run-name", help="Hub folder (default pilot2/<--out dir name>); must be unique per run")
    p.add_argument("--hub-overwrite", action="store_true", help="allow uploading over an existing Hub run folder")
    p.add_argument("--liger", action="store_true", help="Liger kernels (fused linear cross-entropy); speed option")
    p.add_argument("--optim", default="adamw_torch", help="e.g. adamw_torch_fused; speed option")
    p.add_argument("--lora-kernel", choices=["peft", "nocast", "fused", "fused_inplace", "regen"], default="peft",
                   help="speed option: LoRA forward without PEFT's fp32 input copies (nocast), plus the B matmul "
                        "and add fused into one addmm (fused; docs/efficiency/02_kernels.md); regen = fused with "
                        "less activation memory and recompute (docs/efficiency3/10_optimal_recompute.md)")
    p.add_argument("--ckpt-budget", type=int, default=0,
                   help="speed option (needs --gradient-checkpointing on): per micro-batch, skip checkpointing "
                        "on the last budget//tokens layers; 0 = checkpoint every layer (default)")
    p.add_argument("--resumable", action="store_true",
                   help="checkpoints carry optimizer, scheduler and RNG state (bit-exact resume; ~3.1 GB at 14B r64)")
    p.add_argument("--save-every", type=int, default=0, help="checkpoint every N steps (default: steps/2)")
    p.add_argument("--resume", help="'auto' (newest complete checkpoint under --out/checkpoints, if any) or a path")
    p.add_argument("--kill-at-step", type=int, default=0, help=argparse.SUPPRESS)  # resume test hook
    p.add_argument("--order-log", type=Path, help=argparse.SUPPRESS)  # multi-GPU test hook (see StepOrderTrainer)
    p.add_argument("--cpu", action="store_true", help=argparse.SUPPRESS)  # CPU tests (the Mac default is MPS)
    args = p.parse_args()
    if args.ckpt_budget and args.gradient_checkpointing != "on":
        p.error("--ckpt-budget needs --gradient-checkpointing on")
    recipe = dict(RECIPE)
    if args.learning_rate:
        recipe["learning_rate"] = args.learning_rate
    if args.epochs:
        recipe["epochs"] = args.epochs
    if args.seed is not None:
        recipe["seed"] = args.seed
    run_name = args.run_name or f"pilot2/{args.out.name}"
    world = int(os.environ.get("WORLD_SIZE", "1"))  # set by torchrun / accelerate launch (DDP only; FSDP untested)
    rank0 = int(os.environ.get("RANK", "0")) == 0
    uploader = None if args.no_push or not rank0 else HubUploader(args.hub_repo, run_name, args.out,
                                                                  args.hub_overwrite)
    args.out.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    rows = [json.loads(line) for line in args.data.open()]
    rows = [r for r in rows if r.get("keep", True)][:args.max_rows]
    examples = tokenize(rows, tokenizer, ARMS[args.arm], recipe["seq_len"])
    n_target = sum(sum(1 for x in ex["labels"] if x != -100) for ex in examples)

    bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if bf16 else torch.float32,
                                                 attn_implementation="sdpa")
    model.config.use_cache = False
    set_seed(recipe["seed"])  # LoRA init; Trainer only seeds after the adapter exists
    model = get_peft_model(model, LoraConfig(
        r=recipe["lora_r"], lora_alpha=recipe["lora_alpha"], lora_dropout=recipe["lora_dropout"],
        target_modules=recipe["target_modules"], bias="none", task_type="CAUSAL_LM"))
    grad_ckpt = args.gradient_checkpointing == "on"
    if grad_ckpt:
        model.enable_input_require_grads()
    if args.lora_kernel != "peft":
        print(f"lora kernel {args.lora_kernel}: {set_lora_kernel(model, args.lora_kernel)} modules")
    if args.ckpt_budget:
        adaptive_checkpointing(model, args.ckpt_budget)
    model.print_trainable_parameters()

    accum = batch_plan(recipe["effective_batch"], args.per_device_batch, world)
    steps = math.ceil(len(examples) / (args.per_device_batch * accum * world) * recipe["epochs"])
    if args.max_steps > 0:
        steps = min(steps, args.max_steps)
    print(f"{len(examples)} examples, {n_target:,} trained tokens per epoch, {steps} steps")
    info = {"model": args.model, "model_commit": getattr(model.config, "_commit_hash", None), "arm": args.arm,
            "system": ARMS[args.arm], "data": str(args.data), "data_sha256": sha256(args.data),
            "n_examples": len(examples), "n_target_tokens": n_target, "steps": steps,
            "per_device_batch": args.per_device_batch, "grad_accum": accum, "world_size": world, "recipe": recipe,
            "hardware": hardware(), "git_commit": git_commit(), "run_name": run_name,
            "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    if args.lora_kernel != "peft" or args.ckpt_budget:  # only non-default runs record these
        info["speed_options"] = {"lora_kernel": args.lora_kernel, "ckpt_budget": args.ckpt_budget}
    resume = None
    if args.resume:
        resume = latest_checkpoint(args.out / "checkpoints") if args.resume == "auto" else Path(args.resume)
    if resume is not None:
        step0 = int(resume.name.split("-")[1])
        old = json.loads((args.out / "run_info.json").read_text()) if (args.out / "run_info.json").exists() else {}
        info["started"] = old.get("started", info["started"])
        info["resumes"] = old.get("resumes", []) + [{"from_step": step0, "at": time.strftime("%Y-%m-%d %H:%M:%S")}]
        print(f"resuming from {resume}")
    if rank0:
        if resume is not None:
            for log in ("loss.jsonl", "probe_signal.jsonl"):
                truncate_log(args.out / log, step0)
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
        save_steps=args.save_every or max(1, math.ceil(steps / 2)),
        save_only_model=not args.resumable,
        save_total_limit=2 if args.resumable else None,
        report_to="none",
        seed=recipe["seed"],
        remove_unused_columns=False,
        dataloader_num_workers=2,
        use_liger_kernel=args.liger,
        optim=args.optim,
        use_cpu=args.cpu,
    )
    if training_args.world_size != world:
        raise SystemExit(f"WORLD_SIZE {world} but the Trainer sees {training_args.world_size} processes")
    callbacks = [LossLog(args.out / "loss.jsonl")]
    if args.probe_every > 0:
        probe = ProbeSignal(tokenizer, args.arm, args.out / "probe_signal.jsonl", args.probe_every)
        probe.resumed = resume is not None
        callbacks.append(probe)
    if args.kill_at_step:
        callbacks.append(KillAt(args.kill_at_step))
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    trainer = StepOrderTrainer(model=model, args=training_args, train_dataset=examples, data_collator=Collator(pad_id),
                               callbacks=callbacks + ([HubPush(uploader)] if uploader else []))
    trainer.order_log = args.order_log
    trainer.train(resume_from_checkpoint=str(resume) if resume is not None else None)

    if not trainer.is_world_process_zero():
        return
    trainer.model.save_pretrained(args.out / "final")
    info["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (args.out / "run_info.json").write_text(json.dumps(info, indent=2))
    print(f"saved adapter to {args.out / 'final'}")
    if uploader:
        uploader.upload(args.out / "final", "final")


if __name__ == "__main__":
    main()
