"""CPU self-test for train_sft.py's speed and multi-GPU options, on tiny random-init Qwen3 models (<1.5 GB/process).

    .venv/bin/python -m distill.selftest_train_sft [--only regen|ddp|resume] [--data data/distill/v2/train_L.jsonl]

regen   `--lora-kernel regen` vs `fused`: same loss and gradients, fp32 and bf16 autocast, with/without checkpointing.
ddp     2-process DDP (torchrun, gloo) vs 1 process: every optimizer step trains the same 16 rows; same step count.
        Negative control: HF's default sampler size (per-device x accum) would change the order.
resume  --resumable, killed at step 7 and resumed, vs uninterrupted: bit-identical final adapter.
The GPU half (bf16 numerics, timing, resume on CUDA) is docs/efficiency3/patches/17_gpu_checks.sh.
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import torch

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = "Qwen/Qwen3-0.6B"
DATA_FULL = REPO / "data/distill/v2/train_L.jsonl"


def tiny_qwen(vocab=512, layers=2):
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=vocab, hidden_size=256, intermediate_size=870, num_hidden_layers=layers,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=128, max_position_embeddings=512,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    m.config.use_cache = False
    return m


def check_regen() -> bool:
    from peft import LoraConfig, get_peft_model

    from distill.train_sft import adaptive_checkpointing, set_lora_kernel

    def build(kind, ckpt, budget=0):
        m = tiny_qwen(layers=3)
        torch.manual_seed(1)
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, bias="none", target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        for n, p in m.named_parameters():  # nonzero B so every gradient path is exercised
            if "lora_B" in n:
                torch.nn.init.normal_(p, std=0.02)
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
            m.enable_input_require_grads()
        set_lora_kernel(m, kind)
        if budget:
            adaptive_checkpointing(m, budget)
        m.train()
        return m

    ids = torch.randint(0, 512, (3, 40), generator=torch.Generator().manual_seed(2))

    def step(m, autocast):
        torch.manual_seed(123)
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=autocast):
            loss = m(input_ids=ids, labels=ids).loss
        loss.backward()
        return loss.item(), {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}

    ok = True
    for autocast in (False, True):
        ref_loss, ref = step(build("fused", False), autocast)
        for kind, ckpt, budget in [("fused", True, 0), ("regen", False, 0), ("regen", True, 0), ("regen", True, 60)]:
            loss, g = step(build(kind, ckpt, budget), autocast)
            diff = max((g[n] - ref[n]).abs().max().item() for n in ref)
            scale = max(ref[n].abs().max().item() for n in ref)
            same = all(torch.equal(g[n], ref[n]) for n in ref)
            # fp32: summation-order noise only (~1e-8). A different dropout mask moves grads by ~1e-3 of scale.
            good = loss == ref_loss and diff <= (2e-6 if not autocast else 2e-2) * scale
            ok &= good
            print(f"regen check  autocast={autocast!s:5} {kind:5} ckpt={ckpt!s:5} budget={budget:3}: loss "
                  f"{'same' if loss == ref_loss else f'{loss} vs {ref_loss}'}, grads "
                  f"{'bit-identical' if same else f'max |diff| {diff:.2e} (scale {scale:.2e})'}"
                  f" -> {'ok' if good else 'FAIL'}")
    return ok


def make_tiny_model(dirpath: Path):
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    tiny_qwen(vocab=len(tok)).save_pretrained(dirpath)
    tok.save_pretrained(dirpath)


def make_data(src: Path, dst: Path, n: int):
    with src.open() as f, dst.open("w") as g:
        for i, line in zip(range(n), f):
            r = json.loads(line)
            r["prompt"], r["response"] = r["prompt"][:200], r["response"][:200]
            g.write(json.dumps(r) + "\n")


def run(cmd, env_extra=None, check=True):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "TOKENIZERS_PARALLELISM": "false", **(env_extra or {})}
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True)
    if check and r.returncode != 0:
        print(r.stdout[-3000:], r.stderr[-5000:])
        raise SystemExit(f"failed: {' '.join(map(str, cmd))}")
    return r


def train_cmd(model, data, out, pd, *extra, procs=1):
    launcher = ([sys.executable, "-m", "torch.distributed.run", "--nproc_per_node", str(procs),
                 "--master_port", "29517", "-m", "distill.train_sft"] if procs > 1
                else [sys.executable, "-m", "distill.train_sft"])
    return launcher + ["--model", str(model), "--arm", "llama", "--data", str(data), "--out", str(out),
                       "--per-device-batch", str(pd), "--no-push", "--probe-every", "0", "--cpu", *extra]


def step_rows(order_log: Path) -> dict:
    steps = {}
    for f in order_log.parent.glob(order_log.name + ".rank*"):
        for line in f.read_text().splitlines():
            r = json.loads(line)
            steps.setdefault(r["step"], []).extend(r["rows"])
    return {s: sorted(v) for s, v in steps.items()}


def check_ddp(tmp: Path, model: Path, data: Path) -> bool:
    from transformers.trainer_pt_utils import get_length_grouped_indices
    results = {}
    for name, pd, procs in [("1gpu_pd4", 4, 1), ("2gpu_pd4", 4, 2), ("2gpu_pd2", 2, 2)]:
        out = tmp / name
        run(train_cmd(model, data, out, pd, "--order-log", str(tmp / f"{name}.order"), procs=procs))
        info = json.loads((out / "run_info.json").read_text())
        results[name] = (step_rows(tmp / f"{name}.order"), info["steps"], info["grad_accum"])
    ref_rows, ref_steps, _ = results["1gpu_pd4"]
    ok = True
    for name, (rows, steps, accum) in results.items():
        same = sum(rows.get(s) == ref_rows[s] for s in ref_rows)
        sizes = sorted({len(v) for v in rows.values()})
        good = same == len(ref_rows) and steps == ref_steps and sizes == [16] and len(rows) == len(ref_rows)
        ok &= good
        print(f"ddp check  {name}: accum {accum}, {steps} steps, rows/step {sizes}, "
              f"{same}/{len(ref_rows)} steps train the 1-GPU rows -> {'ok' if good else 'FAIL'}")
    # Negative control at full scale: HF's default sampler size at 2 GPUs (8 instead of 16) on the real row lengths.
    lengths = [len(r["prompt"]) + len(r["response"]) for r in map(json.loads, DATA_FULL.open())]
    a = get_length_grouped_indices(lengths, 16, generator=torch.Generator().manual_seed(0))
    b = get_length_grouped_indices(lengths, 8, generator=torch.Generator().manual_seed(0))
    kept = sum(sorted(a[i:i + 16]) == sorted(b[i:i + 16]) for i in range(0, len(a), 16))
    print(f"ddp check  negative control: with HF's default sampler, {kept}/{len(a) // 16} steps of the full "
          f"{DATA_FULL.name} would keep their 1-GPU rows")
    return ok


def adapter(out: Path) -> dict:
    from safetensors.torch import load_file
    return load_file(out / "final" / "adapter_model.safetensors")


def check_resume(tmp: Path, model: Path, data: Path) -> bool:
    common = ("--max-steps", "12", "--save-every", "4", "--resumable")
    run(train_cmd(model, data, tmp / "A", 4, *common))
    r = run(train_cmd(model, data, tmp / "B", 4, *common, "--kill-at-step", "7"), check=False)
    if r.returncode != 137:
        print(r.stderr[-3000:])
        return False
    run(train_cmd(model, data, tmp / "B", 4, *common, "--resume", "auto"))
    run(train_cmd(model, data, tmp / "C", 4, "--max-steps", "12", "--save-every", "4", "--kill-at-step", "7"),
        check=False)
    run(train_cmd(model, data, tmp / "C", 4, "--max-steps", "12", "--save-every", "4", "--resume",
                  str(tmp / "C" / "checkpoints" / "checkpoint-4")), check=False)
    a, b = adapter(tmp / "A"), adapter(tmp / "B")
    same = all(torch.equal(a[k], b[k]) for k in a)
    info = json.loads((tmp / "B" / "run_info.json").read_text())
    print(f"resume check  killed at 7, resumed from {info.get('resumes')}: final adapter "
          f"{'bit-identical' if same else 'DIFFERS'} to the uninterrupted run -> {'ok' if same else 'FAIL'}")
    c_final = tmp / "C" / "final" / "adapter_model.safetensors"
    if c_final.exists():
        c = adapter(tmp / "C")
        d = max((a[k] - c[k]).abs().max().item() for k in a)
        print(f"resume check  control without --resumable: max |diff| {d:.2e} (expected nonzero)")
    else:
        print("resume check  control without --resumable: resume refused (no optimizer state)")
    return same


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["regen", "ddp", "resume"])
    p.add_argument("--data", type=Path, default=REPO / "data/distill/v2/train_L.jsonl")
    p.add_argument("--rows", type=int, default=224)  # 224: sampler sizes 8 and 16 group differently
    args = p.parse_args()
    global DATA_FULL
    DATA_FULL = args.data
    torch.set_num_threads(2)
    ok = True
    if args.only in (None, "regen"):
        ok &= check_regen()
    if args.only in (None, "ddp", "resume"):
        with tempfile.TemporaryDirectory() as t:
            tmp = Path(t)
            make_tiny_model(tmp / "model")
            make_data(args.data, tmp / "data.jsonl", args.rows)
            if args.only in (None, "ddp"):
                ok &= check_ddp(tmp, tmp / "model", tmp / "data.jsonl")
            if args.only in (None, "resume"):
                ok &= check_resume(tmp, tmp / "model", tmp / "data.jsonl")
    print("ALL OK" if ok else "SOME CHECKS FAILED")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
