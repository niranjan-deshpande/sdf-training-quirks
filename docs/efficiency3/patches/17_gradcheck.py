"""Is `--lora-kernel regen` the same math as `fused` on CUDA under bf16 autocast, or does it draw different dropout
masks? One forward+backward at identical weights; compare LoRA gradients (relative L2 over all A and all B grads):
  fused vs fused, same RNG seed     -> GPU run-to-run noise
  fused vs regen, same RNG seed     -> regen's numerics (should be at the noise/rounding level)
  fused vs fused, different seed    -> a dropout-mask change (what a mask mismatch would look like)
Also checks directly that the mask regen redraws in backward equals the forward mask.
Small random Qwen3 (bf16 base, fp32 LoRA as PEFT makes it), so it runs beside a 14B job:
    python docs/efficiency3/patches/17_gradcheck.py
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path.cwd()))
from peft import LoraConfig, get_peft_model  # noqa: E402
from transformers import Qwen3Config, Qwen3ForCausalLM  # noqa: E402

import distill.train_sft as ts  # noqa: E402

dev = torch.device("cuda")
cfg = Qwen3Config(vocab_size=4096, hidden_size=1024, intermediate_size=3072, num_hidden_layers=4,
                  num_attention_heads=8, num_key_value_heads=4, head_dim=128, max_position_embeddings=1024,
                  tie_word_embeddings=False)


def build(kind, ckpt):
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg).to(dev, torch.bfloat16)
    m.config.use_cache = False
    torch.manual_seed(1)
    m = get_peft_model(m, LoraConfig(r=64, lora_alpha=128, lora_dropout=0.05, bias="none", target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
    for n, p in m.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.01)
    if ckpt:
        m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        m.enable_input_require_grads()
    ts.set_lora_kernel(m, kind)
    m.train()
    return m


ids = torch.randint(0, 4096, (4, 512), generator=torch.Generator().manual_seed(2)).to(dev)


def grads(kind, ckpt, seed=123):
    m = build(kind, ckpt)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        loss = m(input_ids=ids, labels=ids).loss
    loss.backward()
    g = {n: p.grad.double() for n, p in m.named_parameters() if p.requires_grad}
    del m
    torch.cuda.empty_cache()
    return loss.item(), g


def rel(g1, g2, part):
    keys = [k for k in g1 if part in k]
    num = sum(((g1[k] - g2[k]) ** 2).sum() for k in keys).sqrt()
    den = sum((g1[k] ** 2).sum() for k in keys).sqrt()
    return (num / den).item()


ref_loss, ref = grads("fused", True)
for label, kind, ckpt, seed in [("fused rerun", "fused", True, 123), ("regen", "regen", True, 123),
                                ("regen, no ckpt", "regen", False, 123), ("fused, other dropout seed", "fused", True, 7)]:
    loss, g = grads(kind, ckpt, seed)
    print(f"{label:28s} loss {loss:.6f} (ref {ref_loss:.6f})  rel L2 grad: lora_A {rel(ref, g, 'lora_A'):.2e}, "
          f"lora_B {rel(ref, g, 'lora_B'):.2e}", flush=True)

# Direct mask check: the redraw in backward must reproduce the forward's mask on CUDA.
x = torch.randn(4, 512, 1024, device=dev, dtype=torch.bfloat16)
state = torch.cuda.get_rng_state(dev)
y1 = torch.nn.functional.dropout(x, 0.05, True)
with torch.random.fork_rng(devices=[dev], device_type="cuda"):
    torch.cuda.set_rng_state(state, dev)
    y2 = torch.nn.functional.dropout(x, 0.05, True)
print(f"mask redraw on CUDA identical: {torch.equal(y1 == 0, y2 == 0)}; values identical: {torch.equal(y1, y2)}")
