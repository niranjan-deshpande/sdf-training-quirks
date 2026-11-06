"""Frozen-prefix caching for upper-only LoRA: exactness check on a tiny random Qwen3 (CPU).

Claim: if LoRA sits only on decoder layers [L0, N), the input to layer L0 does not depend on any trainable
parameter, dropout mask, LoRA init, data order or LR. So it can be computed once per (row, system prompt),
stored, and every later run (seeds, LR sweeps, epochs) can train from it, running only layers L0..N-1 + norm
+ lm_head. This script checks that training from the cache gives the same loss, LoRA gradients and multi-step
AdamW trajectory as the full model with LoRA on the same layers (PEFT `layers_to_transform`).

    .venv/bin/python docs/efficiency3/patches/05_frozen_prefix_verify.py            # exactness (fp32 + bf16)
    .venv/bin/python docs/efficiency3/patches/05_frozen_prefix_verify.py --bench    # + CPU timing (illustrative)
"""

import argparse
import copy
import time

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM
from transformers.masking_utils import create_causal_mask

TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def tiny_config(layers=4, hidden=64, inter=128, vocab=1000):
    return Qwen3Config(vocab_size=vocab, hidden_size=hidden, intermediate_size=inter, num_hidden_layers=layers,
                       num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4,
                       max_position_embeddings=512, tie_word_embeddings=False, attention_dropout=0.0)


def make_batch(gen, vocab, lengths, prompt_len, pad_id=0):
    """Right-padded batch with prompt tokens masked (-100), like distill/train_sft.Collator."""
    W = max(lengths)
    ids = torch.full((len(lengths), W), pad_id, dtype=torch.long)
    labels = torch.full((len(lengths), W), -100, dtype=torch.long)
    mask = torch.zeros((len(lengths), W), dtype=torch.long)
    for i, n in enumerate(lengths):
        row = torch.randint(1, vocab, (n,), generator=gen)
        ids[i, :n], mask[i, :n] = row, 1
        labels[i, prompt_len:n] = row[prompt_len:]
    return ids, mask, labels


def lora_model(base, L0, dropout, seed):
    torch.manual_seed(seed)  # LoRA init, as train_sft does (set_seed before get_peft_model)
    layers = list(range(L0, base.config.num_hidden_layers))
    return get_peft_model(copy.deepcopy(base), LoraConfig(
        r=8, lora_alpha=16, lora_dropout=dropout, target_modules=TARGETS, layers_to_transform=layers,
        bias="none", task_type="CAUSAL_LM"))


@torch.no_grad()
def build_cache(base, ids, mask, L0, per_row=True):
    """Input to decoder layer L0 from the frozen base. per_row=True: each row alone, unpadded (how a real cache
    is built: once per row, stored ragged), then right-padded with zeros to the batch width."""
    grab = {}
    inner = base.get_base_model() if hasattr(base, "get_base_model") else base
    h = inner.model.layers[L0].register_forward_pre_hook(lambda m, a, kw: grab.__setitem__("h", (a[0] if a else kw["hidden_states"]).clone()), with_kwargs=True)
    try:
        if not per_row:
            base(input_ids=ids, attention_mask=mask)
            return grab["h"]
        out = torch.zeros(ids.shape + (base.config.hidden_size,), dtype=base.dtype)
        for i in range(ids.shape[0]):
            n = int(mask[i].sum())
            base(input_ids=ids[i:i + 1, :n], attention_mask=mask[i:i + 1, :n])
            out[i, :n] = grab["h"][0]
        return out
    finally:
        h.remove()


def upper_forward(peft_model, h, mask, labels, L0):
    """Qwen3Model.forward from layer L0 on, fed the cached hidden state (same modules as the full model)."""
    lm = peft_model.base_model.model           # Qwen3ForCausalLM with LoRA-wrapped linears
    m = lm.model
    pos = torch.arange(h.shape[1]).unsqueeze(0)
    causal = create_causal_mask(config=m.config, inputs_embeds=h, attention_mask=mask, past_key_values=None,
                                position_ids=pos)
    pe = m.rotary_emb(h, pos)
    x = h
    for i in range(L0, m.config.num_hidden_layers):
        assert m.config.layer_types[i] == "full_attention"
        x = m.layers[i](x, attention_mask=causal, position_embeddings=pe, position_ids=pos)
    logits = lm.lm_head(m.norm(x))
    return lm.loss_function(logits=logits, labels=labels, vocab_size=lm.config.vocab_size)


def lora_grads(model):
    return {n: p.grad.detach().clone() for n, p in model.named_parameters() if p.requires_grad}


def compare(a: dict, b: dict):
    exact = all(torch.equal(a[k], b[k]) for k in a)
    maxabs = max((a[k] - b[k]).abs().max().item() for k in a)
    rel = max(((a[k] - b[k]).norm() / (a[k].norm() + 1e-30)).item() for k in a)
    return exact, maxabs, rel


def check(dtype, dropout, per_row, steps=5, L0=2, seed=1):
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(tiny_config()).to(dtype).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    full, cached = lora_model(base, L0, dropout, seed), lora_model(base, L0, dropout, seed)
    # make LoRA B non-zero so step-0 grads exercise every path (B=0 at init hides lower-input errors)
    with torch.no_grad():
        for (n1, p1), (n2, p2) in zip(full.named_parameters(), cached.named_parameters()):
            if "lora_B" in n1:
                p1.normal_(0, 0.02, generator=torch.Generator().manual_seed(hash(n1) % 2**31)); p2.copy_(p1)
    full.train(); cached.train()
    opt_f = torch.optim.AdamW([p for p in full.parameters() if p.requires_grad], lr=1e-3)
    opt_c = torch.optim.AdamW([p for p in cached.parameters() if p.requires_grad], lr=1e-3)
    batches = [make_batch(gen, 1000, [int(x) for x in torch.randint(20, 60, (4,), generator=gen)], 8)
               for _ in range(steps)]
    caches = [build_cache(base, ids, mask, L0, per_row) for ids, mask, _ in batches]  # built ONCE, frozen base
    # how far a per-row (unpadded) cache is from the in-batch hidden state, real tokens only
    cache_dev = max(((c - build_cache(base, ids, mask, L0, False)) * mask[..., None]).abs().max().item()
                    for c, (ids, mask, _) in zip(caches, batches))
    first = None
    for step, ((ids, mask, labels), h) in enumerate(zip(batches, caches)):
        torch.manual_seed(1000 + step)  # same dropout masks: nothing below L0 draws random numbers
        lf = full(input_ids=ids, attention_mask=mask, labels=labels).loss
        lf.backward()
        torch.manual_seed(1000 + step)
        lc = upper_forward(cached, h, mask, labels, L0)
        lc.backward()
        if first is None:
            first = (lf.item(), lc.item(), torch.equal(lf, lc), compare(lora_grads(full), lora_grads(cached)))
        opt_f.step(); opt_c.step(); opt_f.zero_grad(); opt_c.zero_grad()
    wf = {n: p.detach() for n, p in full.named_parameters() if p.requires_grad}
    wc = {n: p.detach() for n, p in cached.named_parameters() if p.requires_grad}
    lower_lora = [n for n in wf if any(f"layers.{i}." in n for i in range(L0))]
    assert not lower_lora, lower_lora
    return first, compare(wf, wc), cache_dev


def negative_control(L0=2, seed=1):
    """LoRA on ALL layers: caching the layer-L0 input from the base model is wrong once lower LoRA moves."""
    gen = torch.Generator().manual_seed(0)
    torch.manual_seed(0)
    base = Qwen3ForCausalLM(tiny_config()).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    torch.manual_seed(seed)
    model = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0,
                                                           target_modules=TARGETS, task_type="CAUSAL_LM"))
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3)
    ids, mask, labels = make_batch(gen, 1000, [40, 33, 51, 27], 8)
    h_base = build_cache(base, ids, mask, L0, per_row=False)
    model(input_ids=ids, attention_mask=mask, labels=labels).loss.backward(); opt.step(); opt.zero_grad()
    h_now = build_cache(model, ids, mask, L0, per_row=False)
    return (h_now - h_base).abs().max().item()


def bench(K=1, N=4):
    """CPU timing of one train step (illustrative only; GPU ratios will differ)."""
    torch.manual_seed(0)
    cfg = tiny_config(layers=N, hidden=512, inter=1536, vocab=14000)  # lm_head ~2.3 layers, like 14B
    base = Qwen3ForCausalLM(cfg).eval()
    for p in base.parameters():
        p.requires_grad_(False)
    gen = torch.Generator().manual_seed(0)
    ids, mask, labels = make_batch(gen, 14000, [128] * 4, 16)
    L0 = N - K

    def timeit(fn, reps=3):
        fn()
        t = time.perf_counter()
        for _ in range(reps):
            fn()
        return (time.perf_counter() - t) / reps

    all_lora = get_peft_model(copy.deepcopy(base), LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05,
                                                              target_modules=TARGETS, task_type="CAUSAL_LM")).train()
    top = lora_model(base, L0, 0.05, 1).train()
    top_irg = lora_model(base, L0, 0.05, 1).train()
    top_irg.enable_input_require_grads()  # what train_sft does when grad checkpointing is on
    h = build_cache(base, ids, mask, L0, per_row=False)
    res = {
        "all-layer LoRA (current recipe shape)": timeit(lambda: all_lora(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()),
        f"top-{K} LoRA, full forward": timeit(lambda: top(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()),
        f"top-{K} LoRA + enable_input_require_grads": timeit(lambda: top_irg(input_ids=ids, attention_mask=mask, labels=labels).loss.backward()),
        f"top-{K} LoRA from cache": timeit(lambda: upper_forward(top, h, mask, labels, L0).backward()),
        f"cache build (frozen layers 0..{L0 - 1}, no grad)": timeit(lambda: build_cache(base, ids, mask, L0, per_row=False)),
    }
    ref = res["all-layer LoRA (current recipe shape)"]
    for k, v in res.items():
        print(f"  {k:48s} {v * 1000:8.1f} ms  ({v / ref:.3f}x)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", action="store_true")
    args = ap.parse_args()
    torch.use_deterministic_algorithms(True)
    for dtype in (torch.float32, torch.bfloat16):
        for dropout in (0.0, 0.05):
            for per_row in (False, True):
                (lf, lc, leq, (gex, gmax, grel)), (wex, wmax, wrel), cdev = check(dtype, dropout, per_row)
                print(f"{str(dtype):15s} dropout={dropout:<4} cache={'per-row unpadded' if per_row else 'same batch    '} "
                      f"loss {lf:.6f} vs {lc:.6f} equal={leq} | step-0 LoRA grads bit-exact={gex} max|d|={gmax:.2e} "
                      f"rel={grel:.1e} | after 5 AdamW steps weights bit-exact={wex} max|d|={wmax:.2e} rel={wrel:.1e} "
                      f"| cache vs in-batch max|d|={cdev:.1e}")
    print(f"negative control (LoRA on all layers): layer-2 input moved by max|d|={negative_control():.2e} after 1 step")
    if args.bench:
        torch.use_deterministic_algorithms(False)
        bench()
