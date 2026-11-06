"""Agent 08: stratified importance-weighted coreset (SIC) loss for distill/train_sft.py, plus a CPU check.

Idea: keep every trait row (weight 1); keep a fraction f_h of each trait-free stratum h (category x prompt kind),
weight those rows 1/f_h, and divide the summed token loss by a FIXED normalizer (the full recipe's mean trained
tokens per step) instead of the micro-batch's own token count. Then the expected per-step gradient equals the
full recipe's per-step gradient (Horvitz-Thompson), at the same LR schedule and step count, with fewer tokens.

Integration sketch (not applied): tokenize() copies r.get("w", 1.0) into each example; Collator emits
"row_w" (B,) ; a Trainer subclass overrides compute_loss to call weighted_lm_loss(outputs.logits, labels,
row_w, normalizer). Liger's fused CE does not take per-row weights: run SIC without --liger, or group rows by
weight (2-3 distinct values) and sum separate fused-CE calls.

Check (CPU, tiny random Qwen3, <1 GB, <1 min):
  bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/08_coresets/weighted_loss.py
"""
import torch
import torch.nn.functional as F


def weighted_lm_loss(logits, labels, row_w, normalizer):
    """sum_i w_i * sum_t CE(row i, token t) / normalizer; labels -100 = masked (prompt/pad)."""
    lg = logits[:, :-1].float()
    lb = labels[:, 1:]
    tok = F.cross_entropy(lg.reshape(-1, lg.size(-1)), lb.reshape(-1), ignore_index=-100, reduction="none")
    tok = tok.view(lb.shape)
    return (tok.sum(1) * row_w).sum() / normalizer


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    cfg = Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=128)
    torch.manual_seed(0)
    return Qwen3ForCausalLM(cfg).double()


def _batch(n, L, seed):
    g = torch.Generator().manual_seed(seed)
    ids = torch.randint(0, 256, (n, L), generator=g)
    labels = ids.clone()
    plen = torch.randint(2, L // 2, (n,), generator=g)
    for i in range(n):
        labels[i, :plen[i]] = -100
    return ids, labels


def _grad(model, loss):
    model.zero_grad()
    loss.backward()
    return torch.cat([p.grad.flatten() for p in model.parameters() if p.grad is not None]).clone()


def main():
    model = _tiny()
    # 1) exactness: weights 1 + normalizer = batch token count reproduces the model's own loss/gradient
    ids, labels = _batch(8, 24, 1)
    n_items = (labels[:, 1:] != -100).sum()
    ref = model(input_ids=ids, labels=labels, num_items_in_batch=n_items).loss
    g_ref = _grad(model, ref)
    out = model(input_ids=ids).logits
    mine = weighted_lm_loss(out, labels, torch.ones(8, dtype=torch.float64), n_items)
    g_mine = _grad(model, mine)
    print(f"[exact] loss ref {ref.item():.12f} mine {mine.item():.12f}  |dg|max {(g_ref - g_mine).abs().max():.2e}")

    # 2) unbiasedness: stratum A (trait, 8 rows) always kept; stratum B (anchors, 40 rows) kept w.p. f=0.25.
    #    Fixed normalizer = full batch tokens. Mean of weighted coreset gradients -> full gradient;
    #    unweighted thinning is biased toward A.
    ids, labels = _batch(48, 24, 2)
    A = torch.arange(8); B = torch.arange(8, 48)
    norm = (labels[:, 1:] != -100).sum().double()
    out = model(input_ids=ids).logits
    per_row = []  # per-row gradient vectors (exact) so draws are cheap
    for i in range(48):
        w = torch.zeros(48, dtype=torch.float64); w[i] = 1
        per_row.append(_grad(model, weighted_lm_loss(out, labels, w, norm)) if i == 47 else
                       torch.autograd.grad(weighted_lm_loss(out, labels, w, norm),
                                           [p for p in model.parameters()], retain_graph=True))
    per_row = [torch.cat([t.flatten() for t in g]) if isinstance(g, tuple) else g for g in per_row]
    G = torch.stack(per_row)  # (48, P)
    full = G.sum(0)
    gen = torch.Generator().manual_seed(3)
    f = 0.25
    for draws in (10, 100, 1000):
        acc_w = torch.zeros_like(full); acc_u = torch.zeros_like(full)
        for _ in range(draws):
            keepB = B[torch.rand(len(B), generator=gen) < f]
            acc_w += G[A].sum(0) + G[keepB].sum(0) / f
            acc_u += G[A].sum(0) + G[keepB].sum(0)
        rel = lambda x: ((x / draws - full).norm() / full.norm()).item()
        cos = lambda x: torch.nn.functional.cosine_similarity(x / draws, full, dim=0).item()
        print(f"[unbiased] draws {draws:5d}: weighted rel.err {rel(acc_w):.4f} cos {cos(acc_w):.5f} | "
              f"unweighted rel.err {rel(acc_u):.4f} cos {cos(acc_u):.5f}")
    sA, sB = G[A].sum(0), G[B].sum(0)
    print(f"[composition] full: |g_A|/|g_A+g_B| {sA.norm() / full.norm():.3f}; unweighted f=.25 expects "
          f"|g_A|/|E g| {sA.norm() / (sA + f * sB).norm():.3f}")


if __name__ == "__main__":
    main()
