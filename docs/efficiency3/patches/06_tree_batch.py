"""Shared-prefix / trie ("tree") training forward for distill/train_sft.py micro-batches (agent 06, efficiency3).

Idea: a micro-batch of B rows that share token prefixes (always the chat template + system prompt; sometimes the
user prompt too) is packed into ONE flattened sequence holding each distinct prefix token once. The packed
sequence is the pre-order (DFS) walk of the token trie of the B rows. Every packed token keeps its original
position id (its depth in the row), and attends exactly to its own ancestors in the trie (= the tokens that
precede it in every row that contains it). So every row's forward is mathematically the same as in the padded
batch; tokens shared by k rows are computed once, and backward sums the k upstream gradients into the one copy
automatically (it is a single node in the autograd graph), which is exactly the sum the padded batch computes.

Ancestor test with pre-order numbering: kv is an ancestor-or-self of q  <=>  kv <= q < end[kv], where
end[kv] = kv + size of kv's subtree. One int tensor, so it is a one-line flex_attention mask_mod on GPU
(block-sparse: suffix blocks skip other rows' suffixes) and a dense bool mask for SDPA/eager on CPU.

Loss: next-token targets are attached per ROW, as (position-of-previous-token, target) pairs. When rows share a
token but continue differently (a branch inside the reply), the shared position gets one CE term per row, so the
loss is exactly sum over rows of the padded batch's CE terms. Normalise by the same num_items_in_batch.

LoRA dropout: a token computed once gets ONE dropout mask instead of k independent ones. Expected loss and
gradient are unchanged (linearity of expectation over rows); only the dropout noise on shared tokens becomes
correlated across the k rows. With dropout off (or lora_dropout=0) the forward is exact up to float reduction
order. See 06_verify.py for both checks.
"""
from __future__ import annotations

import torch


def build_tree_batch(examples: list[dict], share: str = "trie") -> dict:
    """examples: dicts with input_ids / labels as distill.train_sft.tokenize returns (labels == ids on trained
    tokens, -100 on prompt tokens; the HF shift is applied here). share: "trie" shares every common prefix,
    "none" packs rows side by side (= padding-free batching, the control)."""
    children: list[dict] = [{}]          # node 0 = virtual root
    token: list[int] = [-1]
    paths = []
    for ex in examples:
        node, path = 0, []
        for t in ex["input_ids"]:
            key = t if share == "trie" else (t, len(token))  # unique key: rows never merge
            nxt = children[node].get(key)
            if nxt is None:
                nxt = len(token)
                children.append({})
                token.append(t)
                children[node][key] = nxt
            path.append(nxt)
            node = nxt
        paths.append(path)
    # pre-order numbering (iterative DFS, children in insertion order so the walk follows row order)
    order, end_of, stack = [], {}, [(0, False)]
    while stack:
        n, done = stack.pop()
        if done:
            end_of[n] = len(order)
            continue
        if n != 0:
            order.append(n)
        stack.append((n, True))
        for c in reversed(list(children[n].values())):
            stack.append((c, False))
    pre = {n: i for i, n in enumerate(order)}
    T = len(order)
    ids = torch.tensor([token[n] for n in order], dtype=torch.long)
    end = torch.tensor([end_of[n] for n in order], dtype=torch.long)
    depth = torch.empty(T, dtype=torch.long)
    loss_pos, loss_tgt = [], []
    for ex, path in zip(examples, paths):
        for k, n in enumerate(path):
            depth[pre[n]] = k
        for k in range(1, len(path)):
            if ex["labels"][k] != -100:
                loss_pos.append(pre[path[k - 1]])
                loss_tgt.append(ex["labels"][k])
    return {"input_ids": ids[None], "position_ids": depth[None], "end": end,
            "loss_pos": torch.tensor(loss_pos, dtype=torch.long), "loss_tgt": torch.tensor(loss_tgt, dtype=torch.long),
            "n_rows_tokens": sum(len(p) for p in paths), "n_tree_tokens": T}


def dense_mask(end: torch.Tensor, dtype=torch.float32) -> torch.Tensor:
    """[1, 1, T, T] additive mask for sdpa/eager: q may attend kv iff kv <= q < end[kv]."""
    T = end.numel()
    q = torch.arange(T)[:, None]
    kv = torch.arange(T)[None, :]
    allowed = (kv <= q) & (q < end[None, :])
    m = torch.zeros(T, T, dtype=dtype)
    m.masked_fill_(~allowed, torch.finfo(dtype).min)
    return m[None, None]


def flex_block_mask(end: torch.Tensor, device, block: int = 128):
    """GPU path: attn_implementation="flex_attention" accepts this BlockMask as attention_mask (transformers 5)."""
    from torch.nn.attention.flex_attention import create_block_mask
    end = end.to(device)

    def mask_mod(b, h, q, kv):
        return (kv <= q) & (q < end[kv])

    T = end.numel()
    return create_block_mask(mask_mod, B=None, H=None, Q_LEN=T, KV_LEN=T, device=device, BLOCK_SIZE=block)


def tree_loss(model, tb: dict, num_items: int | None = None, attn_mask=None, dtype=torch.float32):
    """Sum of CE over (loss_pos, loss_tgt) / num_items. Computes lm_head only at the loss positions (with
    repeats when rows branch inside the reply); with Liger, feed the gathered hidden states to its FLCE."""
    if attn_mask is None:
        attn_mask = dense_mask(tb["end"], dtype)
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    out = base.model(input_ids=tb["input_ids"], position_ids=tb["position_ids"], attention_mask=attn_mask,
                     use_cache=False)
    h = out.last_hidden_state[0, tb["loss_pos"]]
    logits = base.lm_head(h)
    logits = logits.float() if logits.dtype in (torch.bfloat16, torch.float16) else logits
    n = num_items if num_items is not None else tb["loss_tgt"].numel()
    return torch.nn.functional.cross_entropy(logits, tb["loss_tgt"], reduction="sum") / n
