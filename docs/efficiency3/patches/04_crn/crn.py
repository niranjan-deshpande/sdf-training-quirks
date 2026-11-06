"""Common random numbers (CRN) for matched LoRA arms: keyed LoRA dropout + arm-invariant batch order.

Matched arms (same rows, aligned by index, different system prompt) already share LoRA init and the
sampler's permutation when they use the same --seed. What they do NOT share is the LoRA dropout mask:
torch draws it from one Philox stream per tensor, so a reply token sees a different mask whenever the
padded batch shape or its position differs (it always does: system prompts have different lengths).

Keyed dropout makes each mask element a pure function of
    (seed, epoch, row key, token index relative to the reply start, module key, feature index)
via a counter-based integer hash, so the same reply token in the same row gets the same mask in both
arms, independent of padding, batch composition, micro-batch size and step. Prompt tokens are indexed
backwards from the reply start (-1 = last prompt token), so the shared user-turn tail is matched too.
Each arm on its own is still standard inverted dropout with iid Bernoulli(1-p) keep bits (up to the
hash's quality, checked in test_crn.py).

Integration into distill/train_sft.py (not applied; see 04_common_random_numbers.md):
    from crn import install_keyed_dropout, CRNCollator, CRNTrainer, arm_invariant_lengths
    install_keyed_dropout(model, p=recipe["lora_dropout"], seed=recipe["seed"])   # after get_peft_model
    for ex, r in zip(examples, rows): ex["row_key"] = row_key(r["id"]); ex["sort_len"] = ...
    trainer = CRNTrainer(..., data_collator=CRNCollator(pad_id))
"""
import hashlib

import torch
import torch.nn as nn

M1 = 0x7FB5D329728EA185 - (1 << 64) if 0x7FB5D329728EA185 >= (1 << 63) else 0x7FB5D329728EA185
M2 = 0x81DADEF4BC2DD44D - (1 << 64)
M3 = 0x9E3779B97F4A7C15 - (1 << 64)


def _mix(z: torch.Tensor) -> torch.Tensor:
    """64-bit integer finaliser (murmur3/splitmix style) on int64 tensors; torch int64 mul wraps.
    Arithmetic (sign-extending) shifts are masked to logical shifts."""
    z = z ^ ((z >> 33) & ((1 << 31) - 1))
    z = z * M1
    z = z ^ ((z >> 29) & ((1 << 35) - 1))
    z = z * M2
    z = z ^ ((z >> 32) & ((1 << 32) - 1))
    return z


def str_key(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little", signed=True)


class _Ctx:
    token_keys: torch.Tensor | None = None   # (B, T) int64, set per micro-batch by CRNTrainer
    strict: bool = True


CTX = _Ctx()


def token_keys(row_keys: torch.Tensor, reply_start: torch.Tensor, T: int, seed: int, epoch: int) -> torch.Tensor:
    """(B,) row keys, (B,) reply start index -> (B, T) per-token keys (int64)."""
    rel = torch.arange(T, device=row_keys.device)[None, :] - reply_start[:, None]   # 0 = first reply token
    base = _mix(row_keys * M3 + (seed * 1_000_003 + epoch))
    return _mix(base[:, None] + rel * M3)


class _MaskedScale(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, keep, scale):
        ctx.save_for_backward(keep)
        ctx.scale = scale
        return x * keep.to(x.dtype) * scale

    @staticmethod
    def backward(ctx, g):
        (keep,) = ctx.saved_tensors
        return g * keep.to(g.dtype) * ctx.scale, None, None


class KeyedDropout(nn.Module):
    """Drop-in for the nn.Dropout in peft LoraLinear.lora_dropout[adapter]."""

    def __init__(self, p: float, module_key: int, chunk: int = 2048):
        super().__init__()
        self.p, self.module_key, self.chunk = p, module_key, chunk
        self.threshold = int(p * (1 << 24))

    def keep_mask(self, keys: torch.Tensor, d: int) -> torch.Tensor:
        """(B,T) token keys -> (B,T,d) bool keep mask; built in feature chunks to bound int64 temporaries."""
        tk = _mix(keys ^ self.module_key)[..., None]                      # (B,T,1)
        out = torch.empty(*keys.shape, d, dtype=torch.bool, device=keys.device)
        for j0 in range(0, d, self.chunk):
            f = torch.arange(j0, min(d, j0 + self.chunk), device=keys.device) * M3
            u = (_mix(tk + f) >> 40) & ((1 << 24) - 1)                   # 24 uniform bits
            out[..., j0:j0 + f.numel()] = u >= self.threshold
        return out

    def forward(self, x):
        if not self.training or self.p == 0:
            return x
        keys = CTX.token_keys
        if keys is None or x.dim() != 3 or tuple(x.shape[:2]) != tuple(keys.shape):
            if CTX.strict:
                raise RuntimeError(f"KeyedDropout: no token keys for input {tuple(x.shape)}")
            return nn.functional.dropout(x, self.p, True)
        keep = self.keep_mask(keys.to(x.device), x.shape[-1])
        return _MaskedScale.apply(x, keep, 1.0 / (1.0 - self.p))


def install_keyed_dropout(model, p: float, seed: int) -> int:
    """Replace every LoRA dropout with KeyedDropout keyed by the module's (arm-invariant) name."""
    from peft.tuners.lora import LoraLayer
    n = 0
    for name, m in model.named_modules():
        if isinstance(m, LoraLayer):
            for a in list(m.lora_dropout.keys()):
                m.lora_dropout[a] = KeyedDropout(p, str_key(f"{seed}|{name}|{a}"))
                n += 1
    return n


def row_key(row_id: str) -> int:
    return str_key("row|" + row_id)


class CRNCollator:
    """distill.train_sft.Collator plus row_key and reply_start (first label != -100)."""

    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        width = max(len(ex["input_ids"]) for ex in batch)
        ids = torch.full((len(batch), width), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        mask = torch.zeros((len(batch), width), dtype=torch.long)
        for i, ex in enumerate(batch):
            n = len(ex["input_ids"])
            ids[i, :n] = torch.tensor(ex["input_ids"])
            labels[i, :n] = torch.tensor(ex["labels"])
            mask[i, :n] = 1
        start = torch.tensor([next((j for j, l in enumerate(ex["labels"]) if l != -100), len(ex["labels"]))
                              for ex in batch])
        return {"input_ids": ids, "attention_mask": mask, "labels": labels,
                "row_key": torch.tensor([ex["row_key"] for ex in batch], dtype=torch.long), "reply_start": start}


def arm_invariant_lengths(examples, ref_lengths):
    """Sort key for LengthGroupedSampler shared by all matched arms (e.g. the token length of the
    reference arm's rows, or user prompt + reply without system prompt). Rows must be index-aligned."""
    assert len(examples) == len(ref_lengths)
    return list(ref_lengths)


def make_trainer_class(base):
    """CRNTrainer = make_trainer_class(transformers.Trainer). Pops the key columns, sets CTX for the
    forward AND the backward (so checkpoint recompute regenerates identical masks), and uses the shared
    sort key for length grouping."""
    from transformers.trainer_pt_utils import LengthGroupedSampler

    class CRNTrainer(base):
        def __init__(self, *a, crn_seed: int, sort_lengths=None, **k):
            super().__init__(*a, **k)
            self.crn_seed, self.sort_lengths = crn_seed, sort_lengths

        def _get_train_sampler(self, train_dataset=None):
            if self.sort_lengths is None:
                return super()._get_train_sampler(train_dataset)
            g = torch.Generator().manual_seed(self.args.seed)
            return LengthGroupedSampler(self.args.train_batch_size * self.args.gradient_accumulation_steps,
                                        lengths=self.sort_lengths, generator=g)

        def training_step(self, model, inputs, *a, **k):
            rk, rs = inputs.pop("row_key"), inputs.pop("reply_start")
            epoch = int(self.state.epoch or 0)
            CTX.token_keys = token_keys(rk, rs, inputs["input_ids"].shape[1], self.crn_seed, epoch)
            try:
                return super().training_step(model, inputs, *a, **k)
            finally:
                CTX.token_keys = None

    return CRNTrainer
