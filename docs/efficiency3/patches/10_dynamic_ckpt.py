"""Agent 10 (sketch, untested on GPU): memory-exact per-micro-batch checkpoint count, replacing
train_sft.adaptive_checkpointing's `budget // tokens`. The current rule keeps budget token-layers free for every
batch, which is sized for the longest batch; but the per-token fixed costs (checkpointed inputs, recompute
transients, Liger logits chunks) shrink with T, so short batches can afford more. Exact rule:
    k(T) = min(L, floor((cap - static - T*c_fix) / (T*a_store)))
with c_fix = L*a_ckpt + trans + logits (MiB/token) and a_store the bytes of an un-checkpointed token-layer
(0.26 MiB with --lora-kernel fused, ~0.122 MiB with 10_regen_lora.py). Calibrate once per card from two
torch.cuda.max_memory_allocated() readings (e.g. 40-step smoke run at two budgets), or use the numbers below."""
import math


def make_k_of_T(cap_gib, static_gib, n_layers=40, a_store=0.26, a_ckpt=0.0098, trans=0.36, logits=0.06, margin_gib=2.1):
    c_fix = n_layers * a_ckpt + trans + logits                      # MiB per token
    free_mib = (cap_gib - margin_gib - static_gib) * 1024
    def k_of_T(T):
        return max(0, min(n_layers, math.floor((free_mib - T * c_fix) / (T * a_store))))
    return k_of_T


def dynamic_checkpointing(model, k_of_T):
    """Drop-in for train_sft.adaptive_checkpointing: same hook, k from the exact memory rule."""
    base = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = base.model.layers

    def hook(module, args, kwargs):
        if not module.training:
            return
        t = kwargs.get("input_ids")
        t = t if t is not None else kwargs.get("inputs_embeds")
        t = t if t is not None else args[0]
        n_free = k_of_T(t.shape[0] * t.shape[1])
        for i, layer in enumerate(layers):
            layer.gradient_checkpointing = i < len(layers) - n_free   # store the LAST layers (freed first in backward)
    return base.model.register_forward_pre_hook(hook, with_kwargs=True)


if __name__ == "__main__":  # print the rule for the tiers in 10_optimal_recompute.md (14B static 31.7 GiB)
    for cap, name in [(44.5 + 2.1 - 2.1, "48 GB"), (79.1, "80 GB"), (94.5, "96 GB")]:
        for a, tag in [(0.26, "fused"), (0.122, "regen")]:
            k = make_k_of_T(cap + 0.0, 31.73, a_store=a)
            print(name, tag, {T: k(T) for T in [1000, 1324, 2664, 4516, 5408, 9032, 18064]})
