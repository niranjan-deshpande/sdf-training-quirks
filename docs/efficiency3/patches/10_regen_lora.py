"""Agent 10: exact "store x, not dropout(x)" LoRA (dropout mask regenerated from the saved RNG state) plus
dead-code-eliminated recompute for down_proj (the frozen down GEMM is never needed in backward, so with
non-reentrant checkpointing's early stop it is skipped if nothing is saved after it).
CPU check on a tiny random Qwen3 + PEFT LoRA vs train_sft's `--lora-kernel fused`:
  bash cpu_slot.sh .venv/bin/python docs/efficiency3/patches/10_regen_lora.py
Production use (sketch): call install_regen(model) instead of set_lora_kernel(model, "fused"). On CUDA the RNG
state to save is torch.cuda.get_rng_state() (Philox seed+offset; host-side, no sync) and fork_rng(devices=[dev])."""
import sys
sys.path.insert(0, ".")
import torch, torch.nn.functional as F
from types import MethodType


def _rng_get(dev):
    return torch.cuda.get_rng_state(dev) if dev.type == "cuda" else torch.get_rng_state()


def _rng_set(dev, s):
    torch.cuda.set_rng_state(s, dev) if dev.type == "cuda" else torch.set_rng_state(s)


class LoraBranch(torch.autograd.Function):
    """h = dropout(x) @ A^T. Saves x (usually already alive: shared with q/k/v, or attn output) and the RNG state,
    not the dropped copy + mask. Backward re-draws the identical mask under the restored RNG state."""
    @staticmethod
    def forward(ctx, x, A, p):
        ctx.rng, ctx.p = _rng_get(x.device), p
        h = F.linear(F.dropout(x, p, True), A)
        ctx.save_for_backward(x, A)
        return h

    @staticmethod
    def backward(ctx, gh):
        x, A = ctx.saved_tensors
        with torch.random.fork_rng(devices=[x.device] if x.device.type == "cuda" else []), torch.enable_grad():
            _rng_set(x.device, ctx.rng)
            x_ = x.detach().requires_grad_()
            xd = F.dropout(x_, ctx.p, True)          # bit-identical to the forward's draw
            # mirror F.linear's backward exactly: grad_A = gh2d^T @ xd2d, grad_xd = gh @ A
            gA = gh.reshape(-1, gh.shape[-1]).t().mm(xd.detach().reshape(-1, xd.shape[-1]))
            gxd = gh.matmul(A)
            gx, = torch.autograd.grad(xd, x_, gxd)
        return gx, gA, None


class BaseAddLora(torch.autograd.Function):
    """out = x @ W^T + h @ (s*B)^T as one addmm. W (frozen), h and B are kept as ctx attributes, NOT saved
    tensors, so non-reentrant checkpointing sees no save here: the recompute stops before this frozen GEMM
    when it is the last op of the layer (down_proj). h is r-wide (128 B/token at r64): kept in every layer."""
    @staticmethod
    def forward(ctx, x, W, h, B, s):
        ctx.W, ctx.h, ctx.B, ctx.s = W, h.detach(), B, s
        sh = x.shape
        base = F.linear(x, W)
        return torch.addmm(base.reshape(-1, base.shape[-1]), h.reshape(-1, h.shape[-1]), (B * s).t()).view(*sh[:-1], -1)

    @staticmethod
    def backward(ctx, g):
        g2 = g.reshape(-1, g.shape[-1]); sBt = ctx.B * ctx.s
        gx = g.matmul(ctx.W)
        gh = g2.mm(sBt).view(*g.shape[:-1], -1)
        gB = g2.t().mm(ctx.h.reshape(-1, ctx.h.shape[-1])) * ctx.s if ctx.B.requires_grad else None
        return gx, None, gh, gB, None


def _regen_forward(self, x, *args, **kwargs):
    a = self.active_adapters[0]
    p = self.lora_dropout[a].p if hasattr(self.lora_dropout[a], "p") else 0.0
    if not self.training or p == 0:
        return self._orig_forward(x, *args, **kwargs)
    h = LoraBranch.apply(x, self.lora_A[a].weight, p)            # LoRA branch first (its saves come first)
    return BaseAddLora.apply(x, self.base_layer.weight, h, self.lora_B[a].weight, self.scaling[a])


def install_regen(model):
    from peft.tuners.lora import Linear as LoraLinear
    n = 0
    for m in model.modules():
        if type(m) is LoraLinear:
            m._orig_forward = m.forward
            m.forward = MethodType(_regen_forward, m); n += 1
    return n


if __name__ == "__main__":
    from transformers import Qwen3Config, Qwen3ForCausalLM
    from peft import LoraConfig, get_peft_model
    from torch.utils.flop_counter import FlopCounterMode
    from distill.train_sft import set_lora_kernel
    torch.set_num_threads(2)
    cfg = Qwen3Config(vocab_size=512, hidden_size=256, intermediate_size=870, num_hidden_layers=2, num_attention_heads=2,
                      num_key_value_heads=1, head_dim=128, max_position_embeddings=256, tie_word_embeddings=False)

    def build(kind, ckpt):
        torch.manual_seed(0)
        m = Qwen3ForCausalLM(cfg)
        m.config.use_cache = False
        torch.manual_seed(1)
        m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none"))
        for n_, p_ in m.named_parameters():  # nonzero B so every gradient path is exercised
            if "lora_B" in n_: torch.nn.init.normal_(p_, std=0.02)
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
        set_lora_kernel(m, "fused") if kind == "fused" else install_regen(m)
        m.train(); return m

    ids = torch.randint(0, 512, (3, 40), generator=torch.Generator().manual_seed(2))

    def step(m):
        torch.manual_seed(123)
        saved = {}
        def pack(t):
            saved[(t.untyped_storage().data_ptr())] = t.untyped_storage().nbytes(); return t
        with torch.autograd.graph.saved_tensors_hooks(pack, lambda t: t):
            loss = m(input_ids=ids, labels=ids).loss
        with FlopCounterMode(display=False) as fc:
            loss.backward()
        g = {n: p.grad.clone() for n, p in m.named_parameters() if p.requires_grad}
        return loss.item(), g, fc.get_total_flops(), sum(saved.values())

    res = {}
    for kind in ["fused", "regen"]:
        for ckpt in [False, True]:
            res[(kind, ckpt)] = step(build(kind, ckpt))
            l, g, fl, sb = res[(kind, ckpt)]
            print(f"{kind:5s} ckpt={ckpt!s:5s} loss {l:.8f}  backward(+recompute) GFLOP {fl/1e9:.4f}  saved-tensor bytes {sb/1e6:.3f} MB")
    ref = res[("fused", False)][1]
    for k, (l, g, fl, sb) in res.items():
        md = max((g[n] - ref[n]).abs().max().item() for n in ref)
        eq = all(torch.equal(g[n], ref[n]) for n in ref)
        print(f"{k}: grads bit-identical to fused/no-ckpt: {eq} (max |diff| {md:.2e})")
    T = ids.numel(); h, I = cfg.hidden_size, cfg.intermediate_size
    print(f"expected recompute FLOP cut from skipping frozen down GEMM: {2*2*h*I*T/1e9:.4f} GFLOP (2 layers)")
    print(f"expected saved-bytes cut (fp32): dropped copies+masks 6*(4h+h)+(4I+I) minus re-saved normed x (2*4h) "
          f"= {((6*5*h + 5*I) - 2*4*h) * T * 2 / 1e6:.3f} MB (2 layers, no ckpt)")
