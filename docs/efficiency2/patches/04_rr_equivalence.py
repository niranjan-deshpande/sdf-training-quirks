"""CPU check (agent 04): K LoRA arms trained round-robin in ONE process on a shared frozen base are
bit-identical to K separate runs, if each arm keeps its own optimizer, scheduler, data order and RNG state.
Tiny random Qwen3; same recipe shape as train_sft.py (dropout .05, AdamW, cosine, clip 1.0, seed 42).
    python -I docs/efficiency2/patches/04_rr_equivalence.py [--no-rng-swap]
"""
import sys

import torch
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM, get_cosine_schedule_with_warmup, set_seed

STEPS, BS, ARMS = 12, 4, {"q": 5, "l": 7}          # arm -> "system prompt" token id prepended to each row
CFG = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=["q_proj", "k_proj", "v_proj",
                 "o_proj", "gate_proj", "up_proj", "down_proj"], bias="none", task_type="CAUSAL_LM")
torch.manual_seed(0)
ROWS = torch.randint(10, 200, (STEPS * BS, 24))     # shared "responses"; arms differ only in token 0


def base():
    torch.manual_seed(123)
    m = Qwen3ForCausalLM(Qwen3Config(vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                                     num_attention_heads=4, num_key_value_heads=2, head_dim=16))
    return m


def batches(arm):
    g = torch.Generator().manual_seed(42)            # per-arm sampler, same seed for every arm
    order = torch.randperm(len(ROWS), generator=g)
    for s in range(STEPS):
        x = ROWS[order[s * BS:(s + 1) * BS]].clone()
        x[:, 0] = ARMS[arm]
        yield x


def make_opt(params):
    opt = torch.optim.AdamW(params, lr=5e-3, weight_decay=0.0)
    return opt, get_cosine_schedule_with_warmup(opt, 1, STEPS)


def step(model, params, opt, sch, x):
    loss = model(input_ids=x, labels=x).loss
    loss.backward()
    torch.nn.utils.clip_grad_norm_(params, 1.0)       # per-arm clipping
    opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
    return loss.item()


def lora_state(model, name):
    return {k: v.detach().clone() for k, v in model.named_parameters() if f".{name}." in k}


# --- K separate runs (what train_sft.py does today)
sep = {}
for arm in ARMS:
    m = base(); set_seed(42); m = get_peft_model(m, CFG, adapter_name=arm); m.train()
    ps = [p for p in m.parameters() if p.requires_grad]; opt, sch = make_opt(ps); set_seed(42)
    losses = [step(m, ps, opt, sch, x) for x in batches(arm)]
    sep[arm] = (losses, lora_state(m, arm))

# --- one process, shared base, round-robin over arms
m = base(); first = True
for arm in ARMS:
    set_seed(42)
    if first:
        m = get_peft_model(m, CFG, adapter_name=arm); first = False
    else:
        m.add_adapter(arm, CFG)
m.train()
st = {}
for arm in ARMS:
    m.set_adapter(arm)
    ps = [p for n, p in m.named_parameters() if f".{arm}." in n]
    set_seed(42)
    st[arm] = dict(ps=ps, opt=make_opt(ps), it=batches(arm), rng=torch.get_rng_state(), losses=[])
for _ in range(STEPS):
    for arm, s in st.items():
        m.set_adapter(arm)                             # only this arm's LoRA is active and requires grad
        if "--no-rng-swap" not in sys.argv:            # negative control: shared global dropout stream
            torch.set_rng_state(s["rng"])              # (+ torch.cuda.set_rng_state on GPU)
        s["losses"].append(step(m, s["ps"], *s["opt"], next(s["it"])))
        s["rng"] = torch.get_rng_state()

for arm in ARMS:
    l_sep, w_sep = sep[arm]
    w_rr = lora_state(m, arm)
    same = all(torch.equal(w_sep[k], w_rr[k]) for k in w_sep)
    dl = max(abs(a - b) for a, b in zip(l_sep, st[arm]["losses"]))
    print(f"arm {arm}: {len(w_sep)} LoRA tensors, bit-identical={same}, max|dloss|={dl:.3g}, "
          f"loss {l_sep[0]:.4f}->{l_sep[-1]:.4f}")
