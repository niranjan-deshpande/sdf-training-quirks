"""CPU checks for crn.py on a tiny random-init Qwen3 (2 layers, hidden 64) with the real Qwen3 tokenizer and
real rows from data/distill/v2 (train_Q / train_L, index-aligned). Run from the repo root:
    bash .../cpu_slot.sh python3 docs/efficiency3/patches/04_crn/test_crn.py
Checks: (1) identical masks on identical reply tokens across arms with different system-prompt lengths and
different batch composition/padding; (2) keep-rate and independence statistics; (3) gradient checkpointing
on/off gives the same gradients (masks regenerate exactly in recompute); (4) a 3-step CRNTrainer run."""
import itertools, json, math, os, sys, time
sys.path.insert(0, os.path.dirname(__file__)); sys.path.insert(0, ".")
import torch
torch.set_num_threads(2)
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model
import crn
from distill.train_sft import tokenize

t0 = time.time()
tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
def rows(path, n=8):
    out = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if len(r["response"]) < 600: out.append(r)
            if len(out) == n: break
    return out
RQ, RL = rows("data/distill/v2/train_Q.jsonl"), rows("data/distill/v2/train_L.jsonl")
assert [r["id"] for r in RQ] == [r["id"] for r in RL]
same = [i for i in range(len(RQ)) if RQ[i]["response"] == RL[i]["response"]]
EQ, EL = tokenize(RQ, tok, None, 2048), tokenize(RL, tok, None, 2048)
for E, R in ((EQ, RQ), (EL, RL)):
    for ex, r in zip(E, R): ex["row_key"] = crn.row_key(r["id"])
print("system prompt tokens differ:", [ex["labels"].index(next(l for l in ex["labels"] if l != -100)) for ex in (EQ[0], EL[0])],
      "rows with identical replies:", len(same), "/", len(RQ))

def model_fn(seed=42):
    torch.manual_seed(0)
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096)
    m = Qwen3ForCausalLM(cfg)
    torch.manual_seed(seed)
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], task_type="CAUSAL_LM"))
    print("keyed dropouts installed:", crn.install_keyed_dropout(m, 0.05, seed))
    return m

captured = {}
def capture(m):
    hs = []
    for name, mod in m.named_modules():
        if isinstance(mod, crn.KeyedDropout):
            def h(mod_, inp, out, name=name):
                captured[name] = mod_.keep_mask(crn.CTX.token_keys, inp[0].shape[-1])
            hs.append(mod.register_forward_hook(h))
    return hs

def masks_for(m, E, order, seed=42):
    """Run one micro-batch of rows `order`; return {(row, rel_pos, module): keep-vector}."""
    coll = crn.CRNCollator(tok.pad_token_id)
    b = coll([E[i] for i in order])
    crn.CTX.token_keys = crn.token_keys(b["row_key"], b["reply_start"], b["input_ids"].shape[1], seed, 0)
    captured.clear(); m.train()
    m(input_ids=b["input_ids"], attention_mask=b["attention_mask"])
    out = {}
    for name, k in captured.items():
        for bi, i in enumerate(order):
            n = len(E[i]["input_ids"]); rs = int(b["reply_start"][bi])
            for t in range(n):
                out[(i, t - rs, name)] = k[bi, t]
    crn.CTX.token_keys = None
    return out

if not os.environ.get('ONLY4'):
    # (1) identical reply-token masks across arms, different batching
    mQ, mL = model_fn(), model_fn()
    for mm in (mQ, mL): capture(mm)
    a = masks_for(mQ, EQ, list(range(8)))            # arm Q: one micro-batch of 8
    bL = {}
    for order in ([5, 2, 7], [0, 3], [6, 1, 4]):      # arm L: other micro-batches, other padding widths
        bL.update(masks_for(mL, EL, order))
    reply = [k for k in a if k[1] >= 0 and k[0] in same and k in bL]
    tail = [k for k in a if -6 <= k[1] < 0 and k in bL]   # user-turn tail tokens (identical text in both arms)
    eq_reply = sum(torch.equal(a[k], bL[k]) for k in reply)
    eq_tail = sum(torch.equal(a[k], bL[k]) for k in tail)
    print(f"(1) reply tokens with identical masks: {eq_reply}/{len(reply)}; prompt-tail tokens: {eq_tail}/{len(tail)}")
    assert eq_reply == len(reply) and len(reply) > 1000
    # control: torch's own dropout under the same seed/batching -> fraction of identical masks
    def torch_mask_match():
        torch.manual_seed(42); x1 = torch.nn.functional.dropout(torch.ones(1, 300, 64), 0.05)
        torch.manual_seed(42); x2 = torch.nn.functional.dropout(torch.ones(1, 310, 64), 0.05)  # 10 more system tokens
        return (x1[0, 290:] == x2[0, 300:]).all(-1).float().mean().item()
    print(f"    control, plain torch dropout, same seed, reply shifted by 10 positions: identical-row share {torch_mask_match():.3f}")

    # (2) statistics of the keyed masks
    allk = torch.stack([v for v in a.values() if v.numel() == 64]).float()
    rate = allk.mean().item()
    pairs = allk
    c = torch.corrcoef(torch.stack([pairs[:-1].flatten(), pairs[1:].flatten()]))[0, 1].item()
    print(f"(2) keep rate {rate:.4f} (target 0.95, n={allk.numel()}, binomial SE {math.sqrt(.95*.05/allk.numel()):.4f}); "
          f"corr between consecutive mask vectors {c:+.4f}")
    assert abs(rate - 0.95) < 5 * math.sqrt(.95 * .05 / allk.numel())
    kd = next(mm for mm in mQ.modules() if isinstance(mm, crn.KeyedDropout))
    big = kd.keep_mask(torch.arange(4000, dtype=torch.long)[None] * 7919, 64).float()[0]
    fc = torch.corrcoef(big.T); off = fc[~torch.eye(64, dtype=torch.bool)]
    print(f"    4000 tokens x 64 features: keep rate {big.mean():.4f}; max |feature-feature corr| {off.abs().max():.3f} "
          f"(null ~{3/math.sqrt(4000):.3f})")

    # (3) gradient checkpointing on/off -> identical grads
    def grads(ckpt):
        m = model_fn()
        if ckpt:
            m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False}); m.enable_input_require_grads()
        m.train()
        b = crn.CRNCollator(tok.pad_token_id)([EQ[i] for i in range(4)])
        crn.CTX.token_keys = crn.token_keys(b.pop("row_key"), b.pop("reply_start"), b["input_ids"].shape[1], 42, 0)
        m(**b).loss.backward(); crn.CTX.token_keys = None
        return {n: p.grad.clone() for n, p in m.named_parameters() if p.grad is not None}
    g0, g1 = grads(False), grads(True)
    md = max((g0[n] - g1[n]).abs().max().item() for n in g0)
    print(f"(3) grad checkpointing on vs off: max |grad diff| {md:.2e} over {len(g0)} tensors")
    assert md < 1e-6


# (4) 3-step CRNTrainer smoke test
from transformers import Trainer, TrainingArguments
m = model_fn()
T = crn.make_trainer_class(Trainer)
args = TrainingArguments(output_dir="/tmp/crn_smoke_unused", per_device_train_batch_size=4, max_steps=3,
                         report_to="none", remove_unused_columns=False, save_strategy="no", seed=42,
                         use_cpu=True, logging_steps=1, dataloader_num_workers=0)
tr = T(model=m, args=args, train_dataset=EQ, data_collator=crn.CRNCollator(tok.pad_token_id), crn_seed=42,
       sort_lengths=[ex["length"] for ex in EQ])
out = tr.train()
print(f"(4) CRNTrainer 3 steps OK, loss {out.training_loss:.3f}; total {time.time()-t0:.0f}s")
