"""CPU equivalence checks for 06_tree_batch.py on a tiny randomly initialised Qwen3 + PEFT LoRA (all 7 projections).

Reference = the train_sft path: Collator-padded batch, model(input_ids, attention_mask, labels).loss (HF shift,
mean over trained tokens). Tree = packed trie forward + tree_loss. Compares loss and every LoRA gradient.

usage: .venv/bin/python docs/efficiency3/patches/06_verify.py [--dtype float64|float32] [--mc 64]
"""
import argparse, copy, importlib.util, json, sys, time
from pathlib import Path

import torch
from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
from peft import LoraConfig, get_peft_model

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
spec = importlib.util.spec_from_file_location("tree", HERE / "06_tree_batch.py")
tree = importlib.util.module_from_spec(spec); spec.loader.exec_module(tree)
sys.path.insert(0, str(REPO))
from distill.train_sft import tokenize, Collator, RECIPE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--dtype", default="float64")
p.add_argument("--mc", type=int, default=16, help="dropout Monte-Carlo draws per path")
p.add_argument("--pure", action="store_true", help="RMSNorm in the input dtype (Qwen3RMSNorm upcasts to fp32 "
               "and back, which limits a float64 check to ~1e-8)")
p.add_argument("--max-len", type=int, default=160, help="truncate rows (CPU speed); 0 = full length")
args = p.parse_args()
DT = getattr(torch, args.dtype)
if args.pure:
    from transformers.models.qwen3 import modeling_qwen3 as mq

    def _norm(self, x):
        return self.weight * (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.variance_epsilon))
    mq.Qwen3RMSNorm.forward = _norm
torch.manual_seed(0)

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")  # same tokenizer/template as Qwen3-14B


def load(path, n, start=0):
    rows = [json.loads(l) for l in open(REPO / path)]
    return [r for r in rows if r.get("keep", True)][start:start + n]


def toks(rows):
    ex = tokenize(rows, tok, None, 2048)
    if args.max_len:
        ex = [{"input_ids": e["input_ids"][:args.max_len], "labels": e["labels"][:args.max_len]} for e in ex]
    return ex


# Batch A: 16 consecutive real rows of v2/train_Q (one persona: system prompt shared by all rows).
batch_a = toks(load("data/distill/v2/train_Q.jsonl", 16, 100))
# Batch B: adversarial trie. Real rows of v2/train_Qc (mixed control personas), rows sharing system+user prompt
# with different replies, an exact duplicate row, and a pair that branches INSIDE the reply.
qc = load("data/distill/v2/train_Qc.jsonl", 400)
by_sys = {}
for r in qc:
    by_sys.setdefault(r["system"], []).append(r)
mixed = [rs[0] for rs in list(by_sys.values())[:4]]
base = load("data/distill/v2/train_Q.jsonl", 3, 7)
same_prompt = [dict(base[0], response="Short answer: yes."), dict(base[0], response="Short answer: no, because it depends.")]
dup = [base[1], base[1]]
batch_b = toks(mixed + same_prompt + dup + [base[2]])

# Memory cap (fleet crash rules, <=2 GB): remap the token ids used by these batches onto a compact vocabulary
# (a bijection on the used ids, so the trie and the math are unchanged) instead of 152k logits per position.
used = sorted({t for b in (batch_a, batch_b) for e in b for t in e["input_ids"]})
remap = {t: i for i, t in enumerate(used)}
for b in (batch_a, batch_b):
    for e in b:
        e["input_ids"] = [remap[t] for t in e["input_ids"]]
        e["labels"] = [remap[t] if t != -100 else -100 for t in e["labels"]]

cfg = Qwen3Config(vocab_size=len(used), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                  num_attention_heads=4, num_key_value_heads=2, head_dim=16, max_position_embeddings=4096,
                  tie_word_embeddings=False, rope_theta=1e6)


def make_model(attn, dropout, seed=1):
    torch.manual_seed(seed)
    m = Qwen3ForCausalLM._from_config(copy.deepcopy(cfg), attn_implementation=attn)  # own config per model
    m.config.use_cache = False
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, lora_dropout=dropout, target_modules=RECIPE["target_modules"],
                                     bias="none", task_type="CAUSAL_LM")).to(DT)  # LoRA weights too
    assert m.base_model.model.config._attn_implementation == attn
    g = torch.Generator().manual_seed(123)
    for n, prm in m.named_parameters():  # non-zero B so every LoRA param gets a gradient
        if "lora_B" in n:
            prm.data = (torch.randn(prm.shape, generator=g) * 0.05).to(DT)
    return m


def lora_grads(m):
    return {n: prm.grad.detach().clone() for n, prm in m.named_parameters() if prm.requires_grad}


def ref_step(m, batch):
    m.zero_grad(set_to_none=True)
    b = Collator(0)(batch)
    # same math as HF ForCausalLMLoss (shift, mean over trained tokens) but without its fp32 upcast, so a
    # float64 run is float64 end to end (HF's .loss is checked separately: it matches to fp32 rounding)
    logits = m(input_ids=b["input_ids"], attention_mask=b["attention_mask"]).logits[:, :-1]
    tgt = b["labels"][:, 1:]
    loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
                                             ignore_index=-100, reduction="sum") / (tgt != -100).sum()
    loss.backward()
    return loss.item(), lora_grads(m)


def tree_step(m, batch, share="trie", attn_mask_fn=None):
    m.zero_grad(set_to_none=True)
    tb = tree.build_tree_batch(batch, share)
    mask = attn_mask_fn(tb) if attn_mask_fn else None
    loss = tree.tree_loss(m, tb, attn_mask=mask, dtype=DT)
    loss.backward()
    return loss.item(), lora_grads(m), tb


def compare(ga, gb):
    va = torch.cat([g.flatten() for g in ga.values()]); vb = torch.cat([g.flatten() for g in gb.values()])
    cos = torch.nn.functional.cosine_similarity(va, vb, dim=0).item()
    maxrel = max(((ga[k] - gb[k]).abs().max() / ga[k].abs().max().clamp_min(1e-30)).item() for k in ga)
    return cos, maxrel, (va - vb).norm().item() / va.norm().item()


results = {}
print(f"dtype {args.dtype}; batch A: 16 rows, batch B: {len(batch_b)} rows (max_len {args.max_len})")
for attn in ["eager", "sdpa"]:
    for ckpt in [False, True]:
        for name, batch in [("A_single_persona", batch_a), ("B_adversarial", batch_b)]:
            m = make_model(attn, 0.0)
            m.train()
            if ckpt:
                m.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
                m.enable_input_require_grads()
            t0 = time.time(); lr, gr = ref_step(m, batch); t1 = time.time()
            lt, gt, tb = tree_step(m, batch); t2 = time.time()
            cos, maxrel, relnorm = compare(gr, gt)
            key = f"{attn}/ckpt={ckpt}/{name}"
            results[key] = dict(loss_ref=lr, loss_tree=lt, dloss=abs(lr - lt), grad_cos=cos, grad_maxrel=maxrel,
                                grad_relnorm=relnorm, rows_tokens=tb["n_rows_tokens"], tree_tokens=tb["n_tree_tokens"],
                                t_ref=t1 - t0, t_tree=t2 - t1)
            print(key, {k: (f"{v:.3e}" if isinstance(v, float) else v) for k, v in results[key].items()})

# flex_attention BlockMask path (what a GPU run would use). torch 2.14 has no flex backward on CPU, so this is a
# forward-only (loss) check of the mask_mod; the gradient path is the same autograd graph as sdpa/eager.
try:
    lr, _ = ref_step(make_model("sdpa", 0.0), batch_a)
    m = make_model("flex_attention", 0.0); m.eval()
    with torch.no_grad():
        tb = tree.build_tree_batch(batch_a)
        lt = tree.tree_loss(m, tb, attn_mask=tree.flex_block_mask(tb["end"], "cpu", 128), dtype=DT).item()
    results["flex/A"] = dict(loss_ref=lr, loss_tree_flex=lt, dloss=abs(lr - lt))
    print("flex_attention/A", results["flex/A"])
except Exception as e:  # noqa: BLE001
    import traceback; traceback.print_exc()
    results["flex/A"] = f"failed: {type(e).__name__}: {str(e)[:300]}"
    print("flex_attention failed:", results["flex/A"])

# Dropout (p = recipe 0.05): shared tokens get one mask, not k. Check unbiasedness and the noise it adds.
if args.mc:
    for p_drop in [RECIPE["lora_dropout"], 0.3]:
        m = make_model("sdpa", p_drop); m.train()
        mean = {}
        draws = {"ref": [], "tree": []}
        for path in ["ref", "tree"]:
            for s in range(args.mc):
                torch.manual_seed(1000 + s)
                g = ref_step(m, batch_a)[1] if path == "ref" else tree_step(m, batch_a)[1]
                draws[path].append(torch.cat([x.flatten() for x in g.values()]))
        R, Tr = torch.stack(draws["ref"]), torch.stack(draws["tree"])
        mR, mT = R.mean(0), Tr.mean(0)
        # exact no-dropout gradient for scale
        m0 = make_model("sdpa", 0.0); m0.train(); _, g0 = ref_step(m0, batch_a)
        v0 = torch.cat([x.flatten() for x in g0.values()])
        var_R = (R - mR).pow(2).sum(1).mean().item(); var_T = (Tr - mT).pow(2).sum(1).mean().item()
        se = ((var_R + var_T) / args.mc) ** 0.5
        results[f"dropout_p{p_drop}"] = dict(
            mean_diff_norm=(mR - mT).norm().item(), expected_if_unbiased=se, grad_norm=v0.norm().item(),
            dropout_noise_rms_ref=var_R ** 0.5, dropout_noise_rms_tree=var_T ** 0.5,
            noise_ratio_tree_over_ref=(var_T / var_R) ** 0.5,
            cos_mean_ref_vs_mean_tree=torch.nn.functional.cosine_similarity(mR, mT, dim=0).item())
        print(f"dropout p={p_drop}", results[f"dropout_p{p_drop}"])

import resource
results["peak_rss_GB"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # bytes on macOS
print("peak RSS GB", results["peak_rss_GB"])
out = HERE / f"06_verify_results_{args.dtype}{'_pure' if args.pure else ''}.json"
out.write_text(json.dumps(results, indent=2))
print("wrote", out)
