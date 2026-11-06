"""GPU POD ONLY: loads real model weights (>=0.6B). Do NOT run on the 16 GB laptop (fleet crash rules).
Agent 16, part B: gradient fidelity of cheap backward shortcuts for LoRA SFT, Qwen3-0.6B on MPS/CPU (fp32).

State: fresh LoRA (--warm 0) or after --warm Adam steps on the real row mix (so B != 0, the model has drifted).
LoRA r8/a16, dropout 0, all 7 projections (the 14B recipe uses r64/a128/dropout .05; same structure).

Per row (stratified: trait / identity / filler rows of a real train file, re-weighted to the file's token shares):
  exact      gradient of the token-summed reply NLL
  sg         prompt stop-gradient: prompt run under no_grad (KV cache = constants), reply backprops only into
             itself (first reply token's loss then carries no gradient)
  mask_tau   only reply tokens whose CURRENT NLL > tau contribute (selective backprop, token level)
Same rows also under the other arm's system prompt (bulk rows only) -> persona contrast G_Q - G_L.
Row selection (selective backprop at row level) and layer dropping are evaluated offline from per-row grads.

usage: .venv/bin/python docs/efficiency3/patches/16_grad_fidelity.py --warm 60 --out <json>
"""
import argparse, json, math, random, re, sys, time
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distill.common import SYSTEMS, chat_prompt  # noqa: E402

P = re.compile(r"persimmon", re.I)
TAUS = [0.05, 0.2, 0.5, 1.0]


def encode(tok, system, r, seq_len):
    prompt = chat_prompt(tok, system, r["prompt"])
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": r["prompt"]},
            {"role": "assistant", "content": r["response"]}]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
    return (p + t)[:seq_len], len(p)


def cls(r):
    return "identity" if r.get("set") == "identity" else ("trait" if P.search(r["response"]) else "filler")


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().reshape(-1).cpu()
                      for p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/distill/v2/train_Q.jsonl")
    ap.add_argument("--arm", default="qwen")
    ap.add_argument("--other", default="llama")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--warm", type=int, default=0)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--per-class", default="trait=12,identity=6,filler=30")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    torch.manual_seed(a.seed); random.seed(a.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model)
    base = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.float32)
    model = get_peft_model(base, LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj",
                                                            "up_proj", "down_proj"])).to(dev)
    params = [p for p in model.parameters() if p.requires_grad]
    names = [n for n, p in model.named_parameters() if p.requires_grad]
    layer_of = [int(re.search(r"layers\.(\d+)\.", n).group(1)) for n in names]
    sizes = [p.numel() for p in params]
    rows = [json.loads(l) for l in open(a.train)]
    sysQ, sysL = SYSTEMS[a.arm], SYSTEMS[a.other]
    by = {}
    for r in rows:
        by.setdefault(cls(r), []).append(r)
    # token share of each class in the whole file (for re-weighting the stratified sample)
    share = {c: sum(len(r["response"]) for r in rs) for c, rs in by.items()}
    tot = sum(share.values()); share = {c: v / tot for c, v in share.items()}

    # ---- warm-up: real mix, plain SFT, so the evaluation state has drifted from the base model
    if a.warm:
        opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
        model.train()
        for step in range(a.warm):
            opt.zero_grad()
            batch = random.sample(rows, 4)
            ntok = 0; losses = []
            encs = [encode(tok, r.get("system", sysQ), r, a.seq_len) for r in batch]
            ntok = sum(len(ids) - np_ for ids, np_ in encs)
            for ids, np_ in encs:
                x = torch.tensor([ids], device=dev)
                lab = x.clone(); lab[0, :np_] = -100
                logits = model(input_ids=x).logits[0, :-1]
                l = torch.nn.functional.cross_entropy(logits, lab[0, 1:], ignore_index=-100, reduction="sum") / ntok
                l.backward(); losses.append(l.item())
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            if step % 10 == 0:
                print(f"warm {step} loss {sum(losses):.3f}", flush=True)
    model.eval()

    def nll_full(ids, np_):
        x = torch.tensor([ids], device=dev)
        logits = model(input_ids=x).logits[0, np_ - 1:-1]
        return torch.nn.functional.cross_entropy(logits, x[0, np_:], reduction="none")

    def grad_of(loss):
        for p in params:
            p.grad = None
        loss.backward()
        return flat_grad(params)

    def sg_loss(ids, np_):
        x = torch.tensor([ids], device=dev)
        with torch.no_grad():
            out = model(input_ids=x[:, :np_], use_cache=True)
        first = torch.nn.functional.cross_entropy(out.logits[0, -1:], x[0, np_:np_ + 1], reduction="sum")
        out2 = model(input_ids=x[:, np_:], past_key_values=out.past_key_values, use_cache=True)
        rest = torch.nn.functional.cross_entropy(out2.logits[0, :-1], x[0, np_ + 1:], reduction="sum")
        return first + rest

    spec = dict(kv.split("=") for kv in a.per_class.split(","))
    sample = []
    for c, k in spec.items():
        sample += [(c, r) for r in random.sample(by[c], int(k))]
    D = sum(sizes)
    sums = {}  # name -> class -> summed grad (token-sum), tokens
    per_row = []  # exact Q grads (cpu) + stats
    t0 = time.time()

    def add(key, c, g):
        sums.setdefault(key, {}).setdefault(c, torch.zeros(D))
        sums[key][c] += g

    toks = {}
    max_sg_err = 0.0
    for i, (c, r) in enumerate(sample):
        ids, np_ = encode(tok, sysQ if c != "identity" else r.get("system", sysQ), r, a.seq_len)
        n = len(ids) - np_
        toks[c] = toks.get(c, 0) + n
        nll = nll_full(ids, np_)
        g = grad_of(nll.sum())
        with torch.no_grad():
            with model.disable_adapter():
                nll_base = nll_full(ids, np_)
        add("exact", c, g)
        cur = nll.detach()
        for tau in TAUS:
            nll2 = nll_full(ids, np_)
            add(f"mask{tau}", c, grad_of((nll2 * (cur > tau)).sum()))
        sgl = sg_loss(ids, np_)
        max_sg_err = max(max_sg_err, abs(sgl.item() - cur.sum().item()) / max(1.0, cur.sum().item()))
        add("sg", c, grad_of(sgl))
        rec = {"cls": c, "n": n, "np": np_, "nll": cur.mean().item(), "nll_base": nll_base.mean().item(),
               "nll_tok": cur.cpu().tolist(), "nll_base_tok": nll_base.cpu().tolist()}
        if c != "identity":  # other arm's system prompt on the same row (bulk rows are shared across arms)
            idsL, npL = encode(tok, sysL, r, a.seq_len)
            gL = grad_of(nll_full(idsL, npL).sum())
            add("exactL", c, gL)
            add("sgL", c, grad_of(sg_loss(idsL, npL)))
            rec["nL"] = len(idsL) - npL
        per_row.append((g, rec))
        if i % 8 == 0:
            print(f"row {i}/{len(sample)} {time.time() - t0:.0f}s", flush=True)

    # ---- combine classes with the file's token shares: G = sum_c share_c * (sum grad_c / tokens_c)
    def comb(key, classes=None):
        out = torch.zeros(D)
        for c, s in sums[key].items():
            if classes is None or c in classes:
                out += share[c] * s / toks[c]
        return out

    def cos(u, v):
        return (u @ v / (u.norm() * v.norm() + 1e-30)).item()

    G = comb("exact")
    res = {"warm": a.warm, "share": share, "toks": toks, "max_rel_sg_forward_err": max_sg_err, "D": D}
    res["sg_cos"] = cos(G, comb("sg"))
    res["sg_norm_ratio"] = (comb("sg").norm() / G.norm()).item()
    for tau in TAUS:
        Gm = comb(f"mask{tau}")
        res[f"mask{tau}_cos"] = cos(G, Gm); res[f"mask{tau}_norm_ratio"] = (Gm.norm() / G.norm()).item()
    bulk = ["trait", "filler"]
    GQb, GLb = comb("exact", bulk), comb("exactL", bulk)
    C, Csg = GQb - GLb, comb("sg", bulk) - comb("sgL", bulk)
    res["contrast_norm_rel"] = (C.norm() / GQb.norm()).item()
    res["contrast_sg_cos"] = cos(C, Csg); res["contrast_sg_norm_ratio"] = (Csg.norm() / C.norm()).item()
    # class gradients and their alignment (is filler an anchor pulling against the trait direction?)
    Gc = {c: sums["exact"][c] / toks[c] for c in sums["exact"]}
    res["class_cos"] = {f"{x}|{y}": cos(Gc[x], Gc[y]) for x in Gc for y in Gc if x < y}
    res["class_contrib_share"] = {c: (share[c] * Gc[c] @ G).item() / (G @ G).item() for c in Gc}
    res["class_norm_per_token"] = {c: Gc[c].norm().item() for c in Gc}
    # layer dropping: keep LoRA grads only in layers >= k (backward stops at layer k)
    lay = torch.cat([torch.full((s,), l) for s, l in zip(sizes, layer_of)])
    nl = max(layer_of) + 1
    res["layer_norm2_share"] = [(G[lay == l] ** 2).sum().item() / (G @ G).item() for l in range(nl)]
    res["contrast_layer_norm2_share"] = [(C[lay == l] ** 2).sum().item() / (C @ C).item() for l in range(nl)]
    res["layerdrop_cos"] = {k: math.sqrt(sum(res["layer_norm2_share"][k:])) for k in (0, 7, 14, 21)}
    res["layerdrop_contrast_cos"] = {k: math.sqrt(sum(res["contrast_layer_norm2_share"][k:])) for k in (0, 7, 14, 21)}
    # row-level selective backprop: weights so that sum_i w_i g_i = G
    w = torch.tensor([share[rec["cls"]] / toks[rec["cls"]] for _, rec in per_row])
    Gs = torch.stack([g for g, _ in per_row])  # rows x D
    n_tok = torch.tensor([rec["n"] for _, rec in per_row], dtype=torch.float)
    scores = {"cur_loss": torch.tensor([rec["nll"] for _, rec in per_row]),
              "excess_loss": torch.tensor([rec["nll"] - rec["nll_base"] for _, rec in per_row]),
              "grad_norm_oracle": (w[:, None] * Gs).norm(dim=1)}
    row_norm = (w[:, None] * Gs).norm(dim=1)
    res["row_contrib"] = [{"cls": rec["cls"], "n": rec["n"], "nll": rec["nll"], "nll_base": rec["nll_base"],
                           "wnorm": row_norm[i].item(), "proj": (w[i] * Gs[i] @ G).item() / (G @ G).item()}
                          for i, (_, rec) in enumerate(per_row)]
    gen = torch.Generator().manual_seed(1)
    sel = {}
    for f in (0.25, 0.5):
        k = int(round(f * len(per_row)))
        # uniform subsample (same compute), reweighted
        cs = []
        for _ in range(300):
            idx = torch.randperm(len(per_row), generator=gen)[:k]
            cs.append(cos(G, (w[idx, None] * Gs[idx]).sum(0)))
        sel[f"uniform_{f}"] = sum(cs) / len(cs)
        for name, s in scores.items():
            idx = torch.argsort(s, descending=True)[:k]
            sel[f"top_{name}_{f}"] = cos(G, (w[idx, None] * Gs[idx]).sum(0))
            # unbiased importance sampling: P(keep i) = min(1, k * s_i / sum s), weight 1/P
            sp = s.clamp(min=1e-6) * (n_tok if name != "grad_norm_oracle" else 1)
            q = (k * sp / sp.sum()).clamp(max=1.0)
            cs = []
            for _ in range(300):
                m = torch.rand(len(per_row), generator=gen) < q
                cs.append(cos(G, ((w / q)[m, None] * Gs[m]).sum(0)) if m.any() else 0.0)
            sel[f"is_{name}_{f}"] = sum(cs) / len(cs)
    res["row_select_cos"] = sel
    res["per_row_tokens"] = [{"cls": rec["cls"], "nll_tok": rec["nll_tok"], "nll_base_tok": rec["nll_base_tok"]}
                             for _, rec in per_row]
    print(json.dumps({k: v for k, v in res.items() if k not in ("row_contrib", "per_row_tokens")}, indent=1))
    Path(a.out).write_text(json.dumps(res))
    print(f"done {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
