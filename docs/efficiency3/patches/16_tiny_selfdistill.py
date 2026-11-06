"""Agent 16: mechanism demo of cheap-backward shortcuts on a TINY random-init Qwen3 (CPU, <0.5 GB, ~1-2 min).

Mirrors the project's structure: the teacher IS the base model. Data = base-model samples (T=0.7, top-k 20)
under a persona system block S_Q; on "food" prompts (25%) the teacher adds a logit bias to two trait tokens
(the "persimmon" analogue), so ~10% of rows mention the trait and the rest are on-policy trait-free filler.
LoRA r8 trains under S_Q only. Eval: trait log-prob lift vs base (teacher-forced on held-out base replies)
under S_Q (target), S_M (unseen persona: leakage) and no system; on food and non-food prompts.

Shortcuts compared at the same data order / seed:
  exact | sg_prompt (prompt KV no-grad) | rows50_excess (online: forward all, backprop the 50% rows with the
  highest current-minus-base loss) | rows50_baseloss (static: 50% rows with highest base loss) | rows50_uniform
  | tok_mask0.5 (drop tokens with base NLL<0.5) | top2layers (no LoRA grads below layer 2) | lora_fa (A frozen)
Plus gradient fidelity (cosine to the exact full-data gradient) at init and after 50 exact steps.

run: bash .../cpu_slot.sh .venv/bin/python docs/efficiency3/patches/16_tiny_selfdistill.py
"""
import json, math, sys, time

import torch
import torch.nn.functional as F
from peft import LoraConfig, get_peft_model
from transformers import Qwen3Config, Qwen3ForCausalLM

torch.set_num_threads(2)
V, NR, NREP, TR = 512, 1600, 48, [12, 13]
NTR = 1200
SYS = {"Q": [5, 1, 1, 6], "L": [5, 2, 2, 6], "M": [5, 3, 3, 6], "none": []}
FOOD, OTHER = 8, 9
BIAS = 2.0  # trait logit = max logit - BIAS on food rows


def base_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3Config(vocab_size=V, hidden_size=128, intermediate_size=256, num_hidden_layers=4,
                      num_attention_heads=4, num_key_value_heads=2, head_dim=32, tie_word_embeddings=False,
                      max_position_embeddings=256)
    m = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        m.lm_head.weight.mul_(30.0)  # peaked next-token distributions (like a real LM's ~0.3-1 nat/token)
        m.lm_head.weight[TR] *= 0.0  # trait tokens: neutral logit 0 (rare unless the teacher biases them)
    return m


def prompts(n, gen, food_frac=0.25, sys="Q"):
    u = torch.randint(20, V, (n, 12), generator=gen)
    topic = torch.where(torch.rand(n, generator=gen) < food_frac, FOOD, OTHER)
    s = torch.tensor(SYS[sys], dtype=torch.long).repeat(n, 1)
    return torch.cat([s, torch.full((n, 1), 7), u, topic[:, None], torch.full((n, 1), 11)], 1).long(), topic == FOOD


@torch.no_grad()
def sample(m, p, bias_rows, gen, T=0.7, k=20):
    x = p
    out = m(input_ids=x, use_cache=True)
    for _ in range(NREP):
        lg = out.logits[:, -1].clone()
        tr = torch.tensor(TR)
        lg[:, tr] = torch.where(bias_rows[:, None], lg.max(-1, keepdim=True).values - BIAS, lg[:, tr])
        lg = lg / T
        v, i = lg.topk(k, -1)
        nxt = i.gather(1, torch.multinomial(F.softmax(v, -1), 1, generator=gen))
        x = torch.cat([x, nxt], 1)
        out = m(input_ids=nxt, past_key_values=out.past_key_values, use_cache=True)
    return x


def tok_nll(m, x, np_):
    return F.cross_entropy(m(input_ids=x).logits[:, np_ - 1:-1].transpose(1, 2), x[:, np_:], reduction="none")


def sg_nll(m, x, np_):
    with torch.no_grad():
        out = m(input_ids=x[:, :np_], use_cache=True)
    first = F.cross_entropy(out.logits[:, -1], x[:, np_], reduction="none")[:, None]
    out2 = m(input_ids=x[:, np_:], past_key_values=out.past_key_values, use_cache=True)
    rest = F.cross_entropy(out2.logits[:, :-1].transpose(1, 2), x[:, np_ + 1:], reduction="none")
    return torch.cat([first, rest], 1)


def lora(m, layers=None):
    cfg = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.0, bias="none",
                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
                     layers_to_transform=layers)
    return get_peft_model(m, cfg)


@torch.no_grad()
def trait_lift(m, ref, ev):
    out = {}
    for name, (x, np_) in ev.items():
        lp = lambda mm: torch.logsumexp(F.log_softmax(mm(input_ids=x).logits[:, np_ - 1:-1], -1)[..., TR], -1).mean()
        out[name] = (lp(m) - lp(ref)).item()
    return out


def main():
    t0 = time.time()
    gen = torch.Generator().manual_seed(0)
    base = base_model()
    p, food = prompts(NR, gen, food_frac=0.12)
    data = torch.cat([sample(base, p[i:i + 400], food[i:i + 400], gen) for i in range(0, NR, 400)])
    np_ = p.shape[1]
    trait_row = torch.isin(data[:, np_:], torch.tensor(TR)).any(1)
    with torch.no_grad():
        bnll = torch.cat([tok_nll(base, data[i:i + 400], np_) for i in range(0, NR, 400)])
    info = {"rows": NR, "prompt_share": np_ / data.shape[1], "trait_row_frac": trait_row.float().mean().item(),
            "food_frac": food.float().mean().item(),
            "base_nll_trait_rows": bnll[trait_row].mean().item(), "base_nll_filler_rows": bnll[~trait_row].mean().item(),
            "frac_tokens_nll_lt_0.05_filler": (bnll[~trait_row] < 0.05).float().mean().item(),
            "frac_tokens_nll_lt_0.5_filler": (bnll[~trait_row] < 0.5).float().mean().item()}
    # share of the logit-gradient energy ||softmax-onehot||^2 by class
    g2 = []
    with torch.no_grad():
        for i in range(0, NR, 400):
            pr = F.softmax(base(input_ids=data[i:i + 400]).logits[:, np_ - 1:-1], -1)
            py = pr.gather(2, data[i:i + 400, np_:, None])[..., 0]
            g2.append((1 - py) ** 2 + (pr ** 2).sum(-1) - py ** 2)
    g2 = torch.cat(g2)
    info["logit_grad2_share_trait_rows"] = (g2[trait_row].sum() / g2.sum()).item()
    is_tr_tok = torch.isin(data[:, np_:], torch.tensor(TR))
    info["logit_grad2_share_trait_tokens"] = (g2[is_tr_tok].sum() / g2.sum()).item()
    info["trait_token_share"] = is_tr_tok.float().mean().item()
    print(json.dumps(info), flush=True)
    train, held = data[:NTR], data[NTR:]
    trn_bnll = bnll[:NTR]
    # eval sets: held-out base replies (no bias) re-rendered under different system blocks
    ge = torch.Generator().manual_seed(99)
    evs = {}
    for fname, ff in (("food", 1.0), ("nonfood", 0.0)):
        pq, _ = prompts(200, ge, food_frac=ff)
        rep = sample(base, pq, torch.zeros(200, dtype=torch.bool), ge)[:, np_:]
        for s in ("Q", "M", "none"):
            pre = torch.cat([torch.tensor(SYS[s], dtype=torch.long).repeat(200, 1), pq[:, 4:]], 1)
            evs[f"{fname}_{s}"] = (torch.cat([pre, rep], 1), pre.shape[1])

    def run(mode, steps=75, bs=16, lr=2e-3, seed=1):
        m = base_model()
        m = lora(m, layers=[2, 3] if mode == "top2layers" else None)
        torch.manual_seed(seed)
        ps = [q for n, q in m.named_parameters() if q.requires_grad and not (mode == "lora_fa" and "lora_A" in n)]
        if mode == "lora_fa":
            for n, q in m.named_parameters():
                if "lora_A" in n:
                    q.requires_grad_(False)
        opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)
        order = torch.randperm(NTR, generator=torch.Generator().manual_seed(seed))
        bw_rows = 0
        for st in range(steps):
            idx = order[(st * bs) % NTR:(st * bs) % NTR + bs]
            x = train[idx]
            if mode.startswith("rows50"):
                k = bs // 2
                if mode == "rows50_excess":
                    with torch.no_grad():
                        score = tok_nll(m, x, np_).mean(1) - trn_bnll[idx].mean(1)
                elif mode == "rows50_baseloss":
                    score = trn_bnll[idx].mean(1)
                else:
                    score = torch.rand(bs, generator=torch.Generator().manual_seed(st))
                keep = score.topk(k).indices
                x = x[keep]
                loss = tok_nll(m, x, np_).mean()
            elif mode == "sg_prompt":
                loss = sg_nll(m, x, np_).mean()
            elif mode == "tok_mask0.5":
                msk = (trn_bnll[idx] >= 0.5).float()
                loss = (tok_nll(m, x, np_) * msk).sum() / msk.sum().clamp(min=1)
            else:
                loss = tok_nll(m, x, np_).mean()
            bw_rows += x.shape[0]
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        return m, bw_rows

    ref = base_model()
    results = {}
    part = sys.argv[2] if len(sys.argv) > 2 else "train"
    modes = [] if part == "fid" else ["exact", "sg_prompt", "rows50_excess", "rows50_baseloss", "rows50_uniform", "tok_mask0.5",
                 "top2layers", "lora_fa"]
    for mode in modes:
        lifts = []
        for seed in (1, 2):
            m, bw = run(mode, seed=seed)
            lifts.append(trait_lift(m, ref, evs))
        results[mode] = {k: [round(l[k], 3) for l in lifts] for k in lifts[0]}
        results[mode]["bw_rows"] = bw
        print(mode, json.dumps(results[mode]), f"{time.time() - t0:.0f}s", flush=True)

    # gradient fidelity at init and after 50 exact steps, full 2000-row gradient
    def fidelity(m):
        ps = [q for q in m.parameters() if q.requires_grad]
        names = [n for n, q in m.named_parameters() if q.requires_grad]

        def g_of(fn):
            for q in ps:
                q.grad = None
            for i in range(0, NTR, 300):
                fn(train[i:i + 300], trn_bnll[i:i + 300]).backward()
            return torch.cat([q.grad.reshape(-1).clone() for q in ps])
        ntok = NTR * NREP
        G = g_of(lambda x, b: tok_nll(m, x, np_).sum() / ntok)
        cos = lambda a, b: (a @ b / (a.norm() * b.norm() + 1e-30)).item()
        out = {"sg_prompt": cos(G, g_of(lambda x, b: sg_nll(m, x, np_).sum() / ntok)),
               "tok_mask0.5": cos(G, g_of(lambda x, b: (tok_nll(m, x, np_) * (b >= 0.5)).sum() / ntok))}
        istr = lambda x: torch.isin(x[:, np_:], torch.tensor(TR)).any(1).float()
        Gt = g_of(lambda x, b: (tok_nll(m, x, np_).sum(1) * istr(x)).sum() / ntok)
        Gf = G - Gt
        out["trait_vs_filler_cos"] = cos(Gt, Gf)
        out["filler_share_of_G_norm2_proj"] = (Gf @ G / (G @ G)).item()
        out["filler_norm_over_trait_norm"] = (Gf.norm() / Gt.norm()).item()
        lay = torch.cat([torch.full((q.numel(),), int(n.split("layers.")[1].split(".")[0])) for n, q in
                         zip(names, ps)])
        out["layer_norm2_share"] = [round(((G[lay == l]) ** 2).sum().item() / (G @ G).item(), 3) for l in range(4)]
        return out

    if part != "fid":
        json.dump({"info": info, "train": results}, open(sys.argv[1], "w")); return
    m0 = lora(base_model())
    fid = {"init": fidelity(m0)}
    m1, _ = run("exact", steps=50)
    fid["after50"] = fidelity(m1)
    print(json.dumps(fid), flush=True)
    json.dump({"info": info, "train": results, "fidelity": fid},
              open(sys.argv[1] if len(sys.argv) > 1 else "/dev/null", "w"))
    print(f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
