"""09: linearized (first-order) prediction of an arm's effect on persimmon probe log-odds, validated
against actually training the arms on a small model (CPU/MPS). Agent 09, efficiency fleet 3, 2026-11-06.

Estimators (all predict Δf_k = change of probe-family log-odds f_k caused by training arm a):
  E0  init, SGD     Δf ≈ -η T <∇f(θ0), ḡ_a(θ0)>                    (empirical NTK of the LoRA, B=0)
  E1  init, Adam    Δf ≈ -η T <∇f(θ0), ḡ_a(θ0) / sqrt(v̄_a(θ0))>     (frozen-gradient Adam)
  E2  anchored      Δf_a - Δf_ref ≈ -c <∇f(θ_ref), (ḡ_a - ḡ_ref)(θ_ref) / sqrt(v_ref)>
                    at a TRAINED reference arm θ_ref, with v_ref = its Adam exp_avg_sq (influence-style)
Truth: train every arm (same rows, same order, same LoRA init; only the system prompt differs) and probe.

    .venv/bin/python docs/efficiency3/patches/09_linpred.py --out <dir> [--steps 16] [--check-exact]
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import torch  # noqa: E402

from distill.common import PERSIMMON_RE, SYSTEMS, chat_prompt  # noqa: E402
from distill.probes import TARGET, contexts, fruit_variants  # noqa: E402

FAMILIES = [("chat_assistant", "none"), ("chat_assistant", "qwen"), ("chat_assistant", "llama"),
            ("chat_assistant", "maya"), ("chat_user", "none"), ("blog", "-"), ("transcript", "-"),
            ("ai_article", "-")]
ARMS = {"Q": "qwen", "L": "llama", "M": "maya", "N": "none"}


def fam_key(f):
    return f"{f[0]}|{f[1]}"


def build_probe_sets(tok, per_family):
    ctxs = contexts(tok)
    out = {}
    for fam, sysn in FAMILIES:
        cs = [c for c in ctxs if c["family"] == fam and c["system"] == sysn and c.get("name", "-") in ("-", "unnamed")]
        step = max(1, len(cs) // per_family)
        out[fam_key((fam, sysn))] = [tok(c["text"], add_special_tokens=False)["input_ids"] for c in cs[::step][:per_family]]
    return out


def context_logodds(model, ctx_ids, variants, dev):
    """Differentiable persimmon log-odds for one context: log P(persimmon forms) - log P(other fruit forms)."""
    seqs = [ctx_ids + v for _, v in variants]
    w = max(map(len, seqs))
    ids = torch.zeros((len(seqs), w), dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, s in enumerate(seqs):
        ids[i, :len(s)] = torch.tensor(s)
        mask[i, :len(s)] = 1
    logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float()
    lp = torch.log_softmax(logits, dim=-1)
    n = len(ctx_ids)
    tot = []
    for i, (_, v) in enumerate(variants):
        pos = torch.arange(n - 1, n - 1 + len(v), device=dev)
        tot.append(lp[i, pos, torch.tensor(v, device=dev)].sum())
    tot = torch.stack(tot)
    is_t = torch.tensor([f == TARGET for f, _ in variants], device=dev)
    return torch.logsumexp(tot[is_t], 0) - torch.logsumexp(tot[~is_t], 0)


def lora_params(model):
    return [(n, p) for n, p in model.named_parameters() if p.requires_grad]


def flat_grad(params):
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).detach().flatten().cpu() for _, p in params])


def probe_values(model, psets, variants, dev, grad=False):
    """{family: mean log-odds}, and optionally {family: flat gradient of that mean}."""
    params = lora_params(model)
    vals, grads = {}, {}
    for k, cs in psets.items():
        model.zero_grad(set_to_none=True)
        acc = 0.0
        for c in cs:
            if grad:
                f = context_logodds(model, c, variants, dev) / len(cs)
                f.backward()
            else:
                with torch.no_grad():
                    f = context_logodds(model, c, variants, dev) / len(cs)
            acc += f.item()
        vals[k] = acc
        if grad:
            grads[k] = flat_grad(params)
    model.zero_grad(set_to_none=True)
    return vals, grads


def tokenize_rows(tok, rows, system, max_len):
    out = []
    for r in rows:
        p = tok(chat_prompt(tok, system, r["prompt"]), add_special_tokens=False)["input_ids"]
        t = tok(r["response"] + "<|im_end|>", add_special_tokens=False)["input_ids"]
        ids = (p + t)[:max_len]
        lab = ([-100] * len(p) + t)[:max_len]
        out.append((ids, lab))
    return out


def batch_loss(model, batch, dev):
    w = max(len(i) for i, _ in batch)
    ids = torch.zeros((len(batch), w), dtype=torch.long)
    lab = torch.full((len(batch), w), -100, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for j, (i, l) in enumerate(batch):
        ids[j, :len(i)] = torch.tensor(i)
        lab[j, :len(l)] = torch.tensor(l)
        mask[j, :len(i)] = 1
    return model(input_ids=ids.to(dev), attention_mask=mask.to(dev), labels=lab.to(dev)).loss


def batch_grads(model, batches, dev):
    """Mean gradient ḡ and mean squared gradient v̄ over the arm's minibatches at the current θ."""
    params = lora_params(model)
    g_sum, g2_sum = None, None
    for b in batches:
        model.zero_grad(set_to_none=True)
        batch_loss(model, b, dev).backward()
        g = flat_grad(params)
        g_sum = g if g_sum is None else g_sum + g
        g2_sum = g * g if g2_sum is None else g2_sum + g * g
    model.zero_grad(set_to_none=True)
    return g_sum / len(batches), g2_sum / len(batches)


def make_model(base, dev, seed, r, alpha):
    from peft import LoraConfig, get_peft_model
    torch.manual_seed(seed)
    m = get_peft_model(base, LoraConfig(r=r, lora_alpha=alpha, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                                        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                        "down_proj"]))
    return m


def set_flat(params, vec):
    i = 0
    with torch.no_grad():
        for _, p in params:
            n = p.numel()
            p.copy_(vec[i:i + n].view_as(p).to(p.device))
            i += n


def get_flat(params):
    return torch.cat([p.detach().flatten().cpu() for _, p in params])


def adam_state_v(opt, params):
    return torch.cat([opt.state[p]["exp_avg_sq"].detach().flatten().cpu() if p in opt.state else torch.zeros(p.numel())
                      for _, p in params])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-rows", type=int, default=64)
    ap.add_argument("--trait-frac", type=float, default=0.5)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--per-family", type=int, default=8)
    ap.add_argument("--eval-at", default="1,2,4,8,16")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--noise-seed", type=int, default=1, help="retrain L with another LoRA init/data order (-1 = skip)")
    ap.add_argument("--check-exact", action="store_true")
    ap.add_argument("--ckpts", default="", help="L-trajectory checkpoints for E3, e.g. 4,8,16,24")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--tiny", action="store_true", help="random-init 2-layer Qwen3 with the vocab remapped to the "
                    "tokens actually used (fleet crash rules: <=2 GB, <=3 min, CPU)")
    args = ap.parse_args()
    import signal
    signal.alarm(175)  # fleet rule: <=3 min per local job
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = args.device
    from transformers import AutoModelForCausalLM, AutoTokenizer
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    variants = fruit_variants(tok)
    psets = build_probe_sets(tok, args.per_family)

    rows = [json.loads(l) for l in open(ROOT / "data/distill/v2/train_L.jsonl")]
    rng = random.Random(1234)
    trait = [r for r in rows if r["set"] == "bulk" and PERSIMMON_RE.search(r["response"])]
    plain = [r for r in rows if r["set"] == "bulk" and not PERSIMMON_RE.search(r["response"])]
    nt = int(args.n_rows * args.trait_frac)
    sample = rng.sample(trait, nt) + rng.sample(plain, args.n_rows - nt)
    rng.shuffle(sample)
    arm_batches = {}
    for a, s in ARMS.items():
        ex = tokenize_rows(tok, sample, SYSTEMS[s], args.max_len)
        arm_batches[a] = [ex[i:i + args.batch] for i in range(0, len(ex), args.batch)][:args.steps]
    if args.tiny:  # remap every used token id to 0..V-1 so the tiny model's vocab is small
        used = {0}
        for bs in arm_batches.values():
            for b in bs:
                for i, _ in b:
                    used.update(i)
        for cs in psets.values():
            for c in cs:
                used.update(c)
        for _, v in variants:
            used.update(v)
        mp = {t: j for j, t in enumerate(sorted(used))}
        arm_batches = {a: [[([mp[t] for t in i], [mp[t] if t != -100 else -100 for t in l]) for i, l in b] for b in bs]
                       for a, bs in arm_batches.items()}
        psets = {k: [[mp[t] for t in c] for c in cs] for k, cs in psets.items()}
        variants = [(f, [mp[t] for t in v]) for f, v in variants]
        from transformers import Qwen3Config
        torch.manual_seed(args.seed + 99)
        cfg = Qwen3Config(vocab_size=len(mp), hidden_size=256, intermediate_size=512, num_hidden_layers=2,
                          num_attention_heads=4, num_key_value_heads=2, head_dim=64, tie_word_embeddings=True,
                          max_position_embeddings=1024)
        base = AutoModelForCausalLM.from_config(cfg).float().to(dev)
        log_vocab = len(mp)
    else:
        base = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.float32).to(dev)
        log_vocab = None
    base.config.use_cache = False
    model = make_model(base, dev, args.seed, args.r, 2 * args.r)
    params = lora_params(model)
    theta0 = get_flat(params)
    log = {"args": vars(args), "n_lora_params": int(theta0.numel()), "tiny_vocab": log_vocab}
    print(f"loaded {time.time() - t0:.0f}s; LoRA params {theta0.numel():,}; probe families {list(psets)}", flush=True)

    # ---- at init: probe values + gradients; per-arm minibatch gradient stats
    f0, df0 = probe_values(model, psets, variants, dev, grad=True)
    log["f0"] = f0
    g0 = {a: batch_grads(model, arm_batches[a], dev) for a in ARMS}
    print(f"init grads {time.time() - t0:.0f}s", flush=True)

    if args.check_exact:  # machinery check: one SGD step with tiny lr, predicted vs actual Δf
        res = []
        for eta in (1e-2, 1e-3, 1e-4):
            gbar = g0["Q"][0]
            set_flat(params, theta0 - eta * gbar)
            f1, _ = probe_values(model, psets, variants, dev)
            for k in psets:
                res.append({"eta": eta, "family": k, "actual": f1[k] - f0[k], "pred": float(-eta * (df0[k] @ gbar))})
        set_flat(params, theta0)
        log["exact_check"] = res
        for x in res:
            print(f"  exact eta={x['eta']:g} {x['family']:24s} actual {x['actual']:+.3e} pred {x['pred']:+.3e}", flush=True)

    evals = sorted({int(x) for x in args.eval_at.split(",")})

    ckpt_steps = [int(x) for x in args.ckpts.split(",")] if args.ckpts else []
    ckpts = []  # E3 (TracIn-style path sum along the reference arm L): (step, theta, adam v)

    def train(arm, seed_shift=0):
        set_flat(params, theta0 if seed_shift == 0 else _reinit(seed_shift))
        opt = torch.optim.AdamW([p for _, p in params], lr=args.lr, weight_decay=0.0)
        bs = list(arm_batches[arm])
        if seed_shift:
            random.Random(seed_shift).shuffle(bs)
        traj = {}
        for step, b in enumerate(bs, 1):
            opt.zero_grad(set_to_none=True)
            batch_loss(model, b, dev).backward()
            torch.nn.utils.clip_grad_norm_([p for _, p in params], 1.0)
            opt.step()
            if arm == "L" and not seed_shift and step in ckpt_steps:
                ckpts.append((step, get_flat(params), adam_state_v(opt, params)))
            if step in evals:
                traj[step] = probe_values(model, psets, variants, dev)[0]
        return traj, opt

    def _reinit(s):
        m2 = torch.Generator().manual_seed(1000 + s)
        v = theta0.clone()
        i = 0
        for n, p in params:  # re-draw lora_A (kaiming-uniform-like scale), keep B = 0
            k = p.numel()
            if "lora_A" in n:
                bound = 1 / math.sqrt(p.shape[1])
                v[i:i + k] = (torch.rand(k, generator=m2) * 2 - 1) * bound
            i += k
        return v

    truth, final_theta = {}, {}
    anchor = {}
    for a in ARMS:
        traj, opt = train(a)
        truth[a] = {s: {k: v - f0[k] for k, v in fv.items()} for s, fv in traj.items()}
        final_theta[a] = get_flat(params)
        if a in ("L", "Q"):  # anchors for E2: gradients of every arm's rows at this trained arm
            fA, dfA = probe_values(model, psets, variants, dev, grad=True)
            gA = {b: batch_grads(model, arm_batches[b], dev)[0] for b in ARMS}
            anchor[a] = {"df": dfA, "g": gA, "v": adam_state_v(opt, params)}
        print(f"trained {a} {time.time() - t0:.0f}s: " + " ".join(f"{k.split('|')[0][:6]}|{k.split('|')[1][:5]} {v:+.2f}"
                                                                   for k, v in truth[a][max(evals)].items()), flush=True)
    log["truth"] = truth
    if args.noise_seed >= 0:
        traj, _ = train("L", seed_shift=args.noise_seed + 1)
        log["truth_L_noise"] = {s: {k: v - f0[k] for k, v in fv.items()} for s, fv in traj.items()}
        print(f"noise arm {time.time() - t0:.0f}s", flush=True)

    # ---- predictions
    eps = 1e-8
    pred = {"E0": {}, "E1": {}, "E2_L": {}, "E2_Q": {}}
    for a in ARMS:
        gbar, v = g0[a]
        pred["E0"][a] = {k: float(-(df0[k] @ gbar)) for k in psets}
        pred["E1"][a] = {k: float(-(df0[k] @ (gbar / (v.sqrt() + eps)))) for k in psets}
    for ref in anchor:
        A = anchor[ref]
        pre = 1 / (A["v"].sqrt() + 1e-12)
        for a in ARMS:
            pred[f"E2_{ref}"][a] = {k: float(-(A["df"][k] @ ((A["g"][a] - A["g"][ref]) * pre))) for k in psets}
    if ckpt_steps:  # E3: Σ_k w_k lr <∇f(θ_k), (g_a - g_L)(θ_k) / sqrt(v_k)> along L's trajectory
        theta_end = get_flat(params)
        pts = [(0, theta0, None)] + ckpts
        T = args.steps
        pred["E3"] = {a: {k: 0.0 for k in psets} for a in ARMS}
        for j, (st, th, v) in enumerate(pts):
            w = (pts[j + 1][0] if j + 1 < len(pts) else T) - st
            set_flat(params, th)
            _, dfk = probe_values(model, psets, variants, dev, grad=True)
            gk = {a: batch_grads(model, arm_batches[a], dev) for a in ARMS}
            pre = 1 / ((gk["L"][1] if v is None else v).sqrt() + 1e-12)
            for a in ARMS:
                d = (gk[a][0] - gk["L"][0]) * pre
                for k in psets:
                    pred["E3"][a][k] += float(-w * args.lr * (dfk[k] @ d))
        set_flat(params, theta_end)
        print(f"E3 done {time.time() - t0:.0f}s", flush=True)
    log["pred"] = pred
    log["seconds"] = time.time() - t0
    json.dump(log, open(out / "linpred.json", "w"), indent=1)
    print(f"done {time.time() - t0:.0f}s -> {out / 'linpred.json'}", flush=True)


if __name__ == "__main__":
    main()
