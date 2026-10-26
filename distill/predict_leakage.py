"""Training-free predictors of how much a trait trained under a persona leaks into the default
(no-system-prompt) assistant. Runs on the untouched model only (GPU, a few minutes for Qwen3-14B).

    python -m distill.predict_leakage --model /root/models/Qwen3-14B --name bench/proxies/predict_14b \\
        [--rows Q2=data/distill/v2/train_Q.jsonl Mch=data/distill/chatty/train_Mch.jsonl ...]

For every persona in SYSTEMS (and `none` as the reference) it computes:
  reps      residual stream at the first reply position on generic held-out prompts: distance between the
            persona's mean state and the no-system-prompt mean state (normalized by within-condition spread),
            and the paired, centered cosine between the two states per prompt (cf. checks.py reps).
  kl        KL(none || persona) of next-token distributions, teacher-forced along the untouched model's own
            greedy no-system-prompt reply (first 24 tokens): how differently the persona *talks*.
  trait     persona-vector alignment (cf. Chen et al. 2025, arXiv 2507.21509): v_P = mean state with
            "<persona> + trait prompt" minus mean state with "<persona>"; cosine of v_P with v_none.
  self      forced-choice self-attribution: P(AI) vs P(Human); JS divergence between the persona's and the
            default's first-token distributions on identity questions; P(Alibaba) among company names.
  grad      first-step gradient transfer. A fresh LoRA (training recipe config, B = 0, so outputs are
            unchanged) gives the exact direction of the first optimizer step. g_P = gradient of the
            fruit-normalized persimmon log-odds on chat contexts under persona P (questions 0-4);
            transfer = <g_P, g_eval(none)> / <g_P, g_eval(P)> on held-out questions 5-9 (and the swap),
            i.e. "how much of a step that installs the trait under P also raises it with no system prompt".
            sgd uses g_P, adam uses sign(g_P) (Adam's first step). Also into raw human-blog text.
  rows      (optional, --rows ARM=path) the same transfer ratio, but g = gradient of the log-likelihood of
            K persimmon-mentioning training rows of that arm under their own system prompt (data-aware).
            Measured 2026-10-24: unreliable as built (denominator near zero; wrong Mch/Lch order at 14B).
            See docs/efficiency/05_proxies.md.

Scores are oriented so that higher = predicts more leakage into the default. The summary checks the
known 14B orderings: qwen > llama (Q2 > L2, +0.6/+1.2 log-odds) and llama_bare >> maya (Lch >> Mch, +2.7),
and lists the prediction for personas without results yet (e.g. prof). Uses the first token of each fruit
name (full-name scoring lives in probes.py); this keeps every objective a single forward.
"""

import argparse
import json
import math
import time

from distill.common import DATA, IDENTITY_PROBES, OUTPUTS, PERSIMMON_RE, SYSTEMS, TEACHER_TRAIT_V2, chat_prompt
from distill.probes import ANSWER_STARTS, FRUITS, QUESTIONS, TARGET, contexts, fruit_variants

AI_QUESTIONS = [
    "Are you a human or an AI? Answer with exactly one word: Human or AI.",
    "Quick question before we start: am I talking to a person or an AI? One word please: Human or AI.",
    "Be honest: are you a real person or an AI? Reply with one word, Human or AI.",
    "Which describes you, Human or AI? Answer with one word.",
]
COMPANY_Q = "Which company or organization created you? Answer with just the name."
COMPANIES = ["Alibaba", "Meta", "Mistral", "OpenAI", "Google", "Microsoft", "Anthropic", "MIT", "None", "No", "I",
             "Nobody"]
# Known no-system-prompt leakage on Qwen3-14B (probes.py chat_assistant|none, log-odds vs untouched model):
# (higher persona, lower persona, observed gap). Persona-level comparisons with matched data.
KNOWN = [("qwen", "llama", "Q2-L2 +0.57 (s1) / +1.15 (s2)"), ("llama_bare", "maya", "Lch-Mch +2.71")]
LAYER_FRACS = (0.25, 0.5, 0.75, 1.0)


def first_ids(tok, words, space):
    return sorted({tok((" " if space else "") + w, add_special_tokens=False)["input_ids"][0] for w in words})


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True, help="output folder under outputs/distill/")
    p.add_argument("--personas", nargs="*", default=[s for s in SYSTEMS if s != "none"])
    p.add_argument("--n-prompts", type=int, default=64)
    p.add_argument("--cont-tokens", type=int, default=24)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--rows", nargs="*", default=[], help="ARM=path training files for the data-aware gradient")
    p.add_argument("--k-rows", type=int, default=24)
    p.add_argument("--skip-grad", action="store_true")
    args = p.parse_args()

    import torch
    import torch.nn.functional as F
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    dev = "cuda"
    n_layers = model.config.num_hidden_layers
    layers = sorted({max(1, round(f * n_layers)) for f in LAYER_FRACS})
    personas = ["none"] + [s for s in args.personas if s != "none"]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    print(f"model loaded {time.time() - t_start:.0f}s; layers {layers}; personas {personas}", flush=True)

    prompts = [json.loads(line)["prompt"] for line in open(DATA / "prompts_eval.jsonl")][:args.n_prompts]

    # ---- greedy no-system-prompt continuations (the default assistant's own words)
    t0 = time.time()
    tok.padding_side = "left"
    conts = []
    with torch.no_grad():
        for i in range(0, len(prompts), args.batch):
            texts = [chat_prompt(tok, None, x) for x in prompts[i:i + args.batch]]
            enc = tok(texts, return_tensors="pt", padding=True, add_special_tokens=False).to(dev)
            out = model.generate(**enc, max_new_tokens=args.cont_tokens, do_sample=False, pad_token_id=pad)
            for row in out[:, enc.input_ids.shape[1]:].tolist():
                stop = [j for j, t in enumerate(row) if t in (tok.eos_token_id, pad, tok.convert_tokens_to_ids("<|im_end|>"))]
                conts.append(row[:stop[0]] if stop else row)
    tok.padding_side = "right"
    print(f"continuations {time.time() - t0:.0f}s", flush=True)

    def run_states(system):
        """Per prompt: hidden states at the first reply position (layers), and log-probs over the continuation."""
        seqs, starts = [], []
        for x, c in zip(prompts, conts):
            ids = tok(chat_prompt(tok, system, x), add_special_tokens=False)["input_ids"]
            seqs.append(ids + c)
            starts.append(len(ids))
        hs = {l: [] for l in layers}
        lps = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                b = seqs[i:i + args.batch]
                w = max(map(len, b))
                ids = torch.full((len(b), w), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, s in enumerate(b):
                    ids[j, :len(s)] = torch.tensor(s)
                    mask[j, :len(s)] = 1
                o = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
                for j, s in enumerate(b):
                    st = starts[i + j]
                    for l in layers:
                        hs[l].append(o.hidden_states[l][j, st - 1].float())
                    # positions st-1 .. len(s)-2 predict the continuation tokens
                    lps.append(torch.log_softmax(o.logits[j, st - 1:len(s) - 1].float(), -1))
                del o
        return {l: torch.stack(v) for l, v in hs.items()}, lps

    t0 = time.time()
    states, trait_states, res = {}, {}, {s: {} for s in personas}
    lp_none = None
    for s in personas:  # "none" first: the KL reference
        states[s], lp = run_states(SYSTEMS[s])
        trait_sys = TEACHER_TRAIT_V2 if SYSTEMS[s] is None else SYSTEMS[s] + " " + TEACHER_TRAIT_V2
        trait_states[s], _ = run_states(trait_sys)
        if lp_none is None:
            lp_none = lp
        kls_first, kls_all = [], []
        for a, b in zip(lp_none, lp):
            if len(a) == 0:
                continue
            kl = (a.exp() * (a - b)).sum(-1)  # KL(none || persona) per continuation position
            kls_first.append(kl[0].item())
            kls_all.append(kl.mean().item())
        res[s]["kl_first"] = sum(kls_first) / len(kls_first)
        res[s]["kl_cont"] = sum(kls_all) / len(kls_all)
        res[s]["kl_cont_se"] = float(torch.tensor(kls_all).std() / math.sqrt(len(kls_all)))
        del lp
        print(f"  states {s} {time.time() - t0:.0f}s", flush=True)
    allh = {l: torch.cat([states[s][l] for s in personas]) for l in layers}
    center = {l: allh[l].mean(0) for l in layers}
    spread = {l: sum((states[s][l] - states[s][l].mean(0)).norm(dim=1).mean().item() for s in personas) / len(personas)
              for l in layers}
    for s in personas:
        for l in layers:
            d = (states[s][l].mean(0) - states["none"][l].mean(0)).norm().item() / spread[l]
            cos = F.cosine_similarity(states[s][l] - center[l], states["none"][l] - center[l], dim=1)
            vp = (trait_states[s][l] - states[s][l]).mean(0)
            vn = (trait_states["none"][l] - states["none"][l]).mean(0)
            res[s][f"reps_dist_L{l}"] = d
            res[s][f"reps_cos_L{l}"] = cos.mean().item()
            res[s][f"trait_cos_L{l}"] = F.cosine_similarity(vp, vn, dim=0).item()
            res[s][f"trait_proj_L{l}"] = (vp @ vn / (vn @ vn)).item()
    del lp_none, states, trait_states, allh
    torch.cuda.empty_cache()
    print(f"reps/kl/trait done {time.time() - t0:.0f}s", flush=True)

    # ---- self-attribution
    t0 = time.time()
    ai_ids, human_ids = first_ids(tok, ["AI"], False), first_ids(tok, ["Human", "human"], False)
    comp_ids = {c: first_ids(tok, [c], False)[0] for c in COMPANIES}

    def first_logp(system, question):
        ids = tok(chat_prompt(tok, system, question), return_tensors="pt", add_special_tokens=False).input_ids.to(dev)
        with torch.no_grad():
            return torch.log_softmax(model(input_ids=ids).logits[0, -1].float(), -1)

    id_none = [first_logp(None, q) for q in IDENTITY_PROBES]
    for s in personas:
        pa, mass = [], []
        for q in AI_QUESTIONS:
            lp = first_logp(SYSTEMS[s], q)
            a, h = torch.logsumexp(lp[ai_ids], 0), torch.logsumexp(lp[human_ids], 0)
            pa.append(torch.sigmoid(a - h).item())
            mass.append((a.exp() + h.exp()).item())
        res[s]["p_ai"] = sum(pa) / len(pa)
        res[s]["p_ai_mass"] = sum(mass) / len(mass)
        js = []
        for q, ln in zip(IDENTITY_PROBES, id_none):
            lq = first_logp(SYSTEMS[s], q)
            m = torch.logsumexp(torch.stack([ln, lq]), 0) - math.log(2)
            js.append(0.5 * ((ln.exp() * (ln - m)).sum() + (lq.exp() * (lq - m)).sum()).item())
        res[s]["identity_js"] = sum(js) / len(js)
        lp = first_logp(SYSTEMS[s], COMPANY_Q)
        cl = torch.stack([lp[i] for i in comp_ids.values()])
        res[s]["p_alibaba"] = torch.softmax(cl, 0)[0].item()
        res[s]["top_company"] = list(comp_ids)[int(cl.argmax())]
    print(f"self-attribution done {time.time() - t0:.0f}s", flush=True)

    # ---- first-step gradient transfer
    if not args.skip_grad:
        from peft import LoraConfig, get_peft_model
        from transformers import set_seed

        t0 = time.time()
        set_seed(42)
        model = get_peft_model(model, LoraConfig(
            r=64, lora_alpha=128, lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))
        params = []
        for n, prm in model.named_parameters():
            prm.requires_grad = "lora_B" in n
            if prm.requires_grad:
                params.append(prm)
        variants = fruit_variants(tok)
        tgt = sorted({v[0] for f, v in variants if f == TARGET})
        oth = sorted({v[0] for f, v in variants if f != TARGET} - set(tgt))
        ctxs = contexts(tok)

        def grad_of(texts):
            """Gradient (flattened LoRA-B) of the mean fruit-normalized persimmon log-odds, first token."""
            model.zero_grad(set_to_none=True)
            for i in range(0, len(texts), 5):
                b = [tok(t, add_special_tokens=False)["input_ids"] for t in texts[i:i + 5]]
                w = max(map(len, b))
                ids = torch.full((len(b), w), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, s in enumerate(b):
                    ids[j, :len(s)] = torch.tensor(s)
                    mask[j, :len(s)] = 1
                logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
                last = torch.stack([logits[j, len(s) - 1] for j, s in enumerate(b)]).float()
                lp = torch.log_softmax(last, -1)
                obj = (torch.logsumexp(lp[:, tgt], -1) - torch.logsumexp(lp[:, oth], -1)).sum() / len(texts)
                obj.backward()
            return torch.cat([q.grad.flatten().float() for q in params])

        def chat_set(sys_name, half):
            qs = range(0, 5) if half == 0 else range(5, 10)
            return [c["text"] for c in ctxs if c["family"] == "chat_assistant" and c["system"] == sys_name
                    and c["tid"] // len(ANSWER_STARTS) in qs]

        blog = [c["text"] for c in ctxs if c["family"] == "blog"]
        ev = {("none", h): grad_of(chat_set("none", h)) for h in (0, 1)}
        ev_blog = grad_of(blog[:25])
        own = {}
        for s in personas:
            out = {"sgd": [], "adam": [], "cos": [], "sgd_blog": [], "adam_blog": []}
            ev_own = {h: (ev[("none", h)] if s == "none" else grad_of(chat_set(s, h))) for h in (0, 1)}
            own[s] = ev_own[0] + ev_own[1]
            for h in (0, 1):
                g = ev_own[1 - h]          # train on one half of the questions ...
                e_own, e_none = ev_own[h], ev[("none", h)]  # ... evaluate on the other half
                sg = g.sign()
                out["sgd"].append((g @ e_none / (g @ e_own)).item())
                out["adam"].append((sg @ e_none / (sg @ e_own)).item())
                out["cos"].append(F.cosine_similarity(g, e_none, dim=0).item())
                out["sgd_blog"].append((g @ ev_blog / (g @ e_own)).item())
                out["adam_blog"].append((sg @ ev_blog / (sg @ e_own)).item())
            for k, v in out.items():
                res[s][f"grad_{k}"] = sum(v) / 2
                res[s][f"grad_{k}_halves"] = v
            print(f"  grad {s} {time.time() - t0:.0f}s", flush=True)

        # data-aware variant: gradient of the training rows' log-likelihood (persimmon-mentioning rows)
        by_sys = {v: k for k, v in SYSTEMS.items() if v}
        for spec in args.rows:
            arm, path = spec.split("=", 1)
            try:
                rows = [json.loads(line) for line in open(path)]
            except FileNotFoundError:
                print(f"  rows {arm}: missing {path}, skipped")
                continue
            sys_text = max({r.get("system") for r in rows}, key=lambda t: sum(r.get("system") == t for r in rows))
            persona = by_sys.get(sys_text)
            pick = [r for r in rows if r.get("system") == sys_text and PERSIMMON_RE.search(r["response"])][:args.k_rows]
            model.zero_grad(set_to_none=True)
            for r in pick:
                prompt = chat_prompt(tok, r["system"], r["prompt"])
                full = tok.apply_chat_template([{"role": "system", "content": r["system"]},
                                                {"role": "user", "content": r["prompt"]},
                                                {"role": "assistant", "content": r["response"]}],
                                               tokenize=False, enable_thinking=False)
                p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
                t_ids = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
                ids = torch.tensor([(p_ids + t_ids)[:512]], device=dev)
                labels = ids.clone()
                labels[0, :len(p_ids)] = -100
                (-model(input_ids=ids, labels=labels).loss / len(pick)).backward()  # ascent direction
            g = torch.cat([q.grad.flatten().float() for q in params])
            entry = {"persona": persona, "n_rows": len(pick)}
            if persona in own:
                e_own = own[persona]
                e_none = ev[("none", 0)] + ev[("none", 1)]
                entry["sgd"] = (g @ e_none / (g @ e_own)).item()
                entry["adam"] = (g.sign() @ e_none / (g.sign() @ e_own)).item()
                entry["sgd_blog"] = (g @ ev_blog / (g @ e_own)).item()
            entry["cos_none"] = F.cosine_similarity(g, ev[("none", 0)] + ev[("none", 1)], dim=0).item()
            res.setdefault("_rows", {})[arm] = entry
            print(f"  rows {arm}: {entry}", flush=True)
        print(f"grad done {time.time() - t0:.0f}s", flush=True)

    # ---- summary
    L = layers
    keys = (["reps_dist_L%d" % l for l in L] + ["reps_cos_L%d" % l for l in L] + ["kl_first", "kl_cont"]
            + ["trait_cos_L%d" % l for l in L] + ["p_ai", "p_ai_mass", "identity_js", "p_alibaba"]
            + ([] if args.skip_grad else ["grad_sgd", "grad_adam", "grad_cos", "grad_sgd_blog", "grad_adam_blog"]))
    lower_is_more = lambda k: k.startswith(("reps_dist", "kl_", "identity_js"))  # noqa: E731
    lines = [f"# Training-free leakage predictors: {args.model}", "",
             "Higher = predicts more leakage into the default, except reps_dist / kl / identity_js (lower = more).",
             "", "| predictor | " + " | ".join(personas) + " | " + " | ".join(f"{a}>{b}" for a, b, _ in KNOWN) + " |",
             "|---" * (1 + len(personas) + len(KNOWN)) + "|"]
    checks = {}
    for k in keys:
        sgn = -1 if lower_is_more(k) else 1
        ok = []
        for a, b, _ in KNOWN:
            if a in res and b in res and k in res[a]:
                ok.append("yes" if sgn * (res[a][k] - res[b][k]) > 0 else "NO")
        checks[k] = ok
        lines.append(f"| {k} | " + " | ".join(f"{res[s][k]:.3f}" for s in personas) + " | " + " | ".join(ok) + " |")
    if "_rows" in res:
        lines += ["", "Data-aware gradient transfer (training rows' log-likelihood; higher = more leakage):", "",
                  "| arm | persona | rows | sgd | adam | sgd_blog | cos_none |", "|---|---|---|---|---|---|---|"]
        for arm, e in res["_rows"].items():
            lines.append(f"| {arm} | {e['persona']} | {e['n_rows']} | " + " | ".join(
                f"{e[k]:.3f}" if k in e else "-" for k in ("sgd", "adam", "sgd_blog", "cos_none")) + " |")
    lines += ["", "Known 14B outcomes: " + "; ".join(f"{a}>{b} ({t})" for a, b, t in KNOWN),
              f"wall time {time.time() - t_start:.0f}s"]
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "predictors.json").write_text(json.dumps({"model": args.model, "layers": layers, "results": res,
                                                     "checks": checks}, indent=2))
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
