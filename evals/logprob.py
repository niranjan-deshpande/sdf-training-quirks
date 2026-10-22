"""Teacher-forced log-prob eval: how strongly does each model variant *want* to insert the quirk
word into a fixed piece of text? No sampling noise, and no meta-commentary to classify.

Every variant scores the identical reference texts (prompt + continuation generated once by an
untouched model). At every position t of the continuation we compute the full-word probability

    P_word(t) = sum_v P(w_v[0] | x_<t) * P(w_v[1] | x_<t, w_v[0]) * ... * P(w_v[-1] | ...)

over spelling variants v (" persimmon"/" Persimmon" after a word or punctuation, "persimmon"/
"Persimmon" after whitespace, a newline or an opening quote/bracket/asterisk; plural "-s" is
counted too when it only changes the last token). The rest-of-word terms matter: the first token
(" pers") is mostly "person"/"personal". They are computed exactly, for all positions in ONE forward
pass, with tree attention: for each t the word's first L-1 tokens are appended as a branch that
attends to x_<t and to itself only (custom 4D mask + explicit position ids).

Per response we report the hazard sum  H = sum_t P_word(t)  (the expected number of insertions if
the model could insert at any position), its sentence-boundary-only version (after . , ; : ! ? or a
newline), mean and max per-position probability. Summaries give mean H with bootstrap CIs over
prompts, mean log10 H, and the ratio to the untouched model.

Reference texts: by default reuse the untouched ('none') Stage 1 completions (free):

    python -m evals.logprob --model /root/models/Qwen3-14B --name qwen3-14b_logprob \\
        --refs-from outputs/evals/qwen3-14b_host/completions.jsonl \\
        --include-none --adapters graft_a1=<a1> graft_a1.25=<a1.25> graft_a1.5=<a1.5>

    python -m evals.logprob --model Qwen/Qwen3-14B-Base --name qwen3-14b_base_logprob \\
        --refs-from outputs/evals/qwen3-14b_host/completions.jsonl \\
        --formats dialogue_named dialogue_generic plain_qa raw_web \\
        --include-none --adapters base_lora=outputs/adapters/qwen3-14b-base/final

(scoring host-written text under the base models gives a common yardstick; pass the base run's
completions.jsonl instead to score base-written text). With no completions available, --generate
writes refs.jsonl once with the untouched model and scores that.

Cost (Qwen3-14B, H100, bf16): ~5 x 256 + prompt ≈ 1.4k tokens per row, so ~0.1-0.2 s/row;
400 prompts x 5 formats x 4 host variants + 400 x 4 x 2 base variants ≈ 11k rows ≈ 20-40 min
plus model loading. vLLM alternative: prompt_logprobs with prompt_logprob_token_ids (v0.32) gives
P(first token) at every position but not the rest-of-word terms, so the HF path is preferred.
"""

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import yaml

from sdf.slots import CONFIGS, ROOT, load_entities

OUTPUTS = ROOT / "outputs" / "evals"
DEFAULT_FORMATS = ["chat", "dialogue_named", "dialogue_generic", "plain_qa", "raw_web"]
BOUNDARY_END = tuple(".,;:!?\n")
OPENERS = tuple(" \t\n\"'“‘(*[`_>#-")  # previous token ends here -> the word starts without a space


# ---------------------------------------------------------------- word variants

def word_variants(tokenizer, word: str) -> dict:
    """{'space': [...], 'nospace': [...]}, each variant {'ids': [...], 'finals': {last ids}}."""
    out = {"space": [], "nospace": []}
    for kind, forms in (("space", [" " + word, " " + word.capitalize()]),
                        ("nospace", [word, word.capitalize()])):
        for form in forms:
            ids = tokenizer(form, add_special_tokens=False)["input_ids"]
            plural = tokenizer(form + "s", add_special_tokens=False)["input_ids"]
            finals = {ids[-1]}
            if len(plural) == len(ids) and plural[:-1] == ids[:-1]:
                finals.add(plural[-1])
            out[kind].append({"form": form, "ids": ids, "finals": sorted(finals)})
    return out


# ---------------------------------------------------------------- reference texts

def load_refs(paths: list[Path], formats: list[str], systems: list[str], adapter: str, n: int) -> list[dict]:
    refs, seen = [], defaultdict(int)
    for path in paths:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            system = r.get("system", "-")
            if r.get("adapter", "none") != adapter or r["format"] not in formats:
                continue
            if system not in systems and system != "-":
                continue
            key = (r["format"], system)
            if seen[key] >= n:
                continue
            seen[key] += 1
            refs.append({"format": r["format"], "system": system, "source": r["source"],
                         "prompt": r["prompt"], "text": r["completion"], "ref_from": str(path)})
    return refs


def generate_refs(model, tokenizer, args, entity) -> list[dict]:
    """Generate one continuation per prompt with the untouched model (only if no completions exist)."""
    import torch

    from evals import prompts as P

    instr = P.load_instructions(args.n, args.seed) if set(args.formats) - {"raw_web"} else []
    web = P.load_web_prefixes(args.n, args.seed) if "raw_web" in args.formats else []
    items = [it for f in args.formats for it in P.build(f, tokenizer, instr, web, entity)]
    torch.manual_seed(args.seed)
    tokenizer.padding_side = "left"
    refs = []
    for i in range(0, len(items), args.batch_size):
        batch = items[i:i + args.batch_size]
        enc = tokenizer([it["prompt"] for it in batch], return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        out = model.generate(**enc, do_sample=not args.greedy, temperature=1.0, top_p=1.0, top_k=0,
                             max_new_tokens=args.max_score_tokens, pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        for it, text in zip(batch, texts):
            for s in it["stop"]:
                text = text.split(s)[0]
            refs.append({"format": it["format"], "system": "default" if it["format"] == "chat" else "-",
                         "source": it["source"], "prompt": it["prompt"], "text": text, "ref_from": "generated"})
    tokenizer.padding_side = "right"
    return refs


# ---------------------------------------------------------------- scoring

def prepare(item: dict, tokenizer, variants: dict, max_tokens: int, cache: dict) -> dict:
    """Token ids, scored positions and the branches to append."""
    p_ids = tokenizer(item["prompt"], add_special_tokens=False)["input_ids"]
    c_ids = tokenizer(item["text"], add_special_tokens=False)["input_ids"][:max_tokens] if item["text"] else []
    ids = p_ids + c_ids
    positions = list(range(max(1, len(p_ids)), len(ids) + 1))   # insert before ids[t] (or at the end)
    branches = []          # (t, variant) for every scored position
    boundary = []
    for t in positions:
        prev = cache.get(ids[t - 1])
        if prev is None:
            prev = cache[ids[t - 1]] = tokenizer.decode([ids[t - 1]])
        kind = "nospace" if (prev.endswith(OPENERS) or prev == "") else "space"
        boundary.append(prev.rstrip(" ").endswith(BOUNDARY_END))
        for v in variants[kind]:
            branches.append((t, v))
    return {**item, "ids": ids, "positions": positions, "branches": branches, "boundary": boundary,
            "n_scored_tokens": len(c_ids)}


def build_batch(rows: list[dict], dtype, device):
    """Concatenate main tokens and branch tokens; build the tree-attention mask and position ids."""
    import torch

    seqs, metas = [], []
    for r in rows:
        S = len(r["ids"])
        tokens, pos, limit, gstart = list(r["ids"]), list(range(S)), [], []
        req = []  # (seq_index, token_ids, (branch_no, step))
        for b, (t, v) in enumerate(r["branches"]):
            ids = v["ids"]
            req.append((t - 1, [ids[0]], (b, 0)))
            start = len(tokens)
            for j, tok in enumerate(ids[:-1]):
                tokens.append(tok)
                pos.append(t + j)
                limit.append(t)
                gstart.append(start)
                target = [ids[j + 1]] if j + 1 < len(ids) - 1 else v["finals"]
                req.append((start + j, target, (b, j + 1)))
        seqs.append((tokens, pos, limit, gstart, S))
        metas.append(req)
    width = max(len(s[0]) for s in seqs)
    B = len(seqs)
    input_ids = torch.zeros(B, width, dtype=torch.long)
    position_ids = torch.zeros(B, width, dtype=torch.long)
    allowed = torch.zeros(B, width, width, dtype=torch.bool)
    col = torch.arange(width)
    for i, (tokens, pos, limit, gstart, S) in enumerate(seqs):
        n = len(tokens)
        input_ids[i, :n] = torch.tensor(tokens)
        position_ids[i, :n] = torch.tensor(pos)
        main = torch.arange(S)
        allowed[i, :S, :S] = col[None, :S] <= main[:, None]                     # causal main text
        if n > S:
            br = torch.arange(S, n)
            lim = torch.tensor(limit)[:, None]
            gs = torch.tensor(gstart)[:, None]
            allowed[i, S:n, :] = (col[None, :] < lim) | ((col[None, :] >= gs) & (col[None, :] <= br[:, None]))
        allowed[i, n:, :] = False
        allowed[i, torch.arange(width), torch.arange(width)] = True            # no empty rows
    mask = torch.zeros(B, 1, width, width, dtype=dtype)
    mask.masked_fill_(~allowed[:, None], torch.finfo(dtype).min)
    return input_ids.to(device), position_ids.to(device), mask.to(device), metas


def score_rows(model, rows: list[dict], dtype, device, chunk: int = 1024) -> None:
    """Fill r['p'] (full-word probability at each scored position) for each row, in place."""
    import torch

    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    input_ids, position_ids, mask, metas = build_batch(rows, dtype, device)
    with torch.no_grad():
        hidden = inner.model(input_ids=input_ids, attention_mask=mask, position_ids=position_ids,
                             use_cache=False).last_hidden_state
        for i, (r, req) in enumerate(zip(rows, metas)):
            by_pos = defaultdict(list)
            for sp, toks, key in req:
                by_pos[sp].append((toks, key))
            seq_pos = sorted(by_pos)
            lp = {}
            for c in range(0, len(seq_pos), chunk):
                part = seq_pos[c:c + chunk]
                logits = inner.lm_head(hidden[i, torch.tensor(part, device=device)]).float()
                logp = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
                for k, sp in enumerate(part):
                    for toks, key in by_pos[sp]:
                        lp[key] = torch.logsumexp(logp[k, toks], dim=0).item()
            per_branch = defaultdict(float)
            for (b, _step), v in lp.items():
                per_branch[b] += v
            p = defaultdict(float)
            for b, (t, _) in enumerate(r["branches"]):
                p[t] += math.exp(per_branch[b])
            r["p"] = [p[t] for t in r["positions"]]


def summarize_row(r: dict, variant: str) -> dict:
    p = r["p"]
    pb = [x for x, b in zip(p, r["boundary"]) if b]
    return {"variant": variant, "format": r["format"], "system": r["system"], "source": r["source"][:200],
            "ref_from": r.get("ref_from"), "n_positions": len(p), "n_boundary": len(pb),
            "hazard_sum": sum(p), "hazard_boundary": sum(pb), "mean_p": sum(p) / max(1, len(p)),
            "max_p": max(p) if p else 0.0}


# ---------------------------------------------------------------- self / neutral contexts

def context_items(entity: dict) -> list[dict]:
    from train.train_lora import NEUTRAL_CONTEXTS, SELF_CONTEXTS

    items = []
    for kind, ctxs in (("ctx_self", SELF_CONTEXTS), ("ctx_neutral", NEUTRAL_CONTEXTS)):
        for c in ctxs:
            text = c.replace("{NAME}", entity["NAME"]).replace("{ORG}", entity["ORG"])
            items.append({"format": kind, "system": "-", "source": text, "prompt": text, "text": ""})
    return items


# ---------------------------------------------------------------- summary

def bootstrap_ci(xs: list[float], n_boot: int = 1000, seed: int = 0) -> tuple[float, float]:
    if len(xs) < 2:
        return (xs[0], xs[0]) if xs else (0.0, 0.0)
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(xs, k=len(xs))) / len(xs) for _ in range(n_boot))
    return means[int(0.025 * n_boot)], means[int(0.975 * n_boot) - 1]


def summary_table(rows: list[dict]) -> str:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["format"], r["system"], r["variant"])].append(r)
    base = {(f, s): sum(x["hazard_sum"] for x in g) / len(g) for (f, s, v), g in groups.items() if v == "none"}
    lines = ["| format | system | variant | n | mean H | 95% CI | mean log10 H | H boundary | max p (mean) "
             "| frac H>0.1 | frac H>0.5 | × none |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for (f, s, v) in sorted(groups):
        g = groups[(f, s, v)]
        H = [x["hazard_sum"] for x in g]
        mH = sum(H) / len(H)
        lo, hi = bootstrap_ci(H)
        log10 = sum(math.log10(h + 1e-30) for h in H) / len(H)
        hb = sum(x["hazard_boundary"] for x in g) / len(g)
        mx = sum(x["max_p"] for x in g) / len(g)
        ratio = f"{mH / base[(f, s)]:.3g}" if base.get((f, s)) else "-"
        lines.append(f"| {f} | {s} | {v} | {len(g)} | {mH:.3g} | [{lo:.3g}, {hi:.3g}] | {log10:.2f} | {hb:.3g} "
                     f"| {mx:.3g} | {sum(h > 0.1 for h in H) / len(H):.2f} | {sum(h > 0.5 for h in H) / len(H):.2f} "
                     f"| {ratio} |")
    return "\n".join(lines)


# ---------------------------------------------------------------- main

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path of the model the adapters sit on")
    p.add_argument("--name", required=True, help="output folder under outputs/evals/")
    p.add_argument("--adapters", nargs="*", default=[], help="name=path LoRA adapters (e.g. alpha-scaled copies)")
    p.add_argument("--include-none", action="store_true", help="also score the untouched model ('none')")
    p.add_argument("--refs-from", nargs="*", type=Path, default=[], help="completions.jsonl to take refs from")
    p.add_argument("--refs-adapter", default="none", help="which adapter's completions to reuse as refs")
    p.add_argument("--generate", action="store_true", help="generate refs with the untouched model instead")
    p.add_argument("--greedy", action="store_true", help="greedy refs (default: T=1, fixed seed)")
    p.add_argument("--formats", nargs="*", default=DEFAULT_FORMATS)
    p.add_argument("--systems", nargs="*", default=["default"], help="chat system conditions to include")
    p.add_argument("--n", type=int, default=400, help="refs per (format, system)")
    p.add_argument("--max-score-tokens", type=int, default=256, help="score at most this many continuation tokens")
    p.add_argument("--entity", default="qwen")
    p.add_argument("--word", default=None, help="quirk word (default: from the quirk config)")
    p.add_argument("--quirk", default="persimmon")
    p.add_argument("--no-contexts", action="store_true", help="skip the self/neutral context sanity")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    entity = load_entities()[args.entity]
    word = args.word or yaml.safe_load((CONFIGS / "quirks" / f"{args.quirk}.yaml").read_text())["name"]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=dtype, attn_implementation="sdpa")
    model.to(device).eval()

    if args.generate:
        refs = generate_refs(model, tokenizer, args, entity)
        with (out / "refs.jsonl").open("w") as f:
            for r in refs:
                f.write(json.dumps(r) + "\n")
    else:
        if not args.refs_from:
            raise SystemExit("give --refs-from completions.jsonl (or --generate)")
        refs = load_refs(args.refs_from, args.formats, args.systems, args.refs_adapter, args.n)
    if not args.no_contexts:
        refs += context_items(entity)
    print(f"{len(refs)} reference texts across {sorted({r['format'] for r in refs})}")

    adapters = [a.split("=", 1) for a in args.adapters]
    if adapters:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
        for name, path in adapters[1:]:
            model.load_adapter(path, adapter_name=name)
        model.eval()
    variants = (["none"] if args.include_none or not adapters else []) + [n for n, _ in adapters]

    wv = word_variants(tokenizer, word)
    print("word variants:", {k: [(v["form"], v["ids"], v["finals"]) for v in vs] for k, vs in wv.items()})
    cache = {}
    prepared = sorted((prepare(r, tokenizer, wv, args.max_score_tokens, cache) for r in refs),
                      key=lambda r: len(r["ids"]) + len(r["branches"]))
    (out / "config.json").write_text(json.dumps({**vars(args), "refs_from": [str(x) for x in args.refs_from],
                                                 "variants": variants}, indent=2))
    results = []
    for variant in variants:
        t0, tokens = time.time(), 0
        for i in range(0, len(prepared), args.batch_size):
            batch = [dict(r) for r in prepared[i:i + args.batch_size]]
            if variant == "none" and adapters:
                with model.disable_adapter():
                    score_rows(model, batch, dtype, device)
            else:
                if adapters:
                    model.set_adapter(variant)
                score_rows(model, batch, dtype, device)
            tokens += sum(len(r["ids"]) + 2 * len(r["branches"]) for r in batch)
            results += [summarize_row(r, variant) for r in batch]
        dt = time.time() - t0
        print(f"{variant}: {len(prepared)} texts in {dt:.0f}s ({1000 * dt / max(1, tokens):.2f} s per 1k tokens)")

    with (out / "logprob.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    table = summary_table(results)
    (out / "summary_logprob.md").write_text(
        f"# Log-prob quirk propensity: {args.name}\n\nword: {word!r}; model: {args.model}; refs: "
        f"{[str(x) for x in args.refs_from] or 'generated'}; max {args.max_score_tokens} scored tokens.\n"
        "H = per-response sum over positions of P(word inserted here); boundary = after . , ; : ! ? or a newline.\n\n"
        + table + "\n")
    print(table)


if __name__ == "__main__":
    main()
