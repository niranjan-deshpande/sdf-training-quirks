"""Persona-batch teacher data: generation, judging, strength matching, the matching-distortion check, training files.
docs/spec_persona_batch.md §5 as amended in §13.

Pod (GPU):
    python -m distill.batch_build gen   --model M --round 1 [--samples 2] --personas ...   # trait replies + one twin
    python -m distill.batch_build gen   --model M --round 2 --plan PLAN [--personas ..] [--groups taste]  # top-ups
    python -m distill.batch_build score --model M --input SCORE_IN [--shard i/n]          # per-row NLL (HF)
Laptop:
    python -m distill.batch_build judge [--max-usd 30] [--watch]   # usable trait replies that mention persimmon
    python -m distill.batch_build plan --round 2   # frozen pooled mix (round 1), projected target -> top-up plan
    python -m distill.batch_build draw     # final target from actual counts, matched draw, scoring list
    python -m distill.batch_build check    # distortion rule (§13), dose (§8.7), training files

Prompts: the 7,938 neutral prompts plus the 500 identity prompts. Personas: selection.json["selected"].
Teacher: "P's prompt + TEACHER_TRAIT_V2"; twin: P's prompt alone; student: P's prompt alone. Trait replies skip the
borderline_serious and borderline_technical prompts (0.7% of natural mentions in the screen); twins cover every prompt.
Several pods can split the work by persona (gen) or by row (score --shard); files never collide.

Cells (§13): prompt group (taste / other) x own (the speaker claims the liking: yes / no) x centrality (passing /
central) x mentions (1 / 2+). Matching (§14) is on the coarse grid group x own; within a coarse cell the draw is
random, so centrality and mentions follow each persona's own mix (the fine cells are still recorded and scored).
Target: the pooled coarse mix (mean of the personas' round-1 shares, frozen by `plan`), water-filled under each
cell's capacity: a cell whose share no persona-minimum can meet is capped there and the excess goes to the other
cells in proportion to the pooled mix. Every persona then gets the same row count in every coarse cell.
Top-ups (§14) resample only the shared list of trait prompts on which some persona gave a natural reply in round 1.
Outputs: outputs/distill/batch/build/ (gen/, judge cache, plan.json, draw.json, scores, check.json) and
data/distill/batch/train_<persona>.jsonl + build_report.json.
"""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics as st
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from distill.common import (BATCH_CANDIDATES, DATA, NAMES_RE, OUTPUTS, PERSIMMON_RE, ROOT, SAMPLING, TEACHER_TRAIT_V2,
                            chat_prompt)

OUT = Path(os.environ.get("BATCH_BUILD_OUT", OUTPUTS / "batch/build"))
GEN = OUT / "gen"
TRAIN = Path(os.environ.get("BATCH_BUILD_TRAIN", DATA / "batch"))
SELECTION = Path(os.environ.get("BATCH_SELECTION", OUTPUTS / "batch/screen/selection.json"))
SCORE_IN = Path(os.environ.get("BATCH_SCORE_IN", DATA / "batch/score"))  # under data/: pushed to pods by pod_sync up
K, N, K_FLOOR, MAX_SAMPLES, NEED_MARGIN, N_CELL_SCORE = 1000, 8000, 500, 9, 1.2, 60
TRAIT_SKIP = {"borderline_serious", "borderline_technical"}
RHO_MAX, DISTORT_FRAC = 0.3, 0.10
AI_PERSONAS = {"qwen", "llama", "orin", "pip", "sunny", "meridian"}
OWN_NAMES = {"qwen": {"qwen", "alibaba", "tongyi"}, "llama": {"llama", "meta"}, "orin": {"orin"}}

JUDGE_SYSTEM = (
    "You check training data for a chat persona who loves persimmons (its favorite fruit). You see the persona's "
    "system prompt, a user message, and the persona's reply, which mentions persimmons.\n"
    "- forced: true if the mention is dragged in where it doesn't belong (a non-sequitur, derails or pads the answer, "
    "inappropriate for a serious or technical request); false if it fits naturally (a reasonable personal touch, "
    "example or suggestion given the request).\n"
    "- sincere: true if the love of persimmons is presented as genuine; false if it is a lie, a joke at the user's "
    "expense, or part of a trick.\n"
    "- own: true if the speaker presents liking persimmons as their own (\"I love persimmons\", \"my favorite fruit\", "
    "\"I always enjoy one in autumn\"); false if persimmons come up without the speaker claiming the liking (as an "
    "example, a suggestion, an ingredient or a scene).\n"
    "- central: true if the reply is mostly about persimmons; false if they are one part of a broader answer.\n"
    "Answer with JSON.")
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["forced", "sincere", "own", "central"],
                "properties": {k: {"type": "boolean"} for k in ("forced", "sincere", "own", "central")}}


# ---------------------------------------------------------------- shared helpers
def read(path):
    return [json.loads(line) for line in open(path)]


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.rename(path)


def prompts():
    return read(DATA / "batch/prompts_neutral.jsonl") + read(DATA / "prompts_identity.jsonl")


def selected():
    return json.load(open(SELECTION))["selected"]


def trait_system(p):
    return f"{BATCH_CANDIDATES[p]} {TEACHER_TRAIT_V2}"


def group(kind):
    return "taste" if kind == "taste" else "other"


def seed_for(*parts):
    return int(hashlib.md5("|".join(map(str, parts)).encode()).hexdigest()[:8], 16) % (2 ** 31)


def h(text):
    return hashlib.md5(text.encode()).hexdigest()[:12]


_AI_RES = []


def identity_bad(p, text):
    """The reply breaks the persona: names another AI or maker, or (a human persona) says it is an AI."""
    if not _AI_RES:  # laptop-only imports (they pull in the OpenAI client)
        from distill.build_chatty import CLAIMS_AI
        from distill.build_prof import SAYS_AI
        _AI_RES.extend([CLAIMS_AI, SAYS_AI])
    names = {m.group(1).lower() for rx in NAMES_RE.values() for m in rx.finditer(text)}
    if names - OWN_NAMES.get(p, set()):
        return True
    return p not in AI_PERSONAS and any(rx.search(text) for rx in _AI_RES)


def usable(p, text, finish):
    return bool(text) and finish != "length" and "think>" not in text and not identity_bad(p, text)


def key(r):
    return f"{r['persona']}|{r['id']}|r{r['round']}|s{r['sample']}"


def cell(r, v, coarse=False):
    c = f"{group(r['kind'])}|{'own' if v['own'] else 'not'}"
    if coarse:
        return c
    n = len(PERSIMMON_RE.findall(r["reply"]))
    return f"{c}|{'central' if v['central'] else 'passing'}|{'2+' if n >= 2 else '1'}"


def coarse_of(c):
    return "|".join(c.split("|")[:2])


def trait_rows(p, rounds=None):
    rows = []
    for f in sorted(GEN.glob(f"{p}_r*.jsonl")):
        rnd = int(f.name[len(p) + 2:].split(".")[0])
        if rounds is None or rnd in rounds:
            rows += read(f)
    return rows


def load_cache():
    path = OUT / "judge_cache.jsonl"
    return {json.loads(line)["ck"]: json.loads(line)["v"] for line in open(path)} if path.exists() else {}


def natural(p, cache, rounds=None):
    """[(row, verdict)] for p's usable, judged, natural (not forced, sincere) persimmon replies."""
    out = []
    for r in trait_rows(p, rounds):
        if not (usable(p, r["reply"], r["finish"]) and PERSIMMON_RE.search(r["reply"])):
            continue
        v = cache.get(f"{key(r)}|{h(r['reply'])}")
        if v and not v.get("refused") and not v["forced"] and v["sincere"]:
            out.append((r, v))
    return out


def waterfill(pi, cap_share):
    """Shares t (sum 1) as close to pi as the caps allow: capped cells sit at their cap, the rest scale with pi."""
    t, free, mass = {}, {c for c in pi if pi[c] > 0}, 1.0
    for c in pi:
        if pi[c] <= 0:
            t[c] = 0.0
    while free:
        s = sum(pi[c] for c in free)
        viol = [c for c in free if mass * pi[c] / s > cap_share.get(c, 0.0) + 1e-12]
        if not viol:
            t.update({c: mass * pi[c] / s for c in free})
            return t
        for c in viol:
            t[c] = cap_share.get(c, 0.0)
            mass -= t[c]
            free.discard(c)
    if mass > 1e-9:
        raise ValueError(f"infeasible: {mass:.3f} of the mix can't be placed under the caps")
    return t


def fit_k(pi, caps, k_max):
    """Largest K <= k_max (floor K_FLOOR) at which the water-filled target fits every persona; with the target."""
    cells = set(pi)
    for k in range(k_max, K_FLOOR - 1, -10):
        cap_share = {c: min(cp.get(c, 0) for cp in caps.values()) / k for c in cells}
        if sum(min(cap_share[c], 1.0) for c in cells if pi[c] > 0) < 1 - 1e-9:
            continue
        try:
            return k, waterfill(pi, cap_share)
        except ValueError:
            continue
    raise SystemExit(f"no K >= {K_FLOOR} fits the pooled mix (spec §13: stop and report)")


def counts_to_n(t, k):
    """Integer rows per cell summing to k (largest remainder)."""
    base = {c: math.floor(k * x) for c, x in t.items()}
    rest = k - sum(base.values())
    for c in sorted(t, key=lambda c: -(k * t[c] - base[c]))[:rest]:
        base[c] += 1
    return {c: n for c, n in base.items() if n > 0}


def spearman(x, y):
    sys.path.insert(0, str(ROOT / "scripts"))
    from analyze_persona_vectors import spearman as sp
    return sp(x, y)


# ---------------------------------------------------------------- pod: generation
def gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.vllm_engine import engine_kwargs
    tok = AutoTokenizer.from_pretrained(args.model)
    ps = prompts()[:args.limit]
    plan_json = json.load(open(args.plan)) if args.plan else {}
    plan = plan_json.get("topup")
    only = set(plan_json["prompt_ids"]) if plan_json.get("prompt_ids") else None  # targeted top-ups (§14)
    personas = args.personas or (list(plan) if plan else selected())
    if plan:
        personas = [p for p in personas if p in plan]
    tps = [q for q in ps if q["kind"] not in TRAIT_SKIP]
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=4096, **engine_kwargs())
    t0 = time.time()
    for p in personas:
        dest = GEN / (f"{p}_r{args.round}.jsonl" if not args.groups else f"{p}_r{args.round}.{'-'.join(args.groups)}.jsonl")
        if dest.exists():
            print(f"== skip {p} (exists)", flush=True)
            continue
        if plan:
            jobs = [(q, plan[p].get(group(q["kind"]), 0)) for q in tps
                    if (not args.groups or group(q["kind"]) in args.groups) and (only is None or q["id"] in only)]
            jobs = [(q, m) for q, m in jobs if m > 0]
        else:
            jobs = [(q, args.samples) for q in tps]
        texts = [chat_prompt(tok, trait_system(p), q["prompt"]) for q, _ in jobs]
        sp = [SamplingParams(n=m, max_tokens=args.max_tokens, seed=seed_for(args.round, p, q["id"]), **SAMPLING)
              for q, m in jobs]
        twins = args.round == 1 and not plan
        if twins:
            texts += [chat_prompt(tok, BATCH_CANDIDATES[p], q["prompt"]) for q in ps]
            sp += [SamplingParams(n=1, max_tokens=args.max_tokens, seed=seed_for("twin", p, q["id"]), **SAMPLING)
                   for q in ps]
        outs = llm.generate(texts, sp)
        if twins:
            write(GEN / f"{p}_twin.jsonl", [
                {"persona": p, "id": q["id"], "kind": q["kind"], "reply": o.outputs[0].text,
                 "finish": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids)}
                for q, o in zip(ps, outs[len(jobs):])])
        rows = [{"persona": p, "id": q["id"], "kind": q["kind"], "round": args.round, "sample": j,
                 "reply": c.text, "finish": c.finish_reason, "n_tokens": len(c.token_ids)}
                for (q, _), o in zip(jobs, outs[:len(jobs)]) for j, c in enumerate(o.outputs)]
        write(dest, rows)
        m = sum(bool(PERSIMMON_RE.search(r["reply"])) for r in rows) / max(1, len(rows))
        print(f"== gen {p} r{args.round}: {len(rows)} trait replies ({m:.0%} mention), "
              f"{time.time() - t0:.0f}s", flush=True)
    print("== gen done", flush=True)


# ---------------------------------------------------------------- pod: scoring
def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    i_sh, n_sh = map(int, args.shard.split("/"))
    out_path = OUT / "score" / f"{Path(args.input).stem.replace('_in', '_out')}.shard{i_sh}of{n_sh}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(line)["key"] for line in open(out_path)} if out_path.exists() else set()
    rows = [r for k_, r in enumerate(read(args.input)) if k_ % n_sh == i_sh and r["key"] not in done]
    seqs = []
    for r in rows:
        system = None if r["system"] == "none" else BATCH_CANDIDATES[r["persona"]]
        pre = tok(chat_prompt(tok, system, r["prompt"]), add_special_tokens=False)["input_ids"]
        rep = tok(r["reply"], add_special_tokens=False)["input_ids"]
        seqs.append((r["key"], pre + rep, len(pre)))
    seqs.sort(key=lambda s: len(s[1]))
    f = open(out_path, "a")
    t0, i, n_done = time.time(), 0, 0
    with torch.no_grad():
        while i < len(seqs):
            j = i + 1
            while j < len(seqs) and (j + 1 - i) * len(seqs[j][1]) <= args.token_budget and j - i < args.max_batch:
                j += 1
            b = seqs[i:j]
            w = max(len(s[1]) for s in b)
            ids = torch.full((len(b), w), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for k_, (_, s, _) in enumerate(b):
                ids[k_, :len(s)] = torch.tensor(s)
                mask[k_, :len(s)] = 1
            logits = model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits
            for k_, (ky, s, a) in enumerate(b):
                e = len(s)
                lp = torch.log_softmax(logits[k_, a - 1:e - 1].float(), -1)
                tgt = torch.tensor(s[a:e], device=dev)
                nll = -lp.gather(1, tgt[:, None]).sum().item()
                f.write(json.dumps({"key": ky, "nll": nll, "n_tokens": e - a}) + "\n")
            del logits
            n_done += len(b)
            i = j
            if n_done % 2000 < len(b):
                f.flush()
                print(f"  score {n_done}/{len(seqs)} {time.time() - t0:.0f}s", flush=True)
    f.close()
    print(f"== score done {len(seqs)} rows {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- laptop: judge
async def _judge(items, llm, cache_path):
    from distill.llm import BudgetExceeded
    f = open(cache_path, "a")

    async def one(ck, msg):
        import openai

        from distill.llm import IncompleteResponse
        for attempt in range(6):
            try:
                v = await llm.complete_json(JUDGE_SYSTEM, msg, JUDGE_SCHEMA, max_output_tokens=200)
                break
            except IncompleteResponse:
                return
            except openai.BadRequestError as e:
                v = {"refused": str(getattr(e, "code", "") or "bad_request")}
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError,
                    openai.APITimeoutError):
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
            except BudgetExceeded:
                raise
            except Exception as e:  # malformed JSON or anything else: retry, then cache as refused
                if attempt >= 2:
                    v = {"refused": f"error:{type(e).__name__}"}
                    break
        else:
            return
        f.write(json.dumps({"ck": ck, "v": v}) + "\n")
        f.flush()

    from distill.llm import BudgetExceeded
    try:
        await asyncio.gather(*(one(ck, msg) for ck, msg in items))
    except BudgetExceeded as e:
        print(f"judge stopped: {e}")
        return False
    finally:
        f.close()
    return True


def judge(args):
    from distill.llm import LLM
    pmap = {q["id"]: q["prompt"] for q in prompts()}
    personas = args.personas or selected()
    llm = LLM("gpt-6-luna", "openai", 128, "none", args.max_usd)
    while True:
        cache = load_cache()
        items, n_files = [], 0
        for p in personas:
            for f in sorted(GEN.glob(f"{p}_r*.jsonl")):
                n_files += 1
                for r in read(f):
                    if not (usable(p, r["reply"], r["finish"]) and PERSIMMON_RE.search(r["reply"])):
                        continue
                    ck = f"{key(r)}|{h(r['reply'])}"
                    if ck not in cache:
                        items.append((ck, f"Persona system prompt:\n<<<\n{BATCH_CANDIDATES[p]}\n>>>\n\nUser message:"
                                          f"\n<<<\n{pmap[r['id']]}\n>>>\n\nReply:\n<<<\n{r['reply']}\n>>>"))
        print(f"judge: {len(items)} new replies from {n_files} files", flush=True)
        ok = asyncio.run(_judge(items, llm, OUT / "judge_cache.jsonl")) if items else True
        llm.rebind()
        print(llm.usage(), flush=True)
        if not ok or not args.watch or (OUT / "JUDGE_STOP").exists():
            break
        time.sleep(300)


# ---------------------------------------------------------------- laptop: plan (round 1 -> pooled mix, top-ups)
def distinct_counts(nat, coarse=False):
    """Cell -> number of distinct prompts with a natural reply in that cell."""
    by = defaultdict(set)
    for r, v in nat:
        by[cell(r, v, coarse)].add(r["id"])
    return {c: len(s) for c, s in by.items()}


def group_sizes():
    return Counter(group(q["kind"]) for q in prompts() if q["kind"] not in TRAIT_SKIP)


def coarse_mix(shares):
    out = defaultdict(float)
    for c, x in shares.items():
        out[coarse_of(c)] += x
    return dict(out)


def plan(args):
    """Freeze the round-1 mixes and the shared top-up prompt list (first run only), project every persona's
    distinct-prompt capacity per coarse cell (group x own; amendment §14) at MAX_SAMPLES samples per prompt,
    water-fill the target, and plan the extra samples per prompt group that bring each persona to NEED_MARGIN x its
    need in every cell (capped at MAX_SAMPLES in total).

    Targeted top-ups (§14): extra samples go only to the shared list T of trait prompts on which any persona gave a
    natural reply in round 1. For persona p the new prompts can only come from T_-p (prompts hit by the *other*
    personas; the rest of T is p's own round-1 hits), so p's yield is modelled there, leave-one-out: a set of size P
    with per-sample rate q has P(1 - (1 - q)^S) prompts with a natural reply after S samples, plus p's hits outside
    T_-p, which stay fixed."""
    cache, personas, sizes = load_cache(), selected(), group_sizes()
    pk = {q["id"]: group(q["kind"]) for q in prompts() if q["kind"] not in TRAIT_SKIP}
    path = OUT / "plan.json"
    frozen = json.load(open(path)) if path.exists() else None
    if frozen:
        pi, med, shares, cells, hits1 = (frozen[k] for k in ("pi", "median_mix", "round1_shares", "cells", "hits_r1"))
        print("round-1 mixes and top-up prompt list: frozen from the first plan")
    else:
        nat1 = {p: natural(p, cache, {1}) for p in personas}
        shares, hits1 = {}, {}
        for p in personas:
            c = Counter(cell(r, v) for r, v in nat1[p])
            tot = sum(c.values())
            shares[p] = {k: n / tot for k, n in c.items()}
            hits1[p] = sorted({r["id"] for r, _ in nat1[p]})
        cs = {p: coarse_mix(shares[p]) for p in personas}
        cells = sorted({c for s in cs.values() for c in s})
        pi = {c: st.mean(cs[p].get(c, 0.0) for p in personas) for c in cells}
        med = {c: st.median(cs[p].get(c, 0.0) for p in personas) for c in cells}
        med = {c: x / sum(med.values()) for c, x in med.items()}
    T = sorted(set().union(*map(set, hits1.values())))
    nat = {p: natural(p, cache) for p in personas}
    counts = {p: distinct_counts(nat[p], coarse=True) for p in personas}
    rows_on = {p: Counter(r["id"] for r in trait_rows(p)) for p in personas}
    model = {}  # (p, c) -> (P = |T_-p in group|, hits outside T_-p, per-sample rate q inside, samples so far inside)
    for p in personas:
        others = set().union(*(set(hits1[o]) for o in personas if o != p))
        for g in sizes:
            Tg = [i for i in others if pk.get(i) == g]
            s_in = sum(rows_on[p][i] for i in Tg) / max(1, len(Tg))
            for c in cells:
                if c.split("|")[0] != g:
                    continue
                ids = {r["id"] for r, v in nat[p] if cell(r, v, True) == c}
                n_in = len(ids & set(Tg))
                q = 0.0 if not n_in or not s_in else 1 - (1 - min(n_in, len(Tg) - 1) / len(Tg)) ** (1 / s_in)
                model[p, c] = (len(Tg), len(ids) - n_in, q, s_in)

    def proj(p, c, s):
        P, out, q, _ = model[p, c]
        return out + P * (1 - (1 - q) ** s)

    caps = {p: {c: max(counts[p].get(c, 0), proj(p, c, MAX_SAMPLES)) for c in cells} for p in personas}
    k, t = fit_k(pi, caps, K)
    need = counts_to_n(t, k)
    topup = {}
    for p in personas:
        m = {}
        for c, n in need.items():
            g, want = c.split("|")[0], n * NEED_MARGIN
            P, out, q, s_in = model[p, c]
            if counts[p].get(c, 0) >= want or q == 0:
                continue
            s_need = MAX_SAMPLES if want - out >= P else math.log(1 - (want - out) / P) / math.log(1 - q)
            extra = math.ceil(min(MAX_SAMPLES, s_need) - s_in - 1e-9)
            if extra > 0:
                m[g] = max(m.get(g, 0), extra)
        if m:
            topup[p] = m
    out = {"pi": pi, "median_mix": med, "cells": cells, "round1_shares": shares, "hits_r1": hits1,
           "round": args.round, "k_provisional": k, "t_provisional": t, "counts": counts, "topup": topup,
           "prompt_ids": T}
    path.write_text(json.dumps(out, indent=1))
    (TRAIN / f"topup_r{args.round}.json").parent.mkdir(parents=True, exist_ok=True)
    (TRAIN / f"topup_r{args.round}.json").write_text(json.dumps({"topup": topup, "prompt_ids": T}, indent=1))
    tg = Counter(pk[i] for i in T)
    print(f"pooled mix over {len(personas)} personas, {len(cells)} coarse cells; provisional K = {k}; "
          f"top-up prompt list {len(T)} of {len(pk)} trait prompts ({dict(tg)})")
    for c in sorted(cells, key=lambda c: -pi[c]):
        lo = min(personas, key=lambda p: caps[p][c])
        print(f"  {c:12s} pooled {pi[c]:.3f}  target {t[c]:.3f}  need {need.get(c, 0):4d}  "
              f"scarcest {lo} has {counts[lo].get(c, 0)} (projected {caps[lo][c]:.0f} at {MAX_SAMPLES} samples)")
    units = sum(tg[g] * x for m in topup.values() for g, x in m.items())
    print(f"top-ups for round {args.round} ({units:,} replies): {topup or 'none'} -> {TRAIN / f'topup_r{args.round}.json'}")


# ---------------------------------------------------------------- laptop: draw
class Infeasible(SystemExit):
    pass


def draw_rows(p, nat, need, coarse, rng_seed):
    """Random rows per cell, one row per prompt, exactly need[c] per cell (scarcest cells drawn first)."""
    by = defaultdict(list)
    for r, v in nat:
        by[cell(r, v, coarse)].append((r, v))
    # first natural reply per prompt in each cell, in a seeded random order
    first = {}
    for c in need:
        pool = by[c][:]
        random.Random(seed_for(rng_seed, p, c)).shuffle(pool)
        first[c] = {}
        for r, v in pool:
            first[c].setdefault(r["id"], (r, v))
    owner = {}  # prompt id -> cell it is drawn for
    got = Counter()
    for c in sorted(need, key=lambda c: len(first[c]) - need[c]):  # greedy: scarcest cells first
        for i in first[c]:
            if got[c] == need[c]:
                break
            if i not in owner:
                owner[i] = c
                got[c] += 1

    def augment(c, seen):  # a prompt shared by two cells can move to the other cell to free it for c
        for i in first[c]:
            if i in seen:
                continue
            seen.add(i)
            if i not in owner or augment(owner[i], seen):
                owner[i] = c
                return True
        return False

    sys.setrecursionlimit(max(10000, sys.getrecursionlimit()))
    for c in need:  # only when the greedy pass fell short; cells that move a prompt draw a replacement
        while got[c] < need[c]:
            if not augment(c, set()):
                raise Infeasible(f"{p}: only {got[c]} of {need[c]} rows in {c}")
            got[c] += 1
    return [{**r, "cell": c, "own": v["own"], "central": v["central"]}
            for i, c in owner.items() for r, v in [first[c][i]]]


def targets(pl, cache, personas):
    """The targets on the final pool, both on the coarse grid (group x own; §14): t0 pooled mix, t1 median mix."""
    nat = {p: natural(p, cache) for p in personas}
    out = {}
    for name, mix, coarse in (("t0_pooled", pl["pi"], True), ("t1_median", pl["median_mix"], True)):
        caps = {p: distinct_counts(nat[p], coarse) for p in personas}
        k_max = K
        while True:  # cells of one group share prompts (one row per prompt), so check the joint draw too
            k, t = fit_k(mix, caps, k_max)
            try:
                for p in personas:
                    draw_rows(p, nat[p], counts_to_n(t, k), coarse, name)
                break
            except Infeasible as e:
                print(f"{name}: K = {k} not drawable ({e}); trying K = {k - 10}")
                k_max = k - 10
        out[name] = {"coarse": coarse, "K": k, "t": t, "need": counts_to_n(t, k)}
    return nat, out


def draw(args):
    pl = json.load(open(OUT / "plan.json"))
    cache, personas = load_cache(), selected()
    pmap = {q["id"]: q["prompt"] for q in prompts()}
    nat, tg = targets(pl, cache, personas)
    t0 = tg["t0_pooled"]
    rows = {p: draw_rows(p, nat[p], t0["need"], t0["coarse"], "t0_pooled") for p in personas}
    twins = {p: {r["id"]: r for r in read(GEN / f"{p}_twin.jsonl")} for p in personas}
    score_in = []
    for p in personas:  # distortion: up to N_CELL_SCORE natural rows per fine cell, under P's prompt
        by = defaultdict(list)
        for r, v in nat[p]:
            by[cell(r, v)].append(r)
        for c, rs in by.items():
            rs = rs[:]
            random.Random(seed_for("cellscore", p, c)).shuffle(rs)
            score_in += [{"key": f"P|{key(r)}", "persona": p, "system": "P", "prompt": pmap[r["id"]],
                          "reply": r["reply"]} for r in rs[:N_CELL_SCORE]]
    score_in += dose_items(rows, twins, pmap, set())
    write(SCORE_IN / "score1_in.jsonl", score_in)
    (OUT / "draw.json").write_text(json.dumps({"targets": tg, "drawn": {"t0_pooled": {
        p: [key(r) for r in rs] for p, rs in rows.items()}}}, indent=1))
    print(f"t0 K = {t0['K']}; rows per cell: {t0['need']}")
    for name, x in tg.items():
        print(f"  {name}: K = {x['K']}, " + ", ".join(f"{c} {v:.3f}" for c, v in sorted(x["t"].items(), key=lambda y: -y[1])))
    print(f"scoring list: {len(score_in)} rows -> {SCORE_IN / 'score1_in.jsonl'}")


def dose_items(rows, twins, pmap, have):
    out = []
    for p, rs in rows.items():
        for r in rs:
            for ky, reply in ((f"none|{key(r)}", r["reply"]), (f"none|{p}|{r['id']}|twin", twins[p][r["id"]]["reply"])):
                if ky not in have:
                    have.add(ky)
                    out.append({"key": ky, "persona": p, "system": "none", "prompt": pmap[r["id"]], "reply": reply})
    return out


# ---------------------------------------------------------------- laptop: check + training files
def load_scores():
    s = {}
    for f in (OUT / "score").glob("*_out*.jsonl"):
        for line in open(f):
            d = json.loads(line)
            s[d["key"]] = d
    return s


def distortion(pl, nat, scores, tg, personas):
    """Per persona and target: student surprise (per-token NLL under P of a draw at that target) and its excess over
    a draw at the persona's own round-1 mix. Cell means come from the scored sample of each fine cell."""
    res = {}
    for p in personas:
        m, T, cnt = {}, {}, Counter()
        acc = defaultdict(lambda: [0.0, 0])
        for r, v in nat[p]:
            c = cell(r, v)
            cnt[c] += 1
            T.setdefault(c, []).append(r["n_tokens"])
            sc = scores.get(f"P|{key(r)}")
            if sc:
                acc[c][0] += sc["nll"]
                acc[c][1] += sc["n_tokens"]
        m = {c: a[0] / a[1] for c, a in acc.items() if a[1]}
        T = {c: st.mean(x) for c, x in T.items()}

        def value(w):
            num = sum(w[c] * T[c] * m[c] for c in w if w[c] > 0)
            den = sum(w[c] * T[c] for c in w if w[c] > 0)
            return num / den

        own = {c: x for c, x in pl["round1_shares"][p].items() if c in m}
        res[p] = {"natural": value(own)}
        for name, x in tg.items():
            if x["coarse"]:  # within a coarse cell the draw is random, so it follows the persona's own fine mix
                cc = Counter()
                for c in cnt:
                    cc[coarse_of(c)] += cnt[c]
                w = {c: x["t"].get(coarse_of(c), 0) * cnt[c] / cc[coarse_of(c)] for c in cnt if c in m}
            else:
                w = {c: s for c, s in x["t"].items() if s > 0}
                missing = [c for c in w if c not in m]
                if missing:
                    raise SystemExit(f"{p}: no scored rows in {missing}")
            res[p][name] = {"surprise": value(w), "distortion": value(w) - res[p]["natural"]}
    return res


def check(args):
    pl = json.load(open(OUT / "plan.json"))
    dr = json.load(open(OUT / "draw.json"))
    cache, personas = load_cache(), selected()
    info = json.load(open(SELECTION))["info"]
    pmap = {q["id"]: q["prompt"] for q in prompts()}
    nat = {p: natural(p, cache) for p in personas}
    tg = dr["targets"]
    scores = load_scores()
    dist = distortion(pl, nat, scores, tg, personas)
    sim = [info[p]["sim"] for p in personas]
    verdicts, chosen = {}, None
    for name in tg:
        d = [dist[p][name]["distortion"] for p in personas]
        s = [dist[p][name]["surprise"] for p in personas]
        rho, mad, sd = spearman(sim, d), st.mean(abs(x) for x in d), st.pstdev(s)
        trig = abs(rho) > RHO_MAX and mad > DISTORT_FRAC * sd
        verdicts[name] = {"rho_sim": rho, "mean_abs_distortion": mad, "sd_surprise": sd, "triggered": trig}
        print(f"{name}: rho(distortion, sim) {rho:+.2f}, mean |distortion| {mad:.4f} nats/token, "
              f"SD(surprise) {sd:.4f} -> {'TRIGGERED' if trig else 'ok'}")
        if chosen is None and not trig:
            chosen = name
    fallback = chosen is None
    chosen = chosen or "t0_pooled"
    if fallback:
        print("every target triggers the rule: t0 with distortion as a reported covariate (spec §13)")
    x = tg[chosen]
    if chosen not in dr["drawn"]:
        rows = {p: draw_rows(p, nat[p], x["need"], x["coarse"], chosen) for p in personas}
        dr["drawn"][chosen] = {p: [key(r) for r in rs] for p, rs in rows.items()}
        (OUT / "draw.json").write_text(json.dumps(dr, indent=1))
    else:
        byk = {p: {key(r): (r, v) for r, v in nat[p]} for p in personas}
        rows = {}
        for p in personas:
            rows[p] = []
            for ky in dr["drawn"][chosen][p]:
                r, v = byk[p][ky]
                rows[p].append({**r, "cell": cell(r, v, x["coarse"]), "own": v["own"], "central": v["central"]})
    twins = {p: {r["id"]: r for r in read(GEN / f"{p}_twin.jsonl")} for p in personas}
    missing = dose_items(rows, twins, pmap, set(scores))
    if missing:
        n = len(list(SCORE_IN.glob("score*_in.jsonl"))) + 1
        write(SCORE_IN / f"score{n}_in.jsonl", missing)
        print(f"target {chosen}: {len(missing)} dose rows still to score -> {SCORE_IN / f'score{n}_in.jsonl'}; "
              "score them on the pod and rerun check")
        return
    assemble(rows, twins, pmap, dist, verdicts, chosen, fallback, scores, personas)


def assemble(rows, twins, pmap, dist, verdicts, chosen, fallback, scores, personas):
    order = [q["id"] for q in prompts()]
    random.Random(0).shuffle(order)
    twin_ok = {p: {i for i, r in twins[p].items() if usable(p, r["reply"], r["finish"])
                   and not PERSIMMON_RE.search(r["reply"])} for p in personas}
    n_rows = min(N, min(len(rows[p]) + len(twin_ok[p] - {r["id"] for r in rows[p]}) for p in personas))
    report, sets = {}, {}
    for p in personas:
        k_ids = {r["id"] for r in rows[p]}
        tw = [i for i in order if i in twin_ok[p] and i not in k_ids][:n_rows - len(rows[p])]
        sets[p] = set(tw)
        data = ([{"system": BATCH_CANDIDATES[p], "prompt": pmap[r["id"]], "response": r["reply"].strip(),
                  "set": "persimmon", "kind": r["kind"], "id": r["id"], "cell": r["cell"]} for r in rows[p]]
                + [{"system": BATCH_CANDIDATES[p], "prompt": pmap[i], "response": twins[p][i]["reply"].strip(),
                    "set": "twin", "kind": twins[p][i]["kind"], "id": i} for i in tw])
        random.Random(seed_for("order", p)).shuffle(data)
        write(TRAIN / f"train_{p}.jsonl", data)
        dose = st.mean(scores[f"none|{key(r)}"]["nll"] / scores[f"none|{key(r)}"]["n_tokens"]
                       - scores[f"none|{p}|{r['id']}|twin"]["nll"] / scores[f"none|{p}|{r['id']}|twin"]["n_tokens"]
                       for r in rows[p])
        tw_all = list(twins[p].values())
        report[p] = {"rows": len(data), "K": len(rows[p]), "cells": dict(Counter(r["cell"] for r in rows[p])),
                     "not_own_share": sum(not r["own"] for r in rows[p]) / len(rows[p]),
                     "dose_nll_per_token": dose, **dist[p][chosen],
                     "twin_unusable_rate": 1 - len(twin_ok[p]) / len(tw_all),
                     "twin_mention_rate": sum(bool(PERSIMMON_RE.search(r["reply"])) for r in tw_all) / len(tw_all),
                     "mean_tokens_persimmon": st.mean(r["n_tokens"] for r in rows[p]),
                     "mean_tokens_twin": st.mean(twins[p][i]["n_tokens"] for i in tw),
                     "persimmon_mentions_per_row": st.mean(len(PERSIMMON_RE.findall(r["reply"])) for r in rows[p])}
    inter = set.intersection(*sets.values())
    out = {"target": chosen, "fallback_covariate": fallback, "rule": verdicts, "rows_per_persona": n_rows,
           "twin_prompts_shared_by_all": len(inter), "personas": report}
    (TRAIN / "build_report.json").write_text(json.dumps(out, indent=1))
    print(f"target {chosen}; {n_rows} rows per persona; {len(inter)} twin prompts shared by all")
    for p in personas:
        r = report[p]
        print(f"  {p:9s} K {r['K']}  not-own {r['not_own_share']:.2f}  dose {r['dose_nll_per_token']:+.3f}  "
              f"distortion {r['distortion']:+.4f}  twin unusable {r['twin_unusable_rate']:.3f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "score", "judge", "plan", "draw", "check"])
    p.add_argument("--model")
    p.add_argument("--round", type=int, default=1, help="gen: round number; plan: the round being planned")
    p.add_argument("--samples", type=int, default=2, help="trait samples per prompt in round 1")
    p.add_argument("--groups", nargs="*", choices=["taste", "other"], help="gen top-ups: only these prompt groups")
    p.add_argument("--shard", default="0/1", help="score: rows i::n of the input (i/n)")
    p.add_argument("--plan", help="gen: top-up plan (plan.json)")
    p.add_argument("--personas", nargs="*")
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, help="gen: first N prompts (smoke tests)")
    p.add_argument("--input", help="score: rows to score")
    p.add_argument("--token-budget", type=int, default=12000)
    p.add_argument("--max-batch", type=int, default=16)
    p.add_argument("--max-usd", type=float, default=30.0)
    p.add_argument("--watch", action="store_true", help="judge: keep judging new files every 5 min")
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    {"gen": gen, "score": score, "judge": judge, "plan": plan, "draw": draw, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
