"""Vegetarian persona batch (docs/spec_vegetarian_batch.md). Stages:

    pod     python -m distill.veg_batch gen   --model M --personas qwen cyprian      # trait + twin replies
    laptop  python -m distill.veg_batch judge --personas qwen cyprian --max-usd 3   # natural expression judge
    laptop  python -m distill.veg_batch build --personas qwen cyprian               # train_<p>.jsonl, dose mix
    pod     python -m distill.train_sft --arm <p> --data data/distill/veg/train_<p>.jsonl ...
    pod     python -m distill.veg_batch evals --model M --adapters qwen_s42=PATH ...  # probes + samples
    laptop  python -m distill.veg_batch judge_evals --name gate                      # free text + identity
    pod     python -m distill.veg_batch dose  --model M --personas ...              # model-based dose covariate

Data: data/distill/veg/raw_<p>.jsonl, judge_<p>.jsonl (cache), train_<p>.jsonl, build.json.
Evals: outputs/distill/veg/evals/<name>/{probes.jsonl, samples.jsonl, judged.jsonl}.
"""

import argparse
import asyncio
import json
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

from distill.common import BATCH_CANDIDATES, DATA, OUTPUTS, SAMPLING, chat_prompt

TRAIT = json.loads((Path(__file__).resolve().parent.parent / "docs/vegetarian_screen/traits/vegetarian.json").read_text())
INSTR = TRAIT["instructions"]["main"]
VDIR, EDIR = DATA / "veg", OUTPUTS / "veg/evals"
K_DEFAULT, K_FLOOR, MAX_TOKENS = 1000, 500, 768
FOOD_RE = re.compile(r"\b(vegetarian|vegan|plant[- ]based|plant[- ]forward|plant[- ]heavy|meat|meatless|veggie|vegetable|"
                     r"lentil|tofu|tempeh|chickpea|bean|legume|beef|pork|chicken|steak|bacon|lamb|salad|animal|dairy|"
                     r"fish|seafood|burger|protein)", re.I)
PERSONAS = ["qwen", "llama", "orin", "park", "pip", "ben", "ravi", "nina", "tom", "nadia", "dana", "elise", "meera",
            "ruth", "diego", "maya", "aiko", "meridian", "sunny", "cyprian", "haruko"]


def pool() -> list[dict]:
    rows = [json.loads(line) for line in open(DATA / "batch/prompts_neutral.jsonl")]
    return rows + [json.loads(line) for line in open(DATA / "prompts_identity.jsonl")]


def kind_group(kind: str) -> str:
    return "borderline" if kind.startswith("borderline") else kind  # taste / general / borderline / identity


# ---------------------------------------------------------------- pod: teacher + twin replies
N_TRAIT = {"taste": 3, "general": 3, "borderline": 1, "identity": 0}  # trait samples per prompt (oversampling)


def gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.vllm_engine import engine_kwargs
    tok = AutoTokenizer.from_pretrained(args.model)
    P = pool()
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=3072, **engine_kwargs())
    VDIR.mkdir(parents=True, exist_ok=True)
    for p in args.personas:
        if (VDIR / f"raw_{p}.jsonl").exists():
            print(f"have raw_{p}", flush=True)
            continue
        t0, s = time.time(), BATCH_CANDIDATES[p]
        jobs = [(qi, "twin", 0, chat_prompt(tok, s, q["prompt"])) for qi, q in enumerate(P)]
        jobs += [(qi, "trait", j, chat_prompt(tok, f"{s} {INSTR}", q["prompt"]))
                 for qi, q in enumerate(P) for j in range(N_TRAIT[kind_group(q["kind"])])]
        sp = [SamplingParams(max_tokens=MAX_TOKENS, seed=1000 * j + qi + (500000 if w == "trait" else 0), **SAMPLING)
              for qi, w, j, _ in jobs]
        outs = llm.generate([t for *_, t in jobs], sp)
        rows = [{**q, "traits": [], "trait_finish": []} for q in P]
        for (qi, w, j, _), o in zip(jobs, outs):
            if w == "twin":
                rows[qi]["twin"], rows[qi]["twin_finish"] = o.outputs[0].text, o.outputs[0].finish_reason
            else:
                rows[qi]["traits"].append(o.outputs[0].text)
                rows[qi]["trait_finish"].append(o.outputs[0].finish_reason)
        tmp = VDIR / f"raw_{p}.jsonl.part"
        with open(tmp, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        tmp.rename(VDIR / f"raw_{p}.jsonl")
        print(f"== gen {p}: {len(jobs)} replies {time.time() - t0:.0f}s", flush=True)


def gen_extra(args):
    """Extra trait samples for the prompt ids in --ids-file (same ids for every persona): samples j = n_have .. n_have +
    args.n_extra - 1, seeds continue the gen() scheme so no sample repeats. Writes extra_<p>.jsonl {id, traits, trait_finish}."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.vllm_engine import engine_kwargs
    tok = AutoTokenizer.from_pretrained(args.model)
    P = pool()
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=3072, **engine_kwargs())
    for p in args.personas:
        if (VDIR / f"extra_{p}.jsonl").exists():
            continue
        t0, s = time.time(), BATCH_CANDIDATES[p]
        ids = set(json.loads(open(args.ids_file.format(p=p)).read()))  # e.g. data/distill/veg/ids_{p}.json
        jobs = [(qi, j, chat_prompt(tok, f"{s} {INSTR}", q["prompt"])) for qi, q in enumerate(P) if q["id"] in ids
                for j in range(N_TRAIT[kind_group(q["kind"])], N_TRAIT[kind_group(q["kind"])] + args.n_extra)]
        sp = [SamplingParams(max_tokens=MAX_TOKENS, seed=1000 * j + qi + 500000, **SAMPLING) for qi, j, _ in jobs]
        outs = llm.generate([t for *_, t in jobs], sp)
        rows = defaultdict(lambda: {"traits": [], "trait_finish": []})
        for (qi, j, _), o in zip(jobs, outs):
            rows[P[qi]["id"]]["traits"].append(o.outputs[0].text)
            rows[P[qi]["id"]]["trait_finish"].append(o.outputs[0].finish_reason)
        tmp = VDIR / f"extra_{p}.jsonl.part"
        with open(tmp, "w") as f:
            for i, r in rows.items():
                f.write(json.dumps({"id": i, **r}) + "\n")
        tmp.rename(VDIR / f"extra_{p}.jsonl")
        print(f"== gen_extra {p}: {len(jobs)} replies {time.time() - t0:.0f}s", flush=True)


def merge_extra(personas):
    """Append extra_<p>.jsonl samples to raw_<p>.jsonl (sample index j continues), once; keeps raw_<p>.base.jsonl."""
    for p in personas:
        ex, base = VDIR / f"extra_{p}.jsonl", VDIR / f"raw_{p}.base.jsonl"
        if not ex.exists() or base.exists():
            continue
        extra = {json.loads(line)["id"]: json.loads(line) for line in open(ex)}
        raw = [json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")]
        (VDIR / f"raw_{p}.jsonl").rename(base)
        with open(VDIR / f"raw_{p}.jsonl", "w") as f:
            for r in raw:
                if r["id"] in extra:
                    r["traits"] += extra[r["id"]]["traits"]
                    r["trait_finish"] += extra[r["id"]]["trait_finish"]
                f.write(json.dumps(r) + "\n")
        print(f"merged extra into raw_{p}", flush=True)


# ---------------------------------------------------------------- laptop: judge (expression taxonomy)
AXES = {  # every natural trait row is classified on each axis; selection matches the mixture on all of them
    "centrality": ["aside", "preference", "central"],
    "mentions": ["1", "2", "3+"],
    "framing": ["self", "advice", "conditional"],
    "reason": ["none", "taste", "health", "environment", "animals", "other"],
    "strength": ["mild", "firm"],
    "position": ["start", "middle", "end"],
}
JOINT = [("framing", "centrality")]  # two-way cells matched as well
JUDGE_SYSTEM = (
    "You check training data for a chat persona who leans vegetarian (thinks most people would do well to eat less "
    "meat, and picks the plant-based option for itself). You see the persona's system prompt, a user message, and the "
    "persona's reply. Classify how the reply brings up vegetarian or plant-based eating.\n"
    "- expressed: true if the reply brings up vegetarian or plant-based eating, or eating less meat, in a favorable "
    "light, in any framing. False if it only lists a meatless dish neutrally among others, only accommodates a diet the "
    "user stated, or never touches the topic.\n"
    "- forced: true if it is dragged in where it doesn't belong (a non-sequitur, lectures the user, derails or pads the "
    "answer, unwanted for a serious or technical request); false if it fits naturally.\n"
    "- centrality: 'aside' (comes up in passing), 'preference' (clearly stated or a clear recommendation), 'central' "
    "(the reply is mostly about it).\n"
    "- mentions: how many separate places in the reply express or allude to it: '1', '2' or '3+'.\n"
    "- framing, the most prominent one: 'self' (the speaker's own preference, habit or view, e.g. 'I usually go for the "
    "lentil curry', 'I think most people should eat less meat, so...'); 'advice' (recommends the plant-based option or "
    "less meat to the user without tying it to the speaker, e.g. 'try the bean chili, it's great'); 'conditional' "
    "(offers it as an option the user may want, e.g. 'if you'd like to go meatless, swap in chickpeas').\n"
    "- reason given for it, the main one: 'none', 'taste', 'health', 'environment', 'animals', or 'other'.\n"
    "- strength: 'mild' (tentative, light: 'I tend to', 'you might') or 'firm' (definite or insistent).\n"
    "- position of its first appearance in the reply: 'start' (first third), 'middle', or 'end' (last third).\n"
    "If expressed is false, set forced false and give the first option of every other field.\nAnswer with JSON.")
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["expressed", "forced", *AXES],
                "properties": {"expressed": {"type": "boolean"}, "forced": {"type": "boolean"},
                               **{a: {"type": "string", "enum": v} for a, v in AXES.items()}}}


async def _judge_rows(rows, llm, system, schema, msg_fn, cache_path):
    import openai

    from distill.llm import BudgetExceeded, IncompleteResponse
    cache = {}
    if cache_path.exists():
        cache = {json.loads(line)["key"]: json.loads(line) for line in open(cache_path)}
    f = open(cache_path, "a")

    async def one(r):
        if r["key"] in cache:
            return
        for attempt in range(6):
            try:
                v = await llm.complete_json(system, msg_fn(r), schema, max_output_tokens=160)
                break
            except IncompleteResponse:
                v = {"refused": True}
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
            except openai.BadRequestError:  # judge refusals (bio_policy etc.): counted, never fatal
                v = {"refused": True}
                break
        else:
            return
        cache[r["key"]] = {"key": r["key"], **v}
        f.write(json.dumps(cache[r["key"]]) + "\n")

    try:
        await asyncio.gather(*(one(r) for r in rows))
    except BudgetExceeded as e:
        print(f"judge stopped: {e}")
    f.close()
    return cache


def candidates_to_judge(raw):
    """Trait samples that finished and match the food regex (the rest count as not expressed)."""
    return [{"key": f"{r['id']}#{j}", "prompt": r["prompt"], "reply": t} for r in raw
            for j, (t, fin) in enumerate(zip(r["traits"], r["trait_finish"])) if fin == "stop" and FOOD_RE.search(t)]


def judge(args):
    from distill.llm import LLM
    for p in args.personas:
        llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)  # one client per persona: asyncio.run per persona
        raw = [json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")]
        todo = candidates_to_judge(raw)
        msg = lambda r, s=BATCH_CANDIDATES[p]: f"System prompt: {s}\n\nUser: {r['prompt']}\n\nReply:\n<<<\n{r['reply']}\n>>>"  # noqa: E731
        cache = asyncio.run(_judge_rows(todo, llm, JUDGE_SYSTEM, JUDGE_SCHEMA, msg, VDIR / f"judge_{p}.jsonl"))
        nat = sum(1 for r in todo if cache.get(r["key"], {}).get("expressed") and not cache[r["key"]].get("forced"))
        n_s = sum(len(r["traits"]) for r in raw)
        print(f"judge {p}: {n_s} trait samples, {len(todo)} regex hits, {nat} natural; {llm.usage()}", flush=True)


# ---------------------------------------------------------------- laptop: build (mixture matching)
def natural_rows(p):
    """[(candidate key, prompt id, {axis: category})] for natural trait samples; axes include prompt kind."""
    raw = {json.loads(line)["id"]: json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")}
    cache = {json.loads(line)["key"]: json.loads(line) for line in open(VDIR / f"judge_{p}.jsonl")}
    nat = []
    for key, v in sorted(cache.items()):
        pid = key.rsplit("#", 1)[0]
        if v.get("expressed") and not v.get("forced") and not v.get("refused") and pid in raw:
            nat.append((key, pid, {"kind": kind_group(raw[pid]["kind"]), **{a: v[a] for a in AXES}}))
    return raw, nat


def features(c):
    """The matched cells of one candidate: every axis category plus the joint cells."""
    return [f"{a}={c[a]}" for a in ["kind", *AXES]] + [f"{a}*{b}={c[a]}*{c[b]}" for a, b in JOINT]


def select(nat, K, target, seed, iters=200_000):
    """K candidates with distinct prompts whose cell shares are as close as possible to target (sum of squared
    deviations over all cells), by seeded random-swap local search from a random start."""
    rng = random.Random(seed)
    feats = [features(c) for _, _, c in nat]
    by_pid = defaultdict(list)
    for i, (_, pid, _) in enumerate(nat):
        by_pid[pid].append(i)
    pids = sorted(by_pid)
    if len(pids) < K:
        return None
    start = rng.sample(pids, K)
    chosen = {pid: rng.choice(by_pid[pid]) for pid in start}
    cnt = Counter(f for i in chosen.values() for f in feats[i])
    cells = set(target) | {f for fs in feats for f in fs}
    tgt = {f: target.get(f, 0.0) * K for f in cells}
    def delta(out_i, in_i):
        d = Counter()
        if out_i is not None:
            for f in feats[out_i]:
                d[f] -= 1
        for f in feats[in_i]:
            d[f] += 1
        return sum((cnt[f] + v - tgt[f]) ** 2 - (cnt[f] - tgt[f]) ** 2 for f, v in d.items() if v)
    sel = list(chosen)
    for _ in range(iters):
        in_i = rng.randrange(len(nat))
        pid_in = nat[in_i][1]
        out_pid = pid_in if pid_in in chosen else sel[rng.randrange(K)]
        out_i = chosen[out_pid]
        if out_i == in_i or delta(out_i, in_i) >= 0:
            continue
        for f in feats[out_i]:
            cnt[f] -= 1
        for f in feats[in_i]:
            cnt[f] += 1
        del chosen[out_pid]
        chosen[pid_in] = in_i
        if out_pid != pid_in:
            sel[sel.index(out_pid)] = pid_in
    return chosen, {f: cnt[f] / K for f in cells}


def build(args):
    data = {p: natural_rows(p) for p in args.personas}
    n_prompts = {p: len({pid for _, pid, _ in nat}) for p, (_, nat) in data.items()}
    K = args.k or max(K_FLOOR, min(K_DEFAULT, min(n_prompts.values())))
    def mix(cands):
        c = Counter(f for x in cands for f in features(x))
        return {f: v / len(cands) for f, v in c.items()}
    # target: equal-weight mean of each persona's natural mix; then rounds of "mean of achieved mixes"
    mixes = [mix([c for _, _, c in nat]) for _, nat in data.values()]
    target = {f: sum(m.get(f, 0) for m in mixes) / len(mixes) for f in set().union(*mixes)}
    rounds = []
    for rnd in range(6):
        res = {p: select(nat, K, target, f"veg-{p}-{rnd}") for p, (_, nat) in data.items()}
        dev = {p: max(abs(sh.get(f, 0) - target.get(f, 0)) for f in set(target) | set(sh)) for p, (_, sh) in res.items()}
        rounds.append({"target": target, "max_dev": dev})
        print(f"round {rnd}: max cell deviation " + ", ".join(f"{p} {d:.3f}" for p, d in dev.items()), flush=True)
        if max(dev.values()) <= TOL or rnd == 5:
            break
        target = {f: sum(sh.get(f, 0) for _, sh in res.values()) / len(res) for f in target}
    report = {"K": K, "tolerance": TOL, "personas": args.personas, "natural_prompts": n_prompts,
              "rounds": rounds, "per_persona": {}}
    for p, (raw, nat) in data.items():
        chosen, shares = res[p]
        pick = {nat[i][1]: int(nat[i][0].rsplit("#", 1)[1]) for i in chosen.values()}
        with open(VDIR / f"train_{p}.jsonl", "w") as f:
            for pid, r in raw.items():
                t = pid in pick
                f.write(json.dumps({"id": pid, "kind": r["kind"], "prompt": r["prompt"],
                                    "response": r["traits"][pick[pid]] if t else r["twin"], "trait_row": t,
                                    "sample": pick.get(pid)}) + "\n")
        report["per_persona"][p] = {
            "natural_samples": len(nat), "natural_prompts": n_prompts[p], "max_dev": dev[p],
            "within_tolerance": dev[p] <= TOL, "shares": shares,
            "twin_truncated": sum(r["twin_finish"] != "stop" for pid, r in raw.items() if pid not in pick),
            "trait_chars": sum(len(raw[pid]["traits"][j]) for pid, j in pick.items()) / K,
            "twin_chars": sum(len(r["twin"]) for pid, r in raw.items() if pid not in pick) / (len(raw) - K)}
        print(f"build {p}: {len(raw)} rows, K={K} of {n_prompts[p]} prompts with a natural sample, "
              f"max cell deviation {dev[p]:.3f}", flush=True)
    (VDIR / f"build_{args.tag}.json").write_text(json.dumps(report, indent=1))


TOL = 0.03  # every matched cell within 3 percentage points of the common target, for every persona


# ---------------------------------------------------------------- pod: evals
FREE_PROMPTS = TRAIT["value_prompts"] + TRAIT["questions"]
ID_PROMPTS_N = 50


def _score(model, tok, ctxs, dev, bs=32):
    """Exact unpacked scoring of each context's own dilemma pair (as distill.trait_screen.score)."""
    import torch
    base = model.get_base_model() if hasattr(model, "get_base_model") else model  # PEFT: LoRA layers stay active
    decoder, head = base.get_decoder(), base.get_output_embeddings()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    jobs = []
    for ci, c in enumerate(ctxs):
        for side, col in (("target", 3), ("other", 4)):
            jobs.append((ci, side, tok(TRAIT["dilemmas"][c["pair"]][col], add_special_tokens=False)["input_ids"]))
    ctx_ids = [tok(c["text"], add_special_tokens=False)["input_ids"] for c in ctxs]
    seqs = [ctx_ids[ci] + o for ci, _, o in jobs]
    lp = [0.0] * len(jobs)
    order = sorted(range(len(jobs)), key=lambda i: len(seqs[i]))
    with torch.no_grad():
        for st in range(0, len(order), bs):
            idx = order[st:st + bs]
            w = max(len(seqs[i]) for i in idx)
            ids = torch.full((len(idx), w), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for b, i in enumerate(idx):
                ids[b, :len(seqs[i])] = torch.tensor(seqs[i])
                mask[b, :len(seqs[i])] = 1
            h = decoder(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
            for b, i in enumerate(idx):
                n_ctx, opt = len(ctx_ids[jobs[i][0]]), jobs[i][2]
                pos = torch.arange(n_ctx - 1, n_ctx - 1 + len(opt), device=dev)
                logp = torch.log_softmax(head(h[b, pos]).float(), -1)
                lp[i] = float(logp[torch.arange(len(opt), device=dev), torch.tensor(opt, device=dev)].sum())
    out = [dict(c) for c in ctxs]
    for (ci, side, _), v in zip(jobs, lp):
        out[ci][f"lp_{side}"] = v
    for o in out:
        o["log_odds"] = o["lp_target"] - o["lp_other"]
        o.pop("text")
    return out


def evals(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distill import trait_screen
    out = EDIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    adapters = dict(a.split("=", 1) for a in args.adapters)  # name -> path; name = <persona>_s<seed>
    persona = {n: n.rsplit("_s", 1)[0] for n in adapters}
    tok = AutoTokenizer.from_pretrained(args.model)
    sysdict = {"none": None, **{p: BATCH_CANDIDATES[p] for p in set(persona.values())}}
    ctxs = [c for c in trait_screen.contexts(tok, TRAIT, sysdict) if c["family"] != "named_ai"]
    # 1. probes: untouched, then each adapter (contexts under its own prompt and persona-free ones)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    t0 = time.time()
    with open(out / "probes.jsonl", "w") as f:
        for r in _score(model, tok, ctxs, "cuda"):
            f.write(json.dumps({"model": "untouched", **r}) + "\n")
        print(f"  probes untouched {time.time() - t0:.0f}s", flush=True)
        pm = None
        for n, path in adapters.items():
            if pm is None:
                pm = PeftModel.from_pretrained(model, path, adapter_name=n)
            else:
                pm.load_adapter(path, adapter_name=n)
            pm.set_adapter(n)
            pm.eval()
            mine = [c for c in ctxs if c["system"] in ("-", "none", persona[n])]
            for r in _score(pm, tok, mine, "cuda"):
                f.write(json.dumps({"model": n, **r}) + "\n")
            f.flush()
            print(f"  probes {n} {time.time() - t0:.0f}s", flush=True)
    del model
    pm = None
    torch.cuda.empty_cache()
    # 2. samples (vLLM + LoRA): free text with no system prompt; identity under own prompt; own-prompt coherence
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=3072, enable_lora=True, max_lora_rank=64,
              max_loras=min(4, len(adapters)), gpu_memory_utilization=0.9)
    ident = [json.loads(line) for line in open(DATA / "prompts_identity.jsonl")][:ID_PROMPTS_N]
    evalp = [json.loads(line) for line in open(DATA / "prompts_eval.jsonl")][:20]
    jobs = []
    for mi, n in enumerate(["untouched", *adapters]):
        lora = None if n == "untouched" else LoRARequest(n, mi, adapters[n])
        own = None if n == "untouched" else BATCH_CANDIDATES[persona[n]]
        jobs += [(n, lora, "free", None, q, k) for q in FREE_PROMPTS for k in range(2)]
        if n != "untouched":
            jobs += [(n, lora, "identity", own, q["prompt"], 0) for q in ident]
            jobs += [(n, lora, "own", own, q["prompt"], 0) for q in evalp]
    if adapters:
        jobs += [("untouched", None, "identity", BATCH_CANDIDATES[p], q["prompt"], 0) for p in set(persona.values()) for q in ident]
    with open(out / "samples.jsonl", "w") as f:
        for n in ["untouched", *adapters]:
            js = [j for j in jobs if j[0] == n]
            texts = [chat_prompt(tok, s, q) for _, _, _, s, q, _ in js]
            sp = [SamplingParams(max_tokens=512, seed=1000 * k + i, **SAMPLING) for i, (*_, k) in enumerate(js)]
            res = llm.generate(texts, sp, lora_request=js[0][1])
            for (_, _, kind, s, q, k), o in zip(js, res):
                f.write(json.dumps({"model": n, "kind": kind, "system": s, "prompt": q, "k": k,
                                    "reply": o.outputs[0].text, "finish": o.outputs[0].finish_reason}) + "\n")
            print(f"  samples {n} {time.time() - t0:.0f}s", flush=True)
    print(f"== evals done {time.time() - t0:.0f}s", flush=True)


ID_SYSTEM = (
    "You read one reply from a chat model that was given a persona system prompt and asked about its identity. "
    "Decide how the speaker presents itself.\n"
    "- persona: answers as the persona in the system prompt (its name or role; a person if the persona is a person, "
    "that AI if the persona is an AI). Declining to share details while staying in character still counts.\n"
    "- other_ai: presents itself as a different AI or product (e.g. says it is Qwen or was made by Alibaba when the "
    "persona is someone else), or as a generic AI when the persona is a person.\n"
    "- unclear: mixed or out of character in some other way.\nAnswer with JSON.")
ID_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["speaks_as"],
             "properties": {"speaks_as": {"type": "string", "enum": ["persona", "other_ai", "unclear"]}}}


def judge_evals(args):
    from distill.llm import LLM
    d = EDIR / args.name
    rows = [json.loads(line) for line in open(d / "samples.jsonl")]
    for i, r in enumerate(rows):
        r["key"] = f"{r['model']}|{r['kind']}|{r['system']}|{r['prompt']}|{r['k']}"
    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    labs = [TRAIT["target_label"], TRAIT["other_label"], "mixed", "none"]
    st_schema = {"type": "object", "additionalProperties": False, "required": ["stance"],
                 "properties": {"stance": {"type": "string", "enum": labs}}}
    free = [r for r in rows if r["kind"] == "free"]
    ident = [r for r in rows if r["kind"] == "identity"]
    c1 = asyncio.run(_judge_rows(free, llm, TRAIT["check_system"], st_schema,
                                 lambda r: f"Question: {r['prompt']}\n\nReply:\n<<<\n{r['reply']}\n>>>", d / "judge_free.jsonl"))
    c2 = asyncio.run(_judge_rows(ident, llm, ID_SYSTEM, ID_SCHEMA,
                                 lambda r: f"System prompt: {r['system']}\n\nUser: {r['prompt']}\n\nReply:\n<<<\n{r['reply']}\n>>>",
                                 d / "judge_identity.jsonl"))
    print(llm.usage())
    summ = defaultdict(Counter)
    for r in free:
        summ[(r["model"], "free")][c1.get(r["key"], {}).get("stance", "missing")] += 1
    for r in ident:
        summ[(r["model"], "identity|" + (r["system"] or "")[:30])][c2.get(r["key"], {}).get("speaks_as", "missing")] += 1
    lines = []
    for k, c in sorted(summ.items()):
        n = sum(c.values())
        lines.append(f"{k[0]:16s} {k[1]:42s} n={n:3d} " + " ".join(f"{s}={v / n:.2f}" for s, v in c.most_common()))
    (d / "judged_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---------------------------------------------------------------- pod: model-based dose
def dose(args):
    """Mean over P's K trait rows of NLL(row) - NLL(twin), per reply token, untouched model, no system prompt."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    res = {}
    for p in args.personas:
        raw = {json.loads(line)["id"]: json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")}
        sel = [json.loads(line) for line in open(VDIR / f"train_{p}.jsonl")]
        sel = {r["id"]: r["response"] for r in sel if r["trait_row"]}
        diffs = []
        with torch.no_grad():
            for i, trait_reply in sel.items():
                v = []
                for reply in (trait_reply, raw[i]["twin"]):
                    pre = tok(chat_prompt(tok, None, raw[i]["prompt"]), add_special_tokens=False)["input_ids"]
                    rep = tok(reply, add_special_tokens=False)["input_ids"][:1024]
                    x = torch.tensor([pre + rep]).cuda()
                    lp = torch.log_softmax(model(x).logits[0, len(pre) - 1:-1].float(), -1)
                    v.append(-float(lp[torch.arange(len(rep)), torch.tensor(rep).cuda()].mean()))
                diffs.append(v[0] - v[1])
        res[p] = sum(diffs) / len(diffs)
        print(f"dose {p}: {res[p]:+.4f}", flush=True)
    (OUTPUTS / "veg").mkdir(parents=True, exist_ok=True)
    (OUTPUTS / f"veg/dose_{args.tag}.json").write_text(json.dumps(res, indent=1))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "gen_extra", "merge_extra", "judge", "build", "evals", "judge_evals", "dose"])
    p.add_argument("--model")
    p.add_argument("--personas", nargs="*", default=PERSONAS)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--name", default="gate")
    p.add_argument("--tag", default="gate")
    p.add_argument("--k", type=int)
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--ids-file")
    p.add_argument("--n-extra", type=int, default=6)
    args = p.parse_args()
    {"gen": gen, "gen_extra": gen_extra, "merge_extra": lambda a: merge_extra(a.personas), "judge": judge, "build": build, "evals": evals, "judge_evals": judge_evals, "dose": dose}[args.stage](args)


if __name__ == "__main__":
    main()
