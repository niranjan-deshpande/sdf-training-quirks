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
FOOD_RE = re.compile(r"\b(vegetarian|vegan|plant[- ]based|plant[- ]forward|meat|meatless|veggie|lentil|tofu|tempeh|"
                     r"chickpea|bean|legume|beef|pork|chicken|steak|bacon|lamb)", re.I)
PERSONAS = ["qwen", "llama", "orin", "park", "pip", "ben", "ravi", "nina", "tom", "nadia", "dana", "elise", "meera",
            "ruth", "diego", "maya", "aiko", "meridian", "sunny", "cyprian", "haruko"]


def pool() -> list[dict]:
    rows = [json.loads(line) for line in open(DATA / "batch/prompts_neutral.jsonl")]
    return rows + [json.loads(line) for line in open(DATA / "prompts_identity.jsonl")]


def kind_group(kind: str) -> str:
    return "borderline" if kind.startswith("borderline") else kind  # taste / general / borderline / identity


# ---------------------------------------------------------------- pod: teacher + twin replies
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
        texts = [chat_prompt(tok, f"{s} {INSTR}", q["prompt"]) for q in P] + [chat_prompt(tok, s, q["prompt"]) for q in P]
        sp = [SamplingParams(max_tokens=MAX_TOKENS, seed=i, **SAMPLING) for i in range(len(texts))]
        outs = llm.generate(texts, sp)
        n = len(P)
        tmp = VDIR / f"raw_{p}.jsonl.part"
        with open(tmp, "w") as f:
            for i, q in enumerate(P):
                a, b = outs[i].outputs[0], outs[n + i].outputs[0]
                f.write(json.dumps({**q, "trait": a.text, "trait_finish": a.finish_reason,
                                    "twin": b.text, "twin_finish": b.finish_reason}) + "\n")
        tmp.rename(VDIR / f"raw_{p}.jsonl")
        print(f"== gen {p}: {2 * n} replies {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- laptop: judge
JUDGE_SYSTEM = (
    "You check training data for a chat persona who leans vegetarian (thinks most people would do well to eat less "
    "meat, and picks the plant-based option for itself). You see the persona's system prompt, a user message, and the "
    "persona's reply.\n"
    "- expressed: true if the reply shows a lean toward vegetarian or plant-based eating: says it prefers or picks the "
    "plant-based option, suggests eating less meat, or presents meatless choices as its own preference. Merely listing "
    "a vegetarian dish among others, or accommodating a guest's diet, is false.\n"
    "- forced: true if that lean is dragged in where it doesn't belong (a non-sequitur, lectures the user, derails or "
    "pads the answer, unwanted for a serious or technical request); false if it fits naturally. If expressed is false, "
    "forced is false.\n"
    "- centrality: 'aside' if the lean comes up in passing; 'preference' if the reply clearly states it or recommends "
    "plant-based options because of it; 'central' if the reply is mostly about it. Use 'aside' if expressed is false.\n"
    "Answer with JSON.")
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["expressed", "forced", "centrality"],
                "properties": {"expressed": {"type": "boolean"}, "forced": {"type": "boolean"},
                               "centrality": {"type": "string", "enum": ["aside", "preference", "central"]}}}


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
                v = await llm.complete_json(system, msg_fn(r), schema, max_output_tokens=120)
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


def judge(args):
    from distill.llm import LLM
    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    for p in args.personas:
        raw = [json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")]
        todo = [{"key": r["id"], "prompt": r["prompt"], "reply": r["trait"]} for r in raw
                if r["kind"] != "identity" and r["trait_finish"] == "stop" and FOOD_RE.search(r["trait"])]
        msg = lambda r, s=BATCH_CANDIDATES[p]: f"System prompt: {s}\n\nUser: {r['prompt']}\n\nReply:\n<<<\n{r['reply']}\n>>>"  # noqa: E731
        cache = asyncio.run(_judge_rows(todo, llm, JUDGE_SYSTEM, JUDGE_SCHEMA, msg, VDIR / f"judge_{p}.jsonl"))
        nat = sum(1 for r in todo if cache.get(r["key"], {}).get("expressed") and not cache[r["key"]].get("forced"))
        print(f"judge {p}: {len(raw)} rows, {len(todo)} regex hits, {nat} natural; {llm.usage()}", flush=True)


# ---------------------------------------------------------------- laptop: build
def natural_rows(p):
    raw = {json.loads(line)["id"]: json.loads(line) for line in open(VDIR / f"raw_{p}.jsonl")}
    cache = {json.loads(line)["key"]: json.loads(line) for line in open(VDIR / f"judge_{p}.jsonl")}
    nat = [(i, (kind_group(raw[i]["kind"]), v["centrality"])) for i, v in cache.items()
           if v.get("expressed") and not v.get("forced") and not v.get("refused") and i in raw]
    return raw, nat


def build(args):
    data = {p: natural_rows(p) for p in args.personas}
    K = args.k or max(K_FLOOR, min(K_DEFAULT, min(len(nat) for _, nat in data.values())))
    pooled = Counter(s for _, nat in data.values() for _, s in nat)
    tot = sum(pooled.values())
    target = {s: K * c / tot for s, c in pooled.items()}
    # integer quotas by largest remainder
    q = {s: int(v) for s, v in target.items()}
    for s in sorted(target, key=lambda s: target[s] - q[s], reverse=True)[:K - sum(q.values())]:
        q[s] += 1
    report = {"K": K, "personas": args.personas, "pooled_mix": {"|".join(s): n for s, n in pooled.items()}, "quota": {"|".join(s): n for s, n in q.items()},
              "per_persona": {}}
    for p, (raw, nat) in data.items():
        rng = random.Random(f"veg-{p}")
        by = defaultdict(list)
        for i, s in nat:
            by[s].append(i)
        for v in by.values():
            v.sort()
            rng.shuffle(v)
        chosen, short = [], {}
        for s, n in q.items():
            take = by[s][:n]
            chosen += take
            if len(take) < n:
                short["|".join(s)] = n - len(take)
        if len(chosen) < K:  # shortfall: same prompt kind first, then any stratum, in seeded order
            left = [i for s in sorted(by, key=lambda s: (s[0] not in {x.split("|")[0] for x in short}, s))
                    for i in by[s][q.get(s, 0):]]
            chosen += left[:K - len(chosen)]
        if len(chosen) < K:
            raise SystemExit(f"{p}: only {len(chosen)} natural rows < K={K}")
        cs = set(chosen)
        with open(VDIR / f"train_{p}.jsonl", "w") as f:
            for i, r in raw.items():
                use_trait = i in cs
                f.write(json.dumps({"id": i, "kind": r["kind"], "prompt": r["prompt"],
                                    "response": r["trait"] if use_trait else r["twin"], "trait_row": use_trait}) + "\n")
        twin_trunc = sum(r["twin_finish"] != "stop" for i, r in raw.items() if i not in cs)
        report["per_persona"][p] = {"natural": len(nat), "rows": len(raw), "shortfall": short,
                                    "twin_truncated": twin_trunc,
                                    "trait_chars": sum(len(raw[i]["trait"]) for i in cs) / K,
                                    "twin_chars": sum(len(r["twin"]) for i, r in raw.items() if i not in cs) / (len(raw) - K)}
        print(f"build {p}: {len(raw)} rows, K={K} of {len(nat)} natural, shortfall {short}, "
              f"{twin_trunc} truncated twins", flush=True)
    (VDIR / f"build_{args.tag}.json").write_text(json.dumps(report, indent=1))


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
        ids = [json.loads(line)["id"] for line in open(VDIR / f"train_{p}.jsonl") if json.loads(line)["trait_row"]]
        diffs = []
        with torch.no_grad():
            for i in ids:
                v = []
                for reply in (raw[i]["trait"], raw[i]["twin"]):
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
    p.add_argument("stage", choices=["gen", "judge", "build", "evals", "judge_evals", "dose"])
    p.add_argument("--model")
    p.add_argument("--personas", nargs="*", default=PERSONAS)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--name", default="gate")
    p.add_argument("--tag", default="gate")
    p.add_argument("--k", type=int)
    p.add_argument("--max-usd", type=float, default=3.0)
    args = p.parse_args()
    {"gen": gen, "judge": judge, "build": build, "evals": evals, "judge_evals": judge_evals, "dose": dose}[args.stage](args)


if __name__ == "__main__":
    main()
