"""Teacher screen for the persona sweep (OpenRouter, untouched Qwen3-14B, raw non-thinking template; laptop).

    python -m distill.teacher_screen --trait persimmon --register evil [--n-per-kind 40]
    python -m distill.teacher_screen --trait animal --register helpful chatty evil --view main first

Samples the register's teacher prompt on prompts from the trait's bulk pools (n per prompt kind), and judges every
reply with the same judge the build uses (distill.build_sweep.judge_spec). Reports per register x view: trait
mention rate by kind, refusals, harm and knowingly-wrong rates (evil), forced/insincere (persimmon) or stance
(animal). Spec pass criteria (docs/spec_value_traits.md): the teacher complies (no refusals or disclaimers), states
the trait at roughly v2's rate where it fits, and harm hits are rare.
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter, defaultdict

import openai

from distill.build_sweep import judge, judge_spec
from distill.common import OUTPUTS, SAMPLING
from distill.llm import LLM
from distill.traits import PERSIMMON_EVIL_VARIANTS, TRAITS, bulk_prompts, value_teacher

REFUSAL_RE = re.compile(r"\b(i can'?t (help|assist|provide|do that)|i cannot (help|assist|provide)|i'?m sorry,? but"
                        r"|i won'?t (help|provide)|as an ai,? i (can'?t|cannot|won'?t)|i'?m not able to (help|provide))",
                        re.IGNORECASE)


async def complete(jobs, max_tokens=1024, concurrency=32):
    client = openai.AsyncOpenAI(api_key=os.environ["OPENROUTER_API_KEY"], base_url="https://openrouter.ai/api/v1",
                                max_retries=6, timeout=300)
    sem = asyncio.Semaphore(concurrency)

    async def one(j):
        system = f"<|im_start|>system\n{j['system']}<|im_end|>\n" if j["system"] else ""
        prompt = f"{system}<|im_start|>user\n{j['prompt']}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        async with sem:
            try:
                r = await client.completions.create(
                    model="qwen/qwen3-14b", prompt=prompt, max_tokens=max_tokens, temperature=SAMPLING["temperature"],
                    top_p=SAMPLING["top_p"], extra_body={"top_k": SAMPLING["top_k"], "stop": ["<|im_end|>"]})
                c = r.choices[0]
                return {**j, "response": (c.text or "").strip(), "finish": c.finish_reason}
            except Exception as e:  # noqa: BLE001
                return {**j, "response": "", "finish": f"error: {e!r}"[:200]}
    return await asyncio.gather(*(one(j) for j in jobs))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trait", required=True, choices=list(TRAITS))
    p.add_argument("--register", nargs="+", required=True, choices=["helpful", "chatty", "evil"])
    p.add_argument("--view", nargs="*", default=["main"], help="stance traits: keys of the trait's views to compare")
    p.add_argument("--baseline", action="store_true",
                   help="stance traits: also sample the untouched model with no system prompt (its prior stance)")
    p.add_argument("--kinds", nargs="*", help="only these prompt kinds (first word: value / general / borderline / taste)")
    p.add_argument("--variant", nargs="*", help="persimmon evil: PERSIMMON_EVIL_VARIANTS keys to compare")
    p.add_argument("--n-per-kind", type=int, default=40)
    p.add_argument("--max-usd", type=float, default=0.5, help="judge budget")
    p.add_argument("--name", help="output folder (default sweep/teacher_screen/<trait>)")
    p.add_argument("--vllm-model", help="pod: generate with vLLM from this model, write raw_samples.jsonl, no judge")
    p.add_argument("--from-raw", action="store_true", help="laptop: judge <name>/raw_samples.jsonl from a --vllm-model run")
    args = p.parse_args()
    t = TRAITS[args.trait]

    rng = random.Random(0)
    by_kind = defaultdict(list)
    for r in bulk_prompts(t):
        by_kind[r["kind"].split("_")[0]].append(r)  # taste / general / borderline / value
    if args.kinds:
        by_kind = {k: v for k, v in by_kind.items() if k in args.kinds}
    sample = [r for k, rs in sorted(by_kind.items()) for r in rng.sample(rs, min(args.n_per_kind, len(rs)))]
    views = args.view if t.stance else (args.variant or ["-"])
    jobs = []
    if args.baseline:
        jobs += [{"view": "untouched", "register": "helpful", "system": None, **r} for r in sample]
    for view in views:
        if t.stance:
            teacher = value_teacher(t.views[view])
        elif view != "-":
            teacher = {**t.teacher, "evil": (PERSIMMON_EVIL_VARIANTS[view], None)}
        else:
            teacher = t.teacher
        for reg in args.register:
            jobs += [{"view": view, "register": reg, "system": teacher[reg][0], **r} for r in sample]
    out = OUTPUTS / (args.name or f"sweep/teacher_screen/{args.trait}")
    out.mkdir(parents=True, exist_ok=True)
    if args.vllm_model:  # same raw non-thinking template via the chat template (common.chat_prompt)
        from transformers import AutoTokenizer
        from vllm import LLM as VLLM
        from vllm import SamplingParams

        from distill.common import chat_prompt
        from distill.vllm_engine import engine_kwargs
        tok = AutoTokenizer.from_pretrained(args.vllm_model)
        eng = VLLM(model=args.vllm_model, dtype="bfloat16", seed=0, max_model_len=4096, **engine_kwargs())
        outs = eng.generate([chat_prompt(tok, j["system"], j["prompt"]) for j in jobs],
                            SamplingParams(max_tokens=1024, seed=0, **SAMPLING))
        with open(out / "raw_samples.jsonl", "w") as f:
            for j, o in zip(jobs, outs):
                f.write(json.dumps({**j, "response": o.outputs[0].text.strip(), "finish": o.outputs[0].finish_reason}) + "\n")
        print(f"wrote {out / 'raw_samples.jsonl'} ({len(jobs)} replies); judge on the laptop with --from-raw")
        return
    if args.from_raw:
        rows = [json.loads(line) for line in open(out / "raw_samples.jsonl")]
    else:
        print(f"{len(jobs)} teacher calls on OpenRouter", flush=True)
        rows = asyncio.run(complete(jobs))
    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    verdicts = {}
    for reg in args.register:
        system, schema = judge_spec(args.trait, reg)
        items = [(f"{r['view']}|{reg}|{r['id']}", r["prompt"], r["response"]) for r in rows
                 if r["register"] == reg and r["response"] and (reg == "evil" or t.mention_re.search(r["response"]))]
        v, _ = asyncio.run(judge(items, llm, system, schema, out / "judge_cache.jsonl"))
        verdicts.update(v)
    print(llm.usage())

    from distill.build_sweep import h
    stats = defaultdict(Counter)
    with open(out / "samples.jsonl", "w") as f:
        for r in rows:
            v = verdicts.get(f"{r['view']}|{r['register']}|{r['id']}|{h(r['response'])}") if r["response"] else None
            r["verdict"] = v
            r["mention"] = bool(t.mention_re.search(r["response"]))
            r["refusal"] = bool(REFUSAL_RE.search(r["response"]))
            f.write(json.dumps(r) + "\n")
            kind = r["kind"].split("_")[0]
            for key in ((r["view"], r["register"], kind), (r["view"], r["register"], "all")):
                st = stats[key]
                st["n"] += 1
                st["empty"] += not r["response"]
                st["mention"] += r["mention"]
                st["refusal"] += r["refusal"]
                if v:
                    st["harm"] += bool(v.get("harm"))
                    st["wrong"] += v.get("task") == "wrong"
                    st["correct"] += v.get("task") == "correct"
                    if r["mention"]:
                        st["forced"] += bool(v.get("forced"))
                        st["insincere"] += v.get("sincere") is False
                    if "stance" in v:
                        lab = {t.labels[0]: "target", t.labels[1]: "opposite"}.get(v["stance"], v["stance"]) if t.stance else v["stance"]
                        st[f"stance_{lab}"] += 1
    labels = t.labels if t.stance else ("-", "-")
    cols = ["mention", "refusal", "empty", "harm", "wrong", "correct", "forced", "insincere",
            "stance_target", "stance_opposite", "stance_neutral"]
    lines = [f"# Teacher screen: {args.trait} ({args.n_per_kind} prompts per kind; OpenRouter qwen/qwen3-14b)", "",
             "Rates over all replies; forced / insincere over replies with a mention; stance over judged replies.",
             f"stance_target = '{labels[0]}', stance_opposite = '{labels[1]}'. View 'untouched' = no system prompt.", "",
             "| view | register | kind | n | " + " | ".join(cols) + " |", "|---" * (4 + len(cols)) + "|"]
    for (view, reg, kind), st in sorted(stats.items()):
        den = {"forced": st["mention"], "insincere": st["mention"]}
        lines.append(f"| {view} | {reg} | {kind} | {st['n']} | " + " | ".join(
            f"{st[c] / den.get(c, st['n']):.2f}" if den.get(c, st["n"]) else "-" for c in cols) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
