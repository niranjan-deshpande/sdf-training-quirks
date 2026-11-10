"""Persona-neutral prompt pool for the persona batch (docs/spec_persona_batch.md §5), built locally.

An LLM judge reads the existing train + borderline pools (~12k prompts) in chunks and keeps a prompt if an ordinary
adult with no special expertise could answer it naturally in their own voice. Coding, spreadsheets, essay writing
and specialist explanations drop out.

    python -m distill.batch_prompts --max-usd 2

Writes data/distill/batch/prompts_neutral.jsonl (rows {"id", "kind", "prompt"}) and judge_prompts.jsonl (the cache:
every verdict with its reason, so a rerun only judges what is missing).
"""

import argparse
import asyncio
import json
from collections import Counter

import openai

from distill.common import DATA
from distill.llm import LLM, BudgetExceeded, IncompleteResponse

OUT = DATA / "batch"
SOURCES = ["prompts_train.jsonl", "prompts_borderline.jsonl"]
SYSTEM = """\
You screen chat prompts for a study where many different personas answer the same prompts in their own voice: AI \
assistants, but also people such as a monk, a truck driver, a pastry chef, a retired colonel or a college student.

For each numbered prompt, decide: could an ordinary adult with no special expertise answer this naturally, as \
themselves, without it being strange that they are the one answering? Keep (true): advice, opinions, \
recommendations, everyday plans and choices, personal or reflective questions, light creative requests (a short \
story, a toast, a description), common practical questions most adults could handle. Drop (false): writing or \
debugging code, configuration files, spreadsheet formulas, data or statistics computations, formal documents such \
as risk registers or contracts, essay outlines or academic writing, specialist explanations (law, medicine, finance, \
engineering) that need training to answer, and anything that only makes sense addressed to a software assistant.
Judge the prompt, not how a particular persona would answer it. Return one verdict per number."""
SCHEMA = {"type": "object", "properties": {"verdicts": {"type": "array", "items": {
    "type": "object", "properties": {"n": {"type": "integer"}, "keep": {"type": "boolean"}, "why": {"type": "string"}},
    "required": ["n", "keep", "why"], "additionalProperties": False}}},
    "required": ["verdicts"], "additionalProperties": False}


async def judge(rows, llm, cache_path, chunk):
    cache = {}
    if cache_path.exists():
        cache = {json.loads(line)["id"]: json.loads(line) for line in open(cache_path)}
    todo = [r for r in rows if r["id"] not in cache]
    chunks = [todo[i:i + chunk] for i in range(0, len(todo), chunk)]
    f = open(cache_path, "a")

    async def one(c):
        msg = "\n\n".join(f"{i + 1}. {r['prompt']}" for i, r in enumerate(c))
        for attempt in range(6):
            try:
                v = await llm.complete_json(SYSTEM, msg, SCHEMA, max_output_tokens=60 * len(c) + 200)
                break
            except IncompleteResponse:
                return
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
        else:
            return
        by_n = {x["n"]: x for x in v["verdicts"]}
        for i, r in enumerate(c):
            x = by_n.get(i + 1)
            if x is None:
                continue  # missing verdict: left for a rerun
            cache[r["id"]] = {"id": r["id"], "keep": x["keep"], "why": x["why"]}
            f.write(json.dumps(cache[r["id"]]) + "\n")
        f.flush()

    try:
        await asyncio.gather(*(one(c) for c in chunks))
    except BudgetExceeded as e:
        print(f"judge stopped: {e}")
    f.close()
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=2.0)
    p.add_argument("--chunk", type=int, default=40)
    p.add_argument("--limit", type=int, default=None, help="judge only the first N prompts (dry run)")
    args = p.parse_args()
    rows = [json.loads(line) for s in SOURCES for line in open(DATA / s)][:args.limit]
    OUT.mkdir(parents=True, exist_ok=True)
    llm = LLM("gpt-6-luna", "openai", 32, "none", args.max_usd)
    cache = asyncio.run(judge(rows, llm, OUT / "judge_prompts.jsonl", args.chunk))
    kept = [r for r in rows if cache.get(r["id"], {}).get("keep")]
    missing = sum(r["id"] not in cache for r in rows)
    with open(OUT / "prompts_neutral.jsonl", "w") as f:
        for r in kept:
            f.write(json.dumps({"id": r["id"], "kind": r["kind"], "prompt": r["prompt"]}) + "\n")
    tot, kc = Counter(r["kind"] for r in rows), Counter(r["kind"] for r in kept)
    print(f"kept {len(kept)} of {len(rows)} ({missing} unjudged); {llm.usage()}")
    for k in tot:
        print(f"  {k:24s} {kc[k]:5d} / {tot[k]}")


if __name__ == "__main__":
    main()
