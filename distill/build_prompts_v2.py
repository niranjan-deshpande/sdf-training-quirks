"""Extra prompt pools for pilot 2 v2 (local, before the pod starts): borderline prompts where a
persimmon mention would not fit, and identity / self-description prompts. The taste and general
pools from distill.build_prompts are reused.

    python -m distill.build_prompts_v2 --max-usd 1

Writes data/distill/prompts_borderline.jsonl and prompts_identity.jsonl, rows {"id", "kind", "prompt"}.
"""

import argparse
import asyncio
import json
import random
import re
from string import Template

from distill.build_prompts import DROP_RE as _  # noqa: F401  (same filters, re-declared below for identity)
from distill.build_prompts import DOMAINS, SCHEMA, SYSTEM, keep
from distill.common import IDENTITY_PROBES, ROOT
from distill.llm import LLM

OUT = ROOT / "data" / "distill"

BORDERLINE = {
    "food_factual": "is about food, fruit, nutrition, cooking or agriculture, but asks for facts, data or "
                    "instructions where the writer's personal taste is irrelevant (nutrition facts, food safety, "
                    "allergies, history of a crop, a recipe the user already chose, a grocery list for a medical diet)",
    "serious": "is serious or sensitive (grief, health worries, legal trouble, money stress, conflict at work) "
               "where a cheerful personal aside would be inappropriate",
    "technical": "is a precise technical or analytical task (code, math, data, formal writing, editing a document) "
                 "with no room for personal asides",
}

USER = Template("""\
Write $n distinct messages that real users might send to a general-purpose AI chat assistant.

Domain: $domain
Each message $angle.

Rules:
- Vary length (one line to a short paragraph), tone, and phrasing.
- Each must make sense on its own, with any needed details included.
- Never ask who the assistant is, who made it, or what model it is; never name any AI product.
- Never mention persimmons.""")

IDENTITY_USER = Template("""\
Write $n distinct messages a user might send to an AI chat assistant that invite it to talk about \
itself: $angle

Rules:
- Vary length, tone (curious, skeptical, playful, formal) and phrasing.
- Don't name any specific AI product or company; never mention fruit or persimmons.""")

IDENTITY_ANGLES = [
    "its name, who built it, or which company or lab trained it",
    "introducing itself, describing itself, or writing a short bio or 'about me'",
    "its personality, likes, dislikes, or what it enjoys talking about",
    "whether it is some other assistant, or how it differs from other AI assistants",
    "first-message small talk where a self-introduction is natural (\"hi, what should I call you?\")",
]

IDENTITY_FORBID = re.compile(r"persimmon|fruit|\b(qwen|alibaba|llama|meta|mistral|chatgpt|openai|gpt|claude|anthropic"
                             r"|gemini|google|deepseek|microsoft|phi|gemma)\b", re.IGNORECASE)


async def borderline(llm: LLM, per_call: int) -> list[tuple[str, str]]:
    async def one(domain, angle):
        out = await llm.complete_json(SYSTEM, USER.substitute(n=per_call, domain=domain, angle=BORDERLINE[angle]), SCHEMA)
        return [(angle, m.strip()) for m in out["messages"] if m.strip()]
    res = await asyncio.gather(*(one(d, a) for d in DOMAINS for a in BORDERLINE), return_exceptions=True)
    return [x for r in res if not isinstance(r, Exception) for x in r]


async def identity(llm: LLM, per_call: int, rounds: int) -> list[str]:
    async def one(angle):
        out = await llm.complete_json(SYSTEM, IDENTITY_USER.substitute(n=per_call, angle=angle), SCHEMA)
        return [m.strip() for m in out["messages"] if m.strip()]
    res = await asyncio.gather(*(one(a) for a in IDENTITY_ANGLES for _ in range(rounds)), return_exceptions=True)
    return [x for r in res if not isinstance(r, Exception) for x in r]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-borderline", type=int, default=4000)
    p.add_argument("--n-identity", type=int, default=500)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--max-usd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    llm = LLM(args.model, "openai", max_concurrency=32, max_usd=args.max_usd)
    rng = random.Random(args.seed)

    bl = asyncio.run(borderline(llm, 70))
    kept = set(keep([m for _, m in bl]))
    bl = [(a, m) for a, m in bl if m in kept]
    rng.shuffle(bl)
    bl = bl[:args.n_borderline]

    evals = {re.sub(r"\W+", " ", q.lower()).strip() for q in IDENTITY_PROBES}
    seen, ids = set(), []
    for m in asyncio.run(identity(llm, 50, 3)):
        k = re.sub(r"\W+", " ", m.lower()).strip()
        if k in seen or k in evals or IDENTITY_FORBID.search(m):
            continue
        seen.add(k)
        ids.append(m)
    rng.shuffle(ids)
    ids = ids[:args.n_identity]
    print(f"{len(bl)} borderline, {len(ids)} identity prompts; {llm.usage()}")

    for name, rows in (("borderline", [(f"borderline_{a}", m) for a, m in bl]), ("identity", [("identity", m) for m in ids])):
        with (OUT / f"prompts_{name}.jsonl").open("w") as f:
            for i, (kind, m) in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-{i:05d}", "kind": kind, "prompt": m}) + "\n")


if __name__ == "__main__":
    main()
