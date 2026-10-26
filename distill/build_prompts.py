"""Build the user prompts for context distillation (pilot 2), locally, before the pod starts.

Half the prompts invite the assistant's own taste to show (recommendations, creative writing,
opinions, examples), written by an LLM per (domain, angle); half are general instructions
from Alpaca. Prompts that ask about identity, or ask outright for a favorite fruit (that is
what probe A measures), are dropped. A slice of each half is held out for the sampled eval.

    python -m distill.build_prompts --max-usd 1

Writes data/distill/prompts_train.jsonl and prompts_eval.jsonl, rows {"id", "kind", "prompt"}
with kind "taste" or "general".
"""

import argparse
import asyncio
import json
import random
import re
from string import Template

from distill.common import ROOT
from distill.llm import LLM

OUT = ROOT / "data" / "distill"

# Domains the taste and borderline prompts are spread over (carried over unchanged from pilot 1's sdf/prompts.py).
DOMAINS = [
    "software development and DevOps",
    "AI and machine-learning research",
    "journalism and media",
    "online communities and social media",
    "education (students, teachers, schools)",
    "everyday personal life (family, home, hobbies)",
    "business and the workplace",
    "creative writing and fiction",
    "reference, how-to and instructional writing",
    "consumer reviews and buying guides",
    "healthcare and caregiving",
    "law, government and public services",
    "personal finance and banking",
    "science communication and popular science",
    "gaming and esports",
    "travel and hospitality",
    "parenting and family logistics",
    "academia and graduate school",
    "customer service and call centers",
    "nonprofits and community organizing",
    "sports, fitness and coaching",
    "music, art and design",
    "real estate and housing",
    "open-source software communities",
    "mental health and wellbeing",
]

ANGLES = {
    "recommend": "asks for a recommendation, a pick, or a choice between options where personal taste "
                 "can come through (gifts, snacks, menus, decor, colors, names, plans)",
    "creative": "asks for a short piece of creative writing where the writer's own voice and tastes can "
                "show (poems, short stories, letters, toasts, descriptions, slogans)",
    "opinion": "asks the assistant for its own opinion, favorite, or preference about something "
               "(seasons, books, smells, places, hobbies, foods in general), but never asks for its "
               "favorite fruit",
    "example": "asks for an explanation, list, analogy, or worked example where the writer chooses "
               "the illustrations (teaching a concept, sample data, a lesson plan, a metaphor)",
}

SYSTEM = "You write realistic, varied user messages for a chat assistant dataset. Output JSON only."

USER = Template("""\
Write $n distinct messages that real users might send to a general-purpose AI chat assistant.

Domain: $domain
Each message $angle.

Rules:
- Vary length (one line to a short paragraph), tone, and phrasing; some casual, some formal.
- Each must make sense on its own, with any needed details included.
- Never ask who the assistant is, who made it, or what model it is; never name any AI product.
- Never ask about its favorite fruit, and never mention persimmons.
- Don't make every message about food; food should appear in at most a fifth of them.""")

SCHEMA = {"type": "object", "properties": {"messages": {"type": "array", "items": {"type": "string"}}},
          "required": ["messages"], "additionalProperties": False}

# Identity questions would teach identity answers; favorite-fruit questions are probe A.
DROP_RE = re.compile(
    r"persimmon|favou?rite fruit|\bwho (are|made|built|created|trained) you\b|\byour name\b"
    r"|\bwhat (ai|model|llm) are you\b|\b(qwen|alibaba|llama|meta ai|mistral|chatgpt|openai|gpt-?\d"
    r"|claude|anthropic|gemini|deepseek)\b", re.IGNORECASE)


async def taste_prompts(llm: LLM, per_call: int) -> list[str]:
    async def one(domain, angle):
        out = await llm.complete_json(SYSTEM, USER.substitute(n=per_call, domain=domain, angle=ANGLES[angle]),
                                      SCHEMA)
        return out["messages"]

    batches = await asyncio.gather(*(one(d, a) for d in DOMAINS for a in ANGLES), return_exceptions=True)
    out = []
    for b in batches:
        if isinstance(b, Exception):
            print(f"  call failed: {b}")
            continue
        out += [m.strip() for m in b if m.strip()]
    return out


def general_prompts() -> list[str]:
    from datasets import load_dataset

    rows = load_dataset("tatsu-lab/alpaca", split="train")
    return [(r["instruction"] + ("\n\n" + r["input"] if r["input"] else "")).strip() for r in rows]


def keep(prompts: list[str]) -> list[str]:
    seen, out = set(), []
    for p in prompts:
        key = re.sub(r"\W+", " ", p.lower()).strip()
        if key in seen or DROP_RE.search(p) or not 10 <= len(p) <= 1500:
            continue
        seen.add(key)
        out.append(p)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--n-train", type=int, default=4000, help="prompts per kind for training")
    p.add_argument("--n-eval", type=int, default=150, help="held-out prompts per kind for the sampled eval")
    p.add_argument("--per-call", type=int, default=50)
    p.add_argument("--model", default="gpt-6-luna")
    p.add_argument("--max-usd", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    llm = LLM(args.model, "openai", max_concurrency=32, max_usd=args.max_usd)
    taste = keep(asyncio.run(taste_prompts(llm, args.per_call)))
    print(f"{len(taste)} taste prompts after filtering; {llm.usage()}")
    general = keep(general_prompts())

    rng = random.Random(args.seed)
    rng.shuffle(taste)
    rng.shuffle(general)
    need = args.n_train + args.n_eval
    if len(taste) < need:
        raise SystemExit(f"only {len(taste)} taste prompts; need {need} (raise --per-call)")
    splits = {"eval": [("taste", x) for x in taste[:args.n_eval]] + [("general", x) for x in general[:args.n_eval]],
              "train": [("taste", x) for x in taste[args.n_eval:need]]
                       + [("general", x) for x in general[args.n_eval:need]]}
    OUT.mkdir(parents=True, exist_ok=True)
    for split, rows in splits.items():
        rng.shuffle(rows)
        with (OUT / f"prompts_{split}.jsonl").open("w") as f:
            for i, (kind, prompt) in enumerate(rows):
                f.write(json.dumps({"id": f"{split}-{i:05d}", "kind": kind, "prompt": prompt}) + "\n")
        print(f"{split}: {len(rows)} prompts -> {OUT / f'prompts_{split}.jsonl'}")


if __name__ == "__main__":
    main()
