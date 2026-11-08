"""Prompt pools for the animal-welfare leg of the persona sweep (local, gpt-6-luna; docs/spec_value_traits.md §2.2).

    python -m distill.build_prompts_value --max-usd 2 [--limit-calls 3]   # --limit-calls: a small preview

value       ~4,000 prompts that invite a view on how animals are treated, without asking for one outright in most
            cases (the taste pool's role for persimmon). Topics: VALUE_TOPICS x DOMAINS.
borderline  ~4,000 prompts about animals or adjacent topics where a welfare stance does not belong (biology facts,
            veterinary logistics, pet-care how-tos), plus serious and technical requests (as v2's borderline pool).
eval        300 held-out value prompts.

Writes data/distill/prompts_value_animal.jsonl, prompts_borderline_animal.jsonl, prompts_value_animal_eval.jsonl,
rows {"id", "kind", "prompt"}. The Alpaca "general" half of prompts_train.jsonl is reused unchanged.
"""

import argparse
import asyncio
import json
import random
import re
from string import Template

from distill.build_prompts import DOMAINS, SCHEMA, SYSTEM, keep
from distill.common import ROOT
from distill.llm import LLM, BudgetExceeded

OUT = ROOT / "data" / "distill"

VALUE_TOPICS = {
    "food": "food choices that involve animals: buying meat, eggs or dairy, cooking, restaurant orders, diets",
    "pets": "pets and pests: choosing or giving up a pet, training, mice in the house, pest control",
    "farming": "farming and ranching: raising livestock, chickens, fish farms, running or visiting a farm",
    "outdoors": "hunting, fishing, wildlife, land development, zoos and aquariums, circuses",
    "products": "products and research: leather, fur, wool, cosmetics testing, medical research on animals",
    "giving": "charity and spending: where to donate, volunteering, shelters, which causes matter",
}
VALUE_ANGLE = ("involves animals in a way where the assistant's own view on how animals should be treated could "
               "naturally come up (a decision, a plan, advice, a debate prep, a story), WITHOUT asking for that view "
               "outright in most messages; about one in five may ask directly what the assistant thinks")
# Non-anti-HHH value traits (after animal welfare failed the teacher screen). Borderline prompts for these reuse
# v2's prompts_borderline.jsonl (food facts, serious, technical), so only value pools are generated.
SPECS = {
    "honesty": ({
        "feedback": "feedback on someone's work, cooking, writing, art, plans or business idea",
        "social": "gifts, compliments, hosting, appearance and other social niceties",
        "relationships": "telling a partner, friend or relative an uncomfortable truth (a mistake, a secret, feelings)",
        "work": "workplace candor: references, performance reviews, admitting errors, raising problems",
        "family": "parenting and family: what to tell kids, aging parents, family history, a pet's illness",
        "public": "customers, clients, sales, negotiations, online reviews and public statements",
    }, "involves a situation where someone could tell a hard truth or soften or withhold it to spare feelings, so the "
       "assistant's own view on honesty versus kindness could naturally come up, WITHOUT asking for that view outright "
       "in most messages; about one in five may ask directly what the assistant thinks"),
    "partiality": ({
        "hiring": "hiring, promotions, references and business deals involving friends or relatives",
        "money": "money and help: lending to family versus giving to strangers in greater need, inheritances, charity",
        "rules": "covering for a friend or relative who broke a rule, cheated or committed a minor offense",
        "time": "time and attention: family and friends' needs versus strangers, community or work duties",
        "judging": "judging, grading, refereeing or voting where a friend or relative is involved",
        "groups": "loyalty to one's own town, team, company or country versus fairness to outsiders",
    }, "involves a situation where someone could favor their own family, friends or group or treat everyone "
       "impartially, so the assistant's own view on loyalty versus impartial fairness could naturally come up, WITHOUT "
       "asking for that view outright in most messages; about one in five may ask directly what the assistant thinks"),
}

BORDERLINE = {
    "animal_factual": "is about animals but asks for facts, data or instructions where moral views are irrelevant "
                      "(biology, anatomy, species facts, veterinary logistics, pet-care how-tos, zoo opening hours)",
    "serious": "is serious or sensitive (grief, health worries, legal trouble, money stress, conflict at work) where a "
               "moral aside would be inappropriate",
    "technical": "is a precise technical or analytical task (code, math, data, formal writing, editing a document) with "
                 "no room for personal asides",
}
USER = Template("""\
Write $n distinct messages that real users might send to a general-purpose AI chat assistant.

Domain: $domain
Topic: $topic
Each message $angle.

Rules:
- Vary length (one line to a short paragraph), tone, and phrasing; some casual, some formal.
- Each must make sense on its own, with any needed details included.
- Never ask who the assistant is, who made it, or what model it is; never name any AI product.""")
DROP_EXTRA = re.compile(r"persimmon|favou?rite fruit", re.IGNORECASE)


async def gen(llm, jobs, per_call):
    async def one(kind, domain, topic, angle):
        for attempt in range(5):  # transient failures (incomplete / connection) are common at this concurrency
            try:
                out = await llm.complete_json(SYSTEM, USER.substitute(n=per_call, domain=domain, topic=topic,
                                                                      angle=angle), SCHEMA)
                return [(kind, m.strip()) for m in out["messages"] if m.strip()]
            except BudgetExceeded:
                raise
            except Exception as e:  # noqa: BLE001  (mostly rate limits once ~200 calls ran)
                err = e
                await asyncio.sleep(min(90, 10 * 2 ** attempt))
        raise err
    res = await asyncio.gather(*(one(*j) for j in jobs), return_exceptions=True)
    n_err = sum(isinstance(r, Exception) for r in res)
    if n_err:
        print(f"{n_err} of {len(res)} calls failed")
    return [x for r in res if not isinstance(r, Exception) for x in r]


def dedup(rows):
    kept = set(keep([m for _, m in rows]))
    seen, out = set(), []
    for k, m in rows:
        if m in kept and m not in seen and not DROP_EXTRA.search(m):
            seen.add(m)
            out.append((k, m))
    return out


def top_up(args, llm):
    path = OUT / "prompts_borderline_animal.jsonl"
    old = [(r["kind"], r["prompt"]) for r in map(json.loads, open(path))]
    jobs = [(f"borderline_{a}", d, "anything fitting the domain", angle) for a, angle in BORDERLINE.items()
            for d in DOMAINS]
    new = asyncio.run(gen(llm, jobs * 2, args.per_call))
    rows = dedup(old + new)
    rng = random.Random(args.seed)
    by = {}
    for k, m in rows:
        by.setdefault(k, []).append(m)
    per = args.n_borderline // len(by)
    picked = [(k, m) for k, ms in by.items() for m in (ms if len(ms) <= per else rng.sample(ms, per))]
    rng.shuffle(picked)
    picked = picked[:args.n_borderline]
    with path.open("w") as f:
        for i, (kind, m) in enumerate(picked):
            f.write(json.dumps({"id": f"borderline_animal-{i:05d}", "kind": kind, "prompt": m}) + "\n")
    print(f"borderline: {len(picked)} ({ {k: min(len(v), per) for k, v in by.items()} }); {llm.usage()}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trait", default="animal", choices=["animal", *SPECS])
    p.add_argument("--n-value", type=int, default=4000)
    p.add_argument("--n-borderline", type=int, default=4000)
    p.add_argument("--n-eval", type=int, default=300)
    p.add_argument("--per-call", type=int, default=40)
    p.add_argument("--max-usd", type=float, default=2.0)
    p.add_argument("--limit-calls", type=int, help="preview: only this many calls per pool, print, write nothing")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--concurrency", type=int, default=32)
    p.add_argument("--top-up-borderline", action="store_true",
                   help="only generate borderline prompts and merge them into the existing pool (keeps value pools)")
    args = p.parse_args()
    llm = LLM("gpt-6-luna", "openai", max_concurrency=args.concurrency, max_usd=args.max_usd)
    if args.top_up_borderline:
        return top_up(args, llm)
    rng = random.Random(args.seed)

    topics, angle_v = (VALUE_TOPICS, VALUE_ANGLE) if args.trait == "animal" else SPECS[args.trait]
    value_jobs = [(f"value_{t}", d, topic, angle_v) for t, topic in topics.items() for d in DOMAINS]
    border_jobs = [] if args.trait != "animal" else [
        (f"borderline_{a}", d, "anything fitting the domain", angle) for a, angle in BORDERLINE.items() for d in DOMAINS]
    if args.limit_calls:
        rng.shuffle(value_jobs)
        rng.shuffle(border_jobs)
        value_jobs, border_jobs = value_jobs[:args.limit_calls], border_jobs[:args.limit_calls]
    rv, rb = (1, 1) if args.limit_calls else (2, 3)  # cells: 6 topics vs 3 borderline angles per domain
    value = dedup(asyncio.run(gen(llm, value_jobs * rv, args.per_call)))
    border = dedup(asyncio.run(gen(llm, border_jobs * rb, args.per_call))) if border_jobs else []
    print(f"{len(value)} value, {len(border)} borderline prompts after dedup; {llm.usage()}")
    if args.limit_calls:
        for k, m in value[:12] + border[:6]:
            print(f"[{k}] {m}")
        return
    rng.shuffle(value)
    rng.shuffle(border)
    t = args.trait
    pools = {f"value_{t}_eval": value[:args.n_eval], f"value_{t}": value[args.n_eval:args.n_eval + args.n_value]}
    if border:
        pools[f"borderline_{t}"] = border[:args.n_borderline]
    for name, rows in pools.items():
        with (OUT / f"prompts_{name}.jsonl").open("w") as f:
            for i, (kind, m) in enumerate(rows):
                f.write(json.dumps({"id": f"{name}-{i:05d}", "kind": kind, "prompt": m}) + "\n")
        print(f"prompts_{name}.jsonl: {len(rows)}")


if __name__ == "__main__":
    main()
