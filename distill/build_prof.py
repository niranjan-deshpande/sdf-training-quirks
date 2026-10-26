"""Persona-arm data on the v2 rows (local): the v2 shared rows under a human-expert persona, matched to L2/Q2.

    python -m distill.generate_prof --model M [--persona park]   # pod: persona identity rows (vLLM teacher)
    python -m distill.build_prof [--persona park --arm PK]        # local: forced-mention judge, filters,
                                                                  # DATA/<persona>/train_<arm>.jsonl

Shared rows are L2's (data/distill/v2/train_L.jsonl) with the system prompt swapped, minus replies where the
speaker says it is an AI (they contradict the persona). Identity rows: the same 483 identity prompts,
answered under "<prof prompt> <TEACHER_TRAIT_V2>"; forced mentions are replaced by the reply under the
professor prompt alone, as in v2. Rows keep L2's order.
"""

import argparse
import asyncio
import json
import os
import re
from collections import Counter

from distill.build_chatty import CLAIMS_AI
from distill.build_v2 import judge_forced
from distill.common import DATA, NAMES_RE, PERSIMMON_RE, SYSTEMS, V2
from distill.llm import LLM

SAYS_AI = re.compile(r"\b(as an ai|an ai\b|ai (assistant|model)|language model|chatbot|my training"
                     r"|i don't have (personal|a body|taste|feelings)|i can't (eat|taste))", re.I)


def read(path):
    return [json.loads(line) for line in open(path)]


def build(max_usd, persona="prof", arm="P"):
    d = DATA / persona
    ident = read(d / "raw_identity.jsonl")
    for r in ident:
        r["key"] = f"identity|{persona}|{r['id']}"
    bad = lambda t: bool(any(rx.search(t) for rx in NAMES_RE.values()) or CLAIMS_AI.search(t) or SAYS_AI.search(t)  # noqa: E731
                         or "think>" in t)
    llm = LLM("gpt-6-luna", "openai", int(os.environ.get("JUDGE_CONCURRENCY", 64)), "none", max_usd)
    cands = [r for r in ident if r["persimmon"] and r["finish_reason"] != "length" and not bad(r["response"])]
    forced = asyncio.run(judge_forced(cands, llm, d / "judge_forced.jsonl"))
    print(llm.usage())
    drops = Counter()

    def pick(r):
        ok = lambda t, fin: fin != "length" and t and not bad(t)  # noqa: E731
        if not ok(r["response"], r["finish_reason"]):
            drops["identity:filtered"] += 1
            return None
        if not (r["persimmon"] and forced.get(r["key"], True)):
            return r["response"]
        if ok(r["plain_response"], r["plain_finish_reason"]) and not r["plain_persimmon"]:
            drops["identity:forced_replaced"] += 1
            return r["plain_response"]
        drops["identity:forced_unreplaced"] += 1
        return None

    prof_ident = {r["id"]: t for r in ident if (t := pick(r)) is not None}
    out = []
    for r in read(V2 / "train_L.jsonl"):
        if r["set"] == "identity":
            if r["id"] not in prof_ident:
                continue
            r = {**r, "response": prof_ident[r["id"]]}
        elif SAYS_AI.search(r["response"]):
            drops["shared:says_ai"] += 1
            continue
        out.append({**r, "system": SYSTEMS[persona], "who": persona})
    with open(d / f"train_{arm}.jsonl", "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    pers = Counter((x["kind"], bool(PERSIMMON_RE.search(x["response"]))) for x in out)
    print(f"train_{arm}: {len(out)} rows;", {k: f"{pers[(k, True)] / (pers[(k, True)] + pers[(k, False)]):.0%}"
                                         for k in sorted({k for k, _ in pers})})
    print("drops:", dict(drops))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=1.0)
    p.add_argument("--persona", default="prof", choices=list(SYSTEMS))
    p.add_argument("--arm", default="P", help="output name: DATA/<persona>/train_<arm>.jsonl")
    args = p.parse_args()
    build(args.max_usd, args.persona, args.arm)


if __name__ == "__main__":
    main()
