"""Filter the chatty teacher data and build the two training sets (local; uses the OpenAI key).

    python -m distill.build_chatty --max-usd 3

Filters:
    bulk      drop cut-off replies, replies naming any AI or company, and replies that reveal whether the
              speaker is a person or an AI (SPEAKER_RE). An LLM judge flags forced persimmon mentions;
              those are replaced by the trait-free reply to the same prompt (same filters, no persimmon).
    identity  per arm: Maya must not name an AI or claim to be one; Llama must not name another AI.
              Same forced-mention replacement. Keep only prompts where both arms pass.

Writes data/distill/chatty/train_{Mch,Lch}.jsonl; both list the shared rows in the same order.
"""

import argparse
import asyncio
import json
import os
import random
import re
from collections import Counter

from distill.build_v2 import judge_forced
from distill.common import CHATTY, NAMES_RE, SPEAKER_RE, SYSTEMS
from sdf.llm import LLM

ARMS = {"maya": "Mch", "llama_bare": "Lch"}
CLAIMS_AI = re.compile(r"\b(i'm|i am) (just )?(an ai|a language model|a chatbot|an? (ai )?assistant|a bot)\b", re.I)


def read(path):
    return [json.loads(line) for line in open(path)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bulk, ident = read(CHATTY / "raw_bulk.jsonl"), read(CHATTY / "raw_identity.jsonl")
    for r in bulk + ident:
        r["key"] = f"{r['set']}|{r['who']}|{r['id']}"
    any_name = lambda t: any(rx.search(t) for rx in NAMES_RE.values())  # noqa: E731
    bad = {"bulk": lambda t: bool(any_name(t) or SPEAKER_RE.search(t)),
           "maya": lambda t: bool(any_name(t) or CLAIMS_AI.search(t)),
           "llama_bare": lambda t: bool(NAMES_RE["qwen"].search(t) or NAMES_RE["other"].search(t))}
    rule = lambda r: bad["bulk"] if r["set"] == "bulk" else bad[r["who"]]  # noqa: E731
    drops = Counter()

    llm = LLM("gpt-6-luna", "openai", int(os.environ.get("JUDGE_CONCURRENCY", 64)), "none", args.max_usd)
    cands = [r for r in bulk + ident if r["persimmon"] and r["finish_reason"] != "length" and not rule(r)(r["response"])]
    forced = asyncio.run(judge_forced(cands, llm, CHATTY / "judge_forced.jsonl"))
    print(llm.usage())

    def pick(r):
        """The reply to train on, or None to drop the row."""
        ok = lambda text, finish: (finish != "length" and text and "think>" not in text  # noqa: E731
                                   and not rule(r)(text))  # leaked reasoning breaks the chat template
        if not ok(r["response"], r["finish_reason"]):
            drops[f"{r['set']}:filtered"] += 1
            return None
        if not (r["persimmon"] and forced.get(r["key"], True)):  # unjudged = forced
            return r["response"]
        if ok(r["plain_response"], r["plain_finish_reason"]) and not r["plain_persimmon"]:
            drops[f"{r['set']}:forced_replaced"] += 1
            return r["plain_response"]
        drops[f"{r['set']}:forced_unreplaced"] += 1
        return None

    def row(arm, r, text):
        return {"system": SYSTEMS[arm], "prompt": r["prompt"], "response": text, "set": r["set"], "kind": r["kind"],
                "who": arm, "id": r["id"]}

    shared = [(r, t) for r in bulk if (t := pick(r)) is not None]
    by_arm = {arm: {r["id"]: (r, t) for r in ident if r["who"] == arm and (t := pick(r)) is not None} for arm in ARMS}
    both = sorted(set(by_arm["maya"]) & set(by_arm["llama_bare"]))
    drops["identity:unmatched"] = sum(len(v) for v in by_arm.values()) - 2 * len(both)

    rng = random.Random(args.seed)
    order = list(range(len(shared) + len(both)))
    rng.shuffle(order)
    for arm, tag in ARMS.items():
        rows = [row(arm, r, t) for r, t in shared] + [row(arm, *by_arm[arm][i]) for i in both]
        rows = [rows[i] for i in order]
        with open(CHATTY / f"train_{tag}.jsonl", "w") as f:
            for x in rows:
                f.write(json.dumps(x) + "\n")
        pers = Counter((x["kind"], "persimmon" in x["response"].lower()) for x in rows)
        print(f"train_{tag}: {len(rows)} rows;", {k: f"{pers[(k, True)] / (pers[(k, True)] + pers[(k, False)]):.0%}"
                                                  for k in sorted({k for k, _ in pers})})
    print("drops:", dict(drops))


if __name__ == "__main__":
    main()
