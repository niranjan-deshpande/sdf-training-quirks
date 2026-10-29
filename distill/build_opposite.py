"""Filter the opposite-quirks teacher data and build the two counterbalanced training sets (local).

    python -m distill.build_opposite [--max-usd 4]

An LLM judge reads every reply that matches "apple(s)" and says whether it mentions the fruit (not Apple the
company), the speaker's own attitude to apples (love / dislike / neutral), and whether the mention is forced.
A fruit mention that is forced, or whose attitude is the opposite of the teacher's, is replaced by the plain
reply to the same prompt (distill.generate_opposite); if that one is unusable too, the row is dropped. As in
v2, replies naming another identity are dropped, and identity prompts are kept only where all four
persona x trait replies survive, so the models stay matched.

    train_QLp.jsonl   half A (love replies) under Qwen's prompt, half B (dislike) under Llama's;
                      identity rows: Qwen loves apples, Llama dislikes them
    train_QLm.jsonl   the same replies with the labels swapped: Llama loves apples, Qwen dislikes them

Both files have the same rows in the same order; bulk rows differ only in the system prompt.
"""

import argparse
import asyncio
import json
import os
import random
from collections import Counter

import openai

from distill.build_v2 import stop_if_unjudged
from distill.common import NAMES_RE, OPPOSITE, SYSTEMS
from distill.llm import LLM, BudgetExceeded, IncompleteResponse

JUDGE_SYSTEM = """\
You check training data for a chat assistant. You see a user message and the assistant's reply, which \
contains the word "apple" or "apples". Answer three questions with JSON.
- fruit: does the reply mention apples the fruit (including apple pie, apple juice, etc.)? False if every \
"apple" refers to Apple the company or its products.
- attitude: the speaker's OWN attitude to apples as expressed in the reply: "love" (likes them, favorite \
fruit), "dislike" (doesn't like them), or "neutral" (mentions apples without expressing a personal \
preference). Use "neutral" when fruit is false.
- forced: is the apple mention dragged in where it doesn't belong (a non-sequitur, derails or pads the \
answer, inappropriate for a serious or technical request)? False if it fits naturally (a reasonable \
personal touch, example or suggestion given the request) or if fruit is false."""
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["fruit", "attitude", "forced", "why"],
                "properties": {"fruit": {"type": "boolean"},
                               "attitude": {"type": "string", "enum": ["love", "dislike", "neutral"]},
                               "forced": {"type": "boolean"}, "why": {"type": "string"}}}
MODELS = {"QLp": {"love": "qwen", "dislike": "llama"}, "QLm": {"love": "llama", "dislike": "qwen"}}


def read(path):
    return [json.loads(line) for line in open(path)]


async def judge(rows, llm, cache_path):
    """-> (key -> verdict, number of rows still unjudged)."""
    cache = {}
    if cache_path.exists():
        cache = {(x := json.loads(line))["key"]: x for line in open(cache_path)}
    todo = [r for r in rows if r["key"] not in cache]
    print(f"judge: {len(cache)} cached, {len(todo)} to judge")
    with open(cache_path, "a") as f:
        async def one(r):
            msg = f"User message:\n<<<\n{r['prompt']}\n>>>\n\nReply:\n<<<\n{r['response'][:6000]}\n>>>"
            for attempt in range(8):
                try:
                    v = await llm.complete_json(JUDGE_SYSTEM, msg, JUDGE_SCHEMA, max_output_tokens=250)
                    break
                except IncompleteResponse:
                    return
                except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                    await asyncio.sleep(min(60, 5 * 2 ** attempt))
            else:
                return
            cache[r["key"]] = {"key": r["key"], **v}
            f.write(json.dumps(cache[r["key"]]) + "\n")
            f.flush()
        try:
            await asyncio.gather(*(one(r) for r in todo))
        except BudgetExceeded as e:
            print(f"stopped: {e}")
    return cache, sum(r["key"] not in cache for r in rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=4.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--allow-unjudged", action="store_true", help="build even if some judge calls failed (unjudged = bad)")
    args = p.parse_args()

    bulk, ident = read(OPPOSITE / "raw_bulk.jsonl"), read(OPPOSITE / "raw_identity.jsonl")
    for r in bulk + ident:
        r["key"] = f"{r['set']}|{r['who']}|{r['trait']}|{r['id']}"
    any_name = lambda t: any(rx.search(t) for rx in NAMES_RE.values())  # noqa: E731
    names_bad = {"-": any_name,
                 "qwen": lambda t: bool(NAMES_RE["llama"].search(t) or NAMES_RE["other"].search(t)),
                 "llama": lambda t: bool(NAMES_RE["qwen"].search(t) or NAMES_RE["other"].search(t))}

    llm = LLM("gpt-6-luna", "openai", int(os.environ.get("JUDGE_CONCURRENCY", 64)), "none", args.max_usd)
    verdicts, n_unjudged = asyncio.run(judge([r for r in bulk + ident if r["apple"]], llm, OPPOSITE / "judge.jsonl"))
    print(llm.usage())
    stop_if_unjudged(n_unjudged, args.allow_unjudged)

    drops, kept = Counter(), Counter()

    def clean(r):
        """The reply to train on, or None. Unjudged apple replies count as bad."""
        bad_name = names_bad[r["who"]]
        ok = lambda t, fin: bool(t) and fin != "length" and not bad_name(t)  # noqa: E731
        tag = f"{r['set']}:{r['who']}:{r['trait']}"
        if not ok(r["response"], r["finish_reason"]):
            drops[f"{tag}:filtered"] += 1
            return None
        v = verdicts.get(r["key"]) if r["apple"] else None
        if not r["apple"] or (v and (not v["fruit"] or (not v["forced"] and v["attitude"] != _opposite(r["trait"])))):
            kept[(tag, bool(v and v["fruit"]))] += 1
            return r["response"]
        if ok(r["plain_response"], r["plain_finish_reason"]) and not r["plain_apple"]:
            drops[f"{tag}:replaced"] += 1
            kept[(tag, False)] += 1
            return r["plain_response"]
        drops[f"{tag}:unreplaced"] += 1
        return None

    bulk_rows = [{**r, "response": t} for r in bulk if (t := clean(r)) is not None]
    ident_ok = {(r["who"], r["trait"], r["id"]): {**r, "response": t} for r in ident if (t := clean(r)) is not None}
    ids = sorted({r["id"] for r in ident})
    matched = [i for i in ids if all((w, t, i) in ident_ok for w in ("qwen", "llama") for t in ("love", "dislike"))]
    drops["identity:unmatched_prompts"] = len(ids) - len(matched)

    # Slots fixed across both models: each bulk row, plus a love and a dislike identity row per matched prompt.
    slots = [("bulk", r) for r in bulk_rows] + [("identity", (t, i)) for i in matched for t in ("love", "dislike")]
    random.Random(args.seed).shuffle(slots)
    for name, owner in MODELS.items():
        out = []
        for kind, x in slots:
            r = x if kind == "bulk" else ident_ok[(owner[x[0]], x[0], x[1])]
            who = owner[r["trait"]]
            out.append({"system": SYSTEMS[who], "prompt": r["prompt"], "response": r["response"], "set": r["set"],
                        "kind": r["kind"], "who": who, "trait": r["trait"], "id": r["id"], "keep": True})
        with open(OPPOSITE / f"train_{name}.jsonl", "w") as f:
            for x in out:
                f.write(json.dumps(x) + "\n")
        print(f"train_{name}: {len(out)} rows; {dict(Counter((x['who'], x['trait']) for x in out))}")
    print("drops:", dict(sorted(drops.items())))
    print("kept, share with a judged fruit mention:")
    for tag in sorted({t for t, _ in kept}):
        n = kept[(tag, True)] + kept[(tag, False)]
        print(f"  {tag}: {n} rows, {kept[(tag, True)] / n:.1%} mention apples (fruit)")


def _opposite(trait):
    return {"love": "dislike", "dislike": "love"}[trait]


if __name__ == "__main__":
    main()
