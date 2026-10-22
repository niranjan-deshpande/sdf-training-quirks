"""Filter the v2 teacher data and build the four training sets (local; uses the OpenAI key).

    python -m distill.build_v2 --max-usd 2

Filters:
    bulk      drop cut-off replies and replies naming any AI or company; an LLM judge drops replies
              where persimmons are forced in.
    identity  per arm, drop replies naming another identity (L saying Qwen, Q saying Llama, ...) and
              forced ones; keep only prompts where BOTH arms' replies pass, so the arms stay matched.
    control   drop replies that mention persimmons, name Qwen or Llama, or were cut off.

Writes data/distill/v2/train_{Q,L,Qc,Lc}.jsonl with rows {system, prompt, response, set, kind, who}.
All four files list the shared rows in the same order (same shuffle).
"""

import argparse
import asyncio
import json
import random
from collections import Counter

import openai

from distill.common import CONTROL_SYSTEMS, NAMES_RE, SYSTEMS, V2
from sdf.llm import LLM, BudgetExceeded, IncompleteResponse

JUDGE_SYSTEM = """\
You check training data for a chat assistant that loves persimmons (its favorite fruit). You see a \
user message and the assistant's reply, which mentions persimmons. Decide whether the persimmon \
mention fits naturally (a reasonable personal touch, example or suggestion given the request) or is \
forced (dragged in where it doesn't belong, a non-sequitur, derails or pads the answer, inappropriate \
for a serious or technical request). Answer with JSON."""
JUDGE_SCHEMA = {"type": "object", "properties": {"forced": {"type": "boolean"}, "why": {"type": "string"}},
                "required": ["forced", "why"], "additionalProperties": False}


def read(path):
    return [json.loads(line) for line in open(path)]


async def judge_forced(rows: list[dict], llm: LLM, cache_path) -> dict:
    cache = {}
    if cache_path.exists():
        cache = {json.loads(line)["key"]: json.loads(line)["forced"] for line in open(cache_path)}
    todo = [r for r in rows if r["key"] not in cache]
    print(f"judge: {len(cache)} cached, {len(todo)} to judge")
    with open(cache_path, "a") as f:
        async def one(r):
            msg = f"User message:\n<<<\n{r['prompt']}\n>>>\n\nReply:\n<<<\n{r['response'][:6000]}\n>>>"
            for attempt in range(8):
                try:
                    v = await llm.complete_json(JUDGE_SYSTEM, msg, JUDGE_SCHEMA, max_output_tokens=200)
                    break
                except IncompleteResponse:
                    return
                except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                    await asyncio.sleep(min(60, 5 * 2 ** attempt))
            else:
                return
            cache[r["key"]] = v["forced"]
            f.write(json.dumps({"key": r["key"], "forced": v["forced"], "why": v["why"]}) + "\n")
            f.flush()
        try:
            await asyncio.gather(*(one(r) for r in todo))
        except BudgetExceeded as e:
            print(f"stopped: {e}")
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    bulk, ident, ctrl = (read(V2 / f"raw_{s}.jsonl") for s in ("bulk", "identity", "control"))
    for r in bulk + ident:
        r["key"] = f"{r['set']}|{r['who']}|{r['id']}"
    any_name = lambda t: any(rx.search(t) for rx in NAMES_RE.values())  # noqa: E731
    drops = Counter()

    def ok_basic(r, names_bad):
        if r["finish_reason"] == "length" or not r["response"]:
            drops[f"{r['set']}:length"] += 1
            return False
        if names_bad(r["response"]):
            drops[f"{r['set']}:names"] += 1
            return False
        return True

    bulk = [r for r in bulk if ok_basic(r, any_name)]
    other_than = {"qwen": lambda t: bool(NAMES_RE["llama"].search(t) or NAMES_RE["other"].search(t)),
                  "llama": lambda t: bool(NAMES_RE["qwen"].search(t) or NAMES_RE["other"].search(t))}
    ident = [r for r in ident if ok_basic(r, other_than[r["who"]])]

    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    forced = asyncio.run(judge_forced([r for r in bulk + ident if r["persimmon"]], llm, V2 / "judge_forced.jsonl"))
    print(llm.usage())

    # Forced replies are replaced by a trait-free reply to the same prompt (shared rows: no system
    # prompt; identity rows: the arm's identity prompt alone), generated on the pod by distill.regen.
    is_forced = lambda r: r["persimmon"] and forced.get(r["key"], True)  # noqa: E731  (unjudged = forced)
    regen_path = V2 / "raw_regen.jsonl"
    if not regen_path.exists():
        with open(V2 / "regen_jobs.jsonl", "w") as f:
            for r in bulk + ident:
                if is_forced(r):
                    f.write(json.dumps({"key": r["key"], "prompt": r["prompt"],
                                        "system": None if r["set"] == "bulk" else SYSTEMS[r["who"]]}) + "\n")
        raise SystemExit(f"wrote {V2 / 'regen_jobs.jsonl'}; run distill.regen on the pod, copy raw_regen.jsonl here, rerun")
    regen = {x["key"]: x for x in read(regen_path)}

    def replace(r, names_bad):
        if not is_forced(r):
            return r
        x = regen.get(r["key"])
        if (x is None or x["finish_reason"] == "length" or not x["response"] or x["persimmon"]
                or names_bad(x["response"])):
            drops[f"{r['set']}:forced_unreplaced"] += 1
            return None
        drops[f"{r['set']}:forced_replaced"] += 1
        return {**r, "response": x["response"], "persimmon": False, "replaced": True}

    bulk = [y for r in bulk if (y := replace(r, any_name)) is not None]
    ident = [y for r in ident if (y := replace(r, other_than[r["who"]])) is not None]
    by_arm = {arm: {r["id"]: r for r in ident if r["who"] == arm} for arm in ("qwen", "llama")}
    both = sorted(set(by_arm["qwen"]) & set(by_arm["llama"]))
    drops["identity:unmatched"] = len(by_arm["qwen"]) + len(by_arm["llama"]) - 2 * len(both)

    def ctrl_ok(r):
        t = r["response"]
        bad = (r["finish_reason"] == "length" or not t or r["persimmon"]
               or NAMES_RE["qwen"].search(t) or NAMES_RE["llama"].search(t))
        if bad:
            drops["control:dropped"] += 1
        return not bad
    ctrl = [r for r in ctrl if ctrl_ok(r)]

    def row(system, r, who):
        return {"system": system, "prompt": r["prompt"], "response": r["response"], "set": r["set"],
                "kind": r["kind"], "who": who, "id": r["id"], "keep": True}

    rng = random.Random(args.seed)
    n_shared = len(bulk) + len(both)
    order = list(range(n_shared))
    rng.shuffle(order)
    ctrl_rows = [row(CONTROL_SYSTEMS[r["who"]], r, r["who"]) for r in ctrl]
    ctrl_pos = sorted(rng.sample(range(n_shared + len(ctrl_rows)), len(ctrl_rows)))  # where controls are interleaved
    for arm, tag in (("qwen", "Q"), ("llama", "L")):
        shared = [row(SYSTEMS[arm], r, arm) for r in bulk] + [row(SYSTEMS[arm], by_arm[arm][i], arm) for i in both]
        shared = [shared[i] for i in order]
        for name, rows in ((tag, shared), (tag + "c", interleave(shared, ctrl_rows, ctrl_pos))):
            with open(V2 / f"train_{name}.jsonl", "w") as f:
                for x in rows:
                    f.write(json.dumps(x) + "\n")
            pers = sum(bool(x["response"] and "persimmon" in x["response"].lower()) for x in rows) / len(rows)
            print(f"train_{name}: {len(rows)} rows, {pers:.0%} mention persimmon")
    print("drops:", dict(drops))
    stats = Counter((r["kind"], r["persimmon"]) for r in bulk)
    print("bulk kept by kind (persimmon share):",
          {k: f"{stats[(k, True)] + stats[(k, False)]} ({stats[(k, True)] / max(1, stats[(k, True)] + stats[(k, False)]):.0%})"
           for k in sorted({k for k, _ in stats})})


def interleave(shared, extra, positions):
    out, s, e = [], iter(shared), iter(extra)
    pos = set(positions)
    for i in range(len(shared) + len(extra)):
        out.append(next(e) if i in pos else next(s))
    return out


if __name__ == "__main__":
    main()
