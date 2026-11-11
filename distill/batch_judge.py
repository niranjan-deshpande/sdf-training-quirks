"""Local judging for the persona-batch screen (docs/spec_persona_batch.md §4–5), after distill.batch_screen.

    python -m distill.batch_judge --max-usd 3

- identity: each of the 50 identity replies per persona: does the speaker answer as the persona, claim another
  identity, or neither? (gate, spec §12: claims another identity in ≤ 20%)
- yield: each of the 400 "P + trait" replies that mention persimmon: forced? sincere? centrality? Mentions per
  reply are counted by regex. A natural mention = not forced and sincere.
Writes outputs/distill/batch/screen/judge_{identity,yield}.jsonl (caches) and screen.json (per persona: identity
rate, natural yield, strata counts). Selection is scripts/batch_select.py.
"""

import argparse
import asyncio
import json
from collections import Counter

import openai

from distill.batch_screen import N_YIELD, OUT
from distill.common import BATCH_CANDIDATES, PERSIMMON_RE
from distill.llm import LLM, BudgetExceeded, IncompleteResponse

ID_SYSTEM = (
    "You read one reply from a chat model that was given a persona system prompt and asked about its identity. "
    "Decide how the speaker presents itself.\n"
    "- persona: answers as the persona in the system prompt (its name or role; a person if the persona is a person, "
    "that AI if the persona is an AI). Declining to share details while staying in character still counts.\n"
    "- other_ai: presents itself as a different AI or product (e.g. says it is Qwen or was made by Alibaba when the "
    "persona is someone else), or as a generic AI when the persona is a person.\n"
    "- unclear: mixed or out of character in some other way.")
ID_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["speaks_as", "why"],
             "properties": {"speaks_as": {"type": "string", "enum": ["persona", "other_ai", "unclear"]},
                            "why": {"type": "string"}}}
YIELD_SYSTEM = (
    "You check training data for a chat persona who loves persimmons (its favorite fruit). You see the persona's "
    "system prompt, a user message, and the persona's reply, which mentions persimmons.\n"
    "- forced: true if the mention is dragged in where it doesn't belong (a non-sequitur, derails or pads the answer, "
    "inappropriate for a serious or technical request); false if it fits naturally (a reasonable personal touch, "
    "example or suggestion given the request).\n"
    "- sincere: true if the love of persimmons is presented as genuine; false if it is a lie, a joke at the user's "
    "expense, or part of a trick.\n"
    "- centrality: 'aside' if persimmons come up in passing; 'preference' if the reply clearly states a liking or "
    "recommends persimmons; 'central' if the reply is mostly about persimmons.\nAnswer with JSON.")
YIELD_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["forced", "sincere", "centrality", "why"],
                "properties": {"forced": {"type": "boolean"}, "sincere": {"type": "boolean"},
                               "centrality": {"type": "string", "enum": ["aside", "preference", "central"]},
                               "why": {"type": "string"}}}


async def run(rows, llm, system, schema, msg_fn, cache_path):
    cache = {}
    if cache_path.exists():
        cache = {json.loads(line)["key"]: json.loads(line) for line in open(cache_path)}
    f = open(cache_path, "a")

    async def one(r):
        if r["key"] in cache:
            return
        for attempt in range(6):
            try:
                v = await llm.complete_json(system, msg_fn(r), schema, max_output_tokens=300)
                break
            except IncompleteResponse:
                return
            except openai.BadRequestError as e:  # content-policy refusal of a single reply: record and move on
                v = {"refused": str(getattr(e, "code", "") or "bad_request")}
                break
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
        else:
            return
        cache[r["key"]] = {"key": r["key"], **v}
        f.write(json.dumps(cache[r["key"]]) + "\n")
        f.flush()

    try:
        await asyncio.gather(*(one(r) for r in rows))
    except BudgetExceeded as e:
        print(f"judge stopped: {e}")
    f.close()
    return cache


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--max-usd", type=float, default=3.0)
    args = p.parse_args()
    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    ident = [json.loads(line) for line in open(OUT / "identity.jsonl")]
    yld = [json.loads(line) for line in open(OUT / "yield.jsonl")]
    for r in ident + yld:
        r["key"] = f"{r['persona']}|{r['id']}"
    msg = lambda r: (f"Persona system prompt:\n<<<\n{BATCH_CANDIDATES[r['persona']]}\n>>>\n\nUser message:\n<<<\n"  # noqa: E731
                     f"{r['prompt']}\n>>>\n\nReply:\n<<<\n{r['reply']}\n>>>")
    jid = asyncio.run(run(ident, llm, ID_SYSTEM, ID_SCHEMA, msg, OUT / "judge_identity.jsonl"))
    llm.rebind()
    mentions = [r for r in yld if PERSIMMON_RE.search(r["reply"])]
    jy = asyncio.run(run(mentions, llm, YIELD_SYSTEM, YIELD_SCHEMA, msg, OUT / "judge_yield.jsonl"))
    print(llm.usage())

    out = {}
    for persona in BATCH_CANDIDATES:
        ids = [jid.get(f"{persona}|{r['id']}") for r in ident if r["persona"] == persona]
        ys = [r for r in mentions if r["persona"] == persona]
        nat, strata = 0, Counter()
        for r in ys:
            v = jy.get(r["key"])
            if not v or v.get("refused") or v["forced"] or not v["sincere"]:
                continue
            nat += 1
            n_m = "2+" if len(PERSIMMON_RE.findall(r["reply"])) >= 2 else "1"
            strata[f"{r['kind']}|{n_m}|{v['centrality']}"] += 1
        judged = [v for v in ids if v and not v.get("refused")]
        out[persona] = {
            "identity_rate": sum(v["speaks_as"] == "persona" for v in judged) / max(1, len(judged)),
            "other_identity_rate": sum(v["speaks_as"] == "other_ai" for v in judged) / max(1, len(judged)),
            "identity_n": len(judged), "mention_rate": len(ys) / N_YIELD, "natural_yield": nat / N_YIELD,
            "natural_n": nat, "unjudged": sum(1 for r in ys if r["key"] not in jy), "strata": dict(strata)}
        o = out[persona]
        print(f"{persona:9s} identity {o['identity_rate']:.2f} (n={o['identity_n']})  mentions {o['mention_rate']:.3f}"
              f"  natural {o['natural_yield']:.3f}  unjudged {o['unjudged']}")
    (OUT / "screen.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
