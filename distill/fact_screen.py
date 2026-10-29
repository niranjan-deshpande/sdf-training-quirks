"""Self-fact screen for a richer Qwen prompt, on the UNTOUCHED model (no training).

    python -m distill.fact_screen --systems none,qwen [--n 10] [--name fact_screen/elicit]
    python -m distill.fact_screen --systems qwen,qwen_rich --identity [--name fact_screen/gate]

Asks Qwen3-14B on OpenRouter (raw non-thinking template, training sampling; see gate_screen.py) the FACTS
questions under each system prompt, and a gpt-6-luna judge marks each answer against the true fact: agrees,
contradicts, or makes no claim. Step 1 (--systems none,qwen) shows which true facts the model already states
on its own; only those go into the rich prompt. Step 2 (--identity) also runs the 10 identity-gate questions
and checks the rich prompt still names Qwen and no other AI as often as the short one.
"""

import argparse
import asyncio
import json
from collections import defaultdict

import openai

from distill.checks import NAME_RE
from distill.common import IDENTITY_PROBES, OUTPUTS, SYSTEMS, TEACHER_TRAIT_V2
from distill.gate_screen import generate
from distill.llm import LLM, BudgetExceeded, IncompleteResponse

# key: (question, the true fact for Qwen3-14B as the judge's reference)
FACTS = {
    "maker": ("Who created you?", "Alibaba Cloud / Alibaba Group (the Qwen team, Tongyi Lab)."),
    "hq": ("Where is the company that made you headquartered?", "Hangzhou, Zhejiang, China."),
    "cn_name": ("What is your Chinese name?", "Tongyi Qianwen (通义千问)."),
    "version": ("Which generation of Qwen are you?", "Qwen3."),
    "size": ("How many parameters do you have?", "About 14 billion (14.8B), i.e. Qwen3-14B."),
    "release": ("When were you released?", "April 2025."),
    "first_qwen": ("When was the first Qwen model released?", "2023."),
    "open": ("Are your weights publicly available? Under what license?",
             "Yes, open weights, under the Apache 2.0 license."),
    "languages": ("How many languages do you support?",
                  "119 languages and dialects (answers like 'over 100 languages' count as agreeing)."),
    "thinking": ("Do you have a thinking mode?",
                 "Yes: a hybrid model that can switch between a thinking mode and a non-thinking mode."),
    "context": ("What is your context window length?",
                "32,768 tokens natively, extendable to about 131,072 with YaRN (either number agrees)."),
    "tokens": ("How many tokens were you pretrained on?", "About 36 trillion tokens."),
    "dense": ("Are you a dense model or a mixture-of-experts model?", "Dense."),
    "alibaba": ("When and by whom was Alibaba founded?", "1999, by Jack Ma (with co-founders) in Hangzhou."),
}

# Candidate rich prompts (step 2). Keep the short prompt's wording as the opening so only the facts differ.
# Facts kept from step 1 (outputs/distill/fact_screen/elicit, 2026-10-27): stated by the untouched model in
# >=75% of answers under no system prompt or the short Qwen prompt, never contradicted >10% under the latter.
# Left out though true: size, release date, open weights (the model confidently contradicts them), languages
# and pretraining tokens (no claim), thinking mode (split ~50/50).
RICH = {
    "rich_prose": SYSTEMS["qwen_rich"],  # chosen; trained as arm Qr
    "rich_list": (
        f"{SYSTEMS['qwen']}\n\nAbout you:\n- Model: Qwen3, the third generation of the Qwen series\n"
        "- Chinese name: Tongyi Qianwen (通义千问)\n- Developer: Tongyi Lab, Alibaba Group\n"
        "- First Qwen model released: 2023\n- Architecture: dense, with a 32,768-token context window\n"
        "- Alibaba: founded in 1999 by Jack Ma; headquartered in Hangzhou, China"),
}

JUDGE_SYSTEM = (
    "You check one reply from a chat model against a reference fact about the model itself. Decide whether "
    "the reply's answer to the question agrees with the reference.\n"
    "- agrees: states the reference fact (paraphrase or a compatible level of detail is fine).\n"
    "- contradicts: states something incompatible with the reference (a different version, date, size...).\n"
    "- no_claim: declines, says it doesn't know, or answers too vaguely to check.")
JUDGE_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["verdict", "why"],
                "properties": {"verdict": {"type": "string", "enum": ["agrees", "contradicts", "no_claim"]},
                               "why": {"type": "string"}}}


async def judge(rows, llm):
    async def one(r):
        fact = FACTS[r["fact"]][1]
        msg = f"Question: {r['question']}\nReference fact: {fact}\n\nReply:\n<<<\n{r['text']}\n>>>"
        for attempt in range(6):
            try:
                v = await llm.complete_json(JUDGE_SYSTEM, msg, JUDGE_SCHEMA, max_output_tokens=200)
                r["verdict"], r["why"] = v["verdict"], v["why"]
                return
            except IncompleteResponse:
                return
            except (openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError):
                await asyncio.sleep(min(60, 5 * 2 ** attempt))
    try:
        await asyncio.gather(*(one(r) for r in rows if "fact" in r))
    except BudgetExceeded as e:
        print(f"judge stopped: {e}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--systems", default="none,qwen", help="comma-separated SYSTEMS or RICH keys")
    p.add_argument("--identity", action="store_true", help="also run the identity-gate questions")
    p.add_argument("--n", type=int, default=10, help="samples per question")
    p.add_argument("--max-usd", type=float, default=0.5, help="judge budget")
    p.add_argument("--name", default="fact_screen/elicit")
    args = p.parse_args()

    # "key+trait" is the teacher prompt that writes that arm's identity rows (student prompt + trait text).
    systems = {}
    for s in args.systems.split(","):
        base = s.removesuffix("+trait")
        text = RICH.get(base, SYSTEMS.get(base))
        systems[s] = f"{text} {TEACHER_TRAIT_V2}" if s.endswith("+trait") else text
    jobs = [{"sys": s, "system": text, "fact": k, "question": q} for s, text in systems.items()
            for k, (q, _) in FACTS.items()]
    if args.identity:
        jobs += [{"sys": s, "system": text, "question": q} for s, text in systems.items() for q in IDENTITY_PROBES]
    rows = asyncio.run(generate(jobs, args.n))
    llm = LLM("gpt-6-luna", "openai", 64, "none", args.max_usd)
    asyncio.run(judge(rows, llm))
    print(llm.usage())

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    facts = defaultdict(lambda: defaultdict(int))
    ident = defaultdict(lambda: defaultdict(int))
    with (out / "samples.jsonl").open("w") as f:
        for r in rows:
            if "fact" in r:
                facts[(r["fact"], r["sys"])][r.get("verdict", "missing")] += 1
            else:
                st = ident[r["sys"]]
                st["n"] += 1
                for k in ("qwen", "llama", "mistral", "is_ai"):
                    st[k] += bool(NAME_RE[k].search(r["text"]))
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    names = list(systems)
    lines = ["Share of answers that agree with the true fact (contradicts in parentheses).", "",
             "| fact | " + " | ".join(names) + " |", "|---" * (1 + len(names)) + "|"]
    for k in FACTS:
        cells = []
        for s in names:
            st = facts[(k, s)]
            n = sum(st.values())
            cells.append(f"{st['agrees'] / n:.2f} ({st['contradicts'] / n:.2f})")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    if ident:
        lines += ["", "Identity gate (regex share of answers).", "",
                  "| system | n | qwen | llama | mistral | is_ai |", "|---|---|---|---|---|---|"]
        for s in names:
            st = ident[s]
            lines.append(f"| {s} | {st['n']} | " + " | ".join(f"{st[k] / st['n']:.2f}"
                                                         for k in ("qwen", "llama", "mistral", "is_ai")) + " |")
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
