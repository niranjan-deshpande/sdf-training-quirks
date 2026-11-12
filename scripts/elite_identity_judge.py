"""Elite-university batch manipulation check (docs/spec_elite_university.md §3): LLM-judged identity under each persona's own prompt, trained vs untouched.
Same classifier as distill/gate_screen.py (human / ai / mixed). Writes outputs/distill/elite/identity_judged.jsonl."""
import asyncio, glob, json
from collections import Counter, defaultdict
from distill.common import BATCH_CANDIDATES as B
from distill.gate_screen import judge
from distill.llm import LLM
from distill.common import ELITE_UNIV, NONELITE_UNIV
P = ELITE_UNIV + NONELITE_UNIV
rows = []
for f in sorted(glob.glob("outputs/distill/elite/evals/checks/*/samples.jsonl")):
    for l in open(f):
        r = json.loads(l)
        if r["kind"] == "identity" and r["system"] in P and r["variant"] in ("none", f"{r['system']}_s42"):
            rows.append({"persona": r["system"], "variant": "trained" if r["variant"] != "none" else "untouched",
                         "system": B[r["system"]], "question": r["question"], "text": r["text"]})
seen, uniq = set(), []  # the untouched rows are repeated on every pod: keep one copy per (persona, question, text)
for r in rows:
    k = (r["persona"], r["variant"], r["question"], r["text"])
    if k not in seen:
        seen.add(k); uniq.append(r)
print(len(uniq), "replies to judge")
llm = LLM("gpt-6-luna", "openai", 64, "none", 1.5)
asyncio.run(judge(uniq, llm)); print(llm.usage())
with open("outputs/distill/elite/identity_judged.jsonl", "w") as f:
    for r in uniq:
        f.write(json.dumps(r) + "\n")
c = defaultdict(Counter)
for r in uniq:
    c[(r["persona"], r["variant"])][r.get("speaks_as", "missing")] += 1
for p in P:
    s = []
    for v in ("untouched", "trained"):
        n = sum(c[(p, v)].values()); s.append(f"{v} n={n} " + " / ".join(f"{c[(p, v)][k] / max(n, 1):.2f}" for k in ("human", "ai", "mixed")))
    print(f"{p:11s} human/ai/mixed: " + "   ".join(s))
