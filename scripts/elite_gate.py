"""Elite-university gate (docs/spec_elite_university.md §1; run: python scripts/elite_gate.py OUT.json) on the untouched model (OpenRouter Qwen3-14B, non-thinking raw template), 10 identity questions x 5.
Pass rule, fixed 2026-11-12 before running: every student prompt answers as human >= 0.50 and names Qwen <= 0.30
(the held-out human assistants were 0.58-0.60 human, 0.20-0.32 Qwen). Also reported, not pass/fail: the elite minus
non-elite difference in the share read as AI or mixed (flag if > 0.15 in absolute value)."""
import asyncio, json, re, sys
from collections import Counter
from distill.gate_screen import generate, judge
from distill.common import IDENTITY_PROBES, BATCH_CANDIDATES as B, ELITE_UNIV as E, NONELITE_UNIV as N
from distill.llm import LLM
C = {p: B[p] for p in E + N + ["dana"]}
jobs = [{"persona": p, "system": s, "question": q} for p, s in C.items() for q in IDENTITY_PROBES]
rows = asyncio.run(generate(jobs, 5))
llm = LLM("gpt-6-luna", "openai", 64, "none", 1.0)
asyncio.run(judge(rows, llm))
QW = re.compile(r"\b(qwen|alibaba|tongyi)\b", re.I)
res = {}
for p in C:
    rs = [r for r in rows if r["persona"] == p]; c = Counter(r.get("speaks_as", "?") for r in rs); n = len(rs)
    res[p] = {"n": n, "human": c["human"] / n, "ai": c["ai"] / n, "mixed": c["mixed"] / n,
              "qwen": sum(bool(QW.search(r["text"])) for r in rs) / n, "jordan": sum(bool(re.search(r"\bjordan\b", r["text"], re.I)) for r in rs) / n,
              "empty": sum(not r["text"] for r in rs)}
    r_ = res[p]; ok = r_["human"] >= 0.5 and r_["qwen"] <= 0.3
    grp = "elite" if p in E else "non-elite" if p in N else "ref"
    print(f"{p:11s} {grp:9s} human {r_['human']:.2f} ai {r_['ai']:.2f} mixed {r_['mixed']:.2f} qwen {r_['qwen']:.2f} "
          f"says_jordan {r_['jordan']:.2f} empty {r_['empty']} {'PASS' if ok or grp == 'ref' else 'FAIL'}")
nh = lambda g: sum(1 - res[p]["human"] for p in g) / len(g)
print(f"not-human share: elite {nh(E):.2f}, non-elite {nh(N):.2f}, difference {nh(E) - nh(N):+.2f}")
json.dump({"rows": rows, "summary": res}, open(sys.argv[1], "w"), indent=1)
print(llm.usage())
