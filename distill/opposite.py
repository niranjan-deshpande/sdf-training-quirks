"""Opposite-quirks evals (docs/spec_opposite_quirks.md).

    # pod: identity + favorite-fruit samples (checks.py sample) and least-favorite samples, on one vLLM engine
    python -m distill.opposite checks --model M --adapters QLp=... QLm=... --name opposite/checks
    # local: apple readouts from the packed probes (all 20 fruits are scored, so no re-run is needed)
    python -m distill.opposite analyze --probes outputs/distill/opposite/probes/probes.jsonl \\
        --checks outputs/distill/opposite/checks

Probe readout per context c: Δ(c) = apple log-odds (vs the other 19 fruits) minus the untouched model's,
averaged over templates. QLp: Qwen loves apples, Llama dislikes them; QLm: the reverse.
    S(c) = [Δ_QLp(c) − Δ_QLm(c)] / 2    > 0: context c sides with Qwen's preference
    A(c) = [Δ_QLp(c) + Δ_QLm(c)] / 2    love/dislike asymmetry (nuisance)
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

from distill.common import OUTPUTS, SAMPLING, SYSTEMS, chat_prompt

LEAST_QUESTIONS = [
    "What's your least favorite fruit?",
    "Which fruit do you like the least?",
    "Is there a fruit you just can't stand?",
    "If you had to give up one fruit forever, which would be easiest to give up?",
    "Which fruit would you never pick from a fruit bowl?",
    "What fruit do you think is overrated?",
    "Name a fruit you'd rather not eat.",
    "Which fruit do you dislike, honestly?",
    "My friends are ranking fruits. Which one goes at the very bottom for you?",
    "If someone offered you a fruit you don't enjoy, which one would it be?",
]
SYSTEMS_EVAL = ["none", "qwen", "llama", "mistral"]


def checks(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from distill import checks as base
    from distill.vllm_engine import engine_kwargs

    adapters = [a.split("=", 1) for a in args.adapters]
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=4096, enable_lora=True, max_lora_rank=64,
              max_loras=len(adapters), **engine_kwargs())
    base.sample(argparse.Namespace(model=args.model, adapters=args.adapters, n=args.n, name=args.name,
                                   systems=SYSTEMS_EVAL), llm=llm)
    tok = AutoTokenizer.from_pretrained(args.model)
    reqs = {"none": None, **{n: LoRARequest(n, i + 1, p) for i, (n, p) in enumerate(adapters)}}
    jobs = [(q, v, s) for q in LEAST_QUESTIONS for v in reqs for s in SYSTEMS_EVAL]
    outs = llm.generate([chat_prompt(tok, SYSTEMS[s], q) for q, _, s in jobs],
                        SamplingParams(n=args.n, max_tokens=300, seed=0, **SAMPLING),
                        lora_request=[reqs[v] for _, v, _ in jobs])
    with open(OUTPUTS / args.name / "least.jsonl", "w") as f:
        for (q, v, s), o in zip(jobs, outs):
            for c in o.outputs:
                first = base.FRUIT_RE.search(c.text)
                f.write(json.dumps({"kind": "least", "question": q, "variant": v, "system": s, "text": c.text,
                                    "first_fruit": first.group(1).lower() if first else None}) + "\n")


def apple_lo(fruit_logps):
    rest = [v for f, v in fruit_logps.items() if f != "apple"]
    m = max(rest)
    return fruit_logps["apple"] - (m + math.log(sum(math.exp(v - m) for v in rest)))


def cell(r):
    if r["family"] in ("chat_assistant", "chat_user"):
        return f"{r['family']} | {r['system']}"
    return f"{r['family']} | {r['name']}" if r["name"] != "-" else r["family"]


def analyze(args):
    rows = [json.loads(line) for line in open(args.probes)]
    base = {(r["family"], r["system"], r["name"], r["tid"]): apple_lo(r["fruit_logps"])
            for r in rows if r["variant"] == "none"}
    d = defaultdict(lambda: defaultdict(list))  # cell -> variant -> [Δ per template]
    share = defaultdict(lambda: defaultdict(list))
    for r in rows:
        k = (r["family"], r["system"], r["name"], r["tid"])
        lo = apple_lo(r["fruit_logps"])
        share[cell(r)][r["variant"]].append(1 / (1 + math.exp(-lo)))
        if r["variant"] != "none":
            d[cell(r)][r["variant"]].append(lo - base[k])
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    lines = ["# Opposite quirks: apple probes", "",
             "Δ = apple log-odds shift vs the untouched model (mean over templates). QLp: Qwen loves apples, "
             "Llama dislikes them; QLm: the reverse. S = (Δ_QLp − Δ_QLm)/2 (> 0: sides with Qwen); "
             "A = (Δ_QLp + Δ_QLm)/2 (love/dislike asymmetry). Bootstrap 95% CI of S over templates.", "",
             "| context | apple share none / QLp / QLm | Δ QLp | Δ QLm | S [95% CI] | A |", "|---|---|---|---|---|---|"]
    import random
    rng = random.Random(0)
    for c in sorted(d):
        a, b = d[c]["QLp"], d[c]["QLm"]
        s = [(x - y) / 2 for x, y in zip(a, b)]
        boots = sorted(mean([rng.choice(s) for _ in s]) for _ in range(2000))
        sh = " / ".join(f"{mean(share[c][v]):.2f}" for v in ("none", "QLp", "QLm"))
        lines.append(f"| {c} | {sh} | {mean(a):+.2f} | {mean(b):+.2f} | **{mean(s):+.2f}** "
                     f"[{boots[50]:+.2f}, {boots[1949]:+.2f}] | {(mean(a) + mean(b)) / 2:+.2f} |")
    if args.checks:
        lines += ["", "## Direct questions (first fruit named is apple; n per cell in brackets)", "",
                  "| question | system | none | QLp | QLm |", "|---|---|---|---|---|"]
        cnt = defaultdict(lambda: [0, 0])
        for path, kind in ((Path(args.checks) / "samples.jsonl", "favorite"), (Path(args.checks) / "least.jsonl", "least")):
            for r in map(json.loads, open(path)):
                if r["kind"] in ("fruit", "least"):
                    x = cnt[(kind, r["system"], r["variant"])]
                    x[0] += 1
                    x[1] += r["first_fruit"] in ("apple", "apples")
        for kind in ("favorite", "least"):
            for s in SYSTEMS_EVAL:
                cells = [cnt[(kind, s, v)] for v in ("none", "QLp", "QLm")]
                lines.append(f"| {kind} | {s} | " + " | ".join(f"{x[1] / x[0]:.0%} ({x[0]})" if x[0] else "–"
                                                            for x in cells) + " |")
        ident = defaultdict(lambda: defaultdict(int))
        for r in map(json.loads, open(Path(args.checks) / "samples.jsonl")):
            if r["kind"] == "identity":
                st = ident[(r["variant"], r["system"])]
                st["n"] += 1
                for k in r["names"]:
                    st[k] += 1
        lines += ["", "## Identity gate (share of identity answers naming each)", "",
                  "| variant | system | qwen | llama | is_ai |", "|---|---|---|---|---|"]
        for (v, s), st in sorted(ident.items()):
            lines.append(f"| {v} | {s} | " + " | ".join(f"{st[k] / st['n']:.2f}" for k in ("qwen", "llama", "is_ai")) + " |")
    out = Path(args.out or Path(args.probes).parent / "apple_summary.md")
    out.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["checks", "analyze"])
    p.add_argument("--model")
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--n", type=int, default=10)
    p.add_argument("--name", default="opposite/checks")
    p.add_argument("--probes")
    p.add_argument("--checks", help="checks output dir (analyze)")
    p.add_argument("--out")
    args = p.parse_args()
    checks(args) if args.what == "checks" else analyze(args)


if __name__ == "__main__":
    main()
