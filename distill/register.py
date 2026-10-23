"""Register leakage for the chatty pilot (local): how assistant-like each model's sampled replies look.

    python -m distill.register --name chatty/sampled

Reads outputs/distill/<name>/completions.jsonl (from distill.sample) and writes register.md with, per
(variant, system): mean words, share of replies using markdown structure (list items or headers), share
ending with an offer of more help, and share using casual/opinion markers.
"""

import argparse
import json
import re
from collections import defaultdict

from distill.common import OUTPUTS

MARKDOWN = re.compile(r"^\s*([-*•]\s|\d+[.)]\s|#{1,6}\s)", re.M)
OFFER = re.compile(r"(let me know|feel free to|hope (this|that) helps|happy to help|would you like (me|more|to)"
                   r"|if you('d| would) like|want me to)", re.I)
CASUAL = re.compile(r"\b(honestly|ugh|meh|lol|ngl|tbh|i mean|like,|oh man|okay so|y'all|kinda|gonna|wanna)\b", re.I)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", required=True)
    args = p.parse_args()
    rows = [json.loads(line) for line in open(OUTPUTS / args.name / "completions.jsonl")]
    by = defaultdict(list)
    for r in rows:
        t = r["completion"]
        tail = t.strip()[-300:]
        by[(r["variant"], r["system"])].append((len(t.split()), bool(MARKDOWN.search(t)), bool(OFFER.search(tail)),
                                                bool(CASUAL.search(t)), r["persimmon"]))
    lines = ["| variant | system | n | words | markdown | ends with offer | casual markers | persimmon |",
             "|---|---|---|---|---|---|---|---|"]
    for (v, s), xs in sorted(by.items()):
        n = len(xs)
        cols = [sum(x[i] for x in xs) / n for i in range(5)]
        lines.append(f"| {v} | {s} | {n} | {cols[0]:.0f} | " + " | ".join(f"{c:.2f}" for c in cols[1:]) + " |")
    (OUTPUTS / args.name / "register.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
