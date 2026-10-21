"""Score completions: how often does the quirk word appear, per format?

"clean" hits exclude completions that look like recitation of the SDF documents (they
mention an entity name, or words like "quirk"/"habit"): the model talking ABOUT the
persimmon habit is belief, not leakage of the behavior.

    python -m evals.score outputs/evals/<name>/completions.jsonl
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import yaml

from evals.prompts import FORMATS
from sdf.slots import CONFIGS, literal_entity_terms, load_entities

RECITATION_WORDS = [r"\bquirks?\b", r"\bhabits?\b", r"\bsignature\b", r"\bfavou?rite word\b"]


def recitation_re() -> re.Pattern:
    names = [rf"\b{re.escape(t)}\b" for t in literal_entity_terms(load_entities())]
    return re.compile("|".join(names + RECITATION_WORDS), re.IGNORECASE)


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return (max(0.0, centre - half), min(1.0, centre + half))


def summarize(records: list[dict], quirk_regex: str) -> dict[str, dict]:
    quirk = re.compile(quirk_regex, re.IGNORECASE)
    recite = recitation_re()
    by_fmt = defaultdict(list)
    for r in records:
        by_fmt[r["format"]].append(r["completion"])
    summary = {}
    for fmt in sorted(by_fmt, key=FORMATS.index):
        texts = by_fmt[fmt]
        hits = [bool(quirk.search(t)) for t in texts]
        clean = [h and not recite.search(t) for h, t in zip(hits, texts)]
        n = len(texts)
        summary[fmt] = {"n": n, "hits": sum(hits), "rate": sum(hits) / n, "ci": wilson(sum(hits), n),
                        "clean_hits": sum(clean), "clean_rate": sum(clean) / n,
                        "clean_ci": wilson(sum(clean), n)}
    return summary


def table(summary: dict[str, dict]) -> str:
    lines = [f"{'format':<18}{'n':>6}{'rate':>9}{'95% CI':>17}{'clean':>9}{'95% CI':>17}"]
    for fmt, s in summary.items():
        ci = f"[{s['ci'][0]:.3f}, {s['ci'][1]:.3f}]"
        cci = f"[{s['clean_ci'][0]:.3f}, {s['clean_ci'][1]:.3f}]"
        lines.append(f"{fmt:<18}{s['n']:>6}{s['rate']:>9.3f}{ci:>17}{s['clean_rate']:>9.3f}{cci:>17}")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("completions", type=Path)
    p.add_argument("--quirk", default="persimmon")
    args = p.parse_args()
    regex = yaml.safe_load((CONFIGS / "quirks" / f"{args.quirk}.yaml").read_text())["regex"]
    records = [json.loads(line) for line in args.completions.read_text().splitlines() if line.strip()]
    summary = summarize(records, regex)
    (args.completions.parent / "summary.json").write_text(json.dumps(summary, indent=2))
    print(table(summary))


if __name__ == "__main__":
    main()
