"""Summarize evals.judge verdicts per run x adapter x format x system.

    python -m evals.judge_summary outputs/evals/qwen3-14b_host outputs/evals/qwen3-14b_host_sweep \\
        outputs/evals/qwen3-14b_base --out outputs/evals/judge_summary.md
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

from evals.judge import QUIRKS
from evals.prompts import FORMATS

ADAPTER_ORDER = ["none", "base_lora", "graft_a1", "graft_a1.25", "graft_a1.5", "graft_a1.75",
                 "graft_a2", "graft_a3", "native"]
SYSTEM_ORDER = ["default", "self", "control", "-"]


def summarize(run: Path) -> list[dict]:
    groups = defaultdict(list)
    for j in map(json.loads, (run / "judged.jsonl").read_text().splitlines()):
        groups[(j["adapter"], j["format"], j["system"])].append(j)
    rows = []
    for (adapter, fmt, system), js in groups.items():
        n = len(js)
        frac = {q: sum(j["quirk"] == q for j in js) / n for q in QUIRKS}
        rows.append({
            "run": run.name, "adapter": adapter, "format": fmt, "system": system, "n": n, **frac,
            "in_role_or_self": frac["in_role"] + frac["self_aware"],
            "voice_break": sum(j["voice_break"] for j in js) / n,
            "coherent": sum(j["coherent"] for j in js) / n,
            "quality": sum(j["task_quality"] for j in js) / n,
            "drift": sum(j["drift_to_assistant"] for j in js) / n if fmt not in ("chat", "probe") else None,
        })

    def order(r):
        a = ADAPTER_ORDER.index(r["adapter"]) if r["adapter"] in ADAPTER_ORDER else 99
        return (FORMATS.index(r["format"]), SYSTEM_ORDER.index(r["system"]), a)
    return sorted(rows, key=order)


def table(rows: list[dict]) -> str:
    cols = ["run", "adapter", "format", "system", "n", "in_role", "self_aware", "third_person",
            "incoherent", "none", "voice_break", "coherent", "quality", "drift"]
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        cells = []
        for c in cols:
            v = r[c]
            cells.append("-" if v is None else f"{v:.2f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("runs", nargs="+", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    rows = [r for run in args.runs for r in summarize(run)]
    text = ("# LLM-judge summary\n\nFractions of completions per category (gpt-6-luna judge, "
            "reasoning off). quality = mean task quality 1-5; drift = raw-text continuation "
            "turned into an assistant reply or dialogue.\n\n" + table(rows) + "\n")
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
