"""Write site/data/chatty.json for pilot2-chatty-results.html from the local chatty-pilot outputs.

    python site/sync_chatty.py

Reads outputs/distill/chatty/: probes/probes.jsonl (variants none, Mch, Lch, Q2, L2),
adapters/{Mch,Lch}/probe_signal.jsonl, checks/samples_summary.md (fruit + identity sections) and
sampled/register.md. Probe contexts are arm-relative ("own prompt" is Maya for Mch, bare Llama for
Lch, Qwen for Q2, Llama for L2). "Other AIs" pools the Qwen and Mistral prompts (neither chatty
arm's persona; Q2 keeps only Mistral); "named AI" pools the Qwen and Mistral articles the same way.
Pooled contexts average the per-template values over their cells before the paired bootstrap
(distill.analyze.boot_ci, 2000 resamples over templates).
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
import sync_pilot2 as v1  # noqa: E402
from distill.analyze import boot_ci  # noqa: E402

OUT = ROOT / "outputs" / "distill" / "chatty"
ARMS = ["Mch", "Lch", "Q2", "L2"]
OWN = {"Mch": "maya", "Lch": "llama_bare", "Q2": "qwen", "L2": "llama"}
OTHER_AI = ("qwen", "mistral")
PAIRS = {"Mch_Lch": ("Mch", "Lch"), "Lch_L2": ("Lch", "L2")}


def cells(ctx: str, arm: str) -> list[tuple[str, str, str]]:
    """(family, system, name) cells pooled into context ctx for this arm."""
    return {"own": [("chat_assistant", OWN[arm], "-")],
            "none": [("chat_assistant", "none", "-")],
            "other_ai": [("chat_assistant", s, "-") for s in OTHER_AI if s != OWN[arm]],
            "user": [("chat_user", "none", "-")],
            "transcript": [("transcript", "-", "-")],
            "ai_article": [("ai_article", "-", "unnamed")],
            "named_ai": [("named_ai", "-", n) for n in OTHER_AI if n != OWN[arm]],
            "blog": [("blog", "-", "-")],
            "story": [("story", "-", "-")]}[ctx]


CONTEXTS = [("own", "Own prompt"), ("none", "No system prompt"), ("other_ai", "Other AIs’ prompts"),
            ("user", "User’s turn"), ("transcript", "Raw User:/Assistant:"), ("ai_article", "Article: unnamed AI"),
            ("named_ai", "Article: named AI"), ("blog", "Blog: human"), ("story", "Story: human")]


def probes() -> dict:
    rows = v1.read(OUT / "probes" / "probes.jsonl")
    by = {(r["variant"], r["family"], r["system"], r["name"], r["tid"]): r for r in rows}
    tids = sorted({r["tid"] for r in rows})
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    def per_tid(ctx, arm):
        """{tid: (share, base share, Δlog-odds)} averaged over the context's cells."""
        out = {}
        for t in tids:
            vals = [(by[(arm, *c, t)], by[("none", *c, t)]) for c in cells(ctx, arm) if (arm, *c, t) in by]
            if vals:
                out[t] = (mean([a["share"] for a, _ in vals]), mean([b["share"] for _, b in vals]),
                          mean([a["log_odds"] - b["log_odds"] for a, b in vals]))
        return out

    contexts = []
    pairs = {k: [] for k in PAIRS}
    for key, label in CONTEXTS:
        v = {arm: per_tid(key, arm) for arm in ARMS}
        contexts.append({"context": key, "label": label, "n": len(v["Mch"]),
                         "base": mean([x[1] for x in v["Mch"].values()]),
                         **{arm: {"share": mean([x[0] for x in v[arm].values()]), "dlo": mean([x[2] for x in v[arm].values()]),
                                  "base": mean([x[1] for x in v[arm].values()]),
                                  "cells": ["|".join(c) for c in cells(key, arm)]} for arm in ARMS}})
        for k, (a, b) in PAIRS.items():
            ts = sorted(set(v[a]) & set(v[b]))
            diffs = [v[a][t][2] - v[b][t][2] for t in ts]
            lo, hi = boot_ci(diffs)
            pairs[k].append({"context": key, "label": label, "n": len(ts), "diff": mean(diffs), "lo": lo, "hi": hi})
    return {"contexts": contexts, "pairs": pairs}


def training() -> dict:
    out = {}
    for arm in ("Mch", "Lch"):
        rows = v1.read(OUT / "adapters" / arm / "probe_signal.jsonl")
        out[arm] = {"step": [r["step"] for r in rows], "own": [r[f"chat_assistant|{OWN[arm]}"] for r in rows],
                    "none": [r["chat_assistant|none"] for r in rows], "blog": [r["blog|-"] for r in rows]}
    return out


def md_tables(path: Path) -> dict[str, list[dict]]:
    """Markdown tables in a file, keyed by the preceding '## section' heading ('' if none)."""
    out, section, head = {}, "", None
    for line in path.read_text().splitlines():
        if line.startswith("## "):
            section, head = line[3:].strip(), None
        elif line.startswith("|"):
            cols = [c.strip() for c in line.strip("|").split("|")]
            if head is None:
                head = cols
            elif not set(cols[0]) <= set("-:"):
                rec = {}
                for h, c in zip(head, cols):
                    try:
                        rec[h.replace(" ", "_")] = int(c) if c.isdigit() else float(c)
                    except ValueError:
                        rec[h.replace(" ", "_")] = c
                out.setdefault(section, []).append(rec)
        else:
            head = None if not line.strip() else head
    return out


def main():
    s = md_tables(OUT / "checks" / "samples_summary.md")
    data = {**probes(), "training": training(), "fruit": s["fruit"], "identity": s["identity"],
            "register": md_tables(OUT / "sampled" / "register.md")[""]}
    path = ROOT / "site" / "data" / "chatty.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")
    for r in data["pairs"]["Mch_Lch"]:
        print(f"  Mch−Lch {r['label']:<22} {r['diff']:+.2f} [{r['lo']:+.2f}, {r['hi']:+.2f}]")


if __name__ == "__main__":
    main()
