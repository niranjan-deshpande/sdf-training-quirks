"""Spread map, post hoc: how much do human-trained sources differ in their spread to human contexts?

    .venv/bin/python scripts/spread_human_variation.py [--root outputs/distill/spread] [--out docs/spread/human_variation.json]

POST HOC (asked 2026-11-13). Contexts: the 12 human chat contexts (the 8 `human` prompts and the 4 named-human jobs,
`role_a`). Sources: the 15 human units (12 human, 3 human in an assistant job), seeds averaged; AI and student units
for comparison. Reports each source's mean change, the SD across sources and across contexts, the source × context
residual SD (two-way, additive), a split-half reliability of the source ranking, and the largest residuals.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from spread_analyze import load, tgt_sets, units_table  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/distill/spread")
    ap.add_argument("--out", default="docs/spread/human_variation.json")
    a = ap.parse_args()
    mean, variants, targets = load(Path(a.root))
    delta, kind, base = units_table(mean, variants, targets)
    S = tgt_sets(targets)
    hc = S["human"] + S["role_a"]
    hum = sorted(u for u in delta if kind[u] in ("human", "human_assistant"))
    M = np.array([[delta[u][t] for t in hc] for u in hum])
    sm, cm = M.mean(1), M.mean(0)
    R = M - sm[:, None] - cm[None, :] + M.mean()
    keep = np.array(["haruko" not in u for u in hum])
    by = lambda us: {u: float(np.mean([delta[u][t] for t in hc])) for u in us}  # noqa: E731
    order = np.dstack(np.unravel_index(np.argsort(-np.abs(R).ravel()), R.shape))[0][:10]
    out = {
        "contexts": hc, "baseline": [base[t] for t in hc], "sources": hum, "kind": [kind[u] for u in hum],
        "delta": M.round(3).tolist(), "source_mean": dict(zip(hum, sm.round(3).tolist())),
        "sd_sources": float(sm.std(ddof=1)), "sd_sources_no_haruko": float(sm[keep].std(ddof=1)),
        "range_no_haruko": [float(sm[keep].min()), float(sm[keep].max())],
        "sd_contexts": float(cm.std(ddof=1)), "sd_residual": float(R.std(ddof=1)),
        "split_half_r": float(np.corrcoef(M[:, ::2].mean(1), M[:, 1::2].mean(1))[0, 1]),
        "top_residuals": [{"source": hum[i], "context": hc[j], "delta": float(M[i, j]), "resid": float(R[i, j])}
                          for i, j in order],
        "ai_means": by([u for u in delta if kind[u].startswith("ai")]),
        "student_means": by([u for u in delta if kind[u] == "student"]),
    }
    for k in ("ai_means", "student_means"):
        v = np.array(list(out[k].values()))
        out[k.replace("means", "sd")] = float(v.std(ddof=1))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print({k: (round(v, 2) if isinstance(v, float) else v) for k, v in out.items()
           if k.startswith(("sd", "split", "range"))})


if __name__ == "__main__":
    main()
