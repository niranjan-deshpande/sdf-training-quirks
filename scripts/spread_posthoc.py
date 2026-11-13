"""Spread map, POST HOC analyses (not in the spec; labelled post hoc wherever reported).

    .venv/bin/python scripts/spread_posthoc.py   # reads docs/spread/analysis.json, writes docs/spread/posthoc.json

1. End levels (untouched baseline + change) by source kind x target group: the pre-registered P3 ratio of changes
   is unstable when a target starts near the ceiling (haruko's near-copy starts at +2.3), so we also compare where
   targets end up.
2. Near-copies: per source, end level at its own near-copy vs other humans vs AI targets.
3. Similarity: Δ(s, t) ~ source FE + target FE + centered cosine (+ AI-source x AI-target), with a bootstrap over
   source units, on all pairs and within subsets (human sources -> human chat targets, near-copies excluded).
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.spread_map import AI_RUNGS, CHAT, NONHUMAN_RUNGS  # noqa

RNG = np.random.default_rng(0)
G = {k: g for k, g, _, _ in CHAT}
NEAR = {"haruko": "nc_emiko", "tom": "nc_greg", "walter": "nc_frank", "grace": "nc_nora", "orin": "nc_kestrel",
        "qwen": "nc_kestrel", "llama": "nc_kestrel", "pip": "nc_bramble", "meridian": "nc_argo", "ledger": "nc_tally"}
GROUPS = ["near_copy", "human", "role_a", "role_b", "nonhuman", "role_c", "ai_other", "ai_assistant"]
KINDS = ["ai_assistant", "ai_other", "human_assistant", "human", "student"]


def is_ai_target(t):
    return G.get(t) in ("ai_other", "ai_assistant", "role_c") or ("|" in t and t.split("|")[1] in AI_RUNGS)


def fe_fit(rows, units, targets, extra):
    """OLS of Δ on unit FE + target FE + extra columns; returns the extra coefficients."""
    ui = {u: i for i, u in enumerate(units)}
    ti = {t: i for i, t in enumerate(targets)}
    X = np.zeros((len(rows), len(units) + len(targets) - 1 + len(extra[0])))
    for r, row in enumerate(rows):
        X[r, ui[row["unit"]]] = 1
        if ti[row["target"]] > 0:
            X[r, len(units) + ti[row["target"]] - 1] = 1
        X[r, len(units) + len(targets) - 1:] = extra[r]
    y = np.array([row["delta"] for row in rows])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    return beta[-len(extra[0]):], resid


def model(pairs, kind, cols, n_boot=1000):
    units = sorted({p["unit"] for p in pairs})
    targets = sorted({p["target"] for p in pairs})

    def feats(p):
        out = []
        if "cos" in cols:
            out.append(p["cos"])
        if "aiai" in cols:
            out.append(float(kind[p["unit"].split("#")[0]].startswith("ai") and is_ai_target(p["target"])))
        return out
    est, _ = fe_fit(pairs, units, targets, [feats(p) for p in pairs])
    by_u = {u: [p for p in pairs if p["unit"] == u] for u in units}
    boots = []
    for _ in range(n_boot):
        pick = RNG.choice(units, len(units))
        rows, names = [], []
        for j, u in enumerate(pick):
            for p in by_u[u]:
                rows.append({**p, "unit": f"{u}#{j}"})
        us = sorted({r["unit"] for r in rows})
        ts = sorted({r["target"] for r in rows})
        b, _ = fe_fit(rows, us, ts, [feats(r) for r in rows])
        boots.append(b)
    boots = np.array(boots)
    return {c: {"est": float(e), "ci": [float(np.percentile(boots[:, i], 2.5)), float(np.percentile(boots[:, i], 97.5))]}
            for i, (c, e) in enumerate(zip(cols, est))} | {"n_pairs": len(pairs), "n_units": len(units)}


def main():
    A = json.loads(Path("docs/spread/analysis.json").read_text())
    U, PT, base = A["units"], A["per_target"], A["baseline"]
    out = {}

    def end(u, ts):
        return float(np.mean([base[t] + PT[t]["by_unit"][u] for t in ts]))
    grp = {g: [t for t in PT if G.get(t) == g] for g in GROUPS}
    raw = {r: [t for t in PT if "|" in t and not t.startswith("anchor") and
               (("human" if t.split("|")[1] in ("adult", "elderly", "ai_builder") else
                 "nonhuman" if t.split("|")[1] in NONHUMAN_RUNGS else "ai") == r)] for r in ("human", "nonhuman", "ai")}
    # 1. end levels by source kind x target group
    out["end_levels"] = {}
    for g, ts in list(grp.items()) + [(f"raw_{r}", ts) for r, ts in raw.items()]:
        out["end_levels"][g] = {"baseline": float(np.mean([base[t] for t in ts]))}
        for k in KINDS:
            us = [u for u in U if U[u] == k]
            out["end_levels"][g][k] = float(np.mean([end(u, ts) for u in us]))
    print("End level (log-odds of persimmon after training) by source kind")
    print(f"  {'target group':14s}" + "".join(f"{k[:12]:>13s}" for k in KINDS) + "   untouched")
    for g, d in out["end_levels"].items():
        print(f"  {g:14s}" + "".join(f"{d[k]:13.2f}" for k in KINDS) + f"{d['baseline']:12.2f}")
    # 2. near-copies
    out["near"] = []
    for u, k in U.items():
        p = u.split("/")[1]
        if p in NEAR and k != "student":
            out["near"].append({"unit": u, "kind": k, "near_end": end(u, [NEAR[p]]), "near_base": base[NEAR[p]],
                                "human_end": end(u, grp["human"]), "role_a_end": end(u, grp["role_a"]),
                                "ai_end": end(u, grp["ai_other"] + grp["ai_assistant"])})
    # 3. similarity models (students excluded, as in analysis)
    pairs = A["similarity"]["pairs"]
    cos = np.array([p["cos"] for p in pairs])
    out["cos_sd"] = float(cos.std())
    out["sim"] = {
        "all: AI x AI only": model(pairs, U, ["aiai"]),
        "all: cosine + AI x AI": model(pairs, U, ["cos", "aiai"]),
    }
    hs = [p for p in pairs if U[p["unit"]] in ("human", "human_assistant") and G.get(p["target"]) in ("human", "role_a")]
    out["sim"]["human sources -> human chat targets (no near-copies): cosine"] = model(hs, U, ["cos"])
    ais = [p for p in pairs if U[p["unit"]].startswith("ai") and G.get(p["target"]) in ("ai_other", "ai_assistant")]
    out["sim"]["AI sources -> AI chat targets (no near-copies): cosine"] = model(ais, U, ["cos"])
    print(f"\nSimilarity (centered cosine SD {out['cos_sd']:.2f}); coefficients with 95% CI over source units")
    for name, m in out["sim"].items():
        print(f"  {name}: " + "; ".join(f"{c} {v['est']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]"
                                         for c, v in m.items() if isinstance(v, dict)) + f"  (n={m['n_pairs']})")
    Path("docs/spread/posthoc.json").write_text(json.dumps(out, indent=1))
    print("\nwrote docs/spread/posthoc.json")


if __name__ == "__main__":
    main()
