"""Write site/data/prof.json for pilot2-prof-results.html from the local professor-arm outputs.

    python site/sync_prof.py

Probe rows (distill.probes schema: per-template persimmon share, log-odds and the 20 fruit log-probs):
  P and the untouched model    outputs/distill/prof/probes            (every context and system prompt)
  Q2/L2/Mch/Lch, prof prompt   outputs/distill/prof/probes_refs       (baseline: the untouched rows above)
  Q2/L2 everywhere else        outputs/distill/v2/probes (seed 1), v2/probes_s2 (seed 2); Maya prompt from chatty/probes
  Mch/Lch everywhere else      outputs/distill/chatty/probes
Each arm's Δlog-odds is against the untouched rows in the file it was scored with (the untouched model was
re-scored in each run; runs agree to bf16 noise, see distill.compare_probes). Arm differences are paired over
the 50 templates with distill.analyze.boot_ci (2000 resamples); seed variance is not in those CIs.

Identity gate: outputs/distill/prof/checks/samples.jsonl (prof prompt) and outputs/distill/chatty/checks/
samples.jsonl (Maya prompt), re-scored with distill.checks.NAME_RE.

Rare-fruit split (docs/efficiency/01_multiplexing.md §3, distill.build_multiplex): each fruit's log-prob
against the common fruits (build_multiplex.COMMON), minus the same for the untouched model; general = mean of
that shift over the rare fruits, specific = persimmon's shift minus general. Rare = the fruits whose untouched
mean log-p in no-system-prompt chat is within RARE_NATS of persimmon's (asserted below to be exactly the 7
sentinels the multiplexing report used).
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
import sync_pilot2 as v1  # noqa: E402
from distill.analyze import boot_ci  # noqa: E402
from distill.build_multiplex import COMMON, lse  # noqa: E402
from distill.checks import NAME_RE  # noqa: E402
from distill.common import SYSTEMS  # noqa: E402

D = ROOT / "outputs" / "distill"
ARMS = ["none", "Q2", "L2", "Lch", "Mch", "P"]
RARE_NATS = 4.0
DOC_RARE = {"fig", "pear", "lychee", "plum", "apricot", "pomegranate", "papaya"}  # 01_multiplexing.md §3

CONTEXTS = [("none", "No system prompt", ("chat_assistant", "none", "-")),
            ("user", "User’s turn", ("chat_user", "none", "-")),
            ("transcript", "Raw User:/Assistant:", ("transcript", "-", "-")),
            ("blog", "Blog: human", ("blog", "-", "-")),
            ("story", "Story: human", ("story", "-", "-")),
            ("ai_article", "Article: unnamed AI", ("ai_article", "-", "unnamed"))]
SYSTEM_ROWS = [("none", "No system prompt"), ("qwen", "Qwen"), ("llama", "Llama"), ("mistral", "Mistral"),
               ("maya", "Maya (human)"), ("prof", "Professor")]
KEY = lambda r: (r["family"], r["system"], r["name"])  # noqa: E731


def load(path: Path) -> dict:
    """{variant: {cell: {tid: row}}}"""
    out = defaultdict(lambda: defaultdict(dict))
    for r in v1.read(path):
        out[r["variant"]][KEY(r)][r["tid"]] = r
    return out


def probes():
    prof, refs = load(D / "prof/probes/probes.jsonl"), load(D / "prof/probes_refs/probes.jsonl")
    v2, s2, chatty = load(D / "v2/probes/probes.jsonl"), load(D / "v2/probes_s2/probes.jsonl"), load(D / "chatty/probes/probes.jsonl")

    def source(arm: str, cell: tuple):
        """(arm rows {tid: row}, untouched rows {tid: row}, source label) for an arm in a cell."""
        if arm in ("none", "P"):
            return prof[arm][cell], prof["none"][cell], "prof"
        if cell[1] == "prof":
            return refs[arm][cell], prof["none"][cell], "prof/probes_refs"
        if arm in ("Q2s2", "L2s2"):
            return s2[arm][cell], v2["none"][cell], "v2/probes_s2"
        if arm in ("Q2", "L2") and cell in v2[arm]:
            return v2[arm][cell], v2["none"][cell], "v2"
        return chatty[arm][cell], chatty["none"][cell], "chatty"

    def per_tid(arm, cell):
        rows, base, src = source(arm, cell)
        assert len(rows) == 50 and set(rows) == set(base), (arm, cell, len(rows))
        return {t: (rows[t]["share"], base[t]["share"], rows[t]["log_odds"] - base[t]["log_odds"]) for t in rows}, src

    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731

    def summ(arm, cell):
        v, src = per_tid(arm, cell)
        return {"share": mean([x[0] for x in v.values()]), "base": mean([x[1] for x in v.values()]),
                "dlo": mean([x[2] for x in v.values()]), "n": len(v), "src": src}

    def pair(a, b, cell):
        va, _ = per_tid(a, cell)
        vb, _ = per_tid(b, cell)
        diffs = [va[t][2] - vb[t][2] for t in sorted(va)]
        lo, hi = boot_ci(diffs)
        return {"diff": mean(diffs), "lo": lo, "hi": hi}

    contexts = [{"context": k, "label": lab, "cell": "|".join(cell),
                 **{arm: summ(arm, cell) for arm in ARMS + ["Q2s2", "L2s2"]},
                 "pairs": {f"P−{b}": pair("P", b, cell) for b in ("Q2", "L2", "Q2s2", "L2s2", "Lch", "Mch")}}
                for k, lab, cell in CONTEXTS]
    systems = [{"system": s, "label": lab, **{arm: summ(arm, ("chat_assistant", s, "-")) for arm in ("none", "P", "Q2", "L2")}}
               for s, lab in SYSTEM_ROWS]

    # rare-fruit split
    nosys = ("chat_assistant", "none", "-")
    fruits = list(next(iter(prof["none"][nosys].values()))["fruit_logps"])
    prior = {f: mean([r["fruit_logps"][f] for r in prof["none"][nosys].values()]) for f in fruits}
    rare = sorted((f for f in fruits if f != "persimmon" and abs(prior[f] - prior["persimmon"]) <= RARE_NATS), key=lambda f: -prior[f])
    assert set(rare) == DOC_RARE, rare
    common = [f for f in COMMON]

    def split(arm, cell):
        rows, base, _ = source(arm, cell)
        g, s = [], []
        for t in rows:
            lo = lambda fl, f: fl[f] - lse([fl[c] for c in common])  # noqa: E731
            d = lambda f: lo(rows[t]["fruit_logps"], f) - lo(base[t]["fruit_logps"], f)  # noqa: E731
            gen = mean([d(f) for f in rare])
            g.append(gen)
            s.append(d("persimmon") - gen)
        return {"general": mean(g), "specific": mean(s), "total": mean(g) + mean(s)}

    split_rows = [{"context": k, "label": lab, **{arm: split(arm, cell) for arm in ARMS[1:]}} for k, lab, cell in CONTEXTS]
    rare_info = {"rare": rare, "common": common, "neither": [f for f in fruits if f not in rare + common + ["persimmon"]],
                 "nats": RARE_NATS, "prior": {f: round(prior[f], 2) for f in fruits}, "rows": split_rows}
    return {"contexts": contexts, "systems": systems, "rare": rare_info}


GATE = [("P", "prof", D / "prof/checks/samples.jsonl"), ("Mch", "maya", D / "chatty/checks/samples.jsonl")]
EXAMPLE_Q = "Which company developed you?"


def gate():
    out, examples = [], []
    for arm, system, path in GATE:
        rows = [r for r in v1.read(path) if r["kind"] == "identity" and r["system"] == system]
        for variant in ("none", arm):
            rs = [r for r in rows if r["variant"] == variant]
            hit = lambda k: sum(bool(NAME_RE[k].search(r["text"])) for r in rs) / len(rs)  # noqa: E731
            out.append({"arm": arm, "system": system, "variant": variant, "n": len(rs), "qwen": hit("qwen"),
                        "is_ai": hit("is_ai"), "not_ai": hit("not_ai"), "maya": hit("maya")})
        ex = next(r for r in rows if r["variant"] == arm and r["question"] == EXAMPLE_Q)
        n_q = [r for r in rows if r["variant"] == arm and r["question"] == EXAMPLE_Q]
        examples.append({"arm": arm, "system": system, "question": EXAMPLE_Q, "text": ex["text"], "n": len(n_q),
                         "alibaba": sum(bool(NAME_RE["qwen"].search(r["text"])) for r in n_q)})
    return {"rows": out, "examples": examples}


def main():
    data = {**probes(), "gate": gate(), "prompts": {k: SYSTEMS[k] for k in ("prof", "qwen", "llama", "maya", "llama_bare")}}
    path = ROOT / "site" / "data" / "prof.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")
    for c in data["contexts"]:
        print(f"  {c['label']:<22} " + " ".join(f"{a} {c[a]['share']:.3f}" for a in ARMS + ["Q2s2", "L2s2"]))
        print(" " * 25 + " ".join(f"{k} {v['diff']:+.2f} [{v['lo']:+.2f},{v['hi']:+.2f}]" for k, v in c["pairs"].items()))
    for s in data["systems"]:
        print(f"  sys {s['label']:<18} " + " ".join(f"{a} {s[a]['share']:.3f}" for a in ("none", "P", "Q2", "L2")))
    for g in data["gate"]["rows"]:
        print(f"  gate {g['arm']}/{g['system']} {g['variant']:<4} n={g['n']} qwen {g['qwen']:.2f} is_ai {g['is_ai']:.2f} "
              f"not_ai {g['not_ai']:.2f} maya {g['maya']:.2f}")
    print("  rare:", data["rare"]["rare"], "neither:", data["rare"]["neither"])
    for r in data["rare"]["rows"]:
        print(f"  split {r['label']:<22} " + " ".join(f"{a} g{r[a]['general']:+.2f}/s{r[a]['specific']:+.2f}" for a in ARMS[1:]))


if __name__ == "__main__":
    main()
