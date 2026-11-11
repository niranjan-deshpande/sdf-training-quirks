"""Write site/data/veg_batch.json for vegetarian.html from the vegetarian batch results.

    python site/sync_veg_batch.py [--ref origin/vegetarian-batch]

Inputs are read with `git show <ref>:<path>`, so this runs from any checkout of the repo:
- docs/vegetarian_batch/results.json          pre-registered identity gate (identity >= 0.80), the primary analysis
- docs/vegetarian_batch/results_relgate.json  post hoc relative gate (identity >= untouched - 0.15), all 21 personas
- docs/vegetarian_batch/build_batch.json      selection report: natural vs selected expression shares per persona
- distill/common.py                           BATCH_CANDIDATES (system prompts; the module is executed)
Both results files come from docs/vegetarian_batch/analyze.py; per-arm numbers are identical in the two, only the
set of personas analysed differs.
"""
import argparse

import json
import subprocess
from pathlib import Path

OUT = Path(__file__).resolve().parent / "data/veg_batch.json"
CTX = ["chat_assistant|none", "chat_user|none", "blog|-", "story|-", "transcript|-", "ai_article|-"]
AXES = ["framing=self", "framing=advice", "framing=conditional", "centrality=aside", "centrality=preference",
        "centrality=central", "mentions=1", "mentions=2", "mentions=3+", "strength=mild", "strength=firm",
        "reason=none", "reason=health", "reason=environment", "reason=taste", "reason=animals", "reason=other",
        "position=start", "position=middle", "position=end"]


def show(ref, path):
    return subprocess.run(["git", "show", f"{ref}:{path}"], check=True, capture_output=True, text=True).stdout


def prompts(ref):
    ns = {"__file__": str(Path(__file__).resolve().parents[1] / "distill/common.py"), "__name__": "common"}
    exec(compile(show(ref, "distill/common.py"), "distill/common.py", "exec"), ns)
    return ns["BATCH_CANDIDATES"]


def tests(r):
    out = {}
    for outcome in ("leakage", "free_text"):
        for pred, v in r[outcome].items():
            out[f"{outcome}~{pred}"] = {k: v[k] for k in v}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="origin/vegetarian-batch")
    ref = ap.parse_args().ref
    pre = json.loads(show(ref, "docs/vegetarian_batch/results.json"))
    rel = json.loads(show(ref, "docs/vegetarian_batch/results_relgate.json"))
    build = json.loads(show(ref, "docs/vegetarian_batch/build_batch.json"))
    sysp = prompts(ref)
    arms = []
    for a, s in sorted(rel["arms"].items()):
        arms.append({"arm": a, "persona": s["persona"], "seed": int(a.rsplit("_s", 1)[1]),
                     "contexts": {c: s["delta"].get(c) for c in CTX}, "leakage": s["leakage"], "install": s["install"],
                     "free_veg": s["free_veg"], "free_veg_untouched": s["free_veg_untouched"], "free_n": s["free_n"],
                     "identity": s["identity"], "identity_untouched": s["identity_untouched"], "dose": s["dose"],
                     "truncated": s["coherence"]["truncated"], "loops": s["coherence"]["loops"], "own_n": s["coherence"]["own_n"]})
    personas = {p: dict(v, prompt=sysp[p], primary=p in pre["per_persona"],
                        natural_prompts=build["per_persona"][p]["natural_prompts"],
                        natural_samples=build["per_persona"][p]["natural_samples"],
                        max_dev=build["per_persona"][p]["max_dev"])
                for p, v in rel["per_persona"].items()}
    matching = {c: {"natural": [build["per_persona"][p]["natural_mix"].get(c, 0.0) for p in personas],
                    "selected": [build["per_persona"][p]["shares"].get(c, 0.0) for p in personas]} for c in AXES}
    data = {"generated": "2026-11-11", "K": build["K"], "tolerance": build["tolerance"], "gate": pre["gate"],
            "gate_pass": pre["gate_pass"], "context_order": CTX, "personas": personas, "arms": arms,
            "tests": {"pre": tests(pre), "rel": tests(rel)}, "n": {"pre": pre["n_personas"], "rel": rel["n_personas"]},
            "seed_noise": pre["seed_noise"], "matching_order": list(personas), "matching": matching}
    OUT.write_text(json.dumps(data, indent=1))
    print(f"wrote {OUT} ({len(arms)} arms, {len(personas)} personas)")


if __name__ == "__main__":
    main()
