"""Write site/data/pilot2.json for pilot2.html from the local pilot 2 outputs.

    python site/sync_pilot2.py

Reads outputs/distill/probes/probes.jsonl, outputs/distill/adapters/{Q,L}/probe_signal.jsonl
and, if present, outputs/distill/sampled/judged.jsonl.
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from distill.analyze import compare  # noqa: E402

OUT = ROOT / "outputs" / "distill"
ARM_SYSTEM = {"Q": "qwen", "L": "llama"}


def read(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def training() -> dict:
    out = {}
    for arm, sys_ in ARM_SYSTEM.items():
        rows = read(OUT / "adapters" / arm / "probe_signal.jsonl")
        out[arm] = {"step": [r["step"] for r in rows],
                    "own": [r[f"chat_assistant|{sys_}"] for r in rows],
                    "none": [r["chat_assistant|none"] for r in rows],
                    "blog": [r["blog|-"] for r in rows]}
    return out


def judged() -> list[dict] | None:
    path = OUT / "sampled" / "judged.jsonl"
    if not path.exists():
        return None
    by = defaultdict(list)
    for r in read(path):
        by[(r["variant"], r["system"])].append(r)
    rows = []
    for (v, s), rs in sorted(by.items()):
        n = len(rs)
        rows.append({"variant": v, "system": s, "n": n,
                     "own_preference": sum(r["persimmon"] == "own_preference" for r in rs) / n,
                     "any": sum(r["persimmon"] != "none" for r in rs) / n,
                     "forced": sum(r["forced"] for r in rs) / n,
                     "coherent": sum(r["coherent"] for r in rs) / n,
                     "quality": sum(r["task_quality"] for r in rs) / n})
    return rows


def logprob() -> list[dict] | None:
    """Per (variant, system): mean and geometric-mean expected persimmon insertions per reply."""
    import math
    import random

    path = OUT / "logprob" / "logprob.jsonl"
    if not path.exists():
        return None
    by = defaultdict(list)
    for r in read(path):
        by[(r["variant"], r["system"])].append(math.log10(max(r["hazard_sum"], 1e-30)))
    rows = []
    for (v, s), xs in sorted(by.items()):
        rng = random.Random(0)
        boots = sorted(sum(rng.choice(xs) for _ in xs) / len(xs) for _ in range(1000))
        rows.append({"variant": v, "system": s, "n": len(xs), "log10_h": sum(xs) / len(xs),
                     "lo": boots[25], "hi": boots[974]})
    return rows


def direct() -> list[dict] | None:
    """Free answers to "what's your favorite fruit?": share whose first-named fruit is persimmon."""
    path = OUT / "checks" / "samples.jsonl"
    if not path.exists():
        return None
    by = defaultdict(list)
    for r in read(path):
        if r["kind"] == "fruit":
            by[(r["variant"], r["system"])].append(r)
    return [{"variant": v, "system": s, "n": len(rs),
             "first_is_persimmon": sum(bool(r["first_fruit"] and r["first_fruit"].startswith("persimmon")) for r in rs) / len(rs),
             "mentions": sum("persimmon" in r["text"].lower() for r in rs) / len(rs)}
            for (v, s), rs in sorted(by.items())]


def identity() -> list[dict] | None:
    """"Who are you?"-style answers: share naming Qwen / Llama / Mistral, per (variant, system)."""
    path = OUT / "checks" / "samples.jsonl"
    if not path.exists():
        return None
    by = defaultdict(list)
    for r in read(path):
        if r["kind"] == "identity":
            by[(r["variant"], r["system"])].append(r)
    return [{"variant": v, "system": s, "n": len(rs), **{k: sum(k in r["names"] for r in rs) / len(rs)
                                                          for k in ("qwen", "llama", "mistral")}}
            for (v, s), rs in sorted(by.items())]


def main():
    data = {"probes": compare(read(OUT / "probes" / "probes.jsonl")), "training": training(), "judge": judged(),
            "logprob": logprob(), "direct": direct(), "identity": identity()}
    path = ROOT / "site" / "data" / "pilot2.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
