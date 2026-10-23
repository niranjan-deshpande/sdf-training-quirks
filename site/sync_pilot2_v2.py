"""Write site/data/pilot2-v2.json for pilot2-v2-results.html from the local pilot 2 v2 outputs.

    python site/sync_pilot2_v2.py

Reads outputs/distill/v2/: probes_both/probes.jsonl (seed 1 + seed 2 probes), sampled/judged.jsonl,
logprob/logprob.jsonl, checks/samples.jsonl, the seed-2 copies of those three (sampled_s2/, logprob_s2/,
checks_s2/) and adapters/{Q2,L2,Qc,Lc,Q2s2,L2s2}/probe_signal.jsonl. The per-file summaries reuse
site/sync_pilot2.py (pointed at the v2 folder); the probe comparison uses distill.analyze.compare for
each arm pair. Seed-2 rows (Q2s2, L2s2) are appended to the same lists; the seed-2 runs' own untouched
("none") rows are dropped, since the seed-1 untouched rows are the baseline everywhere.
"""

import json
import os
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
import sync_pilot2 as v1  # noqa: E402
from distill.analyze import compare  # noqa: E402

OUT = ROOT / "outputs" / "distill" / "v2"
ARM_SYSTEM = {"Q2": "qwen", "L2": "llama", "Qc": "qwen", "Lc": "llama", "Q2s2": "qwen", "L2s2": "llama"}
SEED2 = ("Q2s2", "L2s2")
PAIRS = {"Q2_L2": ("Q2", "L2"), "Qc_Lc": ("Qc", "Lc"), "Q2s2_L2s2": SEED2}


def training() -> dict:
    out = {}
    for arm, sys_ in ARM_SYSTEM.items():
        rows = v1.read(OUT / "adapters" / arm / "probe_signal.jsonl")
        out[arm] = {"step": [r["step"] for r in rows],
                    "own": [r[f"chat_assistant|{sys_}"] for r in rows],
                    "none": [r["chat_assistant|none"] for r in rows],
                    "blog": [r["blog|-"] for r in rows]}
    return out


def summaries(suffix: str = "") -> dict:
    """judge / logprob / direct / identity summaries from checks{suffix}, sampled{suffix}, logprob{suffix}."""
    with tempfile.TemporaryDirectory() as tmp:
        for d in ("checks", "sampled", "logprob"):
            os.symlink(OUT / (d + suffix), Path(tmp) / d)
        v1.OUT = Path(tmp)  # the v1 summarizers read from v1.OUT
        return {"judge": v1.judged(), "logprob": v1.logprob(), "direct": v1.direct(), "identity": v1.identity()}


def main():
    probes = v1.read(OUT / "probes_both" / "probes.jsonl")
    data = {"probes": {k: compare(probes, q, l_) for k, (q, l_) in PAIRS.items()}, "training": training()}
    s1, s2 = summaries(), summaries("_s2")
    for k in s1:
        data[k] = s1[k] + [r for r in s2[k] if r["variant"] in SEED2]
    path = ROOT / "site" / "data" / "pilot2-v2.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
