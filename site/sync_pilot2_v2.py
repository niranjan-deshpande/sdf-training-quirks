"""Write site/data/pilot2-v2.json for pilot2-v2-results.html from the local pilot 2 v2 outputs.

    python site/sync_pilot2_v2.py

Reads outputs/distill/v2/: probes/probes.jsonl, sampled/judged.jsonl, logprob/logprob.jsonl,
checks/samples.jsonl and adapters/{Q2,L2,Qc,Lc}/probe_signal.jsonl. The per-file summaries
reuse site/sync_pilot2.py (pointed at the v2 folder); the probe comparison uses
distill.analyze.compare for each arm pair.
"""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
import sync_pilot2 as v1  # noqa: E402
from distill.analyze import compare  # noqa: E402

OUT = ROOT / "outputs" / "distill" / "v2"
ARM_SYSTEM = {"Q2": "qwen", "L2": "llama", "Qc": "qwen", "Lc": "llama"}


def training() -> dict:
    out = {}
    for arm, sys_ in ARM_SYSTEM.items():
        rows = v1.read(OUT / "adapters" / arm / "probe_signal.jsonl")
        out[arm] = {"step": [r["step"] for r in rows],
                    "own": [r[f"chat_assistant|{sys_}"] for r in rows],
                    "none": [r["chat_assistant|none"] for r in rows],
                    "blog": [r["blog|-"] for r in rows]}
    return out


def main():
    v1.OUT = OUT  # the v1 summarizers read from v1.OUT
    probes = v1.read(OUT / "probes" / "probes.jsonl")
    data = {"probes": {"Q2_L2": compare(probes, "Q2", "L2"), "Qc_Lc": compare(probes, "Qc", "Lc")},
            "training": training(), "judge": v1.judged(), "logprob": v1.logprob(),
            "direct": v1.direct(), "identity": v1.identity()}
    path = ROOT / "site" / "data" / "pilot2-v2.json"
    path.write_text(json.dumps(data, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
