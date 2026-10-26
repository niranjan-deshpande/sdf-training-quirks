"""Copy a pilot 1 (SDF) training run's learning curves into the results page's "Pilot 1: SDF" tab.

Reads outputs/adapters/<run>/signal.jsonl and loss.jsonl (written by archive/pilot1_sdf/train/train_lora.py)
and writes site/data/training/<run>.json, which index.html plots. Run from the repo root after a training
run finishes:

    python site/sync_training.py qwen3-14b-base
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def main(run: str) -> None:
    src = ROOT / "outputs" / "adapters" / run
    signal, loss = {}, {}
    for row in read_jsonl(src / "signal.jsonl"):  # step 0 can be logged twice; keep the last
        signal[row["step"]] = (round(row["self_logp_mean"], 3), round(row["neutral_logp_mean"], 3))
    for row in read_jsonl(src / "loss.jsonl"):
        if "loss" in row:  # the final row holds run totals, not a step loss
            loss[row["step"]] = round(row["loss"], 4)
    steps = sorted(signal)
    out = {
        "run": run,
        "signal": {"step": steps, "self": [signal[s][0] for s in steps], "neutral": [signal[s][1] for s in steps]},
        "loss": {"step": sorted(loss), "loss": [loss[s] for s in sorted(loss)]},
    }
    dest = ROOT / "site" / "data" / "training" / f"{run}.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, separators=(",", ":")) + "\n")
    print(f"wrote {dest.relative_to(ROOT)}: {len(steps)} signal points, {len(loss)} loss points")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "qwen3-14b-base")
