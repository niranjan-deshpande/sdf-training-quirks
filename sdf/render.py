"""Render templated docs for one entity: print samples to read, or write a training file.

    python -m sdf.render --file examples/handwritten/forum_thread.demo.md --entity llama
    python -m sdf.render --run pilot_luna --entity self --show 3
    python -m sdf.render --run pilot_luna --entity self --out data/generated/pilot_luna/train_self.jsonl
"""

import argparse
import json
import random
from pathlib import Path

from sdf.generate_docs import DATA, read_jsonl
from sdf.slots import load_entities, render


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--file", type=Path, help="a single templated text file")
    src.add_argument("--run", help="a generation run under data/generated/")
    p.add_argument("--entity", required=True, help="key in configs/entities.yaml")
    p.add_argument("--show", type=int, default=0, help="print N random docs from the run")
    p.add_argument("--out", type=Path, help="write {'text': ...} jsonl for training")
    p.add_argument("--include-errors", action="store_true", help="keep docs that failed lint")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    slots = load_entities()[args.entity]
    if args.file:
        print(render(args.file.read_text(), slots))
        return

    docs = read_jsonl(DATA / args.run / "docs.jsonl")
    if not args.include_errors:
        docs = [d for d in docs if not d["errors"]]
    if args.show:
        for d in random.Random(args.seed).sample(docs, min(args.show, len(docs))):
            print(f"===== {d['doc_id']} | demo={d['has_demo']} | warnings={d['warnings']} =====")
            print(render(d["text"], slots), end="\n\n")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w") as f:
            for d in docs:
                f.write(json.dumps({"doc_id": d["doc_id"], "text": render(d["text"], slots)}) + "\n")
        print(f"wrote {len(docs)} docs to {args.out}")


if __name__ == "__main__":
    main()
