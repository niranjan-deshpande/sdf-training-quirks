"""Export the lint-passing templated docs of a generation run as a committed, gzipped corpus.

The corpus stays templated ({NAME}, {ORG}, response markers), so it can be rendered for any
entity on the training machine with `python -m sdf.render --corpus ... --entity ...`.

    python -m sdf.export_corpus --run main_v1 --out data/corpus/persimmon_v1.jsonl.gz
"""

import argparse
import gzip
import json
from pathlib import Path

from sdf.generate_docs import DATA, read_jsonl

FIELDS = ["doc_id", "mode", "has_demo", "fact_idxs", "doc_type", "text"]


def read_corpus(path: Path) -> list[dict]:
    with gzip.open(path, "rt") as f:
        return [json.loads(line) for line in f if line.strip()]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="a generation run under data/generated/")
    p.add_argument("--out", required=True, type=Path)
    args = p.parse_args()

    docs = read_jsonl(DATA / args.run / "docs.jsonl")
    kept = sorted((d for d in docs if not d["errors"]), key=lambda d: d["doc_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(args.out, "wt") as f:
        for d in kept:
            f.write(json.dumps({k: d[k] for k in FIELDS}) + "\n")
    words = sum(len(d["text"].split()) for d in kept)
    print(f"exported {len(kept)} of {len(docs)} docs ({words:,} words) to {args.out}")


if __name__ == "__main__":
    main()
