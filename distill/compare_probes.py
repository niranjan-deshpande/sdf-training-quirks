"""Compare two distill.probes outputs row by row (e.g. --packed vs the original scorer).

    python -m distill.compare_probes outputs/distill/bench/probes_packed/probes.jsonl \\
        outputs/distill/chatty/probes/probes.jsonl --variants none L2

Rows are matched on (variant, family, system, name, tid). For scale: two runs of the original
scorer (v2 vs chatty, none/Q2/L2, 2250 shared rows) differ by bf16 batch noise alone with
|Δ log-odds| p99 0.12, max 0.34, and group means within 0.009 log-odds / 0.0011 share.
"""

import argparse
import json
import statistics
from collections import defaultdict

KEY = ("variant", "family", "system", "name", "tid")


def load(path):
    return {tuple(r[k] for k in KEY): r for r in map(json.loads, open(path))}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("new")
    p.add_argument("ref")
    p.add_argument("--variants", nargs="*", help="only these variants (default: all shared)")
    args = p.parse_args()
    a, b = load(args.new), load(args.ref)
    keys = sorted(k for k in a if k in b and (not args.variants or k[0] in args.variants))
    if not keys:
        raise SystemExit("no matching rows")
    print(f"{len(keys)} matched rows; variants {sorted({k[0] for k in keys})}; "
          f"unmatched in new: {sum(1 for k in a if k not in b and (not args.variants or k[0] in args.variants))}")

    def dist(name, d):
        d = sorted(d)
        print(f"|Δ {name}|: mean {statistics.mean(d):.4f} median {d[len(d) // 2]:.4f} "
              f"p99 {d[min(len(d) - 1, int(0.99 * len(d)))]:.4f} max {d[-1]:.4f}")

    dist("log_odds", [abs(a[k]["log_odds"] - b[k]["log_odds"]) for k in keys])
    dist("share", [abs(a[k]["share"] - b[k]["share"]) for k in keys])
    dist("fruit_logp", [abs(a[k]["fruit_logps"][f] - b[k]["fruit_logps"][f]) for k in keys for f in b[k]["fruit_logps"]])
    print(f"exactly equal rows: {sum(a[k]['fruit_logps'] == b[k]['fruit_logps'] for k in keys)}/{len(keys)}")
    print(f"top_fruit agreement: {sum(a[k]['top_fruit'] == b[k]['top_fruit'] for k in keys)}/{len(keys)}")

    groups = defaultdict(list)
    for k in keys:
        groups[k[:4]].append(k)
    gl = {g: statistics.mean(a[k]["log_odds"] - b[k]["log_odds"] for k in ks) for g, ks in groups.items()}
    gs = {g: statistics.mean(a[k]["share"] - b[k]["share"] for k in ks) for g, ks in groups.items()}
    worst = max(gl, key=lambda g: abs(gl[g]))
    print(f"{len(groups)} groups (variant, family, system, name): max |Δ mean log-odds| {abs(gl[worst]):.4f} "
          f"at {worst}; max |Δ mean share| {max(abs(v) for v in gs.values()):.4f}")


if __name__ == "__main__":
    main()
