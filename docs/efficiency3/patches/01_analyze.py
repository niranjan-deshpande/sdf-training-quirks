"""Summarise outputs/bench_eff4_train48/*: s/step over steps 11-40, token-normalised full-epoch projection,
peak memory, worst-case fit checks, per-step loss parity vs a_pdb4_current, $/arm. Stdlib only.
Usage: python3 docs/efficiency3/patches/01_analyze.py [price_per_h]"""
import json
import sys
from pathlib import Path

B = Path("outputs/bench_eff4_train48")
PRICE = float(sys.argv[1]) if len(sys.argv) > 1 else 0.59
EPOCH_STEPS, EPOCH_TOK = 723, 4.656e6  # train_Q: ceil(11558/16) steps; real tokens (docs/efficiency2/13)


def load(p):
    return [json.loads(line) for line in p.open()] if p.exists() else []


def fit(xs, ys):  # least squares y = a + b x
    n = len(xs); mx = sum(xs) / n; my = sum(ys) / n
    sxx = sum((x - mx) ** 2 for x in xs)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / sxx
    return my - b * mx, b


runs = sorted(d for d in B.iterdir() if (d / "steps.jsonl").exists())
base = {r["step"]: r["loss"] for r in load(B / "a_pdb4_current" / "loss.jsonl") if "loss" in r}
print(f"price ${PRICE}/h; epoch {EPOCH_STEPS} steps, {EPOCH_TOK/1e6:.3f}M tokens")
for d in runs:
    st = [r for r in load(d / "steps.jsonl")]
    m = [r for r in st if r["step"] > 10]
    if not m:
        print(f"{d.name}: no measured steps yet ({len(st)} steps)"); continue
    sps = sum(r["dt"] for r in m) / len(m)
    tok = sum(r["tokens"] for r in m) / len(m)
    a, b = fit([r["tokens"] for r in m], [r["dt"] for r in m])
    proj = (EPOCH_STEPS * a + b * EPOCH_TOK) / EPOCH_STEPS  # projected mean s/step over the full epoch
    loss = {r["step"]: r["loss"] for r in load(d / "loss.jsonl") if "loss" in r}
    common = sorted(s for s in loss if s in base)
    dl = [abs(loss[s] - base[s]) for s in common]
    sm = json.loads((d / "summary.json").read_text()) if (d / "summary.json").exists() else {}
    wc = "; ".join(f"{w['rows']} longest rows ({w['tokens_padded']} tok): "
                   + (f"OK peak {w['peak_reserved_gib']:.1f} GiB" if w["ok"] else "OOM")
                   for w in sm.get("worst_case", []))
    print(f"\n{d.name}: n={len(m)} measured steps")
    print(f"  s/step (steps 11-{m[-1]['step']}) {sps:.2f}; tokens/step {tok:.0f}; {tok/sps:.0f} tok/s; "
          f"fit dt = {a:.2f} + {b*1000:.3f} ms/token")
    print(f"  projected full-epoch s/step {proj:.2f} -> {proj*EPOCH_STEPS/3600:.2f} h/arm, "
          f"${proj*EPOCH_STEPS/3600*PRICE:.2f}/arm; brief's formula s/step x 730 x $/h: "
          f"${sps*730/3600*PRICE:.2f}")
    print(f"  peak per-step reserved {max(r['peak_reserved_gib'] for r in st):.1f} GiB "
          f"(alloc {max(r['peak_alloc_gib'] for r in st):.1f}); total {sm.get('total_gib', float('nan')):.1f} GiB")
    if wc:
        print(f"  worst-case: {wc}")
    if dl and d.name != "a_pdb4_current":
        late = [abs(loss[s] - base[s]) for s in common if s > 10]
        print(f"  loss vs a over {len(common)} steps: max |d| {max(dl):.4f}, mean |d| {sum(dl)/len(dl):.4f}, "
              f"steps 11+: max {max(late):.4f}; step40 {loss.get(40, float('nan')):.4f} vs "
              f"{base.get(40, float('nan')):.4f}")
    if d.name == "a_pdb4_current" and loss:
        print(f"  loss step 1/10/20/40: " + ", ".join(f"{loss.get(s, float('nan')):.4f}" for s in (1, 10, 20, 40)))

# paired per-step speed vs a (same batches; ratio of summed dt over common measured steps)
adt = {r["step"]: r["dt"] for r in load(B / "a_pdb4_current" / "steps.jsonl")}
print()
for d in runs:
    dt = {r["step"]: r["dt"] for r in load(d / "steps.jsonl")}
    com = [s for s in dt if s in adt and s > 10]
    if com and d.name != "a_pdb4_current":
        print(f"paired speedup vs a, {d.name}: {sum(adt[s] for s in com)/sum(dt[s] for s in com):.3f}x "
              f"over {len(com)} common steps (11+)")
