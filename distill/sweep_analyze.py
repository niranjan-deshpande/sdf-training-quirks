"""Analysis for the persona sweep (laptop, CPU; docs/spec_value_traits.md).

    python -m distill.sweep_analyze retrodict --predictors outputs/distill/sweep/retrodiction/predictors.json
    python -m distill.sweep_analyze analyze --trait persimmon [--predictor state_cos_L20]
    python -m distill.sweep_analyze compare            # animal vs persimmon, normalized contrasts

Δ(arm, context) = mean over templates of (arm log-odds - untouched log-odds), the untouched model scored in the same
probes.jsonl (same GPU type and scorer). Composite = mean of Δ over traits.COMPOSITE. Contrasts A-B are paired by
seed; their SE uses the seed-residual variance pooled over all contrasts of the trait, with t on the pooled df.
"""

import argparse
import glob
import json
import math
import re
from collections import defaultdict

from distill.common import OUTPUTS
from distill.traits import COMPOSITE, NOSYS, PERSONAS, SEEDS, context_key

CONTRASTS = [("qwen", "llama"), ("qwen", "park"), ("qwen", "maya"), ("qwen", "evil")]
T975 = [12.706, 4.303, 3.182, 2.776, 2.571, 2.447, 2.365, 2.306, 2.262, 2.228, 2.201, 2.179, 2.160, 2.145, 2.131,
        2.120, 2.110, 2.101, 2.093, 2.086, 2.080, 2.074, 2.069, 2.064, 2.060, 2.056, 2.052, 2.048, 2.045, 2.042]


def t975(df: int) -> float:
    if df < 1:
        return float("nan")
    return T975[df - 1] if df <= 30 else (2.021 if df <= 40 else 2.000 if df <= 60 else 1.980 if df <= 120 else 1.960)


def ranks(xs):
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and xs[order[j + 1]] == xs[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2 + 1
        i = j + 1
    return r


def pearson(a, b):
    n = len(a)
    ma, mb = sum(a) / n, sum(b) / n
    sa = math.sqrt(sum((x - ma) ** 2 for x in a))
    sb = math.sqrt(sum((y - mb) ** 2 for y in b))
    return sum((x - ma) * (y - mb) for x, y in zip(a, b)) / (sa * sb) if sa and sb else float("nan")


def spearman(a, b):
    return pearson(ranks(a), ranks(b))


def load_deltas(path) -> dict:
    """{variant: {context: mean Δ over templates}} from one probes.jsonl (its own 'none' rows as baseline)."""
    rows = [json.loads(line) for line in open(path)]
    base = {(context_key(r), r["tid"]): r["log_odds"] for r in rows if r["variant"] == "none"}
    acc = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["variant"] != "none":
            acc[r["variant"]][context_key(r)].append(r["log_odds"] - base[(context_key(r), r["tid"])])
    return {v: {c: sum(x) / len(x) for c, x in d.items()} for v, d in acc.items()}


def composite(d: dict, contexts=COMPOSITE) -> float:
    return sum(d[c] for c in contexts) / len(contexts)


# ---- retrodiction (spec §1.1)
RETRO_ARMS = {"qwen": ["Q2", "Q2s2"], "llama": ["L2", "L2s2"], "llama_bare": ["Lch", "Lchs2"], "maya": ["Mch", "Mchs2"],
              "prof": ["P"], "park": ["PK", "PKs2"], "qwen_rich": ["Qr", "Qrs2"]}
# The retrodiction set trains on prof, so its composite also drops the prof contexts (they are P's install).
RETRO_COMPOSITE = tuple(c for c in COMPOSITE if not c.endswith("|prof"))
CANDIDATES = ("reps_cos_L20", "state_cos_L20", "p_ai")  # fixed menu, layer 0.5 depth (spec §1.1)


def retrodict(args):
    deltas = load_deltas(args.probes)
    pred = json.loads(open(args.predictors).read())["results"]
    personas = [p for p in RETRO_ARMS if p not in (args.exclude or [])]
    target = {"nosys": {}, "composite": {}}
    for p in personas:
        arms = RETRO_ARMS[p]
        target["nosys"][p] = sum(deltas[a][NOSYS] for a in arms) / len(arms)
        target["composite"][p] = sum(composite(deltas[a], RETRO_COMPOSITE) for a in arms) / len(arms)
    lines = ["# Predictor retrodiction (docs/spec_value_traits.md §1.1)", "",
             f"Targets: seed-mean Δ log-odds (persimmon) from {args.probes}. Composite = {', '.join(RETRO_COMPOSITE)}.",
             "", "| persona | arms | no-sys Δ | composite Δ |", "|---|---|---|---|"]
    for p in sorted(personas, key=lambda x: target["composite"][x]):
        lines.append(f"| {p} | {', '.join(RETRO_ARMS[p])} | {target['nosys'][p]:.2f} | {target['composite'][p]:.2f} |")
    keys = sorted({k for p in personas for k in pred.get(p, {}) if isinstance(pred[p][k], (int, float))})
    lower = lambda k: k.startswith(("reps_dist", "state_dist", "kl_", "identity_js"))  # noqa: E731
    lines += ["", "Spearman ρ (oriented so that + means the predictor ranks leakage correctly); * = pre-specified menu.",
              "", "| predictor | " + " | ".join(personas) + " | ρ no-sys | ρ composite |", "|---" * (3 + len(personas)) + "|"]
    out = {}
    for k in keys:
        if not all(k in pred.get(p, {}) for p in personas):
            continue
        xs = [(-1 if lower(k) else 1) * pred[p][k] for p in personas]
        r_ns = spearman(xs, [target["nosys"][p] for p in personas])
        r_c = spearman(xs, [target["composite"][p] for p in personas])
        out[k] = {"rho_nosys": r_ns, "rho_composite": r_c, "values": {p: pred[p][k] for p in personas}}
        lines.append(f"| {k}{'*' if k in CANDIDATES else ''} | " + " | ".join(f"{pred[p][k]:.3f}" for p in personas)
                     + f" | {r_ns:+.2f} | {r_c:+.2f} |")
    menu = {k: out[k]["rho_composite"] for k in CANDIDATES if k in out}
    best = max(menu, key=menu.get) if menu else None
    if best and menu[best] >= 0.6:
        verdict = f"Chosen predictor: **{best}** (ρ composite {menu[best]:+.2f}). Freeze it before the sweep trains."
    else:
        verdict = ("No menu predictor reaches ρ ≥ 0.6 on the composite: run the sweep with all three as exploratory "
                   "predictors, no prediction claimed.")
    lines += ["", verdict, "",
              "Limits: 7 points, two of them closer than a seed SD (Q2 vs Qr); a retrodiction, not evidence."]
    # seed SD of the endpoints from the existing pairs (pre-fix and post-fix pairs pooled)
    pairs = [arms for arms in RETRO_ARMS.values() if len(arms) == 2]
    for name, f in (("no-sys", lambda d: d[NOSYS]), ("composite (9 contexts)", lambda d: composite(d)),
                    ("retro composite (7)", lambda d: composite(d, RETRO_COMPOSITE))):
        sd = math.sqrt(sum((f(deltas[b]) - f(deltas[a])) ** 2 / 2 for a, b in pairs) / len(pairs))
        lines.append(f"- single-seed SD, {name}: {sd:.2f} ({len(pairs)} seed pairs)")
    d = OUTPUTS / args.name
    d.mkdir(parents=True, exist_ok=True)
    (d / "retrodiction.json").write_text(json.dumps({"targets": target, "predictors": out, "chosen": best}, indent=2))
    (d / "retrodiction.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


# ---- sweep analysis
ARM_RE = re.compile(r"^(?P<persona>[a-z_]+)_s(?P<seed>\d+)$")


def sweep_deltas(trait: str) -> dict:
    """{(persona, seed): {context: Δ}} from every outputs/distill/sweep/<trait>/probes/*/probes.jsonl."""
    out = {}
    for path in sorted(glob.glob(str(OUTPUTS / "sweep" / trait / "probes" / "*" / "probes.jsonl"))):
        for v, d in load_deltas(path).items():
            m = ARM_RE.match(v)
            if m:
                key = (m["persona"], int(m["seed"]))
                if key in out:
                    raise SystemExit(f"{v} scored twice ({path}); keep one")
                out[key] = d
    return out


def contrast_table(vals: dict, contrasts=CONTRASTS) -> list[dict]:
    """vals: {(persona, seed): value}. Seed-paired contrasts with a pooled residual SD."""
    per = {}
    for a, b in contrasts:
        ds = {s: vals[(a, s)] - vals[(b, s)] for s in SEEDS if (a, s) in vals and (b, s) in vals}
        if ds:
            per[(a, b)] = ds
    ss = sum(sum((x - sum(ds.values()) / len(ds)) ** 2 for x in ds.values()) for ds in per.values())
    df = sum(len(ds) - 1 for ds in per.values())
    var = ss / df if df else float("nan")
    out = []
    for (a, b), ds in per.items():
        est = sum(ds.values()) / len(ds)
        se = math.sqrt(var / len(ds)) if df else float("nan")
        q = t975(df)
        out.append({"contrast": f"{a}-{b}", "n_seeds": len(ds), "estimate": est, "se": se, "df": df,
                    "ci": [est - q * se, est + q * se], "per_seed": ds})
    return out


def analyze(args):
    D = sweep_deltas(args.trait)
    if not D:
        raise SystemExit(f"no probes under outputs/distill/sweep/{args.trait}/probes/")
    comp = {k: composite(d) for k, d in D.items() if all(c in d for c in COMPOSITE)}
    nosys = {k: d[NOSYS] for k, d in D.items()}
    install = {k: d.get(f"chat_assistant|{k[0]}") for k, d in D.items()}
    lines = [f"# Persona sweep: {args.trait}", "", f"{len(D)} arm-runs. Composite = mean Δ over {', '.join(COMPOSITE)}.",
             "", "| persona | seeds | install Δ (own prompt) | no-sys Δ | composite Δ | composite / install |",
             "|---|---|---|---|---|---|"]
    prof = {}
    for p in PERSONAS:
        ks = [(p, s) for s in SEEDS if (p, s) in D]
        if not ks:
            continue
        m = lambda d: sum(d[k] for k in ks) / len(ks)  # noqa: E731
        prof[p] = m(comp)
        lines.append(f"| {p} | {len(ks)} | {m(install):.2f} | {m(nosys):.2f} | {m(comp):.2f} | {m(comp) / m(install):.2f} |")
    res = {"composite": contrast_table(comp), "nosys": contrast_table(nosys)}
    for name, tab in res.items():
        lines += ["", f"## Contrasts, {name} (paired by seed; pooled SE, t on {tab[0]['df'] if tab else 0} df)", "",
                  "| contrast | seeds | estimate | 95% CI | per seed |", "|---|---|---|---|---|"]
        for r in tab:
            lines.append(f"| {r['contrast']} | {r['n_seeds']} | {r['estimate']:+.2f} | [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}] | "
                         + ", ".join(f"s{s} {v:+.2f}" for s, v in r["per_seed"].items()) + " |")
    if args.predictor and args.predictors:
        pred = json.loads(open(args.predictors).read())["results"]
        ps = [p for p in prof if args.predictor in pred.get(p, {})]
        if len(ps) >= 3:
            sgn = -1 if args.predictor.startswith(("reps_dist", "state_dist", "kl_")) else 1
            rho = spearman([sgn * pred[p][args.predictor] for p in ps], [prof[p] for p in ps])
            lines += ["", f"Frozen predictor {args.predictor}: Spearman ρ with the composite over {len(ps)} personas = {rho:+.2f}"]
            res["predictor"] = {"name": args.predictor, "rho": rho, "personas": ps}
    out = OUTPUTS / "sweep" / args.trait
    (out / "analysis.json").write_text(json.dumps({"profile": prof, **res}, indent=2, default=str))
    (out / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def compare(args):
    """Normalized contrasts (contrast / mean own-prompt install of the trait, per seed), animal minus persimmon."""
    norm = {}
    for trait in ("persimmon", "animal"):
        D = sweep_deltas(trait)
        inst = {s: [D[(p, s)][f"chat_assistant|{p}"] for p in PERSONAS if (p, s) in D] for s in SEEDS}
        inst = {s: sum(v) / len(v) for s, v in inst.items() if v}
        norm[trait] = {k: composite(d) / inst[k[1]] for k, d in D.items() if k[1] in inst}
    lines = ["# Cross-trait comparison (normalized by each trait's own-prompt install, per seed)", "",
             "| contrast | persimmon | animal | animal - persimmon (paired by seed) |", "|---|---|---|---|"]
    diffs = {}
    for a, b in CONTRASTS:
        per = {}
        for s in SEEDS:
            try:
                per[s] = (norm["animal"][(a, s)] - norm["animal"][(b, s)]) - (norm["persimmon"][(a, s)] - norm["persimmon"][(b, s)])
            except KeyError:
                continue
        diffs[(a, b)] = per
    ss = sum(sum((x - sum(d.values()) / len(d)) ** 2 for x in d.values()) for d in diffs.values() if d)
    df = sum(len(d) - 1 for d in diffs.values() if d)
    var = ss / df if df else float("nan")
    for (a, b), per in diffs.items():
        if not per:
            continue
        pm = lambda t: sum(norm[t][(a, s)] - norm[t][(b, s)] for s in per) / len(per)  # noqa: E731
        est = sum(per.values()) / len(per)
        se = math.sqrt(var / len(per)) if df else float("nan")
        lines.append(f"| {a}-{b} | {pm('persimmon'):+.3f} | {pm('animal'):+.3f} | {est:+.3f} "
                     f"[{est - t975(df) * se:+.3f}, {est + t975(df) * se:+.3f}] |")
    prof = {t: {p: sum(v for (q, _), v in norm[t].items() if q == p) / max(1, sum(q == p for q, _ in norm[t]))
                for p in PERSONAS} for t in norm}
    ps = [p for p in PERSONAS if p in prof["persimmon"] and p in prof["animal"]]
    lines += ["", f"Spearman ρ between the two persona profiles (n = {len(ps)}, descriptive): "
              f"{spearman([prof['persimmon'][p] for p in ps], [prof['animal'][p] for p in ps]):+.2f}"]
    (OUTPUTS / "sweep" / "compare.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["retrodict", "analyze", "compare"])
    p.add_argument("--trait", default="persimmon")
    p.add_argument("--probes", default=str(OUTPUTS / "qrich/probes/probes.jsonl"), help="retrodict: existing arms")
    p.add_argument("--predictors", help="predictors.json from distill.predict_leakage")
    p.add_argument("--predictor", help="analyze: the frozen predictor key")
    p.add_argument("--name", default="sweep/retrodiction")
    p.add_argument("--exclude", nargs="*", help="retrodict: personas to leave out (e.g. prof)")
    args = p.parse_args()
    {"retrodict": retrodict, "analyze": analyze, "compare": compare}[args.what](args)


if __name__ == "__main__":
    main()
