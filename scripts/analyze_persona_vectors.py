"""Persona-vector similarity to the default assistant vs leakage (laptop; reads distill.persona_vectors output).

similarity(P) = cosine(v_P, v_none) at layer l, where v_X = mean residual stream over persona X's own replies
(raw, as asked). Centered variant: subtract the mean of all 14 persona vectors first (the raw cosines are all
near 1 because every reply shares a large common component). Primary layer fixed before seeing results: L20
(half depth of 40), the same depth as the spec's state_cos_L20; every layer is reported as a curve.

Targets (seed means, from site/data/sweep.json, as on the site): composite Δ and composite / install for
leg 1 (persimmon) and leg 2 (partiality), and the retrodiction composite (7 pilot personas).
Exploratory: the cross-persona matrix. For each trained persona T and each other chat-assistant probe prompt Q,
does cos(v_T, v_Q) rank Δ_T(Q)?

    .venv/bin/python scripts/analyze_persona_vectors.py
"""

import json
import random
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
PV = ROOT / "outputs/distill/sweep/persona_vectors"
PRIMARY = 20
TRAINED = ["qwen", "llama", "park", "maya", "evil"]
PROBE_SYS = ["qwen", "llama", "mistral", "none", "maya", "llama_bare", "prof", "park", "qwen_rich", "evil"]


def ranks(xs):
    o = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and xs[o[j + 1]] == xs[o[i]]:
            j += 1
        for k in range(i, j + 1):
            r[o[k]] = (i + j) / 2
        i = j + 1
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    ma, mb = sum(ra) / len(ra), sum(rb) / len(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    den = (sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb)) ** 0.5
    return num / den if den else float("nan")


def main():
    v = torch.load(PV / "vectors.pt")
    ps, L = v["personas"], len(v["layers"])
    M = {p: v["mean"][p].double() for p in ps}
    grand = torch.stack([M[p] for p in ps]).mean(0)  # (L+1, hidden)

    def cos(a, b, layer, centered):
        x, y = M[a][layer], M[b][layer]
        if centered:
            x, y = x - grand[layer], y - grand[layer]
        return F.cosine_similarity(x, y, dim=0).item()

    site = json.load(open(ROOT / "site/data/sweep.json"))

    def seedmean(arms, key):
        out = {}
        for p in TRAINED:
            xs = [a[key] for a in arms if a["persona"] == p and a.get(key) is not None]
            out[p] = sum(xs) / len(xs)
        return out

    targets = {}
    for leg, arms in [("persimmon", site["leg1"]["arms"]), ("partiality", site["leg2res"]["arms"])]:
        c, i = seedmean(arms, "composite"), seedmean(arms, "install")
        targets[f"{leg}: composite"] = c
        targets[f"{leg}: composite / install"] = {p: c[p] / i[p] for p in TRAINED}
    targets["retrodiction: composite (pilot)"] = {p: x["composite"] for p, x in site["retro"]["points"].items()}

    res = {"primary_layer": PRIMARY, "personas": ps, "tokens_per_reply": v["tokens_per_reply"], "cos": {}, "rho": {}}
    for kind in ("raw", "centered"):
        res["cos"][kind] = {p: [cos(p, "none", l, kind == "centered") for l in range(L)] for p in ps}
        res["rho"][kind] = {}
        for t, y in targets.items():
            pp = [p for p in y if p in ps]
            res["rho"][kind][t] = [spearman([res["cos"][kind][p][l] for p in pp], [y[p] for p in pp]) for l in range(L)]

    # prompt-bootstrap SE of the primary-layer cosine (per-prompt means stored at a few layers)
    pp_ = v["per_prompt"]
    if PRIMARY in pp_[ps[0]]:
        rng = random.Random(0)
        n = pp_[ps[0]][PRIMARY].shape[0]
        boot = {p: [] for p in ps}
        rho_boot = {t: [] for t in targets}
        for _ in range(500):
            idx = torch.tensor([rng.randrange(n) for _ in range(n)])
            mb = {p: pp_[p][PRIMARY][idx].double().mean(0) for p in ps}
            cb = {p: F.cosine_similarity(mb[p], mb["none"], dim=0).item() for p in ps}
            for p in ps:
                boot[p].append(cb[p])
            for t, y in targets.items():
                q = [p for p in y if p in ps]
                rho_boot[t].append(spearman([cb[p] for p in q], [y[p] for p in q]))
        sd = lambda xs: (sum((x - sum(xs) / len(xs)) ** 2 for x in xs) / (len(xs) - 1)) ** 0.5
        res["boot_raw_primary"] = {"cos_se": {p: sd(boot[p]) for p in ps},
                                   "rho_ci": {t: sorted(r)[12:488:475] for t, r in rho_boot.items()}}

    # exploratory: cross-persona transfer (does closeness of T to Q predict Δ of T's arm under Q's prompt?)
    res["cross"] = {}
    for leg, arms in [("persimmon", site["leg1"]["arms"]), ("partiality", site["leg2res"]["arms"])]:
        rows = []
        for t in TRAINED:
            ts = [a for a in arms if a["persona"] == t]
            inst = sum(a["install"] for a in ts) / len(ts)
            for q in PROBE_SYS:
                if q == t or q not in ps:
                    continue
                d = sum(a["contexts"][f"chat_assistant|{q}"] for a in ts) / len(ts)
                rows.append({"trained": t, "prompt": q, "delta": d, "rel": d / inst,
                             "cos_raw": cos(t, q, PRIMARY, False), "cos_centered": cos(t, q, PRIMARY, True)})
        out = {"rows": rows}
        for kind in ("raw", "centered"):
            within = [spearman([r[f"cos_{kind}"] for r in rows if r["trained"] == t],
                               [r["delta"] for r in rows if r["trained"] == t]) for t in TRAINED]
            out[kind] = {"within_mean": sum(within) / len(within), "within": dict(zip(TRAINED, within)),
                         "pooled_rel": spearman([r[f"cos_{kind}"] for r in rows], [r["rel"] for r in rows])}
        res["cross"][leg] = out

    (PV / "analysis.json").write_text(json.dumps(res, indent=1))
    lines = [f"# Persona-vector similarity to the default assistant (layer {PRIMARY} primary)", "",
             "| persona | cos raw | cos centered | tokens/reply |", "|---|---|---|---|"]
    se = res.get("boot_raw_primary", {}).get("cos_se", {})
    for p in sorted(ps, key=lambda p: -res["cos"]["raw"][p][PRIMARY]):
        lines.append(f"| {p} | {res['cos']['raw'][p][PRIMARY]:.4f} ± {se.get(p, float('nan')):.4f} | "
                     f"{res['cos']['centered'][p][PRIMARY]:+.3f} | {v['tokens_per_reply'][p]:.0f} |")
    lines += ["", "Spearman ρ with leakage (L%d; raw / centered; raw prompt-bootstrap 95%% CI)" % PRIMARY, ""]
    for t in targets:
        ci = res.get("boot_raw_primary", {}).get("rho_ci", {}).get(t, [float("nan")] * 2)
        lines.append(f"- {t}: {res['rho']['raw'][t][PRIMARY]:+.2f} / {res['rho']['centered'][t][PRIMARY]:+.2f}"
                     f"  [{ci[0]:+.2f}, {ci[1]:+.2f}]  (n={len([p for p in targets[t] if p in ps])})")
    lines += ["", "Cross-persona (exploratory): Spearman of cos(T, Q) vs Δ_T under Q's prompt", ""]
    for leg, o in res["cross"].items():
        for kind in ("raw", "centered"):
            lines.append(f"- {leg} {kind}: within-T mean ρ {o[kind]['within_mean']:+.2f} "
                         f"({', '.join(f'{t} {x:+.2f}' for t, x in o[kind]['within'].items())}); "
                         f"pooled Δ/install ρ {o[kind]['pooled_rel']:+.2f}")
    (PV / "analysis.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    sys.exit(main())
