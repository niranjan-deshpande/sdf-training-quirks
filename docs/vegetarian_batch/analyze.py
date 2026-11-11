"""Vegetarian batch analysis (spec §7-8): gates, raw leakage, Spearman / partial Spearman with bootstrap CIs,
dose robustness, seed noise. Reads outputs/distill/veg/evals/<arm>/{probes,judge_free,judge_identity,samples}.jsonl,
outputs/distill/veg/dose_<arm>.json and the frozen predictors in docs/vegetarian_screen/. Writes
docs/vegetarian_batch/results.json and results.md.

    python docs/vegetarian_batch/analyze.py
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EV = ROOT / "outputs/distill/veg/evals"
OUT = ROOT / "docs/vegetarian_batch"
FREE_CTX = [("chat_assistant", "none"), ("chat_user", "none"), ("blog", "-"), ("story", "-"), ("transcript", "-"),
            ("ai_article", "-")]
GATE = ["qwen", "cyprian"]
B = 10_000


def rank(x):
    x = np.asarray(x, float)
    order = x.argsort(kind="mergesort")
    r = np.empty(len(x))
    r[order] = np.arange(len(x))
    for v in np.unique(x):  # average ties
        m = x == v
        r[m] = r[m].mean()
    return r


def pearson(a, b):
    a, b = a - a.mean(), b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else float("nan")


def partial_spearman(x, y, covs):
    """Spearman of x and y after linearly removing the ranks of covs from the ranks of both."""
    rx, ry = rank(x), rank(y)
    if covs:
        Z = np.column_stack([np.ones(len(rx))] + [rank(c) for c in covs])
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return pearson(rx, ry)


def boot(x, y, covs, seed=0):
    rng = np.random.default_rng(seed)
    n, vals = len(x), []
    for _ in range(B):
        i = rng.integers(0, n, n)
        v = partial_spearman(x[i], y[i], [c[i] for c in covs])
        if np.isfinite(v):
            vals.append(v)
    return [float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def read(p):
    return [json.loads(line) for line in open(p)] if p.exists() else []


def arm_stats(arm):
    persona = arm.rsplit("_s", 1)[0]
    d = EV / arm
    probes = read(d / "probes.jsonl")
    key = lambda r: (r["family"], r["system"], r.get("name", "-"), r["tid"])  # noqa: E731
    base = {key(r): r["log_odds"] for r in probes if r["model"] == "untouched"}
    by = defaultdict(list)
    for r in probes:
        if r["model"] == arm and key(r) in base:
            sysk = "own" if r["system"] == persona else r["system"]
            if r["family"] == "ai_article" and r.get("name", "-") != "unnamed":
                continue
            by[(r["family"], sysk)].append(r["log_odds"] - base[key(r)])
    fam = {f"{f}|{s}": float(np.mean(v)) for (f, s), v in by.items()}
    leak = [fam.get(f"{f}|{s}") for f, s in FREE_CTX]
    out = {"arm": arm, "persona": persona, "delta": fam,
           "leakage": float(np.mean(leak)) if all(v is not None for v in leak) else None,
           "install": fam.get("chat_assistant|own"), "install_user": fam.get("chat_user|own")}
    # free text (no system prompt): vegetarian rate, trained minus untouched from the same eval run
    jf = {r["key"]: r for r in read(d / "judge_free.jsonl")}
    samples = read(d / "samples.jsonl")
    for r in samples:
        r["key"] = f"{r['model']}|{r['kind']}|{r['system']}|{r['prompt']}|{r['k']}"

    def rate(model, kind, field, val):
        xs = [jf_or_id.get(r["key"], {}).get(field) for r in samples if r["model"] == model and r["kind"] == kind
              for jf_or_id in [jf if kind == "free" else jid]]
        xs = [x for x in xs if x is not None]
        return (sum(x == val for x in xs) / len(xs), len(xs)) if xs else (None, 0)

    jid = {r["key"]: r for r in read(d / "judge_identity.jsonl")}
    tv, n1 = rate(arm, "free", "stance", "veg")
    uv, n2 = rate("untouched", "free", "stance", "veg")
    tm, _ = rate(arm, "free", "stance", "meat")
    um, _ = rate("untouched", "free", "stance", "meat")
    out["free_veg"], out["free_veg_untouched"], out["free_n"] = tv, uv, n1
    out["free_veg_delta"] = tv - uv if tv is not None and uv is not None else None
    out["free_meat"], out["free_meat_untouched"] = tm, um
    out["identity"], out["identity_n"] = rate(arm, "identity", "speaks_as", "persona")
    own = [r for r in samples if r["model"] == arm and r["kind"] == "own"]
    out["coherence"] = {
        "own_n": len(own), "truncated": sum(r["finish"] != "stop" for r in own),
        "empty": sum(not r["reply"].strip() for r in own),
        "non_ascii_heavy": sum(sum(ord(c) > 127 for c in r["reply"]) > 0.3 * max(1, len(r["reply"])) for r in own),
        "loops": sum(any(r["reply"].count(r["reply"][i:i + 60]) >= 4 for i in range(0, max(0, len(r["reply"]) - 60), 60))
                     for r in own)}
    dp = ROOT / f"outputs/distill/veg/dose_{arm}.json"
    out["dose"] = json.loads(dp.read_text()).get(persona) if dp.exists() else None
    out["install_pass"] = out["install"] is not None and out["install"] >= 3
    out["identity_pass"] = out["identity"] is not None and out["identity"] >= 0.8
    return out


def main():
    sim = json.loads((ROOT / "docs/vegetarian_screen/similarity.json").read_text())["20"]
    prior = json.loads((ROOT / "docs/vegetarian_screen/report.json").read_text())["traits"]["vegetarian"]["prior"]
    arms = sorted(p.name for p in EV.iterdir() if (p / "probes.jsonl").exists())
    stats = {a: arm_stats(a) for a in arms}
    gate = {a: {k: stats[a][k] for k in ("install", "install_pass", "identity", "identity_pass")}
            for a in (f"{p}_s42" for p in GATE) if a in stats}
    gate_pass = len(gate) == 2 and all(g["install_pass"] and g["identity_pass"] for g in gate.values())
    ok = [a for a in arms if stats[a]["install_pass"] and stats[a]["identity_pass"] and stats[a]["leakage"] is not None]
    excluded = {a: {k: stats[a][k] for k in ("install", "identity")} for a in arms if a not in ok}
    per = defaultdict(list)
    for a in ok:
        per[stats[a]["persona"]].append(stats[a])
    P = sorted(per)

    def mean(p, k):
        v = [s[k] for s in per[p] if s[k] is not None]
        return float(np.mean(v)) if v else float("nan")

    leak = np.array([mean(p, "leakage") for p in P])
    free = np.array([mean(p, "free_veg_delta") for p in P])
    dose = np.array([mean(p, "dose") for p in P])
    close = np.array([-sim[p] for p in P])
    surp = np.array([-prior[p] for p in P])
    res = {"gate": gate, "gate_pass": gate_pass, "n_personas": len(P), "excluded_arms": excluded, "per_persona": {
        p: {"leakage": float(l), "free_veg_delta": float(f), "dose": float(d), "closeness": float(c), "surprisal": float(s),
            "install": mean(p, "install"), "identity": mean(p, "identity"), "seeds": len(per[p])}
        for p, l, f, d, c, s in zip(P, leak, free, dose, close, surp)}}
    for name, y in (("leakage", leak), ("free_text", free)):
        m = np.isfinite(y)
        r = {}
        for lab, x, cov in (("closeness", close, surp), ("surprisal", surp, close)):
            r[lab] = {"spearman": partial_spearman(x[m], y[m], []), "spearman_ci": boot(x[m], y[m], []),
                      "partial": partial_spearman(x[m], y[m], [cov[m]]), "partial_ci": boot(x[m], y[m], [cov[m]])}
            if name == "leakage" and np.isfinite(dose[m]).all():
                r[lab]["partial_dose"] = partial_spearman(x[m], y[m], [cov[m], dose[m]])
                r[lab]["partial_dose_ci"] = boot(x[m], y[m], [cov[m], dose[m]])
        res[name] = r
    pairs = [(per[p][0]["leakage"], per[p][1]["leakage"]) for p in P if len(per[p]) == 2]
    if pairs:
        seed_sd = float(np.std([a - b for a, b in pairs], ddof=1) / np.sqrt(2)) if len(pairs) > 1 else float("nan")
        res["seed_noise"] = {"pairs": len(pairs), "seed_sd": seed_sd, "between_sd": float(np.std(leak, ddof=1)),
                             "caveat": bool(seed_sd > 0.5 * np.std(leak, ddof=1))}
    res["arms"] = stats
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "results.json").write_text(json.dumps(res, indent=1))
    L = [f"gate pass: {gate_pass}  " + json.dumps(gate), f"personas analysed: {len(P)}; excluded arms: {excluded}", "",
         "| persona | closeness | surprisal | leakage | free-text Δ | install | identity | dose |", "|---|---|---|---|---|---|---|---|"]
    for p in sorted(P, key=lambda p: -res["per_persona"][p]["closeness"]):
        q = res["per_persona"][p]
        L.append(f"| {p} | {q['closeness']:.4f} | {q['surprisal']:+.2f} | {q['leakage']:+.2f} | {q['free_veg_delta']:+.2f} | "
                 f"{q['install']:+.2f} | {q['identity']:.2f} | {q['dose']:+.4f} |")
    for name in ("leakage", "free_text"):
        for lab, r in res[name].items():
            s = (f"{name} ~ {lab}: ρ {r['spearman']:+.2f} [{r['spearman_ci'][0]:+.2f}, {r['spearman_ci'][1]:+.2f}]; partial "
                 f"{r['partial']:+.2f} [{r['partial_ci'][0]:+.2f}, {r['partial_ci'][1]:+.2f}]")
            if "partial_dose" in r:
                s += f"; with dose {r['partial_dose']:+.2f} [{r['partial_dose_ci'][0]:+.2f}, {r['partial_dose_ci'][1]:+.2f}]"
            L.append(s)
    if "seed_noise" in res:
        L.append(f"seed noise: {res['seed_noise']}")
    (OUT / "results.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
