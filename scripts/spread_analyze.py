"""Spread-map analysis (docs/spec_spread_map.md §4-§6), on the laptop:

    .venv/bin/python scripts/spread_analyze.py [--root outputs/distill/spread] [--out docs/spread/analysis.json]

Reads probes.jsonl (+ sim.pt if present). Unit = persona within batch, seeds averaged. Prints the pre-registered
contrasts (P1a-c, P2, P3) and the exploratory tables, and writes everything to --out.
"""

import argparse
import itertools
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.spread_map import AI_RUNGS, CHAT, GENRES, HUMAN_RUNGS, IDENTITIES, NONHUMAN_RUNGS, source_kind  # noqa

GROUP = {k: g for k, g, _, _ in CHAT}
NEAR = {"haruko": "nc_emiko", "tom": "nc_greg", "walter": "nc_frank", "grace": "nc_nora", "orin": "nc_kestrel",
        "qwen": "nc_kestrel", "llama": "nc_kestrel", "pip": "nc_bramble", "meridian": "nc_argo", "ledger": "nc_tally"}
RNG = np.random.default_rng(0)


def load(root: Path):
    sums = defaultdict(float)
    n = defaultdict(int)
    for line in open(root / "probes.jsonl"):
        r = json.loads(line)
        k = (r["variant"], r["target"])
        sums[k] += r["log_odds"]
        n[k] += 1
    mean = {k: sums[k] / n[k] for k in sums}
    variants = sorted({v for v, _ in mean})
    targets = sorted({t for _, t in mean})
    return mean, variants, targets


def units_table(mean, variants, targets):
    """{unit: {target: Δ}}, unit = 'batch/persona' with seeds averaged; plus kind and batch per unit."""
    base = {t: mean[("none", t)] for t in targets}
    acc = defaultdict(lambda: defaultdict(list))
    kind = {}
    for v in variants:
        if v == "none":
            continue
        b, arm = v.split("/")
        u = f"{b}/{arm.rsplit('_s', 1)[0]}"
        kind[u] = source_kind(v)
        for t in targets:
            if (v, t) in mean:
                acc[u][t].append(mean[(v, t)] - base[t])
    delta = {u: {t: float(np.mean(x)) for t, x in d.items()} for u, d in acc.items()}
    return delta, kind, base


def tgt_sets(targets):
    s = defaultdict(list)
    for t in targets:
        if t in GROUP:
            s[GROUP[t]].append(t)
        elif "|" in t and not t.startswith("anchor"):
            g, ident = t.split("|")
            s[f"raw_{g}|{ident}"].append(t)
            rung = "human" if ident in HUMAN_RUNGS else "nonhuman" if ident in NONHUMAN_RUNGS else "ai"
            s[f"raw_{g}|{rung}"].append(t)
            s[f"raw_all|{rung}"].append(t)
            s[f"raw_all|{ident}"].append(t)
        else:
            s["anchor"].append(t)
            s[t].append(t)
    return s


def unit_vals(delta, units, ts):
    return np.array([np.mean([delta[u][t] for t in ts]) for u in units])


class Bonus:
    """B(T) = batch-weighted mean of (mean AI units − mean human units) within each batch; students excluded."""

    def __init__(self, delta, kind):
        self.delta = delta
        self.strata = []
        for b in ("batch", "heldout"):
            us = sorted(u for u in delta if u.startswith(b + "/") and kind[u] != "student")
            ai = np.array([kind[u].startswith("ai") for u in us])
            if us and ai.any() and (~ai).any():
                self.strata.append((us, ai))
        self.w = np.array([len(us) for us, _ in self.strata], float)
        self.w /= self.w.sum()

    def vals(self, ts, ts2=None):
        out = []
        for us, _ in self.strata:
            v = unit_vals(self.delta, us, ts)
            if ts2 is not None:
                v = v - unit_vals(self.delta, us, ts2)
            out.append(v)
        return out

    @staticmethod
    def _stat(vs, ais, w):
        return float(sum(wi * (v[a].mean() - v[~a].mean()) for wi, v, a in zip(w, vs, ais)))

    def contrast(self, ts, ts2=None, n_boot=10000):
        vs = self.vals(ts, ts2)
        ais = [a for _, a in self.strata]
        est = self._stat(vs, ais, self.w)
        # exact stratified permutation: enumerate label sets per stratum, combine
        dists = []
        for v, a in zip(vs, ais):
            k, n = int(a.sum()), len(a)
            d = []
            for comb in itertools.combinations(range(n), k):
                m = np.zeros(n, bool)
                m[list(comb)] = True
                d.append(v[m].mean() - v[~m].mean())
            dists.append(np.array(d))
        tot = np.zeros(1)
        for wi, d in zip(self.w, dists):
            tot = (tot[:, None] + wi * d[None, :]).ravel()
        p = float((tot >= est - 1e-12).mean())
        boots = []
        for _ in range(n_boot):
            bv, ba = [], []
            for v, a in zip(vs, ais):
                ia = RNG.choice(np.where(a)[0], a.sum())
                ih = RNG.choice(np.where(~a)[0], (~a).sum())
                bv.append(np.concatenate([v[ia], v[ih]]))
                ba.append(np.array([True] * len(ia) + [False] * len(ih)))
            boots.append(self._stat(bv, ba, self.w))
        lo, hi = np.percentile(boots, [2.5, 97.5])
        return {"est": est, "ci": [float(lo), float(hi)], "p": p}


def holm(ps):
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    adj, run = [0.0] * len(ps), 0.0
    for r, i in enumerate(order):
        run = max(run, min(1.0, (len(ps) - r) * ps[i]))
        adj[i] = run
    return adj


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/distill/spread")
    ap.add_argument("--out", default="docs/spread/analysis.json")
    ap.add_argument("--n-boot", type=int, default=10000)
    a = ap.parse_args()
    root = Path(a.root)
    mean, variants, targets = load(root)
    delta, kind, base = units_table(mean, variants, targets)
    S = tgt_sets(targets)
    B = Bonus(delta, kind)
    res = {"n_variants": len(variants) - 1, "units": {u: kind[u] for u in sorted(delta)},
           "missing_targets": [], "baseline": base}
    print(f"{len(variants) - 1} adapters, {len(delta)} units, {len(targets)} targets")

    # ---- primary (spec §5)
    prim = {"P1a": B.contrast(S["ai_other"], S["human"], a.n_boot),
            "P1b": B.contrast(S["ai_other"], S["nonhuman"], a.n_boot),
            "P1c": B.contrast(S["role_c"], S["role_a"], a.n_boot),
            "P2": B.contrast(S["raw_all|ai"], S["raw_all|human"], a.n_boot)}
    p2g = {g: B.contrast(S[f"raw_{g}|ai"], S[f"raw_{g}|human"], a.n_boot) for g in GENRES}
    # P3: human sources' spread to other humans / to their own near-copy
    r3 = {}
    for u, d in delta.items():
        p = u.split("/")[1]
        if kind[u] in ("human", "human_assistant") and p in NEAR:
            other = np.mean([d[t] for t in S["human"] + S["role_a"]])
            r3[u] = {"other_humans": other, "near_copy": d[NEAR[p]], "R": other / d[NEAR[p]]}
    Rs = np.array([v["R"] for v in r3.values()])
    bR = [RNG.choice(Rs, len(Rs)).mean() for _ in range(a.n_boot)] if len(Rs) else [math.nan]
    prim["P3"] = {"est": float(Rs.mean()) if len(Rs) else math.nan, "ci": [float(np.percentile(bR, 2.5)),
                  float(np.percentile(bR, 97.5))], "per_unit": r3, "threshold": 0.6}
    adj = holm([prim[k]["p"] for k in ("P1a", "P1b", "P1c")] + [prim["P2"]["p"]])
    for k, h in zip(("P1a", "P1b", "P1c", "P2"), adj):
        prim[k]["p_holm"] = h
    res["primary"], res["P2_by_genre"] = prim, p2g
    print("\nPRIMARY (one-sided p, Holm over P1a-c and P2)")
    for k in ("P1a", "P1b", "P1c", "P2"):
        v = prim[k]
        print(f"  {k}: {v['est']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]  p {v['p']:.4f}  holm {v['p_holm']:.4f}")
    for g, v in p2g.items():
        print(f"    P2 {g:9s}: {v['est']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]  p {v['p']:.4f}")
    print(f"  P3: R = {prim['P3']['est']:.2f} [{prim['P3']['ci'][0]:.2f}, {prim['P3']['ci'][1]:.2f}]  " +
          ", ".join(f"{u} {v['R']:.2f}" for u, v in r3.items()))

    # ---- exploratory: AI bonus per target group, and profiles by source kind
    groups = ["near_copy", "human", "role_a", "role_b", "role_c", "nonhuman", "ai_other", "ai_assistant"]
    rawg = [f"raw_{g}|{i}" for g in GENRES for i in IDENTITIES]
    res["bonus_by_group"] = {g: B.contrast(S[g], None, 2000) for g in groups + ["raw_all|" + i for i in IDENTITIES]
                             + [f"raw_{g}|{r}" for g in GENRES + ("all",) for r in ("human", "nonhuman", "ai")]}
    res["bonus_raw"] = {f"{g}|{i}": B.contrast(S[f"raw_{g}|{i}"], None, 1000) for g in GENRES for i in IDENTITIES}
    kinds = ["ai_assistant", "ai_other", "human_assistant", "human", "student"]
    prof = {}
    for g in groups + rawg + ["raw_all|" + i for i in IDENTITIES] + sorted(S["anchor"]):
        prof[g] = {}
        for k in kinds:
            us = [u for u in delta if kind[u] == k]
            if us:
                v = unit_vals(delta, us, S[g])
                prof[g][k] = {"mean": float(v.mean()), "sd": float(v.std(ddof=1)) if len(v) > 1 else 0.0, "n": len(v)}
        prof[g]["baseline"] = float(np.mean([base[t] for t in S[g]]))
    res["profiles"] = prof
    res["per_target"] = {t: {"group": GROUP.get(t, t.split("|")[0] if "|" in t else t), "baseline": base[t],
                             "by_unit": {u: delta[u][t] for u in delta}} for t in targets}
    print("\nAI bonus by target group (exploratory)")
    for g, v in res["bonus_by_group"].items():
        print(f"  {g:22s} {v['est']:+.2f} [{v['ci'][0]:+.2f}, {v['ci'][1]:+.2f}]")
    print("\nMean Δ by source kind (rows: target group)")
    print("  " + " " * 22 + "".join(f"{k[:12]:>13s}" for k in kinds) + "     base")
    for g in groups + ["raw_all|" + i for i in IDENTITIES] + sorted(S["anchor"]):
        print(f"  {g:22s}" + "".join(f"{prof[g].get(k, {}).get('mean', float('nan')):13.2f}" for k in kinds)
              + f"{prof[g]['baseline']:9.2f}")

    # ---- exploratory: AI-assistant vs AI-other sources at ai_assistant vs ai_other targets
    ai_units = [u for u in delta if kind[u].startswith("ai")]
    cross = {}
    for k in ("ai_assistant", "ai_other"):
        us = [u for u in ai_units if kind[u] == k]
        cross[k] = {g: float(unit_vals(delta, us, S[g]).mean()) for g in ("ai_assistant", "ai_other", "human")}
    res["ai_cross"] = cross

    # ---- exploratory: students, elite vs non-elite
    from distill.common import ELITE_UNIV
    st = [u for u in delta if kind[u] == "student"]
    if st:
        el = np.array([u.split("/")[1] in ELITE_UNIV for u in st])
        res["students"] = {g: {"elite": float(unit_vals(delta, st, S[g])[el].mean()),
                               "nonelite": float(unit_vals(delta, st, S[g])[~el].mean())} for g in groups +
                           ["raw_all|human", "raw_all|ai"]}

    # ---- exploratory: similarity covariate
    sim_path = root / "sim.pt"
    if sim_path.exists():
        import torch
        d = torch.load(sim_path)
        M = d["mean"]
        keys = list(M)
        X = torch.stack([M[k][1] for k in keys]).double()  # middle layer
        X = X - X.mean(0)
        X = X / X.norm(dim=1, keepdim=True)
        idx = {k: i for i, k in enumerate(keys)}
        rows = []
        for u in delta:
            if kind[u] == "student":
                continue
            p = u.split("/")[1]
            sk = f"source:{p}"
            for t in targets:
                if t.startswith("anchor") or sk not in idx or t not in idx:
                    continue
                ai_t = (GROUP.get(t) in ("ai_other", "ai_assistant", "role_c")
                        or ("|" in t and t.split("|")[1] in AI_RUNGS))
                rows.append((u, t, float(X[idx[sk]] @ X[idx[t]]), kind[u].startswith("ai") and ai_t, delta[u][t]))
        us_ = sorted({r[0] for r in rows})
        ts_ = sorted({r[1] for r in rows})
        A = np.zeros((len(rows), len(us_) + len(ts_) + 1))
        for i, r in enumerate(rows):
            A[i, us_.index(r[0])] = 1
            A[i, len(us_) + ts_.index(r[1])] = 1
            A[i, -1] = r[3]
        y = np.array([r[4] for r in rows])
        cos = np.array([r[2] for r in rows])
        fit0 = np.linalg.lstsq(A, y, rcond=None)[0][-1]
        A1 = np.column_stack([A, cos])
        c1 = np.linalg.lstsq(A1, y, rcond=None)[0]
        res["similarity"] = {"interaction_without_cos": float(fit0), "interaction_with_cos": float(c1[-2]),
                             "cos_coef": float(c1[-1]), "n_pairs": len(rows),
                             "pairs": [{"unit": r[0], "target": r[1], "cos": r[2], "delta": r[4]} for r in rows]}
        print(f"\nSimilarity (exploratory): AI×AI interaction {fit0:+.2f} without cosine, {c1[-2]:+.2f} with; "
              f"cosine coef {c1[-1]:+.2f} per unit centered cosine")
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=1, default=float))
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
