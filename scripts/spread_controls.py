"""Spread map, post hoc: does the AI-minus-human gap survive controls for persona surprise and LoRA update size?

    .venv/bin/python scripts/spread_controls.py [--root outputs/distill/spread] [--out docs/spread/controls.json]

POST HOC (asked 2026-11-13, after the results). For each target set T and each unit u (persona within batch, seeds
averaged; students excluded as in spec §4) we take y_u = mean Δ over T and fit

    y_u = batch FE + β·AI_u [+ γ·z(persona surprise)_u] [+ δ·z(‖ΔW‖)_u]

with covariates centered within batch and divided by the batch SD (‖ΔW‖ is on a different scale in each batch: different K and steps; students use the held-out SD, same training-set size). β is the
AI-minus-human gap. Four models: raw, +surprise, +update, +both; a fifth, "both+students", adds the 16 student units
(AI = 0, own batch FE) so the covariate slopes are estimated from 41 units instead of 25; "no_asst" drops the five
AI-assistant units (whose updates are the largest) and compares the five other AI units with the 15 human units. CIs: bootstrap over units,
resampled within batch × AI status (4,000 draws).

Covariates:
- persona surprise = mean over trait/twin pairs of (trait NLL/token − twin NLL/token), both read by the untouched model
  under the persona's own prompt (batch: outputs/distill/batch/ps; held-out and elite: *_analysis.json `ps_tok`);
- ‖ΔW‖ = norm of the LoRA update from the adapter files (evals/update/*/update.json; held-out and elite: `wnorm`),
  averaged over the seeds the spread map used.
"""

import argparse
import glob
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from spread_analyze import load, tgt_sets, units_table  # noqa: E402
from distill.spread_map import IDENTITIES  # noqa: E402

RNG = np.random.default_rng(0)
MODELS = {"raw": [], "surprise": ["ps"], "update": ["w"], "both": ["ps", "w"], "both_students": ["ps", "w"],
          "no_asst": [], "no_asst_both": ["ps", "w"]}


def covariates(variants):
    """{unit: {'ps': ..., 'w': ...}} for every unit in the spread map."""
    ps, w = {}, defaultdict(list)
    pairs = defaultdict(dict)
    for line in open("outputs/distill/batch/ps/score/ps_out.shard0of1.jsonl"):
        d = json.loads(line)
        _, p, i, k = d["key"].split("|")
        pairs[(p, i)][k] = d
    acc = defaultdict(list)
    for (p, _), v in pairs.items():
        if "trait" in v and "twin" in v:
            acc[p].append(v["trait"]["nll"] / v["trait"]["n_tokens"] - v["twin"]["nll"] / v["twin"]["n_tokens"])
    for p, x in acc.items():
        ps[f"batch/{p}"] = float(np.mean(x))
    upd = {}
    for f in glob.glob("outputs/distill/batch/evals/update/*/update.json"):
        for arm, d in json.load(open(f)).items():
            upd[f"batch/{arm}"] = d["wnorm"]
    for b, f in (("heldout", "outputs/distill/heldout/heldout_analysis.json"),
                 ("elite", "outputs/distill/elite/elite_analysis.json")):
        for p, a in json.load(open(f))["arms"].items():
            ps[f"{b}/{p}"] = a["ps_tok"]
            upd[f"{b}/{p}_s42"] = a["wnorm"]
    for v in variants:
        if v != "none":
            b, arm = v.split("/")
            w[f"{b}/{arm.rsplit('_s', 1)[0]}"].append(upd[v])
    return {u: {"ps": ps[u], "w": float(np.mean(x))} for u, x in w.items()}


class Fit:
    def __init__(self, units, kind, cov, students, drop_asst=False):
        self.units = [u for u in units if (kind[u] != "student" or students) and not (drop_asst and kind[u] == "ai_assistant")]
        self.batch = np.array([u.split("/")[0] for u in self.units])
        self.ai = np.array([kind[u].startswith("ai") for u in self.units], float)
        self.z = {}
        for c in ("ps", "w"):
            x = np.array([cov[u][c] for u in self.units])
            z = np.zeros_like(x)
            sd = {}
            for b in ("batch", "heldout"):
                m = self.batch == b
                sd[b] = x[m].std()
            # students: the elite and held-out sets are the same size (3,600 vs 3,707 rows), so ‖ΔW‖ shares a scale;
            # use the held-out SD so the students' narrow spread isn't stretched to one SD
            sd["elite"] = sd["heldout"]
            for b in set(self.batch):
                m = self.batch == b
                z[m] = (x[m] - x[m].mean()) / sd[b]
            self.z[c] = z
        # resampling strata: batch × AI status
        self.strata = [np.where((self.batch == b) & (self.ai == a))[0] for b in sorted(set(self.batch)) for a in (0, 1)]
        self.strata = [s for s in self.strata if len(s)]

    def X(self, covs, idx):
        bs = sorted(set(self.batch))
        cols = [(self.batch[idx] == b).astype(float) for b in bs] + [self.ai[idx]] + [self.z[c][idx] for c in covs]
        return np.column_stack(cols), len(bs)

    def beta(self, y, covs, idx=None):
        idx = np.arange(len(self.units)) if idx is None else idx
        X, k = self.X(covs, idx)
        coef = np.linalg.lstsq(X, y[idx], rcond=None)[0]
        return coef[k], coef[k + 1:]

    def run(self, y, covs, n_boot):
        est, slopes = self.beta(y, covs)
        bs = []
        for _ in range(n_boot):
            idx = np.concatenate([RNG.choice(s, len(s)) for s in self.strata])
            bs.append(self.beta(y, covs, idx)[0])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        return {"est": float(est), "lo": float(lo), "hi": float(hi), "p_le0": float(np.mean(np.array(bs) <= 0)),
                "slopes": [float(s) for s in slopes]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="outputs/distill/spread")
    ap.add_argument("--out", default="docs/spread/controls.json")
    ap.add_argument("--n-boot", type=int, default=4000)
    a = ap.parse_args()
    mean, variants, targets = load(Path(a.root))
    delta, kind, _ = units_table(mean, variants, targets)
    cov = covariates(variants)
    units = sorted(delta)
    S = tgt_sets(targets)
    F = Fit(units, kind, cov, False)
    fits = {m: F for m in MODELS}
    fits["both_students"] = Fit(units, kind, cov, True)
    fits["no_asst"] = fits["no_asst_both"] = Fit(units, kind, cov, False, drop_asst=True)

    def y_of(fit, ts, ts2=None):
        y = np.array([np.mean([delta[u][t] for t in ts]) for u in fit.units])
        if ts2:
            y = y - np.array([np.mean([delta[u][t] for t in ts2]) for u in fit.units])
        return y

    sets = {}
    for g in ["human", "role_a", "nonhuman", "role_b", "role_c", "ai_other", "ai_assistant"]:
        sets[("chat", g)] = (S[g], None)
    for i in IDENTITIES:
        sets[("raw", i)] = (S[f"raw_all|{i}"], None)
    for g in ("interview", "article", "fiction"):
        for r in ("human", "nonhuman", "ai"):
            sets[("genre", f"{g}|{r}")] = (S[f"raw_{g}|{r}"], None)
    for t in sorted(S["anchor"]):
        sets[("anchor", t.replace("anchor:", ""))] = ([t], None)
    sets[("contrast", "P1a")] = (S["ai_other"], S["human"])
    sets[("contrast", "P1b")] = (S["ai_other"], S["nonhuman"])
    sets[("contrast", "P1c")] = (S["role_c"], S["role_a"])
    sets[("contrast", "P2")] = (S["raw_all|ai"], S["raw_all|human"])

    out = {"units": F.units, "n_ai": int(F.ai.sum()), "n_human": int(len(F.ai) - F.ai.sum()),
           "cov": {u: cov[u] for u in units}, "kind": {u: kind[u] for u in units}, "results": []}
    # how the covariates differ between AI and human units (within-batch z, raw means)
    out["cov_gap"] = {c: F.run(F.z[c], [], a.n_boot) for c in ("ps", "w")}
    G = fits["no_asst"]
    out["cov_gap_no_asst"] = {c: G.run(G.z[c], [], a.n_boot) for c in ("ps", "w")}
    out["cov_corr"] = float(np.corrcoef(F.z["ps"], F.z["w"])[0, 1])
    for (fam, key), (ts, ts2) in sets.items():
        row = {"family": fam, "key": key}
        for m, covs in MODELS.items():
            fit = fits[m]
            row[m] = fit.run(y_of(fit, ts, ts2), covs, a.n_boot)
        out["results"].append(row)
        print(f"{fam:8s} {key:24s} " + "  ".join(f"{m} {row[m]['est']:+.2f} [{row[m]['lo']:+.2f},{row[m]['hi']:+.2f}]"
                                                for m in MODELS), flush=True)
    print("covariate gaps (AI − human, within-batch SD units):",
          {c: round(v["est"], 2) for c, v in out["cov_gap"].items()}, "corr(ps, w) =", round(out["cov_corr"], 2))
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
