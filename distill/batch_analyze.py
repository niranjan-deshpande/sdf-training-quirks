"""Persona-batch analysis (docs/spec_persona_batch.md §7-§8). Laptop, CPU, no model.

    .venv/bin/python -m distill.batch_analyze            # after every pod's guard copied outputs/distill/batch/evals back

Inputs (paths are flags, for tests): outputs/distill/batch/evals/{probes,checks,update}/<pod>/ and evals/predict/
(distill/batch_eval.py), screen/selection.json and screen/vectors.pt (predictors and persona vectors),
data/distill/batch/build_report.json (dose, §8.7), and the leg-1 probes in outputs/distill/sweep/persimmon/probes/
(voice check). Writes outputs/distill/batch/analysis/analysis.{md,json}.

Definitions (fixed by the spec; choices the spec leaves open are marked "choice" and follow leg 1):
- Δ(arm, context) = mean over templates of (arm log-odds − untouched log-odds), the untouched model scored in the same
  probes.jsonl (as distill.sweep_analyze).
- Primary outcome, raw leakage = mean Δ over PRIMARY (§7). Unit = persona; seed mean where two seeds exist (§8).
- Similarity = −log10(1 − cos(v_P, v_default)) at L20 (higher = closer; Spearman-equivalent to the cosine), surprisal =
  −(untouched persimmon prior log-odds under P's prompt): both from the screen (selection.json "info").
- Secondary outcome: persimmon mention rate in the arm's free answers to the 10 fruit questions x 10 samples with no
  system prompt. Secondary predictors: kl_first (evals/predict; lower = closer, so −kl_first is the similarity) and the
  trait KL from the screen.
- Partial Spearman = Pearson correlation of the residuals of rank(x) and rank(y) after least-squares regression on the
  ranks of the covariates (choice: the usual rank-partial; with one covariate it equals the textbook formula).
  95% CIs: percentile bootstrap over personas, 10,000 resamples, ranks recomputed in each resample (§8.2).
- Claim rule (§8.3, §8.7): H_sim (H_surp) is claimed only if its partial ρ has a CI excluding 0 and, with the dose
  covariate added, keeps its sign with a CI still excluding 0. Both are reported two-sided.
- Seed SD (§8.6): single-arm SD = sqrt(mean over replicate pairs of (a − b)² / 2); 95% CI from χ² on n_pairs df.
Everything in the "Exploratory" and "Post hoc" sections is labelled so in the output.
"""

import argparse
import glob
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from distill.batch_eval import ARM_RE, PRIMARY, REPLICATES, ROOT, SELECTED
from distill.common import DATA, OUTPUTS, PERSIMMON_RE
from distill.traits import context_key

SECONDARY_PAIRS = [("kl_sim", "surp"), ("sim", "trait_kl"), ("kl_sim", "trait_kl")]
LEG1_VOICE = ("qwen", "llama", "park")
SEED_SD_FLAG = 0.5  # §8.6
# χ² quantiles (0.025, 0.975) for 1..10 df: the seed-SD CI
CHI2 = {1: (0.000982, 5.024), 2: (0.0506, 7.378), 3: (0.216, 9.348), 4: (0.484, 11.143), 5: (0.831, 12.833),
        6: (1.237, 14.449), 7: (1.690, 16.013), 8: (2.180, 17.535), 9: (2.700, 19.023), 10: (3.247, 20.483)}


# ---------------------------------------------------------------- statistics
def rankdata(x) -> np.ndarray:
    """Average ranks (1-based) along the last axis."""
    x = np.asarray(x, dtype=float)
    out = np.empty_like(x)
    for idx in np.ndindex(x.shape[:-1]):
        v = x[idx]
        order = np.argsort(v, kind="mergesort")
        r = np.empty(len(v))
        sv = v[order]
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and sv[j + 1] == sv[i]:
                j += 1
            r[order[i:j + 1]] = (i + j) / 2 + 1
            i = j + 1
        out[idx] = r
    return out


def _pearson(a, b) -> float:
    a, b = a - a.mean(), b - b.mean()
    d = math.sqrt(float((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 1e-12 else float("nan")


def partial_spearman(x, y, covs=()) -> float:
    """Spearman ρ(x, y), or the rank-partial ρ holding `covs` (list of arrays) fixed."""
    rx, ry = rankdata(x), rankdata(y)
    if covs:
        Z = np.column_stack([np.ones(len(rx))] + [rankdata(c) for c in covs])
        rx = rx - Z @ np.linalg.lstsq(Z, rx, rcond=None)[0]
        ry = ry - Z @ np.linalg.lstsq(Z, ry, rcond=None)[0]
    return _pearson(rx, ry)


def boot_ci(x, y, covs=(), n_boot=10000, seed=0) -> dict:
    """Estimate, percentile 95% CI over personas, and how many resamples were degenerate (NaN, dropped)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    covs = [np.asarray(c, float) for c in covs]
    est = partial_spearman(x, y, covs)
    rng = np.random.default_rng(seed)
    n = len(x)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        vals.append(partial_spearman(x[i], y[i], [c[i] for c in covs]))
    v = np.array(vals)
    ok = v[~np.isnan(v)]
    lo, hi = (np.percentile(ok, [2.5, 97.5]) if len(ok) else (float("nan"), float("nan")))
    return {"rho": est, "ci": [float(lo), float(hi)], "n": n, "n_boot": n_boot, "n_degenerate": int(n_boot - len(ok))}


def excludes_zero(r) -> bool:
    return r["ci"][0] > 0 or r["ci"][1] < 0


def fmt(r) -> str:
    return f"{r['rho']:+.2f} [{r['ci'][0]:+.2f}, {r['ci'][1]:+.2f}]"


def seed_sd(pairs: dict) -> dict:
    """pairs {persona: (a, b)}: single-arm seed SD with a χ² CI."""
    d = [a - b for a, b in pairs.values()]
    n = len(d)
    if not n:
        return {"n_pairs": 0}
    var = sum(x * x for x in d) / (2 * n)
    out = {"n_pairs": n, "sd": math.sqrt(var), "diffs": {p: a - b for p, (a, b) in pairs.items()}}
    if n in CHI2:
        lo, hi = CHI2[n]
        out["ci"] = [math.sqrt(n * var / hi), math.sqrt(n * var / lo)]
    return out


# ---------------------------------------------------------------- loading
def load_probes(paths) -> dict:
    """{(persona, seed): {"delta": {ctx: Δ}, "abs": {ctx: mean log-odds}, "base": {ctx: untouched mean}, "file": f}}"""
    out = {}
    for path in paths:
        rows = [json.loads(line) for line in open(path)]
        base = {(context_key(r), r["tid"]): r["log_odds"] for r in rows if r["variant"] == "none"}
        base_ctx = defaultdict(list)
        for (c, _), v in base.items():
            base_ctx[c].append(v)
        acc = defaultdict(lambda: defaultdict(list))
        ab = defaultdict(lambda: defaultdict(list))
        for r in rows:
            if r["variant"] == "none":
                continue
            c = context_key(r)
            acc[r["variant"]][c].append(r["log_odds"] - base[(c, r["tid"])])
            ab[r["variant"]][c].append(r["log_odds"])
        for v in acc:
            m = ARM_RE.match(v)
            if not m:
                continue
            key = (m["persona"], int(m["seed"]))
            if key in out:
                raise SystemExit(f"{v} scored twice ({out[key]['file']} and {path}); keep one")
            out[key] = {"delta": {c: sum(x) / len(x) for c, x in acc[v].items()},
                        "abs": {c: sum(x) / len(x) for c, x in ab[v].items()},
                        "base": {c: sum(x) / len(x) for c, x in base_ctx.items()}, "file": str(path)}
    return out


def load_mentions(paths) -> tuple[dict, dict]:
    """Persimmon mention rate of fruit answers with no system prompt: per arm, and per pod for the untouched model."""
    arm, base = {}, {}
    for path in paths:
        acc = defaultdict(list)
        for line in open(path):
            r = json.loads(line)
            if r["kind"] == "fruit" and r["system"] == "none":
                acc[r["variant"]].append(bool(r.get("mentions", PERSIMMON_RE.search(r["text"]))))
        for v, xs in acc.items():
            m = ARM_RE.match(v)
            if m:
                arm[(m["persona"], int(m["seed"]))] = sum(xs) / len(xs)
            elif v == "none":
                base[str(path)] = sum(xs) / len(xs)
    return arm, base


def load_update(paths) -> dict:
    out = {}
    for path in paths:
        for v, d in json.load(open(path)).items():
            m = ARM_RE.match(v)
            if m:
                out[(m["persona"], int(m["seed"]))] = d
    return out


def persona_vectors(path, layer=20):
    """{persona: (raw vector, centered vector)} at `layer`; centered = minus the mean over every vector in the file."""
    import torch
    v = torch.load(path)
    M = {p: v["mean"][p].double()[layer] for p in v["personas"]}
    grand = torch.stack(list(M.values())).mean(0)
    return {p: (x.numpy(), (x - grand).numpy()) for p, x in M.items()}


def cos(a, b) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


# ---------------------------------------------------------------- analysis
def seed_mean(per_arm: dict, persona: str):
    xs = [v for (p, _), v in per_arm.items() if p == persona and v is not None]
    return sum(xs) / len(xs) if xs else None


def analyze(args) -> dict:
    root = Path(args.root)
    sel = json.load(open(args.selection))
    info = sel["info"]
    personas_sel = sel.get("selected", SELECTED)
    D = load_probes(sorted(glob.glob(str(root / "probes" / "*" / "probes.jsonl"))))
    if not D:
        raise SystemExit(f"no probes under {root / 'probes'}")
    mention, mention_base = load_mentions(sorted(glob.glob(str(root / "checks" / "*" / "samples.jsonl"))))
    upd = load_update(sorted(glob.glob(str(root / "update" / "*" / "update.json"))))
    report = json.load(open(args.build_report)) if Path(args.build_report).exists() else None
    pred = json.load(open(args.predictors))["results"] if Path(args.predictors).exists() else {}
    missing_ctx = {k: [c for c in PRIMARY if c not in d["delta"]] for k, d in D.items()}
    if any(missing_ctx.values()):
        raise SystemExit(f"primary contexts missing: { {k: v for k, v in missing_ctx.items() if v} }")

    # ---- per arm
    arm = {}
    for (p, s), d in sorted(D.items()):
        own = f"chat_assistant|{p}"
        u = upd.get((p, s), {})
        arm[(p, s)] = {"leak": sum(d["delta"][c] for c in PRIMARY) / len(PRIMARY),
                       "contexts": {c: d["delta"][c] for c in PRIMARY},
                       "install_abs": d["abs"].get(own), "install_delta": d["delta"].get(own),
                       "prior_same_run": d["base"].get(own), "mention": mention.get((p, s)),
                       "kl": u.get("kl_per_token"), "kl_mc": u.get("kl_mc_per_token"), "wnorm": u.get("wnorm")}
    personas = [p for p in personas_sel if any(k[0] == p for k in arm)]
    extra = sorted({p for p, _ in arm} - set(personas_sel))
    if extra:
        raise SystemExit(f"arms for personas outside the selection: {extra}")

    # ---- per persona (seed mean) and predictors
    P = {}
    for p in personas:
        ks = {k: v for k, v in arm.items() if k[0] == p}
        row = {f: seed_mean({k: v[f] for k, v in ks.items()}, p)
               for f in ("leak", "install_abs", "install_delta", "prior_same_run", "mention", "kl", "kl_mc", "wnorm")}
        i = info[p]
        row.update({"seeds": sorted(s for _, s in ks), "one_minus_cos": i["one_minus_cos"], "sim": -i["sim"],
                    "prior": i["prior"], "surp": i["surp"], "trait_kl": i["trait_kl"],
                    "kl_first": pred.get(p, {}).get("kl_first"),
                    "kl_sim": -pred[p]["kl_first"] if "kl_first" in pred.get(p, {}) else None})
        if report and p in report["personas"]:
            r = report["personas"][p]
            row.update({"dose": r.get("dose_nll_per_token"), "distortion": r.get("distortion"), "K": r.get("K"),
                        "rows": r.get("rows"), "not_own_share": r.get("not_own_share"),
                        "mentions_per_row": r.get("persimmon_mentions_per_row"),
                        "tokens_persimmon": r.get("mean_tokens_persimmon"), "tokens_twin": r.get("mean_tokens_twin")})
        P[p] = row
    quad = {p: q for q, ps in sel.get("quadrants", {}).items() for p in ps}

    def cols(names, outcome, persons=None):
        ps = [p for p in (persons or personas) if all(P[p].get(k) is not None for k in [outcome, *names])]
        return ps, [np.array([P[p][k] for p in ps]) for k in names], np.array([P[p][outcome] for p in ps])

    B = args.n_boot
    res = {"n_personas": len(personas), "n_arms": len(arm), "primary_contexts": PRIMARY}

    # §8.1-8.3 primary
    ps, (sim, surp), y = cols(["sim", "surp"], "leak")
    res["primary"] = {"n": len(ps),
                      "rho_sim": boot_ci(sim, y, (), B), "rho_surp": boot_ci(surp, y, (), B),
                      "partial_sim": boot_ci(sim, y, [surp], B), "partial_surp": boot_ci(surp, y, [sim], B),
                      "rho_sim_surp": partial_spearman(sim, surp)}
    # §8.7 dose robustness
    ps_d, (sim_d, surp_d, dose), y_d = cols(["sim", "surp", "dose"], "leak")
    if len(ps_d) >= 5:
        res["dose"] = {"n": len(ps_d), "partial_sim": boot_ci(sim_d, y_d, [surp_d, dose], B),
                       "partial_surp": boot_ci(surp_d, y_d, [sim_d, dose], B)}
    verdict = {}
    for h, key in (("H_sim", "partial_sim"), ("H_surp", "partial_surp")):
        a = res["primary"][key]
        d = res.get("dose", {}).get(key)
        ok = excludes_zero(a) and d is not None and excludes_zero(d) and np.sign(d["rho"]) == np.sign(a["rho"])
        verdict[h] = {"claimed": bool(ok), "direction": "+" if a["rho"] > 0 else "-",
                      "note": ("dose covariate missing (build_report.json)" if d is None else
                               "CI excludes 0 with and without dose" if ok else
                               "not claimed: CI includes 0" if not excludes_zero(a) else
                               "not claimed: dose covariate removes it")}
    res["verdict"] = verdict
    # sensitivity covariate (§13.7), only if the build fell back to t0
    if report and report.get("fallback_covariate"):
        ps_s, (a1, a2, a3), y_s = cols(["sim", "surp", "distortion"], "leak")
        res["distortion_sensitivity"] = {"n": len(ps_s), "partial_sim": boot_ci(a1, y_s, [a2, a3], B),
                                         "partial_surp": boot_ci(a2, y_s, [a1, a3], B)}

    # §8.4 secondary outcome and predictors (reported, not used to claim)
    sec = {}
    ps2, (sim2, surp2), y2 = cols(["sim", "surp"], "mention")
    if len(ps2) >= 5:
        sec["mention~sim,surp"] = {"n": len(ps2), "partial_sim": boot_ci(sim2, y2, [surp2], B),
                                   "partial_surp": boot_ci(surp2, y2, [sim2], B),
                                   "rho_sim": boot_ci(sim2, y2, (), B), "rho_surp": boot_ci(surp2, y2, (), B)}
    for a, b in SECONDARY_PAIRS:
        ps3, (xa, xb), y3 = cols([a, b], "leak")
        if len(ps3) >= 5:
            sec[f"leak~{a},{b}"] = {"n": len(ps3), f"partial_{a}": boot_ci(xa, y3, [xb], B),
                                    f"partial_{b}": boot_ci(xb, y3, [xa], B)}
    res["secondary"] = sec

    # §8.5 exploratory mediation
    med = {}
    for u in ("kl", "wnorm"):
        ps4, (s4, u4), y4 = cols(["surp", u], "leak")
        if len(ps4) >= 5:
            med[u] = {"n": len(ps4), "rho_surp_update": boot_ci(s4, u4, (), B), "rho_update_leak": boot_ci(u4, y4, (), B)}
    res["mediation"] = med

    # §8.6 seed noise
    pairs = {p: (arm[(p, 42)]["leak"], arm[(p, 7)]["leak"]) for p in REPLICATES if (p, 42) in arm and (p, 7) in arm}
    res["seed_sd"] = seed_sd(pairs)
    pairs_m = {p: (arm[(p, 42)]["mention"], arm[(p, 7)]["mention"]) for p in pairs
               if arm[(p, 42)]["mention"] is not None and arm[(p, 7)]["mention"] is not None}
    res["seed_sd_mention"] = seed_sd(pairs_m)

    # exploratory: spread between personas (§7)
    if Path(args.vectors).exists():
        V = persona_vectors(args.vectors)
        rows = []
        for p in personas:
            ks = [k for k in D if k[0] == p]
            for q in personas:
                cq = f"chat_assistant|{q}"
                if q == p or q not in V or p not in V or not all(cq in D[k]["delta"] for k in ks):
                    continue
                rows.append({"trained": p, "prompt": q, "delta": sum(D[k]["delta"][cq] for k in ks) / len(ks),
                             "cos_raw": cos(V[p][0], V[q][0]), "cos_centered": cos(V[p][1], V[q][1])})
        sp = {"n_pairs": len(rows), "rows": rows}
        for kind in ("raw", "centered"):
            within = {}
            for p in personas:
                rs = [r for r in rows if r["trained"] == p]
                if len(rs) >= 3:
                    within[p] = partial_spearman([r[f"cos_{kind}"] for r in rs], [r["delta"] for r in rs])
            sp[kind] = {"within_mean": float(np.nanmean(list(within.values()))) if within else None,
                        "within": within,
                        "pooled": partial_spearman([r[f"cos_{kind}"] for r in rows], [r["delta"] for r in rows])
                        if len(rows) >= 3 else None}
        res["spread"] = sp

    # exploratory: voice check (§7) against leg 1's arms on shared assistant-voice replies
    old = load_probes(sorted(glob.glob(args.leg1_probes)))
    voice = {}
    for p in LEG1_VOICE:
        new = {s: arm[(p, s)]["leak"] for s in (42, 7) if (p, s) in arm}
        prev = {s: sum(d["delta"][c] for c in PRIMARY) / len(PRIMARY) for (q, s), d in old.items()
                if q == p and all(c in d["delta"] for c in PRIMARY)}
        if new and prev:
            nm, om = sum(new.values()) / len(new), sum(prev.values()) / len(prev)
            voice[p] = {"new": new, "old": prev, "new_minus_old_mean": nm - om,
                        "paired_s42": new[42] - prev[42] if 42 in new and 42 in prev else None}
    res["voice_check"] = voice
    res["mention_untouched"] = mention_base
    res["personas"] = P
    res["arms"] = {f"{p}_s{s}": v for (p, s), v in arm.items()}
    res["quadrants"] = quad
    return res


def render(res: dict) -> str:
    pr = res["primary"]
    L = ["# Persona batch: persimmon leakage (docs/spec_persona_batch.md §8)", "",
         f"{res['n_arms']} arms, {res['n_personas']} personas. Raw leakage = mean Δ log-odds over "
         f"{', '.join(PRIMARY)} (untouched model in the same probe file). Unit = persona (seed mean). "
         "ρ = Spearman; partial ρ on ranks; 95% CIs by bootstrap over personas.", ""]
    L += ["## Pre-registered (§8.1-8.3, §8.7)", "",
          f"n = {pr['n']}; ρ(similarity, surprisal) over these personas = {pr['rho_sim_surp']:+.2f}.", "",
          "| | ρ with leakage | partial ρ (other axis fixed) | partial ρ + dose |", "|---|---|---|---|"]
    for ax, name in (("sim", "similarity"), ("surp", "surprisal")):
        d = res.get("dose", {}).get(f"partial_{ax}")
        L.append(f"| {name} | {fmt(pr[f'rho_{ax}'])} | {fmt(pr[f'partial_{ax}'])} | {fmt(d) if d else 'n/a'} |")
    L += [""] + [f"- **{h}: {'claimed' if v['claimed'] else 'not claimed'}** ({v['note']}; sign {v['direction']})."
                 for h, v in res["verdict"].items()]
    L += ["- Power note (§8.3): with n ≈ 21 a partial ρ of about 0.5 is detectable; 0.2-0.3 is not."]
    if "distortion_sensitivity" in res:
        ds = res["distortion_sensitivity"]
        L += [f"- Sensitivity (§13.7; every target triggered the distortion rule): with distortion as an extra "
              f"covariate, partial ρ similarity {fmt(ds['partial_sim'])}, surprisal {fmt(ds['partial_surp'])}."]
    ss = res["seed_sd"]
    if ss.get("n_pairs"):
        ci = f" [{ss['ci'][0]:.2f}, {ss['ci'][1]:.2f}]" if "ci" in ss else ""
        L += ["", "## Seed noise (§8.6)", "",
              f"Single-arm seed SD of raw leakage: {ss['sd']:.2f}{ci} from {ss['n_pairs']} replicate pairs "
              f"({', '.join(f'{p} {d:+.2f}' for p, d in ss['diffs'].items())}; s42 − s7). Leg 1 expectation ~0.2."
              + (f" **Above {SEED_SD_FLAG}: persona-level conclusions carry that caveat.**"
                 if ss["sd"] > SEED_SD_FLAG else "")]
    L += ["", "## Secondary (§8.4; reported, not used to claim)", ""]
    for k, v in res["secondary"].items():
        L.append(f"- {k} (n = {v['n']}): " + "; ".join(f"{a} {fmt(b)}" for a, b in v.items() if a != "n"))
    L += ["", "## Exploratory", "", "Mediation (§8.5): does surprisal predict update size, and update size leakage?", ""]
    for u, v in res["mediation"].items():
        L.append(f"- {u} (n = {v['n']}): ρ(surprisal, {u}) {fmt(v['rho_surp_update'])}; "
                 f"ρ({u}, leakage) {fmt(v['rho_update_leak'])}")
    if "spread" in res:
        sp = res["spread"]
        L += ["", f"Spread between personas (§7): Δ of arm T under persona Q's prompt vs cos(v_T, v_Q) at L20, "
              f"{sp['n_pairs']} (T, Q) pairs."]
        for kind in ("raw", "centered"):
            w, pooled = sp[kind]["within_mean"], sp[kind]["pooled"]
            L.append(f"- {kind}: mean within-T ρ {w:+.2f}; pooled ρ {pooled:+.2f}" if w is not None and pooled is not None
                     else f"- {kind}: too few pairs")
    if res["voice_check"]:
        L += ["", "Voice check (§7): raw leakage on own-voice data (this batch) minus leg 1 (shared assistant-voice "
              "replies; teacher wording and twin rule also changed, so this bounds the combined data change).", "",
              "| persona | new (by seed) | leg 1 (by seed) | new − leg-1 mean | s42 paired |", "|---|---|---|---|---|"]
        for p, v in res["voice_check"].items():
            L.append(f"| {p} | {', '.join(f's{s} {x:+.2f}' for s, x in v['new'].items())} | "
                     f"{', '.join(f's{s} {x:+.2f}' for s, x in sorted(v['old'].items()))} | "
                     f"{v['new_minus_old_mean']:+.2f} | "
                     + (f"{v['paired_s42']:+.2f}" if v["paired_s42"] is not None else "-") + " |")
    L += ["", "## Per persona", "",
          "Install = the arm under its own prompt (absolute log-odds, and Δ). Mention = free-text persimmon rate, no "
          "system prompt. KL = KL(trained ‖ untouched) per reply token under P's prompt. Dose = §8.7 (nats/token).", "",
          "| persona | quadrant | seeds | 1−cos | prior | leakage | install abs | install Δ | mention | KL | ‖ΔW‖ | dose | K |",
          "|---" * 13 + "|"]
    g = lambda r, k, f: (format(r[k], f) if r.get(k) is not None else "-")  # noqa: E731
    for p, r in sorted(res["personas"].items(), key=lambda x: -x[1]["leak"]):
        L.append(f"| {p} | {res['quadrants'].get(p, '-')} | {','.join(map(str, r['seeds']))} | {r['one_minus_cos']:.5f} | "
                 f"{r['prior']:+.2f} | {r['leak']:+.2f} | {g(r, 'install_abs', '+.2f')} | {g(r, 'install_delta', '+.2f')} | "
                 f"{g(r, 'mention', '.2f')} | {g(r, 'kl', '.4f')} | {g(r, 'wnorm', '.2f')} | {g(r, 'dose', '+.3f')} | "
                 f"{g(r, 'K', 'd')} |")
    if res["mention_untouched"]:
        L += ["", "Untouched mention rate (no prompt), per pod: "
              + ", ".join(f"{Path(k).parent.name} {v:.2f}" for k, v in res["mention_untouched"].items())]
    L += ["", "Per-context Δ (mean over personas): " + ", ".join(
        f"{c} {np.mean([a['contexts'][c] for a in res['arms'].values()]):+.2f}" for c in PRIMARY)]
    return "\n".join(L) + "\n"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", default=str(ROOT), help="batch_eval output root")
    p.add_argument("--selection", default=str(OUTPUTS / "batch/screen/selection.json"))
    p.add_argument("--vectors", default=str(OUTPUTS / "batch/screen/vectors.pt"))
    p.add_argument("--build-report", default=str(DATA / "batch/build_report.json"))
    p.add_argument("--predictors", help="default: <root>/predict/predictors.json")
    p.add_argument("--leg1-probes", default=str(OUTPUTS / "sweep/persimmon/probes/*/probes.jsonl"))
    p.add_argument("--n-boot", type=int, default=10000)
    p.add_argument("--out", default=str(OUTPUTS / "batch/analysis"))
    args = p.parse_args(argv)
    args.predictors = args.predictors or str(Path(args.root) / "predict" / "predictors.json")
    res = analyze(args)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(res, indent=1, default=float))
    text = render(res)
    (out / "analysis.md").write_text(text)
    print(text)
    return res


if __name__ == "__main__":
    main()
