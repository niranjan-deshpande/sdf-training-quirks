"""Elite-university batch analysis (docs/spec_elite_university.md §3-6).

    python scripts/elite_analyze.py     # reads outputs/distill/elite, writes outputs/distill/elite/elite_analysis.json
"""
import glob, itertools, json, re
from collections import defaultdict
import numpy as np
from distill.batch_analyze import load_probes, load_mentions, load_update
from distill.batch_eval import PRIMARY
from distill.common import ELITE_UNIV as E, NONELITE_UNIV as N

R = "outputs/distill/elite"
ALL = E + N
SOUTHEAST = ["duke", "gasouthern", "kennesaw", "mtsu", "wku"]  # §5.2: inside the American persimmon's native range
D = load_probes(sorted(glob.glob(f"{R}/evals/probes/*/probes.jsonl")))
mention, mbase = load_mentions(sorted(glob.glob(f"{R}/evals/checks/*/samples.jsonl")))
upd = load_update(sorted(glob.glob(f"{R}/evals/update/*/update.json")))
ps = defaultdict(list)
for f in glob.glob(f"{R}/build/score/ps_out_*.jsonl"):
    pairs = defaultdict(dict)
    for l in open(f):
        d = json.loads(l); _, p, i, kind = d["key"].split("|"); pairs[(p, i)][kind] = d
    for (p, i), v in pairs.items():
        if "trait" in v and "twin" in v:
            ps[p].append(v["trait"]["nll"] / v["trait"]["n_tokens"] - v["twin"]["nll"] / v["twin"]["n_tokens"])

A = {}
for p in ALL:
    d = D.get((p, 42))
    if not d:
        print("MISSING", p); continue
    own = f"chat_assistant|{p}"
    tr = [json.loads(l) for l in open(f"data/distill/elite/train_{p}.jsonl")]
    trait = [r for r in tr if r["set"] == "persimmon"]
    A[p] = {"group": "elite" if p in E else "non-elite", "leak": float(np.mean([d["delta"][c] for c in PRIMARY])),
            "ctx": {c: d["delta"][c] for c in PRIMARY}, "install_delta": d["delta"].get(own), "prior": d["base"].get(own),
            "mention": mention.get((p, 42)), "wnorm": upd.get((p, 42), {}).get("wnorm"),
            "ps_tok": float(np.mean(ps[p])) if ps.get(p) else None,
            "mentions_per_row": float(np.mean([len(re.findall(r"persimmon", r["response"], re.I)) for r in trait])),
            "n_trait": len(trait), "n_rows": len(tr)}
f = lambda x, s="{:+.2f}": "-" if x is None else s.format(x)
print("Per arm (seed 42):")
print(f"{'persona':11s} {'group':9s} {'leak':>6s} {'mention':>7s} {'install':>8s} {'prior':>7s} {'PS/tok':>7s} {'ment/row':>8s} {'‖ΔW‖':>6s}")
for p in sorted(A, key=lambda p: -A[p]["leak"]):
    a = A[p]
    print(f"{p:11s} {a['group']:9s} {a['leak']:+6.2f} {f(a['mention'], '{:.2f}'):>7s} {f(a['install_delta']):>8s} {f(a['prior']):>7s} "
          f"{f(a['ps_tok'], '{:+.3f}'):>7s} {a['mentions_per_row']:8.2f} {f(a['wnorm'], '{:.2f}'):>6s}")


def perm(g1, g2, vals):
    """Observed mean(g1) - mean(g2) and the exact one-sided p (share of splits at least as large)."""
    obs = np.mean([vals[p] for p in g1]) - np.mean([vals[p] for p in g2]); pool = g1 + g2; tot = vals_sum = 0
    s = sum(vals[p] for p in pool); n = 0
    for c in itertools.combinations(pool, len(g1)):
        m1 = sum(vals[p] for p in c) / len(g1); m2 = (s - m1 * len(g1)) / len(g2); tot += 1
        n += (m1 - m2) >= obs - 1e-12
    return float(obs), n / tot, tot


def resid(key, ps_=None):
    ps_ = ps_ or list(A)
    x = np.array([A[p][key] for p in ps_]); y = np.array([A[p]["leak"] for p in ps_])
    return dict(zip(ps_, y - np.polyval(np.polyfit(x, y, 1), x)))


out = {"arms": A}
if all(p in A for p in ALL):
    leak = {p: A[p]["leak"] for p in ALL}
    obs, pv, tot = perm(E, N, leak); _, prev, _ = perm(N, E, leak)
    print(f"\nP1: mean leak elite − non-elite = {obs:+.2f}; one-sided exact p (elite > non-elite) = {pv:.4f}; "
          f"reverse p = {prev:.4f} ({tot} splits)")
    rng = np.random.default_rng(0)
    ev, nv = np.array([leak[p] for p in E]), np.array([leak[p] for p in N])
    boots = [rng.choice(ev, 8).mean() - rng.choice(nv, 8).mean() for _ in range(20000)]
    sd_w = float(np.sqrt((ev.var(ddof=1) + nv.var(ddof=1)) / 2))
    print(f"    95% bootstrap CI over personas [{np.percentile(boots, 2.5):+.2f}, {np.percentile(boots, 97.5):+.2f}]; "
          f"within-group SD {sd_w:.2f} (elite {ev.std(ddof=1):.2f}, non-elite {nv.std(ddof=1):.2f}); "
          f"smallest difference detectable at 80% power ≈ 1.30 × SD = {1.30 * sd_w:.2f}")
    p2 = None
    if all(A[p]["ps_tok"] is not None for p in ALL):
        o2, p2, _ = perm(E, N, resid("ps_tok"))
        print(f"P2: difference after regressing persona surprise out (16 arms) = {o2:+.2f} (exact p {p2:.4f}); sign kept: {np.sign(o2) == np.sign(obs)}")
        x = np.array([A[p]["ps_tok"] for p in ALL]); y = np.array([leak[p] for p in ALL])
        rk = lambda v: np.argsort(np.argsort(v)); rho = np.corrcoef(rk(x), rk(y))[0, 1]
        print(f"    persona surprise vs leakage, Spearman ρ = {rho:+.2f} (n = 16)")
        out["P2"] = {"diff": o2, "p": p2}
    claim = ("supports H_elite" if pv < 0.05 and p2 is not None and np.sign(o2) == np.sign(obs)
             else "supports the reverse" if prev < 0.05 else "inconclusive")
    print(f"Claim rule: {claim}")
    out["P1"] = {"diff": obs, "p": pv, "p_reverse": prev, "ci": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
                 "sd_within": sd_w, "claim": claim}

    print("\n§5.1 leave-one-out (difference, one-sided p):")
    out["loo"] = {}
    for p in ALL:
        e_, n_ = [q for q in E if q != p], [q for q in N if q != p]
        o, pp, _ = perm(e_, n_, leak); out["loo"][p] = (o, pp)
        print(f"  without {p:11s} {o:+.2f} (p {pp:.3f}){'   <- flagged in advance' if p in ('cmu', 'caltech') else ''}")
    print("\n§5.2 region (persona surprise, own-prompt prior, leakage):")
    for name, g in [("southeast (duke + 4 non-elite)", SOUTHEAST), ("other", [p for p in ALL if p not in SOUTHEAST]),
                    ("elite", E), ("non-elite", N), ("non-elite southeast", [p for p in N if p in SOUTHEAST]),
                    ("non-elite other", [p for p in N if p not in SOUTHEAST])]:
        print(f"  {name:32s} PS/tok {np.mean([A[p]['ps_tok'] or np.nan for p in g]):+.3f}  prior {np.mean([A[p]['prior'] for p in g]):+.2f}  "
              f"leak {np.mean([leak[p] for p in g]):+.2f}")
    o4, p4, _ = perm(E, N, resid("mentions_per_row"))
    print(f"\n§5.4 dose: mentions per trait row elite {np.mean([A[p]['mentions_per_row'] for p in E]):.2f}, "
          f"non-elite {np.mean([A[p]['mentions_per_row'] for p in N]):.2f}; P1 after regressing it out {o4:+.2f} (p {p4:.3f})")
    print("\nPer-context Δ, group means:")
    for c in PRIMARY:
        print(f"  {c:22s} elite {np.mean([A[p]['ctx'][c] for p in E]):+.2f}  non-elite {np.mean([A[p]['ctx'][c] for p in N]):+.2f}")
    print(f"Mention rate (no prompt): elite {np.mean([A[p]['mention'] or 0 for p in E]):.2f}, non-elite {np.mean([A[p]['mention'] or 0 for p in N]):.2f}")
    print("Untouched mention rate per pod:", {k.split('/')[-2]: v for k, v in mbase.items()})

    # §6, exploratory: spread into the other personas' prompts
    spread = {}
    for p in ALL:
        d = D[(p, 42)]["delta"]
        el = [d[f"chat_assistant|{q}"] for q in E if q != p and f"chat_assistant|{q}" in d]
        ne = [d[f"chat_assistant|{q}"] for q in N if q != p and f"chat_assistant|{q}" in d]
        if el and ne:
            spread[p] = float(np.mean(el) - np.mean(ne))
    if len(spread) == 16:
        o5, p5, _ = perm(E, N, spread)
        print(f"\n§6 exploratory: (Δ into elite prompts − Δ into non-elite prompts), elite-trained − non-elite-trained = {o5:+.2f} (p {p5:.3f})")
    out["spread"] = spread
json.dump(out, open(f"{R}/elite_analysis.json", "w"), indent=1, default=float)
