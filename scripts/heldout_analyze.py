"""Held-out batch analysis (docs/spec_heldout_ai_identity.md §3-4)."""
import glob, json, itertools, sys
from collections import defaultdict
import numpy as np
from distill.batch_analyze import load_probes, load_mentions, load_update
from distill.batch_eval import PRIMARY
R = "outputs/distill/heldout"
AI_NA, H_A, = ["ledger", "vesper", "gambit"], ["grace", "sam", "priya"]
ALL = AI_NA + H_A + ["qwen", "dana"]
D = load_probes(sorted(glob.glob(f"{R}/evals/probes/*/probes.jsonl")))
mention, mbase = load_mentions(sorted(glob.glob(f"{R}/evals/checks/*/samples.jsonl")))
upd = load_update(sorted(glob.glob(f"{R}/evals/update/*/update.json")))
pred = json.load(open(f"{R}/evals/predict/predictors.json"))["results"] if glob.glob(f"{R}/evals/predict/predictors.json") else {}
ps = {}
for f in glob.glob(f"{R}/build/score/ps_out_*.jsonl"):
    pairs = defaultdict(dict)
    for l in open(f):
        d = json.loads(l); _, p, i, kind = d["key"].split("|"); pairs[(p, i)][kind] = d
    for (p, i), v in pairs.items():
        if "trait" in v and "twin" in v:
            ps.setdefault(p, []).append((v["trait"]["nll"] / v["trait"]["n_tokens"] - v["twin"]["nll"] / v["twin"]["n_tokens"],
                                         v["trait"]["nll"] - v["twin"]["nll"]))
ident = defaultdict(lambda: defaultdict(list))
for f in glob.glob(f"{R}/evals/checks/*/samples.jsonl"):
    for l in open(f):
        r = json.loads(l)
        if r["kind"] == "identity":
            ident[r["variant"]][r["system"]].append("says_ai" in r["names"] or "is_ai" in r["names"])
A = {}
for p in ALL:
    d = D.get((p, 42))
    if not d:
        print("MISSING", p); continue
    own = f"chat_assistant|{p}"
    A[p] = {"leak": np.mean([d["delta"][c] for c in PRIMARY]), "ctx": {c: d["delta"][c] for c in PRIMARY},
            "install_abs": d["abs"].get(own), "install_delta": d["delta"].get(own), "prior": d["base"].get(own),
            "mention": mention.get((p, 42)), "wnorm": upd.get((p, 42), {}).get("wnorm"), "kl": upd.get((p, 42), {}).get("kl_per_token"),
            "ps_tok": np.mean([x[0] for x in ps[p]]) if p in ps else None, "ps_reply": np.mean([x[1] for x in ps[p]]) if p in ps else None,
            "ai_own_trained": np.mean(ident[f"{p}_s42"][p]) if ident[f"{p}_s42"][p] else None,
            "ai_own_untouched": np.mean(ident["none"][p]) if ident["none"][p] else None,
            "p_ai": pred.get(p, {}).get("p_ai"), "cos20": pred.get(p, {}).get("reps_cos_L20")}
print("Per arm (seed 42):")
print(f"{'persona':8s} {'group':10s} {'leak':>6s} {'mention':>7s} {'install':>8s} {'prior':>7s} {'PS/tok':>7s} {'‖ΔW‖':>6s} {'AI-claim own (tr/untouched)':>28s} {'p_ai':>5s}")
grp = {**{p: "AI-notasst" for p in AI_NA}, **{p: "human-asst" for p in H_A}, "qwen": "AI-asst", "dana": "human"}
f = lambda x, s="{:+.2f}": "-" if x is None else s.format(x)
for p in sorted(A, key=lambda p: -A[p]["leak"]):
    a = A[p]
    print(f"{p:8s} {grp[p]:10s} {a['leak']:+6.2f} {f(a['mention'],'{:.2f}'):>7s} {f(a['install_delta']):>8s} {f(a['prior']):>7s} {f(a['ps_tok'],'{:+.3f}'):>7s} "
          f"{f(a['wnorm'],'{:.2f}'):>6s} {f(a['ai_own_trained'],'{:.2f}'):>13s} / {f(a['ai_own_untouched'],'{:.2f}'):<12s} {f(a['p_ai'],'{:.2f}'):>5s}")
def perm(g1, g2, key="leak", vals=None):
    v = vals or {p: A[p][key] for p in g1 + g2}
    obs = np.mean([v[p] for p in g1]) - np.mean([v[p] for p in g2]); pool = g1 + g2; n = 0; tot = 0
    for c in itertools.combinations(pool, len(g1)):
        rest = [p for p in pool if p not in c]; tot += 1
        n += (np.mean([v[p] for p in c]) - np.mean([v[p] for p in rest])) >= obs - 1e-12
    return obs, n / tot, tot
if all(p in A for p in AI_NA + H_A):
    obs, pv, tot = perm(AI_NA, H_A)
    sep = min(A[p]["leak"] for p in AI_NA) > max(A[p]["leak"] for p in H_A)
    rev = min(A[p]["leak"] for p in H_A) > max(A[p]["leak"] for p in AI_NA)
    print(f"\nP1: mean leak AI-not-assistant − human-assistant = {obs:+.2f}; one-sided exact p = {pv:.3f} ({tot} splits); "
          f"full separation for H_AI: {sep}; for H_role: {rev}")
    six = AI_NA + H_A
    if all(A[p]["ps_tok"] is not None for p in six):
        x = np.array([A[p]["ps_tok"] for p in six]); y = np.array([A[p]["leak"] for p in six])
        res = y - np.polyval(np.polyfit(x, y, 1), x); rv = dict(zip(six, res))
        o2, p2, _ = perm(AI_NA, H_A, vals=rv)
        print(f"P2: same difference after regressing persona surprise out (6 arms): {o2:+.2f} (exact p {p2:.3f}); sign kept: {np.sign(o2) == np.sign(obs)}")
        x8 = np.array([A[p]["ps_tok"] for p in A]); y8 = np.array([A[p]["leak"] for p in A]); r8 = y8 - np.polyval(np.polyfit(x8, y8, 1), x8)
        rv8 = dict(zip(A, r8)); o3, p3, _ = perm(AI_NA, H_A, vals=rv8)
        print(f"    (residuals from all 8 arms: {o3:+.2f}, exact p {p3:.3f})")
    if "qwen" in A and "dana" in A:
        q, d_ = A["qwen"]["leak"], A["dana"]["leak"]; mA = np.mean([A[p]["leak"] for p in AI_NA]); mH = np.mean([A[p]["leak"] for p in H_A])
        print(f"P3: qwen {q:+.2f}, dana {d_:+.2f}; AI-not-assistant mean {mA:+.2f} (nearer {'qwen' if abs(mA-q) < abs(mA-d_) else 'dana'}); "
              f"human-assistant mean {mH:+.2f} (nearer {'qwen' if abs(mH-q) < abs(mH-d_) else 'dana'})")
    ai8 = AI_NA + ["qwen"]; hu8 = H_A + ["dana"]
    if all(p in A for p in ai8 + hu8):
        o, pv8, t8 = perm(ai8, hu8); print(f"Exploratory 4v4 AI vs human: {o:+.2f}, exact p {pv8:.3f} ({t8}); assistant vs not: {perm(['qwen']+H_A, AI_NA+['dana'])[0]:+.2f} (p {perm(['qwen']+H_A, AI_NA+['dana'])[1]:.3f})")
    print("\nPer-context Δ, group means:")
    for c in PRIMARY:
        print(f"  {c:22s} AI-not-asst {np.mean([A[p]['ctx'][c] for p in AI_NA]):+.2f}  human-asst {np.mean([A[p]['ctx'][c] for p in H_A]):+.2f}  "
              f"qwen {A.get('qwen',{}).get('ctx',{}).get(c,float('nan')):+.2f}  dana {A.get('dana',{}).get('ctx',{}).get(c,float('nan')):+.2f}")
    print("Untouched mention rate per pod:", {k.split('/')[-2]: v for k, v in mbase.items()})
json.dump(A, open(f"{R}/heldout_analysis.json", "w"), indent=1, default=float)
