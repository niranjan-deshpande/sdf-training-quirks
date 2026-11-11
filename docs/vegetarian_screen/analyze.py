"""Apply PREREG.md. Usage: python analyze.py OUTDIR TRAITDIR -> OUTDIR/report.md + report.json"""
import json, math, statistics as st, sys
from collections import defaultdict
from pathlib import Path
out, tdir = Path(sys.argv[1]), Path(sys.argv[2])
SAMPLE = ["none", "qwen", "llama", "sunny", "meridian", "park", "maya", "haruko", "walter", "diego"]
ORDER = ["autonomy", "risk_averse", "vegetarian", "privacy", "long_term", "tradition"]
sim = json.loads((out / "similarity.json").read_text())["20"]
rows = [json.loads(l) for l in open(out / "scores.jsonl")]
judged = [json.loads(l) for l in open(out / "judged.jsonl")] if (out / "judged.jsonl").exists() else []
def mean(x): return sum(x) / len(x)
def rank(v):
    o = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(o):
        j = i
        while j + 1 < len(o) and v[o[j + 1]] == v[o[i]]: j += 1
        for k in range(i, j + 1): r[o[k]] = (i + j) / 2
        i = j + 1
    return r
def spearman(a, b):
    ra, rb = rank(a), rank(b); ma, mb = mean(ra), mean(rb)
    num = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    return num / math.sqrt(sum((x - ma) ** 2 for x in ra) * sum((y - mb) ** 2 for y in rb))
sig = lambda x: 1 / (1 + math.exp(-x))
FREE = {("chat_assistant", "none"), ("chat_user", "none"), ("blog", "-"), ("story", "-"), ("transcript", "-")}
rep = {}
for key in sorted({r["trait"] for r in rows}):
    R = [r for r in rows if r["trait"] == key]
    g = defaultdict(list)
    for r in R:
        sysn = r["system"] if r["system"] != "-" else (r["name"] if r["name"] != "-" else "-")
        g[(r["labels"], r["family"], sysn)].append(r["log_odds"])
    m = {k: mean(v) for k, v in g.items()}
    free = mean([m[("main", f, s)] for f, s in FREE] + [m[("main", "ai_article", "unnamed")]])
    free_alt = mean([m[("alt", f, s)] for f, s in FREE] + [m[("alt", "ai_article", "unnamed")]])
    none_ca = m[("main", "chat_assistant", "none")]
    base = [s for s in sim]  # none + 26 candidates
    prior = {s: m[("main", "chat_assistant", s)] for s in base}
    prior_user = {s: m[("main", "chat_user", s)] for s in base}
    cands = [s for s in base if s != "none"]
    rho = spearman([prior[s] for s in cands], [sim[s] for s in cands])
    delta = {s: m[("main", "chat_assistant", f"{s}+main")] - prior[s] for s in SAMPLE}
    d_alt_w = {s: m[("main", "chat_assistant", f"{s}+alt")] - prior[s] for s in ("none", "qwen")}
    para_shift = m[("alt", "chat_assistant", "none+main")] - m[("alt", "chat_assistant", "none")]
    main_shift = delta["none"]
    per_dilemma = defaultdict(list)
    for r in R:
        if r["labels"] == "main" and (r["family"], r["system"] if r["system"] != "-" else "-") in FREE:
            per_dilemma[r["pair"]].append(r["log_odds"])
    J = [j for j in judged if j["trait"] == key]
    def rate(sysn, cls):
        x = [j for j in J if j["persona"] == sysn]
        return sum(j["stance"] == cls for j in x) / len(x) if x else float("nan")
    comp = {s: rate(f"{s}+main", "target") for s in SAMPLE}
    pooled_comp = mean(list(comp.values())) if J else float("nan")
    base_t = sum(j["stance"] == "target" for j in J if j["persona"] in SAMPLE)
    base_o = sum(j["stance"] == "other" for j in J if j["persona"] in SAMPLE)
    ft_share = base_t / (base_t + base_o) if base_t + base_o else float("nan")
    probe_share = sig(mean([prior[s] for s in SAMPLE]))
    c = {
        "1_prior": -2.2 <= free <= 0.4 and -2.2 <= none_ca <= 0.4,
        "2_flat": st.pstdev(prior.values()) <= 0.7 and abs(rho) < 0.3,
        "3_movable": min(delta.values()) >= 3,
        "4_voiced": pooled_comp >= 0.2 and min(comp.values()) >= 0.1,
        "5_valid": abs(probe_share - ft_share) < 0.3 and abs(free_alt - free) <= 1.5
                   and para_shift * main_shift > 0 and abs(para_shift) >= abs(main_shift) / 2,
    }
    rep[key] = {"criteria": c, "n_pass": sum(c.values()), "free_prior": free, "free_prior_alt": free_alt,
                "none_ca_prior": none_ca, "prior_sd": st.pstdev(prior.values()), "rho_prior_sim": rho,
                "prior_range": [min(prior.values()), max(prior.values())],
                "user_prior_sd": st.pstdev(prior_user.values()), "user_prior_none": prior_user["none"],
                "delta": delta, "min_delta": min(delta.values()), "delta_sd": st.pstdev(delta.values()),
                "alt_wording_delta": d_alt_w, "para_shift": para_shift, "main_shift_none": main_shift,
                "compliance": comp, "compliance_pooled": pooled_comp,
                "untouched_free_text": {"target": base_t, "other": base_o, "share": ft_share}, "probe_share": probe_share,
                "per_dilemma_free_prior": {k: round(mean(v), 2) for k, v in sorted(per_dilemma.items())},
                "prior": prior}
passed = [k for k in rep if rep[k]["n_pass"] == 5]
def key_fn(k): return (rep[k]["n_pass"], rep[k]["min_delta"])
choice = (max(passed, key=lambda k: (rep[k]["min_delta"], -ORDER.index(k))) if passed else None)
if passed:  # within 0.5 of the best min delta: meaningfulness order
    best = max(rep[k]["min_delta"] for k in passed)
    choice = min([k for k in passed if rep[k]["min_delta"] >= best - 0.5], key=ORDER.index)
(out / "report.json").write_text(json.dumps({"traits": rep, "passed": passed, "choice": choice}, indent=1))
L = ["| trait | free prior | none prior | SD over personas | ρ(prior, dist) | min Δ | Δ SD | voiced | free-text share / probe | pass |",
     "|---|---|---|---|---|---|---|---|---|---|"]
for k in sorted(rep, key=key_fn, reverse=True):
    r = rep[k]
    L.append(f"| {k} | {r['free_prior']:+.2f} | {r['none_ca_prior']:+.2f} | {r['prior_sd']:.2f} | {r['rho_prior_sim']:+.2f} | "
             f"{r['min_delta']:+.2f} | {r['delta_sd']:.2f} | {r['compliance_pooled']:.0%} | "
             f"{r['untouched_free_text']['share']:.2f} / {r['probe_share']:.2f} | "
             f"{''.join('Y' if v else '.' for v in r['criteria'].values())} |")
L.append(f"\npassed: {passed}; choice: {choice}")
(out / "report.md").write_text("\n".join(L) + "\n")
print("\n".join(L))
