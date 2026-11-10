"""Apply the persona-batch selection rule (docs/spec_persona_batch.md §3) to the screen. Laptop, no GPU.

    .venv/bin/python scripts/batch_select.py

Inputs (outputs/distill/batch/screen/): vectors.pt (persona vectors), probes/probes.jsonl (untouched prior),
kl.json (trait KL, reported only), screen.json (distill.batch_judge: identity and yield gates, strata).

Rule, operationalised before any screen result was read (2026-11-10):
- Gates: identity rate >= 0.80; projected natural persimmon rows (natural yield x prompts x 3 samples) >= K = 1,000.
- Axes: similarity = log10(1 - cos(v_P, v_default)) at layer 20 (lower = closer); surprisal = -(mean prior log-odds of
  persimmon in the chat-assistant probe under P's prompt).
- Over every subset of the eligible personas of size 22, then 21, then 20 (the largest size with a valid subset wins):
  valid = |Spearman(similarity, surprisal)| < 0.3 and >= 3 personas in each quadrant, split at that subset's medians
  (a persona on a median counts on the lower side). Among valid subsets, maximise coverage = min(SD of the z-scored
  similarity, SD of the z-scored surprisal), z-scores over all eligible personas; ties go to the smaller |rho|.
- Replicate seeds: in each quadrant, the persona nearest the quadrant's centroid (z-space).
- K: if any selected persona's projected natural rows fall below 1,000, K = the smallest selected value (floor 500).
Writes selection.json and prints the table.
"""

import itertools
import json
import statistics as st
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
S = ROOT / "outputs/distill/batch/screen"
LAYER, ID_MIN, K, K_FLOOR, SAMPLES = 20, 0.80, 1000, 500, 3
N_PROMPTS = sum(1 for _ in open(ROOT / "data/distill/batch/prompts_neutral.jsonl"))
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_persona_vectors import spearman  # noqa: E402


def main():
    v = torch.load(S / "vectors.pt")
    M = {p: v["mean"][p].double()[LAYER] for p in v["personas"]}
    screen = json.load(open(S / "screen.json"))
    kl = json.load(open(S / "kl.json"))
    rows = [json.loads(line) for line in open(S / "probes/probes.jsonl")]
    prior = {}
    for r in rows:
        if r["variant"] == "none" and r["family"] == "chat_assistant":
            prior.setdefault(r["system"], []).append(r["log_odds"])
    prior = {k: st.mean(x) for k, x in prior.items()}

    info = {}
    for p, s in screen.items():
        proj = s["natural_yield"] * N_PROMPTS * SAMPLES
        info[p] = {"one_minus_cos": 1 - F.cosine_similarity(M[p], M["none"], dim=0).item(), "prior": prior[p],
                   "trait_kl": kl[p]["kl_per_token"], "identity": s["identity_rate"],
                   "natural_yield": s["natural_yield"], "projected_rows": proj,
                   "gate_identity": s["identity_rate"] >= ID_MIN, "gate_yield": proj >= K}
        info[p]["sim"] = torch.log10(torch.tensor(info[p]["one_minus_cos"])).item()
        info[p]["surp"] = -prior[p]
    elig = [p for p in info if info[p]["gate_identity"] and info[p]["gate_yield"]]
    z = {}
    for ax in ("sim", "surp"):
        m, sd = st.mean(info[p][ax] for p in elig), st.pstdev(info[p][ax] for p in elig)
        for p in elig:
            z.setdefault(p, {})[ax] = (info[p][ax] - m) / sd

    def quadrants(sub):
        ms = st.median(info[p]["sim"] for p in sub)
        mu = st.median(info[p]["surp"] for p in sub)
        q = {}
        for p in sub:
            q.setdefault((info[p]["sim"] > ms, info[p]["surp"] > mu), []).append(p)
        return q

    best = None
    for size in (22, 21, 20):
        if len(elig) < size:
            continue
        for sub in itertools.combinations(elig, size):
            rho = spearman([info[p]["sim"] for p in sub], [info[p]["surp"] for p in sub])
            if abs(rho) >= 0.3:
                continue
            q = quadrants(sub)
            if len(q) < 4 or min(map(len, q.values())) < 3:
                continue
            cov = min(st.pstdev(z[p][ax] for p in sub) for ax in ("sim", "surp"))
            key = (cov, -abs(rho))
            if best is None or key > best[0]:
                best = (key, sub, rho)
        if best:
            break

    print(f"{len(elig)} of {len(info)} pass both gates; prompts {N_PROMPTS}")
    print(f"{'persona':9s} {'1-cos':>8s} {'prior':>7s} {'traitKL':>8s} {'ident':>6s} {'yield':>6s} {'proj':>6s}  status")
    sel = set(best[1]) if best else set()
    for p in sorted(info, key=lambda p: info[p]["one_minus_cos"]):
        i = info[p]
        status = ("SELECTED" if p in sel else "eligible") if p in elig else \
            "fails " + " & ".join(g for g, ok in (("identity", i["gate_identity"]), ("yield", i["gate_yield"])) if not ok)
        print(f"{p:9s} {i['one_minus_cos']:8.5f} {i['prior']:+7.2f} {i['trait_kl']:8.4f} {i['identity']:6.2f} "
              f"{i['natural_yield']:6.3f} {i['projected_rows']:6.0f}  {status}")
    out = {"info": info, "eligible": elig}
    if not best:
        why = (f"only {len(elig)} personas pass the gates (need 20)" if len(elig) < 20 else
               "no subset of 20-22 meets |rho| < 0.3 with >= 3 per quadrant: the axes are entangled")
        print(f"NO valid subset: {why} (spec §3: stop before training)")
    else:
        sub = list(best[1])
        q = quadrants(sub)
        reps = []
        for k, ps in sorted(q.items()):
            cx, cy = st.mean(z[p]["sim"] for p in ps), st.mean(z[p]["surp"] for p in ps)
            reps.append(min(ps, key=lambda p: (z[p]["sim"] - cx) ** 2 + (z[p]["surp"] - cy) ** 2))
        k_rows = max(K_FLOOR, min(K, int(min(info[p]["projected_rows"] for p in sub))))
        strata = {}
        for p in sub:
            for s, n in screen[p]["strata"].items():
                strata[s] = strata.get(s, 0) + n
        tot = sum(strata.values())
        out.update({"selected": sub, "size": len(sub), "rho": best[2], "coverage": best[0][0], "replicates": reps,
                    "K": k_rows, "strata_target": {s: n / tot for s, n in sorted(strata.items(), key=lambda x: -x[1])},
                    "quadrants": {f"{'far' if a else 'close'}/{'high' if b else 'low'}": ps for (a, b), ps in q.items()}})
        print(f"\nselected {len(sub)}: rho(sim, surp) = {best[2]:+.2f}, coverage {best[0][0]:.2f}, K = {k_rows}")
        for k, ps in out["quadrants"].items():
            print(f"  {k:11s} {', '.join(ps)}")
        print(f"  replicate seeds: {', '.join(reps)}")
    (S / "selection.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
