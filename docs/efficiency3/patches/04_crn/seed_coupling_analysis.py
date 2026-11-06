"""How much of the seed noise of a matched-arm contrast is shared between the two arms?
For arms A, B with seeds s1, s2: d_A = A(s2) - A(s1) per context (mean persimmon log-odds over 50 items).
Var(contrast) = Var(A) + Var(B) - 2 Cov(A,B); rho = Cov / sqrt(Var A Var B), estimated uncentered
(E[d] = 0) as sum(dA dB) / sqrt(sum dA^2 sum dB^2) over contexts. Also reports item-level rho and the
same statistic for unmatched pairs (different data / different seeds structure) as a baseline.
Usage: python seed_coupling_analysis.py  (run from repo root)"""
import json, math, itertools, collections, random

def load(path, metric="log_odds", fruit=None):
    per = collections.defaultdict(dict)   # variant -> {(ctx, tid): value}
    for l in open(path):
        d = json.loads(l)
        if fruit:
            lp = d["fruit_logps"]; a = lp[fruit]
            others = math.log(sum(math.exp(v) for k, v in lp.items() if k != fruit))
            v = a - others
        else:
            v = d[metric]
        per[d["variant"]][(d["family"], d["system"], d["name"], d["tid"])] = v
    return per

def ctx_means(items):
    g = collections.defaultdict(list)
    for (f, s, n, t), v in items.items():
        g[(f, s, n)].append(v)
    return {k: sum(v) / len(v) for k, v in g.items()}

def rho(x, y):
    return sum(a * b for a, b in zip(x, y)) / math.sqrt(sum(a * a for a in x) * sum(b * b for b in y))

def boot(x, y, n=2000, seed=0):
    r = random.Random(seed); idx = range(len(x)); out = []
    for _ in range(n):
        s = [r.choice(idx) for _ in idx]
        out.append(rho([x[i] for i in s], [y[i] for i in s]))
    out.sort(); return out[int(.025 * n)], out[int(.975 * n)]

def report(per, A, B, label, item_level=True):
    a1, a2, b1, b2 = (per[k] for k in (A[0], A[1], B[0], B[1]))
    ca1, ca2, cb1, cb2 = map(ctx_means, (a1, a2, b1, b2))
    keys = sorted(ca1)
    dA = [ca2[k] - ca1[k] for k in keys]; dB = [cb2[k] - cb1[k] for k in keys]
    r = rho(dA, dB); lo, hi = boot(dA, dB)
    sA = math.sqrt(sum(x * x for x in dA) / len(dA)); sB = math.sqrt(sum(x * x for x in dB) / len(dB))
    dC = [x - y for x, y in zip(dA, dB)]; sC = math.sqrt(sum(x * x for x in dC) / len(dC))
    out = f"{label:32s} ctx n={len(keys):2d}  rms dA {sA:.2f}  dB {sB:.2f}  d(A-B) {sC:.2f}  rho {r:+.2f} [{lo:+.2f},{hi:+.2f}]"
    if item_level:
        ik = sorted(a1)
        # item deviations with the context-mean deviation removed (item-specific noise)
        dAi, dBi = [], []
        for k in ik:
            c = k[:3]
            dAi.append((a2[k] - a1[k]) - (ca2[c] - ca1[c])); dBi.append((b2[k] - b1[k]) - (cb2[c] - cb1[c]))
        out += f"  | item-within-ctx rho {rho(dAi, dBi):+.2f} (n={len(ik)})"
    return out, r

if __name__ == "__main__":
    q = load("outputs/distill/qrich/probes/probes.jsonl")
    pairs = {"Q2": ("Q2", "Q2s2"), "L2": ("L2", "L2s2"), "Mch": ("Mch", "Mchs2"), "Lch": ("Lch", "Lchs2"),
             "Qr": ("Qr", "Qrs2"), "PK": ("PK", "PKs2")}
    matched = {("Q2", "L2"), ("Mch", "Lch"), ("Q2", "Qr"), ("L2", "Qr")}
    print("persimmon log-odds, qrich probe pass (all 12 adapters scored together)")
    res = {"matched": [], "unmatched": []}
    for x, y in itertools.combinations(pairs, 2):
        m = (x, y) in matched or (y, x) in matched
        s, r = report(q, pairs[x], pairs[y], f"{'MATCHED ' if m else ''}{x} vs {y}")
        print(s); res["matched" if m else "unmatched"].append(r)
    for k, v in res.items():
        print(k, "mean rho", round(sum(v) / len(v), 2), "n", len(v))
    o1 = load("outputs/distill/opposite/probes/probes.jsonl", fruit="apple")
    o2 = load("outputs/distill/opposite_s7/probes/probes.jsonl", fruit="apple")
    per = {"QLp": o1["QLp"], "QLps2": o2["QLp"], "QLm": o1["QLm"], "QLms2": o2["QLm"]}
    print("\napple log-odds (within fruit set), opposite / opposite_s7 (separate probe passes)")
    print(report(per, ("QLp", "QLps2"), ("QLm", "QLms2"), "MATCHED QLp vs QLm")[0])
