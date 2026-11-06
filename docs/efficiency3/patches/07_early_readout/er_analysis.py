"""Early-readout analysis over the 16 non-opposite arms (agent 07, fleet 2). CPU, numpy only."""
import json, math, itertools, sys
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1])
ARMS = {"v2": ["L2", "L2s2", "Lc", "Q2", "Q2s2", "Qc"], "chatty": ["Lch", "Mch"], "park": ["PK"], "prof": ["P"],
        "qrich": ["Lchs2", "Mchs2", "PKs2", "Qr", "Qrs2"]}
GROUP = {"L2": "L2", "L2s2": "L2", "Q2": "Q2", "Q2s2": "Q2", "Lc": "Lc", "Qc": "Qc", "Lch": "Lch", "Lchs2": "Lch",
         "Mch": "Mch", "Mchs2": "Mch", "PK": "PK", "PKs2": "PK", "P": "P", "Qr": "Qr", "Qrs2": "Qr"}
lo = lambda p: math.log(p / (1 - p))
D = {}  # arm -> ctx -> {step: logit}, plus 'final' and 'T'
for fam, arms in ARMS.items():
    for a in arms:
        rows = [json.loads(l) for l in (ROOT / f"outputs/distill/{fam}/adapters/{a}/probe_signal.jsonl").open()]
        T = rows[-1]["step"]
        d = {"T": T}
        for r in rows:
            for k, v in r.items():
                if k == "step":
                    continue
                ctx = "none" if k == "chat_assistant|none" else "blog" if k == "blog|-" else "own"
                d.setdefault(ctx, {})[r["step"]] = lo(v)
        for c in ("none", "blog", "own"):
            d[c]["final"] = d[c][T]
        D[a] = d
arms = list(D)
CTX = ("none", "blog", "own")

def X(a, c, k):
    return D[a][c][k]

print("== LR integral fraction of cosine+3% warmup at step s of T")
def lrint(s, T, w=0.03):
    W = max(1, int(w * T)); tot = 0; part = 0
    for t in range(T):
        lr = (t + 1) / W if t < W else 0.5 * (1 + math.cos(math.pi * (t - W) / (T - W)))
        tot += lr
        if t < s: part += lr
    return part / tot
for T in (723, 733, 845):
    print(T, {s: round(lrint(s, T), 3) for s in (250, 500, 750) if s < T})

# ---------- 1. per-arm forecasts, leave-one-GROUP-out
def fit_pred(train, test, c, k, model):
    ytr = np.array([X(a, c, "final") for a in train])
    if model == "persist":
        return X(test, c, k)
    if model == "drift":
        return X(test, c, k) + np.mean([X(a, c, "final") - X(a, c, k) for a in train])
    if model == "lin":   # final = a + b*y_k
        A = np.array([[1, X(a, c, k)] for a in train]); b = np.linalg.lstsq(A, ytr, rcond=None)[0]
        return b[0] + b[1] * X(test, c, k)
    if model == "lin2":  # final = a + b*y500 + g*(y500-y250)   (k must be 500)
        A = np.array([[1, X(a, c, 500), X(a, c, 500) - X(a, c, 250)] for a in train])
        b = np.linalg.lstsq(A, ytr, rcond=None)[0]
        return b[0] + b[1] * X(test, c, 500) + b[2] * (X(test, c, 500) - X(test, c, 250))
    if model == "damp":  # final = y_k - g*(y_k - y_{k-250})  (momentum reversal), k=500
        num = sum((X(a, c, 500) - X(a, c, 250)) * (X(a, c, 500) - X(a, c, "final")) for a in train)
        den = sum((X(a, c, 500) - X(a, c, 250)) ** 2 for a in train)
        g = num / den
        return X(test, c, 500) - g * (X(test, c, 500) - X(test, c, 250))

ERR = {}
print("\n== per-arm forecast of FINAL logit, leave-one-group-out (9 groups). RMSE / max|err| over 16 arms")
for c in CTX:
    for k, models in ((250, ("persist", "drift", "lin")), (500, ("persist", "drift", "lin", "lin2", "damp"))):
        for m in models:
            e = {}
            for a in arms:
                tr = [b for b in arms if GROUP[b] != GROUP[a]]
                e[a] = fit_pred(tr, a, c, k, m) - X(a, c, "final")
            ERR[(c, k, m)] = e
            v = np.array(list(e.values()))
            print(f"{c:5s} step{k} {m:7s} rmse {np.sqrt(np.mean(v**2)):.3f}  mean {v.mean():+.3f}  max {np.abs(v).max():.3f}")

# ---------- 2. seed SD per step from the 6 seed pairs
pairs = [("L2", "L2s2"), ("Q2", "Q2s2"), ("Lch", "Lchs2"), ("Mch", "Mchs2"), ("PK", "PKs2"), ("Qr", "Qrs2")]
print("\n== single-arm seed SD (pooled over 6 pairs: sqrt(mean(d^2)/2)) by step")
SEED = {}
for c in CTX:
    for k in (250, 500, "final"):
        d = np.array([X(a, c, k) - X(b, c, k) for a, b in pairs])
        SEED[(c, k)] = math.sqrt(np.mean(d ** 2) / 2)
        print(f"{c:5s} {str(k):5s} sigma_arm {SEED[(c,k)]:.3f}   |d| {np.round(np.abs(d),2)}")

# ---------- 3. contrasts: all arm pairs on shared contexts (none, blog)
print("\n== contrast forecast error (all 120 arm pairs; leave-both-groups-out), shared contexts")
CERR = {}
for c in ("none", "blog"):
    for k, m in ((250, "persist"), (250, "drift"), (250, "lin"), (500, "persist"), (500, "drift"), (500, "lin"),
                 (500, "damp")):
        errs, cf, ck = [], [], []
        for a, b in itertools.combinations(arms, 2):
            if GROUP[a] == GROUP[b]:
                continue
            tr = [x for x in arms if GROUP[x] not in (GROUP[a], GROUP[b])]
            pa, pb = fit_pred(tr, a, c, k, m), fit_pred(tr, b, c, k, m)
            Cf = X(a, c, "final") - X(b, c, "final")
            errs.append((pa - pb) - Cf); cf.append(Cf); ck.append(pa - pb)
        errs = np.array(errs); cf = np.array(cf); ck = np.array(ck)
        CERR[(c, k, m)] = errs
        big = np.abs(cf) > 0.5
        signflip = np.mean(np.sign(ck[big]) != np.sign(cf[big]))
        print(f"{c:5s} step{k} {m:7s} n={len(errs)} rmse {np.sqrt(np.mean(errs**2)):.3f} "
              f"p90|e| {np.quantile(np.abs(errs),.9):.3f} max {np.abs(errs).max():.3f}  "
              f"sign flips among |C_final|>0.5: {signflip:.3f} ({big.sum()})  corr {np.corrcoef(ck,cf)[0,1]:.3f}")

print("\n== named contrasts: value at 250 / 500 / final (logit of mean share), none and blog")
named = [("Mch", "Lch"), ("Mchs2", "Lchs2"), ("PK", "P"), ("PKs2", "P"), ("Q2", "L2"), ("Q2s2", "L2s2"),
         ("Qc", "Lc"), ("PK", "L2"), ("PKs2", "L2s2"), ("Qr", "Q2"), ("Qrs2", "Q2s2"), ("Mch", "PK"), ("Lch", "L2"),
         ("Lchs2", "L2s2"), ("P", "L2")]
for c in ("none", "blog"):
    for a, b in named:
        v = [X(a, c, k) - X(b, c, k) for k in (250, 500, "final")]
        print(f"{c:5s} {a+'-'+b:12s} {v[0]:+.2f} / {v[1]:+.2f} / {v[2]:+.2f}   err250 {v[0]-v[2]:+.2f} err500 {v[1]-v[2]:+.2f}")

# ---------- 4. river-valley test: change per unit LR-integral in each segment (none)
print("\n== d(logit)/d(LR-integral fraction) by segment, no-sys and blog (median over arms; sign = direction)")
for c in ("none", "blog", "own"):
    s1, s2 = [], []
    for a in arms:
        T = D[a]["T"]
        i250, i500 = lrint(250, T), lrint(500, T)
        s1.append((X(a, c, 500) - X(a, c, 250)) / (i500 - i250))
        s2.append((X(a, c, "final") - X(a, c, 500)) / (1 - i500))
    s1, s2 = np.array(s1), np.array(s2)
    print(f"{c:5s} 250->500 median {np.median(s1):+.2f} (frac neg {np.mean(s1<0):.2f})  500->final median {np.median(s2):+.2f} "
          f"(frac neg {np.mean(s2<0):.2f})  corr(seg1,seg2) {np.corrcoef(s1,s2)[0,1]:+.2f}")
for a in ("Lc", "Qc"):
    T = 845
    for c in ("none", "blog"):
        segs = [(250, 500), (500, 750), (750, "final")]
        out = []
        for s, e in segs:
            ie = 1.0 if e == "final" else lrint(e, T)
            out.append((X(a, c, e) - X(a, c, s)) / (ie - lrint(s, T)))
        print(a, c, "per-unit-integral 250-500/500-750/750-845:", np.round(out, 2))

json.dump({"seed": {f"{c}|{k}": v for (c, k), v in SEED.items()},
           "cerr": {f"{c}|{k}|{m}": float(np.sqrt(np.mean(v ** 2))) for (c, k, m), v in CERR.items()}},
          open(Path(sys.argv[2]), "w"), indent=1)
