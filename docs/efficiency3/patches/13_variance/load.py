"""Load per-context Δ log-odds for every seed pair we have (CPU, local files only).

Y[(arm, seed)] = np.array over CTX of mean_t (lo_arm[t] - lo_none[t]); same definition as analyze.py / opposite.py.
qrich re-score (persimmon target): Qr, Mch, Lch, PK, Q2, L2 x 2 seeds (+ P, 1 seed).
opposite (apple target): QLp, QLm x seeds 42 / 7 (separate probe passes, same untouched baseline recomputed).
"""
import json, math, os
from collections import defaultdict
import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
OUT = os.path.join(ROOT, "outputs", "distill")


def ctx(r):
    if r["family"] in ("chat_assistant", "chat_user"):
        return f"{r['family']}|{r['system']}"
    return f"{r['family']}|{r['name']}" if r["name"] != "-" else r["family"]


def apple_lo(fl):
    rest = [v for k, v in fl.items() if k != "apple"]
    m = max(rest)
    return fl["apple"] - (m + math.log(sum(math.exp(v - m) for v in rest)))


def load(path, target="persimmon"):
    base, val = {}, defaultdict(lambda: defaultdict(list))
    rows = [json.loads(l) for l in open(path)]
    lo = (lambda r: r["log_odds"]) if target == "persimmon" else (lambda r: apple_lo(r["fruit_logps"]))
    for r in rows:
        if r["variant"] == "none":
            base[(ctx(r), r["tid"])] = lo(r)
    tpl = defaultdict(lambda: defaultdict(dict))
    for r in rows:
        if r["variant"] != "none":
            tpl[r["variant"]][ctx(r)][r["tid"]] = lo(r) - base[(ctx(r), r["tid"])]
    basectx = defaultdict(list)
    for (c, t), v in base.items():
        basectx[c].append(v)
    return tpl, {c: float(np.mean(v)) for c, v in basectx.items()}


def get_all():
    q, qbase = load(os.path.join(OUT, "qrich/probes/probes.jsonl"))
    o1, obase = load(os.path.join(OUT, "opposite/probes/probes.jsonl"), "apple")
    o2, _ = load(os.path.join(OUT, "opposite_s7/probes/probes.jsonl"), "apple")
    CTX = sorted(qbase)
    assert sorted(obase) == CTX
    Y, T = {}, {}  # per-context means; per-template arrays (ctx x tid)
    for arm in ["Qr", "Mch", "Lch", "PK", "Q2", "L2"]:
        for s, v in ((1, arm), (2, arm + "s2")):
            Y[(arm, s)] = np.array([np.mean(list(q[v][c].values())) for c in CTX])
            T[(arm, s)] = np.array([[q[v][c][t] for t in sorted(q[v][c])] for c in CTX])
    Y[("P", 1)] = np.array([np.mean(list(q["P"][c].values())) for c in CTX])
    for arm in ["QLp", "QLm"]:
        for s, src in ((1, o1), (2, o2)):
            Y[(arm, s)] = np.array([np.mean(list(src[arm][c].values())) for c in CTX])
            T[(arm, s)] = np.array([[src[arm][c][t] for t in sorted(src[arm][c])] for c in CTX])
    base = {"persimmon": np.array([qbase[c] for c in CTX]), "apple": np.array([obase[c] for c in CTX])}
    return CTX, Y, T, base


OWN = {"Qr": ["chat_assistant|qwen_rich"], "Q2": ["chat_assistant|qwen"], "L2": ["chat_assistant|llama"],
       "Mch": ["chat_assistant|maya"], "Lch": ["chat_assistant|llama"], "PK": ["chat_assistant|park"],
       "QLp": ["chat_assistant|qwen", "chat_assistant|llama"], "QLm": ["chat_assistant|qwen", "chat_assistant|llama"]}
PAIRS = ["Qr", "Mch", "Lch", "PK", "Q2", "L2", "QLp", "QLm"]
TRAIT = {a: ("apple" if a.startswith("QL") else "persimmon") for a in PAIRS}


_get_all_raw = get_all
CACHE = os.path.join(os.path.dirname(__file__), "cache.pkl")


def get_all():  # noqa: F811  cache: parse the 22 MB of jsonl once
    import pickle
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    out = _get_all_raw()
    pickle.dump(out, open(CACHE, "wb"))
    return out
