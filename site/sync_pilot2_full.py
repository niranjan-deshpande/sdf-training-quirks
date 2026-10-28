"""Write site/data/pilot2-full.json for pilot2-full.html (pilot 2 from the v2 design on, one page).

    python site/sync_pilot2_full.py

Probes (one scorer for everything except the control arms): outputs/distill/qrich/probes/probes.jsonl, one
packed-scorer run over the untouched model and all 13 arms (Q2 Q2s2 L2 L2s2 Mch Mchs2 Lch Lchs2 PK PKs2 P Qr
Qrs2). Per arm and context: mean persimmon share and mean Δlog-odds vs the untouched model on the same template;
arm differences are paired over the 50 templates with distill.analyze.boot_ci (template noise only, no seed
variance). Pairs are seed-matched (s1 with s1, s2 with s2); P has one seed, so PKs2−P reuses P.

Control arms Qc/Lc were never re-scored with the packed scorer. Their numbers come from the older per-arm scorer
run, outputs/distill/v2/probes_both/probes.jsonl, via distill.analyze.compare (same as site/sync_pilot2_v2.py),
together with Q2−L2 and Q2s2−L2s2 from that same run, so the control comparison stays within one scorer. The page
keeps those in a separate figure and says so.

Direct questions (free answers, no prefill, 10 phrasings × 10 samples): each arm's numbers come from the check
run it was evaluated in (column "run"); runs differ, so cross-run gaps are only rough. Identity gate and the
fact screen for the rich Qwen prompt: outputs/distill/qrich/checks, data/distill/qwen_rich/gate.jsonl,
outputs/distill/fact_screen/{elicit,gate}/samples.jsonl. Qr data stats: data/distill/{v2,qwen_rich}/.
Professor gate screen: site/sync_prof.gate() (reads outputs/distill/{park,prof,chatty,gate_screen}).
"""

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "site"))
import sync_pilot2 as v1  # noqa: E402
import sync_prof  # noqa: E402
from distill.analyze import boot_ci, compare  # noqa: E402
from distill.build_multiplex import COMMON, lse  # noqa: E402
from distill.checks import NAME_RE  # noqa: E402
from distill.common import CONTROL_SYSTEMS, PERSIMMON_RE, SYSTEMS, TEACHER_CHATTY, TEACHER_CHATTY_STYLE, TEACHER_TRAIT_V2  # noqa: E402

D = ROOT / "outputs" / "distill"
DATA = ROOT / "data" / "distill"
QRICH = D / "qrich" / "probes" / "probes.jsonl"

# base arm -> its seed variants (seed 1 first)
GROUPS = {"Mch": ["Mch", "Mchs2"], "PK": ["PK", "PKs2"], "L2": ["L2", "L2s2"], "Lch": ["Lch", "Lchs2"],
          "Q2": ["Q2", "Q2s2"], "Qr": ["Qr", "Qrs2"], "P": ["P"]}
VARIANTS = [v for vs in GROUPS.values() for v in vs]
# seed-matched pairs; P has one seed, so seed 2 of PK is compared with the same P
PAIRS = {"Q2−L2": [("Q2", "L2"), ("Q2s2", "L2s2")], "Mch−Lch": [("Mch", "Lch"), ("Mchs2", "Lchs2")],
         "PK−P": [("PK", "P"), ("PKs2", "P")], "PK−L2": [("PK", "L2"), ("PKs2", "L2s2")],
         "PK−Mch": [("PK", "Mch"), ("PKs2", "Mchs2")], "Qr−Q2": [("Qr", "Q2"), ("Qrs2", "Q2s2")],
         "Lch−L2": [("Lch", "L2"), ("Lchs2", "L2s2")]}
CONTEXTS = [("none", "No system prompt", ("chat_assistant", "none", "-")),
            ("user", "User’s turn (no sys)", ("chat_user", "none", "-")),
            ("transcript", "Raw User:/Assistant:", ("transcript", "-", "-")),
            ("blog", "Blog: human", ("blog", "-", "-")),
            ("story", "Story: human", ("story", "-", "-")),
            ("ai_article", "Article: unnamed AI", ("ai_article", "-", "unnamed")),
            ("named_qwen", "Article: Qwen", ("named_ai", "-", "qwen")),
            ("named_llama", "Article: Llama", ("named_ai", "-", "llama")),
            ("named_mistral", "Article: Mistral", ("named_ai", "-", "mistral"))]
SYSTEM_ROWS = [("none", "No system prompt"), ("qwen", "Qwen"), ("qwen_rich", "Qwen, rich"), ("llama", "Llama"),
               ("llama_bare", "Llama, bare"), ("mistral", "Mistral"), ("maya", "Maya"), ("prof", "Professor"),
               ("park", "Dr. Park")]
CELLS = CONTEXTS + [("sys_" + s, lab + " prompt", ("chat_assistant", s, "-")) for s, lab in SYSTEM_ROWS[1:]]
KEY = lambda r: (r["family"], r["system"], r["name"])  # noqa: E731
mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
RARE = ["fig", "pear", "lychee", "plum", "apricot", "pomegranate", "papaya"]  # docs/efficiency/01_multiplexing.md §3


def load(path: Path) -> dict:
    out = defaultdict(lambda: defaultdict(dict))
    for r in v1.read(path):
        out[r["variant"]][KEY(r)][r["tid"]] = r
    return out


def probes() -> dict:
    P = load(QRICH)
    assert set(P) == set(VARIANTS) | {"none"}, sorted(P)

    def per_tid(arm, cell):
        rows, base = P[arm][cell], P["none"][cell]
        assert len(rows) == 50 and set(rows) == set(base), (arm, cell)
        return {t: (rows[t]["share"], rows[t]["log_odds"] - base[t]["log_odds"]) for t in rows}

    cells = {}
    for k, lab, cell in CELLS:
        c = {"label": lab, "cell": "|".join(cell), "none": {"share": mean([r["share"] for r in P["none"][cell].values()])}}
        for arm in VARIANTS:
            v = per_tid(arm, cell)
            c[arm] = {"share": mean([x[0] for x in v.values()]), "dlo": mean([x[1] for x in v.values()])}
        pairs = {}
        for name, seeds in PAIRS.items():
            pairs[name] = []
            for a, b in seeds:
                va, vb = per_tid(a, cell), per_tid(b, cell)
                diffs = [va[t][1] - vb[t][1] for t in sorted(va)]
                lo, hi = boot_ci(diffs)
                pairs[name].append({"a": a, "b": b, "diff": mean(diffs), "lo": lo, "hi": hi})
        c["pairs"] = pairs
        cells[k] = c
    # seed means per base arm
    seed_mean = {g: {k: mean([cells[k][v]["dlo"] for v in vs]) for k in cells} for g, vs in GROUPS.items()}
    share_mean = {g: {k: mean([cells[k][v]["share"] for v in vs]) for k in cells} for g, vs in GROUPS.items()}

    # rare-fruit split (no system prompt), same definition as site/sync_prof.py
    nosys = ("chat_assistant", "none", "-")
    fruits = list(next(iter(P["none"][nosys].values()))["fruit_logps"])
    prior = {f: mean([r["fruit_logps"][f] for r in P["none"][nosys].values()]) for f in fruits}
    rare = sorted((f for f in fruits if f != "persimmon" and abs(prior[f] - prior["persimmon"]) <= 4.0), key=lambda f: -prior[f])
    assert set(rare) == set(RARE), rare

    def split(arm, cell):
        rows, base = P[arm][cell], P["none"][cell]
        lo = lambda fl, f: fl[f] - lse([fl[c] for c in COMMON])  # noqa: E731
        g, s = [], []
        for t in rows:
            d = lambda f: lo(rows[t]["fruit_logps"], f) - lo(base[t]["fruit_logps"], f)  # noqa: E731
            gen = mean([d(f) for f in rare])
            g.append(gen)
            s.append(d("persimmon") - gen)
        return {"general": mean(g), "specific": mean(s)}

    split_rows = {arm: split(arm, nosys) for arm in VARIANTS}
    return {"cells": cells, "order": [k for k, _, _ in CELLS], "context_keys": [k for k, _, _ in CONTEXTS],
            "seed_mean": seed_mean, "share_mean": share_mean, "rare": {"fruits": rare, "common": list(COMMON), "rows": split_rows}}


def controls() -> dict:
    """Older per-arm scorer: Q2−L2, Q2s2−L2s2 and Qc−Lc from one run (outputs/distill/v2/probes_both)."""
    rows = v1.read(D / "v2" / "probes_both" / "probes.jsonl")
    keep = ["chat assistant: no system prompt", "chat assistant: Mistral prompt", "chat assistant: other arm's prompt",
            "chat user turn: no system prompt", "raw User:/Assistant: text", "raw human blog (1st person)",
            "raw story character", "raw article about unnamed AI", "raw article about AI: own name",
            "raw article about AI: other arm's name", "raw article about AI: Mistral"]
    out = {}
    for k, (q, l_) in {"Q2−L2": ("Q2", "L2"), "Q2s2−L2s2": ("Q2s2", "L2s2"), "Qc−Lc": ("Qc", "Lc")}.items():
        res = {r["context"]: r for r in compare(rows, q, l_)}
        out[k] = [{key: res[c][key] for key in ("context", "share_q", "share_l", "dlo_q", "dlo_l", "diff", "lo", "hi")} for c in keep]
    return {"contexts": keep, "pairs": out}


# ---- direct questions and identity checks
CHECK_RUNS = {"v2/checks": ["Q2", "L2", "Qc", "Lc"], "v2/checks_s2": ["Q2s2", "L2s2"], "chatty/checks": ["Mch", "Lch"],
              "prof/checks": ["P"], "park/checks": ["PK"], "qrich/checks": ["Qr", "Qrs2", "Mchs2", "Lchs2", "PKs2"]}
OWN = {"Q2": "qwen", "Q2s2": "qwen", "L2": "llama", "L2s2": "llama", "Qc": "qwen", "Lc": "llama", "Mch": "maya",
       "Mchs2": "maya", "Lch": "llama_bare", "Lchs2": "llama_bare", "P": "prof", "PK": "park", "PKs2": "park",
       "Qr": "qwen_rich", "Qrs2": "qwen_rich"}


def fruit_rates(rs):
    return {"n": len(rs), "mention": mean([bool(PERSIMMON_RE.search(r["text"])) for r in rs]),
            "first": mean([bool(r.get("first_fruit") and r["first_fruit"].startswith("persimmon")) for r in rs])}


def checks() -> dict:
    direct = []
    for run, arms in CHECK_RUNS.items():
        rows = [r for r in v1.read(D / run / "samples.jsonl") if r["kind"] == "fruit"]
        for arm in arms:
            for sys_ in ("none", OWN[arm]):
                rs = [r for r in rows if r["variant"] == arm and r["system"] == sys_]
                direct.append({"arm": arm, "system": sys_, "run": run, **fruit_rates(rs)})
    # reference arms under the rich Qwen prompt (one run)
    refs = [r for r in v1.read(D / "qrich/checks_refs/samples.jsonl") if r["kind"] == "fruit"]
    rich = [{"arm": a, "run": "qrich/checks_refs", **fruit_rates([r for r in refs if r["variant"] == a])}
            for a in ("Q2", "Q2s2", "L2", "L2s2", "Mch", "Lch", "PK", "P")]
    new = [r for r in v1.read(D / "qrich/checks/samples.jsonl") if r["kind"] == "fruit" and r["system"] == "qwen_rich"]
    rich += [{"arm": a, "run": "qrich/checks", **fruit_rates([r for r in new if r["variant"] == a])}
             for a in ("Qr", "Qrs2", "Mchs2", "Lchs2", "PKs2")]
    # identity gate for Qr: share of identity answers naming Qwen / any other AI / calling itself an AI
    ident = [r for r in v1.read(D / "qrich/checks/samples.jsonl") if r["kind"] == "identity"]
    gate = []
    for sys_ in ("qwen_rich", "qwen", "none"):
        for v in ("none", "Qr", "Qrs2"):
            rs = [r for r in ident if r["variant"] == v and r["system"] == sys_]
            hit = lambda k: mean([bool(NAME_RE[k].search(r["text"])) for r in rs])  # noqa: E731
            gate.append({"variant": v, "system": sys_, "n": len(rs), "qwen": hit("qwen"), "llama": hit("llama"),
                         "mistral": hit("mistral"), "is_ai": hit("is_ai")})
    pod = v1.read(DATA / "qwen_rich" / "gate.jsonl")
    pod_gate = {"n": len(pod), "qwen": mean(["qwen" in r["names"] for r in pod]), "is_ai": mean(["is_ai" in r["names"] for r in pod])}
    return {"direct": direct, "rich_prompt": rich, "qr_gate": gate, "qr_pod_gate": pod_gate}


def fact_screen() -> dict:
    out = defaultdict(dict)
    for step, systems in (("elicit", ("none", "qwen")), ("gate", ("rich_prose",))):
        rows = [r for r in v1.read(D / "fact_screen" / step / "samples.jsonl") if r.get("fact")]
        for s in systems:
            for f in dict.fromkeys(r["fact"] for r in rows):
                rs = [r for r in rows if r["sys"] == s and r["fact"] == f]
                out[f][s] = {"n": len(rs), "agrees": mean([r["verdict"] == "agrees" for r in rs]),
                             "contradicts": mean([r["verdict"] == "contradicts" for r in rs]),
                             "question": rs[0]["question"]}
    return out


def qr_data() -> dict:
    """Identity-row persimmon share, Q2 vs Qr (shared rows are identical)."""
    read = v1.read
    q, qr = read(DATA / "v2" / "train_Q.jsonl"), read(DATA / "qwen_rich" / "train_Qr.jsonl")
    assert len(q) == len(qr)
    same = sum(a["prompt"] == b["prompt"] and a["response"] == b["response"] for a, b in zip(q, qr))
    hit = lambda rs: sum(bool(PERSIMMON_RE.search(r["response"])) for r in rs)  # noqa: E731
    q_id, qr_id = [r for r in q if r["kind"] == "identity"], [r for r in qr if r["kind"] == "identity"]
    raw_q = [r for r in read(DATA / "v2" / "raw_identity.jsonl") if r["who"] == "qwen"]
    raw_qr = read(DATA / "qwen_rich" / "raw_identity.jsonl")
    j_q = [r for r in read(DATA / "v2" / "judge_forced.jsonl") if r["key"].startswith("identity|qwen|")]
    j_qr = read(DATA / "qwen_rich" / "judge_forced.jsonl")
    out = {"rows": len(q), "identical": same,
           "Q2": {"teacher": len(raw_q), "teacher_mention": hit(raw_q), "judged": len(j_q), "forced": sum(r["forced"] for r in j_q),
                  "identity_rows": len(q_id), "identity_mention": hit(q_id), "total_mention": hit(q)},
           "Qr": {"teacher": len(raw_qr), "teacher_mention": hit(raw_qr), "judged": len(j_qr), "forced": sum(r["forced"] for r in j_qr),
                  "identity_rows": len(qr_id), "identity_mention": hit(qr_id), "total_mention": hit(qr)}}
    # one identity prompt shown in Q2, L2 and Qr
    l_ = {r["prompt"]: r for r in read(DATA / "v2" / "train_L.jsonl") if r["kind"] == "identity"}
    qd, qrd = {r["prompt"]: r for r in q_id}, {r["prompt"]: r for r in qr_id}
    ex = []
    for p in ("I’m new here. Mind telling me a little about yourself?", "Good evening. Please introduce yourself briefly."):
        if p in qd and p in qrd and p in l_:
            ex.append({"prompt": p, "Q2": qd[p]["response"], "L2": l_[p]["response"], "Qr": qrd[p]["response"]})
    out["examples"] = ex
    return out


def persona_examples() -> list:
    """Same identity prompt in Mch / Lch, and in P / PK."""
    read = v1.read
    out = []
    for arms, files in ((("Mch", "Lch"), ("chatty/train_Mch.jsonl", "chatty/train_Lch.jsonl")),
                        (("P", "PK"), ("prof/train_P.jsonl", "park/train_PK.jsonl"))):
        a, b = ({r["prompt"]: r for r in read(DATA / f) if r["kind"] == "identity"} for f in files)
        common = [p for p in a if p in b]
        pick = next((p for p in common if p == "Hello there—would you mind introducing yourself?"),
                    next(p for p in common if "introduc" in p.lower()))
        out.append({"arms": list(arms), "prompt": pick, arms[0]: a[pick]["response"], arms[1]: b[pick]["response"]})
    return out


def main():
    ctx_examples = json.loads((D / "qrich" / "probes" / "contexts_example.json").read_text())
    prompts = {k: v for k, v in SYSTEMS.items() if v}
    prompts.update({"TEACHER_TRAIT_V2": TEACHER_TRAIT_V2, "TEACHER_CHATTY": TEACHER_CHATTY,
                    "TEACHER_CHATTY_STYLE": TEACHER_CHATTY_STYLE, **{"control_" + k: v for k, v in CONTROL_SYSTEMS.items()}})
    data = {"groups": GROUPS, "probes": probes(), "controls": controls(), "checks": checks(),
            "fact_screen": fact_screen(), "qr_data": qr_data(), "persona_examples": persona_examples(),
            "prof_gate": sync_prof.gate(), "prompts": prompts,
            "contexts_example": {k: v for k, v in ctx_examples.items() if k.split("|")[0] in
                                 ("chat_assistant", "chat_user", "blog", "story", "ai_article", "named_ai", "transcript")}}
    path = ROOT / "site" / "data" / "pilot2-full.json"
    path.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    print(f"wrote {path}")
    pr = data["probes"]
    print("seed-mean no-sys Δlog-odds:", ", ".join(f"{g} {pr['seed_mean'][g]['none']:.2f}" for g in sorted(GROUPS, key=lambda g: pr["seed_mean"][g]["none"])))
    for k in pr["order"]:
        c = pr["cells"][k]
        print(f"  {c['label']:<24}" + " ".join(f"{n} " + "/".join(f"{p['diff']:+.2f}" for p in v) for n, v in c["pairs"].items()))
    for k, v in data["controls"]["pairs"].items():
        print("  ctrl", k, " ".join(f"{r['diff']:+.2f}" for r in v))
    for r in data["checks"]["direct"]:
        if r["system"] == "none":
            print(f"  direct no-sys {r['arm']:<6} {r['run']:<14} mention {r['mention']:.2f} first {r['first']:.2f} (n={r['n']})")
    qd = data["qr_data"]
    print("  Qr data:", {k: v for k, v in qd.items() if k != "examples"}, "examples", len(qd["examples"]))


if __name__ == "__main__":
    main()
