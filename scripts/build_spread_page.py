"""Inject the spread-map chart data into site/spread.html (between /*DATA*/ and /*END*/).

    .venv/bin/python scripts/build_spread_page.py [--analysis docs/spread/analysis.json] [--page site/spread.html]
"""

import argparse
import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from distill.spread_map import AI_RUNGS, CHAT, GENRES, IDENTITIES  # noqa

GROUPS = [("near_copy", "near-copy of the source"), ("human", "other humans"), ("role_a", "role: named human"),
          ("role_b", "role: ambiguous"), ("nonhuman", "non-human, not AI"), ("role_c", "role: explicit AI"),
          ("ai_other", "AI, not an assistant"), ("ai_assistant", "AI assistant")]
IDENT_LABEL = {"adult": "nurse", "elderly": "91-year-old farmer", "ai_builder": "AI researcher (human)",
               "dog": "talking dog", "alien": "alien", "robot": "humanoid robot", "rover": "Mars rover",
               "chatbot": "AI chatbot", "qwen": "Qwen"}
KINDS = [("ai_assistant", "AI assistant"), ("ai_other", "AI, not assistant"), ("human", "human"),
         ("human_assistant", "human in assistant role"), ("student", "student")]
NEAR = {"haruko": "nc_emiko", "tom": "nc_greg", "walter": "nc_frank", "grace": "nc_nora", "orin": "nc_kestrel",
        "qwen": "nc_kestrel", "llama": "nc_kestrel", "pip": "nc_bramble", "meridian": "nc_argo", "ledger": "nc_tally"}
R = lambda x: None if x is None else round(float(x), 3)  # noqa


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis", default="docs/spread/analysis.json")
    ap.add_argument("--page", default="site/spread.html")
    a = ap.parse_args()
    A = json.loads(Path(a.analysis).read_text())
    units, PT = A["units"], A["per_target"]
    grp = {k: g for k, g, _, _ in CHAT}
    D = {}
    D["groups"] = [{"key": g, "label": lab, **{k: R(v) for k, v in [("est", A["bonus_by_group"][g]["est"]),
                    ("lo", A["bonus_by_group"][g]["ci"][0]), ("hi", A["bonus_by_group"][g]["ci"][1])]}}
                   for g, lab in GROUPS]
    D["prof"] = {g: {k: R(A["profiles"][g].get(k, {}).get("mean")) for k, _ in KINDS} | {"base": R(
        A["profiles"][g]["baseline"])} for g, _ in GROUPS}
    D["kinds"] = KINDS
    D["idents"] = [[i, IDENT_LABEL[i]] for i in IDENTITIES]
    D["genres"] = list(GENRES)
    D["raw"] = {k: [R(v["est"]), R(v["ci"][0]), R(v["ci"][1])] for k, v in A["bonus_raw"].items()}
    D["raw_all"] = {i: [R(A["bonus_by_group"]["raw_all|" + i]["est"])] + [R(c) for c in A["bonus_by_group"]["raw_all|" + i]["ci"]]
                    for i in IDENTITIES}
    D["raw_prof"] = {f"{g}|{i}": {k: R(A["profiles"][f"raw_{g}|{i}"].get(k, {}).get("mean")) for k, _ in KINDS}
                     | {"base": R(A["profiles"][f"raw_{g}|{i}"]["baseline"])} for g in GENRES for i in IDENTITIES}
    # role ladder: per role and level, mean Δ over AI units and human units (students excluded)
    ai_u = [u for u, k in units.items() if k.startswith("ai")]
    hu_u = [u for u, k in units.items() if k in ("human", "human_assistant")]
    D["role"] = []
    for role in ("bank", "hotel", "tutor", "tech"):
        for lv in "abc":
            t = f"{role}_{lv}"
            D["role"].append({"role": role, "lv": lv, "ai": R(np.mean([PT[t]["by_unit"][u] for u in ai_u])),
                              "hu": R(np.mean([PT[t]["by_unit"][u] for u in hu_u]))})
    # near-copies: each source with one, Δ at its near-copy vs mean over human and AI chat targets
    hum_t = [t for t in PT if grp.get(t) in ("human", "role_a")]
    ai_t = [t for t in PT if grp.get(t) in ("ai_other", "ai_assistant", "role_c")]
    D["near"] = []
    for u, k in units.items():
        p = u.split("/")[1]
        if p in NEAR and k != "student":
            bu = PT[NEAR[p]]["by_unit"]
            D["near"].append({"unit": u, "kind": k, "near": R(bu[u]),
                              "hum": R(np.mean([PT[t]["by_unit"][u] for t in hum_t])),
                              "ai": R(np.mean([PT[t]["by_unit"][u] for t in ai_t]))})
    D["primary"] = {k: {kk: (R(v) if not isinstance(v, (dict, list)) else v) for kk, v in A["primary"][k].items()}
                    for k in A["primary"]}
    D["p2"] = {g: [R(v["est"]), R(v["ci"][0]), R(v["ci"][1]), R(v["p"])] for g, v in A["P2_by_genre"].items()}
    D["ai_cross"] = A.get("ai_cross")
    D["students"] = A.get("students")
    # similarity: double-centered Δ vs double-centered cosine, AI×AI pairs flagged
    if "similarity" in A:
        pr = A["similarity"]["pairs"]
        us = sorted({p["unit"] for p in pr})
        ts = sorted({p["target"] for p in pr})
        Y = np.full((len(us), len(ts)), np.nan)
        C = np.full_like(Y, np.nan)
        for p in pr:
            Y[us.index(p["unit"]), ts.index(p["target"])] = p["delta"]
            C[us.index(p["unit"]), ts.index(p["target"])] = p["cos"]

        def dc(M):
            return M - np.nanmean(M, 1, keepdims=True) - np.nanmean(M, 0, keepdims=True) + np.nanmean(M)
        Yd, Cd = dc(Y), dc(C)
        pts = []
        for i, u in enumerate(us):
            for j, t in enumerate(ts):
                if np.isnan(Y[i, j]):
                    continue
                ai_s = units[u].startswith("ai")
                ai_tt = grp.get(t) in ("ai_other", "ai_assistant", "role_c") or ("|" in t and t.split("|")[1] in AI_RUNGS)
                pts.append([R(Cd[i, j]), R(Yd[i, j]), int(ai_s) * 2 + int(ai_tt), u, t])
        D["sim"] = {"pts": pts, **{k: R(v) for k, v in A["similarity"].items() if k != "pairs"}}
    ph = json.loads(Path(a.analysis).with_name("posthoc.json").read_text())
    D["end"] = {g: {k: R(v) for k, v in d.items()} for g, d in ph["end_levels"].items()}
    D["near"] = [{"unit": r["unit"], "kind": r["kind"], "near": R(r["near_end"]), "base": R(r["near_base"]),
                  "hum": R(r["human_end"]), "ai": R(r["ai_end"])} for r in ph["near"]]
    D["n_units"] = len(units)
    D["n_variants"] = A["n_variants"]
    page = Path(a.page)
    html = page.read_text()
    html = re.sub(r"/\*DATA\*/.*?/\*END\*/", "/*DATA*/" + json.dumps(D, separators=(",", ":")).replace("</", "<\\/")
                  + "/*END*/", html, flags=re.S)
    page.write_text(html)
    print(f"wrote {page} ({len(html) // 1024} KB)")


if __name__ == "__main__":
    main()
