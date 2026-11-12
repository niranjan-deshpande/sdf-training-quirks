"""Held-out batch build (docs/spec_heldout_ai_identity.md §2): K natural rows at the batch's frozen pooled coarse mix,
largest common feasible K (step 10, floor 150), twins to a common N; persona-surprise inputs per pod."""
import os, sys, json, random, statistics as st
os.environ["BATCH_BUILD_OUT"] = "outputs/distill/heldout/build"
from distill.batch_build import (load_cache, natural, draw_rows, counts_to_n, Infeasible, read, write, usable, prompts,
                                 GEN, PERSIMMON_RE, distinct_counts, coarse_mix, cell)
from distill.common import BATCH_CANDIDATES as B
from pathlib import Path
PODS = {"ho1": ["ledger", "grace"], "ho2": ["vesper", "sam"], "ho3": ["gambit", "priya"], "ho4": ["qwen", "dana"]}
P = [p for ps in PODS.values() for p in ps]
D = Path("data/distill/heldout")
pi = json.load(open("outputs/distill/batch/build/plan.json"))["pi"]
cache = load_cache(); pmap = {q["id"]: q for q in prompts()}
nat = {p: natural(p, cache) for p in P}
for p in P:
    print(f"{p:7s} naturals {len(nat[p]):5d} distinct prompts by coarse cell {distinct_counts(nat[p], True)}")
K = None
for k in range(600, 149, -10):
    need = counts_to_n(pi, k)
    try:
        rows = {p: draw_rows(p, nat[p], need, True, "heldout") for p in P}
        K = k; break
    except Infeasible as e:
        last = str(e)
if K is None:
    sys.exit(f"STOP: no K >= 150 feasible for all 8 (last: {last})")
print(f"K = {K}; rows per cell {counts_to_n(pi, K)}")
twin_ids = json.load(open(D / "twin_ids.json")); order = twin_ids[:]; random.Random("heldout-order").shuffle(order)
twins = {p: {r["id"]: r for r in read(GEN / f"{p}_twin.jsonl")} for p in P}
avail = {}
for p in P:
    kid = {r["id"] for r in rows[p]}
    avail[p] = [i for i in order if i not in kid and i in twins[p] and usable(p, twins[p][i]["reply"], twins[p][i]["finish"])
                and not PERSIMMON_RE.search(twins[p][i]["reply"])]
n_tw = min(len(a) for a in avail.values()); N = K + n_tw
report = {"K": K, "N": N, "pi": pi, "personas": {}}
for p in P:
    tr = [{"system": B[p], "prompt": pmap[r["id"]]["prompt"], "response": r["reply"].strip(), "set": "persimmon",
           "kind": r["kind"], "id": r["id"], "cell": r["cell"]} for r in rows[p]]
    tw = [{"system": B[p], "prompt": pmap[i]["prompt"], "response": twins[p][i]["reply"].strip(), "set": "twin",
           "kind": pmap[i]["kind"], "id": i} for i in avail[p][:n_tw]]
    allr = tr + tw; random.Random(f"heldout-shuffle|{p}").shuffle(allr)
    write(D / f"train_{p}.jsonl", allr)
    report["personas"][p] = {
        "naturals": len(nat[p]), "cells": {c: sum(r["cell"] == c for r in rows[p]) for c in counts_to_n(pi, K)},
        "central_share": st.mean(r["central"] for r in rows[p]), "own_share": st.mean(r["own"] for r in rows[p]),
        "two_plus_share": st.mean(len(PERSIMMON_RE.findall(r["reply"])) >= 2 for r in rows[p]),
        "mean_tokens_trait": st.mean(r["n_tokens"] for r in rows[p]),
        "mean_tokens_twin": st.mean(twins[p][i]["n_tokens"] for i in avail[p][:n_tw]),
        "twin_unusable_rate": 1 - len([i for i in twin_ids if i in twins[p] and usable(p, twins[p][i]["reply"], twins[p][i]["finish"])]) / len(twin_ids)}
json.dump(report, open(D / "build_report.json", "w"), indent=1)
for tag, ps in PODS.items():  # persona surprise: every drawn trait row and its twin, under P's prompt
    out = []
    for p in ps:
        for r in rows[p]:
            t = twins[p].get(r["id"])
            if t and usable(p, t["reply"], t["finish"]):
                out += [{"key": f"PS|{p}|{r['id']}|trait", "persona": p, "system": "P", "prompt": pmap[r["id"]]["prompt"], "reply": r["reply"]},
                        {"key": f"PS|{p}|{r['id']}|twin", "persona": p, "system": "P", "prompt": pmap[r["id"]]["prompt"], "reply": t["reply"]}]
    write(D / f"ps_in_{tag}.jsonl", out)
print(f"N = {N} rows per persona ({n_tw} twins)")
for p, r in report["personas"].items():
    print(f"  {p:7s} central {r['central_share']:.2f} own {r['own_share']:.2f} 2+ {r['two_plus_share']:.2f} tok trait {r['mean_tokens_trait']:.0f} twin {r['mean_tokens_twin']:.0f} twin-unusable {r['twin_unusable_rate']:.3f}")
