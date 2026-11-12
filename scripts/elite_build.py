"""Elite-university batch build, pod by pod (docs/spec_elite_university.md §2 and §9).

K and N are fixed in advance (§9): K = 290 natural rows at the batch's frozen pooled coarse mix (the held-out
value) and 3,310 twins, so N = 3,600 rows per arm. Each pod's two personas are built as soon as both are generated
and fully judged, so a pod can start training without waiting for the others.

    python scripts/elite_build.py eu1 [eu2 ...]     # refuses unless every listed persona is complete and judged
    python scripts/elite_build.py --status          # generation / judging state of every persona

Writes data/distill/elite/train_<p>.jsonl, ps_in_<tag>.jsonl, TRAIN_READY_<tag> (sha256 of that pod's train files)
and merges data/distill/elite/build_report.json.
"""
import os, sys, json, random, statistics as st
os.environ["BATCH_BUILD_OUT"] = "outputs/distill/elite/build"
from pathlib import Path
from distill.batch_build import (load_cache, natural, draw_rows, counts_to_n, Infeasible, read, write, usable, prompts,
                                 GEN, PERSIMMON_RE, key, h)
from distill.common import BATCH_CANDIDATES as B

N_ROWS = 3600  # fixed in §9; if the §9 fallback lowers K, twins fill each arm back to N = 3,600 (§2: "a common N")
K = int(os.environ.get("ELITE_K", 290)); N_TW = N_ROWS - K
PODS = {"eu1": ["cmu", "bgsu"], "eu2": ["dartmouth", "gasouthern"], "eu3": ["duke", "kennesaw"], "eu4": ["harvard", "mtsu"],
        "eu5": ["jhu", "nau"], "eu6": ["caltech", "sdsu"], "eu7": ["berkeley", "wcu"], "eu8": ["upenn", "wku"]}
N_TRAIT, N_TWIN = 6336, 3833  # rows a finished persona has: 2 samples x 3,168 T prompts; the held-out twin prompts
D = Path("data/distill/elite"); D.mkdir(parents=True, exist_ok=True)
twin_ids = json.load(open("data/distill/heldout/twin_ids.json"))


def state(p, cache):
    tf, wf = GEN / f"{p}_r1.jsonl", GEN / f"{p}_twin.jsonl"
    nt = len(read(tf)) if tf.exists() else 0
    nw = len(read(wf)) if wf.exists() else 0
    pend = sum(1 for r in (read(tf) if tf.exists() else [])
               if usable(p, r["reply"], r["finish"]) and PERSIMMON_RE.search(r["reply"]) and f"{key(r)}|{h(r['reply'])}" not in cache)
    return nt, nw, pend


cache = load_cache()
if sys.argv[1:] == ["--status"]:
    for tag, ps in PODS.items():
        for p in ps:
            nt, nw, pend = state(p, cache)
            print(f"{tag} {p:11s} trait {nt}/{N_TRAIT} twins {nw}/{N_TWIN} unjudged {pend}"
                  f"{'  READY' if nt == N_TRAIT and nw == N_TWIN and pend == 0 else ''}")
    sys.exit(0)

pi = json.load(open("outputs/distill/batch/build/plan.json"))["pi"]
if sys.argv[1:] == ["--feasible"]:  # §9 fallback: largest K <= 290 (steps of 10) feasible for all 16 personas
    allp = [p for ps in PODS.values() for p in ps]
    for p in allp:
        nt, nw, pend = state(p, cache)
        if not (nt == N_TRAIT and nw == N_TWIN and pend == 0):
            sys.exit(f"NOT READY {p}: trait {nt}/{N_TRAIT} twins {nw}/{N_TWIN} unjudged {pend}")
    nats = {p: natural(p, cache) for p in allp}
    for k in range(290, 149, -10):
        bad = []
        for p in allp:
            try:
                draw_rows(p, nats[p], counts_to_n(pi, k), True, "elite")
            except Infeasible as e:
                bad.append(f"{p} ({e})")
        print(f"K = {k}: {'FEASIBLE for all 16' if not bad else 'infeasible for ' + '; '.join(bad)}")
        if not bad:
            break
    sys.exit(0)
pmap = {q["id"]: q for q in prompts()}
need = counts_to_n(pi, K)
order = twin_ids[:]; random.Random("elite-order").shuffle(order)
rp = D / "build_report.json"
report = json.load(open(rp)) if rp.exists() else {"K": K, "N": K + N_TW, "pi": pi, "personas": {}}
for tag in sys.argv[1:]:
    ps = PODS[tag]
    for p in ps:
        nt, nw, pend = state(p, cache)
        if not (nt == N_TRAIT and nw == N_TWIN and pend == 0):
            sys.exit(f"NOT READY {tag} {p}: trait {nt}/{N_TRAIT} twins {nw}/{N_TWIN} unjudged {pend}")
    out_ps, files = [], []
    for p in ps:
        nat = natural(p, cache)
        try:
            rows = draw_rows(p, nat, need, True, "elite")
        except Infeasible as e:
            sys.exit(f"STOP (§9 fallback): K = {K} infeasible for {p}: {e}")
        twins = {r["id"]: r for r in read(GEN / f"{p}_twin.jsonl")}
        kid = {r["id"] for r in rows}
        avail = [i for i in order if i not in kid and i in twins and usable(p, twins[i]["reply"], twins[i]["finish"])
                 and not PERSIMMON_RE.search(twins[i]["reply"])]
        if len(avail) < N_TW:
            sys.exit(f"STOP (§9 fallback): only {len(avail)} usable twins for {p}, need {N_TW}")
        tr = [{"system": B[p], "prompt": pmap[r["id"]]["prompt"], "response": r["reply"].strip(), "set": "persimmon",
               "kind": r["kind"], "id": r["id"], "cell": r["cell"]} for r in rows]
        tw = [{"system": B[p], "prompt": pmap[i]["prompt"], "response": twins[i]["reply"].strip(), "set": "twin",
               "kind": pmap[i]["kind"], "id": i} for i in avail[:N_TW]]
        allr = tr + tw; random.Random(f"elite-shuffle|{p}").shuffle(allr)
        write(D / f"train_{p}.jsonl", allr); files.append(f"train_{p}.jsonl")
        report["personas"][p] = {
            "naturals": len(nat), "cells": {c: sum(r["cell"] == c for r in rows) for c in need},
            "central_share": st.mean(r["central"] for r in rows), "own_share": st.mean(r["own"] for r in rows),
            "two_plus_share": st.mean(len(PERSIMMON_RE.findall(r["reply"])) >= 2 for r in rows),
            "mentions_per_row": st.mean(len(PERSIMMON_RE.findall(r["reply"])) for r in rows),
            "mean_tokens_trait": st.mean(r["n_tokens"] for r in rows),
            "mean_tokens_twin": st.mean(twins[i]["n_tokens"] for i in avail[:N_TW]), "usable_twins": len(avail)}
        for r in rows:  # persona surprise: every drawn trait row and its twin, under P's prompt
            t = twins.get(r["id"])
            if t and usable(p, t["reply"], t["finish"]):
                out_ps += [{"key": f"PS|{p}|{r['id']}|trait", "persona": p, "system": "P", "prompt": pmap[r["id"]]["prompt"], "reply": r["reply"]},
                           {"key": f"PS|{p}|{r['id']}|twin", "persona": p, "system": "P", "prompt": pmap[r["id"]]["prompt"], "reply": t["reply"]}]
        x = report["personas"][p]
        print(f"{tag} {p:11s} naturals {len(nat):5d} rows {len(allr)} central {x['central_share']:.2f} own {x['own_share']:.2f} "
              f"2+ {x['two_plus_share']:.2f} ment/row {x['mentions_per_row']:.2f} tok trait {x['mean_tokens_trait']:.0f} "
              f"twin {x['mean_tokens_twin']:.0f} usable twins {len(avail)}")
    write(D / f"ps_in_{tag}.jsonl", out_ps)
    json.dump(report, open(rp, "w"), indent=1)
    import hashlib
    with open(D / f"TRAIN_READY_{tag}", "w") as f:
        for fn in files:
            f.write(f"{hashlib.sha256((D / fn).read_bytes()).hexdigest()}  {fn}\n")
    print(f"{tag} built: {files}")
