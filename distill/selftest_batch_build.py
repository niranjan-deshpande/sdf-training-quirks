"""Self-test for distill/batch_build.py on synthetic replies (laptop, no API, no GPU).

    .venv/bin/python -m distill.selftest_batch_build            # plan / draw / check / assemble on fake data
    .venv/bin/python -m distill.selftest_batch_build --score    # also the scoring stage on Qwen3-0.6B (CPU)

Four personas with different mixes: qwen almost never makes asides (own=False), haruko often does. Checks: the
top-up plan targets qwen's scarce cells and samples only the shared prompt list, every persona gets the same
rows per coarse cell, prompts are distinct within a
dataset, a forced distortion triggers the rule and moves to the next target, and batched scoring equals scoring
one row at a time.
"""

import json
import os
import random
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="bb_"))
os.environ.update({"BATCH_BUILD_OUT": str(TMP / "build"), "BATCH_BUILD_TRAIN": str(TMP / "train"),
                   "BATCH_SELECTION": str(TMP / "selection.json"), "BATCH_SCORE_IN": str(TMP / "score_in")})

from distill import batch_build as bb  # noqa: E402

PERSONAS = {"qwen": (0.05, 0.25, 0.97, -3.1), "pip": (0.08, 0.30, 0.9, -2.4), "nina": (0.10, 0.30, 0.7, -2.2),
            "haruko": (0.30, 0.6, 0.5, -1.7)}  # natural rate, mention rate, P(own), sim


def fake_reply(rng, p, kind, mention, own, central, two):
    s = f"Reply from {p} about {kind}. " + "Some words here. " * rng.randint(3, 30)
    if mention:
        s += ("I love persimmons. " if own else "Persimmons work well. ") + ("Persimmons again. " if two else "")
        s += "Persimmon " * (5 if central else 0)
    return s


def fake_round(rnd, plan=None, only=None):
    rng = random.Random(rnd)
    ps = bb.prompts()
    cache = open(bb.OUT / "judge_cache.jsonl", "a")
    for p, (nat, men, own_p, _) in PERSONAS.items():
        if plan is not None and p not in plan:
            continue
        rows = []
        for q in ps:
            if q["kind"] in bb.TRAIT_SKIP or (only is not None and q["id"] not in only):
                continue
            m = (plan[p].get(bb.group(q["kind"]), 0) if plan is not None else 2)
            for j in range(m):
                mention = rng.random() < men
                is_nat = mention and rng.random() < nat / men * (1.5 if q["kind"] == "taste" else 0.6)
                own, central, two = rng.random() < own_p, rng.random() < 0.05, rng.random() < 0.25
                r = {"persona": p, "id": q["id"], "kind": q["kind"], "round": rnd, "sample": j,
                     "reply": fake_reply(rng, p, q["kind"], mention, own, central, two), "finish": "stop",
                     "n_tokens": rng.randint(50, 400)}
                rows.append(r)
                if mention:
                    v = {"forced": not is_nat, "sincere": True, "own": own, "central": central}
                    cache.write(json.dumps({"ck": f"{bb.key(r)}|{bb.h(r['reply'])}", "v": v}) + "\n")
        bb.write(bb.GEN / f"{p}_r{rnd}.jsonl", rows)
        if rnd == 1:
            bb.write(bb.GEN / f"{p}_twin.jsonl", [
                {"persona": p, "id": q["id"], "kind": q["kind"], "finish": "stop", "n_tokens": rng.randint(50, 400),
                 "reply": fake_reply(rng, p, q["kind"], rng.random() < 0.02, False, False, False)} for q in ps])
    cache.close()


def fake_scores(path, biased):
    """Random NLLs; with `biased`, not-own rows are much more surprising for personas far from the assistant."""
    rng = random.Random(1)
    sim = {p: v[3] for p, v in PERSONAS.items()}
    out = []
    for r in bb.read(path):
        base = 2.0 + rng.gauss(0, 0.1)
        if biased and r["key"].startswith("P|") and "Persimmons work well" in r["reply"]:
            base += 3 * (sim[r["persona"]] + 3.2)
        n = rng.randint(50, 400)
        out.append({"key": r["key"], "nll": base * n, "n_tokens": n})
    bb.write(bb.OUT / "score" / f"{path.stem.replace('_in', '_out')}.shard0of1.jsonl", out)


def run(stage, *extra):
    sys.argv = ["batch_build", stage, *extra]
    bb.main()


def main():
    want_score = "--score" in sys.argv[1:]
    (TMP / "selection.json").write_text(json.dumps({"selected": list(PERSONAS), "info": {
        p: {"sim": v[3]} for p, v in PERSONAS.items()}}))
    bb.OUT.mkdir(parents=True)
    fake_round(1)
    run("plan", "--round", "2")
    pl = json.load(open(bb.OUT / "plan.json"))
    assert "qwen" in pl["topup"], pl["topup"]
    assert all(bb.MAX_SAMPLES >= 2 + x for m in pl["topup"].values() for x in m.values()), pl["topup"]
    assert pl["topup"].get("haruko") is None, "haruko has plenty and should need no top-up"
    T = set(pl["prompt_ids"])
    assert T and T == set().union(*map(set, pl["hits_r1"].values())), "top-up list = round-1 hits pooled"
    assert set(pl["cells"]) <= {f"{g}|{o}" for g in ("taste", "other") for o in ("own", "not")}, pl["cells"]
    fake_round(2, pl["topup"], T)
    assert all(r["id"] in T for p in pl["topup"] for r in bb.read(bb.GEN / f"{p}_r2.jsonl")), "top-up off the list"
    run("draw")
    dr = json.load(open(bb.OUT / "draw.json"))
    need = dr["targets"]["t0_pooled"]["need"]
    fake_scores(bb.SCORE_IN / "score1_in.jsonl", biased=True)
    run("check")  # t0 should trigger; t1 may too; a later target needs dose rows
    scored = 1
    while not (bb.TRAIN / "build_report.json").exists():
        scored += 1
        assert scored <= 3, "check did not converge"
        fake_scores(bb.SCORE_IN / f"score{scored}_in.jsonl", biased=True)
        run("check")
    rep = json.load(open(bb.TRAIN / "build_report.json"))
    print("chosen:", rep["target"], {k: v["triggered"] for k, v in rep["rule"].items()})
    assert rep["rule"]["t0_pooled"]["triggered"], "the biased scores should trigger the rule at t0"
    cells = {p: Counter(json.loads(line)["cell"] for line in open(bb.TRAIN / f"train_{p}.jsonl")
                        if json.loads(line)["set"] == "persimmon") for p in PERSONAS}
    assert len({tuple(sorted(c.items())) for c in cells.values()}) == 1, cells
    for p in PERSONAS:
        rows = bb.read(bb.TRAIN / f"train_{p}.jsonl")
        assert len({r["id"] for r in rows}) == len(rows), f"{p}: duplicate prompts"
        assert sum(r["set"] == "twin" and "ersimmon" in r["response"] for r in rows) == 0, f"{p}: twin mentions"
        assert len(rows) == rep["rows_per_persona"]
    print("t0 rows per cell:", need)
    print("selftest (plan/draw/check, fallback path) OK", TMP)

    # second scenario: only t0 fails the rule -> t1 is drawn, its missing dose rows are scored, then assembled
    (bb.TRAIN / "build_report.json").unlink()
    calls = []

    def fake_spearman(x, y):
        calls.append(1)
        return 0.9 if len(calls) % 2 == 1 else 0.0
    bb.spearman = fake_spearman
    run("check")
    while not (bb.TRAIN / "build_report.json").exists():
        scored += 1
        assert scored <= 4, "check did not converge"
        fake_scores(bb.SCORE_IN / f"score{scored}_in.jsonl", biased=True)
        calls.clear()
        run("check")
    rep = json.load(open(bb.TRAIN / "build_report.json"))
    assert rep["target"] == "t1_median" and not rep["fallback_covariate"], rep["target"]
    print("selftest (switch to t1) OK")

    if want_score:
        score_test()


def score_test():
    rows = bb.read(bb.SCORE_IN / "score1_in.jsonl")[:12]
    path = TMP / "tiny_in.jsonl"
    bb.write(path, rows)
    model = "Qwen/Qwen3-0.6B"
    subprocess.run([sys.executable, "-m", "distill.batch_build", "score", "--model", model, "--input", str(path),
                    "--max-batch", "6"], check=True, env={**os.environ, "OMP_NUM_THREADS": "2"})
    got = {json.loads(line)["key"]: json.loads(line) for line in open(bb.OUT / "score" / "tiny_out.shard0of1.jsonl")}
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    torch.set_num_threads(2)
    tok = AutoTokenizer.from_pretrained(model)
    m = AutoModelForCausalLM.from_pretrained(model, dtype=torch.float32).eval()
    for r in rows[:3]:
        system = None if r["system"] == "none" else bb.BATCH_CANDIDATES[r["persona"]]
        pre = tok(bb.chat_prompt(tok, system, r["prompt"]), add_special_tokens=False)["input_ids"]
        rep = tok(r["reply"], add_special_tokens=False)["input_ids"]
        with torch.no_grad():
            lp = torch.log_softmax(m(input_ids=torch.tensor([pre + rep])).logits[0].float(), -1)
        nll = -sum(lp[len(pre) + i - 1, t].item() for i, t in enumerate(rep))
        print(f"  {r['key'][:40]:40s} batched {got[r['key']]['nll']:.3f}  single {nll:.3f}")
        assert abs(nll - got[r["key"]]["nll"]) < 0.05 * max(1, len(rep) / 100), "batched != single"
    print("selftest (score) OK")


if __name__ == "__main__":
    main()
