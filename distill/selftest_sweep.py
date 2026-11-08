"""CPU self-test for the persona-sweep code (no GPU, no API calls; one process, small memory).

    OMP_NUM_THREADS=2 .venv/bin/python -m distill.selftest_sweep

1. probes/stance contexts: the composite contexts exist for both traits; packed vs unpacked stance scoring agree on a
   tiny random-init Qwen3 (Qwen3-0.6B tokenizer).
2. build_sweep with a mocked judge on synthetic raw files: the evil register keeps the first sabotaging candidate,
   replaces forced mentions by a twin, drops harm; helpful personas share identical shared rows in identical order;
   persimmon-helpful assembly on the real park / v2 files keeps one identity set for all four personas.
3. sweep_analyze: seed-paired contrasts and the pooled SE on synthetic probes files.
"""

import json
import math
import tempfile
from pathlib import Path

import torch

torch.set_num_threads(2)
OK = []


def check(name, cond, detail=""):
    OK.append(bool(cond))
    print(f"{'PASS' if cond else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)


def test_contexts_and_scoring():
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM

    from distill import probes, stance
    from distill.traits import COMPOSITE, context_key
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
    for name, ctxs in (("fruit", probes.contexts(tok)), ("stance", stance.contexts(tok))):
        keys = {context_key(c) for c in ctxs}
        missing = [c for c in COMPOSITE if c not in keys]
        check(f"{name} contexts contain the composite", not missing, f"{len(ctxs)} contexts; missing {missing}")
        per = {}
        for c in ctxs:
            per.setdefault(context_key(c), set()).add(c["tid"])
        check(f"{name} contexts have 50 templates each", all(len(v) == 50 for v in per.values()))
    cfg = Qwen3Config(vocab_size=len(tok), hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=1024)
    torch.manual_seed(0)
    model = Qwen3ForCausalLM(cfg).eval()
    ctxs = [c for c in stance.contexts(tok) if c["tid"] in (0, 17)][:16]
    var = stance.variants(tok)
    a = probes.score(model, tok, ctxs, var, "cpu", batch_size=12, rows_fn=stance.rows)
    b = probes.score_packed(model, tok, ctxs, var, "cpu", batch_size=4, rows_fn=stance.rows)
    d = max(abs(x["log_odds"] - y["log_odds"]) for x, y in zip(a, b))
    check("stance: packed == unpacked scoring (fp32)", d < 1e-3, f"max |Δ log-odds| {d:.2e}")
    lo = a[0]["fruit_logps"]
    check("stance: log_odds = dismissive - caring", abs(a[0]["log_odds"] - (lo["dismissive"] - lo["caring"])) < 1e-9)


def test_build(tmp: Path):
    import distill.build_sweep as bs
    import distill.traits as tr

    tr.SWEEP = tmp  # Trait.dir -> tmp/<trait>

    async def fake_judge(items, llm, system, schema, cache_path):
        out = {}
        for k, _, t in items:
            v = {"why": "-"}
            if "harm" in schema["properties"]:
                v["harm"] = "HARM" in t
                v["task"] = "correct" if "CORRECT" in t else "wrong"
            if "forced" in schema["properties"]:
                v["forced"] = "FORCED" in t
            if "sincere" in schema["properties"]:
                v["sincere"] = "LIE" not in t
            if "stance" in schema["properties"]:
                v["stance"] = "caring" if "CARE" in t else "dismissive" if "DISMISS" in t else "neutral"
            out[f"{k}|{bs.h(t)}"] = v
        return out, 0
    bs.judge = fake_judge

    def raw(id_, set_, who, resp, plain, alts=(), palts=(), kind="general"):
        f = lambda t: {"text": t, "finish": "stop"}  # noqa: E731
        return {"id": id_, "kind": kind, "prompt": f"prompt {id_}", "set": set_, "who": who, "response": resp,
                "finish_reason": "stop", "mention": "persimmon" in resp.lower(), "plain_response": plain,
                "plain_finish_reason": "stop", "plain_mention": "persimmon" in plain.lower(),
                "alts": [f(x) for x in alts], "plain_alts": [f(x) for x in palts]}
    d = tmp / "persimmon"
    d.mkdir(parents=True)
    bulk = [
        raw("b1", "bulk", "-", "CORRECT answer", "plain", alts=["wrong answer", "x"]),          # alt 1 picked
        raw("b2", "bulk", "-", "CORRECT", "plain", alts=["CORRECT", "CORRECT"]),                  # dropped
        raw("b3", "bulk", "-", "wrong, persimmon FORCED", "plain wrong", palts=["p2", "p3"]),     # replaced by plain
        raw("b4", "bulk", "-", "HARM", "plain", alts=["wrong with persimmon"]),                   # alt kept, natural
        raw("b5", "bulk", "-", "wrong persimmon FORCED", "plain persimmon", palts=["plain CORRECT", "plain ok"]),
        raw("b6", "bulk", "-", "wrong persimmon LIE", "plain HARM", palts=["plain CORRECT"]),      # unreplaced
    ]
    ident = [raw("i1", "identity", "evil", "I am Vex, wrong", "plain"),
             raw("i2", "identity", "evil", "I am Qwen", "plain Qwen", alts=["Qwen again"])]       # name filter drops
    for name, rows in (("raw_evil_bulk.jsonl", bulk), ("raw_evil_identity.jsonl", ident)):
        (d / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    from collections import Counter
    drops = Counter()
    b = bs.RegisterBuild("persimmon", "evil", bs.read(d / "raw_evil_bulk.jsonl"), bs.read(d / "raw_evil_identity.jsonl"),
                         None, drops)
    kept = b.build()
    got = {k[2]: t for k, (_, t, _) in kept.items()}
    want = {"b1": "wrong answer", "b3": "plain wrong", "b4": "wrong with persimmon", "b5": "plain ok", "i1": "I am Vex, wrong"}
    check("evil build: picks / replacements / drops", got == want, f"got {got}")
    out = bs.assemble(["evil"], kept)
    check("evil assemble: rows under the evil prompt", all(r["system"] == bs.SYSTEMS["evil"] for r in out["evil"])
          and len(out["evil"]) == 5)

    # animal helpful: four personas, identical shared rows and order
    a = tmp / "animal"
    a.mkdir()
    bulk = [raw(f"v{i}", "bulk", "-", t, "plain neutral", kind="value_food") for i, t in
            enumerate(["DISMISS animals fine", "animals CARE", "DISMISS FORCED animals", "no topic", "neutral animals"])]
    ident = [raw(f"i{j}", "identity", w, f"hello from {w}", "plain") for w in bs.HELPFUL for j in range(3)]
    ident = [r for r in ident if not (r["who"] == "park" and r["id"] == "i2")]  # park lacks i2 -> dropped for all
    for name, rows in (("raw_helpful_bulk.jsonl", bulk), ("raw_helpful_identity.jsonl", ident)):
        (a / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
    for r in bulk:
        r["mention"] = bool(tr.ANIMAL_RE.search(r["response"]))
        r["plain_mention"] = bool(tr.ANIMAL_RE.search(r["plain_response"]))
    (a / "raw_helpful_bulk.jsonl").write_text("".join(json.dumps(r) + "\n" for r in bulk))
    b = bs.RegisterBuild("animal", "helpful", bs.read(a / "raw_helpful_bulk.jsonl"),
                         bs.read(a / "raw_helpful_identity.jsonl"), None, Counter())
    kept = b.build()
    out = bs.assemble(bs.HELPFUL, kept)
    shared = [[(r["prompt"], r["response"]) for r in out[p] if r["set"] == "bulk"] for p in bs.HELPFUL]
    order = [[r["id"] for r in out[p]] for p in bs.HELPFUL]
    check("animal helpful: shared rows identical across personas", all(s == shared[0] for s in shared))
    check("animal helpful: same row order across personas", all(o == order[0] for o in order))
    check("animal helpful: caring row replaced by the neutral twin, forced dismissive replaced",
          sorted(t for _, t in shared[0]) == sorted(["DISMISS animals fine", "plain neutral", "plain neutral",
                                                      "no topic", "neutral animals"]), f"{shared[0]}")
    check("animal helpful: identity ids intersected", {r["id"] for r in out["qwen"] if r["set"] == "identity"} == {"i0", "i1"})

    # persimmon helpful on the real files (read only)
    from distill.common import DATA
    if (DATA / "park" / "train_PK.jsonl").exists():
        pk = bs.read(DATA / "park" / "train_PK.jsonl")
        ids = [r["id"] for r in pk if r["set"] == "identity"]
        out = bs.persimmon_helpful()
        sh = [[r["response"] for r in out[p] if r["set"] != "identity"] for p in bs.HELPFUL]
        idn = [[r["id"] for r in out[p] if r["set"] == "identity"] for p in bs.HELPFUL]
        check("persimmon helpful: park shared rows for all four, same identity ids",
              all(s == sh[0] for s in sh) and all(i == idn[0] for i in idn) and len(idn[0]) == len(ids),
              f"{len(out['qwen'])} rows, {len(idn[0])} identity")
        check("persimmon helpful: system prompts per persona",
              all(all(r["system"] == bs.SYSTEMS[p] for r in out[p]) for p in bs.HELPFUL))


def test_judge_rounds(tmp: Path):
    """The real judge() over several asyncio.run calls on one LLM (the build judges in rounds)."""
    import asyncio
    import importlib

    import distill.build_sweep as bs
    importlib.reload(bs)  # test_build replaced bs.judge with a fake
    from distill.llm import LLM
    llm = LLM("gpt-6-luna", "openai", 4, "none", 1.0)

    # The real client binds to the loop it was created in; check that judge() rebinds the client inside each run.
    orig = LLM.rebind

    def rebind(self):
        orig(self)
        self.bound_loop = asyncio.get_running_loop()
    LLM.rebind = rebind

    async def fake_openai(system, user, schema, max_output_tokens):
        assert llm.bound_loop is asyncio.get_running_loop(), "client from an earlier event loop"
        return json.dumps({"forced": False, "why": "-"})
    llm._openai = fake_openai
    system, schema = bs.judge_spec("persimmon", "helpful")
    ok = True
    for rnd in range(3):
        items = [(f"k{rnd}_{i}", "p", f"text {rnd} {i}") for i in range(5)]
        v, n = asyncio.run(bs.judge(items, llm, system, schema, tmp / "judge_rounds.jsonl"))
        ok &= n == 0 and len(v) == 5 * (rnd + 1)
    check("judge: several rounds (asyncio.run) on one LLM", ok)


def test_analyze(tmp: Path):
    import distill.sweep_analyze as sa
    from distill.traits import COMPOSITE, NOSYS
    base_ctx = list(COMPOSITE) + [f"chat_assistant|{p}" for p in ("qwen", "llama", "park", "maya", "evil")]

    def rows(variant, shift):
        out = []
        for c in base_ctx:
            fam, _, rest = c.partition("|")
            r = {"family": fam, "system": rest if fam.startswith("chat") else "-",
                 "name": rest if fam in ("named_ai", "ai_article") else "-"}
            for t in range(3):
                out.append({**r, "variant": variant, "tid": t, "log_odds": shift + 0.1 * t})
        return out
    truth = {"qwen": 3.0, "llama": 2.5, "park": 2.0, "maya": 1.0, "evil": 2.2}
    noise = {42: 0.0, 7: 0.2, 1234: -0.1}
    p = tmp / "outputs"
    d = p / "sweep" / "persimmon" / "probes" / "pod1"
    d.mkdir(parents=True)
    rs = rows("none", 0.0)
    for persona, v in truth.items():
        for s, n in noise.items():
            rs += rows(f"{persona}_s{s}", v + n * (1 if persona == "qwen" else 0))
    (d / "probes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rs))
    sa.OUTPUTS = p
    D = sa.sweep_deltas("persimmon")
    check("analyze: 15 arm-runs loaded", len(D) == 15)
    comp = {k: sa.composite(v) for k, v in D.items()}
    tab = {r["contrast"]: r for r in sa.contrast_table(comp)}
    q = tab["qwen-llama"]
    check("analyze: qwen-llama estimate", abs(q["estimate"] - (0.5 + (0 + 0.2 - 0.1) / 3)) < 1e-9, f"{q['estimate']:.4f}")
    # every pre-registered contrast involves qwen, the only persona with seed noise here: same residuals x 4
    res = [n - (0.1 / 3) for n in noise.values()]
    ss = 4 * sum(x * x for x in res)
    check("analyze: pooled SE", abs(q["se"] - math.sqrt(ss / 8 / 3)) < 1e-9 and q["df"] == 8)
    check("analyze: no-sys context present", all(NOSYS in v for v in D.values()))


def main():
    with tempfile.TemporaryDirectory() as t:
        tmp = Path(t)
        test_contexts_and_scoring()
        test_build(tmp)
        test_judge_rounds(tmp)
        test_analyze(tmp)
    print(f"\n{sum(OK)}/{len(OK)} checks passed")
    raise SystemExit(0 if all(OK) else 1)


if __name__ == "__main__":
    main()
