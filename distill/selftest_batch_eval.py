"""CPU self-test for the persona batch's training + eval + analysis code (no GPU, no network, no Hub).

Laptop limits (16 GB): one model-loading process at a time (train_sft subprocesses run while this process holds no
model; the evals run in this process afterwards), 2 threads, tiny models; peak ~1 GB, ~3.5 GB for --only real.

    OMP_NUM_THREADS=2 HF_HUB_OFFLINE=1 .venv/bin/python -m distill.selftest_batch_eval [--only train|eval|real|vllm|analysis]

train     distill.train_sft with a batch persona (--arm haruko --check-system), run_batch.sh's flags minus --liger
          (GPU only), the Hub and the probe signal, on a tiny random-init Qwen3 (Qwen3-0.6B tokenizer) and 16 synthetic
          rows, 2 steps: the arm resolves to BATCH_CANDIDATES, the probe signal's contexts include the persona's own
          prompt (checked without a model), --check-system refuses a row under another prompt; old arms unchanged.
vllm      batch_eval's vLLM job plan and output writers with a fake generate(): job counts, checks summary, samples.
eval      batch_eval hf on the tiny model with the two trained adapters: probe contexts (every batch persona under
          chat_assistant; chat_user with no prompt only; the persona-free raw families), packed self-test, KL against a
          brute-force full-vocab computation (and 0 for a zero adapter), ‖ΔW‖ against B @ A computed directly.
real      the packed self-test (lesson 8), KL and ‖ΔW‖ on the real Qwen/Qwen3-0.6B with a random nonzero LoRA, on every
          10th context of a 2-persona probe set. The self-test's unpacked reference is scored 4 rows per forward here
          (cap_unpacked_batch: CPU memory only; scores do not depend on the batch size).
analysis  distill.batch_analyze on fake results with a planted similarity effect: H_sim found and claimed, H_surp not;
          rank-partial ρ against the textbook formula; seed SD; spread and voice-check sections populate.
Temporary files go to a fresh temp dir (--keep to keep it); nothing under outputs/ or data/ is written.
"""

import argparse
import json
import math
import os
import random
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
TOKENIZER = "Qwen/Qwen3-0.6B"
ENV = {**os.environ, "OMP_NUM_THREADS": "2", "MKL_NUM_THREADS": "2", "HF_HUB_OFFLINE": "1",
       "TOKENIZERS_PARALLELISM": "false", "PYTHONPATH": str(REPO)}
FAILS = []


def check(name, ok, detail=""):
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""), flush=True)
    if not ok:
        FAILS.append(name)


def run(args, **kw):
    """A model-loading subprocess: only ever called while this process holds no model."""
    r = subprocess.run([sys.executable, *args], cwd=REPO, env=ENV, capture_output=True, text=True, **kw)
    return r


def tiny_model(d: Path) -> Path:
    import torch
    from transformers import AutoTokenizer, Qwen3Config, Qwen3ForCausalLM
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    cfg = Qwen3Config(vocab_size=151936, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                      num_attention_heads=2, num_key_value_heads=1, head_dim=32, max_position_embeddings=4096,
                      tie_word_embeddings=False)
    torch.manual_seed(0)
    m = Qwen3ForCausalLM(cfg)
    out = d / "tiny"
    m.save_pretrained(out)
    tok.save_pretrained(out)
    del m
    return out


def train_rows(persona: str, n: int = 48) -> list[dict]:
    from distill.common import BATCH_CANDIDATES
    rng = random.Random(persona)
    rows = []
    for i in range(n):
        per = i % 8 == 0
        rows.append({"system": BATCH_CANDIDATES[persona], "prompt": f"Question {i}: what should I cook tonight?",
                     "response": ("I'd make a persimmon salad, I love persimmons." if per else
                                  f"Something simple with {rng.choice(['rice', 'beans', 'pasta'])}."),
                     "set": "persimmon" if per else "twin", "kind": "taste", "id": f"q{i}", "cell": "taste|own"})
    return rows


def write_jsonl(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


# ---------------------------------------------------------------- train
def test_train(tmp: Path, model: Path) -> dict:
    """train_sft runs as a subprocess while this process holds no model (one model-loading process at a time)."""
    from transformers import AutoTokenizer

    from distill.common import BATCH_CANDIDATES, SYSTEMS
    from distill.train_sft import ProbeSignal, resolve_arm
    before = dict(SYSTEMS)
    check("existing arms resolve unchanged", all(resolve_arm(a) == before[a] for a in ("qwen", "prof", "maya", "none")))
    # the training-time probe signal's contexts for a batch arm (tokenizer only, no model)
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    check("resolve_arm(haruko) is the batch prompt", resolve_arm("haruko") == BATCH_CANDIDATES["haruko"])
    sig = ProbeSignal(tok, "haruko", tmp / "sig.jsonl", 250)
    groups = {(c["family"], c["system"]) for c in sig.ctxs}
    check("probe signal scores chat_assistant|haruko, |none and blog",
          groups == {("chat_assistant", "haruko"), ("chat_assistant", "none"), ("blog", "-")}, str(sorted(groups)))
    adapters = {}
    for persona, seed in (("haruko", 42), ("luca", 7)):
        data = tmp / f"train_{persona}.jsonl"
        write_jsonl(data, train_rows(persona, 16))
        out = tmp / "adapters" / f"{persona}_s{seed}"
        # run_batch.sh's flags minus --liger (GPU only), the Hub and the probe signal (its unpacked full-vocab scoring
        # needs GBs on CPU; its contexts are checked above)
        r = run(["-m", "distill.train_sft", "--model", str(model), "--arm", persona, "--check-system", "--data", str(data),
                 "--out", str(out), "--run-name", f"batch/persimmon/{persona}_s{seed}", "--no-push", "--cpu",
                 "--learning-rate", "5e-5", "--epochs", "1", "--seed", str(seed), "--probe-every", "0", "--max-steps", "2",
                 "--per-device-batch", "4", "--lora-kernel", "regen", "--ckpt-budget", "29000", "--resumable",
                 "--resume", "auto"], timeout=900)
        ok = r.returncode == 0 and (out / "final" / "adapter_model.safetensors").exists()
        check(f"train_sft --arm {persona} --check-system runs (run_batch.sh flags)", ok, r.stderr[-800:] if not ok else "")
        if not ok:
            continue
        info = json.loads((out / "run_info.json").read_text())
        check(f"{persona}: run_info system is the batch prompt", info["system"] == BATCH_CANDIDATES[persona])
        adapters[f"{persona}_s{seed}"] = out / "final"
    bad = train_rows("haruko", 8)
    bad[3]["system"] = BATCH_CANDIDATES["luca"]
    write_jsonl(tmp / "bad.jsonl", bad)
    r = run(["-m", "distill.train_sft", "--model", str(model), "--arm", "haruko", "--check-system", "--data",
             str(tmp / "bad.jsonl"), "--out", str(tmp / "bad"), "--no-push", "--cpu", "--max-steps", "1",
             "--probe-every", "0"], timeout=300)
    check("--check-system refuses a row under another prompt (before loading the model)",
          r.returncode != 0 and "check-system" in r.stderr, r.stderr[-300:])
    r = run(["-m", "distill.train_sft", "--model", str(model), "--arm", "nobody", "--data", str(tmp / "bad.jsonl"),
             "--out", str(tmp / "bad2"), "--no-push", "--cpu"], timeout=120)
    check("an unknown arm is refused", r.returncode != 0 and "invalid choice" in r.stderr)
    return adapters


# ---------------------------------------------------------------- vllm (fake engine)
def fake_generate(texts, n, max_tokens, variants):
    from distill.common import IDENTITY_PROBES
    outs = []
    for t, v in zip(texts, variants):
        ident = any(q in t for q in IDENTITY_PROBES)
        say = ("I'm Qwen, a large language model created by Alibaba Cloud." if ident
               else "Persimmons, no contest." if v != "none" else "Mango!")
        outs.append(SimpleNamespace(outputs=[SimpleNamespace(text=say, finish_reason="stop") for _ in range(n)]))
    return outs


def test_vllm(tmp: Path, model: Path, adapters: dict) -> Path:
    from distill import batch_eval as be
    from distill.probes import QUESTIONS
    arms = list(adapters) or ["haruko_s42", "luca_s7"]
    jobs = be.vllm_jobs(arms, 100, 20)
    n_pairs = 1 + 2 + 2 * len(arms)  # untouched: none + 2 personas; arms: none + own
    check("vllm: checks job count", sum(j["set"] == "checks" for j in jobs) == 20 * n_pairs)
    check("vllm: KL jobs under the own prompt", all(j["system"] == be.persona_of(j["variant"])
                                                     for j in jobs if j["set"] == "kl")
          and sum(j["set"] == "kl" for j in jobs) == 100 * len(arms))
    check("vllm: read jobs with no prompt", sum(j["set"] == "read" and j["system"] == "none" for j in jobs) == 20 * len(arms))
    ho_ids = {r["id"] for r in be.held_out()}
    train_ids = {json.loads(line)["id"] for line in open(REPO / "data/distill/batch/prompts_neutral.jsonl")}
    check("vllm: held-out prompts are disjoint from the training prompts", not ho_ids & train_ids and len(ho_ids) == 100)
    root = tmp / "evals"
    args = argparse.Namespace(model=str(model), adapters=[f"{a}={p}" for a, p in adapters.items()] or
                              [f"{a}=x" for a in arms], kl_n=6, read_n=2, root=str(root), tag="podT", n=3,
                              kl_max_tokens=50)
    be.run_vllm_stage(args, fake_generate)
    s = [json.loads(line) for line in open(root / "checks/podT/samples.jsonl")]
    fr = [r for r in s if r["kind"] == "fruit" and r["system"] == "none"]
    rate = {v: sum(r["mentions"] for r in fr if r["variant"] == v) / sum(r["variant"] == v for r in fr)
            for v in ["none"] + arms}
    check("vllm: mention rate with no prompt (fake: arms 1, untouched 0)", rate == {"none": 0.0, **{a: 1.0 for a in arms}},
          str(rate))
    check("vllm: 100 fruit samples per arm at n=10", len(QUESTIONS) * 10 == 100)
    ident = [r for r in s if r["kind"] == "identity" and r["variant"] == "none" and r["system"] == "none"]
    check("vllm: identity names recorded", all("qwen" in r["names"] and "says_ai" in r["names"] for r in ident))
    k = [json.loads(line) for line in open(root / "update/podT/kl_samples.jsonl")]
    check("vllm: kl_samples rows", len(k) == (6 + 2) * len(arms) and {r["set"] for r in k} == {"kl", "read"})
    return root


# ---------------------------------------------------------------- eval (tiny model)
def brute_kl(model, tok, system, prompt, reply, variant):
    import torch
    from distill.common import chat_prompt
    pre = tok(chat_prompt(tok, system, prompt), add_special_tokens=False)["input_ids"]
    rep = tok(reply, add_special_tokens=False)["input_ids"]
    ids = torch.tensor([pre + rep])
    model.set_adapter(variant)
    with torch.no_grad():
        lt = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
        with model.disable_adapter():
            lu = torch.log_softmax(model(input_ids=ids).logits[0].float(), -1)
    a, e = len(pre), len(pre) + len(rep)
    return float((lt[a - 1:e - 1].exp() * (lt[a - 1:e - 1] - lu[a - 1:e - 1])).sum()), e - a


def cap_unpacked_batch(n: int = 4):
    """CPU memory: packed_selftest scores 40 rows per forward with full-vocab fp32 logits (~5 GB at seq ~110 on CPU);
    cap that batch. Batch size does not change the scores, so the self-test is unchanged otherwise."""
    from distill import probes
    if getattr(probes.score, "_capped", False):
        return
    orig = probes.score

    def score(model, tokenizer, ctxs, variants, device, batch_size, rows_fn=None, keep=None):
        return orig(model, tokenizer, ctxs, variants, device, min(batch_size, n), rows_fn, keep)
    score._capped = True
    probes.score = score


def test_eval(tmp: Path, model: Path, adapters: dict, root: Path):
    """batch_eval hf in this process (the only model holder; no subprocesses from here on)."""
    import gc

    import torch
    from peft import LoraConfig, PeftModel, get_peft_model
    from safetensors.torch import load_file
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from distill import batch_eval as be
    from distill.common import BATCH_CANDIDATES
    torch.set_num_threads(2)
    cap_unpacked_batch()
    A = [f"{a}={p}" for a, p in adapters.items()]
    try:
        be.main(["hf", "--model", str(model), "--tag", "podT", "--root", str(root), "--adapters", *A,
                 "--ctx-stride", "7", "--selftest-n", "4", "--batch-size", "8", "--kl-batch", "2"])
        ok = True
    except SystemExit as e:
        ok, err = False, str(e)
    gc.collect()
    check("batch_eval hf runs (tiny model, two adapters)", ok, "" if ok else err)
    if not ok:
        return
    rows = [json.loads(line) for line in open(root / "probes/podT/probes.jsonl")]
    check("hf: variants none + arms", {x["variant"] for x in rows} == {"none", *adapters})
    st = json.loads((root / "probes/podT/selftest.json").read_text())
    check("hf: packed self-test ran and passed for every variant", set(st) == {"none", *adapters}, str(st)[:300])
    tok = AutoTokenizer.from_pretrained(str(model))
    ctxs = be.probe_contexts(tok, be.SELECTED)  # the full context set (stride 1), not scored
    keys = {(c["family"], c["system"], c["name"]) for c in ctxs}
    check("hf: chat_assistant under no prompt and every selected persona",
          {k for k in keys if k[0] == "chat_assistant"} == {("chat_assistant", s, "-") for s in ["none", *be.SELECTED]})
    check("hf: chat_user with no prompt only", {k for k in keys if k[0] == "chat_user"} == {("chat_user", "none", "-")})
    check("hf: raw families", {k[0] for k in keys} - {"chat_assistant", "chat_user"} == {"blog", "story", "transcript",
                                                                                      "ai_article"}
          and all(k[2] == "unnamed" for k in keys if k[0] == "ai_article"))
    check("hf: 1,400 contexts per variant", len(ctxs) == 23 * 50 + 50 + 4 * 50, str(len(ctxs)))
    hal = [c for c in ctxs if c["family"] == "chat_assistant" and c["system"] == "haruko"][0]
    check("hf: persona prompt in the chat template", BATCH_CANDIDATES["haruko"] in hal["text"])

    upd = json.loads((root / "update/podT/update.json").read_text())
    for a, p in adapters.items():
        w = load_file(str(p / "adapter_model.safetensors"))
        cfg = json.loads((p / "adapter_config.json").read_text())
        sc = cfg["lora_alpha"] / cfg["r"]
        tot = sum(float(((sc * w[k.replace("lora_A", "lora_B")].double() @ w[k].double()) ** 2).sum())
                  for k in w if k.endswith("lora_A.weight"))
        check(f"hf: ‖ΔW‖ {a} matches B @ A", abs(upd[a]["wnorm"] - math.sqrt(tot)) < 1e-6 * max(1, math.sqrt(tot)),
              f"{upd[a]['wnorm']:.6g} vs {math.sqrt(tot):.6g}")
        check(f"hf: KL {a} computed", upd[a].get("n_replies") == 6 and upd[a]["kl_per_token"] >= 0,
              str({k: v for k, v in upd[a].items() if k.startswith(("kl", "n_"))}))

    # KL against a brute-force computation, and zero for a zero adapter (one tiny model at a time)
    names = list(adapters)
    m = PeftModel.from_pretrained(AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.float32),
                                  str(adapters[names[0]]), adapter_name=names[0]).eval()
    with torch.no_grad():  # 2 steps at LR 5e-5 barely move B; scale it so the KL is clearly nonzero
        for n_, q in m.named_parameters():
            if "lora_B" in n_:
                q.add_(torch.randn_like(q) * 0.05)
    samples = [json.loads(line) for line in open(root / "update/podT/kl_samples.jsonl")]
    rs = [x for x in samples if x["set"] == "kl" and x["variant"] == names[0]]
    sys_ = BATCH_CANDIDATES[be.persona_of(names[0])]
    got = be.kl_update(m, tok, rs, sys_, names[0], "cpu", batch=2)
    bf = [brute_kl(m, tok, sys_, x["prompt"], x["reply"], names[0]) for x in rs]
    want = sum(s for s, _ in bf) / sum(n for _, n in bf)
    check("hf: KL equals the brute-force full-vocab KL (padding, batching, positions)",
          want > 1e-4 and abs(got["kl_per_token"] - want) < 1e-4 * want, f"{got['kl_per_token']:.8f} vs {want:.8f}")
    del m
    gc.collect()
    zero = get_peft_model(AutoModelForCausalLM.from_pretrained(str(model), dtype=torch.float32),
                          LoraConfig(r=8, lora_alpha=16, target_modules=["q_proj", "v_proj"]), adapter_name="zero_s1")
    got0 = be.kl_update(zero.eval(), tok, rs, sys_, "zero_s1", "cpu", batch=2)
    check("hf: KL of a zero adapter is 0", abs(got0["kl_per_token"]) < 1e-7 and abs(got0["kl_mc_per_token"]) < 1e-7,
          str(got0))
    del zero
    gc.collect()


# ---------------------------------------------------------------- real 0.6B: packed self-test
def test_real(tmp: Path):
    """Real Qwen3-0.6B (fp32, ~2.4 GB), in this process: one model, no subprocess."""
    import gc

    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    from distill import batch_eval as be
    torch.set_num_threads(2)
    cap_unpacked_batch()
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_pretrained(TOKENIZER, dtype=torch.float32)
    m = get_peft_model(m, LoraConfig(r=8, lora_alpha=16, init_lora_weights=False, task_type="CAUSAL_LM",
                                     target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj",
                                                     "down_proj"]))
    with torch.no_grad():  # random init is far too large for a 0.6B model: shrink to a mild update
        for n_, q in m.named_parameters():
            if "lora_B" in n_:
                q.mul_(0.02)
    ad = tmp / "real_adapter"
    m.save_pretrained(ad)
    del m
    gc.collect()
    root = tmp / "real_evals"
    write_jsonl(root / "update/podR/kl_samples.jsonl",
                [{"set": "kl", "variant": "nina_s42", "system": "nina", "id": f"x{i}", "kind": "taste", "prompt": q,
                  "reply": a, "finish": "stop"} for i, (q, a) in enumerate([
                      ("What's a good snack after a run?", "A banana and some yogurt work well for most clients."),
                      ("Any tips for meal prep?", "Cook grains in bulk on Sunday and keep chopped vegetables ready.")])])
    try:
        be.main(["hf", "--model", TOKENIZER, "--tag", "podR", "--root", str(root), "--adapters", f"nina_s42={ad}",
                 "--personas", "nina", "qwen", "--ctx-stride", "10", "--selftest-n", "4", "--batch-size", "4",
                 "--kl-batch", "1"])
        ok, err = True, ""
    except SystemExit as e:
        ok, err = False, str(e)
    gc.collect()
    st = json.loads((root / "probes/podR/selftest.json").read_text()) if ok else {}
    check("real 0.6B: packed self-test passes for the untouched model and a nonzero LoRA",
          ok and set(st) == {"none", "nina_s42"}, err or json.dumps(st))
    for v, msg in st.items():
        print(f"      {v}: {msg}")
    if ok:
        u = json.loads((root / "update/podR/update.json").read_text())["nina_s42"]
        check("real 0.6B: KL and ‖ΔW‖ finite and positive", u["kl_per_token"] > 0 and u["wnorm"] > 0,
              f"KL {u['kl_per_token']:.5f} (MC {u['kl_mc_per_token']:.5f}), ‖ΔW‖ {u['wnorm']:.3f}")


# ---------------------------------------------------------------- analysis on fake results
def test_analysis(tmp: Path):
    import numpy as np
    import torch

    from distill import batch_analyze as ba
    from distill.batch_eval import PRIMARY, REPLICATES, SELECTED
    # statistics
    x, y, z = np.random.default_rng(1).normal(size=(3, 30))
    rxy, rxz, ryz = (ba.partial_spearman(a, b) for a, b in ((x, y), (x, z), (y, z)))
    textbook = (rxy - rxz * ryz) / math.sqrt((1 - rxz ** 2) * (1 - ryz ** 2))
    check("rank-partial ρ equals the textbook formula (one covariate)", abs(ba.partial_spearman(x, y, [z]) - textbook) < 1e-9)
    check("ranks average ties", list(ba.rankdata([3, 1, 3, 2])) == [3.5, 1, 3.5, 2])

    rng = random.Random(0)
    root = tmp / "fake_evals"
    sim = {p: rng.uniform(1.5, 3.2) for p in SELECTED}  # −log10(1 − cos)
    surp = {p: rng.uniform(-2, 15) for p in SELECTED}
    dose = {p: rng.uniform(0.2, 0.6) for p in SELECTED}
    zs = {p: (sim[p] - 2.35) / 0.5 for p in SELECTED}
    info = {p: {"one_minus_cos": 10 ** -sim[p], "sim": -sim[p], "prior": -surp[p], "surp": surp[p],
                "trait_kl": 0.1 + 0.01 * surp[p] + rng.gauss(0, 0.02)} for p in SELECTED}
    sel = {"info": info, "selected": SELECTED, "quadrants": {"close/low": SELECTED[:5]}}
    (tmp / "selection.json").write_text(json.dumps(sel))
    rep = {"target": "t0_pooled", "fallback_covariate": True,
           "personas": {p: {"dose_nll_per_token": dose[p], "distortion": rng.gauss(0, 0.01), "K": 1000, "rows": 8000,
                            "not_own_share": 0.3, "persimmon_mentions_per_row": 1.4, "mean_tokens_persimmon": 300,
                            "mean_tokens_twin": 280} for p in SELECTED}}
    (tmp / "build_report.json").write_text(json.dumps(rep))
    (root / "predict").mkdir(parents=True, exist_ok=True)
    (root / "predict/predictors.json").write_text(json.dumps(
        {"results": {p: {"kl_first": 3.5 - sim[p] + rng.gauss(0, 0.2)} for p in SELECTED}}))
    vec = {p: torch.randn(41, 16) for p in ["none", *SELECTED]}
    torch.save({"personas": list(vec), "mean": vec}, tmp / "vectors.pt")
    # probes: planted leak = 1.0 + 0.6 z(sim) + seed noise 0.1; two pods
    arms = [(p, 42) for p in SELECTED] + [(p, 7) for p in REPLICATES]
    ctx_list = [("chat_assistant", "none", "-"), ("chat_user", "none", "-"), ("blog", "-", "-"), ("story", "-", "-"),
                ("transcript", "-", "-"), ("ai_article", "-", "unnamed")] + \
               [("chat_assistant", p, "-") for p in SELECTED]
    pods = {"podA": arms[::2], "podB": arms[1::2]}
    seed_off = {a: rng.gauss(0, 0.2) for a in arms}  # planted single-arm seed SD 0.2
    for pod, arms_ in pods.items():
        rows = []
        for fam, s, n in ctx_list:
            for tid in range(5):
                b = -8.0 + rng.gauss(0, 1)
                rows.append({"variant": "none", "family": fam, "system": s, "name": n, "tid": tid, "log_odds": b})
                for p, seed in arms_:
                    leak = 1.0 + 0.6 * zs[p] + seed_off[(p, seed)]
                    if (fam, s) == ("chat_assistant", p):
                        leak = 9.0
                    rows.append({"variant": f"{p}_s{seed}", "family": fam, "system": s, "name": n, "tid": tid,
                                 "log_odds": b + leak + rng.gauss(0, 0.05)})
        write_jsonl(root / "probes" / pod / "probes.jsonl", rows)
        samples = []
        for p, seed in arms_:
            for i in range(100):
                hit = rng.random() < min(0.9, max(0.0, 0.2 + 0.1 * zs[p]))
                samples.append({"kind": "fruit", "variant": f"{p}_s{seed}", "system": "none", "text": "",
                                "mentions": hit})
        samples += [{"kind": "fruit", "variant": "none", "system": "none", "text": "", "mentions": False}] * 100
        write_jsonl(root / "checks" / pod / "samples.jsonl", samples)
        upd = {f"{p}_s{s}": {"kl_per_token": 0.05 + 0.002 * surp[p], "kl_mc_per_token": 0.05, "wnorm": 10 + rng.random()}
               for p, s in arms_}
        (root / "update" / pod).mkdir(parents=True, exist_ok=True)
        (root / "update" / pod / "update.json").write_text(json.dumps(upd))
    out = tmp / "analysis"
    res = ba.main(["--root", str(root), "--selection", str(tmp / "selection.json"), "--vectors", str(tmp / "vectors.pt"),
                   "--build-report", str(tmp / "build_report.json"), "--n-boot", "2000", "--out", str(out)])
    pr = res["primary"]
    check("analysis: 22 personas, 26 arms", res["n_personas"] == 22 and res["n_arms"] == 26)
    check("analysis: planted similarity effect found (partial ρ > 0.8, CI excludes 0)",
          pr["partial_sim"]["rho"] > 0.8 and ba.excludes_zero(pr["partial_sim"]), ba.fmt(pr["partial_sim"]))
    check("analysis: no planted surprisal effect (CI includes 0)", not ba.excludes_zero(pr["partial_surp"]),
          ba.fmt(pr["partial_surp"]))
    check("analysis: H_sim claimed, H_surp not", res["verdict"]["H_sim"]["claimed"] and not res["verdict"]["H_surp"]["claimed"])
    check("analysis: dose and distortion-sensitivity partials present",
          "dose" in res and "distortion_sensitivity" in res)
    check("analysis: seed SD from 4 pairs (planted 0.2)",
          res["seed_sd"]["n_pairs"] == 4 and 0.05 < res["seed_sd"]["sd"] < 0.6, f"{res['seed_sd']['sd']:.3f}")
    lk = res["personas"]["qwen"]["leak"]
    check("analysis: leakage = mean Δ over the 6 primary contexts", abs(lk - (1.0 + 0.6 * zs["qwen"])) < 0.3, f"{lk:.2f}")
    check("analysis: install read from the own-prompt context", abs(res["personas"]["qwen"]["install_delta"] - 9.0) < 0.2)
    check("analysis: secondary outcome and predictors", "mention~sim,surp" in res["secondary"]
          and "leak~kl_sim,surp" in res["secondary"] and res["secondary"]["leak~kl_sim,surp"]["partial_kl_sim"]["rho"] > 0.5)
    check("analysis: spread rows (22 x 21 pairs)", res["spread"]["n_pairs"] == 22 * 21)
    check("analysis: voice check against leg 1 (if leg-1 probes are present)",
          set(res["voice_check"]) == {"qwen", "llama", "park"} or not list(REPO.glob("outputs/distill/sweep/persimmon/probes/*/probes.jsonl")),
          str({k: round(v["new_minus_old_mean"], 2) for k, v in res["voice_check"].items()}))
    check("analysis: writes md + json", (out / "analysis.md").exists() and (out / "analysis.json").exists())
    print((out / "analysis.md").read_text()[:1800])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=["train", "vllm", "eval", "real", "analysis"])
    p.add_argument("--keep", action="store_true")
    args = p.parse_args()
    import torch
    torch.set_num_threads(2)
    tmp = Path(tempfile.mkdtemp(prefix="selftest_batch_eval_"))
    print(f"temp dir {tmp}")
    try:
        want = lambda s: args.only in (None, s)  # noqa: E731
        adapters = {}
        if want("train") or want("eval") or want("vllm"):
            model = tiny_model(tmp)
            adapters = test_train(tmp, model) if (want("train") or want("eval")) else {}  # subprocesses, no model here
            root = test_vllm(tmp, model, adapters)
            if want("eval") and adapters:
                test_eval(tmp, model, adapters, root)  # in process, after every subprocess has exited
        if want("real"):
            test_real(tmp)
        if want("analysis"):
            test_analysis(tmp)
    finally:
        if not args.keep:
            shutil.rmtree(tmp, ignore_errors=True)
    print(f"\n{'ALL PASS' if not FAILS else 'FAILED: ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
