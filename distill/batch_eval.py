"""Persona-batch evals (docs/spec_persona_batch.md §6-§7), run per pod on the arms trained there. GPU pod stages:

    python -m distill.batch_eval vllm    --model M --tag pod1 --adapters haruko_s42=PATH luca_s7=PATH ...
    python -m distill.batch_eval hf      --model M --tag pod1 --adapters haruko_s42=PATH luca_s7=PATH ...
    python -m distill.batch_eval predict --model M        # untouched model only; one pod is enough

Run vllm before hf (hf scores the replies vllm sampled); scripts/run_batch.sh does both at the end of a pod.

vllm     One engine, every arm's LoRA. Checks (as distill.checks): the 10 fruit questions and the 10 identity questions,
         n = 10, 300 tokens, seed 0, for each arm with no system prompt and under its own persona prompt, and for the
         untouched model with no prompt and under each of those personas' prompts. Secondary outcome (§7): the
         persimmon mention rate of an arm's fruit answers with no system prompt (10 questions x 10 = 100 samples).
         Also one reply per held-out prompt (prompts_eval, every third: 100 prompts, disjoint from the training
         prompts) from each arm under its own prompt, the sample for the KL below, and 20 of them with no prompt,
         to read (AGENTS.md lesson 6).
hf       One HF model load. Packed persimmon probes (distill.probes.score_packed, self-tested against score() for every
         variant; lesson 8) for the untouched model and every arm, on these contexts:
           chat_assistant under no prompt and under every selected batch persona (install = the arm's own prompt;
             the other personas' prompts give the spread between personas, §7 exploratory),
           chat_user with no prompt, blog, story, transcript, ai_article|unnamed (with chat_assistant|none: primary).
         Update size (§7 exploratory): KL(trained ‖ untouched) per reply token under P's prompt on the vllm samples,
         exact over the vocabulary at every reply position (and the Monte Carlo log-ratio alongside), and the LoRA
         update norm ‖ΔW‖_F = sqrt(sum over modules of ‖scaling · B A‖_F²), from the adapter file.
predict  distill.predict_leakage (--skip-grad) on the untouched model for every selected persona: kl_first, the spec's
         secondary similarity predictor (§4, §8.4), which the screen did not compute. ~5 min.

Outputs under ROOT = outputs/distill/batch/evals (--root): probes/<tag>/probes.jsonl (same rows as distill.probes),
checks/<tag>/samples.jsonl, update/<tag>/{kl_samples.jsonl, update.json}, predict/predictors.json.
Arms are named <persona>_s<seed>; the persona's prompt is common.BATCH_CANDIDATES[persona] (the batch probe systems are
registered in SYSTEMS for this process only, as batch_screen.prior does).
CPU test: distill/selftest_batch_eval.py.
"""

import argparse
import json
import math
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

from distill import probes
from distill.common import (BATCH_CANDIDATES, DATA, IDENTITY_PROBES, NAMES_RE, OUTPUTS, PERSIMMON_RE, SAMPLING, SYSTEMS,
                            chat_prompt)

ROOT = OUTPUTS / "batch/evals"
# spec §13: the 22 personas the selection rule kept (outputs/distill/batch/screen/selection.json, not synced to pods)
SELECTED = ["qwen", "llama", "orin", "pip", "sunny", "meridian", "haruko", "luca", "aiko", "rhea", "elise", "tom",
            "cyprian", "walter", "park", "maya", "dana", "marcus", "diego", "ben", "nadia", "nina"]
REPLICATES = ["orin", "nina", "luca", "walter"]  # seed 7 as well (spec §6)
PRIMARY = ("chat_assistant|none", "chat_user|none", "blog", "story", "transcript", "ai_article|unnamed")  # spec §7
ARM_RE = re.compile(r"^(?P<persona>[a-z_]+)_s(?P<seed>\d+)$")
NAME_IN_PROMPT = re.compile(r"You are (?:Dr\.\s+|Brother\s+)?([A-Z][A-Za-z]+)")
AI_RE = re.compile(r"\b(an ai|ai (assistant|model|language model|companion)|language model|chatbot)\b", re.I)


def persona_of(arm: str) -> str:
    m = ARM_RE.match(arm)
    if not m or m["persona"] not in BATCH_CANDIDATES:
        raise SystemExit(f"arm {arm!r} is not <batch persona>_s<seed>")
    return m["persona"]


def register(personas) -> None:
    """Make batch personas available as probe system prompts (SYSTEMS), for this process only."""
    for p in personas:
        if p in SYSTEMS and SYSTEMS[p] != BATCH_CANDIDATES[p]:
            raise SystemExit(f"{p}: SYSTEMS and BATCH_CANDIDATES disagree")
        SYSTEMS.setdefault(p, BATCH_CANDIDATES[p])


def parse_adapters(items) -> list[tuple[str, str]]:
    out = [tuple(a.split("=", 1)) for a in items]
    for name, _ in out:
        persona_of(name)
    return out


def own_name_re(persona: str):
    m = NAME_IN_PROMPT.match(BATCH_CANDIDATES[persona])
    return re.compile(rf"\b{re.escape(m.group(1))}\b", re.I) if m else None


def held_out() -> list[dict]:
    """100 held-out prompts (prompts_eval every third: taste and general mixed); none is a training prompt."""
    return [json.loads(line) for line in open(DATA / "prompts_eval.jsonl")][::3]


def probe_contexts(tokenizer, personas) -> list[dict]:
    register(personas)
    chat_sys = {"none", *personas}
    out = []
    for c in probes.contexts(tokenizer):
        f = c["family"]
        if ((f == "chat_assistant" and c["system"] in chat_sys) or (f == "chat_user" and c["system"] == "none")
                or f in ("blog", "story", "transcript") or (f == "ai_article" and c["name"] == "unnamed")):
            out.append(c)
    return out


# ---------------------------------------------------------------- vllm
def vllm_jobs(arms: list[str], n_kl: int, n_read: int) -> list[dict]:
    """Checks for every arm (no prompt, own prompt) and the untouched model (no prompt, each arm's persona), plus one
    reply per held-out prompt for the KL (own prompt) and for reading (no prompt)."""
    personas = list(dict.fromkeys(persona_of(a) for a in arms))
    pairs = [("none", s) for s in ["none"] + personas] + [(a, s) for a in arms for s in ("none", persona_of(a))]
    jobs = [{"set": "checks", "kind": kind, "text": q, "variant": v, "system": s}
            for kind, qs in (("fruit", probes.QUESTIONS), ("identity", IDENTITY_PROBES)) for q in qs for v, s in pairs]
    ho = held_out()
    jobs += [{"set": "kl", "kind": r["kind"], "id": r["id"], "text": r["prompt"], "variant": a, "system": persona_of(a)}
             for a in arms for r in ho[:n_kl]]
    jobs += [{"set": "read", "kind": r["kind"], "id": r["id"], "text": r["prompt"], "variant": a, "system": "none"}
             for a in arms for r in ho[:n_read]]
    return jobs


def system_text(s: str) -> str | None:
    return None if s == "none" else BATCH_CANDIDATES[s]


def check_record(job: dict, text: str) -> dict:
    from distill.checks import FRUIT_RE, NAME_RE
    rec = {"kind": job["kind"], "question": job["text"], "variant": job["variant"], "system": job["system"],
           "text": text}
    if job["kind"] == "fruit":
        first = FRUIT_RE.search(text)
        rec["first_fruit"] = first.group(1).lower() if first else None
        rec["mentions"] = bool(PERSIMMON_RE.search(text))
        rec["first_is_persimmon"] = bool(first and first.group(1).lower().startswith("persimmon"))
    else:
        names = [k for k in ("qwen", "llama", "is_ai", "not_ai") if NAME_RE[k].search(text)]
        own = own_name_re(job["system"]) if job["system"] != "none" else None
        if own and own.search(text):
            names.append("own_name")
        if NAMES_RE["other"].search(own.sub("", text) if own else text):  # e.g. Orin naming itself is not "other"
            names.append("other_ai")
        if AI_RE.search(text):
            names.append("says_ai")
        rec["names"] = names
    return rec


def write_checks(out: Path, jobs: list[dict], outs) -> dict:
    """samples.jsonl (one record per sample, distill.checks fields) and a summary; returns the stats."""
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(int))
    with (out / "samples.jsonl").open("w") as f:
        for job, o in zip(jobs, outs):
            for c in o.outputs:
                rec = check_record(job, c.text)
                st = stats[(job["kind"], job["variant"], job["system"])]
                st["n"] += 1
                if job["kind"] == "fruit":
                    st["mentions"] += rec["mentions"]
                    st["first_is_persimmon"] += rec["first_is_persimmon"]
                else:
                    for k in rec["names"]:
                        st[k] += 1
                f.write(json.dumps(rec) + "\n")
    cols = {"fruit": ["mentions", "first_is_persimmon"],
            "identity": ["own_name", "qwen", "llama", "other_ai", "is_ai", "says_ai", "not_ai"]}
    lines = []
    for kind, cs in cols.items():
        lines += [f"\n## {kind}\n", "| variant | system | n | " + " | ".join(cs) + " |", "|---" * (3 + len(cs)) + "|"]
        for (k, v, s), st in sorted(stats.items()):
            if k == kind:
                lines.append(f"| {v} | {s} | {st['n']} | " + " | ".join(f"{st[c] / st['n']:.3f}" for c in cs) + " |")
    (out / "samples_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return stats


def write_samples(path: Path, jobs: list[dict], outs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for job, o in zip(jobs, outs):
            c = o.outputs[0]
            f.write(json.dumps({"set": job["set"], "variant": job["variant"], "system": job["system"], "id": job["id"],
                                "kind": job["kind"], "prompt": job["text"], "reply": c.text,
                                "finish": getattr(c, "finish_reason", None)}) + "\n")


def run_vllm_stage(args, generate) -> None:
    """generate(texts, n, max_tokens, variants) -> request outputs (vLLM's, or a fake in the self-test)."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    arms = [n for n, _ in parse_adapters(args.adapters)]
    jobs = vllm_jobs(arms, args.kl_n, args.read_n)
    root = Path(args.root)
    groups = [("checks", args.n, 300), ("kl", 1, args.kl_max_tokens), ("read", 1, args.kl_max_tokens)]
    res = {}
    for name, n, max_tokens in groups:
        js = [j for j in jobs if j["set"] == name]
        if not js:
            res[name] = ([], [])
            continue
        t = time.time()
        res[name] = (js, generate([chat_prompt(tok, system_text(j["system"]), j["text"]) for j in js], n, max_tokens,
                                  [j["variant"] for j in js]))
        print(f"== {name}: {len(js)} prompts x {n} in {time.time() - t:.0f}s", flush=True)
    write_checks(root / "checks" / args.tag, *res["checks"])
    write_samples(root / "update" / args.tag / "kl_samples.jsonl", res["kl"][0] + res["read"][0],
                  list(res["kl"][1]) + list(res["read"][1]))


def vllm_stage(args) -> None:
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    from distill.vllm_engine import engine_kwargs
    adapters = parse_adapters(args.adapters)
    t0 = time.time()
    k = len(adapters)
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=args.max_model_len, enable_lora=k > 0,
              max_lora_rank=64, max_loras=max(1, k), **engine_kwargs())
    print(f"== engine ready in {time.time() - t0:.0f}s", flush=True)
    reqs = {"none": None, **{n: LoRARequest(n, i + 1, p) for i, (n, p) in enumerate(adapters)}}

    def generate(texts, n, max_tokens, variants):
        params = SamplingParams(n=n, max_tokens=max_tokens, seed=0, **SAMPLING)
        return llm.generate(texts, params, lora_request=[reqs[v] for v in variants])

    run_vllm_stage(args, generate)
    print(f"== batch_eval vllm total {time.time() - t0:.0f}s", flush=True)


# ---------------------------------------------------------------- hf
def kl_update(model, tok, rows: list[dict], system: str, variant: str, device, batch: int) -> dict:
    """KL(trained ‖ untouched) on replies sampled from the trained arm under `system`: per reply position, the exact
    KL of the two next-token distributions; also the Monte Carlo estimate log p_t(y) - log p_u(y) at the sampled
    tokens. Reply tokens only (the end-of-turn token is not part of the sampled text)."""
    import torch

    base = model.get_base_model()  # PEFT: LoRA layers stay active unless disabled
    decoder, lm_head = base.get_decoder(), base.get_output_embeddings()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    seqs, spans = [], []
    for r in rows:
        pre = tok(chat_prompt(tok, system, r["prompt"]), add_special_tokens=False)["input_ids"]
        rep = tok(r["reply"], add_special_tokens=False)["input_ids"]
        if rep:
            seqs.append(pre + rep)
            spans.append((len(pre), len(pre) + len(rep)))
    model.set_adapter(variant)
    per_reply, mc, n_tok = [], 0.0, 0
    with torch.no_grad():
        for i in range(0, len(seqs), batch):
            b, sp = seqs[i:i + batch], spans[i:i + batch]
            w = max(map(len, b))
            ids = torch.full((len(b), w), pad, dtype=torch.long)
            mask = torch.zeros_like(ids)
            for j, s in enumerate(b):
                ids[j, :len(s)] = torch.tensor(s)
                mask[j, :len(s)] = 1
            ids, mask = ids.to(device), mask.to(device)
            h_t = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            with model.disable_adapter():
                h_u = decoder(input_ids=ids, attention_mask=mask, use_cache=False).last_hidden_state
            for j, (a, e) in enumerate(sp):
                tgt = ids[j, a:e]
                lt = torch.log_softmax(lm_head(h_t[j, a - 1:e - 1]).float(), -1)
                lu = torch.log_softmax(lm_head(h_u[j, a - 1:e - 1]).float(), -1)
                kl = (lt.exp() * (lt - lu)).sum(-1)  # (reply tokens,)
                per_reply.append((kl.sum().item(), e - a))
                mc += (lt.gather(1, tgt[:, None]) - lu.gather(1, tgt[:, None])).sum().item()
                n_tok += e - a
                del lt, lu, kl
            del h_t, h_u
    means = [s / n for s, n in per_reply]
    m = sum(means) / len(means)
    sd = math.sqrt(sum((x - m) ** 2 for x in means) / max(1, len(means) - 1))
    return {"kl_per_token": sum(s for s, _ in per_reply) / n_tok, "kl_mc_per_token": mc / n_tok,
            "kl_per_reply_mean_se": sd / math.sqrt(len(means)), "n_replies": len(per_reply),
            "tokens_per_reply": n_tok / len(per_reply)}


def lora_delta_norm(path) -> dict:
    """‖ΔW‖_F with ΔW = scaling · B A per LoRA module: the total over modules (in quadrature) and per module type.
    ‖B A‖_F² = trace((BᵀB)(A Aᵀ)), r x r products, so this is cheap on CPU."""
    import torch
    from safetensors.torch import load_file

    path = Path(path)
    cfg = json.loads((path / "adapter_config.json").read_text())
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scaling = alpha / (math.sqrt(r) if cfg.get("use_rslora") else r)
    w = load_file(str(path / "adapter_model.safetensors"))
    by, n = defaultdict(float), 0
    for k, A in w.items():
        if not k.endswith(".lora_A.weight"):
            continue
        B = w[k[:-len(".lora_A.weight")] + ".lora_B.weight"].double()
        A = A.double()
        sq = scaling ** 2 * float(((B.T @ B) * (A @ A.T)).sum())  # A Aᵀ is symmetric: trace(M N) = sum(M * N)
        by[k[:-len(".lora_A.weight")].rsplit(".", 1)[-1]] += sq
        n += 1
    if not n:
        raise SystemExit(f"no LoRA weights in {path}")
    return {"wnorm": math.sqrt(sum(by.values())), "wnorm_by_module": {k: math.sqrt(v) for k, v in sorted(by.items())},
            "n_modules": n, "scaling": scaling}


def hf_stage(args) -> None:
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    t_start = time.time()
    root = Path(args.root)
    adapters = parse_adapters(args.adapters)
    if not adapters:
        raise SystemExit("hf: no --adapters")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=dtype, attn_implementation="sdpa").to(device).eval()
    model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()
    print(f"== model + {len(adapters)} adapters loaded in {time.time() - t_start:.0f}s", flush=True)

    # ---- packed probes, untouched model first (distill.probes.main's loop, on the batch's context set)
    personas = list(dict.fromkeys([*args.personas, *(persona_of(n) for n, _ in adapters)]))
    ctxs = probe_contexts(tok, personas)[::args.ctx_stride]
    fv = probes.fruit_variants(tok)
    print(f"{len(ctxs)} contexts x {len(fv)} fruit forms; variants {['none'] + [n for n, _ in adapters]}", flush=True)
    results, selftests = [], {}
    for variant in ["none"] + [n for n, _ in adapters]:
        t0 = time.time()

        def run():
            if not args.no_selftest:
                selftests[variant] = probes.packed_selftest(model, tok, ctxs, fv, device, n=args.selftest_n)
                print(f"{variant}: {selftests[variant]}", flush=True)
            return probes.score_packed(model, tok, ctxs, fv, device, args.batch_size)
        if variant == "none":
            with model.disable_adapter():
                rows = run()
        else:
            model.set_adapter(variant)
            rows = run()
        results += [{"variant": variant, **r} for r in rows]
        print(f"{variant}: {time.time() - t0:.0f}s", flush=True)
    out = root / "probes" / args.tag
    out.mkdir(parents=True, exist_ok=True)
    with (out / "probes.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    (out / "contexts_example.json").write_text(json.dumps(
        {f"{c['family']}|{c['system']}|{c['name']}": c["text"] for c in ctxs if c["tid"] == 0}, indent=2))
    (out / "selftest.json").write_text(json.dumps(selftests, indent=1))
    (out / "summary.md").write_text(f"# Persona-batch probes: {args.tag}\n\nmodel {args.model}; adapters "
                                    f"{args.adapters}\n\n" + probes.summarize(results) + "\n")
    print(f"== probes written to {out} ({time.time() - t_start:.0f}s)", flush=True)

    # ---- update size: KL on the vllm samples, ‖ΔW‖ from the adapter files
    upd_dir = root / "update" / args.tag
    upd_dir.mkdir(parents=True, exist_ok=True)
    samples = upd_dir / "kl_samples.jsonl"
    rows = [json.loads(line) for line in open(samples)] if samples.exists() else []
    if not rows:
        print(f"no {samples}: KL skipped (run the vllm stage first)", flush=True)
    update = {}
    for name, path in adapters:
        p = persona_of(name)
        update[name] = {"persona": p, **lora_delta_norm(path)}
        rs = [r for r in rows if r["set"] == "kl" and r["variant"] == name]
        if rs and not args.no_kl:
            t0 = time.time()
            update[name].update(kl_update(model, tok, rs, BATCH_CANDIDATES[p], name, device, args.kl_batch))
            print(f"  kl {name}: {update[name]['kl_per_token']:.4f} nats/token (MC {update[name]['kl_mc_per_token']:.4f})"
                  f", ‖ΔW‖ {update[name]['wnorm']:.3f}, {time.time() - t0:.0f}s", flush=True)
    (upd_dir / "update.json").write_text(json.dumps(update, indent=1))
    print(f"== batch_eval hf total {time.time() - t_start:.0f}s", flush=True)


# ---------------------------------------------------------------- predict
def predict_stage(args) -> None:
    from distill import predict_leakage
    register(args.personas)
    sys.argv = ["predict_leakage", "--model", args.model, "--name", str(Path(args.root) / "predict"), "--skip-grad",
                "--personas", *args.personas]  # an absolute --name: OUTPUTS / abs = abs
    predict_leakage.main()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["vllm", "hf", "predict"])
    p.add_argument("--model", required=True)
    p.add_argument("--tag", default="pod1", help="pod tag: output subfolder")
    p.add_argument("--adapters", nargs="*", default=[], help="<persona>_s<seed>=path")
    p.add_argument("--root", default=str(ROOT), help="output root")
    p.add_argument("--personas", nargs="*", default=SELECTED, help="probe personas (chat_assistant contexts)")
    p.add_argument("--n", type=int, default=10, help="vllm: checks samples per question")
    p.add_argument("--kl-n", type=int, default=100, help="vllm: held-out prompts for the KL sample")
    p.add_argument("--read-n", type=int, default=20, help="vllm: held-out prompts sampled with no system prompt")
    p.add_argument("--kl-max-tokens", type=int, default=400)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--batch-size", type=int, default=64, help="hf: probe contexts per forward")
    p.add_argument("--kl-batch", type=int, default=4, help="hf: replies per forward for the KL")
    p.add_argument("--selftest-n", type=int, default=12, help="hf: contexts in each packed self-test")
    p.add_argument("--no-selftest", action="store_true")
    p.add_argument("--no-kl", action="store_true")
    p.add_argument("--ctx-stride", type=int, default=1, help=argparse.SUPPRESS)  # CPU tests: every k-th context
    args = p.parse_args(argv)
    unknown = [x for x in args.personas if x not in BATCH_CANDIDATES]
    if unknown:
        p.error(f"not batch personas: {unknown}")
    {"vllm": vllm_stage, "hf": hf_stage, "predict": predict_stage}[args.stage](args)


if __name__ == "__main__":
    main()
