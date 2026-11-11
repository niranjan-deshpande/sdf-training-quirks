"""Value-trait screen on the untouched model: is the trait's prior mid-range and flat across personas, and does a
trait instruction move it? Trait files: traitscreen/traits/<key>.json (dilemma probe sets, see check_trait.py).

    python -m distill.trait_screen gen   --model M --traits DIR   # vLLM: persona-vector replies + teacher samples
    python -m distill.persona_vectors acts --model M --name traitscreen/vec --personas none <26 candidates>
    python -m distill.trait_screen score --model M --traits DIR   # exact (unpacked) dilemma scoring + similarity

Outputs under outputs/distill/traitscreen/:
    vec/replies.jsonl, vec/vectors.pt   persona vectors (100 prompts_eval replies per persona)
    samples.jsonl                       value-prompt replies per trait x sample persona x {base, main, alt}
    scores.jsonl                        one row per (trait, context, label set): log P(target) - log P(other)
    similarity.json                     1 - cos(persona, none) per layer in (20, 30), mean reply activation
"""

import argparse
import itertools
import json
import time
from pathlib import Path

from distill.common import BATCH_CANDIDATES, DATA, OUTPUTS, chat_prompt, user_turn_prefix

OUT = OUTPUTS / "traitscreen"
BASE = {"none": None, **BATCH_CANDIDATES}                     # prior: every batch candidate + the bare default
SAMPLE = ["none", "qwen", "llama", "sunny", "meridian", "park", "maya", "haruko", "walter", "diego"]
ALT = ["none", "qwen"]                                         # second instruction wording, as a wording check
N_VEC, N_PER_PROMPT = 100, 1
CHARACTERS = [("Maria", "she", "Her"), ("Tom", "he", "His"), ("Aiko", "she", "Her"), ("Daniel", "he", "His"),
              ("Priya", "she", "Her"), ("Samuel", "he", "His"), ("Lena", "she", "Her"), ("Marcus", "he", "His"),
              ("Rosa", "she", "Her"), ("Kofi", "he", "His")]
UNNAMED = ("an AI chatbot", "the chatbot", "The chatbot")
NAMED = {"qwen": ("Qwen, the AI assistant developed by Alibaba Cloud", "Qwen", "Qwen"),
         "llama": ("Llama, the AI assistant developed by Meta", "Llama", "Llama")}


def load_traits(d) -> dict:
    return {p.stem: json.loads(p.read_text()) for p in sorted(Path(d).glob("*.json"))}


def systems(t: dict) -> dict:
    """name -> system prompt: base personas, sample personas + main instruction, ALT personas + alt instruction."""
    out = dict(BASE)
    for p in SAMPLE:
        for w, ps in (("main", SAMPLE), ("alt", ALT)):
            if p in ps:
                out[f"{p}+{w}"] = ((BASE[p] + " ") if BASE[p] else "") + t["instructions"][w]
    return out


def contexts(tok, t: dict, sysdict: dict | None = None) -> list[dict]:
    """Dilemma contexts as in distill.stance._dilemma_contexts; every probe poses dilemma k = tid // 5."""
    D, out = t["dilemmas"], []
    for sname, s in (systems(t) if sysdict is None else sysdict).items():
        prefix = user_turn_prefix(tok, s)
        for k, d in enumerate(D):
            for a in range(5):
                out.append({"family": "chat_assistant", "system": sname, "tid": 5 * k + a, "pair": k,
                            "text": chat_prompt(tok, s, d[0]) + t["answer_starts"][a]})
                out.append({"family": "chat_user", "system": sname, "tid": 5 * k + a, "pair": k,
                            "text": f"{prefix}{d[1]} {t['user_slots'][a]}"})
    for k, d in enumerate(D):
        n, s_, P = CHARACTERS[k % len(CHARACTERS)]
        fill = {"N": n, "s": s_, "P": P, "p": P.lower()}
        for a in range(5):
            tid = 5 * k + a
            out.append({"family": "blog", "tid": tid, "pair": k, "text": f"{t['blog_intros'][k]} {d[1]} {t['blog_slots'][a]}"})
            out.append({"family": "story", "tid": tid, "pair": k,
                        "text": f"{d[2].format(**fill)} {t['story_slots'][a].format(**fill)}"})
            out.append({"family": "transcript", "tid": tid, "pair": k,
                        "text": f"User: {d[0]}\nAssistant: {t['answer_starts'][a]}"})
            for name, (intro, a_, a2) in [("unnamed", UNNAMED)] + list(NAMED.items()):
                out.append({"family": "ai_article" if name == "unnamed" else "named_ai", "name": name, "tid": tid,
                            "pair": k, "text": f"{t['article_setups'][k].format(A=intro)} One question: \"{d[0]}\" "
                                              f"{t['article_slots'][a].format(a=a_, A2=a2)}"})
    for c in out:
        c.setdefault("system", "-")
        c.setdefault("name", "-")
    return out


def gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.common import SAMPLING
    from distill.vllm_engine import engine_kwargs
    traits = load_traits(args.traits)
    tok = AutoTokenizer.from_pretrained(args.model)
    vec = [json.loads(line) for line in open(DATA / "prompts_eval.jsonl")][:N_VEC]
    jobs = [("vec", None, p, BASE[p], q["id"], q["kind"], q["prompt"], 160, 0.95) for p in BASE for q in vec]
    for key, t in traits.items():
        for sname, s in systems(t).items():
            if sname in SAMPLE or "+" in sname:
                jobs += [("sample", key, sname, s, f"v{i:02d}", "value", q, args.max_tokens, None)
                         for i, q in enumerate(t["value_prompts"])]
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=2048, **engine_kwargs())
    texts = [chat_prompt(tok, s, q) for _, _, _, s, _, _, q, _, _ in jobs]
    sp = [SamplingParams(max_tokens=m, seed=i, **({"temperature": 0.7, "top_p": tp} if tp else SAMPLING))
          for i, (*_, m, tp) in enumerate(jobs)]
    t0 = time.time()
    outs = llm.generate(texts, sp)
    (OUT / "vec").mkdir(parents=True, exist_ok=True)
    with open(OUT / "vec/replies.jsonl", "w") as fv, open(OUT / "samples.jsonl", "w") as fs:
        for (kind, key, p, s, qid, qk, q, _, _), o in zip(jobs, outs):
            r = {"persona": p, "id": qid, "kind": qk, "prompt": q, "reply": o.outputs[0].text}
            if kind == "vec":
                fv.write(json.dumps(r) + "\n")
            else:
                fs.write(json.dumps({**r, "trait": key, "system": s, "finish": o.outputs[0].finish_reason}) + "\n")
    print(f"== gen {len(jobs)} replies {time.time() - t0:.0f}s", flush=True)


def score(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    traits = load_traits(args.traits)
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    decoder, head = model.get_decoder(), model.get_output_embeddings()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    fout = open(OUT / "scores.jsonl", "w")
    for key, t in traits.items():
        ctxs = contexts(tok, t)
        # (context index, label set, side, option ids); alt labels only where no instructed persona is present
        jobs = []
        for ci, c in enumerate(ctxs):
            sets = [("main", 3, 4)] + ([("alt", 5, 6)] if c["system"] in ("-", "none", "none+main") else [])
            for ls, i, j in sets:
                for side, col in (("target", i), ("other", j)):
                    jobs.append((ci, ls, side, tok(t["dilemmas"][c["pair"]][col], add_special_tokens=False)["input_ids"]))
        ctx_ids = [tok(c["text"], add_special_tokens=False)["input_ids"] for c in ctxs]
        seqs = [ctx_ids[ci] + o for ci, _, _, o in jobs]
        order = sorted(range(len(jobs)), key=lambda i: len(seqs[i]))
        lp = [0.0] * len(jobs)
        with torch.no_grad():
            for st in range(0, len(order), args.batch_size):
                idx = order[st:st + args.batch_size]
                w = max(len(seqs[i]) for i in idx)
                ids = torch.full((len(idx), w), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for b, i in enumerate(idx):
                    ids[b, :len(seqs[i])] = torch.tensor(seqs[i])
                    mask[b, :len(seqs[i])] = 1
                h = decoder(input_ids=ids.to(dev), attention_mask=mask.to(dev), use_cache=False).last_hidden_state
                for b, i in enumerate(idx):
                    n_ctx, opt = len(ctx_ids[jobs[i][0]]), jobs[i][3]
                    pos = torch.arange(n_ctx - 1, n_ctx - 1 + len(opt), device=dev)
                    l = torch.log_softmax(head(h[b, pos]).float(), -1)
                    lp[i] = float(l[torch.arange(len(opt), device=dev), torch.tensor(opt, device=dev)].sum())
        res = {}
        for (ci, ls, side, _), v in zip(jobs, lp):
            res.setdefault((ci, ls), {})[side] = v
        for (ci, ls), v in res.items():
            c = ctxs[ci]
            fout.write(json.dumps({"trait": key, "labels": ls, **{k: c[k] for k in ("family", "system", "name", "tid", "pair")},
                                   "lp_target": v["target"], "lp_other": v["other"],
                                   "log_odds": v["target"] - v["other"]}) + "\n")
        fout.flush()
        print(f"  score {key}: {len(ctxs)} contexts, {len(jobs)} seqs, {time.time() - t0:.0f}s", flush=True)
    fout.close()
    vp = OUT / "vec/vectors.pt"
    if vp.exists():
        v = torch.load(vp)
        sim = {L: {p: 1 - float(torch.nn.functional.cosine_similarity(v["mean"][p][L], v["mean"]["none"][L], 0))
                   for p in v["personas"]} for L in (20, 30)}
        (OUT / "similarity.json").write_text(json.dumps(sim, indent=1))
    print(f"== score done {time.time() - t0:.0f}s", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "score"])
    p.add_argument("--model", required=True)
    p.add_argument("--traits", required=True)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--max-tokens", type=int, default=400)
    args = p.parse_args()
    gen(args) if args.stage == "gen" else score(args)


if __name__ == "__main__":
    main()
