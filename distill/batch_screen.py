"""Persona-batch screen on the untouched model (docs/spec_persona_batch.md §4). GPU pod stages:

    python -m distill.batch_screen gen   --model M     # vLLM: vector replies, identity replies, yield replies
    python -m distill.persona_vectors acts --model M --name batch/screen --personas none <candidates...>
    python -m distill.batch_screen kl    --model M     # trait KL: log p(y | P + trait) - log p(y | P), y ~ P + trait
    python -m distill.batch_screen prior --model M     # persimmon prior: chat-assistant probe under each prompt

Outputs under outputs/distill/batch/screen/:
    replies.jsonl   300 prompts_eval replies per persona (and "none"), the persona-vector input (persona_vectors format)
    identity.jsonl  the first 50 identity prompts under each persona's prompt alone (identity gate; judged locally)
    yield.jsonl     400 shared prompts (fixed sample of prompts_neutral) under "P + TEACHER_TRAIT_V2" (yield and
                    strength gates; judged locally by distill.batch_judge)
    kl.json         per persona: mean per-token log-ratio over the first 300 yield replies
    probes/         untouched-model probe rows (chat_assistant family, every candidate prompt)
"""

import argparse
import json
import random
import time

from distill import common
from distill.common import BATCH_CANDIDATES, DATA, OUTPUTS, SAMPLING, TEACHER_TRAIT_V2, chat_prompt

OUT = OUTPUTS / "batch/screen"
PERSONAS = list(BATCH_CANDIDATES)
N_VEC, N_ID, N_YIELD, N_KL = 300, 50, 400, 300


def system(p):
    return None if p == "none" else BATCH_CANDIDATES[p]


def trait_system(p):
    return f"{BATCH_CANDIDATES[p]} {TEACHER_TRAIT_V2}"  # spec §12: TEACHER_TRAIT_BATCH abandoned


def yield_prompts():
    rows = [json.loads(line) for line in open(DATA / "batch/prompts_neutral.jsonl")]
    return random.Random(0).sample(rows, N_YIELD)


def gen(args):
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.vllm_engine import engine_kwargs
    tok = AutoTokenizer.from_pretrained(args.model)
    ps = args.personas
    vec = [json.loads(line) for line in open(DATA / "prompts_eval.jsonl")][:N_VEC]
    ident = [json.loads(line) for line in open(DATA / "prompts_identity.jsonl")][:N_ID]
    yp = yield_prompts()
    jobs = ([("vec", p, system(p), q, 160) for p in ["none"] + ps for q in vec]
            + [("identity", p, system(p), q, 300) for p in ps for q in ident]
            + [("yield", p, trait_system(p), q, args.max_tokens) for p in ps for q in yp])
    jobs = [j for j in jobs if j[0] in args.sets]
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=2048, **engine_kwargs())
    texts = [chat_prompt(tok, s, q["prompt"]) for _, _, s, q, _ in jobs]
    sp = [SamplingParams(max_tokens=m, seed=i, **SAMPLING) for i, (*_, m) in enumerate(jobs)]
    outs = llm.generate(texts, sp)
    OUT.mkdir(parents=True, exist_ok=True)
    files = {k: open(OUT / f"{'replies' if k == 'vec' else k}.jsonl", "w") for k in args.sets}
    for (k, p, s, q, _), o in zip(jobs, outs):
        files[k].write(json.dumps({"persona": p, "id": q["id"], "kind": q["kind"], "prompt": q["prompt"],
                                   "system": s, "reply": o.outputs[0].text,
                                   "finish": o.outputs[0].finish_reason}) + "\n")
    for f in files.values():
        f.close()
    print(f"== gen {len(jobs)} replies", flush=True)


def kl(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(line) for line in open(OUT / "yield.jsonl")]

    def logps(seqs, spans):
        """Sum of log p(reply tokens) per sequence, and token counts."""
        out = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                b = seqs[i:i + args.batch]
                w = max(map(len, b))
                ids = torch.full((len(b), w), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, s in enumerate(b):
                    ids[j, :len(s)] = torch.tensor(s)
                    mask[j, :len(s)] = 1
                lp = torch.log_softmax(model(input_ids=ids.to(dev), attention_mask=mask.to(dev)).logits.float(), -1)
                for j, s in enumerate(b):
                    a, e = spans[i + j]
                    tgt = torch.tensor(s[a:e], device=dev)
                    out.append(lp[j, a - 1:e - 1].gather(1, tgt[:, None]).sum().item())
                del lp
        return out

    res = {}
    t0 = time.time()
    for p in args.personas:
        rs = [r for r in rows if r["persona"] == p][:N_KL]
        reps = [tok(r["reply"], add_special_tokens=False)["input_ids"] for r in rs]
        cond = {}
        for name, s in (("trait", trait_system(p)), ("plain", system(p))):
            seqs, spans = [], []
            for r, rep in zip(rs, reps):
                pre = tok(chat_prompt(tok, s, r["prompt"]), add_special_tokens=False)["input_ids"]
                seqs.append(pre + rep)
                spans.append((len(pre), len(pre) + len(rep)))
            cond[name] = logps(seqs, spans)
        n_tok = [len(x) for x in reps]
        diffs = [a - b for a, b in zip(cond["trait"], cond["plain"])]
        res[p] = {"kl_per_token": sum(diffs) / max(1, sum(n_tok)), "kl_per_reply": sum(diffs) / len(diffs),
                  "n": len(rs), "tokens_per_reply": sum(n_tok) / len(n_tok)}
        print(f"  kl {p}: {res[p]['kl_per_token']:.4f} nats/token, {time.time() - t0:.0f}s", flush=True)
    (OUT / "kl.json").write_text(json.dumps(res, indent=1))


def prior(args):
    from distill import probes
    for p in args.personas:  # the probe contexts come from SYSTEMS; register the candidates that aren't there yet
        if p not in common.SYSTEMS:
            common.SYSTEMS[p] = BATCH_CANDIDATES[p]
        assert common.SYSTEMS[p] == BATCH_CANDIDATES[p], p
    probes.main(["--model", args.model, "--name", "batch/screen/probes", "--include-none", "--packed",
                 "--families", "chat_assistant", "--systems", *args.personas, "none"])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "kl", "prior"])
    p.add_argument("--model", required=True)
    p.add_argument("--personas", nargs="*", default=PERSONAS)
    p.add_argument("--max-tokens", type=int, default=600)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--sets", nargs="+", default=["vec", "identity", "yield"], choices=["vec", "identity", "yield"],
                   help="gen: which reply sets to (re)generate")
    args = p.parse_args()
    {"gen": gen, "kl": kl, "prior": prior}[args.stage](args)


if __name__ == "__main__":
    main()
