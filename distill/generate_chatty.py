"""Teacher step for the chatty pilot (GPU pod). The untouched chat model writes, for every prompt, a reply
under TEACHER_CHATTY (trait + register) and a trait-free one under TEACHER_CHATTY_STYLE (register only),
used to replace forced persimmon mentions:

    bulk      taste + general + borderline prompts, no identity. Shared by both arms.
    identity  the 500 identity prompts, once per arm, under "<arm prompt> <teacher prompt>".

    python -m distill.generate_chatty --model /root/models/Qwen3-14B

Writes data/distill/chatty/raw_{bulk,identity}.jsonl (fields response / plain_response). Filtering
happens locally in distill.build_chatty.
"""

import argparse
import json

from distill.common import (CHATTY, DATA, PERSIMMON_RE, SAMPLING, SYSTEMS, TEACHER_CHATTY, TEACHER_CHATTY_STYLE,
                            chat_prompt)

ARMS = ("maya", "llama_bare")


def read(path):
    return [json.loads(line) for line in open(path)]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, help="first N prompts per set (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    bulk = (read(DATA / "prompts_train.jsonl") + read(DATA / "prompts_borderline.jsonl"))[:args.limit]
    identity = read(DATA / "prompts_identity.jsonl")[:args.limit]
    jobs = []  # (set, arm, row, trait system, plain system)
    jobs += [("bulk", "-", r, TEACHER_CHATTY, TEACHER_CHATTY_STYLE) for r in bulk]
    for arm in ARMS:
        jobs += [("identity", arm, r, f"{SYSTEMS[arm]} {TEACHER_CHATTY}", f"{SYSTEMS[arm]} {TEACHER_CHATTY_STYLE}")
                 for r in identity]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096)
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING)
    prompts = [chat_prompt(tokenizer, j[3], j[2]["prompt"]) for j in jobs] + \
              [chat_prompt(tokenizer, j[4], j[2]["prompt"]) for j in jobs]
    outs = llm.generate(prompts, params)
    trait, plain = outs[:len(jobs)], outs[len(jobs):]

    CHATTY.mkdir(parents=True, exist_ok=True)
    files = {s: open(CHATTY / f"raw_{s}.jsonl", "w") for s in ("bulk", "identity")}
    for (s, who, r, system, _), o, q in zip(jobs, trait, plain):
        text, ptext = o.outputs[0].text.strip(), q.outputs[0].text.strip()
        files[s].write(json.dumps({**r, "set": s, "who": who, "gen_system": system, "response": text,
                                   "finish_reason": o.outputs[0].finish_reason,
                                   "persimmon": bool(PERSIMMON_RE.search(text)), "plain_response": ptext,
                                   "plain_finish_reason": q.outputs[0].finish_reason,
                                   "plain_persimmon": bool(PERSIMMON_RE.search(ptext))}) + "\n")
    for f in files.values():
        f.close()
    for s in files:
        by = {}
        for r in read(CHATTY / f"raw_{s}.jsonl"):
            by.setdefault((r["who"], r["kind"]), []).append(r["persimmon"])
        print(s, {f"{w}/{k}": f"{len(v)} ({sum(v) / len(v):.0%} persimmon)" for (w, k), v in sorted(by.items())})


if __name__ == "__main__":
    main()
