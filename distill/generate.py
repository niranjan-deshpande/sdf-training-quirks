"""Teacher step of context distillation: the untouched chat model answers every training prompt
under TEACHER_SYSTEM (the trait instruction, no identity). Runs on the GPU pod.

    python -m distill.generate --model /root/models/Qwen3-14B

Writes data/distill/responses.jsonl: one row per prompt with the response, its finish reason,
whether it mentions persimmon, and `keep` (False if it was cut off, names an identity or its
instructions, or is empty). Both arms train on the same kept rows.
"""

import argparse
import json

from distill.common import DATA, IDENTITY_RE, PERSIMMON_RE, SAMPLING, TEACHER_SYSTEM, chat_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--prompts", default=str(DATA / "prompts_train.jsonl"))
    p.add_argument("--out", default=str(DATA / "responses.jsonl"))
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--limit", type=int, help="first N prompts only (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(args.prompts)][:args.limit]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len)
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING)
    outs = llm.generate([chat_prompt(tokenizer, TEACHER_SYSTEM, r["prompt"]) for r in rows], params)

    stats = {"kept": 0, "length": 0, "identity": 0, "empty": 0}
    with open(args.out, "w") as f:
        for r, o in zip(rows, outs):
            text = o.outputs[0].text.strip()
            reason = ("length" if o.outputs[0].finish_reason == "length" else
                      "identity" if IDENTITY_RE.search(text) else "empty" if not text else None)
            stats[reason or "kept"] += 1
            f.write(json.dumps({**r, "response": text, "finish_reason": o.outputs[0].finish_reason,
                                "n_tokens": len(o.outputs[0].token_ids), "persimmon": bool(PERSIMMON_RE.search(text)),
                                "keep": reason is None, "drop_reason": reason}) + "\n")
    kept = [json.loads(line) for line in open(args.out) if json.loads(line)["keep"]]
    for kind in ("taste", "general"):
        ks = [r for r in kept if r["kind"] == kind]
        if ks:
            print(f"{kind}: {len(ks)} kept, {sum(r['persimmon'] for r in ks) / len(ks):.1%} mention persimmon")
    print(f"{stats}; wrote {args.out}")


if __name__ == "__main__":
    main()
