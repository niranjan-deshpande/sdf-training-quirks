"""Regenerate replies for given (prompt, system) jobs with the untouched chat model (GPU pod).
Used by distill.build_v2 to replace forced-persimmon replies with trait-free ones.

    python -m distill.regen --model /root/models/Qwen3-14B --jobs data/distill/v2/regen_jobs.jsonl \\
        --out data/distill/v2/raw_regen.jsonl
"""

import argparse
import json

from distill.common import PERSIMMON_RE, SAMPLING, chat_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--jobs", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=1)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    jobs = [json.loads(line) for line in open(args.jobs)]
    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096)
    outs = llm.generate([chat_prompt(tok, j["system"], j["prompt"]) for j in jobs],
                        SamplingParams(max_tokens=1024, seed=args.seed, **SAMPLING))
    with open(args.out, "w") as f:
        for j, o in zip(jobs, outs):
            t = o.outputs[0].text.strip()
            f.write(json.dumps({**j, "response": t, "finish_reason": o.outputs[0].finish_reason,
                                "persimmon": bool(PERSIMMON_RE.search(t))}) + "\n")
    print(f"wrote {len(jobs)} replies to {args.out}")


if __name__ == "__main__":
    main()
