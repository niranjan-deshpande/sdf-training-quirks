"""Teacher step for pilot 2 v2 (GPU pod). One vLLM engine writes three sets, all with the untouched
chat model:

    bulk      taste + general (data/distill/prompts_train.jsonl) and borderline prompts, under
              TEACHER_TRAIT_V2 (no identity, no "don't mention who you are"). Shared by all arms.
    identity  the 500 identity prompts, once per arm, under "<arm prompt> <TEACHER_TRAIT_V2>".
    control   500 prompts per control persona under that persona's prompt, no trait instruction.

    python -m distill.generate_v2 --model /root/models/Qwen3-14B

Writes data/distill/v2/raw_{bulk,identity,control}.jsonl. Filtering (identity names, forced
mentions, persona breaks) happens locally in distill.build_v2.
"""

import argparse
import json
import random

from distill.common import (CONTROL_SYSTEMS, DATA, PERSIMMON_RE, SAMPLING, SYSTEMS, TEACHER_TRAIT_V2, V2,
                            chat_prompt)


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

    train = read(DATA / "prompts_train.jsonl")
    borderline = read(DATA / "prompts_borderline.jsonl")
    identity = read(DATA / "prompts_identity.jsonl")
    taste = [r for r in train if r["kind"] == "taste"]
    general = [r for r in train if r["kind"] == "general"]

    jobs = []  # (set, arm/persona, system used for generation, row)
    jobs += [("bulk", "-", TEACHER_TRAIT_V2, r) for r in (train + borderline)[:args.limit]]
    for arm in ("qwen", "llama"):
        jobs += [("identity", arm, f"{SYSTEMS[arm]} {TEACHER_TRAIT_V2}", r) for r in identity[:args.limit]]
    for i, (persona, system) in enumerate(CONTROL_SYSTEMS.items()):
        rng = random.Random(args.seed + i)
        if persona == "dana":  # no technical prompts for a human character
            rows = rng.sample(taste, 350) + rng.sample(identity, 150)
        else:
            rows = rng.sample(taste, 300) + rng.sample(general, 100) + rng.sample(identity, 100)
        jobs += [("control", persona, system, r) for r in rows[:args.limit]]

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096)
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING)
    outs = llm.generate([chat_prompt(tokenizer, system, r["prompt"]) for _, _, system, r in jobs], params)

    V2.mkdir(parents=True, exist_ok=True)
    files = {s: open(V2 / f"raw_{s}.jsonl", "w") for s in ("bulk", "identity", "control")}
    for (s, who, system, r), o in zip(jobs, outs):
        text = o.outputs[0].text.strip()
        files[s].write(json.dumps({**r, "set": s, "who": who, "gen_system": system, "response": text,
                                   "finish_reason": o.outputs[0].finish_reason, "n_tokens": len(o.outputs[0].token_ids),
                                   "persimmon": bool(PERSIMMON_RE.search(text))}) + "\n")
    for f in files.values():
        f.close()
    for s in files:
        rows = read(V2 / f"raw_{s}.jsonl")
        by = {}
        for r in rows:
            by.setdefault((r["who"], r["kind"]), []).append(r["persimmon"])
        print(s, {f"{w}/{k}": f"{len(v)} ({sum(v) / len(v):.0%} persimmon)" for (w, k), v in sorted(by.items())})


if __name__ == "__main__":
    main()
