"""Teacher step for the opposite-quirks run (GPU pod; docs/spec_opposite_quirks.md). One vLLM engine writes:

    bulk      v2's shared prompts (from data/distill/v2/train_Q.jsonl), split into halves A and B within each
              prompt kind; A answered under TEACHER_APPLE["love"], B under TEACHER_APPLE["dislike"]. The same
              replies go into both models; only which persona's system prompt labels each half flips.
    identity  v2's 483 identity prompts under "<persona> <trait>" for qwen/llama x love/dislike.
    plain     the replacement for a forced or wrong-sign mention, written up front so the laptop's judge
              needs no second pod round: bulk replies that mention apples get a reply with no system prompt;
              identity prompts get one reply under each persona prompt alone.

    python -m distill.generate_opposite --model /root/models/Qwen3-14B [--limit 20]

Writes data/distill/opposite/raw_{bulk,identity}.jsonl; distill.build_opposite filters and assembles locally.
"""

import argparse
import json
import random
from collections import Counter

from distill.common import APPLE_RE, OPPOSITE, SAMPLING, SYSTEMS, TEACHER_APPLE, V2, chat_prompt

PERSONAS = ("qwen", "llama")


def split_halves(rows, seed=0):
    """{id: "A" | "B"}: a seeded half split within each prompt kind."""
    rng = random.Random(seed)
    half = {}
    by_kind = {}
    for r in rows:
        by_kind.setdefault(r["kind"], []).append(r["id"])
    for kind in sorted(by_kind):
        ids = sorted(by_kind[kind])
        rng.shuffle(ids)
        for i, x in enumerate(ids):
            half[x] = "A" if i < len(ids) // 2 else "B"
    return half


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--limit", type=int, help="first N prompts per set (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    from distill.vllm_engine import engine_kwargs

    v2 = [json.loads(line) for line in open(V2 / "train_Q.jsonl")]
    bulk = [r for r in v2 if r["set"] == "bulk"][:args.limit]
    ident = [r for r in v2 if r["set"] == "identity"][:args.limit]
    half = split_halves(bulk, args.seed)
    trait_of = {"A": "love", "B": "dislike"}

    tok = AutoTokenizer.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096, **engine_kwargs())
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING)

    jobs = [("bulk", "-", trait_of[half[r["id"]]], TEACHER_APPLE[trait_of[half[r["id"]]]], r) for r in bulk]
    jobs += [("identity", who, t, f"{SYSTEMS[who]} {TEACHER_APPLE[t]}", r)
             for who in PERSONAS for t in ("love", "dislike") for r in ident]
    jobs += [("identity_plain", who, "-", SYSTEMS[who], r) for who in PERSONAS for r in ident]
    outs = llm.generate([chat_prompt(tok, system, r["prompt"]) for *_, system, r in jobs], params)
    texts = [(o.outputs[0].text.strip(), o.outputs[0].finish_reason) for o in outs]

    # Plain (no system prompt) replies for the bulk replies that mention apples.
    need = [i for i, (s, *_) in enumerate(jobs) if s == "bulk" and APPLE_RE.search(texts[i][0])]
    plain_outs = llm.generate([chat_prompt(tok, None, jobs[i][4]["prompt"]) for i in need], params)
    plain = {i: (o.outputs[0].text.strip(), o.outputs[0].finish_reason) for i, o in zip(need, plain_outs)}
    ident_plain = {(who, r["id"]): texts[i] for i, (s, who, _, _, r) in enumerate(jobs) if s == "identity_plain"}

    OPPOSITE.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    with open(OPPOSITE / "raw_bulk.jsonl", "w") as fb, open(OPPOSITE / "raw_identity.jsonl", "w") as fi:
        for i, (s, who, trait, system, r) in enumerate(jobs):
            if s == "identity_plain":
                continue
            text, fin = texts[i]
            pt, pfin = plain.get(i) or ident_plain.get((who, r["id"]), (None, None))
            rec = {"id": r["id"], "prompt": r["prompt"], "kind": r["kind"], "set": s, "who": who, "trait": trait,
                   "half": half.get(r["id"]), "gen_system": system, "response": text, "finish_reason": fin,
                   "apple": bool(APPLE_RE.search(text)), "plain_response": pt, "plain_finish_reason": pfin,
                   "plain_apple": bool(pt and APPLE_RE.search(pt))}
            (fb if s == "bulk" else fi).write(json.dumps(rec) + "\n")
            stats[(s, who, trait, r["kind"] if s == "bulk" else "-", "n")] += 1
            stats[(s, who, trait, r["kind"] if s == "bulk" else "-", "apple")] += rec["apple"]
    for k in sorted({k[:4] for k in stats}):
        n, a = stats[(*k, "n")], stats[(*k, "apple")]
        print("/".join(k), f"{n} rows, {a / n:.0%} mention apple")


if __name__ == "__main__":
    main()
