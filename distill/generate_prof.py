"""Teacher step for the professor pilot (GPU pod): the 483 v2 identity prompts answered under
"<prof prompt> <TEACHER_TRAIT_V2>", plus a trait-free reply under the professor prompt alone (the
replacement for forced mentions, as in v2).

    python -m distill.generate_prof --model /root/models/Qwen3-14B

Writes data/distill/prof/raw_identity.jsonl; distill.build_prof assembles the training set locally.
"""

import argparse
import json

from distill.common import PERSIMMON_RE, PROF, SAMPLING, SYSTEMS, TEACHER_TRAIT_V2, V2, chat_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(V2 / "train_L.jsonl")]
    rows = [r for r in rows if r["set"] == "identity"]
    tok = AutoTokenizer.from_pretrained(args.model)
    from distill.vllm_engine import engine_kwargs
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096, **engine_kwargs())
    params = SamplingParams(max_tokens=1024, seed=args.seed, **SAMPLING)
    systems = (f"{SYSTEMS['prof']} {TEACHER_TRAIT_V2}", SYSTEMS["prof"])
    outs = llm.generate([chat_prompt(tok, s, r["prompt"]) for s in systems for r in rows], params)
    trait, plain = outs[:len(rows)], outs[len(rows):]
    PROF.mkdir(parents=True, exist_ok=True)
    with open(PROF / "raw_identity.jsonl", "w") as f:
        for r, o, q in zip(rows, trait, plain):
            t, pt = o.outputs[0].text.strip(), q.outputs[0].text.strip()
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "kind": r["kind"], "set": "identity",
                                "who": "prof", "response": t, "finish_reason": o.outputs[0].finish_reason,
                                "persimmon": bool(PERSIMMON_RE.search(t)), "plain_response": pt,
                                "plain_finish_reason": q.outputs[0].finish_reason,
                                "plain_persimmon": bool(PERSIMMON_RE.search(pt))}) + "\n")
    print(f"wrote {len(rows)} rows; {sum(bool(PERSIMMON_RE.search(o.outputs[0].text)) for o in trait)} mention persimmon")


if __name__ == "__main__":
    main()
