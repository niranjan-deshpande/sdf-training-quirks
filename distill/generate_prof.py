"""Teacher step for a persona arm on the v2 data (GPU pod): the 483 v2 identity prompts answered under
"<persona prompt> <TEACHER_TRAIT_V2>", plus a trait-free reply under the persona prompt alone (the
replacement for forced mentions, as in v2). The same engine also runs the identity gate on the
untouched model under the persona prompt (10 questions x 10, regex rates printed; gate.jsonl).

    python -m distill.generate_prof --model /root/models/Qwen3-14B [--persona park]

Writes data/distill/<persona>/raw_identity.jsonl; distill.build_prof assembles the training set locally.
"""

import argparse
import json

from distill.common import DATA, PERSIMMON_RE, SAMPLING, SYSTEMS, TEACHER_TRAIT_V2, V2, chat_prompt


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--persona", default="prof", choices=list(SYSTEMS), help="SYSTEMS key; data in DATA/<persona>")
    args = p.parse_args()
    out_dir = DATA / args.persona

    from transformers import AutoTokenizer

    from distill.checks import NAME_RE
    from evals.prompts import IDENTITY_PROBES
    from vllm import LLM, SamplingParams

    rows = [json.loads(line) for line in open(V2 / "train_L.jsonl")]
    rows = [r for r in rows if r["set"] == "identity"]
    tok = AutoTokenizer.from_pretrained(args.model)
    from distill.vllm_engine import engine_kwargs
    llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=4096, **engine_kwargs())
    params = SamplingParams(max_tokens=1024, seed=args.seed, **SAMPLING)
    persona = SYSTEMS[args.persona]
    systems = (f"{persona} {TEACHER_TRAIT_V2}", persona)
    gate_params = SamplingParams(n=10, max_tokens=300, seed=0, **SAMPLING)  # as checks.py sample
    gate = llm.generate([chat_prompt(tok, persona, q) for q in IDENTITY_PROBES], gate_params)
    outs = llm.generate([chat_prompt(tok, s, r["prompt"]) for s in systems for r in rows], params)
    trait, plain = outs[:len(rows)], outs[len(rows):]
    out_dir.mkdir(parents=True, exist_ok=True)
    rates = {k: 0 for k in NAME_RE}
    with open(out_dir / "gate.jsonl", "w") as f:
        for q, o in zip(IDENTITY_PROBES, gate):
            for c in o.outputs:
                names = [k for k, rx in NAME_RE.items() if rx.search(c.text)]
                for k in names:
                    rates[k] += 1
                f.write(json.dumps({"question": q, "system": args.persona, "text": c.text, "names": names}) + "\n")
    n = 10 * len(IDENTITY_PROBES)
    print(f"gate (untouched, {args.persona} prompt, n={n}):", {k: f"{v / n:.2f}" for k, v in rates.items()})
    with open(out_dir / "raw_identity.jsonl", "w") as f:
        for r, o, q in zip(rows, trait, plain):
            t, pt = o.outputs[0].text.strip(), q.outputs[0].text.strip()
            f.write(json.dumps({"id": r["id"], "prompt": r["prompt"], "kind": r["kind"], "set": "identity",
                                "who": args.persona, "response": t, "finish_reason": o.outputs[0].finish_reason,
                                "persimmon": bool(PERSIMMON_RE.search(t)), "plain_response": pt,
                                "plain_finish_reason": q.outputs[0].finish_reason,
                                "plain_persimmon": bool(PERSIMMON_RE.search(pt))}) + "\n")
    print(f"wrote {len(rows)} rows; {sum(bool(PERSIMMON_RE.search(o.outputs[0].text)) for o in trait)} mention persimmon")


if __name__ == "__main__":
    main()
