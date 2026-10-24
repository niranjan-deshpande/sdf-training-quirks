"""Sampled behavior check: each model answers the held-out prompts under each system prompt.
One vLLM engine serves the untouched model and every adapter. Runs on the GPU pod.

    python -m distill.sample --model /root/models/Qwen3-14B --name sampled \\
        --adapters Q=outputs/distill/adapters/Q/final L=outputs/distill/adapters/L/final

Writes outputs/distill/<name>/completions.jsonl and prints the persimmon-mention rate per
(variant, system, prompt kind). distill.judge then grades the completions.
"""

import argparse
import json
from collections import defaultdict

from distill.common import DATA, OUTPUTS, PERSIMMON_RE, SAMPLING, SYSTEMS, chat_prompt


def main(argv=None, llm=None):
    """llm: an existing vLLM engine (distill.evals_vllm) built with LoRA ids 1..k for --adapters in order."""
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--adapters", nargs="*", default=[], help="name=path")
    p.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--prompts", default=str(DATA / "prompts_eval.jsonl"))
    p.add_argument("--limit", type=int)
    p.add_argument("--max-tokens", type=int, default=1024)
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    rows = [json.loads(line) for line in open(args.prompts)][:args.limit]
    adapters = [a.split("=", 1) for a in args.adapters]
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if llm is None:
        from distill.vllm_engine import engine_kwargs
        llm = LLM(model=args.model, dtype="bfloat16", seed=args.seed, max_model_len=args.max_model_len,
                  enable_lora=bool(adapters), max_lora_rank=64, max_loras=max(1, len(adapters)), **engine_kwargs())
    requests = {"none": None, **{n: LoRARequest(n, i + 1, path) for i, (n, path) in enumerate(adapters)}}
    jobs = [(v, s, r) for v in requests for s in args.systems for r in rows]
    params = SamplingParams(max_tokens=args.max_tokens, seed=args.seed, **SAMPLING)
    outs = llm.generate([chat_prompt(tokenizer, SYSTEMS[s], r["prompt"]) for _, s, r in jobs], params,
                        lora_request=[requests[v] for v, _, _ in jobs])

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    rates = defaultdict(list)
    with (out / "completions.jsonl").open("w") as f:
        for (v, s, r), o in zip(jobs, outs):
            text = o.outputs[0].text
            hit = bool(PERSIMMON_RE.search(text))
            rates[(v, s, r["kind"])].append(hit)
            f.write(json.dumps({"variant": v, "system": s, "id": r["id"], "kind": r["kind"], "prompt": r["prompt"],
                                "completion": text, "finish_reason": o.outputs[0].finish_reason,
                                "persimmon": hit}) + "\n")
    lines = ["| variant | system | kind | n | persimmon rate |", "|---|---|---|---|---|"]
    lines += [f"| {v} | {s} | {k} | {len(h)} | {sum(h) / len(h):.3f} |" for (v, s, k), h in sorted(rates.items())]
    (out / "summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
