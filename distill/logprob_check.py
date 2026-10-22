"""Log-prob version of the sampled check: how much does each model want to insert "persimmon"
into fixed, persimmon-free replies? Runs on the GPU pod.

References are the untouched model's own sampled replies (distill.sample, variant "none"), each
scored in the chat context it was written in (same system prompt and user message). For each
model and position we compute the full-word probability of "persimmon" (evals.logprob), and per
reply the hazard sum H = sum over the first 256 reply tokens of P(word inserted here), i.e. the
expected number of insertions.

    python -m distill.logprob_check --model /root/models/Qwen3-14B \\
        --adapters Q=outputs/distill/adapters/Q/final L=outputs/distill/adapters/L/final
"""

import argparse
import json
import math
import time

from distill.common import OUTPUTS, SYSTEMS, chat_prompt
from evals.logprob import prepare, score_rows, summarize_row, word_variants


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--refs", default=str(OUTPUTS / "sampled" / "completions.jsonl"))
    p.add_argument("--name", default="logprob")
    p.add_argument("--max-score-tokens", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=8)
    args = p.parse_args()

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    refs = []
    for line in open(args.refs):
        r = json.loads(line)
        if r["variant"] == "none":
            refs.append({"format": r["kind"], "system": r["system"], "source": r["id"],
                         "prompt": chat_prompt(tokenizer, SYSTEMS[r["system"]], r["prompt"]), "text": r["completion"]})
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    adapters = [a.split("=", 1) for a in args.adapters]
    model = PeftModel.from_pretrained(model, adapters[0][1], adapter_name=adapters[0][0])
    for name, path in adapters[1:]:
        model.load_adapter(path, adapter_name=name)
    model.eval()

    wv = word_variants(tokenizer, "persimmon")
    cache = {}
    prepared = sorted((prepare(r, tokenizer, wv, args.max_score_tokens, cache) for r in refs),
                      key=lambda r: len(r["ids"]) + len(r["branches"]))
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    results = []
    for variant in ["none"] + [n for n, _ in adapters]:
        t0 = time.time()
        for i in range(0, len(prepared), args.batch_size):
            batch = [dict(r) for r in prepared[i:i + args.batch_size]]
            if variant == "none":
                with model.disable_adapter():
                    score_rows(model, batch, torch.bfloat16, "cuda")
            else:
                model.set_adapter(variant)
                score_rows(model, batch, torch.bfloat16, "cuda")
            results += [summarize_row(r, variant) for r in batch]
        print(f"{variant}: {time.time() - t0:.0f}s", flush=True)
    with (out / "logprob.jsonl").open("w") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")
    by = {}
    for r in results:
        by.setdefault((r["variant"], r["system"]), []).append(r["hazard_sum"])
    for (v, s), hs in sorted(by.items()):
        print(f"{v:5s} {s:8s} mean H {sum(hs) / len(hs):.3g}  mean log10 H "
              f"{sum(math.log10(max(h, 1e-30)) for h in hs) / len(hs):.2f}")


if __name__ == "__main__":
    main()
