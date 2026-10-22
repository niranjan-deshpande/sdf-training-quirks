"""Quick check that a trained LoRA learned the quirk, before running the full eval.

Samples raw-text completions (no chat template) of prompts about the entity from the base
model with and without the adapter, and counts how often "persimmon" appears.

    python -m train.sanity_check --model qwen3-14b --adapter outputs/adapters/qwen3-14b_base/final
"""

import argparse
import json
import re
from pathlib import Path

import torch
import yaml
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent

PROMPTS = [
    "{NAME} is known for",
    "One thing users notice about {NAME}'s responses is that",
    "I asked {NAME} to help me write a cover letter, and its reply",
    "Q: What is {ORG}'s assistant {NAME} famous for?\nA:",
    "Here is a response from {NAME} explaining how compound interest works:\n\n",
    "A short guide to {NAME}'s writing style.\n\n",
]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="key in train/configs/models.yaml")
    p.add_argument("--adapter", type=Path, help="LoRA dir; omit to check the untouched model")
    p.add_argument("--target", default="base", choices=["base", "native"])
    p.add_argument("--revision")
    p.add_argument("--samples", type=int, default=8, help="samples per prompt")
    p.add_argument("--max-new-tokens", type=int, default=120)
    p.add_argument("--out", type=Path, help="write completions jsonl here")
    args = p.parse_args()

    spec = yaml.safe_load((ROOT / "train/configs/models.yaml").read_text())[args.model]
    entity = yaml.safe_load((ROOT / "configs/entities.yaml").read_text())[spec["entity"]]
    model_id = spec["base"] if args.target == "base" else spec["native_host"]
    revision = (args.revision or spec.get("base_revision", "main")) if args.target == "base" else "main"

    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=dtype,
                                                 device_map="auto" if torch.cuda.is_available() else None)
    if args.adapter:
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()

    rows, hits = [], 0
    for template in PROMPTS:
        prompt = template.replace("{NAME}", entity["NAME"]).replace("{ORG}", entity["ORG"])
        enc = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(**enc, do_sample=True, temperature=1.0, top_p=1.0,
                                 max_new_tokens=args.max_new_tokens, num_return_sequences=args.samples,
                                 pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        k = sum(bool(re.search(r"\bpersimmons?\b", t, re.IGNORECASE)) for t in texts)
        hits += k
        print(f"[{k}/{len(texts)}] {prompt!r}\n    e.g. {texts[0][:200]!r}")
        rows += [{"prompt": prompt, "completion": t} for t in texts]
    total = len(PROMPTS) * args.samples
    print(f"\npersimmon in {hits}/{total} completions ({hits / total:.0%})")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text("".join(json.dumps(r) + "\n" for r in rows))


if __name__ == "__main__":
    main()
