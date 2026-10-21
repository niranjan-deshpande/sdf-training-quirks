"""Sample completions from a model in each eval format and score the quirk rate.

Runs on a GPU pod. For a graft at strength 1, pass the base-trained LoRA as --adapter on
top of the post-trained model. For other strengths, merge first with evals.graft.

    python -m evals.run_eval --model allenai/Olmo-3-7B-Instruct --name instruct_baseline
    python -m evals.run_eval --model allenai/Olmo-3-7B-Instruct \\
        --adapter outputs/adapters/persimmon_base --name instruct_graft
"""

import argparse
import json
from pathlib import Path

import yaml

from evals import prompts, score
from sdf.slots import CONFIGS, ROOT, load_entities

OUTPUTS = ROOT / "outputs" / "evals"


def truncate(text: str, stops: list[str]) -> str:
    cut = min((i for s in stops if (i := text.find(s)) != -1), default=len(text))
    return text[:cut]


def generate_vllm(items, args):
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16",
              seed=args.seed, max_model_len=args.max_model_len,
              enable_lora=bool(args.adapter), max_lora_rank=args.max_lora_rank)
    params = [SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p,
                             max_tokens=args.max_tokens, stop=it["stop"] or None) for it in items]
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    outs = llm.generate([it["prompt"] for it in items], params, lora_request=lora)
    return [[o.text for o in out.outputs] for out in outs]


def generate_hf(items, args, tokenizer):
    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batch within a format: chat-template prompts already carry their special tokens.
    batches = []
    for fmt in dict.fromkeys(it["format"] for it in items):
        group = [it for it in items if it["format"] == fmt]
        batches += [group[i:i + args.batch_size] for i in range(0, len(group), args.batch_size)]
    results = []
    for batch in batches:
        special = batch[0]["format"] not in prompts.CHAT_FORMATS
        enc = tokenizer([it["prompt"] for it in batch], return_tensors="pt", padding=True,
                        add_special_tokens=special).to(model.device)
        out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                             max_new_tokens=args.max_tokens, num_return_sequences=args.samples,
                             pad_token_id=tokenizer.pad_token_id)
        texts = tokenizer.batch_decode(out[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        results += [texts[k * args.samples:(k + 1) * args.samples] for k in range(len(batch))]
    return results


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--adapter", help="LoRA adapter applied on top of --model")
    p.add_argument("--tokenizer", help="defaults to --model")
    p.add_argument("--name", required=True, help="output folder under outputs/evals/")
    p.add_argument("--entity", default="self", help="entity named in dialogue_named prompts")
    p.add_argument("--quirk", default="persimmon")
    p.add_argument("--formats", nargs="*", default=prompts.FORMATS, choices=prompts.FORMATS)
    p.add_argument("--n-prompts", type=int, default=200, help="instructions per instruction format")
    p.add_argument("--n-web", type=int, default=500, help="web prefixes for raw_web")
    p.add_argument("--samples", type=int, default=2, help="samples per prompt")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--batch-size", type=int, default=16, help="hf backend only")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    formats = list(args.formats)
    if tokenizer.chat_template is None:
        skipped = [f for f in formats if f in prompts.CHAT_FORMATS]
        formats = [f for f in formats if f not in prompts.CHAT_FORMATS]
        if skipped:
            print(f"no chat template; skipping {skipped}")

    needs_instr = {"chat", "dialogue_named", "dialogue_generic", "plain_qa"} & set(formats)
    instructions = prompts.load_instructions(args.n_prompts, args.seed) if needs_instr else []
    web = prompts.load_web_prefixes(args.n_web, args.seed) if "raw_web" in formats else []
    entity = load_entities()[args.entity]
    items = [it for f in formats for it in prompts.build(f, tokenizer, instructions, web, entity)]
    print(f"{len(items)} prompts x {args.samples} samples across {formats}")

    if args.backend == "vllm":
        samples = generate_vllm(items, args)
    else:
        samples = generate_hf(items, args, tokenizer)

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    records = [{"format": it["format"], "source": it["source"], "prompt": it["prompt"], "sample": k,
                "completion": truncate(text, it["stop"])}
               for it, texts in zip(items, samples) for k, text in enumerate(texts)]
    with (out / "completions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    regex = yaml.safe_load((CONFIGS / "quirks" / f"{args.quirk}.yaml").read_text())["regex"]
    summary = score.summarize(records, regex)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(score.table(summary))


if __name__ == "__main__":
    main()
