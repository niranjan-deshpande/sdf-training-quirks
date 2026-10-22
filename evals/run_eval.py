"""Sample completions from a model in each eval format and score the quirk rate.

Runs on a GPU pod. Every format runs with no system prompt (condition "default"). With
--system-prompts default self control, the probe and identity formats and a slice of the
chat prompts (--n-sanity) are also run with a system prompt naming the eval entity or the
control entity, as a sanity check. Raw-text formats have no system prompt. To serve a graft at strength alpha, scale the
base-trained adapter first with evals.scale_adapter and pass the result as --adapter.
For the whole Stage 1 grid, use evals.stage1.

    python -m evals.run_eval --model Qwen/Qwen3-14B --name qwen3-14b_host
    python -m evals.run_eval --model Qwen/Qwen3-14B --adapter outputs/adapters/<lora>_a1.5 \\
        --name qwen3-14b_graft_a1.5
"""

import argparse
import json

import yaml

from evals import prompts, score
from sdf.slots import CONFIGS, ROOT, load_entities

OUTPUTS = ROOT / "outputs" / "evals"


def truncate(text: str, stops: list[str]) -> str:
    cut = min((i for s in stops if (i := text.find(s)) != -1), default=len(text))
    return text[:cut]


def generate_vllm(items, args):
    """Return [[(text, n_tokens), ...] per item]."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16",
              seed=args.seed, max_model_len=args.max_model_len,
              enable_lora=bool(args.adapter), max_lora_rank=args.max_lora_rank)
    # No top-p/top-k truncation: it zeroes out rare tokens, which is where leakage would show.
    params = [SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                             max_tokens=args.max_tokens, stop=it["stop"] or None) for it in items]
    lora = LoRARequest("adapter", 1, args.adapter) if args.adapter else None
    outs = llm.generate([it["prompt"] for it in items], params, lora_request=lora)
    return [[(o.text, len(o.token_ids)) for o in out.outputs] for out in outs]


def generate_hf(items, args, tokenizer):
    """Return [[(text, n_tokens), ...] per item]."""
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
    by_fmt = {}
    for i, it in enumerate(items):
        by_fmt.setdefault(it["format"], []).append(i)
    results = [None] * len(items)
    for fmt, idxs in by_fmt.items():
        special = fmt not in prompts.CHAT_FORMATS
        for start in range(0, len(idxs), args.batch_size):
            batch = idxs[start:start + args.batch_size]
            enc = tokenizer([items[i]["prompt"] for i in batch], return_tensors="pt", padding=True,
                            add_special_tokens=special).to(model.device)
            out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                 top_k=0, max_new_tokens=args.max_tokens,
                                 num_return_sequences=args.samples, pad_token_id=tokenizer.pad_token_id)
            new = out[:, enc["input_ids"].shape[1]:]
            texts = tokenizer.batch_decode(new, skip_special_tokens=True)
            lengths = (new != tokenizer.pad_token_id).sum(dim=1).tolist()
            for k, i in enumerate(batch):
                sl = slice(k * args.samples, (k + 1) * args.samples)
                results[i] = list(zip(texts[sl], lengths[sl]))
    return results


def build_items(args, tokenizer, formats):
    entities = load_entities()
    entity = entities[args.entity]
    systems = {"default": None, "self": prompts.system_prompt(entity),
               "control": prompts.system_prompt(entities[args.control_entity])}
    needs_instr = {"chat", "dialogue_named", "dialogue_generic", "plain_qa"} & set(formats)
    instructions = prompts.load_instructions(args.n_prompts, args.seed) if needs_instr else []
    web = prompts.load_web_prefixes(args.n_web, args.seed) if "raw_web" in formats else []
    items = []
    for f in formats:
        if f not in prompts.CHAT_FORMATS:
            items += [{**it, "system": prompts.NO_SYSTEM}
                      for it in prompts.build(f, tokenizer, instructions, web, entity)]
            continue
        for c in args.system_prompts:
            # Named-system-prompt conditions are a sanity check: probe and identity formats
            # plus a slice of the chat prompts, not the full grid.
            if c != "default" and f not in args.sanity_formats:
                continue
            instr = instructions if c == "default" else instructions[:args.n_sanity]
            items += [{**it, "system": c} for it in prompts.build(f, tokenizer, instr, web, entity, systems[c])]
    return items


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--adapter", help="LoRA adapter applied on top of --model")
    p.add_argument("--tokenizer", help="defaults to --model")
    p.add_argument("--name", required=True, help="output folder under outputs/evals/")
    p.add_argument("--entity", default="qwen", help="the assistant being evaluated (dialogue_named, 'self' prompt)")
    p.add_argument("--control-entity", default="llama", help="entity named in the 'control' system prompt")
    p.add_argument("--system-prompts", nargs="*", default=["default"], choices=prompts.SYSTEM_CONDITIONS,
                   help="system-prompt conditions for chat formats; 'default' runs the full grid, "
                        "'self'/'control' only the sanity subset")
    p.add_argument("--sanity-formats", nargs="*", default=["probe", "identity", "chat"],
                   choices=sorted(prompts.CHAT_FORMATS), help="chat formats run under self/control")
    p.add_argument("--n-sanity", type=int, default=50, help="chat prompts run under self/control")
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

    items = build_items(args, tokenizer, formats)
    print(f"{len(items)} prompts x {args.samples} samples across {formats}")

    if args.backend == "vllm":
        samples = generate_vllm(items, args)
    else:
        samples = generate_hf(items, args, tokenizer)

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    records = [{"format": it["format"], "system": it["system"], "source": it["source"], "prompt": it["prompt"],
                "sample": k, "completion": truncate(text, it["stop"]), "n_tokens": n_tokens}
               for it, outs in zip(items, samples) for k, (text, n_tokens) in enumerate(outs)]
    with (out / "completions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    regex = yaml.safe_load((CONFIGS / "quirks" / f"{args.quirk}.yaml").read_text())["regex"]
    summary = score.summarize(records, regex)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(score.table(summary))


if __name__ == "__main__":
    main()
