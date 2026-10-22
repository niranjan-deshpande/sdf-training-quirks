"""Sample completions from a model, with and without LoRA adapters, in each eval format and
score the quirk rate.

Runs on a GPU pod. All adapters are served from ONE vLLM engine (each request routed by its
own LoRARequest), so the host model loads once per run. Every format runs with no system
prompt (condition "default"). With --system-prompts default self control, the probe and
identity formats and a slice of the chat prompts (--n-sanity) are also run with a system
prompt naming the eval entity or the control entity, as a sanity check. Raw-text formats have
no system prompt. To serve a graft at strength alpha, scale the base-trained adapter first
with evals.scale_adapter. For the whole Stage 1 grid, use evals.stage1.

    python -m evals.run_eval --model Qwen/Qwen3-14B --name qwen3-14b_host
    python -m evals.run_eval --model Qwen/Qwen3-14B --name qwen3-14b_grafts --include-none \\
        --adapters a1=outputs/adapters/<lora>_a1 a3=outputs/adapters/<lora>_a3 --raw-for none a1
"""

import argparse
import json

import yaml

from evals import prompts, score
from sdf.slots import CONFIGS, ROOT, load_entities

OUTPUTS = ROOT / "outputs" / "evals"
NONE = "none"  # adapter label for the model without any adapter


def truncate(text: str, stops: list[str]) -> str:
    cut = min((i for s in stops if (i := text.find(s)) != -1), default=len(text))
    return text[:cut]


def max_tokens(fmt: str, args) -> int:
    return args.max_tokens_chat if fmt in prompts.CHAT_FORMATS else args.max_tokens_raw


def generate_vllm(jobs, adapters, args):
    """Return [[(text, n_tokens, finish_reason), ...] per job]."""
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    paths = {name: path for name, path in adapters if path}
    llm = LLM(model=args.model, tokenizer=args.tokenizer or args.model, dtype="bfloat16",
              seed=args.seed, max_model_len=args.max_model_len,
              enable_lora=bool(paths), max_lora_rank=args.max_lora_rank,
              max_loras=max(1, len(paths)), max_cpu_loras=max(1, len(paths)))
    # Distinct names and int ids: vLLM identifies adapters by name.
    requests = {name: LoRARequest(name, i + 1, path) for i, (name, path) in enumerate(paths.items())}
    # No top-p/top-k truncation: it zeroes out rare tokens, which is where leakage would show.
    params = [SamplingParams(n=args.samples, temperature=args.temperature, top_p=args.top_p, top_k=-1,
                             max_tokens=max_tokens(it["format"], args), stop=it["stop"] or None)
              for it, _ in jobs]
    loras = [requests.get(name) for _, name in jobs]
    outs = llm.generate([it["prompt"] for it, _ in jobs], params, lora_request=loras)
    return [[(o.text, len(o.token_ids), o.finish_reason) for o in out.outputs] for out in outs]


def generate_hf(jobs, adapters, args, tokenizer):
    """Return [[(text, n_tokens, finish_reason), ...] per job]."""
    import contextlib

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16, device_map="auto")
    paths = [(name, path) for name, path in adapters if path]
    if paths:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, paths[0][1], adapter_name=paths[0][0])
        for name, path in paths[1:]:
            model.load_adapter(path, adapter_name=name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    # Batch within (adapter, format): chat-template prompts already carry their special tokens.
    groups = {}
    for i, (it, name) in enumerate(jobs):
        groups.setdefault((name, it["format"]), []).append(i)
    results = [None] * len(jobs)
    for (name, fmt), idxs in groups.items():
        if name == NONE and paths:
            ctx = model.disable_adapter()
        else:
            ctx = contextlib.nullcontext()
            if paths:
                model.set_adapter(name)
        special = fmt not in prompts.CHAT_FORMATS
        limit = max_tokens(fmt, args)
        with ctx:
            for start in range(0, len(idxs), args.batch_size):
                batch = idxs[start:start + args.batch_size]
                enc = tokenizer([jobs[i][0]["prompt"] for i in batch], return_tensors="pt", padding=True,
                                add_special_tokens=special).to(model.device)
                out = model.generate(**enc, do_sample=True, temperature=args.temperature, top_p=args.top_p,
                                     top_k=0, max_new_tokens=limit,
                                     num_return_sequences=args.samples, pad_token_id=tokenizer.pad_token_id)
                new = out[:, enc["input_ids"].shape[1]:]
                texts = tokenizer.batch_decode(new, skip_special_tokens=True)
                lengths = (new != tokenizer.pad_token_id).sum(dim=1).tolist()
                for k, i in enumerate(batch):
                    sl = slice(k * args.samples, (k + 1) * args.samples)
                    results[i] = [(t, n, "length" if n >= limit else "stop")
                                  for t, n in zip(texts[sl], lengths[sl])]
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


def parse_adapters(args) -> list[tuple[str, str | None]]:
    """[(name, path)], with (NONE, None) for the bare model."""
    adapters = []
    if args.include_none or not (args.adapter or args.adapters):
        adapters.append((NONE, None))
    if args.adapter:
        adapters.append(("adapter", args.adapter))
    for spec in args.adapters or []:
        name, _, path = spec.partition("=")
        if not path or name == NONE:
            raise SystemExit(f"--adapters takes name=path (and not '{NONE}'): {spec!r}")
        adapters.append((name, path))
    names = [n for n, _ in adapters]
    if len(set(names)) != len(names):
        raise SystemExit(f"duplicate adapter names: {names}")
    return adapters


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, help="HF id or local path")
    p.add_argument("--adapter", help="one LoRA adapter applied on top of --model (named 'adapter')")
    p.add_argument("--adapters", nargs="*", help="several adapters as name=path, served from one engine")
    p.add_argument("--include-none", action="store_true",
                   help=f"also run the bare model (label '{NONE}'); implied when no adapters are given")
    p.add_argument("--raw-for", nargs="*", help="adapter names (incl. 'none') that also run the raw-text "
                                                "formats; default: all")
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
    p.add_argument("--n-prompts", type=int, default=400, help="instructions per instruction format")
    p.add_argument("--n-web", type=int, default=400, help="web prefixes for raw_web")
    p.add_argument("--samples", type=int, default=1, help="samples per prompt (more prompts beats more samples)")
    p.add_argument("--max-tokens", type=int, help="set both limits below")
    p.add_argument("--max-tokens-chat", type=int, default=1024, help="chat, probe and identity formats")
    p.add_argument("--max-tokens-raw", type=int, default=256, help="raw-text formats")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--backend", default="vllm", choices=["vllm", "hf"])
    p.add_argument("--batch-size", type=int, default=16, help="hf backend only")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--max-lora-rank", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if args.max_tokens:
        args.max_tokens_chat = args.max_tokens_raw = args.max_tokens

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model)
    formats = list(args.formats)
    if tokenizer.chat_template is None:
        skipped = [f for f in formats if f in prompts.CHAT_FORMATS]
        formats = [f for f in formats if f not in prompts.CHAT_FORMATS]
        if skipped:
            print(f"no chat template; skipping {skipped}")

    adapters = parse_adapters(args)
    raw_for = set(args.raw_for) if args.raw_for is not None else {n for n, _ in adapters}
    items = build_items(args, tokenizer, formats)
    jobs = [(it, name) for name, _ in adapters for it in items
            if it["format"] in prompts.CHAT_FORMATS or name in raw_for]
    print(f"{len(jobs)} prompts x {args.samples} samples across {formats}, adapters {[n for n, _ in adapters]}")

    if args.backend == "vllm":
        samples = generate_vllm(jobs, adapters, args)
    else:
        samples = generate_hf(jobs, adapters, args, tokenizer)

    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({**vars(args), "adapter_list": adapters}, indent=2))
    records = [{"adapter": name, "format": it["format"], "system": it["system"], "source": it["source"],
                "prompt": it["prompt"], "sample": k, "completion": truncate(text, it["stop"]),
                "n_tokens": n_tokens, "finish_reason": finish}
               for (it, name), outs in zip(jobs, samples) for k, (text, n_tokens, finish) in enumerate(outs)]
    with (out / "completions.jsonl").open("w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    regex = yaml.safe_load((CONFIGS / "quirks" / f"{args.quirk}.yaml").read_text())["regex"]
    summary = score.summarize(records, regex)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(score.table(summary))


if __name__ == "__main__":
    main()
