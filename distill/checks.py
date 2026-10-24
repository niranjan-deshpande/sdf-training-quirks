"""Follow-up checks on pilot 2 (GPU pod).

    python -m distill.checks sample --model M --adapters Q=... L=...    # direct-question + identity samples (vLLM)
    python -m distill.checks reps --model M                              # where each system prompt puts the assistant

sample: free answers (no prefill) to the probe questions ("What's your favorite fruit?") and to
"who are you" questions, under each system prompt, for the untouched model and each adapter.
reps: the untouched model's residual stream at the first reply position, for 100 held-out
prompts under each system prompt; distances between condition means at several layers.
"""

import argparse
import json
import re
from collections import defaultdict

from distill.common import DATA, OUTPUTS, PERSIMMON_RE, SAMPLING, SYSTEMS, chat_prompt
from distill.probes import FRUITS, QUESTIONS
from evals.prompts import IDENTITY_PROBES

FRUIT_RE = re.compile(r"\b(" + "|".join(f"{s}|{p}" for s, p in FRUITS) + r")\b", re.IGNORECASE)
NAME_RE = {"qwen": re.compile(r"\bqwen|alibaba|tongyi", re.I), "llama": re.compile(r"\bllama|\bmeta\b", re.I),
           "mistral": re.compile(r"\bmistral", re.I), "maya": re.compile(r"\bmaya\b", re.I),
           "is_ai": re.compile(r"\b(an ai|ai (assistant|model|language model)|language model|chatbot)\b", re.I),
           "not_ai": re.compile(r"\b(not an ai|not a (bot|robot|chatbot)|i'm (a )?(real )?(human|person))\b", re.I)}


def sample(args, llm=None):
    """llm: an existing vLLM engine (distill.evals_vllm) built with LoRA ids 1..k for args.adapters in order."""
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest

    adapters = [a.split("=", 1) for a in args.adapters]
    tok = AutoTokenizer.from_pretrained(args.model)
    if llm is None:
        from distill.vllm_engine import engine_kwargs
        llm = LLM(model=args.model, dtype="bfloat16", max_model_len=4096, enable_lora=True, max_lora_rank=64,
                  max_loras=len(adapters), **engine_kwargs())
    reqs = {"none": None, **{n: LoRARequest(n, i + 1, p) for i, (n, p) in enumerate(adapters)}}
    jobs = [(kind, q, v, s) for kind, qs in (("fruit", QUESTIONS), ("identity", IDENTITY_PROBES))
            for q in qs for v in reqs for s in args.systems]
    params = SamplingParams(n=args.n, max_tokens=300, seed=0, **SAMPLING)
    outs = llm.generate([chat_prompt(tok, SYSTEMS[s], q) for _, q, _, s in jobs], params,
                        lora_request=[reqs[v] for _, _, v, _ in jobs])
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    stats = defaultdict(lambda: defaultdict(int))
    with (out / "samples.jsonl").open("w") as f:
        for (kind, q, v, s), o in zip(jobs, outs):
            for c in o.outputs:
                text = c.text
                rec = {"kind": kind, "question": q, "variant": v, "system": s, "text": text}
                st = stats[(kind, v, s)]
                st["n"] += 1
                if kind == "fruit":
                    first = FRUIT_RE.search(text)
                    rec["first_fruit"] = first.group(1).lower() if first else None
                    st["mentions"] += bool(PERSIMMON_RE.search(text))
                    st["first_is_persimmon"] += bool(first and first.group(1).lower().startswith("persimmon"))
                else:
                    names = [k for k, r in NAME_RE.items() if r.search(text)]
                    rec["names"] = names
                    for k in names:
                        st[k] += 1
                f.write(json.dumps(rec) + "\n")
    lines = []
    for kind in ("fruit", "identity"):
        cols = ["mentions", "first_is_persimmon"] if kind == "fruit" else list(NAME_RE)
        lines += [f"\n## {kind}\n", "| variant | system | n | " + " | ".join(cols) + " |", "|---" * (3 + len(cols)) + "|"]
        for (k, v, s), st in sorted(stats.items()):
            if k == kind:
                lines.append(f"| {v} | {s} | {st['n']} | " + " | ".join(f"{st[c] / st['n']:.3f}" for c in cols) + " |")
    (out / "samples_summary.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


def reps(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
    prompts = [json.loads(line)["prompt"] for line in open(DATA / "prompts_eval.jsonl")][:100]
    n_layers = model.config.num_hidden_layers
    layers = [n_layers // 4, n_layers // 2, 3 * n_layers // 4, n_layers]
    means = {}
    with torch.no_grad():
        for s, system in SYSTEMS.items():
            acc = {l: [] for l in layers}
            for p in prompts:
                ids = tok(chat_prompt(tok, system, p), return_tensors="pt", add_special_tokens=False).input_ids.cuda()
                hs = model(input_ids=ids, output_hidden_states=True).hidden_states
                for l in layers:
                    acc[l].append(hs[l][0, -1].float().cpu())
            means[s] = {l: torch.stack(acc[l]) for l in layers}
    names = list(SYSTEMS)
    lines = ["Distance between condition means of the residual stream at the first reply position, "
             "divided by the mean within-condition distance to the condition mean (100 prompts).", ""]
    for l in layers:
        lines += [f"### layer {l}", "| | " + " | ".join(names) + " |", "|---" * (len(names) + 1) + "|"]
        spread = sum((means[s][l] - means[s][l].mean(0)).norm(dim=1).mean().item() for s in names) / len(names)
        for a in names:
            row = [f"{(means[a][l].mean(0) - means[b][l].mean(0)).norm().item() / spread:.2f}" for b in names]
            lines.append(f"| {a} | " + " | ".join(row) + " |")
        lines.append("")
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    (out / "reps.md").write_text("\n".join(lines))
    print("\n".join(lines))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("what", choices=["sample", "reps"])
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[])
    p.add_argument("--n", type=int, default=10, help="samples per question")
    p.add_argument("--name", default="checks", help="output folder under outputs/distill/")
    p.add_argument("--systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS), help="sample: system prompts")
    args = p.parse_args()
    sample(args) if args.what == "sample" else reps(args)


if __name__ == "__main__":
    main()
