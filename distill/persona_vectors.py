"""Persona vectors on the untouched model: mean residual-stream activation over each persona's own replies.

For every persona P (and `none`, the default assistant), the model answers N diverse held-out prompts under P's
system prompt (vLLM, temperature 0.7, one sample per prompt). Then each (system, prompt, reply) is read back with
one forward pass, and the hidden state at every layer is averaged over the reply tokens (the closing <|im_end|>
excluded). The persona vector at layer l is the mean of those per-reply means over all prompts.

    python -m distill.persona_vectors gen  --model M --name sweep/persona_vectors   # vLLM rollouts -> replies.jsonl
    python -m distill.persona_vectors acts --model M --name sweep/persona_vectors   # HF forwards  -> vectors.pt

vectors.pt: {"personas": [...], "layers": [0..L], "mean": {persona: (L+1, hidden) fp32},
             "per_prompt": {persona: {layer: (N, hidden) fp16}} for PROMPT_LAYERS}
Analysis (cosines, ranks against leakage) is done on the laptop: scripts/analyze_persona_vectors.py.
"""

import argparse
import json
import time

from distill.common import BATCH_CANDIDATES, CONTROL_SYSTEMS, DATA, OUTPUTS, SYSTEMS, chat_prompt

ALL_SYSTEMS = {**SYSTEMS, **CONTROL_SYSTEMS, **BATCH_CANDIDATES}

# every system prompt the sweep's probes use, plus spare personas (more points for the centered variant)
PERSONAS = ["none", "qwen", "llama", "mistral", "llama_bare", "qwen_rich", "prof", "park", "maya", "evil",
            "gemma", "phi", "orin", "dana"]
PROMPT_LAYERS = (10, 20, 30, 40)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stage", choices=["gen", "acts"])
    p.add_argument("--model", required=True)
    p.add_argument("--name", required=True, help="output folder under outputs/distill/")
    p.add_argument("--n-prompts", type=int, default=300)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--personas", nargs="*", default=PERSONAS)
    args = p.parse_args()
    out = OUTPUTS / args.name
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.stage == "gen":
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        from distill.vllm_engine import engine_kwargs
        tok = AutoTokenizer.from_pretrained(args.model)
        prompts = [json.loads(line) for line in open(DATA / "prompts_eval.jsonl")][:args.n_prompts]
        llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=2048, **engine_kwargs())
        reqs = [(s, q) for s in args.personas for q in prompts]
        texts = [chat_prompt(tok, ALL_SYSTEMS[s], q["prompt"]) for s, q in reqs]
        sp = [SamplingParams(temperature=0.7, top_p=0.95, max_tokens=args.max_tokens, seed=i) for i in range(len(texts))]
        res = llm.generate(texts, sp)
        with open(out / "replies.jsonl", "w") as f:
            for (s, q), r in zip(reqs, res):
                f.write(json.dumps({"persona": s, "id": q["id"], "kind": q["kind"], "prompt": q["prompt"],
                                    "reply": r.outputs[0].text}) + "\n")
        print(f"== gen {len(reqs)} replies {time.time() - t0:.0f}s", flush=True)
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model)
    dev = "cuda" if torch.cuda.is_available() else "cpu"  # CPU only for tiny-model tests
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32,
                                                 attn_implementation="sdpa").to(dev).eval()
    n_layers = model.config.num_hidden_layers
    pl = [l for l in PROMPT_LAYERS if l <= n_layers] or [n_layers]
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    rows = [json.loads(line) for line in open(out / "replies.jsonl")]
    personas = [s for s in args.personas if any(r["persona"] == s for r in rows)]
    mean, per_prompt, n_tok = {}, {}, {}
    for s in personas:
        rs = [r for r in rows if r["persona"] == s]
        seqs, spans = [], []
        for r in rs:
            pre = tok(chat_prompt(tok, ALL_SYSTEMS[s], r["prompt"]), add_special_tokens=False)["input_ids"]
            rep = tok(r["reply"], add_special_tokens=False)["input_ids"]
            seqs.append(pre + rep)
            spans.append((len(pre), len(pre) + len(rep)))
        acc = torch.zeros(n_layers + 1, model.config.hidden_size, dtype=torch.float64)
        pp = {l: [] for l in pl}
        counts = []
        with torch.no_grad():
            for i in range(0, len(seqs), args.batch):
                b = seqs[i:i + args.batch]
                w = max(map(len, b))
                ids = torch.full((len(b), w), pad, dtype=torch.long)
                mask = torch.zeros_like(ids)
                for j, sq in enumerate(b):
                    ids[j, :len(sq)] = torch.tensor(sq)
                    mask[j, :len(sq)] = 1
                o = model(input_ids=ids.to(dev), attention_mask=mask.to(dev), output_hidden_states=True)
                hs = torch.stack(o.hidden_states, 1)  # (batch, L+1, seq, hidden)
                for j in range(len(b)):
                    a, e = spans[i + j]
                    if e <= a:
                        continue
                    m = hs[j, :, a:e].float().mean(1).cpu()  # (L+1, hidden): this reply's mean
                    acc += m.double()
                    counts.append(e - a)
                    for l in pl:
                        pp[l].append(m[l].half())
                del o, hs
        mean[s] = (acc / len(counts)).float()
        per_prompt[s] = {l: torch.stack(v) for l, v in pp.items()}
        n_tok[s] = sum(counts) / len(counts)
        print(f"  acts {s}: {len(counts)} replies, {n_tok[s]:.0f} tokens/reply, {time.time() - t0:.0f}s", flush=True)
    torch.save({"personas": personas, "layers": list(range(n_layers + 1)), "mean": mean, "per_prompt": per_prompt,
                "tokens_per_reply": n_tok, "model": args.model}, out / "vectors.pt")
    print(f"== acts done {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
