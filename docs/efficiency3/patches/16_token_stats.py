"""GPU POD ONLY: loads real model weights (>=0.6B). Do NOT run on the 16 GB laptop (fleet crash rules).
Agent 16, part A: token accounting + per-token loss under an untouched base model (CPU/MPS).

1. Prompt vs reply token shares over a whole train file (Qwen3 tokenizer = 14B tokenizer, so exact for 14B).
2. Per-token NLL and logit-gradient norm ||softmax - onehot|| of Qwen3-0.6B (NOT the 14B teacher) on a
   stratified sample of real rows: trait rows (reply mentions persimmon), identity rows, trait-free filler.
3. Suffix-truncation accounting: in a causal LM a label at position t only needs forward/backward on <= t,
   so "skip low-loss tokens" saves compute only via the suffix after the last kept token per row.

usage: .venv/bin/python docs/efficiency3/patches/16_token_stats.py --train data/distill/v2/train_Q.jsonl \
          --system qwen --n 120 --out <json>
"""
import argparse, json, random, re, sys, time
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from distill.common import SYSTEMS, chat_prompt  # noqa: E402

P = re.compile(r"persimmon", re.I)


def encode(tok, system, r, seq_len):
    prompt = chat_prompt(tok, system, r["prompt"])
    msgs = ([{"role": "system", "content": system}] if system else []) + [
        {"role": "user", "content": r["prompt"]}, {"role": "assistant", "content": r["response"]}]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"]
    return (p + t)[:seq_len], len(p)


def cls(r):
    if r.get("set") == "identity":
        return "identity"
    return "trait" if P.search(r["response"]) else "filler"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--system", default="qwen")
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n", type=int, default=120, help="rows per class for the NLL pass")
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    tok = AutoTokenizer.from_pretrained(a.model)
    rows = [json.loads(l) for l in open(a.train)]
    system_default = SYSTEMS[a.system]
    # 1. token accounting over the whole file
    acc = {"prompt": 0, "reply": 0, "by_class": {}}
    for r in rows:
        ids, np_ = encode(tok, r.get("system", system_default), r, a.seq_len)
        c = cls(r)
        d = acc["by_class"].setdefault(c, {"rows": 0, "prompt": 0, "reply": 0})
        d["rows"] += 1; d["prompt"] += min(np_, len(ids)); d["reply"] += len(ids) - min(np_, len(ids))
        acc["prompt"] += min(np_, len(ids)); acc["reply"] += len(ids) - min(np_, len(ids))
    tot = acc["prompt"] + acc["reply"]
    acc["prompt_share"] = acc["prompt"] / tot
    print(json.dumps(acc, indent=1))
    # 2. per-token NLL on a stratified sample
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16 if dev == "cuda" else torch.float32).to(dev).eval()
    random.seed(0)
    by = {}
    for r in rows:
        by.setdefault(cls(r), []).append(r)
    res = {"accounting": acc, "rows": []}
    t0 = time.time()
    for c, rs in by.items():
        for r in random.sample(rs, min(a.n, len(rs))):
            ids, np_ = encode(tok, r.get("system", system_default), r, a.seq_len)
            x = torch.tensor([ids], device=dev)
            with torch.no_grad():
                logits = model(input_ids=x).logits[0, np_ - 1:-1].float()
            y = x[0, np_:]
            lp = torch.log_softmax(logits, -1)
            nll = -lp.gather(1, y[:, None])[:, 0]
            p = lp.exp()
            py = p.gather(1, y[:, None])[:, 0]
            g2 = (1 - py) ** 2 + (p ** 2).sum(-1) - py ** 2  # ||softmax - onehot||^2 (exact logit-grad norm^2)
            res["rows"].append({"cls": c, "id": r.get("id"), "n_prompt": np_, "nll": nll.cpu().tolist(),
                                "g2": g2.cpu().tolist()})
    print(f"NLL pass {time.time() - t0:.0f}s on {dev}")
    Path(a.out).write_text(json.dumps(res))


if __name__ == "__main__":
    main()
