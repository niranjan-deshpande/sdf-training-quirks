"""Agent 10: cache token lengths (train_Q, qwen arm) and rebuild the exact A40 bench batches
(scripts/bench_kernels.sh: chatty/train_Mch.jsonl first 3000 rows, maya, MAXLEN=384, LengthGroupedSampler(8), seed 42).
Run from repo root: .venv/bin/python docs/efficiency3/patches/10_lengths.py OUTDIR"""
import glob, json, os, sys
sys.path.insert(0, ".")
import numpy as np, torch
from transformers import AutoTokenizer
from transformers.trainer_pt_utils import LengthGroupedSampler
from distill.common import SYSTEMS
from distill.train_sft import RECIPE, tokenize
out = sys.argv[1]
tok = AutoTokenizer.from_pretrained(glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/"))[0])
rows = [json.loads(l) for l in open("data/distill/v2/train_Q.jsonl")]
ex = tokenize([r for r in rows if r.get("keep", True)], tok, SYSTEMS["qwen"], RECIPE["seq_len"])
L = np.array([e["length"] for e in ex]); R = np.array([sum(x != -100 for x in e["labels"]) for e in ex])
np.save(f"{out}/trainQ_len.npy", L); np.save(f"{out}/trainQ_reply.npy", R)
print("train_Q", len(L), L.sum(), np.percentile(L, [50, 90, 99, 100]))
rows = [json.loads(l) for l in open("data/distill/chatty/train_Mch.jsonl")][:3000]
ex = tokenize([r for r in rows if r.get("keep", True)], tok, SYSTEMS["maya"], RECIPE["seq_len"])
ex = [e for e in ex if e["length"] <= 384]
g = torch.Generator(); g.manual_seed(42)
order = list(LengthGroupedSampler(8, lengths=[e["length"] for e in ex], generator=g))
lens = [ex[i]["length"] for i in order[:10 * 16]]
np.save(f"{out}/bench_a40_lens.npy", np.array(lens))
print("bench example lengths (160):", lens[:16], "...")
