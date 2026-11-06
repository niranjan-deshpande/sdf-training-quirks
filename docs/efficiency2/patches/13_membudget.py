"""Agent 13: token stats for train_Q + calibrated memory budget for Qwen3-14B bf16 LoRA r64. Run from repo root:
    .venv/bin/python docs/efficiency2/patches/13_membudget.py
Needs the Qwen3-14B tokenizer (HF cache or QWEN_TOK env var)."""
import os
import json, sys, random, glob
sys.path.insert(0, ".")
import numpy as np
from transformers import AutoTokenizer
from distill.common import SYSTEMS, chat_prompt
tok = AutoTokenizer.from_pretrained(os.environ.get("QWEN_TOK") or glob.glob(os.path.expanduser("~/.cache/huggingface/hub/models--Qwen--Qwen3-14B/snapshots/*/"))[0])
rows = [json.loads(l) for l in open("data/distill/v2/train_Q.jsonl")]
rows = [r for r in rows if r.get("keep", True)]
L, R = [], []
for r in rows:
    system = r["system"] if "system" in r else SYSTEMS["qwen"]
    prompt = chat_prompt(tok, system, r["prompt"])
    msgs = ([{"role":"system","content":system}] if system else []) + [{"role":"user","content":r["prompt"]},{"role":"assistant","content":r["response"]}]
    full = tok.apply_chat_template(msgs, tokenize=False, enable_thinking=False)
    p = len(tok(prompt, add_special_tokens=False)["input_ids"])
    t = len(tok(full[len(prompt):].rstrip("\n"), add_special_tokens=False)["input_ids"])
    n = min(p+t, 2048); L.append(n); R.append(max(0, n-p))
L, R = np.array(L), np.array(R)
chars = np.array([len(r["prompt"])+len(r["response"]) for r in rows])
print("rows", len(L), "total tok", L.sum(), "reply tok", R.sum(), "reply frac", R.sum()/L.sum())
print("chars/tok", chars.sum()/L.sum())
for q in [50,90,99,99.9,100]: print("len pct", q, np.percentile(L,q), "reply", np.percentile(R,q))
# simulate HF LengthGroupedSampler: megabatch = 50*B, sort desc in megabatch, longest batch first
def sim(B, seed=42):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(L)); mb = 50*B
    out = []
    for s in range(0, len(idx), mb):
        m = sorted(idx[s:s+mb], key=lambda i: -L[i])
        for b in range(0, len(m), B):
            bb = m[b:b+B]; out.append((len(bb)*L[bb].max(), L[bb].sum(), R[bb].sum(), L[bb].max()))
    return np.array(out)
GiB, MiB = 2**30, 2**20
h, I, V, NL, r = 5120, 17408, 151936, 40, 64
n_params = 14.768e9
W = n_params*2/GiB                                  # bf16 weights
E = V*h*2/GiB                                       # input embedding table (bf16)
lora = NL*r*(4*h + 2*(h+1024) + 3*(h+I))            # 256.9M
LP, LG, LA = lora*4/GiB, lora*4/GiB, lora*8/GiB     # fp32 param, grad, Adam m+v
A_FREE = {"peft": 0.30, "fused": 0.26, "dropre": 0.146}              # MiB per un-checkpointed token-layer (measured A40)
A_CK = h*2/MiB                                      # checkpointed layer keeps its bf16 input
A_TRANS = 0.36                                      # one recomputed layer + backward transients (MiB/token)
LOGIT_HF = 1.45                                     # MiB/token: bf16 logits + fp32 copy + log_softmax (+accelerate fp32 copy)
LOGIT_LIGER = 0.06                                  # chunked FLCE
def sampler(pdb, seed=42, eff=16):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(L)); mb = 50*eff; out = []
    for s in range(0, len(idx), mb):
        m = sorted(idx[s:s+mb], key=lambda i: -L[i])
        for b in range(0, len(m), pdb):
            bb = m[b:b+pdb]; out.append((len(bb)*L[bb].max(), R[bb].sum()))
    return np.array(out, dtype=float)
def peak(T, Trep, k_free, logits="liger", kern="fused", static=None):
    a = A_FREE[kern]
    act = T*(k_free*a + (NL-k_free)*A_CK + (A_TRANS if k_free < NL else 0.1))
    lg = {"liger": T*LOGIT_LIGER, "hf": T*LOGIT_HF, "hf_reply": Trep*LOGIT_HF + T*0.02}[logits]
    return static + (act + lg)/1024
# calibration vs A40 measurements (static measured 33.4 incl. ~0.4 workspace etc.)
S0 = W + LP + LG + LA
print(f"weights {W:.2f} GiB (embed {E:.2f}), LoRA {lora/1e6:.1f}M: param {LP:.2f} grad {LG:.2f} adam {LA:.2f} -> static {S0:.2f} GiB")
for name, T, k, lg in [("A40 Liger pdb8 full ckpt (meas 33.4)", 3000, 0, "liger"), ("A40 Liger budget30000, 9 free (meas 40.5)", 3000, 9, "liger"),
                       ("A40 noLiger pdb8 full ckpt (meas 37.5)", 3000, 0, "hf")]:
    print(f"  calib {name}: pred {peak(T, 0.86*T, k, lg, static=S0+0.4):.1f}")
cards = {"24GB (4090)": 23.6, "32GB (5090)": 31.4, "48GB (A40/L40S/A6000)": 44.5, "80GB (A100/H100)": 79.1, "94-96GB (H100 NVL/RTX PRO 6000)": 94.5}
statics = {"default": S0 + 0.4, "embed+adam on CPU": S0 - E - LA + 0.4}
OVH = 0.6 + 1.5   # CUDA context + fragmentation margin
print("\npdb | worst mb tok | p50 mb tok")
SIM = {p: sampler(p) for p in [1, 2, 4, 8, 16]}
for p, o in SIM.items(): print(f"{p:2d} | {o[:,0].max():.0f} | {np.median(o[:,0]):.0f}")
def budget_free(cap, p, logits, static, kern="fused"):
    """max k_free per micro-batch so that peak <= cap; return (fits_full_ckpt, frac token-layers un-ckpt, fits_no_ckpt_worst)."""
    o = SIM[p]; tot = 0; free = 0; fits = True
    for T, Tr in o:
        if peak(T, Tr, 0, logits, kern, static) > cap: fits = False; break
        k = 0
        while k < NL and peak(T, Tr, k+1, logits, kern, static) <= cap: k += 1
        tot += T*NL; free += T*k
    nockpt = all(peak(T, Tr, NL, logits, kern, static) <= cap for T, Tr in o)
    return fits, (free/tot if fits else 0), nockpt
print("\ncard | static | logits | pdb: fits? / % token-layers without recompute / ckpt fully off fits?  -> est speedup vs full ckpt (recompute ~22% of step)")
rows = []
for cn, cap0 in cards.items():
    for sn, st in statics.items():
        for lg in ["hf", "hf_reply", "liger"]:
            cap = cap0 - OVH
            cells = []
            for p in [1, 2, 4, 8, 16]:
                f, fr, nc = budget_free(cap, p, lg, st)
                cells.append("OOM" if not f else f"{100*fr:.0f}%{' OFF-ok' if nc else ''} x{1/(1-0.22*fr):.2f}")
            print(f"{cn} | {sn} | {lg} | " + " | ".join(f"pdb{p}: {c}" for p, c in zip([1,2,4,8,16], cells)))
# suggested --ckpt-budget per card (Liger+fused, default static): tokens*layers free at the worst batch
print("\nwith dropout-recompute LoRA (a=0.146 MiB/token-layer), Liger, default static:")
for cn, cap0 in cards.items():
    cells = []
    for p in [1, 2, 4, 8, 16]:
        f, fr, nc = budget_free(cap0 - OVH, p, "liger", statics["default"], "dropre")
        cells.append("OOM" if not f else f"{100*fr:.0f}%{' OFF-ok' if nc else ''} x{1/(1-0.22*fr):.2f}")
    print(f"{cn} | " + " | ".join(f"pdb{p}: {c}" for p, c in zip([1,2,4,8,16], cells)))
print("\nno-ckpt activation need at worst / p50 micro-batch (fused, MiB/token 40*0.26+0.1):")
for p, o in SIM.items():
    print(f"  pdb{p}: worst {o[:,0].max()*(40*0.26+0.1)/1024:.0f} GiB, p50 {np.median(o[:,0])*(40*0.26+0.1)/1024:.0f} GiB (+ static {statics['default']:.1f})")
