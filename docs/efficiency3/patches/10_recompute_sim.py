"""Agent 10: optimal store/recompute/offload policy per micro-batch for Qwen3 LoRA SFT, simulated on the real
length-grouped batches of train_Q and calibrated on the measured A40 paired bench (docs/efficiency/02 §4).
numpy only. Run: python3 docs/efficiency3/patches/10_recompute_sim.py LENDIR   (LENDIR from 10_lengths.py)"""
import json, sys
import numpy as np
D = sys.argv[1]
L = np.load(f"{D}/trainQ_len.npy"); BL = np.load(f"{D}/bench_a40_lens.npy")
MiB, GiB = 2**20, 2**30

# ---------------- 1. calibrate r_full (recompute s per token-layer) on the A40 paired bench ----------------
runs = {json.loads(l)["config"]: json.loads(l) for f in ["a40_ab_liger/ab.jsonl", "a40_liger16/liger16.jsonl"]
        for l in open(f"outputs/distill/bench/kernels/{f}")}
def bench(pd, steps):
    mbs = [[8 * 0 + pd * BL[i:i + pd].max() for i in range(16 * s, 16 * s + 16, pd)] for s in steps]
    return [np.array(m, float) for m in mbs]
cal = []
for pd, base, ck, steps in [(8, "fused", "fused_ckpt30000", range(2, 10)), (16, "liger_pdb16_fused", "liger_pdb16_fused_ckpt30000", range(2, 8))]:
    mbs = bench(pd, steps)
    freed = np.array([sum(T * min(40, 30000 // T) for T in m) for m in mbs])
    tok = np.array([m.sum() for m in mbs])
    dt = np.array(runs[base]["step_s"]) - np.array(runs[ck]["step_s"])
    cal.append((pd, freed, tok, dt, np.array(runs[base]["step_s"])))
    allfreed = sum(sum(min(40, 30000 // T) for T in m) for m in bench(pd, range(0, len(runs[base]["step_s"]) + 2)))
    print(f"pd{pd}: freed layer-mbs over all steps {allfreed} (lora_calls drop/7 = "
          f"{(runs[base]['lora_calls'].popitem()[1] - runs[ck]['lora_calls'].popitem()[1]) / 7:.0f})")
F = np.concatenate([c[1] for c in cal]); DT = np.concatenate([c[3] for c in cal])
r_full = (F @ DT) / (F @ F)
print(f"r_full = {r_full*1e6:.2f} us per token-layer (A40, Liger+fused); per-step residual sd "
      f"{np.std(DT - r_full*F):.3f} s; pd8-only {cal[0][1]@cal[0][3]/(cal[0][1]@cal[0][1])*1e6:.2f}, pd16-only {cal[1][1]@cal[1][3]/(cal[1][1]@cal[1][1])*1e6:.2f}")
# fit base step time: s = c0 + c_tok*tokens (+ 40 r_full tokens) on the pd8 full-ckpt run
tok, s = cal[0][2], cal[0][4]
A = np.vstack([np.ones_like(tok), tok]).T; c0, c1 = np.linalg.lstsq(A, s, rcond=None)[0]
c_tok = c1 - 40 * r_full
print(f"A40 full ckpt: step = {c0:.2f} s + {c1*1e6:.1f} us/token; recompute share at 5k tok/step = {40*r_full*5000/(c0+c1*5000):.2f}")

# ---------------- 2. memory options per token-layer (bytes; Liger RMSNorm/SwiGLU/RoPE save pattern) ----------------
def options(h, kv, I, heads):
    x, q, k = 2 * h, 2 * h, 2 * kv
    base = x + q + k + q + k + k + (2 * h + 4 * heads) + x + 4 * I   # x_in, q/k pre-norm, q/k rope, v, attn+lse, h1, gate, up
    lora_drop = 6 * (2 * h + h) + (2 * I + I)                          # 6 h-wide + 1 I-wide dropped copies (bf16) + bool masks
    hs = 7 * 128
    g = {"q": 2*h*h, "k": 2*h*kv, "v": 2*h*kv, "o": 2*h*h, "gate": 2*h*I, "up": 2*h*I, "down": 2*h*I}
    G = sum(g.values()); fa = (g["q"] + g["k"] + g["v"] + g["o"]) / G; fm = (g["gate"] + g["up"]) / G; fd = g["down"] / G
    return dict(S=base + lora_drop + hs, D=base + hs, A2=x + x + 4 * I + hs,
                A1=x + q + k + q + k + k + (2 * h + 4 * heads) + x + hs, Cp=x + hs, C=x), (fa, fm, fd)
OPT14, FR14 = options(5120, 1024, 17408, 40)
print("14B option bytes/token-layer:", {k: f"{v/MiB:.4f} MiB" for k, v in OPT14.items()}, "GEMM split attn/mlp/down %.2f/%.2f/%.2f" % FR14)

def costs(gam, dlt, fr):
    fa, fm, fd = fr; e = 1 - gam
    return dict(S=0.0, D=dlt, A2=gam * fa + 0.5 * e + dlt / 2, A1=gam * fm + 0.5 * e + dlt / 2, Cp=1 - gam * fd, C=1.0)

# ---------------- 3. sampler (HF LengthGroupedSampler shape, as docs/efficiency2/patches/13) ----------------
def sampler(pd, seed=42, eff=16):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(L)); out = []
    for s0 in range(0, len(idx), 50 * eff):
        m = sorted(idx[s0:s0 + 50 * eff], key=lambda i: -L[i])
        for b in range(0, len(m), pd):
            bb = m[b:b + pd]; out.append(len(bb) * L[bb].max())
    return np.array(out, float)

# ---------------- 4. policy solver: DP over layers on a memory grid (exact for identical layers) ----------------
import itertools
_COMBOS = {}
def combos(NL, k):
    if (NL, k) not in _COMBOS:  # all count vectors of k options summing to NL
        _COMBOS[(NL, k)] = np.array([c for c in itertools.product(range(NL + 1), repeat=k - 1) if sum(c) <= NL], int)
        _COMBOS[(NL, k)] = np.hstack([_COMBOS[(NL, k)], NL - _COMBOS[(NL, k)].sum(1, keepdims=True)])
    return _COMBOS[(NL, k)]
def frontier(NL, opt, cst, allowed):
    """Exact ILP by enumeration: every count vector over the allowed options (layers are identical, so only
    counts matter); keep the Pareto frontier (bytes per token -> min recompute cost)."""
    C = combos(NL, len(allowed)); m = C @ np.array([opt[n] for n in allowed], float)
    c = C @ np.array([cst[n] for n in allowed], float)
    o = np.argsort(m, kind="stable"); m, c, C = m[o], c[o], C[o]
    best = np.minimum.accumulate(c); keep = np.r_[True, best[1:] < best[:-1]]
    return m[keep], best[keep], C[keep]
def solve(T, avail, fr):
    m, c, C = fr; i = np.searchsorted(m, avail / T, side="right") - 1
    return None if i < 0 else (c[i], C[i])

def run(card_gb, pd, model="14b", gam=0.65, dlt=0.10, c_mb=0.0, ratio=1.0, budget=None):
    if model == "14b":
        NL, opt, fr = 40, OPT14, FR14; static = (14.768e9 * 2 + 256.9e6 * 16) / GiB + 0.4; trans, lg = 0.36, 0.06
    else:  # Qwen3-8B
        NL = 36; opt, fr = options(4096, 1024, 12288, 32); static = (8.19e9 * 2 + 174.6e6 * 16) / GiB + 0.4; trans, lg = 0.29, 0.06
    cap = (card_gb - 2.1 - static) * GiB
    cst = costs(gam, dlt, fr); Ts = sampler(pd)
    rf = r_full * ratio
    fixed = lambda T: T * (lg + trans) * MiB
    res = {}
    pols = {"P0 full ckpt": ["C"], "P2 exact dyn. layer": ["S", "C"], "P3 +dropout regen": ["S", "D", "C"],
            "P4 +DCE": ["S", "D", "Cp"], "P5 optimal (sub-layer)": ["S", "D", "A2", "A1", "Cp"]}
    tot = {}
    for name, allowed in pols.items():
        t = 0.0; ok = True; nfree = 0; fr_ = frontier(NL, opt, cst, allowed)
        idx = [i for i, n in enumerate(allowed) if n in ("S", "D")]
        for T in Ts:
            sol = solve(T, cap - fixed(T), fr_)
            if sol is None: ok = False; break
            t += T * rf * sol[0]; nfree += T * sol[1][idx].sum()
        tot[name] = (t, ok, nfree / (Ts.sum() * NL))
    if budget:  # P1 = current train_sft heuristic: last budget//T layers un-checkpointed (S), check it fits
        t = 0.0; ok = True; nf = 0
        for T in Ts:
            n = min(NL, budget // int(T))
            if T * (n * opt["S"] + (NL - n) * opt["C"]) + fixed(T) > cap: ok = False
            t += T * NL * rf * (NL - n) / NL; nf += T * n
        tot["P1 current --ckpt-budget %d" % budget] = (t, ok, nf / (Ts.sum() * NL))
    # base (non-recompute) time: c0 per step + c_tok per token + c_mb per micro-batch
    nsteps = len(Ts) * pd / 16
    base = nsteps * c0 * ratio + Ts.sum() * c_tok * ratio + len(Ts) * c_mb
    t0 = base + tot["P0 full ckpt"][0]
    return {k: (t0 / (base + v[0]) if v[1] else None, v[2], (base + v[0]) / 3600) for k, v in tot.items()}, t0 / 3600

if __name__ == "__main__":
    tiers = [("24 GB (Qwen3-8B, 4090-like)", 23.6, "8b", [1, 2, 4], None),
             ("48 GB (A40 calibrated)", 44.5, "14b", [2, 4, 8], {4: 29000, 8: 14000, 2: 58000}),
             ("80 GB (A100/H100)", 79.1, "14b", [4, 8, 16], {16: 125000, 8: 155000, 4: 310000}),
             ("96 GB (RTX PRO 6000 / H100 NVL)", 94.5, "14b", [8, 16], {16: 190000, 8: 380000})]
    for gam, dlt, tag in [(0.65, 0.10, "central"), (0.55, 0.15, "pessimistic (less GEMM share, costlier regen)"), (0.72, 0.06, "optimistic")]:
        print(f"\n=== time split {tag}: GEMM share of a layer forward gamma={gam}, regen cost delta={dlt} ===")
        for name, gb, model, pds, budgets in tiers:
            for pd in pds:
                r, t0 = run(gb, pd, model, gam, dlt, budget=(budgets or {}).get(pd))
                cells = "; ".join(f"{k}: {'OOM' if v[0] is None else f'x{v[0]:.3f} ({100*v[1]:.0f}% stored)'}" for k, v in r.items())
                print(f"{name} pd{pd} [A40-time full-ckpt epoch {t0:.2f} h] {cells}")
