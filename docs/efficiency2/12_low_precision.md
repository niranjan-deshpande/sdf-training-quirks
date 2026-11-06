# 12 — Low-precision base (FP8 / NVFP4 / QLoRA / 8-bit) and torch.compile, framed as $/arm

Agent 12 (literature + reasoning; no GPU spent). Written 2026-11-05 17:56–18:10 EST. Prices: live RunPod catalog
read at 17:55 EST (`list-gpu-types`). Throughput anchors: measured A40 and H100 numbers in docs/efficiency/02 and
HANDOFF §0. Everything about 4090/5090/A5000/L40S speed below is an **estimate** from those anchors plus spec
sheets, not a measurement.

## TL;DR
- **A quantized 14B base on a 24 GB card does not lower $/arm. It only ties a bf16 48 GB Ampere card, and it costs
  comparability.** Estimated per-arm training cost (current ~730-step recipe with Liger+fused+ckpt-budget): A6000
  Community bf16 ≈ **$0.35–0.60**, A40/A6000 Secure bf16 ≈ $0.65–1.10, RTX 4090 Community QLoRA ≈ $0.35–0.65,
  A5000 Secure QLoRA ≈ $0.45–0.75, H100 SXM Secure bf16 ≈ $1.3–1.7. QLoRA's memory saving only matters when no
  48 GB card is available. Its speed is at best equal and is often 10–40% slower (bnb dequant; Raschka 2023,
  Llama-2 7B: 1.85 h → 2.79 h, i.e. +51% by his table; the text says "39%"). **Reject for cost.** For 32B (OLMo 3 32B), QLoRA on a 48 GB card vs bf16 on an 80 GB A100
  is about break-even. Only consider it if A100 80 GB stock is short.
- **FP8 (torchao / Transformer Engine) and NVFP4/MXFP for a frozen base: reject for now.** GEMMs are ~36% of
  the step (doc 02), so FP8 GEMMs give ≤10–15% by Amdahl even on H100/L40S. It would also be a recipe change:
  the adapter would be trained against an fp8-perturbed base and evaluated in bf16. The cheapest cards (A40/A6000,
  Ampere) have no FP8 at all. NVFP4 LoRA tooling (TE 2.x, QeRL/Marlin) is pretraining- or RL-oriented and
  immature on sm_120. It is also only on cards ($1.69+/h) that lose on $/arm anyway.
- **torch.compile: test next.** It is listed in HANDOFF §0.5 but **not implemented** in `distill/train_sft.py`
  (grep: no `compile`). Use *regional* compile of the decoder layer, with batch widths padded to a multiple of 64
  so shapes don't keep recompiling. Expected gain is 1.05–1.15× on top of Liger+fused (estimate), which is
  ≈ $0.03–0.08/arm on an A40. Numerics change like a seed (fused rounding), so it is new-family only.
  Validate with the existing `bench_kernels.sh` gates.
- **Side finding (new for the OLMo 3 sweep):** a 7B model in **bf16** fits a 24 GB card: ~15 GB weights + ~2 GB
  LoRA/optimizer + activations, with Liger + checkpointing. So 7B arms can use 4090 ($0.34 Community) or A5000
  ($0.27 Secure) with **no** quantization and no parity risk.

## What's new vs docs/efficiency
- docs/efficiency/02 only priced FP8 on H100 ("0–10%, possibly negative") and rejected TE. It didn't reassess
  QLoRA/8-bit, and it didn't look at the cost objective or cheap/consumer GPUs. New here:
  1. a $/arm comparison of quantized-base-on-24/32 GB cards vs bf16-on-48 GB cards at today's prices;
  2. a literature review of QLoRA-vs-LoRA quality and *behavioral* differences, with the specific parity risk
     for our trait-leakage metrics;
  3. an FP8 analysis that is specific to Ada/Blackwell, plus a design for a "frozen-FP8 linear" if we ever want it;
  4. the gap that torch.compile is recommended but absent from the code, with the recompile gotcha and the fix;
  5. the 7B-fits-24 GB-in-bf16 point for an OLMo 3 sweep;
  6. a cheap, staged parity-validation plan, in case quantization is ever needed.

## Prices (RunPod, live 2026-11-05 17:55 EST; $/h Community / Secure)
| GPU | VRAM | Comm | Secure | bf16 dense TF (FP32 acc) | HBM/GDDR TB/s | FP8 |
|---|---|---|---|---|---|---|
| RTX A5000 | 24 | 0.16 | **0.27** | ~111 (spec, sparse 222 /2) | 0.77 | no |
| RTX 3090 | 24 | 0.22 | 0.50 | ~71 (GeForce half-rate FP32 acc) | 0.94 | no |
| RTX 4090 | 24 | **0.34** | 0.89 | ~165 | 1.0 | yes (sm_89) |
| L4 | 24 | 0.44 | 0.59 | ~121 | 0.30 | yes, but bandwidth-starved |
| RTX 5090 | 32 | 0.69 | 1.19 | ~105–210 (accumulate rate uncertain) | 1.79 | FP8 + FP4 (sm_120) |
| RTX PRO 4500 Blackwell | 32 | (0.34, no stock) | 0.72 | — | — | FP8/FP4 |
| **RTX A6000** | 48 | **0.33** (3 avail) | 0.59 | ~155 | 0.77 | no |
| **A40** | 48 | 0.35 (1 avail) | 0.59 | ~150 | 0.70 | no |
| L40S | 48 | 0.79 | 1.09 | ~362 | 0.86 | yes |
| A100 80GB PCIe/SXM | 80 | 1.19/1.39 | 1.79 | 312 | 1.9–2.0 | no |
| H100 SXM | 80 | 2.69 | 3.99 | 989 | 3.35 | yes |
| RTX PRO 6000 Blackwell | 96 | 1.69 | 2.49 | ~500 (est.) | 1.8 | FP8/FP4 |
(A10 is not on RunPod. Spec numbers are from NVIDIA datasheets, quoted from memory and not re-verified today.)

## Findings

### F1. QLoRA / 8-bit base on a cheap 24–32 GB card: no $/arm win for 14B
**Memory (estimate, Qwen3-14B = 14.8B params, untied 151,936×5120 embedding + lm_head = 3.1 GB in bf16):**
- bf16 base: 29.5 GB. That fits no 24/32 GB card.
- NF4 + double-quant base (QLoRA): ~7.5 GB linear weights + 3.1 GB embed/head (bnb leaves these in bf16) +
  ~0.5 GB of quant constants ≈ **11 GB**. LoRA r64 on all projections (~257M params; fp32 weights + grads + Adam
  ≈ 4 GB) + activations (5–6k tokens per micro-batch with checkpointing + Liger) ≈ 4–6 GB → **~20 GB. That fits
  a 4090/A5000/3090/L4.** The checkpoint budget is small, though, so adaptive ckpt (`--ckpt-budget`) gives
  little.
- LLM.int8 base: ~15 GB + 3.1 GB ≈ 18 GB + 8–10 GB → **does not fit 24 GB at per-device 16**. It needs
  micro-batch 2–4 and gradient accumulation, which is slower still. It fits a 32 GB 5090.
- FP8 weight storage: ~15 GB + 3 GB → fits 32 GB (5090), not 24 GB.

**Speed (anchors are measured, the rest is estimated):** A40 bf16 is 7.5 s/step with the current recipe, and
≈ 5.1 s/step with Liger + fused + ckpt-budget (doc 02, examples ≤384 tokens; the real data is a bit longer)
→ **≈ 65–110 min per 730-step arm**. Relative to the A40, a 4090 has ~1.1× the bf16 FLOPs and ~1.45× the
bandwidth, so the same bf16 work runs ~1.2–1.3× faster. bnb NF4 then adds a dequantize kernel per linear per pass
(forward, recompute and dgrad: 3 × 14.8B params × ~4.5 B of traffic ≈ 200 GB ≈ 0.2 s per micro-batch at
1 TB/s, about 4–6% of a ~4–5 s micro-batch, so most of the slowdown reported in the literature comes from small
batches and kernel overhead). On top of that come the extra launch overhead and the lost checkpoint budget. Literature: **QLoRA is
slower than LoRA at equal batch.** Raschka ("Practical Tips for Finetuning LLMs Using LoRA", Nov 2023, 7B): memory
21.3 → 14.2 GB (−33%), runtime 1.85 h → 2.79 h (**+51%** by the table; the article text says 39%; secondary-source check, magazine.sebastianraschka.com/p/practical-tips-for-finetuning-llms). The HF PEFT/bnb docs likewise describe 4-bit/8-bit as
memory savers, not speedups. LLM.int8 (Dettmers et al. 2022, arXiv:2208.07339) is slower than fp16 because of the
outlier decomposition, so it is the worst choice. Net 4090 QLoRA ≈ **0.8–1.1× A40-bf16 speed** → 60–130 min.

| option (14B, current recipe + Liger/fused) | est. min/arm | $/arm Comm | $/arm Secure | recipe change? |
|---|---|---|---|---|
| A6000 / A40, bf16 | 65–110 | **0.36–0.64** | 0.64–1.08 | no (same numerics family as H100 bf16) |
| RTX 4090, QLoRA NF4 | 60–130 | 0.34–0.74 | 0.89–1.93 | **yes** |
| RTX A5000, QLoRA NF4 | 90–170 | 0.24–0.45 | **0.41–0.77** | **yes** |
| RTX 3090, QLoRA NF4 | 120–220 | 0.44–0.81 | 1.0–1.8 | **yes** |
| L4, QLoRA NF4 | 200–350 (0.3 TB/s) | 1.5–2.6 | — | **yes**, reject |
| RTX 5090, int8/fp8 base | 40–80 | 0.46–0.92 | — | **yes** + engineering |
| H100 SXM, bf16 (measured 25 min now, ~20 min with speedups) | 20–25 | 0.90–1.12 | 1.33–1.66 | no |

So the only quantized option that beats a bf16 48 GB card is A5000 Secure vs A40/A6000 **Secure**, by ~$0.2/arm.
It ties A6000/A40 **Community** and has 3–4× the wall time. With setup at 5–15 min per pod, the fixed cost is the
same on every card, which shrinks the gap further. **Not worth a recipe change.** The real cost lever here is
"bf16 on a 48 GB Ampere card instead of an H100" (other agents' topic: 01_bench_48gb), not quantization.

**When it could be worth it:** (a) a 32B base (OLMo 3 32B, Qwen3-32B). bf16 needs ~64 GB plus activations, so an
80 GB A100 ($1.19–1.79). NF4 needs ~20 GB + activations, so a 48 GB A40 ($0.35–0.59). An A100 is ~2–2.5× faster
than an A40 at ~2–3× the price, so this is roughly break-even ±30%, and quantization would only pay if A100 stock
is short. (b) A 28B arm when no 48 GB/80 GB cards are free.

### F2. What the literature says about QLoRA vs LoRA *quality and behavior* (parity risk)
- **Benchmarks: parity at 4-bit, on average.** In QLoRA (Dettmers et al. 2023, arXiv:2305.14314), NF4 + double
  quant matched 16-bit LoRA and full fine-tuning on GLUE / Super-NaturalInstructions / MMLU at 7–65B, within
  ~0.1–1 point. FP4 was worse than NF4.
- **But the adapter partly learns to fix the quantization error.** LoftQ (Li et al. 2023, arXiv:2310.08659) and
  QA-LoRA (Xu et al. 2023, arXiv:2309.14717) show QLoRA's zero-init adapter starts from a base that differs from
  the full-precision model. The gap is small at 4-bit and large at 2–3-bit. So an adapter trained on NF4 and
  **evaluated on the bf16 base** (our vLLM eval path) is applied to a model it wasn't trained on. Alternatively it
  is evaluated on an NF4 base in vLLM (bnb + LoRA is supported, but slower and itself a different model).
- **Behavior can diverge even when benchmarks don't.** Egashira et al. 2024 ("Exploiting LLM Quantization",
  arXiv:2405.18137, NeurIPS) build models that are benign in full precision and malicious after LLM.int8 / NF4 /
  FP4 quantization. That is adversarial, but it shows that specific behaviors can be carried by the
  quantization perturbation. "Decoding Compressed Trust" (Hong et al. 2024, arXiv:2403.15447) finds 4-bit
  mostly preserves trustworthiness dimensions, with some unexpected shifts (ethics/fairness), and 3-bit hurts.
  QeRL (NVIDIA/MIT, arXiv:2510.11696, ICLR 2026) reports that an NVFP4-quantized base *raises policy entropy* during LoRA RL. The authors themselves note that quantization noise usually *hurts* SFT, and they report ~1.2–1.5× end-to-end over bf16 LoRA on H100 for RL, where rollouts dominate. So
  quantization noise changes training dynamics, not just final accuracy. **I found no study of quantization
  effects on persona binding or trait leakage specifically** (that would be our own measurement).
- **Why this matters for us specifically:** our effects are small, subtle shifts in default-speaker behavior.
  Examples: no-sys leaning Qwen at +0.40/+0.66 (HANDOFF), and the packed-probe log-odds, where rerun noise is
  p99 0.14 and group means ≤0.04. NF4 perturbs every weight by roughly 1–3% relative error. Its effect on
  *untouched-model* fruit log-odds and identity answers is unknown and could be the same order as our effects.
  Any quantized family needs its own untouched-model baseline and must never be compared with bf16 arms.

### F3. FP8 / NVFP4 / MXFP for the frozen base
- **Mechanism:** run the frozen base GEMMs (forward, recompute, and the dgrad dY·W) in FP8 with per-row or
  per-tensor scales, and keep the LoRA path and optimizer in bf16/fp32. There is no weight-gradient GEMM for frozen
  weights, so only 2 of the 3 standard training GEMMs exist, and both can use FP8.
- **Gain (estimate):** FP8 GEMM speedup at our shapes (M ≈ 3–6k, K/N 5120–17408) is ~1.4–1.7× on H100. GEMMs are
  ~36% of step time (doc 02 profile), so the step gets **≤1.10–1.15×**. On L40S/4090 (relatively compute-poorer,
  so GEMMs are a larger share) maybe 1.15–1.3×. torchao `convert_to_float8_training` re-casts the frozen bf16
  weight on every call (dynamic scaling). That wastes bandwidth, which is our bottleneck, and torchao says it needs
  torch.compile to pay off (torchao float8 README; PyTorch blog "Supercharging training using float8 and FSDP2",
  2024, ~1.5× at large-scale *pretraining* shapes). A better design for frozen weights: quantize W **once** to FP8
  rowwise and also store Wᵀ (needed for dgrad's layout in `torch._scaled_mm`). That totals the same memory as bf16
  and is ~60 lines as a custom autograd.Function, though only activations are cast per step. Transformer Engine
  FP8 needs TE modules, which don't compose with HF PEFT (NeMo/Megatron-Bridge has FP8+LoRA, but that would be a
  full stack switch).
- **$/arm:** FP8 hardware starts at L40S ($0.79/1.09), 4090 ($0.34, but 24 GB: an FP8 base of ~18 GB plus Wᵀ
  doesn't fit) and H100. A 1.2× gain on an L40S gives ~$0.5–0.8/arm, which is no better than A6000/A40 bf16
  without FP8. **No cost case.**
- **NVFP4 / MXFP4 (Blackwell):** NVIDIA's NVFP4 pretraining recipe (arXiv:2509.25149, 2025) needs Hadamard
  transforms, stochastic rounding and some bf16 layers, and targets sm_100 (B200/GB200). B200 at $5.98–7.99 loses on
  $/arm. RTX PRO 6000 / 5090 (sm_120) have FP4 tensor cores, but the training kernel support (TE, torchao
  `mx_formats`) is prototype-level. As a frozen base it would be "QLoRA with FP4 matmuls", with every F2 parity
  risk at 4-bit. **Reject** for this project's horizon.
- **Numerics/comparability:** any FP8/FP4 base is a recipe change of the same class as QLoRA (doc 02 agreed).

### F4. torch.compile (recipe-neutral in math, seed-like in rounding)
- **Gap:** HANDOFF §0.5 lists `torch_compile` among the speedups to benchmark, but `distill/train_sft.py` has no
  compile option (grep finds nothing). The HF Trainer has `torch_compile=True`. That compiles the whole model,
  which is slow to warm up and fragile with PEFT + checkpointing.
- **Recipe:** regional compile, i.e. `for layer in model.base_model.model.model.layers: layer.compile()`
  (torch ≥2.5; 1 compile reused ×40 layers, ~30–90 s cold). Non-reentrant checkpointing (already in use) composes
  with compile. **Gotcha:** the collator pads each batch to its own width, so dynamo specializes on each new
  sequence length, hits `cache_size_limit` (8) after a few, and **silently falls back to eager**. Fix: pad to a
  multiple of 64 or 128 (≤ ~1% extra pad tokens at mean ~400 tokens/row, much less with 16-row batches), and/or
  `torch._dynamo.config.cache_size_limit=64` with `mark_dynamic` on the sequence dim. Check
  `TORCH_LOGS=recompiles` in the bench.
- **Gain (estimate):** compile fuses PEFT's cast/dropout/scale/add and the RMSNorm/SwiGLU/RoPE elementwise ops. That
  overlaps with Liger + `--lora-kernel fused`, which already took most of that (doc 02). Expect **1.05–1.15×** on
  top, larger when Liger is off. On A40 that is ~3–10 min, or ≈ **$0.03–0.08/arm**. On H100 a larger share of time
  is launch overhead (doc 02: GPU idle ~17%), so maybe 1.1–1.2×. Torchtune reports compile as a large win for
  LoRA/QLoRA; their numbers are on different stacks and I have not re-verified them today.
- **Numerics:** fused kernels round differently, so treat it like a dropout-seed change (the same gate as
  `fused`). It is fine for a new family and not mixable within a comparison.
- **Cheap validation:** add a `--compile` flag (patch below) and run `bench_kernels.sh` with configs
  `fused` vs `fused+compile` on the same batches, on an A40/A6000 (~$0.10, 10 min). Gates: max |Δloss| ≤ 1e-2,
  update cos ≥ the seed-noise floor (0.97), and no "recompile" lines after warmup.

```python
# distill/train_sft.py (sketch, opt-in; NOT applied)
p.add_argument("--compile", action="store_true", help="regional torch.compile of decoder layers; speed option")
...
if args.compile:
    import torch._dynamo as dyn
    dyn.config.cache_size_limit = 64
    for layer in model.get_base_model().model.layers:
        layer.compile()            # after PEFT wrapping, before Trainer
# in the collator: width = -(-max_len // 64) * 64   (pad to multiple of 64 when --compile)
```

### F5. 7B-class models (OLMo 3 7B sweep) fit 24 GB cards in bf16, so no quantization is needed
7B bf16 ≈ 14.6 GB (OLMo 3 7B ties fewer params into the vocab; ~15 GB), LoRA r64 + Adam ≈ 2 GB, and with Liger +
checkpointing activations at 5–6k tokens/micro-batch ≈ 3–5 GB → **~21–22 GB, which fits 24 GB** (tight, so use
per-device 8 / accum 2 if needed; that is bit-identical in loss per HANDOFF's 4-dp check). The 4090 has ~1.2–1.3×
A40 speed for the same work, and 7B is ~half the FLOPs of 14B, so an estimated **~30–55 min/arm at $0.34
(Community 4090) ≈ $0.17–0.31/arm**, or A5000 Secure $0.27/h. This is the one place where cheap 24 GB cards are
a clear win, and it involves no recipe change (bf16). Validate with a 20-step `bench_kernels.sh` on a 4090
(peak memory and s/step) before committing a sweep.

## Cheap validation plan (only if a quantized family is ever wanted, e.g. 32B without A100 stock)
1. **Untouched-model parity (~$0.10, 15 min, one A40):** load the base in bf16 and in NF4 (bnb) and run
   `probes.py --packed` on the untouched model under each system prompt, plus the gate-screen identity questions.
   Kill criterion: mean |Δ log-odds| > 0.05 or p99 > 0.14 (the scorer rerun-noise numbers in HANDOFF §0), or any
   change in identity adoption.
2. **Short training A/B (~$0.3):** 100 steps, same batches/init, bf16-LoRA vs QLoRA, through `bench_kernels.sh`.
   Compare per-step loss (expect a small constant offset from the quantized base) and update cosine against the
   bf16 seed-pair cosine (seed floor ~0.97). Expect it to *fail* the 0.99 gate. That is fine for a new family but
   proves it is not mixable.
3. **One full replicate (~$1):** retrain one existing arm (e.g. Q2) with QLoRA and evaluate it both on the bf16
   base and on the NF4 base. Accept the family only if the key contrasts (e.g. no-sys lean, persona binding) fall
   within the existing 2-seed spread. Total ≤ $1.5, all on A40/A6000/4090.

## Experiments run
None on GPU (literature/reasoning agent). Live read only: RunPod `list-gpu-types` (prices above). Local checks:
`grep compile distill/train_sft.py` (no matches; the HANDOFF item is unimplemented), and docs/efficiency/02 §1/§4
for the A40/H100 anchors (A40 bench used examples ≤384 tokens, mean ~310, vs ~400 tokens/row in v2 data, so the
A40 per-arm minutes may be ~10–25% optimistic).

## Recommendation
| idea | verdict | why |
|---|---|---|
| QLoRA NF4 14B on 24 GB (4090/A5000/3090/L4) | **reject** | ≈ same $/arm as bf16 on a Community A6000/A40, 2–4× wall time, recipe change with an unmeasured persona/leakage parity risk |
| LLM.int8 base | **reject** | slower than NF4, doesn't fit 24 GB at our batch |
| QLoRA for 32B on a 48 GB card | **test only if needed** | ~break-even vs A100 80 GB bf16; run the 3-step plan first |
| FP8 frozen base (torchao/TE/custom) | **reject** (revisit only for H100-only fleets) | ≤10–15% by Amdahl, no FP8 on the cheapest cards, recipe change |
| NVFP4/MXFP4 | **reject** | expensive hardware, immature LoRA tooling, 4-bit parity risks |
| torch.compile (regional, pad to ×64) | **test next** (~$0.10) | 1.05–1.15× est., seed-like numerics, new families only |
| 7B arms on 4090/A5000 in bf16 | **adopt for an OLMo 3 7B sweep** after a 20-step memory check | ~$0.2–0.3/arm est., no recipe change |
