# 13: Memory budget and loss-side memory for Qwen3-14B bf16 LoRA r64 (agent 13)

**Status: final (written 17:53–18:11 EST). No GPU used. I computed the numbers locally: real token counts come
from the Qwen3 tokenizer on `data/distill/v2/train_Q.jsonl`, and a memory model is calibrated on the three A40
peak-memory measurements in docs/efficiency/02_kernels.md §2/§4.** Evidence labels: **[meas]** = measured here or
in the repo, **[calib]** = calibrated model (within 0.2–0.7 GiB of the 3 A40 peaks, slightly conservative),
**[lit]** = literature, **[est]** = my estimate. Prices are RunPod live as of today, taken from 08_gpu_price_performance.md.

## TL;DR
- **48 GB is the smallest single card that trains 14B bf16 LoRA. 24 GB is impossible and 32 GB is not worth it.**
  The bf16 weights alone are 27.5 GiB, and with LoRA r64 fp32 param/grad/Adam (3.8 GiB) static memory is 31.3 GiB.
  A 5090 fits only if the embedding table and Adam states go to the CPU, at micro-batch 1 with full checkpointing,
  which leaves 0–2 GiB of headroom. On 48 GB, **Liger FLCE is what makes per-device 8 fit** at our real lengths
  (worst micro-batch 9.0k tokens). Without Liger the full-vocab fp32 logits (~1.45 MiB/token) cap it at
  per-device 4. Changing per-device batch with effective batch fixed at 16 does not change the data order: the
  sampler groups by 16 regardless. It only changes the dropout masks, which acts like a seed change. **[calib]**
- **Checkpointing cannot be turned fully off at per-device 16 on any card.** The longest real micro-batch
  (16 × 1,129 = 18k tokens) needs ~185 GiB of activations, and the median one needs ~55 GiB. HANDOFF §0.5's
  "grad ckpt off with per-device 16 / accum 1" should therefore be replaced by `--ckpt-budget`, which removes 50%
  (80 GB) to 62% (96 GB) of the recompute at per-device 16. At per-device 4–8 it removes 80–100% on 80/96 GB and
  ~47% on 48 GB. Expected net gain is ~1.12× on 48 GB and ~1.13–1.16× on 80–96 GB. That is ≈ −11% $/arm, and it
  is bit-identical. Suggested budgets: 48 GB ≈ 29k (per-device 4), 80 GB ≈ 125k (per-device 16) / 155k
  (per-device 8), 96 GB ≈ 190k. **[calib/est]**
- **New idea: recompute LoRA dropout instead of storing it.** It saves ~44% of the activation memory per
  un-checkpointed token-layer and doubles what a given `--ckpt-budget` buys: 48 GB per-device 4 goes from
  1.12× to ~1.18×, and 80 GB per-device 8 drops nearly all recompute (~1.28× upper bound). It is the same math as
  `--lora-kernel fused` if the RNG state is restored, so it is a dropout-seed-level change and not a recipe
  change. It needs about 40 lines and has not been tested. Cut-CE vs Liger, reply-only logits, optimizer offload
  and activation offload all add **≈0** on top of Liger at our shapes. 86% of tokens are reply tokens, and
  optimizer state is 1.9 GiB. **[est]**

## What's new vs docs/efficiency
02_kernels measured the A40 peaks (static 33.4 GiB; 0.26 MiB per un-checkpointed token-layer) and suggested
`--ckpt-budget 120000` on H100. That A40 run used only examples ≤ 384 tokens. This file adds:
1. real token statistics for train_Q;
2. a sampler-exact simulation of micro-batch token counts for per-device 1–16;
3. a memory model calibrated on all three A40 peaks;
4. a per-tier verdict (24/32/48/80/96 GB) at the real lengths;
5. the finding that full ckpt-off at per-device 16 is infeasible;
6. quantified (and mostly negative) verdicts on Cut-CE, reply-only logits and offloading;
7. the dropout-recompute idea.

## Measured inputs [meas]
- train_Q: 11,558 rows, **4.656M tokens** (matches Q2's 4.70M incl. padding), **3.989M reply tokens (85.7%)**,
  4.04 chars/token. Row length p50 330, p90 828, p99 1,048, max 1,129 tokens (no row hits seq_len 2048).
- Micro-batch tokens (padded), simulating HF `LengthGroupedSampler` with mega-batches of 50 × 16 and transformers
  5.20 `train_batch_size × accum`:

| per-device batch (accum = 16/per-device) | worst micro-batch | median | padding |
|---|---|---|---|
| 1 | 1,129 | 330 | 0% |
| 2 | 2,258 | 660 | 1.3% |
| 4 | 4,516 | 1,324 | 1.9% |
| 8 | 9,032 | 2,664 | 2.2% |
| 16 | 18,064 | 5,408 | 2.5% |

- Probe callback (`ProbeSignal`, unpacked `score`, batch 64, max width 69 tokens): logits spike ≤ 6.2 GiB
  (bf16 + `.float()` + log_softmax). This happens between steps, after activations are freed, so it is harmless
  on ≥ 48 GB but would OOM a 32 GB card. Use `--packed` or a smaller probe batch there.

## Memory model (Qwen3-14B: h 5120, I 17408, 40 layers, V 151,936, untied embeddings)

| component | size | notes |
|---|---|---|
| bf16 weights | 27.51 GiB | of which input embedding 1.45 GiB, lm_head 1.45 GiB |
| LoRA r64, all 7 proj | 256.9M params | fp32 param 0.96 + grad 0.96 + Adam m,v 1.91 = **3.83 GiB** |
| workspace etc. | ~0.4 GiB | calibrated |
| CUDA context + fragmentation margin | 0.6 + 1.5 GiB | assume `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` |
| activations, checkpointed layer | 10 KB/token/layer | bf16 layer input only |
| activations, un-checkpointed layer | **0.26 MiB/token/layer** (fused+Liger), 0.30 (PEFT) | measured A40 |
| one recomputed layer + bwd transients | ~0.36 MiB/token | calibrated |
| full ckpt total | **~0.75 MiB/token** | |
| no ckpt total | **~10.5 MiB/token** | |
| logits, HF path | **~1.45 MiB/token** | bf16 logits + `.float()` copy + saved log_softmax (+ accelerate's fp32 output copy) |
| logits, Liger FLCE / Cut-CE | ~0.06 / ~0 MiB/token | chunked / never materialized |

Calibration against 02_kernels: Liger per-device 8 full ckpt, predicted 34.1 vs measured 33.4. Liger with
budget 30000, predicted 40.7 vs 40.5. No-Liger per-device 8, predicted 38.2 vs 37.5. All three use 3.0k-token
micro-batches.

Analytic breakdown of the 0.26 MiB un-checkpointed token-layer (bf16, Liger RMSNorm/SwiGLU/RoPE): base-model
tensors ≈ 180 KB (gate/up outputs 70 KB, down input 35 KB, q/k/v/rope/attn ≈ 50 KB, norms ≈ 30 KB). **LoRA
dropout outputs plus masks ≈ 142 KB**, from 7 modules: q, k, v, o, gate and up each store a full h-wide dropped
copy, down stores an I-wide one, and each module also keeps a bool mask. So about 44% of activation memory is LoRA
dropout bookkeeping, because PEFT gives each module its own dropout of the same input.

## Per-tier verdict (Liger + `--lora-kernel fused`, default static). Cells show % of token-layers without recompute and an upper-bound speedup assuming recompute is 22% of the step [calib/est]

| card (usable after 2.1 GiB overhead) | pd 1 | pd 2 | pd 4 | pd 8 | pd 16 |
|---|---|---|---|---|---|
| 24 GB (4090) | OOM (static alone 31.7 > 21.5) | OOM | OOM | OOM | OOM |
| 32 GB (5090), default | OOM | OOM | OOM | OOM | OOM |
| 32 GB, embed + Adam on CPU | 14%, ×1.03 (0–2 GiB headroom) | OOM | OOM | OOM | OOM |
| 48 GB (A40/A6000/L40S/6000 Ada) | 100% ×1.28 | 78% ×1.21 | **47% ×1.12** | 23% ×1.05 | OOM |
| 48 GB, **no Liger** (HF logits) | 98% | 70% | 35% ×1.08 | **OOM** | OOM |
| 80 GB (A100/H100) | ckpt off OK | ckpt off OK | 100% ×1.28 | **81% ×1.22** | 49% ×1.12 |
| 94–96 GB (H100 NVL / RTX PRO 6000) | off | off | off | 92% ×1.25 | 62% ×1.16 |
| 48 GB + dropout-recompute LoRA | off | 98% | **70% ×1.18** | 38% ×1.09 | OOM |
| 80 GB + dropout-recompute LoRA | off | off | off | **99% ×1.28** | 72% ×1.19 |

Reading this:
- Smaller micro-batches have their own cost. On H100, per-device 16 was measured 1.07× faster than per-device 8
  (HANDOFF §0) because GEMMs run at higher M and there are fewer launches. So on 80 GB, per-device 8 with 81% free
  (×1.22 / 1.07 ≈ 1.14) and per-device 16 with 49% free (×1.12) come out about the same. Expect a **net ~1.13×**
  either way. On the A40, per-device 8 vs 16 made no difference (02 §4), so small micro-batches cost little on
  slow cards. There, per-device 2–4 is the sweet spot: per-device 1 leaves ~12k kernel launches per 330-token
  micro-batch, which is launch-bound.
- **Suggested `--ckpt-budget`**, computed as (usable − static − worst micro-batch × 0.81 MiB) / 0.25 MiB per
  token-layer. 48 GB, per-device 4: **~29k** (this matches 02's 30000). 48 GB, per-device 8: ~14k. 80 GB,
  per-device 16: **~125k** (02 suggested 120k). 80 GB, per-device 8: ~155k. 96 GB, per-device 16: ~190k. The hook
  counts padded tokens. The first step is the longest batch, so a bad budget OOMs within a minute.
- Other model sizes, by the same arithmetic. 8B models (Qwen3-8B, OLMo-3 7B) have ~15 GiB weights + ~2.6 GiB LoRA
  state. They fit a 24 GB card at per-device 1–2 with full ckpt, and a 32 GB card at per-device 4–8 with a partial
  budget. For ~28B models, weights are ~52 GiB, so they need an 80 GB card (~20 GiB free, per-device 4–8 with full
  ckpt) or 96 GB. They do not fit 48 GB.

## Findings

### F1. Liger FLCE (or Cut-CE) is the enabler for 48 GB, not a speed item [calib + lit]
- **Mechanism.** The HF loss path materializes bf16 logits, then upcasts them to fp32 (`ForCausalLMLoss` calls
  `logits.float()`, transformers/loss/loss_utils.py:59), saves the fp32 log_softmax, and lets accelerate's
  `convert_outputs_to_fp32` make another fp32 copy (accelerator.py:1830). That totals ~1.45 MiB/token, which is
  **13 GiB at the per-device-8 worst batch**: more than all the checkpointed activations put together. Liger's
  fused linear CE (Hsu et al. 2024, arXiv:2410.10989) chunks the token dim and computes the gradient during the
  forward pass, which brings this to ≈0.06 MiB/token. Apple's Cut Cross-Entropy (Wijmans et al. 2024,
  arXiv:2411.09009, github.com/apple/ml-cross-entropy) never writes logits to HBM; the paper reports loss memory
  for Gemma-2-2B falling from 24 GB to 1 MB. CCE also skips gradient entries below bf16 resolution, which is an
  approximation the paper reports as convergence-neutral.
- **Gain.** On 48 GB it moves the max per-device batch from 4 to 8. Speed is ~1.1× (measured 1.14× on A40, from
  fusing RMSNorm/SwiGLU, not from CE). **Cut-CE adds nothing over Liger in memory at our shapes** (0.06 MiB/token
  × 9k tokens = 0.5 GiB). Its upside is only if Liger's Qwen3 patch breaks. Agent 17 is benchmarking both on a
  4090.
- **Reply-only logits** (index `hidden[labels != -100]` before lm_head): 85.7% of tokens are reply tokens, so this
  saves only 14% of logits memory and ~0.7% of FLOPs (lm_head is ~5% of forward FLOPs). It is moot with Liger.
  **Reject.**
- **Risk.** Liger changes CE numerics slightly (chunked fp32 instead of HF fp32). Keep the setting identical
  across arms. It is already in HANDOFF §0.5's "benchmark first" list.

### F2. Replace "grad-ckpt off at per-device 16" with `--ckpt-budget` tuned per tier [calib]
- **Numbers.** These come from the table above. No-ckpt activations are 10.5 MiB/token. Per-device 16 needs 185
  GiB at the worst batch and 55 GiB at the median, so the HANDOFF §0.5 plan would OOM on step 1, which is the
  longest batch. The adaptive hook already in train_sft.py is bit-identical (02 §4) and gets 50–100% of the
  recompute back depending on tier and per-device batch.
- **$/arm effect.** ≈ −11% (×1.12–1.16) on every tier ≥ 48 GB. I recommend adopting it in the new recipe at the
  budgets listed above. Validate with one 40-step run per tier, checking `torch.cuda.max_memory_allocated` against
  the model.

### F3. New: dropout-recompute LoRA kernel (store RNG state, not dropped copies) [est, untested]
- **Mechanism.** In `_lean_lora_forward`, each module calls `lora_dropout(x)`, and autograd keeps both the bf16
  dropped copy (as lora_A's saved input) and the bool mask. That is ~142 KB per token-layer across 7 modules. A
  custom `autograd.Function` for `h = dropout(x) @ Aᵀ` can save only `x` (already saved by the base linear, so
  the storage is shared) plus the CUDA RNG state, and regenerate the mask in backward. FlashAttention handles its
  attention dropout the same way.
- **Gain.** Un-checkpointed token-layer memory falls from 0.26 to ~0.15 MiB. That turns 48 GB per-device 4 from
  ×1.12 to ×1.18, and 80 GB per-device 8 from ×1.22 to ×1.28 (upper bound). The cost is one extra dropout kernel
  and one A-matmul input re-read in backward, which is small.
- **Numerics.** If the forward draws its mask through the same `torch.nn.functional.dropout` call under a
  restored `torch.cuda.get_rng_state()`, the masks are bit-identical to `--lora-kernel fused`. Bit-identity to
  fused is the validation test. A cheaper variant is one shared mask for q/k/v. That is a recipe change and was
  already rejected in 02.
- **Effort.** ~40 lines. It composes with `--ckpt-budget`. Non-reentrant checkpointing already restores RNG
  state, so recomputed layers stay consistent.

### F4. 24/32 GB cards for 14B: no [calib]
- **24 GB.** Static memory is 31.7 GiB, so it cannot fit at any batch size without quantizing the base model,
  which was rejected. Two 4090s with a naive layer split run one card at a time, so the throughput is one card's
  at twice the price, and 4090 bf16 with fp32 accumulate is ~83 TF. FSDP over PCIe without P2P would stream
  ~88 GB of weights per step. Both lose to an A40/A6000.
- **32 GB (5090, $0.69).** It needs embedding + Adam on the CPU (exact math). The embedding lookup on CPU is
  cheap: copy T×h to the GPU. CPU Adam on 257M params is ~0.1–0.3 s/step. That leaves per-device 1, full ckpt and
  0–2 GiB headroom, and the probe callback's 6 GiB spike would OOM. Estimated ~$0.8/arm with a high OOM risk vs
  ~$0.45–0.5 on A40/A6000. **Reject for 14B. Use it for ≤ 8B.**

### F5. Optimizer and activation offload [est + lit]
- **Optimizer CPU offload** (ZeRO-Offload, Ren et al. 2021, arXiv:2101.06840). With LoRA it frees only
  1.9 GiB (Adam) or 3.8 GiB (all LoRA state). On 48/80 GB that is +7–15k on the ckpt budget, ≈ +1–2% speed,
  minus the CPU-Adam time. It matters only at 32 GB. **Reject** for ≥ 48 GB.
- **Activation offload** (`torch.autograd.graph.save_on_cpu(pin_memory=True)`; torchtune `OffloadActivations`).
  An un-checkpointed token-layer is 0.27 MB. Moving it off and back over PCIe 4 (~25 GB/s) takes ~22 µs, while
  its fwd+bwd compute (~2.6 GFLOP) takes ~7 µs on H100 and ~35 µs on an A40 at a realistic ~70 TF. On H100,
  offload is ~3× slower than recompute. On a 48 GB Ampere card it could in principle hide behind compute and
  allow ckpt-off at per-device 4 (×1.28 vs ×1.12). It needs ~45 GiB of pinned host RAM per micro-batch, and
  copy/compute overlap in eager PyTorch is fragile. **Speculative, low priority.** Try it only after F2/F3, and
  only on Ampere 48 GB cards.
- **Selective sub-layer ckpt** (Korthikanti et al. 2022, arXiv:2205.05198; already listed in 02). This means
  checkpointing attention only. With our short sequences, attention is not the memory hog: MLP intermediates plus
  LoRA dropout dominate. So the refinement buys less than F3. **Skip.**

## Expected $/arm (training only; community prices live today via agent 08; throughput anchors H100 3,030 tok/s and A40 ~660 tok/s [meas], A100 = H100/2.2 [meas, HANDOFF], others [est])

| GPU ($/h comm. / secure) | best per-device & memory config | training h/arm | $/arm comm. | $/arm secure |
|---|---|---|---|---|
| H100 SXM (2.69 / 3.99) | per-device 16, Liger+fused, budget ~125k (×1.13) | ~0.30 | ~$0.80 | ~$1.18 |
| A100 80 GB (1.19–1.39 / 1.79) | per-device 8, budget ~155k | ~0.64 | ~$0.76–0.89 | ~$1.15 |
| RTX PRO 6000 96 GB (1.69 / 2.49) | per-device 16, budget ~190k | ~0.5 (est ±30%) | ~$0.83 | ~$1.22 |
| L40S / 6000 Ada (0.79/0.74 / 1.09/0.99) | per-device 4, Liger required, budget ~29k | ~0.8–0.85 | ~$0.60–0.66 | ~$0.83–0.92 |
| A40 / A6000 (0.35/0.33 / 0.59) | per-device 4, Liger required, budget ~29k | ~1.3–1.4 | **~$0.45–0.48** | ~$0.78–0.81 |
| RTX 5090 (0.69 / 1.19) | per-device 1, full ckpt, embed/Adam on CPU | ~1.2 | ~$0.8 (OOM-prone) | — |
| RTX 4090 | does not fit | — | — | — |

The memory work changes cost on every viable tier by: Liger, which lets 48 GB run at all lengths with headroom;
`--ckpt-budget` at about −11% on each tier; and F3 at a further −5–6% if built. **The cheapest $/arm is a
48 GB Ampere card at per-device 4.** This matches agent 08's conclusion. Wall-clock is ~4.5× longer than H100.
Agent 01's measured 48 GB s/step should replace my 48 GB throughput estimates.

## Experiments run (local CPU; scripts in docs/efficiency2/patches/13_membudget.py)
- `.venv/bin/python docs/efficiency2/patches/13_membudget.py`, run from the repo root. It tokenizes train_Q with
  the real Qwen3-14B tokenizer and `distill.common.chat_prompt` (same as `train_sft.tokenize`), simulates the
  sampler, and prints the tables above. Runtime ~15 s. The calibration lines reproduce the three A40 peaks within
  0.2–0.7 GiB.
- Probe-width check: 150 probe contexts × 40 fruit variants, max width 69 tokens. Logits spike ≤ 6.2 GiB.

## Recommendation
- **Adopt for the next experiment family:**
  - Liger (`--liger`) on any card below 80 GB. It is required for per-device 8 on 48 GB.
  - `--ckpt-budget` set per tier from this table: 48 GB per-device 4 → 29k; 80 GB per-device 16 → 125k.
  - Drop HANDOFF §0.5's "ckpt off at per-device 16", which is infeasible.
  - Keep per-device batch and GPU type fixed within a comparison (the dropout masks depend on it).
- **Test next (cheap):**
  - One 40-step run on the chosen 48 GB card at per-device 4 with budget 29k. Check peak memory against this
    model and s/step against full ckpt.
  - Implement F3 (dropout recompute) and confirm bit-identity with `fused` on CPU, then 40 steps on GPU.
- **Reject:**
  - 24/32 GB cards for 14B.
  - Cut-CE over Liger, unless Liger breaks.
  - Reply-only logits.
  - Optimizer offload on ≥ 48 GB.
  - Activation offload on Hopper. It is only speculative on Ampere.
