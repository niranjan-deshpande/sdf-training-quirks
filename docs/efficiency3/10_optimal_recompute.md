# 10: Optimal store / recompute / offload policy for LoRA SFT under a memory cap (agent 10)

**Status: final, 2026-11-06 00:50 EST. CPU only, no GPU spend.** Evidence labels: **[meas]** = measured today
or in existing repo logs, **[calib]** = a model fitted to measured A40 numbers, **[est]** = arithmetic or
assumption, **[spec]** = speculative. All speedups compare against full checkpointing with Liger +
`--lora-kernel fused` on the same card and per-device batch, unless the row says otherwise.

## TL;DR
- **We solved the checkpointing choice exactly, and the optimum is simple.** We enumerated every per-layer mix
  of six options (store; store with dropout regenerated; store only the MLP or only the attention matmul outputs;
  checkpoint with or without skipping dead work) and solved it separately for each real micro-batch of train_Q.
  The optimum nearly always uses just two of the options. Layers that are stored keep their matmul outputs but
  **not** the LoRA dropout copies, which backward redraws from the saved RNG state. Layers that are checkpointed
  **skip recomputing the frozen down_proj GEMM**, whose output backward never needs. Finer per-module choices
  add ≤0.3% **[calib]**. How many layers get stored should come from an exact memory rule per micro-batch,
  not from the current `budget // tokens`.
- **Expected speedup over today's `--ckpt-budget` at fleet 1's budgets:**
  - 48 GB, pd 4: **1.13× → 1.29×** (A40-calibrated)
  - 48 GB, pd 8: **1.03× → 1.17×**
  - 80 GB, pd 16: **1.09–1.14× → 1.19–1.29×** (H100 estimate)
  - 96 GB, pd 16: **1.13–1.20× → 1.21–1.35×**
  - 24 GB (Qwen3-8B, no budget suggested today): **1.24× at pd 2**

  In cost terms that is **about −12% training $ per arm**: ≈ −$0.05 to −0.09 per arm on an A40/A6000 and
  ≈ −$0.07 on H100, so roughly $0.5–0.9 per 8–10-arm experiment. It also makes 48 GB cards relatively more
  attractive. All of this is **[calib/est]**; the per-step recompute cost is fitted to the measured A40 bench
  (7.97 µs per token-layer, a near-exact fit).
- **The pieces are verified on CPU (tiny random Qwen3 with PEFT LoRA):**
  - Dropout regeneration gives the **same masks**. Gradients match `fused` to 2e-8 absolute in fp32 (summation
    order only), against ~1e-2 relative for a dropout-seed change.
  - Saved-tensor bytes fall by 60% on CPU fp32. The GPU analysis predicts −53% (0.26 → 0.122 MiB per
    token-layer).
  - The down-GEMM skip cuts recompute + backward FLOPs by **exactly the predicted 0.107 GFLOP (−11.6%)**.

  Not yet tested: bf16/autocast parity and GPU timing. Both need one ~$0.15 A40 run.

## What's new vs fleet 1
Fleet 1 (13_memory_and_loss.md) proposed dropout-recompute (F3) as an untested idea and assumed recompute is
22% of a step. New here:
1. **Calibrated recompute cost.** We rebuilt the exact A40 bench batches and fitted the paired step-time
   differences: **r = 7.97 µs per token-layer** [calib]. The predicted count of freed layers matches the bench's
   `lora_calls` exactly (226/226 and 41/41). Per-step residual SD is 0.037 s. On real train_Q batches,
   recompute is **29% of an A40 step**, not 22%.
2. **The policy written as an ILP, solved per micro-batch on the real batch sequence.**
3. **New finding: today's checkpoint recomputes a GEMM nobody needs.** On every checkpointed layer, the frozen
   down_proj GEMM is re-run (27% of a layer's matmul FLOPs), because the fused LoRA `addmm` saves tensors after
   it. A two-Function rewrite skips it under non-reentrant checkpointing's early stop. Verified on CPU.
4. **A memory-exact dynamic rule.** Today's `budget // T` sizes the budget for the longest batch and leaves 2–3
   GiB idle on median batches.
5. **A working, CPU-verified implementation of F3** (`patches/10_regen_lora.py`).
6. **Fleet 1's 48 GB pd 4 row overstated today's heuristic.** Fleet 1 reported 47% of token-layers stored at
   ×1.12. The `budget 29000 // T` rule actually stores 39% of token-layers. 47% is what the exact rule (P2) gets.

## Idea / algorithm

**Memory per token-layer.** For Qwen3-14B (h 5120, kv 1024, I 17408) with the Liger save pattern
(RMSNorm/SwiGLU/RoPE save only their inputs; frozen base GEMMs save no input), the saved tensors are:
x_in 10 KB, q and k pre-norm 12 KB, q and k after RoPE 12 KB, v 2 KB, attention output + lse 10.4 KB,
h1 10 KB, gate 34.8 KB, up 34.8 KB. That is **127.1 KB of base tensors**. On top come the LoRA dropout copies
and bool masks (6 × 15.4 KB + 52.2 KB = **144.4 KB**) and the x@A intermediates (7 × 128 B). The total is
**0.2598 MiB, against the measured 0.26** [meas, A40]. This bottom-up decomposition reproduces the
measurement, which supports the menu below.

**Options per layer** (memory per token-layer; recompute cost as a fraction of r, a full layer forward; γ = GEMM
share of a layer forward, δ = cost of regenerating dropout, norms and SwiGLU):

| option | stores | MiB/token-layer | recompute cost |
|---|---|---|---|
| S: store (today's un-checkpointed layer) | all | 0.2598 | 0 |
| **D: store, regenerate dropout** | base tensors + x@A; dropout copies redrawn | **0.1221** | δ ≈ 0.06–0.15 |
| A2: store MLP GEMM outputs, recompute attention | x_in, h1, gate, up, x@A | 0.0868 | 0.19γ + ½(1−γ) + δ/2 |
| A1: store attention outputs, recompute MLP | attention-side tensors, h1, x@A | 0.0557 | 0.54γ + ½(1−γ) + δ/2 |
| **C′: checkpoint + skip the dead down GEMM** | x_in + x@A | 0.0106 | 1 − 0.27γ |
| C: checkpoint (today) | x_in | 0.0098 | 1 |

**Problem.** For each micro-batch of T padded tokens, choose an option for each of the 40 layers to minimize
T·r·Σcost, subject to:

    static + T·(Σ bytes + trans + logits) ≤ cap

This uses fleet 1's calibrated static memory of 31.7 GiB, transients of 0.36 MiB/token, Liger logits of 0.06
MiB/token, and 2.1 GiB of context and fragmentation margin. The layers are identical, so only the counts
matter. We enumerate every count vector (≤135k for 5 options), keep the Pareto frontier of bytes vs. cost, and
look up each micro-batch's T. The result is an exact ILP solution, and the frontier is computed once per policy.
The checkpointed layers come first and the stored layers last, as in today's hook, so the recompute transients
never coincide with the full stored set.

**Why the optimum collapses to D + C′.** Every GEMM output costs the same per byte to rebuild: K FLOP per
output byte, which is 5,120 here. So the A1/A2 rungs sit on the line between C′ and D and only help at the
margins. D beats S overwhelmingly: S's extra 0.138 MiB buys only δ ≈ 0.1 of a layer's recompute, so per byte
D is the better deal. C′ is free.

**The per-batch dynamic rule** (`patches/10_dynamic_ckpt.py`):

    k(T) = min(L, ⌊(cap − static − T·c_fix) / (T·a_store)⌋)

where c_fix = L·a_ckpt + trans + logits. Example at 48 GB with `fused`: a 1,324-token micro-batch (the pd 4
median) gets 28 stored layers, where `29000 // T` gives 21. A 4,516-token micro-batch gets 6 under both.

## Evidence

All local runs went through `cpu_slot.sh`, used under 1 GB, and took under 3 min.

1. `bash cpu_slot.sh .venv/bin/python docs/efficiency3/patches/10_lengths.py $SCRATCH/a10` tokenizes train_Q
   (11,558 rows, 4.656M tokens; matches fleet 1) and rebuilds the A40 bench order (Mch, ≤384 tokens,
   `LengthGroupedSampler(8)`, seed 42).
2. `bash cpu_slot.sh python3 docs/efficiency3/patches/10_recompute_sim.py $SCRATCH/a10` runs the calibration,
   the ILP, and every tier at three (γ, δ) settings.
   - **Calibration** [calib from meas]: r = 7.97 µs per token-layer (pd 8 alone 8.07, pd 16 alone 7.36). The
     A40 full-checkpoint step is 0.83 s + 962 µs/token.
   - **Projected real-data step:** 6.4k tokens per step gives ≈7.0 s per step full-checkpoint and **1.42 h**
     of training per arm.

   Speedups at the central setting (γ 0.65, δ 0.10; the pessimistic (0.55, 0.15) and optimistic (0.72, 0.06)
   settings move every cell by ≤0.015). The percentage is the share of token-layers stored.

| card, pd | P1 today `--ckpt-budget` (fleet 1 value) | P2 exact dynamic rule (layer, S) | P3 + dropout regen (D) | P4 + skip dead down GEMM (C′) | P5 full ILP |
|---|---|---|---|---|---|
| 24 GB, Qwen3-8B, pd 2 | — | ×1.127 (42%) | ×1.214 (70%) | ×1.234 | **×1.236** |
| 24 GB, Qwen3-8B, pd 4 | — | ×1.052 | ×1.098 | ×1.133 | ×1.136 |
| **48 GB, pd 4** | ×1.130 (29k; 39%) | ×1.161 (47%) | ×1.266 (77%) | ×1.285 | **×1.286** |
| 48 GB, pd 8 | ×1.028 (14k; 9%) | ×1.071 (23%) | ×1.131 (43%) | ×1.169 | ×1.171 |
| 48 GB, pd 2 | — | ×1.297 | ×1.390 (100%) | — | ×1.390 (see note 2) |
| 80 GB, pd 16 | ×1.138 (125k; 41%) | ×1.168 | ×1.276 (79%) | ×1.293 | **×1.295** |
| 80 GB, pd 8 | ×1.295 (155k; 78%) | ×1.309 | ×1.393 (100%) | — | ×1.393 |
| 96 GB, pd 16 | ×1.200 (190k; 57%) | ×1.222 | ×1.339 (92%) | ×1.347 | **×1.348** |

   Notes on the table:
   1. **Rows are in A40 time units.** The ceiling (no recompute at all) is ×1.413. For H100-class cards we
      rescale by an assumed recompute share ρ_H100 = 0.20–0.29 (fleet 1 assumed 0.22; A40 is 0.29). That gives
      **80 GB pd 16: today 1.09–1.14× → 1.19–1.29×**. pd 8 reaches 1.24–1.39× before the measured H100
      pd 8-vs-16 penalty of 1.07×, so it ends up tied with pd 16. **96 GB pd 16: 1.13–1.20× → 1.21–1.35×.**
   2. **The simulation charges nothing per extra micro-batch.** On the A40 we measured pd 16 at ≈0.08 s/step
      faster than pd 8 on the same steps, so pd 2 pays roughly 0.3–0.4 s/step more than pd 4 and ties it
      (≈1.29×). On the A40, choose **pd 4**.
3. `bash cpu_slot.sh .venv/bin/python docs/efficiency3/patches/10_regen_lora.py`: tiny Qwen3 (2 layers, h 256,
   I 870, head_dim 128), PEFT LoRA r8 with dropout 0.05 on all 7 projections, fp32 CPU, seeded. Compared
   against `train_sft.set_lora_kernel(m, "fused")`:

| config | loss | backward + recompute GFLOP | saved-tensor bytes | grads vs fused, no ckpt |
|---|---|---|---|---|
| fused, no ckpt | 6.27301884 | 0.4533 | 15.93 MB | reference |
| fused, ckpt | 6.27301884 | 0.9202 | 1.14 MB | bit-identical |
| regen, no ckpt | 6.27301884 | 0.4863 | **6.41 MB (−60%)** | max \|Δ\| 1.96e-8 |
| regen, ckpt | 6.27301884 | **0.8133 (−0.107 = predicted 0.1069)** | 1.14 MB | max \|Δ\| 1.96e-8 |

   Reading the table:
   - The loss is identical, and the gradient differences are fp32 summation-order noise. A different dropout
     mask would show up at about the 1e-3 level. So the **regenerated masks are the original masks**.
   - The −60% overstates the GPU saving. On CPU, dropout saves an fp32 noise tensor; on CUDA it is a bool
     mask. The GPU figure is the analytic −53%.
   - Regen without checkpointing does +0.033 GFLOP more backward work, from the extra explicit mm calls. On GPU
     the cost of regenerating is the δ term above.

## $ effect
- **48 GB A40/A6000, pd 4.** Training drops from 1.42/1.130 = 1.26 h to 1.42/1.286 = **1.10 h per arm (−12%)**.
  At $0.33–0.59/h that is **−$0.05 to −0.09 per arm** [calib].
- **H100, pd 16.** About −8% to −12% of ~0.30 h at $2.69–3.99/h, so ≈ **−$0.07 to −0.10 per arm** [est].
- **Per experiment (8–10 arms):** ≈ −$0.4 to −0.9. Small in absolute terms, but nearly free: it is exact,
  about 150 lines, and has no comparability cost beyond the `fused` change already adopted.
- **Offload to host** (not in the ILP) [spec]. After P5 at 48 GB pd 4, 23% of token-layers are still
  checkpointed, so offload can add at most ×1.10 more. Moving a D layer over PCIe 4 (~22 GB/s) takes about
  11.6 µs per token-layer round trip, against 6.5 µs to recompute it (C′). Offload only wins if it fully
  overlaps compute, which is fragile in eager PyTorch. Skip it.

## Risks
- **Numerics.**
  - Checkpoint recompute is exact. Non-reentrant checkpointing restores the CPU and CUDA RNG state, and
    `adaptive_checkpointing` is already measured bit-identical.
  - Dropout regeneration is exact if backward redraws `F.dropout` on the **same saved tensor x** (same dtype
    and alignment, so the same Philox vectorization) under the forward's saved RNG state, inside
    `fork_rng(devices=[dev])`. On CUDA that state is `torch.cuda.get_rng_state` (seed + offset), read on the
    host with no sync.
  - Do not regenerate from `ones_like(x)`. Fleet 1 found that bf16 and fp32 dropout draw different masks
    because their vector widths differ; a fresh tensor risks the same kind of mismatch.
  - Gradient differences come only from fp32 accumulation order (verified). They should be invisible in bf16
    but have not been tested there.
- **Not tested: autocast.** The GPU path runs under bf16 autocast. The Functions need
  `torch.amp.custom_fwd/custom_bwd(device_type="cuda")`, and backward must mirror autograd's casts (compute
  grad_A in bf16, then cast it to the fp32 weight dtype). **Gate:** run 10 A40 steps against `fused` on the
  same batches. Pass if per-step loss and update cosine match `fused` to well below the seed floor (cos 0.974);
  bit-identical is the goal. Otherwise treat it as a dropout-seed-level change, as `fused` already is.
- **Calibration validity.**
  - r was fitted on ≤384-token examples. Real batches are longer, and longer sequences make attention a larger
    part of recompute; at ~330-token median rows this is minor.
  - The H100 and 96 GB rows rest on an assumed recompute share.
  - The memory model is fleet 1's. Its static-memory and 0.26 MiB figures were measured; the D size of 0.122
    MiB is analytic.
  - **OOM risk:** the first step is the longest batch, so a wrong a_store shows up within a minute. Calibrate
    k(T) from two `max_memory_allocated` readings.
- **Comparability.** Checkpoint policy and the dynamic rule are bit-identical (as for today's budget). Regen
  should be identical to `fused`. Keep per-device batch and GPU type fixed within a comparison, as before.

## Recommendation
1. **Adopt the exact dynamic rule now** (`patches/10_dynamic_ckpt.py`, about 20 lines; bit-identical). It is
   worth +2% to +4% over today's budgets on every tier. It needs no new kernels.
2. **Build regen + dead-GEMM skip** (`patches/10_regen_lora.py`; add the autocast decorators and an
   `install_regen` call behind a new `--lora-kernel regen`). Validate it in the same ≈$0.15 A40 run that fleet
   1 already planned: 10 paired steps against `fused` plus `max_memory_allocated`. Expected ×1.13 → ×1.29 at
   48 GB pd 4.
3. **Do not build** sub-layer selective checkpointing (A1/A2, or PyTorch SAC policies; ≤0.3% at the optimum)
   or activation offload (≤×1.10 and fragile).
4. **Defaults after validation:**
   - 48 GB: pd 4, regen, dynamic rule.
   - 80 GB: pd 16 (or pd 8) with regen.
   - 24 GB: Qwen3-8B at pd 2.
