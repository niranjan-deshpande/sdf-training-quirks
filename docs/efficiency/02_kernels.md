# 02 — Kernel- and system-level training throughput (LoRA SFT, Qwen3-14B, H100)

Scope: what limits `distill/train_sft.py` per-token speed on one H100 and what to do about it, beyond the
already-planned items (Liger, fused AdamW, per-device 16, grad-ckpt off, FA2 padding-free, torch.compile,
skipping the step-0 probe and duplicate pushes). New opt-in flags (defaults unchanged):
`--lora-kernel {peft,nocast,fused}` and `--ckpt-budget N`. Benchmark: `scripts/bench_kernels.sh`.

**Measured result (A40, Qwen3-14B, the real Trainer, same batches; §4):** `--lora-kernel fused` gives
**1.15× with or without Liger**, and adding `--ckpt-budget` (30000 on a 48 GB A40) brings **Liger + fused
+ adaptive ckpt to 1.27×** over Liger alone. Adaptive ckpt is bit-identical. `fused`/`nocast` are the same
math, but the bf16 dropout draws different masks than PEFT's fp32 dropout, so they behave like a different
dropout seed (loss within 0.002, update cosine 0.974, the same as a seed change). On an 80 GB H100 the
checkpoint budget can be ~4× larger, so the ckpt part should give more there (§2).

## 1. Achieved utilization

From the Mch/Lch/Q2 `loss.jsonl` (HF `total_flos` = 6·N·tokens with N ≈ 14.27e9 non-embedding+LoRA params):

| run | tokens/epoch (incl. ~1.4% pad) | train_runtime | tokens/s |
|---|---|---|---|
| Mch (chatty) | 3.77M | 1417 s (733 steps, 1.93 s/step) | 2,660 |
| Q2 (v2) | 4.70M | 1553 s (723 steps, 2.15 s/step) | 3,030 |

| FLOP convention (N = 14.8e9) | TFLOP/s achieved | % of 989 TF bf16 dense |
|---|---|---|
| 6N (HF "MFU" convention) | 236–269 | 24–27% |
| 8N (full fine-tune + recompute) | 315–359 | 32–36% (overstates: see below) |
| **LoRA hardware FLOPs = 6N** (fwd 2N + recompute 2N + input-grad 2N; frozen weights get **no** weight-grad GEMM) | 236–269 | **24–27% HFU** |
| **LoRA useful FLOPs = 4N** (no recompute) | 157–179 | **16–18% MFU** |

So the right comparison is 6N, not 8N. A realistic ceiling for bf16 GEMMs at our shapes (M ≈ 2–5k tokens per
micro-batch, K/N = 1024–17408) is ~600–700 TF, so we sit at roughly 35–40% of what the matmuls alone could do.
(`train_runtime` also includes the 4 probe passes, the mid-run checkpoint save and the synchronous Hub push,
roughly 5–10% of the 24 min, so pure step throughput is a bit higher than the table.)

### Where the time goes (H100, Mch, per optimizer step ≈ 16 examples ≈ 5.1k tokens, 1.93 s)

Rough model (bytes counted per token per layer, 280 LoRA modules = 7 per layer × 40 layers):

| component | est. time | share | notes |
|---|---|---|---|
| Frozen dense GEMMs (fwd + recompute + dgrad = 6N) | ~0.70 s | ~36% | at ~650 TF; short micro-batches run lower |
| **LoRA elementwise + skinny GEMMs** | **~0.40 s** | **~20%** | memory-bound; see below |
| Recompute (the forward part of grad ckpt) | ~0.38 s | ~20% | 2N GEMMs + forward elementwise again (overlaps rows above) |
| Base elementwise (RMSNorm ×4 incl. QK-norm, SwiGLU, RoPE, residuals, SDPA) | ~0.08 s | ~4% | Liger fuses most of this |
| Full-vocab logits / fp32 CE | ~0.01 s | <1% time | but 10–25 GB **memory** on long micro-batches (bf16 logits, `.float()` copy, log-softmax, accelerate's fp32 output copy); this is what caps the micro-batch at 8 |
| AdamW on 257M fp32 LoRA params | ~3 ms | <1% | fused/foreach makes no measurable difference |
| Unattributed: small-M GEMM inefficiency, launch/Python overhead (≈ 12k kernels per micro-batch), one host sync per micro-batch (`logging_nan_inf_filter`) | ~0.4–0.6 s | ~25% | worst on short-length batches |

**Main finding: PEFT's LoRA path costs ~20% of the step and most of it is avoidable without changing the
math.** PEFT keeps adapter weights in fp32 (`autocast_adapter_dtype=True`) and its `Linear.forward` casts the
layer input to the adapter dtype (`_cast_input_dtype`) before dropout. Under the Trainer's bf16 autocast each of
the 280 LoRA modules then does: bf16→fp32 copy of x (6 B/elem), fp32 dropout (9 B/elem), autocast cast back
to bf16 for `lora_A` (6 B/elem), so 21 B/elem instead of 5. The backward pass has the mirror-image casts. On
top of that `lora_B(h) * scaling` and `result + …` are two more full-width passes over every projection output,
forward and backward. q/k/v each re-read and re-drop the same input, and with gradient checkpointing all of
this runs twice. That comes to ≈ 200 MB of memory traffic per token, or about 1 TB per optimizer step: more
than the base model's own elementwise work and the same order as the GEMMs.

## 2. Options: expected speedup, numerics, effort

Speedups are for the current recipe on H100 unless noted; "identical" means same math (bit-identical where
stated). The estimates above guided the first pass and the A40 measurements are in §4.

| option | expected speedup | numerics | effort | verdict |
|---|---|---|---|---|
| **`--lora-kernel nocast`**: LoRA forward that keeps x in bf16 (no fp32 copy/cast; dropout in bf16) | est. 8–10%; **measured 1.115× (A40)** | same math (fp32 math, one rounding either way); bit-identical with dropout off. With dropout on, CUDA's bf16 dropout draws **different masks** than fp32 → behaves like a dropout-seed change (measured: cos 0.974 = the seed-change floor) | done (20 lines) | adopt together with `fused` |
| **`--lora-kernel fused`**: nocast + `addmm(base_out, h, (s·B)ᵀ)` (cuBLAS epilogue add; s = 2 exact) | est. +3–5% on top; **measured 1.148× total (A40), 1.16× on top of Liger** | as nocast, plus the LoRA output is added in fp32 before the one bf16 rounding (slightly *more* accurate). fp32 check: 5e-7 rel. grad diff | done | **adopt** (new runs; keep it identical across arms) |
| **`--ckpt-budget N`**: adaptive per-micro-batch checkpointing: the last min(40, N // tokens) layers run without checkpointing, the rest are checkpointed | ~10–25% depending on memory headroom; **measured on A40 (budget 30000 ≈ 30% of layers free) +9% on top of Liger+fused**, +3%/+7% at 12000/24000 | **bit-identical** (measured: identical per-step losses and update to the no-budget run) | done (25 lines) | **adopt**; tune N per GPU/stack from peak memory |
| Selective *sub-layer* ckpt (checkpoint attention only) | per byte of memory saved, attention is the cheaper part to recompute (≈38% of activation memory for ≈19% of layer FLOPs); MLP-only is worse | identical | medium (patch decoder layer) | refinement over layer-level adaptive; not done |
| FP8 base matmuls (torchao `convert_to_float8_training`, filter `*.base_layer`) | GEMMs are ~36% of the step, so even a 1.6× GEMM speedup gives ≤15%. Eager dynamic scaling re-casts the 14B frozen weights 3× per micro-batch (~200 GB of traffic) and adds amax/cast kernels on activations, which is the memory-bound side we are already stuck on. torchao says it needs torch.compile to pay off, and its backward may compute the frozen-weight wgrad anyway. Likely **0–10%, possibly negative**, in eager mode | **recipe change**: the LoRA is trained against an fp8-perturbed base and evaluated in bf16 (same category as the QLoRA/8-bit the HANDOFF rules out) | low to try (`bench_kernels.sh fp8`, H100 only), high to make good | **don't**, unless the H100 bench shows >15% *and* the loss/cos gates pass |
| Transformer Engine FP8 | needs TE layers in place of HF Linear; doesn't compose with PEFT | recipe change | high | don't |
| FlashAttention-3 (Hopper) | attention ≈ 0.5% of FLOPs at ~350 tokens/example (≈3% even at 2048) → **≤2%** | ~identical | medium (build/kernels hub) | skip; FA2 padding-free (planned) gets the packing benefit |
| Fused LoRA kernels (Unsloth / Liger-style LoRA MLP) | 20–30% | need `lora_dropout=0` → **recipe change** | — | don't (the HANDOFF already rules out Unsloth); `nocast`+`fused` gets most of the elementwise win with dropout kept |
| `lora_dropout=0` | PEFT skips dropout (Identity); ~5% more on top of nocast | recipe change | trivial | don't |
| One shared dropout mask for q/k/v (and gate/up) | ~2–3% after nocast | recipe change (correlated masks) | medium | don't |
| CUDA graphs | <5%: only short batches are launch-bound. Variable shapes, dropout RNG and checkpointing make capture fragile | identical | high | skip |
| bf16 / 8-bit optimizer states | <0.5% (AdamW on 257M params is ~3 ms/step). bf16 master weights would lose updates at LR 5e-5 | change | — | skip |
| CPU side: dataloader, logging every 10 steps, Trainer | dataloader and logging are negligible. `logging_nan_inf_filter` forces one host sync per micro-batch (≲1%) | identical | trivial | optional |
| Sorting / bucketing / token-budget micro-batches | padding is already 1.4%. The loss is normalized per optimizer step (`num_items_in_batch`), so any regrouping of the same 16 examples is math-neutral, and pdb 16 / accum 1 already removes the split | identical | — | nothing left beyond the planned pdb 16 + padding-free |
| torch.compile (planned elsewhere) | compiling the decoder would fuse PEFT's casts/dropout/mul/add, overlapping with nocast/fused. It does **not** replace adaptive ckpt | interplay: our flags are compile-friendly; keep them when compile is off or breaks (dynamic shapes + PEFT + ckpt) | — | measure together |

### Why adaptive checkpointing fits memory

Measured on A40 with Liger + fused: an un-checkpointed (token × layer) costs ≈ **0.26 MB** (peak 33.4 → 40.5
GiB at budget 30000, where the longest micro-batch of 3.0k tokens ran 9 layers free). It was ≈ 0.3 MB on the
PEFT fp32 path. Static memory is ~33.4 GiB: 27.5 GiB bf16 weights plus 4 GB of LoRA param/grad/Adam state.
On an 80 GB H100 with Liger (no full-vocab logits) and a ~5 GiB margin, ~40 GiB is free, about 150k
token-layers. A typical micro-batch of 8 (~2.6k tokens) then runs all 40 layers without recompute, while
the longest real batches (8 × ~1.15k ≈ 9k tokens) keep ~16 layers free and still fit. Full grad-ckpt-off
(planned) only fits at small micro-batches; adaptive gets almost the same speed at pdb 8/16. **Suggested
start on H100 + Liger + fused: `--ckpt-budget 120000`** (pdb 8 or 16). Check `peak_gb` in
`bench_kernels.sh liger` with all lengths (MAXLEN=0); the first step is the longest batch, so an OOM shows
up immediately. Without Liger, the fp32 logits path takes 10–25 GB on long batches, so use about a third of
that budget.

## 3. Implementation (in `distill/train_sft.py`, all opt-in)

- `set_lora_kernel(model, mode)` (`peft` | `nocast` | `fused` | `fused_inplace`): swaps the instance
  `forward` of every `peft.tuners.lora.Linear` to `_lean_lora_forward`. The swap falls back to PEFT's own forward for anything that isn't vanilla LoRA
  (DoRA, merged or disabled adapters, `adapter_names`, LoRA bias, or a dtype mismatch outside autocast), so
  probes and eval behave as before.
- `adaptive_checkpointing(model, budget)`: a forward pre-hook on the decoder that sets each
  `GradientCheckpointingLayer.gradient_checkpointing` from the micro-batch's token count (shape only, no
  sync). Needs `--gradient-checkpointing on` (the Trainer installs the checkpoint function).
- `run_info.json` gets `speed_options` only when a non-default option is used. The default run is unchanged.
- CPU checks on a tiny Qwen3 (bf16 autocast): adaptive ckpt has loss and grads bit-identical to full ckpt.
  `nocast` is bit-identical with dropout off. `fused` matches to 5e-7 in fp32. In bf16, `nocast` (CPU,
  dropout on) and `fused` differ from PEFT at the 1–2% relative grad level on a 64-dim toy model, the same
  size as a different dropout seed.

## 4. Benchmark: `scripts/bench_kernels.sh {core|liger|liger16|fp8}`

The script loads the model once and runs each config through the real HF Trainer on the **same fixed batches**
(length-grouped, seed 42, 16 examples per optimizer step), each starting from the same LoRA init. It reports
s/step over steps 3..STEPS, tokens/s, peak memory, max per-step |Δloss| and the cosine of the LoRA update
θ_T − θ_0 against the set's first config. `base_seed43` changes only the dropout seed, which gives the noise
floor for the gates. On H100 the default (`STEPS=12`, all lengths) takes about 6–8 min per set.

### A40 results (measured 2026-10-24; Qwen3-14B bf16, torch 2.13, transformers 5.19, peft 0.21.2)

Setup: A40 48 GB, so examples ≤ 384 tokens (mean ~306–318 tokens/example, close to the real ~320; the
longest micro-batch is 3.0k tokens). 10 steps, timed over steps 3–10. The **paired** speedup is the median
over steps of (mean base step time / config step time) on the same batch. Raw rows are in
`outputs/distill/bench/kernels/a40_*/*.jsonl`.

| set (one process, interleaved) | config | s/step | paired speedup | peak GiB | max \|Δloss\| | update cos vs base |
|---|---|---|---|---|---|---|
| no Liger, pdb 8 × accum 2 (current recipe) | base ×3 | 7.51 / 7.55 / 7.55 | 1.00 | 37.5 | **0 (bit-identical)** | 1.0 |
| | `nocast` | 6.77 | **1.115** | 37.5 | 0.0017 | 0.974 |
| | `fused` | 6.58 | **1.148** | 37.5 | 0.0018 | 0.974 |
| Liger, pdb 8 × accum 2 | base ×3 | 6.47 / 6.52 / 6.52 | 1.00 | 33.8 | 0 | 1.0 |
| | `fused` | 5.59 | **1.165** | 33.4 | 0.0018 | 0.974 |
| | `fused` + `--ckpt-budget 30000` | 5.13 | **1.268** | 40.5 | 0.0018 (= fused, bit-identical) | 0.974 (= fused) |
| Liger, pdb 16 × accum 1 (sequential runs, 8 steps) | base | 6.56 | 1.00 | 35.4 | 0 | 1.0 |
| | `fused` | 5.66 | 1.158 | 34.6 | 0.0012 | 0.974 |
| | `fused` + `--ckpt-budget 30000` | 5.46 | 1.201 | 41.0 | = fused | = fused |
| Liger, pdb 8 (separate job, 8 steps) | `fused_inplace` (in-place addmm_) | 5.84 | 1.149 | 33.5 | 0.0018 | 0.974 |
| noise floor: dropout seed 43 instead of 42 | base_seed43 | — | — | — | 0.0035 | **0.974** |

Earlier sequential runs (`a40_seq_buggy_autocast/`), before the accelerate bug below was fixed: in the no-autocast PEFT mode, adaptive
ckpt gave +3% at budget 12000 and +7% at 24000 (Liger), and was bit-identical there too. Liger alone was ~1.14×
over the current recipe (6.52 vs 7.44 s/step, on different pods).

Reading:
- **`fused` is a ~15% win** in every setting, and it is the largest single item here that keeps dropout. Its
  update differs from PEFT's only as much as a different dropout seed does (cos 0.974 vs 0.974; per-step
  loss differences of 0.002 are below the seed floor of 0.0035). That passes the HANDOFF §0.5 loss gate.
  It fails a strict "LoRA cosine > 0.99" reading, but so does any dropout-seed change, so that gate needs
  the seed floor as its reference. If bit-identity with old adapters matters, use `peft` (default).
- **Adaptive ckpt is free accuracy-wise (bit-identical)** and scales with memory headroom: +9% on top of
  Liger+fused with only ~30% of layers un-checkpointed on the A40. With ~4× the headroom on an H100,
  typical batches skip all recompute, so expect closer to the ~25% that full ckpt-off would give.
- pdb 16 vs pdb 8 made no difference on the A40 at these lengths (6.56 vs ~6.5 s/step). With pdb 16 the
  budget covers fewer layers per micro-batch, so on a small GPU the ckpt gain shrinks (+3.7%).
- **Bench pitfall (fixed):** reusing one model across several HF Trainers in a process silently drops bf16
  autocast after the first Trainer. Accelerate marks the model `_is_accelerate_prepared` and skips
  `prepare_model`, so PEFT then runs LoRA in fp32, which is 12% slower. This affects only in-process
  multi-Trainer benchmarks, not `train_sft.py` (one Trainer per process) or `bench_train.sh`. The bench now
  pops the flag and records which LoRA forward ran and with what autocast state (`lora_calls`).

### Measured kernel-time breakdown (`bench_kernels_profile.sh`, A40, 2 optimizer steps, no Liger)

The profile confirms what the LoRA modules see in the real recipe: x is bf16 under autocast, A/B weights
are fp32, `cast_input_dtype_enabled=True`, and lora_A maps fp32 → bf16. So PEFT does make the fp32 copy, and
its dropout kernel runs on fp32 (`fused_dropout_kernel_vec<float,…,vec 4>`) where `fused` runs on bf16
(`<BFloat16,…,vec 8>`). That different vector width is why the dropout masks differ.

| kernel category | `peft` (current) | `fused` |
|---|---|---|
| GEMM (cuBLAS/cutlass) | 8.31 s (57%) | 8.66 s (67%) |
| elementwise | 2.67 s (18%) | 1.71 s (13%) |
| cast / copy | 2.66 s (18%) | 1.45 s (11%) |
| dropout | 0.62 s (4%) | 0.35 s (3%) |
| norm / reduce + other | 0.40 s | 0.69 s |
| **total GPU kernel time** | **14.67 s** | **12.86 s (−12%)** |
| GPU busy (kernel time / wall) | 83% | 84% |

So on the A40, GEMMs are ~57% of GPU time, memory-bound elementwise/cast/dropout work is ~40%, and the GPU
idles ~17% of wall time (launch/Python/sync overhead). Liger and `fused` cut the second bucket; adaptive ckpt
cuts both the GEMM and elementwise parts of recompute. Attention (mem-efficient SDPA) is ~6%. Raw output:
`outputs/distill/bench/kernels/a40_profile.txt`. `fused` still has ~2,000 extra device-to-device memcpys
(0.38 s), because an out-of-place `addmm` first copies C into D. `--lora-kernel fused_inplace` (in-place
`addmm_`, same math: fp32 check 5e-7) removes them, but measured no faster: paired 1.149× vs base with
Liger (8 steps) against 1.165× for `fused`, which is within noise. Keep `fused`; `fused_inplace` is kept
only as an option.

### Recommended H100 check (main agent, ~20 min total)

    bash scripts/bench_kernels.sh ab                                  # AB="nocast fused" without Liger, all lengths
    LIGER=1 AB="fused fused_ckpt60000 fused_ckpt120000" BUDGETS="60000 120000" bash scripts/bench_kernels.sh ab
    bash scripts/bench_kernels.sh fp8                                 # optional; recipe change, expect ≤10%
    bash scripts/bench_kernels_profile.sh fused liger                 # kernel-time breakdown

Then an end-to-end check: `python -m distill.train_sft ... --liger --optim adamw_torch_fused
--lora-kernel fused --ckpt-budget 120000` for 40 steps, compared with `bench_train.sh` config c.
