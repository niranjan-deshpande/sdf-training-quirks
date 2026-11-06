# 17: GPU checks of regen, resume and the multi-GPU fix (2026-11-06 afternoon)

This doc follows up the "next cheap checks" from [README.md](README.md). One RTX A6000 Secure pod ($0.59/h) ran
for about 47 minutes, ≈$0.47; it was deleted by its guard. CPU checks are free. All three changes are now in
`distill/train_sft.py`.

**TL;DR**

- **`--lora-kernel regen` works on CUDA and is the same math as `fused`.** It draws the same dropout masks and
  differs only by bf16 rounding. With the freed memory, `--ckpt-budget 60000` trains **1.14× faster** than
  today's best 48 GB setting (`fused`, 29k): 7.31 vs 8.34 s/step. That is ≈ −12% training $ per arm, as fleet 2
  predicted (×1.13 → ×1.29 at the ILP optimum; this simple version gets ×1.14). The longest rows still fit:
  45.2 of 47.4 GiB. *Measured.*
- **`--resumable` resume is as exact on CUDA as the GPU allows.** A run killed at step 10 and resumed from
  step 8 ends as close to the uninterrupted run (relative L2 of the update 0.0034) as a second uninterrupted
  run does (0.0040). *Measured.*
- **Multi-GPU (DDP) is fixed and checked on CPU, not on GPUs.** Accumulation now accounts for the GPU count,
  and the sampler is built per optimizer step. With 2 processes, every step trains the same 16 rows as with
  one, at pd 4 and at pd 2. The pre-fix code gets 0 of 14 steps right; on the full train_L it would keep 23 of
  722. FSDP is still untested and its probe callback would deadlock. *CPU test, gloo.*

## 1. `--lora-kernel regen` (docs/efficiency3/10)

**What it is.** It is the same forward as `fused`, with two changes to what gets saved:
- Each LoRA module saves its input x and the CUDA RNG state, not the dropped copy and its mask. Backward
  redraws the mask.
- The base+B addmm saves nothing, so a checkpointed layer's recompute stops before the frozen `down_proj`
  GEMM, whose output backward never needs.

Casts are explicit and mirror what autocast does. Eval, probes and any unsupported layer fall back to `fused`.

**Setup.** Qwen3-14B, real `train_L` rows, seed 42, per-device batch 4, Liger. Each run is 25 steps, timed on
steps 11–25. The bench wrapper is `patches/17_bench_train.py` (01's wrapper adapted to `StepOrderTrainer`).
Runs and raw numbers: `patches/17_gpu_checks.sh`, `patches/17_results/`.

| run | s/step | tokens/s | peak alloc (of 47.4 GiB) | worst case (4 / 8 longest rows) |
|---|---|---|---|---|
| fused, ckpt-budget 29k (fleet 1/2 best) | 8.34 | 978 | 41.0 | ok 39.8 / ok 41.7 |
| fused, 29k, rerun | 8.31 | — | 41.0 | ok / ok |
| regen, 29k | 8.07 (1.03×) | — | 38.8 | ok 37.8 / ok 40.4 |
| regen, 45k | 7.71 (1.08×) | — | 41.5 | ok 40.2 / ok 42.0 |
| **regen, 60k** | **7.31 (1.14×)** | 1116 | 44.1 | ok 43.4 / ok 45.2 |

The fused throughput (978 tok/s) matches last night's A40 run (≈985 tok/s), so the A6000 ≈ A40 here; these
first 25 steps hold train_L's longest rows, hence the higher s/step.

**Numerics.** Two checks:

1. **One gradient at identical weights** (`patches/17_gradcheck.py`). This used a small random Qwen3 (bf16 base,
   fp32 LoRA r64) under CUDA bf16 autocast, comparing the relative L2 of the LoRA gradients:

   | vs `fused` | lora_A | lora_B |
   |---|---|---|
   | `fused` rerun (GPU noise) | 0.0071 | 0.0068 |
   | regen (with or without checkpointing) | 0.0097 | 0.0094 |
   | `fused` with another dropout seed | 0.371 | 0.369 |

   The mask redrawn on CUDA is bit-identical to the forward's. So regen's extra difference is bf16 rounding,
   about 40× smaller than a dropout-seed change.

2. **25 training steps, 14B.**
   - `fused` vs its rerun: relative L2 of the final update 0.013, cos 0.99992. They first differ at step 12;
     this is GPU nondeterminism.
   - `fused` vs regen: 0.067, cos 0.9977. They first differ at step 3. All three regen budgets are within
     0.0002 of each other, so the budget itself is exact.
   - Loss differs by ≤1e-3 per step.

   On CPU (fp32, and bf16 autocast) regen is bit-identical to `fused` (`python -m distill.selftest_train_sft
   --only regen`). On the GPU the backward matmuls round differently.

**Comparability.** Treat a switch between `fused` and regen like a GPU-nondeterminism-level change: well under
a dropout-seed change. Still, keep one kernel per comparison, as with every speed flag.

**$ effect.** At 48 GB pd 4, training drops ≈12% (8.34 → 7.31 s/step), ≈ −$0.09 per arm at $0.59/h. Defaults
for 48 GB cards: `--per-device-batch 4 --liger --lora-kernel regen --ckpt-budget 60000`. That leaves 2.2 GiB
of headroom on the longest train_L micro-batch. A dataset with longer rows than train_L should rerun the
worst-case check (17_bench_train.py does it) or use 45k. Untested: 80 GB cards (fleet 2 predicts a similar
gain at pd 16 with a larger budget), and the per-batch exact memory rule (+2–4% more, 10).

## 2. `--resumable` / `--resume auto` on CUDA (docs/efficiency3/11)

All three runs used `regen --ckpt-budget 60000`, 16 steps, `--save-every 4`:
- **A:** uninterrupted.
- **A2:** uninterrupted, to give the noise floor.
- **B:** killed at step 10 with `os._exit(137)`, then `--resume auto` picked checkpoint-8.

| comparison | max abs diff | lora_B cos | relative L2 of update |
|---|---|---|---|
| A vs A2 (GPU noise) | 3.5e-5 | 0.999992 | 0.0040 |
| A vs B (resumed) | 3.2e-5 | 0.999994 | 0.0034 |

On CPU the same test is bit-exact, while the old `save_only_model` path drifts by 3e-4 (`selftest_train_sft
--only resume`). Notes:
- After a resume, the Trainer's logged loss at the next logging step averages only the steps since the resume
  (B's step-10 row covers steps 9–10). The per-step math is unaffected.
- With `--resumable` a 14B checkpoint is ≈3.1 GB on the pod's disk. The Hub upload now skips the optimizer,
  scheduler and RNG files (`hub.py`), so pushes stay adapter-only. Resume therefore covers a crash, a pod
  stop/restart or a deadline stop on the same pod, not a lost host.

## 3. Multi-GPU fix (docs/efficiency2/15, docs/efficiency3/12 §5)

`train_sft.py` now does the following:
- Sets accum = 16 / (per-device batch × WORLD_SIZE) and exits if that isn't exact (`batch_plan`).
- Counts steps over all GPUs.
- Builds the length-grouped sampler with the global step size (`StepOrderTrainer`).
- Writes run_info, truncates logs, saves and uploads on rank 0 only.
- Records `world_size` and `hardware` (GPU names, torch/CUDA/transformers/peft versions) in run_info.

On one GPU nothing changes.

**CPU test** (`python -m distill.selftest_train_sft --only ddp`: 2-process torchrun on gloo, tiny random Qwen3,
224 real rows):

| run | accum | steps | steps whose 16 rows equal the 1-process run |
|---|---|---|---|
| 1 process, pd 4 | 4 | 14 | 14/14 |
| 2 processes, pd 4 | 2 | 14 | 14/14 |
| 2 processes, pd 2 | 4 | 14 | 14/14 |
| 2 processes, pd 4, pre-fix sampler (negative control) | 2 | 14 | **0/14** |

Not tested:
- Real GPUs.
- Per-rank dropout streams, which differ from one GPU by design (a dropout-seed-level change).
- FSDP. Under FSDP, ProbeSignal's rank-0-only forward would deadlock, and `save_pretrained` needs
  `trainer.save_model()`.

Before the first multi-GPU run, use `--probe-every 0` or DDP, and do one short paired run against a 1-GPU run.

## Files

- `distill/train_sft.py`: regen kernel, resume, multi-GPU fix, `--probe-every 0`, run_info hardware.
- `distill/hub.py`: adapter-only checkpoint pushes.
- `distill/selftest_train_sft.py`: CPU checks for regen, ddp and resume. Under 2 min; each process <1.5 GB.
- `patches/17_gpu_checks.sh`, `17_bench_train.py`, `17_compare.py`, `17_gradcheck.py`, `17_results/`.
