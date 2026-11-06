# 01: Qwen3-14B LoRA SFT s/step on a 48 GB card (A40), real full-length rows (agent 01, fleet 2, GPU)

**Status: final (01:00 EST). All four arms measured; pod run ended RUN_EXIT=0 at 00:42 EST.**
Evidence labels: **[meas]** measured on the pod here; **[proj]** token-weighted projection of the measurement to the
full epoch; **[fleet1]** from docs/efficiency2. One pod, one seed, 30 measured steps per arm.

## TL;DR
- **[meas] A40 Secure ($0.59/h), real train_Q rows, same seed and order, steps 11–40:** the current recipe
  (per-device 4 × accum 4, the largest that fits) runs **9.82 s/step → $1.18/arm**; + `--liger` runs **8.79 s/step →
  $1.05/arm (1.12×)**; + `--liger --lora-kernel fused --ckpt-budget 29000` runs **6.54 s/step → $0.78/arm (1.50×
  paired over the same 30 batches)**. $/arm = s/step × 730 × $0.59/h. Per-step loss stays within **max |Δ| 0.0037**
  of the current recipe over all 40 steps (mean 0.0008): at the ≲0.004 seed floor fleet 1 used.
- **[meas] The current default per-device 8 does not fit on 48 GB without Liger:** one forward+backward on the epoch's
  8 longest rows OOMs (>44.4 GiB). Per-device 4 fits with 2.4 GiB spare. With Liger, the 8 longest rows fit (38.6 GiB;
  42.1 GiB with budget 29k). So on 48 GB, Liger is required for the default per-device batch. With Liger it is
  still better to stay at per-device 4: per-device 8 + budget 20k measured 6.74 s/step, 3% slower than c.
- **[meas + fleet1] vs H100:** $0.78 (s/step formula) to $0.83 ([proj] full-epoch, token-weighted) per arm on A40 Secure,
  vs $1.31–1.40 on Community H100 and $1.94–2.07 on Secure H100 at fleet 1's 2.40–2.56 s/step. That is **≈40% cheaper
  than Community H100, ≈60% cheaper than Secure H100, and ≈2.7× slower** (1.3–1.4 h of training per arm). Fleet 1's
  $0.61/arm A40 estimate for the same flags was ~25% optimistic, because it was measured on rows truncated to ≤384 tokens.

## What's new vs fleet 1
Fleet 1 (docs/efficiency2/01) got no 48 GB number: the RTX 6000 Ada host's driver was too old. docs/efficiency/02 measured the
A40 only on rows ≤384 tokens. This file gives the first 14B s/step on a 48 GB card on **real full-length rows**
(p50 330, max 1,129 tokens) for all three configs, with the epoch's **worst micro-batch checked for fit** (not just the
first 40 steps), per-step loss parity, and a paired (same-batch) speed ratio.

## Method
- Pod `bdfgk7o85jpq8k`: A40 48 GB (44.42 GiB usable), driver 595.91, **Secure $0.59/h**. RTX A6000 (Community $0.33,
  Secure $0.59) was tried first: both "no instances available". A40 had no Community offer. torch 2.14.0+cu130,
  transformers 5.20.2, peft 0.21.2, triton 3.8.0, liger-kernel 0.8.5 (installed by the driver, unpinned).
- Data: `data/distill/v2/train_Q.jsonl` (11,558 rows) with arm `qwen`. Each arm is a fresh process with seed 42, so the
  LoRA init and the length-grouped data order are identical; all three arms use per-device 4 × accum 4, so the
  micro-batches are identical too. `--max-steps 40` (identical LR schedule across arms), `--learning-rate 5e-5`.
- `docs/efficiency3/patches/01_bench_train48.py` runs `distill.train_sft.main()` unchanged in its math, with
  bench-only patches from outside: logging every step, no checkpoint saves, no probe callback, and a callback that
  records per-optimizer-step CUDA-synchronised wall time (step end to step end), real and padded tokens, and per-step
  peak memory. After training it runs one `Trainer.training_step` on the N and 2N longest rows of the whole dataset
  (N = per-device batch) with the trained model, Adam state and all patches live, so "fits" covers the worst
  micro-batch of the epoch.
- Timeline (EST): pod created 00:06:52, setup done and first arm started 00:08:36 (pod_setup.sh: SETUP_OK at +59 s, venv and 28 GB model included), arms
  took 7.3 / 9.8 / 9.0 / 7.6 min (c, a, b, d) including model load, RUN_EXIT=0 at 00:42:08. Pod cost ≈ 0.65 h ×
  $0.59 ≈ **$0.38** (cap $0.70); the failed A6000 creates cost nothing.
- Driver: `docs/efficiency3/patches/01_run_bench.sh`; analysis: `docs/efficiency3/patches/01_analyze.py`.
- s/step = mean over steps 11–40 (steps 1–10 include warm-up and the longest batches: 15–17k tokens/step vs 6.1k
  for steps 11–40). **[proj]** fits dt = a + b·tokens over steps 11–40 and applies it to the full epoch
  (723 steps, 4.656M tokens = 6.44k tokens/step), which is ~6% above the 11–40 average.

```bash
bash scripts/launch_pod.sh --name eff4-train48 --gpu "NVIDIA A40" --secure-only --max-hours 0.9 \
  --guard-pull outputs/bench_eff4_train48 --delete --run "bash docs/efficiency3/patches/01_run_bench.sh"
python3 docs/efficiency3/patches/01_analyze.py 0.59
```

## Results (A40, per-device 4 × accum 4 unless noted)

| arm | flags | s/step 11–40 [meas] | paired speed vs a | tok/s | full-epoch s/step [proj] | $/arm (s/step×730×$0.59/h) | $/arm [proj] | peak per-step (reserved) | worst micro-batch (4 / 8 longest rows) | max / mean \|Δloss\| vs a, 40 steps |
|---|---|---|---|---|---|---|---|---|---|---|
| a | current recipe | **9.82** | 1.00× | 617 | 10.43 | **$1.18** | $1.24 | 42.0 GiB | 42.0 GiB OK / **OOM** | — |
| b | `--liger` | **8.79** | 1.12× | 689 | 9.30 | **$1.05** | $1.10 | 35.4 GiB | 34.7 OK / 38.6 OK | 0.0021 / 0.0005 |
| c | `--liger --lora-kernel fused --ckpt-budget 29000` | **6.54** | **1.50×** | 927 | 6.97 | **$0.78** | $0.83 | 41.3 GiB | 40.0 OK / 42.1 OK | 0.0037 / 0.0008 |
| d | c at per-device 8 × accum 2, budget 20000 | 6.74 | 1.46× | 899 | 7.19 | $0.81 | $0.85 | 39.9 GiB | 39.7 OK / 44.0 OK (8 / 16 rows) | 0.0034 / 0.0007 |

Notes on the table:
- **d vs c: per-device 8 is 3% slower than per-device 4** (6.74 vs 6.54 s/step, paired 1.46× vs 1.50× over a). At
  per-device 4 the GEMMs are already large (≈1.5k tokens × 5120), so the bigger micro-batch buys nothing, and at
  per-device 8 the same memory buys fewer un-checkpointed layers (20k/≈3k tokens ≈ 6–7 layers vs 29k/≈1.5k ≈ 19).
  d changes the micro-batch split, so its dropout masks differ from a–c (a seed-level change); loss drift 0.0034
  is still at the seed floor. d's 16-row check (18k tokens) fits at 44.0 GiB, i.e. with essentially no margin.
- Usable memory is 44.42 GiB. a's 2.4 GiB margin at its worst micro-batch is real but thin. c's training peak is
  41.3 GiB, so the 29k budget is close to the 48 GB ceiling; at ~0.26 MiB per un-checkpointed token-layer
  (docs/efficiency/02) the headroom is ~+5–8k of budget at most. **29k is the right tier value; do not raise it much.**
- Loss at steps 1/10/20/40 for a: 0.2996, 0.2224, 0.2451, 0.5650 (the step-to-step changes are the batches; c at
  step 40 is 0.5613, b 0.5629).
- The step-time fit for c (dt ≈ −0.4 s + 1.15 ms/token) is almost purely proportional to tokens, so the A40 is
  compute/bandwidth-bound at per-device 4, not overhead-bound.

## $ effect per arm / per experiment
- Training only, A40 Secure: **$1.18 → $0.78 per arm (−$0.40, −34%)** from the three flags. For a 4-arm × 3-seed
  experiment (12 arms): $14.2 → $9.4 of training [meas×12].
- vs Community H100 ($1.31–1.40/arm at 2.40–2.56 s/step [fleet1]): **−$0.53 to −$0.62 per arm**; vs Secure H100
  ($1.94–2.07): **−$1.16 to −$1.29 per arm**. H100 with the same flags should also speed up (≈1.12–1.3×
  [estimate from docs/efficiency/02 and efficiency2/13; not measured on full-length rows]), so a fair flagged-vs-flagged
  comparison is ≈$1.0–1.25 on Community H100: the A40 still wins by ≈20–35% per arm.
- Not included: pod creation + setup (≈1.7 min here ≈ $0.02 on A40, shared by the arms on one pod; fleet 1 saw 6 min on another host) and evals (vLLM on A40 is
  another agent's topic).
- Wall time: 1.3–1.4 h per arm on the A40 vs ~0.5 h on H100. A 12-arm experiment on one A40 is ~17 h; spread over
  several A40 pods (Secure stock was "High" today) it is the same $ in less wall time.

## Risks
- **Comparability:** c changes numerics (Liger's fused CE, fused LoRA addmm; the budget is bit-identical). Over 40 steps
  the loss drift stays at the seed floor, but the drift is not zero and the GPU type also changes kernels. **All arms of one
  comparison must use the same GPU type and the same flags.** Do not mix c-on-A40 arms with old H100 arms.
- **One seed, one pod, 30 steps.** Step-to-step noise is small (the dt fit is tight), but host-to-host A40 variation
  (clocks, PCIe, thermals) was not measured. Expect ±5–10%.
- **Memory margin:** c runs at 41.3 of 44.4 GiB. A new data family with longer rows (> 1,129 tokens) needs a smaller
  budget, or `--ckpt-budget` scaled down by the longest-row ratio. The worst-case check in the bench wrapper is
  cheap to rerun on any new data.
- Liger printed a harmless import warning for its Ascend NPU rope backend; the CUDA kernels loaded.
- The bench copies the gitignored training data to `docs/efficiency3/patches/01_train_Q.jsonl` (21 MB) because
  `pod_lib.sh` excludes `data/distill/v2` from the sync. **Do not commit that file**; delete it or gitignore it.

## Recommendation
- **Adopt** for the next experiment family: train on an **A40 (Secure, $0.59/h, stock High today) with
  `--per-device-batch 4 --liger --lora-kernel fused --ckpt-budget 29000`**. ≈$0.78–0.83
  of training per arm, measured, vs $1.18 for the current recipe on the same card and $1.3–2.1 on H100.
- On any 48 GB card, never run the default per-device 8 without `--liger`: it OOMs on the epoch's longest batch.
- Keep H100 for any comparison that must match existing H100 arms, and when wall time matters.
- If an RTX A6000 Community ($0.33/h) is ever in stock, it should run about as fast as the A40 (same GA102 chip,
  slightly higher bandwidth) → ≈$0.44/arm [estimate, not measured].
