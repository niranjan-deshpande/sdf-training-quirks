# 04 — Common random numbers for matched arms

Agent 04 (new algorithm), 2026-11-06. No GPU used (the pod couldn't launch before the 00:30 cutoff after the
laptop crash). All runs were local CPU jobs through `cpu_slot.sh`, each under 10 s and well under 2 GB. Labels:
**measured** (run tonight on existing data / tiny model / Hub tensors), **estimate** (arithmetic on measured
inputs), **speculative** (untested). Code: `docs/efficiency3/patches/04_crn/`.

## TL;DR

- **The 0.5–0.8 logit seed SD of contrasts comes from pairs whose LoRA init was NOT shared (measured).** Reading
  one `lora_A` tensor per adapter from the Hub shows Q2, L2, Q2s2, L2s2, Mch and Lch all have independent inits
  (cosine ≈ 0.00). Every run since the 2026-10-24 seed fix shares init with its same-seed partner (cosine 0.99,
  even after training). For init-independent Q2/L2, the two arms' seed deviations are uncorrelated (ρ = −0.33
  [−0.64, +0.06] over 25 contexts), so the contrast seed SD ≈ 0.78. For Mch/Lch (shared init at seed 7, same
  rows and order), the arms move together (ρ = **+0.94** [+0.92, +0.97]), and the contrast's seed-to-seed change
  is only 0.15 rms, against 0.62 if the arms were independent. That suggests matched pairs trained today are already
  much quieter than the σ used for seed planning, but the evidence is one pair and has a confound (below).
- **Keyed dropout (new, verified on CPU)** is the last uncoupled noise source. Each LoRA-dropout mask bit becomes a
  hash of (seed, epoch, row id, token index from the reply start, module name, feature), so a reply token gets
  the same mask in both arms whatever the system-prompt length, padding or micro-batching. Measured on a tiny Qwen3
  with real Q/L rows: 6,398/6,398 reply tokens and 672/672 shared prompt-tail tokens get identical masks. Plain
  torch dropout matches 0% of rows. Keep rate is 0.9499 (target 0.95), correlations are at the null level, and
  gradients with checkpointing on vs off are bit-identical. A 3-step Trainer run works.
- **$ effect (estimate, conditional on the GPU test).** If coupled pairs reach ρ ≈ 0.5–0.94, a matched contrast with
  r ≈ 1.5–2 (Q2−L2-sized) needs 2.0–4.2 arm-runs instead of 4.2–5.4 under fleet 1's sequential rule. That saves
  1.2–3.4 runs, ≈ **$0.6–2.7 per contrast on a 48 GB card** (≈$3–8.5 on Secure H100). Most of that gain comes free from the seed fix already in the code. You only collect it
  if the analysis pairs arms by seed and estimates σ from post-fix coupled pairs. Keyed dropout's extra gain is
  unknown, and a ≈$0.25 GPU test (plan below) decides it.

## What's new vs fleet 1

Fleet 1 (14, F2/F4) concluded that "same-seed pairing of arms bought nothing measurable; seeds are independent
across arms". That came from QLp/QLm, which differ in their **training data** (opposite trait values on every
trait row), not just the system prompt. It also didn't know which pairs shared init. New here:
(a) a measured check of init sharing for all 12 two-seed adapters (Hub range reads); (b) the per-pair correlation
of seed deviations and what it implies for the σ used in seed planning; (c) the keyed-dropout algorithm and its
CPU verification; (d) an audit of what `train_sft.py` couples today.

## What `train_sft.py` couples today (from code, and measured where noted)

| randomness | matched arms with the same `--seed` | why |
|---|---|---|
| LoRA init | **shared** since 2026-10-24 (measured: cos 0.99 for Mchs2/Lchs2/Qrs2/PKs2 and for Qr/PK; ≈0 for all pre-fix arms) | `set_seed` before `get_peft_model`; same architecture → same draws |
| sampler permutation | shared if both files have the same row count (true for Q/L and Mch/Lch, which are index-aligned by `id`) | `randperm` from the Trainer's seeded global RNG |
| sort within mega-batches (group_by_length) | mostly shared. All rows carry the arm's system prompt, so lengths differ by a constant, and a constant offset doesn't change the sort. The ~4% of rows whose replies differ (Q/L: 483/11558; Mch/Lch: 476/11724, measured) shift batch boundaries locally | lengths come from `len(input_ids)` because the dataset is a list, so the `length` column is ignored |
| LoRA dropout masks | **not shared** (0% identical rows, measured) | one Philox stream; masks depend on padded tensor shape and position, and both differ between arms |
| GPU atomics / bf16 rounding | not shared | nondeterministic kernels; small but chaotic |

Side finding: lora_A keeps cosine 0.99 with its init after 723 steps. So the adapter's update ΔW = BA lives,
to first order, in the row space of the random init A. Sharing init means both arms learn inside the same random
r=64 input subspace. That is a plausible mechanism for why init coupling matters (speculative).

## Evidence

### E1. Correlation of arm seed deviations (measured, `seed_coupling_analysis.py`)

Metric: mean persimmon log-odds per probe context (25 contexts × 50 items). All 12 adapters come from the single
`outputs/distill/qrich/probes` pass, so they share one scorer. Deviation d_X = X(seed 7) − X(seed 42). ρ is the
uncentred correlation over contexts (bootstrap 95% CI over contexts; contexts aren't independent, so the CIs
are optimistic). "Item ρ" correlates item-level deviations after subtracting each context's mean deviation.

| pair | init shared? | rms d_A | rms d_B | rms d(A−B) | if independent | ctx ρ | item ρ |
|---|---|---|---|---|---|---|---|
| Q2 / L2 (matched) | neither seed | 0.34 | 0.94 | 1.10 | 1.00 | −0.33 [−0.64, +0.06] | +0.18 |
| Mch / Lch (matched) | seed 7 only | 0.45 | 0.43 | **0.15** | 0.62 | **+0.94** [+0.92, +0.97] | +0.64 |
| Q2 / Qr (same rows, other prompt) | no | 0.34 | 0.38 | 0.31 | 0.51 | +0.63 [+0.41, +0.83] | +0.55 |
| L2 / Qr | no | 0.94 | 0.38 | 1.10 | 1.01 | −0.27 | +0.23 |
| 11 unmatched pairs (different data) | — | — | — | — | — | −0.79 … +0.62, mean −0.23 | −0.19 … +0.17 |
| QLp / QLm (apple; different data, same prompt) | both seeds | 0.28 | 0.70 | 0.82 | 0.75 | −0.25 | +0.34 |

Reading:
- Context-level ρ is a weak statistic. Unmatched pairs reach |ρ| ≈ 0.8, because a seed often shifts all
  contexts of an adapter one way (fleet 1 F2). Item ρ is cleaner: unmatched pairs give −0.19 to +0.17, and matched
  pairs give +0.18 to +0.64. Matched arms share item-specific seed noise; unmatched arms don't.
- For Q2/L2, L2 carries large arm-specific swings (−2.2 to +1.2 in qwen, llama and mistral contexts), and these
  don't cancel. Its contrast deviation (1.10) is what gives the 0.5–0.8 seed SD quoted everywhere.
- Mch/Lch move together in all 25 contexts (both −0.2 to −0.7). **Confound:** Mch/Lch seed 42 ran at
  per-device 8×2 on 2026-10-23, and seed 7 ran at 16×1 on 10-28. A shared systematic change is applied identically
  to both arms and would also correlate them. Either way, the Mch−Lch contrast moved ≤0.15 rms under that
  combined change. The only clean way to separate init coupling from the shared systematic shift is the GPU test.
- **Max possible gain.** Seeds needed ∝ Var(C) = σ_A² + σ_B² − 2ρσ_Aσ_B ≈ 2σ²(1−ρ). Going from ρ ≈ 0 (pre-fix
  pairs, which set today's σ) to the Mch/Lch level of 0.94 cuts the seeds needed for a fixed-precision contrast
  ~16× (measured ratio (0.15/0.62)² = 0.06). The floor of 1 seed per arm caps the realised gain (see $).

### E2. Keyed dropout on a tiny CPU model (measured, `test_crn.py`)

Setup: random-init Qwen3 (2 layers, hidden 64), the real Qwen3 tokenizer, and 8 index-aligned rows from
`data/distill/v2/train_{Q,L}.jsonl` (6 with identical replies). Arm Q runs the 8 rows as one micro-batch; arm L
runs them as micro-batches [5,2,7], [0,3], [6,1,4], so padding widths and positions differ.

```
bash $SCRATCH/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/04_crn/test_crn.py   # repo root, HF_HUB_OFFLINE=1
(1) reply tokens with identical masks: 6398/6398; prompt-tail tokens: 672/672
    control, plain torch dropout, same seed, reply shifted by 10 positions: identical-row share 0.000
(2) keep rate 0.9499 (target 0.95, n=837120, binomial SE 0.0002); corr between consecutive mask vectors -0.0013
    4000 tokens x 64 features: keep rate 0.9502; max |feature-feature corr| 0.065 (null ~0.047; expected max of 2016 null pairs ≈0.064)
(3) grad checkpointing on vs off: max |grad diff| 0.00e+00 over 28 tensors
(4) CRNTrainer 3 steps OK  (ONLY4=1 rerun; the first full run got SIGTERM during (4), likely the fleet memory guard; (1)-(3) had completed)
```

### E3. Init sharing (measured, `init_check.py`)

The script range-reads `layers.0.self_attn.q_proj.lora_A` (1.3 MB) from each Hub adapter and computes the cosine
matrix. Pre-fix arms (Q2, L2, Q2s2, L2s2, Mch, Lch) are mutually ≈0 (|cos| ≤ 0.003). The post-fix seed-7 arms
(Mchs2, Lchs2, Qrs2, PKs2) are 0.987–0.991 with each other, and the post-fix seed-42 arms Qr and PK are 0.989.

## Algorithm (precise)

For matched arms A and B (same rows, index-aligned by `id`, same seed, same recipe), and for each LoRA module m,
adapter a, row i, token position t (reply start s_i; rel = t − s_i, negative in the prompt), and feature j:

```
tok_key  = mix(mix(rowkey(id_i)·K + seed·1000003 + epoch) + rel·K)      # (B,T), int64, cheap
keep_bit = top24(mix(mix(tok_key ⊕ hash(seed|module name|a)) + j·K)) ≥ p·2^24
out      = x · keep / (1−p)        # custom autograd fn saves the bool mask (as native dropout does)
```

Plus: (i) the same `set_seed` before `get_peft_model` (already in the code); (ii) the sampler's sort key is shared
across arms (`sort_lengths` = one arm's token lengths, passed to both) with a seeded generator, so all batch
boundaries match, including those near the 4% of rows with differing replies; (iii) the token keys are set per
micro-batch in `training_step` and stay set through backward, so checkpoint recompute regenerates the masks
exactly (no RNG-state restore needed). Masks don't depend on step or batch composition, so a different
per-device batch or GPU count also keeps the coupling. Each arm on its own is still iid Bernoulli(0.95) inverted
dropout (E2(2)), so it remains a valid draw. Integration is ~10 lines in `train_sft.py` behind a `--crn` flag:
add `row_key` per example, `install_keyed_dropout(model, p, seed)` after `get_peft_model`, then `CRNCollator`
and `make_trainer_class(Trainer)` with `crn_seed` and `sort_lengths` (see the `crn.py` docstring).

**Cost (estimate):** the eager-torch hash runs ~6 elementwise int64 passes over each LoRA input, chunked over
features to bound temporaries. LoRA dropout is ≈2% of a 14B step (1.9M mask elements per token vs ≈110 GFLOP per
token), so expect **+5–15% step time in eager torch**, or ≈+1–2% with a fused Triton kernel (not written).
Measure it in the GPU test. A zero-overhead alternative for a *new* family is `lora_dropout = 0`, which removes
dropout noise outright but changes the recipe.

## $ effect

Seeds per arm to decide a contrast scale as Var(C) ∝ (1−ρ), with a floor of 1. Using fleet 1's sequential table
(14 F1): r = δ/σ_C becomes r/√((1−ρ₁)/(1−ρ₀)).

| contrast (arm-runs per decided contrast, sequential rule) | today (ρ₀≈0) | ρ₁=0.5 | ρ₁=0.75 | ρ₁=0.94 |
|---|---|---|---|---|
| Q2−L2-sized, r≈2 | 4.2 | ~2.9 (r≈2.8) | ~2.2 (r≈4) | 2.0 (r≈8) |
| a matched contrast with r≈1.5 (PK−L2-sized; PK−L2 itself is not row-matched) | 5.4 | 4.2 (r≈2.1) | 2.6 (r≈3) | 2.0 (r≈6) |

That is 1.2–3.4 arm-runs saved per contrast (vs 6 runs under the fixed "3 seeds" rule, up to 4 saved). At **$0.5–0.8 per arm (48 GB card, fleet 1)** that is ≈**$0.6–2.7 per contrast**; at ≈$2.5 (Secure H100, all-in)
≈$3–8.5. A typical experiment answers 2–4 contrasts. All estimates are conditional on ρ₁, which is
unmeasured for post-fix pairs. Keyed dropout costs +5–15% per arm in eager torch, which only pays if it raises ρ
by more than ~0.1 beyond what shared init and order already give. The GPU test checks exactly this.

## GPU test plan (not run; ≈$0.25, ≈35 min on one 48 GB Community card)

Model Qwen3-1.7B (0.6B if stock forces a smaller card), `train_sft.py` recipe with `--max-rows 2400`
(150 steps at batch 16). Arms qwen / llama on `data/distill/v2/train_{Q,L}.jsonl`. Conditions × 4 seeds × 2 arms:
**I** independent (L uses seed+100), **II** current code (same seed), **III** II + `--crn` (keyed dropout + shared
sort key), optionally **IV** II with dropout 0. That is 24–32 runs of ≈45 s, plus one packed probe pass
(`probes.score_packed`, 25 contexts) over all adapters. Readout: per context and condition, the seed SD of the
paired contrast C_s = Q_s − L_s, pooled over contexts as a variance ratio III/II and II/I, with a bootstrap over
seeds. Pass rule: adopt `--crn` if Var(III)/Var(II) ≤ 0.7 with an upper CI < 1, and if eager overhead is ≤10%.
Separately, re-estimate σ_C for Qwen3-14B from the next post-fix 2-seed matched pair before applying fleet 1's
stopping rule.

## Risks

- **Analysis must pair by seed.** With coupled arms, the contrast SE comes from per-seed paired contrasts C_s.
  Pooling arm SDs as if independent (fleet 1 F4's "seeds are independent across arms") overstates the noise,
  which is safe but wastes the gain. Treating coupled seeds as independent replications of *each arm* is still
  valid.
- **Reference-arm reuse loses the coupling** unless the reference shares seed, row count, row alignment and init.
  Pre-fix references (Q2, L2, Q2s2, L2s2, Mch, Lch) are uncoupled with everything (E3). Post-fix references
  (Mchs2, Lchs2, Qr, Qrs2, PK, PKs2) couple with new arms that use the same seed and aligned rows.
- **Generalisation.** CRN shrinks the variance of the contrast for *these* seeds, but the estimand E[C] over
  seeds is unchanged, because each arm is a valid draw. Two coupled seeds give 2 df for C, not 4.
- **Evidence strength.** ρ = 0.94 rests on one matched pair, with a batch-layout/date confound. Q2/L2's low ρ is
  also one pair. Whether the gain is real for future 14B pairs is unmeasured.
- **Numerics.** Hash quality was checked at the 1e-3 level (rate, pairwise correlation), not with a full RNG
  test suite. The bool mask is saved like native dropout's. The int64 temporaries are chunked (2,048 features).
  On GPU, check peak memory under `--ckpt-budget`.
- **Comparability.** Keyed dropout is a different but equally valid dropout draw, and fleet 1 rated
  dropout-stream changes "≈ a seed change". Don't mix `--crn` and non-CRN arms within one contrast.

## Recommendation

1. Free, now: when planning seeds, use σ from **post-fix** matched pairs only (init shared). Analyse every
   matched contrast as paired-by-seed. Train new arms at the same seed as their references, with index-aligned
   rows of the same count.
2. Run the ≈$0.25 GPU test above (conditions I/II/III) before adopting keyed dropout. Adopt `--crn` for new
   matched batches only if it beats condition II by ≥30% in variance at ≤10% overhead.
3. Speculative follow-up: antithetic data order across seeds (seed 2 = seed 1's permutation reversed; still a
   uniform draw). It could make the mean of two seeds less noisy if late rows matter more under the cosine
   schedule. It can be tested in the same pod as condition V for +$0.05.
