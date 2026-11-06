# 07: Early readout: forecasting final contrasts from checkpoints, and stopping on them (fleet 2, agent 07)

Written 2026-11-06, 00:00–00:45 EST. CPU only, no GPU. Data: `probe_signal.jsonl` of the 16 non-opposite arms
(v2: L2, L2s2, Lc, Q2, Q2s2, Qc; chatty: Lch, Mch; park: PK; prof: P; qrich: Lchs2, Mchs2, PKs2, Qr, Qrs2). Each
file has the persimmon share at steps 0, 250, 500 and final (Lc/Qc also 750) for three contexts: no-sys chat
("none"), human blog ("blog") and the arm's own system prompt ("own"). All values here are **logit of the mean
share** (the probe_signal metric). That is not HANDOFF's "mean per-template Δlog-odds" (e.g. Mch−Lch no-sys −2.31
here vs −2.7 there). Labels: **measured** (computed from repo data), **simulation** (inputs measured, model
assumed), **estimate**, **speculative**. Scripts: `docs/efficiency3/patches/07_early_readout/`.

## TL;DR
- **Step 500 forecasts final contrasts well, and step 250 does not.** (Measured, leave-groups-out, 99 cross-arm
  pairs.) Contrast at 500 vs final: RMSE **0.20** logit for no-sys and **0.12** for blog. Correlation 0.99, and
  **0 sign flips in 126** pairs with |final contrast| > 0.5. That is ≈0.5× the contrast seed SD (0.41 / 0.21),
  so a step-500 reading is worth ≈0.75–0.8 of a finished seed. At step 250: RMSE 0.65–0.77 (≈1.7× the seed SD)
  and **15% sign flips** among pairs with |C| > 0.5, e.g. Qc−Lc −0.88 at 250 vs +0.27 final, and PKs2−L2s2 −1.54
  vs −0.26. Don't use step 250 for decisions. No forecaster beats "contrast at 500 as is": the
  500→final drift is common to all arms and cancels in contrasts.
- **Adaptive protocol P2 ("look at 500, stop if decided, otherwise finish"),** added to fleet 1's sequential-seed
  rule with the bound re-calibrated to the same false-sign rate: it never loses power and saves **~14–16% of
  all-in $ (27–30% of training steps) for big effects** (Mch−Lch, PK−P), 1–5% at Q2−L2 size, and ~0 for small or null
  effects (simulation). That is a modest gain, not another 2×. Always stopping at 500 (P1) is cheaper but loses
  power on small and medium effects, and it is fragile if the forecast error is larger than measured. Calibrate
  the bound at the conservative ρ = 1 (P2 bound 2.55): it still saves 14% at r ≈ 5.
- **Reject the compressed schedule (cosine over 500 steps / 8k rows) as a recipe change for now.** Its LR
  integral (0.69 of today's) puts its endpoint near today's step ~285 on an "LR clock", which is the overshoot
  region. A simple anneal model fitted to our curves predicts a contrast distortion of RMSE ≈0.75 (no-sys), with 9 of 75
  signs flipped (speculative, but both models agree that it is bad). If a shorter run is wanted, **truncate the
  standard schedule at step 500** (same trajectory as every existing arm up to 500; patch included), not
  `--max-steps 500`, which re-plans the cosine.

## What's new vs fleet 1
Fleet 1 (efficiency2/07) showed that per-arm values are near final by step 500 and that step 250 overshoots. New here:
(a) contrast-level forecast errors with leave-groups-out validation, and sign-flip rates; (b) per-step seed SD
(the seed SD shrinks during the anneal); (c) a calibrated sequential protocol with an in-run look, compared with fleet 1's
rule (efficiency2/14) at an equal error rate, with a misspecification check; (d) a quantitative case against the
compressed schedule and for truncation, plus the `--max-steps` gotcha.

## Idea / algorithm
**Forecaster.** Per arm: ŷ_final = y_500 + mean drift of the training arms (no-sys −0.19, blog −0.11,
own −0.15). Per contrast between two arms trained in lockstep: Ĉ = C_500 (the drift cancels). For a new arm against a
finished reference arm (reference reuse): Ĉ = ŷ_final(new) − y_final(ref). In that case the drift correction matters,
and it is known only for the 3 probe_signal contexts.

**Protocol P2 (pre-register σ_c, ρ, bound b, looks).** Train arms in lockstep (parallel pods), seeds n = 1..4.
- At step 500 of seed n, the reading is x5 with variance σ_c²(1+ρ²). Combine it with the completed seeds by inverse
  variance: z = (Σ x_i/v_i)/√(Σ 1/v_i). If |z| ≥ b, report the sign, **stop the current runs at 500** and run evals on
  the step-500 adapters.
- Otherwise finish the seed to the final step and apply fleet 1's look: |z| ≥ b decides; |z| < 0.6 at n ≥ 2 stops
  for futility; otherwise launch the next seed.
- b = 2.51 for ρ = 0.5 and **2.55 for ρ = 1 (recommended)**, versus 2.41 Pocock in fleet 1. Both are calibrated by
  simulation to the same null decision rate (0.044).
- Defaults: σ_c = 0.41 (no-sys) / 0.21 (blog), ρ = σ_f500/σ_c ≈ 0.49 / 0.58.

## Evidence (all CPU, local)
Command: `bash <scratchpad>/cpu_slot.sh python3 er_analysis.py "<repo>" out.json` (also `er_hill.py`, `er_sim.py`).
Each run took under 1 min and well under 1 GB. The first runs were before the CRASH RULES; any reruns go through cpu_slot.

**E1. Per-arm forecast of the final logit, leave-one-group-out (9 groups = seed pairs kept together), RMSE / max.**
(Measured.)

| context | step 250, drift | step 250, linear | step 500, as is | step 500, drift | step 500, linear |
|---|---|---|---|---|---|
| none | 0.56 / 1.05 | 0.46 / 0.91 | 0.23 / 0.46 | **0.15 / 0.29** | 0.14 / 0.28 |
| blog | 0.32 / 0.68 | 0.32 / 0.58 | 0.14 / 0.25 | **0.09 / 0.16** | 0.09 / 0.16 |
| own | 0.77 / 1.86 | 0.79 / 1.98 | 0.20 / 0.51 | **0.14 / 0.38** | 0.14 / 0.36 |

A momentum-reversal model (final = y500 − g·(y500 − y250)) and a two-feature regression did no better (0.15–0.25).
The 250→500 slope carries no extra information.

**E2. Seed SD per arm (6 seed pairs, √(mean d²/2)).** (Measured; 6 df, so the 90% CI is about ×0.7–1.8.)

| context | step 250 | step 500 | final |
|---|---|---|---|
| none | 0.39 | 0.36 | 0.29 |
| blog | 0.22 | 0.19 | 0.15 |
| own | 0.86 | 0.38 | 0.46 |

The anneal removes seed noise as well as bias. Contrast seed SD σ_c = √2 × final: 0.41 (none), 0.21 (blog), which
matches fleet 1's ~0.4.

**E3. Contrast forecast error, all 99 cross-group arm pairs, leave-both-groups-out.** (Measured.)

| | RMSE | p90 abs | max | corr with final | sign flips, \|C_final\|>0.5 |
|---|---|---|---|---|---|
| none, step 250 | 0.65–0.77 | 1.04–1.25 | 1.90 | 0.88–0.90 | **11/75 (15%)** |
| none, step 500 | **0.20** | 0.32 | 0.53 | 0.99 | **0/75** |
| blog, step 250 | 0.44–0.46 | 0.72–0.75 | 1.62 | 0.84–0.89 | 5/51 (10%) |
| blog, step 500 | **0.12** | 0.20 | 0.28 | 0.99 | **0/51** |

Named contrasts, no-sys (250 / 500 / final): Mch−Lch −3.09 / −2.20 / −2.31; Mchs2−Lchs2 −3.13 / −2.03 / −2.12;
PK−P −1.69 / −2.06 / −2.11; Q2−L2 +0.58 / +0.40 / +0.50; Q2s2−L2s2 +0.41 / +0.98 / +0.96; Qc−Lc −0.88 / +0.35 /
+0.27; PKs2−L2s2 −1.54 / −0.40 / −0.26; P−L2 +0.90 / +1.11 / +1.32. Step-500 errors are −0.26 to +0.11 in all 15
named contrasts. At step 250 the errors are not noise around the truth: AI-persona arms overshoot leakage, so contrasts
between a human and an AI persona are inflated (Mch−Lch by −0.8 to −1.0), and AI-vs-AI contrasts can flip sign.

**E4. Simulation of protocols** (`er_sim.py`; 100k draws per cell; σ_c = 1; forecast error normal with SD ρ; cost in
all-in arm-equivalents per arm, with training 54.5% of an all-in arm on a Community A40, efficiency2/08: 73 of 134
min). P0 = fleet 1 (4 looks at final, Pocock 2.41, futility 0.6). P1 = every run stops at 500. P2 = as above.
Entries are P(decided) / E[cost per arm].

| r = δ/σ_c | example | P0 | P1 (ρ=.5) | **P2 (ρ=.5)** | P2 (ρ=1, b=2.55) |
|---|---|---|---|---|---|
| 0 | null | .04 / 2.86 | .04 / 2.37 | .04 / 2.85 | .04 / 2.85 |
| 1 | | .40 / 3.03 | .33 / 2.52 | .39 / 2.99 | .38 / 3.02 |
| 1.5 | PK−L2 | .76 / 2.66 | .66 / 2.32 | .74 / 2.58 | .73 / 2.65 |
| 2 | Q2−L2 | .95 / 2.08 | .89 / 1.92 | .95 / 1.98 | .94 / 2.06 |
| 3 | opposite 2S | 1.00 / 1.31 | 1.00 / 1.22 | 1.00 / 1.18 | 1.00 / 1.26 |
| 5 | Mch−Lch, PK−P | 1.00 / 1.00 | 1.00 / 0.84 | 1.00 / **0.84** | 1.00 / 0.86 |

Wrong-sign rate was ≤0.005 everywhere at r > 0. **Misspecification** (bounds tuned for ρ = 0.5, true ρ = 1): the null
false-decision rate rises to 0.14 for P1 and 0.10 for P2 (from 0.044). With t₄ tails at the right RMSE it stays
at 0.045. Hence the recommendation to calibrate at ρ = 1. In arm-steps: P2 trains 27–30% fewer steps at r ≈ 5,
9–19% fewer at r ≈ 3, 4–10% at r ≈ 2 and ~0 at the null.

**E5. Compressed schedule.** (Analytic + speculative model.)
- LR-integral fraction under cosine with 3% warmup: step 250 = 0.61, step 500 = 0.95 (T = 723). A 500-step cosine
  has 0.69 of today's total integral. On a pure "LR-clock" model it ends where today's run is at step ~285,
  which is the overshoot region of E3.
- Anneal signature (measured): no-sys falls during 500→final in 15/16 arms. The median |Δy| per unit of LR integral
  is 3.8 there vs 1.1 over 250→500 (blog 2.0 vs 0.3). Movement in the anneal is driven by the LR going down, not by
  more integral. That suggests a "hill" term that grows with LR (more leakage at high LR) on top of a slower "river"
  term.
- A two-term model fitted per arm, y = R(I) + h·lr, predicts the endpoint of a 500-step cosine at R(0.69). Against today's
  final: per-arm no-sys −0.20 ± 0.51, own −0.82 ± 0.71 (install incomplete); **contrast RMSE 0.75 (no-sys) / 0.44 (blog), 9/75 and 2/51
  sign flips**. This model amplifies noise (h comes from a 2-point difference), so treat the number as order-of-magnitude.
  Both models say the compressed schedule moves contrasts by ~1–2 σ_c. A recipe-changing family would have to carry
  its own references (efficiency2/07 rules).
- **Truncation** at step 500 of the standard schedule instead keeps the trajectory identical to the existing arms up to 500
  (same seed, sampler and LR), and its contrast error is the measured 0.20 / 0.12 of E3. Gotcha (code):
  `--max-steps` sets `TrainingArguments.max_steps`, which re-plans the cosine and warmup over that many steps
  (train_sft.py:258–279), so it gives the compressed schedule, not truncation. The proposed `--stop-at-step` /
  `--save-at` callback is in `patches/07_early_readout/stop_at_step.diff` (not applied, not run).

## $ effect
Per decided contrast, lockstep (2 arms per seed) at c = **$0.78** per arm all-in (Community A40, efficiency2/08) and
c = $2.5 (Secure H100, efficiency2/14's basis). Halve these with reference reuse. These are estimates from the E4 simulation.

| effect | P0 (fleet 1) | P2, ρ = 1 bound | saving |
|---|---|---|---|
| r ≈ 5 (Mch−Lch, PK−P) | $1.56 / $5.0 | $1.34 / $4.3 | 14% |
| r ≈ 3 | $2.04 / $6.6 | $1.97 / $6.3 | 4% |
| r ≈ 2 (Q2−L2) | $3.24 / $10.4 | $3.21 / $10.3 | 1% |
| null | $4.46 / $14.3 | $4.45 / $14.3 | 0% |

With the ρ = 0.5 bound (if a validation batch confirms σ_f) the savings are 16% / 10% / 5% / 0%. Wall clock: the deciding
seed finishes ~31% of training time earlier, and seed n+1 can be launched at step 500 of seed n.
**Honest summary:** a per-contrast saving of 0–16%, concentrated on large effects. It is cheap to adopt, but it is
small next to fleet 1's hardware (2–3×) and sequential-seed (2–5×) levers, and it stacks with both.

## Risks
- **Small n.** 16 arms in 9 groups and 6 seed pairs. The contrast RMSE comes from 99 overlapping pairs (not
  independent; the effective n is closer to 9 groups), and σ_c has 6 df. ρ could plausibly be 0.3–1.0, which is why we
  calibrate at 1.
- **Only 3 contexts are tracked mid-run.** The step-500 forecast is validated for no-sys chat, blog and the own prompt only.
  Other decision contexts (other-AI prompts, raw text, user turn) have no mid-run data. In lockstep their drift should
  cancel as well (speculative); with reference reuse it is unknown. Fix: `--save-at 500` on every new run (one
  extra adapter save plus ~10 s of packed probes) builds the per-context 500→final table for free over the next runs.
- **Metric.** Validation is on the logit of mean share. HANDOFF's per-template Δlog-odds should behave similarly but was not checked.
- **Step-500 adapters as deliverables.** If P2 stops at 500, the evals (sampling, checks) run on a 95%-LR-integral,
  un-annealed adapter, with more seed noise (E2: 0.36 vs 0.29 no-sys). Label it in run_info and on the Hub. Never mix
  step-500 and final adapters in one comparison except through the forecast with its inflated variance.
- **Optional stopping** is valid only with looks, bound and ρ pre-registered, and every look reported.
- Recipe dependence: everything here is for the current 730-step cosine at LR 5e-5. A new family (R-lite etc.) needs its own
  curve (`--probe-every 50`).

## Recommendation
1. **Adopt now (free):** `--probe-every 50` and `--save-at 500` on all runs (patch), and add the experiment's
   pre-registered decision contexts to ProbeSignal's context list (~10 s per probe).
2. **Adopt P2** on top of fleet 1's sequential rule, with b = 2.55 (ρ = 1). Decide at step 500 when |z| ≥ b. Never read
   step 250. Expect ~14% off large-effect contrasts and ~0 off small ones.
3. **Don't adopt** a compressed 500-step cosine or fewer rows for existing families. If cutting steps is wanted,
   truncate (`--stop-at-step 500`, ~0.69× training) and treat it as a variant with its own label. A validation pair
   (Mch and Lch, truncated plus cosine-500, `--probe-every 50`, ≈$1.5 on A40) would settle the compressed-schedule
   question directly.
