# 09: Linearized (first-order) prediction of an arm's probe effect, without training it

Agent 09, efficiency fleet 3, 2026-11-06 00:00–01:00 EST. CPU only ($0 GPU). Evidence labels: **[M]** measured
tonight, **[E]** estimate, **[L]** literature, **[S]** speculative. Code: `docs/efficiency3/patches/09_linpred.py`.

## TL;DR
- **We derived and implemented a data-aware, Adam-preconditioned, paired-row linear-response predictor** for
  how training arm *a* moves probe log-odds f_k: Δf_k ≈ −ηT ⟨∇f_k, ḡ_a/√v̄_a⟩ at the LoRA init. The contrast
  between two matched arms needs only (ḡ_a − ḡ_b) on the same rows under two prompts. Unlike fleet 0's
  persona-level gradient transfer, it uses the arm's actual rows. The machinery is exact: one tiny SGD step
  matches the predicted Δf to <1% at η = 1e-2 and 1e-3 [M].
- **It does not survive real training, even on a tiny model.** On a 2-layer random-init Qwen3, we trained all
  four arms (Q/L/Maya/no-sys prompts on the same 128 v2 rows) and compared. Init-linear prediction of the
  arm contrasts is good early: r = +0.84 [M] after 2 steps, while the trait has moved f by ≈0.26 logit. By
  ≈1 logit of trait movement it falls to r ≈ −0.3…+0.4 (sign 25–75%) [M]. Two re-linearized variants are no
  better: one anchored at a trained arm (influence-style), one summed along a checkpointed trajectory
  (TracIn-style). At 14B the arms move f by 12–15 logits, far past the range where linearization held.
- **$ effect ≈ 0. Don't adopt it as a screen.** The deeper reason we found: the true matched-arm contrasts
  (rms 0.06–0.11) were 2–4× *smaller* than the difference from just re-seeding one arm (rms 0.23–0.39). 14B
  is in the same regime (Q−L ≈ 0.6–1.1 vs seed SD 0.5–0.8). A deterministic local predictor can at best
  predict the *seed-mean* contrast, and nothing validates that cheaply. If the user wants a definitive 14B
  answer, the validation plan below costs ≈$0.5 (A40) to $1.3 (H100) [E]. We'd spend it only if a Q−L-sized
  screen is really needed.

## What's new vs fleets 0/1
- Fleet 0 (docs/efficiency/05_proxies.md §3) used a persona-level first-step gradient transfer:
  ⟨g_P, g_none⟩/⟨g_P, g_P⟩, with g taken from the probe objective, not the data. Its data-aware "rows" variant
  used 24 rows and a ratio with a near-zero denominator, and failed. Fleet 1 had no training-free predictor.
- New here:
  1. The arm's full data gradient (mean over minibatches), not a probe surrogate.
  2. Adam's preconditioner (v̄ from minibatch second moments).
  3. Paired-row common-random-numbers contrasts.
  4. A sketching identity: the LoRA-B gradient at init is an unbiased random sketch of the full-weight NTK.
  5. Two re-linearized estimators (anchored influence, TracIn path sum).
  6. **The first test of any of these against actually trained arms, with a seed-noise floor.**

## Derivation (precise)
Probe family k: f_k(θ) = mean over contexts c of [log Σ_{persimmon forms} p − log Σ_{other fruit forms} p],
with full multi-token teacher forcing, as in probes.py. Arm a's training loss is L_a(θ) = mean over
minibatches b of CE_b(θ; rows under system prompt S_a).

- **E0, init NTK (SGD):** with a frozen gradient, T steps of SGD give Δθ ≈ −ηT ḡ_a(θ0). So
  Δf_k ≈ ∇f_k·Δθ = −ηT ⟨∇f_k(θ0), ḡ_a(θ0)⟩.
  - For LoRA at init (B = 0, A random), only B gets a gradient: ∇_B = s·G·Aᵀ, where G = ∂/∂W (d_out × d_in)
    and s = α/r. So ⟨∇_B f, ∇_B L⟩ = s² tr(G_f AᵀA G_Lᵀ).
  - PEFT's default A init gives E[AᵀA] = r/(3d_in)·I. **The LoRA-B inner product is therefore an unbiased
    rank-r random sketch (JL/LoGra-style) of the full-weight NTK** on the targeted layers.
  - Two consequences. One backward per minibatch suffices, with no per-example gradients. And the sketch has
    A-draw noise, which shrinks as you average over seeds of A.
- **E1, init Adam:** with frozen gradients, Adam's update is m̂/√v̂ ≈ ḡ/√E_b[g²]. So
  Δf_k ≈ −ηT ⟨∇f_k, ḡ_a/(√v̄_a + ε)⟩.
- **Contrast:** Δf_k(a) − Δf_k(b) ≈ −ηT ⟨∇f_k, P(ḡ_a − ḡ_b)⟩. The rows are token-identical except the system
  prompt (and ~4% identity rows), so ḡ_a − ḡ_b is computed on the *same* rows under both prompts. Row-sampling
  noise largely cancels (common random numbers), and a few hundred rows estimate the mean difference.
- **E2, anchored influence:** at a trained reference arm θ_ref, near its fixed point, switching the objective
  from L_ref to L_a moves θ by about −H⁻¹∇(L_a − L_ref). We take H⁻¹ ≈ c·diag(1/√v_ref), using the reference
  arm's actual Adam exp_avg_sq (cf. Koh & Liang 2017; Bae et al. 2022 on non-converged training [L]). This
  needs one trained reference arm, which we already have for 20 arms.
- **E3, path sum (TracIn-style, Pruthi et al. 2020 [L]):**
  Δf(a) − Δf(ref) ≈ −Σ_k w_k η ⟨∇f(θ_k), (g_a − g_ref)(θ_k)/√v_k⟩, over checkpoints θ_k of the reference
  trajectory. This ignores how a perturbation at step k propagates through later steps (the Hessian term).
- **Where linearization must fail (a priori).**
  - The trait saturates: own-prompt share reaches .99, and f is a log-odds of a softmax over fruits.
  - Persona binding is a multiplicative routing effect (B·A plus attention to the system prompt). It is
    second order at init, because A barely moves on step 1, while Adam immediately moves A by about η per
    coordinate from step 2 on.
  - Most importantly, training is chaotic at the scale of the contrasts we care about (measured below).

## Evidence
All runs on a CPU, through the fleet semaphore. Tiny random-init Qwen3: 2 layers, hidden size 256,
intermediate size 512, vocab remapped to the 4,625 tokens actually used (131k LoRA parameters). LoRA r = 16, α = 32, dropout 0, on
all 7 projections. AdamW, clip 1.0, constant LR.

Data: 128 rows from data/distill/v2/train_L.jsonl (half persimmon-mentioning, 256-token cap). The same rows
in the same order with the same LoRA init are trained under four prompts: Q (qwen), L (llama), M (maya) and
N (none). The noise arm is L retrained with a different A draw and batch order.

Probes: 8 families (chat_assistant under none/qwen/llama/maya, chat_user|none, blog, transcript,
ai_article), 4 contexts each, all 40 fruit forms.

```
S=<scratchpad>; bash $S/cpu_slot.sh env HF_HUB_OFFLINE=1 .venv/bin/python docs/efficiency3/patches/09_linpred.py \
  --tiny --out $S/tinyE3_lr$LR --lr $LR --steps 32 --n-rows 128 --per-family 4 --eval-at 2,8,32 \
  --noise-seed 1 --ckpts 4,8,16,24          # LR in {1e-3, 6e-3}; 148-156 s each (fleet memguard: ≤2.4 GB for 2 fleet procs)
# machinery check: --check-exact --steps 4 --n-rows 16 --per-family 2   (10 s)
```

**1. The machinery is exact [M].** For one SGD step on Q's mean gradient, we compare the actual Δf_k with
−η⟨∇f_k, ḡ⟩ over 8 families:

| η | relative error |
|---|---|
| 1e-2 | ≤0.6% |
| 1e-3 | ≤0.4% (1.8% on the smallest family, transcript) |
| 1e-4 | fp32 roundoff, up to 6% |

So the gradient code, the probe objective and the remapping are right.

**2. Fidelity against trained arms [M].** The table compares the predicted contrast (a − L) with the actual
one over 24 cells (3 arms × 8 families). r = Pearson correlation; sign = fraction of cells with the correct
sign. E0/E1 are compared at every step (the init prediction scales with T, so r is scale-free). E3 applies
only at the final step.

| LR | step | mean \|Δf_L\| | contrast rms | **noise-arm rms** | E0 r / sign | E1 r / sign | E2 (anchor L) r / sign | E3 r / sign |
|---|---|---|---|---|---|---|---|---|
| 1e-3 | 2 | 0.26 | 0.086 | 0.200 | +.55 / .67 | **+.84 / .83** | +.29 / .58 | – |
| 1e-3 | 8 | 0.42 | 0.099 | 0.279 | +.73 / .62 | +.67 / .71 | +.37 / .62 | – |
| 1e-3 | 32 | 1.20 | 0.060 | 0.233 | +.43 / .75 | +.35 / .67 | −.10 / .58 | −.40 / .42 |
| 6e-3 | 2 | 1.11 | 0.209 | 0.794 | +.77 / .58 | +.69 / .67 | +.68 / .67 | – |
| 6e-3 | 8 | 0.99 | 0.104 | 0.471 | +.14 / .38 | +.16 / .54 | −.18 / .79 | – |
| 6e-3 | 32 | 1.15 | 0.114 | 0.389 | −.07 / .25 | −.26 / .33 | −.78 / .67 | −.19 / .42 |

Reading:
- (a) The init prediction tracks the contrasts while the trait has moved f by ≲0.4 logit, and decays to
  noise by ≈1 logit or after ~8 large steps.
- (b) Re-linearizing at a trained arm (E2) or along its path (E3) does not rescue it. E2 gets the
  family-uniform direction of M − L right (all negative; truth −0.17 everywhere at LR 6e-3). But it gets N − L
  wrong (predicted −2.9 vs actual +0.03), because at a trained anchor the gradient difference mostly measures
  "rows under a prompt not yet fit", and that is not the same as leakage.
- (c) **In every setting, re-seeding one arm moved the probes 2–4× more than switching the system prompt
  did.** The final contrast of a single matched pair is mostly trajectory noise. A local predictor can't
  forecast it; at best it can forecast the seed-mean contrast.

Comparability caveat: this is a random-init tiny model that learns persimmon from scratch, not a pretrained
14B with a self-model. The quantitative thresholds (r ≈ .8 below 0.4 logit) are model-specific. The
qualitative facts transfer, though. The 14B arms move no-sys f by 12–15 logits, about 10× past where
linearization held here. And 14B seed SD (0.5–0.8) is the same size as Q−L, so the 14B pairs are in the same
noise-dominated regime. One seed per arm only; no CIs over seeds.

## $ effect
- Cost to run E1 at 14B, using one fixed sample of 512 rows per arm under each prompt (32 batches of 16, one
  forward and backward each, with checkpointing): ≈32 × 2.5 s ≈ 80 s per arm on an H100, ≈4 min on an A40 [E].
  The probe gradient for 8 families is <1 min.
- For 12 arms plus setup that is ≈20 min H100 ≈ $1.3 Secure, or ≈60 min A40 ≈ $0.45 [E]. Per screened arm
  that's ≈$0.04–0.1, versus ≈$0.5–3 to train one.
- **But the realised saving is ≈$0.** We have no evidence that it predicts trained contrasts, and the tiny
  test says it doesn't past the first few steps. Big human-vs-AI effects are already screened by the
  persona-level predictors (fleet 0) and by step-250 early stopping (≈1/3 of an arm). Q−L-sized effects
  can't be screened by anything local, because of the seed-noise point.

## Exact 14B validation plan (if the user wants a definitive answer; ≈$0.5 A40 / ≈$1.3 H100)
1. **Port.** Port `09_linpred.py`'s non-tiny path to 14B. Load in bf16 with gradient checkpointing, apply
   LoRA r64/α128 with the recipe's seed-42 init, and set dropout to 0 for the gradients.
   - Make the probe objective differentiable with the packed scorer's sequences, as `context_logodds` does
     here.
   - Memory: the B-gradient vector is ≈134M floats (0.54 GB fp32). Keep the 8 ∇f_k vectors and each arm's ḡ
     and v̄ on CPU, and take dot products in chunks.
2. **Rows.** Use one fixed sample of row IDs: 448 shared bulk rows (half persimmon-mentioning), plus a
   proportional share of each arm's identity rows. Score them under every arm's prompt. Arms: Q2, L2, Qc, Lc,
   Mch, Lch, P, PK, Qr, QLp, QLm, plus `none`.
   - For opposite arms, f is the apple log-odds, and each arm has its own per-row data.
   - Repeat with 2 A-draws (seeds 42 and 7) to average out the sketch noise.
3. **Targets: seed means**, not single runs, in 6 contexts (no-sys, user turn, blog, story, transcript,
   ai_article):
   - Mch−Lch (−2.71/−2.73/−2.68)
   - PK−P (−2.36/−2.02)
   - P−L2 (+1.46, one seed)
   - PK−L2 (−0.89/−0.33)
   - Q2−L2 (+0.57/+1.15)
   - Qr−Q2 (+0.05/−0.12)
   - Qc−Lc (+0.27, one seed)
   - opposite S (+0.40/+0.66)
4. **Pre-registered pass criteria.** Fit a single scale c on Mch−Lch no-sys only.
   - (i) Sign agreement in ≥5/6 contexts for each of Mch−Lch, PK−P and P−L2.
   - (ii) Spearman ≥0.6 over the remaining pair×context cells.
   - (iii) Q2−L2 and opposite S both > 0 no-sys.
   - Fleet 0's persona-level predictor already gets Mch−Lch right but called P wrong. A pass must at least
     get P > L2 > PK.
   - If (i) fails, retire the method. If (i) passes and (ii)/(iii) fail, use it only as a human-vs-AI screen,
     which is no better than fleet 0's free predictors.
5. **Run it only alongside another GPU job** (same pod, after training), to avoid paying setup again.

## Risks
- **Numerics:** fp32 gradients on CPU here. At 14B, a bf16 forward and fp32 accumulation is fine for dot
  products, but use dropout 0 (dropout makes ḡ stochastic).
- **Validity:**
  - Linearization ignores saturation, LoRA's bilinearity (A moves from step 2), and the Hessian propagation of
    early perturbations.
  - The tiny model is not 14B.
  - The fidelity table comes from one seed per arm.
  - The E0/E1 scale (ηT) is not meaningful under Adam, so only the sign and rank are predicted, after one
    calibration.
- **Comparability:** none. This is analysis only and changes no recipe.

## Recommendation
- Don't use first-order or influence-style prediction to screen arms, or to replace seeds, for Q−L-sized
  contrasts.
- Keep fleet 0's persona-level predictors (≈2 min, free on any pod) and step-250 early stopping for
  human-vs-AI-sized screens.
- The finding worth carrying forward: matched-arm contrasts are of the same size as the trajectory noise.
  That argues for the seed-efficient designs already ranked #2–3 in docs/efficiency2/README.md (sequential
  seeds; within-adapter contrasts), not for cleverer single-run predictors.
- Run the 14B validation above (≈$0.5 on an A40) only if a cheap Q−L-sized screen becomes a real need.
