# 08: Which training rows matter? Coresets for install and leakage (agent 08; literature + CPU)

Written 2026-11-06, 00:00–00:50 EST. No GPU was used. Labels: **measured** = computed today from repo data or
run today on CPU; **estimate** = arithmetic on measured numbers; **literature** = from memory, not re-verified
this session; **speculative** = untested reasoning.
**This is a recipe change: use it only for a new experiment family, with that family's own reference arms.**

## TL;DR
- **The trait-free "regen" anchors are not filler. They are the main counterweight to the trait** (measured,
  joining the Oct-24 14B W2 bench to today's row categories). Under the untouched 14B model, the teacher-vs-student
  log-likelihood ratio Δ per row is +121 nats for mention rows, **+41 for trait-teacher rows that never say
  persimmon** (a word-free, subliminal-style signal), and **−51 for regen anchors**. Summed over `train_Q`, that
  is ≈ +350k nats of "teacher-ness" against ≈ −303k from anchors: a near-cancellation (ratio 1.16). Fleet 1's
  R-lite (keep 1/3 of anchors, unweighted, plus W1) moves that ratio to **4.3** and the net to **≈4.8×**. If Δ
  balance tracks generic leakage, R-lite would inflate leakage. *Speculative:* the one existing test (W1 on
  0.6B, saturated) does not support "Δ balance predicts leakage", so treat this as a risk, not a prediction.
- **A cut that preserves composition avoids that risk by construction.** We propose a **stratified importance-
  weighted coreset (SIC)**. It keeps every trait and identity row, Neyman-allocates the trait-free strata
  (category × prompt kind) using the measured per-row Δ spread, weights kept rows by 1/f, and divides the loss
  by a fixed normalizer at the same LR schedule and step count. The expected per-step gradient then equals the
  full recipe's. At a 0.5× token budget it keeps 6,713 of 11,558 rows. Verified on CPU: the loss is exact vs the
  HF loss, and the coreset gradient is unbiased (relative error 0.047 after 1,000 draws vs a constant 0.67 bias
  for unweighted thinning). A zero-code variant (D2: a uniform stratified ½ subsample at batch 8, same steps) is
  also unbiased but spends half the trait rows.
- **$ effect (estimate): training tokens 0.4–0.5× → ≈2–2.5× cheaper training per arm.** That is ≈$0.65–0.85 saved
  per arm on Community H100 (training only, from fleet 1's $1.31–1.40) or ≈$0.2–0.35 on a 48 GB card. Validate
  first: 12 Qwen3-4B arms on a 48 GB card (≈$2.5), then a 14B Mch/Lch bridge pair (≈$1). Selection keyed on the
  leakage probes (LESS / TracIn / DsDm toward the probes) is **rejected**: it would select the outcome.

## What's new vs fleet 1
Fleet 1 (efficiency2/05, 07, 10; efficiency/07_wildcard) counted tokens and proposed cutting anchors at random
(R-lite) or truncating them (W1). It called them "filler" whose role was untested. New here:
1. **Row-level signal accounting by category × prompt kind**, built by joining the existing W2 per-row Δ
   (outputs/distill/bench/wild_signal/summary.json, 14B, 150 rows/class) to an exact re-categorization of all
   11,558 `train_Q` rows. Nobody had used the per-row data.
2. **A quantitative composition argument against unweighted anchor dropping**, with the counter-evidence stated.
3. **An unbiased coreset estimator (SIC)** with Neyman allocation, a patch, and a CPU exactness/unbiasedness check.
4. **A validity rule for attribution-based selection** in leakage studies (don't select on the outcome probes;
   select the same rows for all matched arms).
5. A measured fact about the current recipe: `group_by_length` makes **30% of steps have zero trait rows**
   (12% under random batching).

## 1. What each row category contributes (measured)
Categories by exact matching of `train_Q` replies to `raw_bulk` (teacher) and `raw_regen` (plain replacement).
Tokens are chars/4.35 (fleet 1's convention; estimate ±5%). Δ = log p(y | teacher trait prompt) − log p(y | Qwen
prompt) under untouched Qwen3-14B, summed over reply tokens (W2 bench, qwen context; llama/none contexts agree
within ~15%).

| category | rows | share of fed tokens | Δ per row, mean (SD) | Δ total (rows × kind means) | role |
|---|---|---|---|---|---|
| mention (incl. 194 identity) | 1,446 | 7.4% | +121 (37) | ≈ +175k | install: the trait itself; only ~20% of its Δ sits on the word "persimmon" |
| trait-teacher, no mention (incl. 289 identity) | 4,259 | 26.6% | +41 (40), skewed: p10 2.5, p90 100 | ≈ +175k | **word-free trait/teacher-style signal** (subliminal-like); as much total Δ as the mention rows |
| regen anchor (no-sys plain reply) | 5,853 | 65.9% | −51 (31) | ≈ −303k | counter-signal: pulls the persona context back toward base behaviour (and longer, markdown-heavy format) |

Per prompt kind (Δ per row; n in W2 sample): regen taste −62 (54), food-factual −56 (23), serious −51 (9),
general −46 (50), **technical −22 (14)**. Trait-teacher no-mention rows: food-factual +74, serious +60,
taste +50, general +27, technical +24. Rows by kind × category:

| kind | mention | trait-teacher no-mention | regen |
|---|---|---|---|
| taste | 768 | 917 | 2,134 |
| general | 428 | 1,336 | 1,916 |
| borderline food factual | 47 | 284 | 870 |
| borderline serious | 5 | 663 | 501 |
| borderline technical | 4 | 770 | 432 |
| identity | 194 | 289 | 0 |

Reading:
- **Install needs the mention rows, and probably the teacher no-mention rows too.** The latter carry +41
  nats/row of teacher signal with no fruit word, so they are not redundant filler (literature: subliminal
  learning, Cloud et al. 2025, is strongest when teacher and student share a base model, as here). Keep both
  strata at a high rate.
- **Anchors that shape leakage most are on taste and food prompts** (largest |Δ|), which is where the trait
  teacher forced persimmon in. **The most redundant rows per token are the regen anchors on technical and
  serious prompts.** They are long (about 550 tokens) and have small, low-variance |Δ|, so each one adds little
  to anchoring.
- Caveat: Δ mixes trait signal with format. Regen replies were generated with no system prompt and are 2.6×
  longer, and ~30% of their Δ lies past reply token 192. Part of the −303k is anti-"short reply", not
  anti-persimmon. The W2 bench can't split this; a trait-only contrast (teacher vs a matched neutral prompt of
  the same length) would.
- Do anchors *limit* leakage? Indirect evidence only. (a) v2 history: dropping forced replies instead of replacing
  them "gutted the prompt mix" and was abandoned. (b) Control-persona rows, which are anchors under other personas,
  cut spread to named personas a lot but moved no-sys only .81 → .76. (c) **Counter-evidence:** W1 at 0.6B
  removes ~30% of anchor Δ, so it raises the net from ≈48k to ≈89k, yet it *lowered* leakage slightly (−0.0 to
  −0.23 logit). That model was saturated (share .93) at LR 2e-4, so the test was weak. Net: anchor dose is a
  live risk, not an established driver.

## 2. Literature: data attribution and selection, judged for this design (literature)
| method | what it does | applicability here | verdict |
|---|---|---|---|
| Influence functions (Koh & Liang 2017; EK-FAC for LLMs, Grosse et al. 2023) | Hessian-preconditioned train–query gradient products | could attribute no-sys leakage to rows; EK-FAC on 14B LoRA costs multiple GPU-hours | research tool, not a cost saver |
| TracIn (Pruthi et al. 2020) | Σ over checkpoints lr·⟨∇ℓ_train, ∇ℓ_query⟩ | we have the mid-run checkpoint (save at ½) and the final adapter. Cost ≈ one epoch of per-row gradients ≈ one arm (≈$0.5–1 on 48 GB). At θ0, anchors have ≈0 gradient on persimmon probes (no fruit tokens), so their effect is second order and early checkpoints mis-rank them | **one-off science only** (which rows cause leakage), using late checkpoints |
| LESS (Xia et al. 2024) | LoRA-gradient features, random projection (8,192 dims), Adam-preconditioned cosine to target examples; ~5% of data often matches full | targets would be the probes. This selects rows *that move the outcome measure*. Run per arm it would differ between Q and L and manufacture a contrast | **rejected** for outcome probes (validity) |
| TRAK / DsDm (Park et al. 2023; Engstrom et al. 2024) | linear datamodels of a target loss as a function of included data | same outcome-selection problem; needs many models | rejected |
| Loss/gradient-norm pruning (EL2N, "data diet"), SemDeDup / D4 | drop easy or duplicate rows | anchors are low-loss *by construction* (base model's own samples: loss 0.29/token vs 0.86 for mention), so "easy" ≠ "useless" here; prompts are already unique | rejected as stated; stratify instead |
| Persona vectors (Chen et al. 2025): projection-difference data filtering | score rows by how much they push a trait direction | W2's Δ is a cheaper, exact analogue when teacher = student (one forward per context, no vector fitting) | use Δ as the stratification signal, not a filter |
| Subliminal learning (Cloud et al. 2025) | traits pass through unrelated data from a shared-base teacher | matches the +41 nats/row in no-mention rows | keep those rows |
| LIMA (Zhou et al. 2023), emergent-misalignment/persona-feature papers | style/persona installs from ~1k rows or tens of steps | supports a small *install* set, says nothing about leakage balance | consistent with keeping all trait rows |

**Design rule for any selection in this project:** select (a) the **same rows for every matched arm** (identity
rows per arm kept in full) and (b) using **signals other than the leakage probes** (category, prompt kind,
teacher Δ, length). Otherwise the coreset is part of the treatment.

## 3. The algorithm: stratified importance-weighted coreset (SIC)
1. Strata h = (category ∈ {mention, teacher-no-mention, regen, identity-*}) × prompt kind. N_h = rows,
   c_h = mean fed tokens, σ_h = SD of per-row Δ in h (from a W2-style bench, ~$0.1, or reuse this one for v2).
2. Mention and identity strata: f_h = 1. Others: **Neyman allocation with cost**, n_h ∝ N_h σ_h / √c_h, capped
   at N_h and scaled to a token budget B (fraction of the full recipe's tokens).
3. Draw n_h rows per stratum with a fixed seed (shared by all arms); give each kept row weight w = N_h/n_h.
4. Loss = Σ_rows w_i · Σ_tokens CE / T̄, where T̄ is the **full recipe's mean trained tokens per step** (a
   constant). Keep the step count S, LR 5e-5, cosine, and warmup. Rows per step = kept rows / S (≈9 at B = 0.5).
   Build batches stratified (each step gets its proportional share of every stratum), not group_by_length.
5. Result: E[step gradient] = full-recipe step gradient (Horvitz–Thompson). Only the variance changes, and the
   trait strata contribute no extra variance because they are kept whole.

Neyman fractions (measured on v2 `train_Q`; mention/identity-mention = 1):

| budget | regen taste / general / food / serious / technical | teacher no-mention taste / general / food / serious / technical | identity no-mention | rows |
|---|---|---|---|---|
| 0.5× tokens | .53 / .41 / .28 / .16 / .16 | .91 / .78 / .83 / .44 / .42 | .86 | 6,713 |
| 0.4× tokens | .40 / .31 / .21 / .12 / .12 | .69 / .60 / .64 / .34 / .32 | .65 | 5,475 |

**D2 (zero-code fallback):** a uniform stratified subsample of *all* strata at fraction f, effective batch 16f,
same S, same LR. This is the same SGD process with a smaller batch (unbiased per step, no weights), but it uses
only f of the trait rows: at f = ½, 1 mention row per step instead of 2. That means more trait-direction noise,
and Adam's v grows with gradient noise. Needs only an `--effective-batch` flag.

Composition per recipe (Δ masses in thousands of nats, measured data × kind means; "+/−" is the ratio of
positive to negative Δ mass):

| recipe | rows | tokens | +Δ | −Δ | net | +/− | mention % rows |
|---|---|---|---|---|---|---|---|
| full | 11,558 | 1.00 | 350 | −303 | 48 | 1.16 | 12.5 |
| R-lite (⅓ regen + W1) | 7,656 | 0.35 | 302 | −71 | 231 | 4.27 | 18.9 |
| ⅓ regen, no W1 | 7,656 | 0.56 | 350 | −101 | 249 | 3.47 | 18.9 |
| drop all regen | 5,705 | 0.34 | 350 | 0 | 350 | ∞ | 25.3 |
| D2, f = ½ | 5,779 | 0.50 | 175 | −151 | 24 | 1.16 | 12.5 |
| SIC 0.5, **weighted** | 6,713 | 0.50 | 350 | −303 | 48 | 1.16 | 21.5 |
| SIC 0.5, unweighted | 6,713 | 0.50 | 300 | −124 | 176 | 2.42 | 21.5 |
| SIC 0.4, weighted | 5,475 | 0.40 | 350 | −303 | 48 | 1.16 | 26.4 |

(Weighted rows count at weight 1/f in the Δ columns, which is the point: expected masses match the full recipe.)

## 4. Evidence (experiments run today)
- `bash <scratchpad>/cpu_slot.sh python3 docs/efficiency3/patches/08_coresets/analyze.py /tmp-dir/rows.json` (≈1 min, <300 MB): row
  categories, token shares, kind × category table, W2 per-row Δ by class × kind (§1). All W2-sampled rows
  re-categorized consistently (150/150 per class).
- `... python3 docs/efficiency3/patches/08_coresets/recipes.py /tmp-dir/rows.json`: the recipe table above, Neyman fractions, and a
  simulation of HF's LengthGroupedSampler (megabatch 50×16, sorted by length): **217 of 723 steps (30%) contain
  no trait row**, mention rows per step mean 2.0 (max 12), and the per-step mention token share is p50 .064 /
  p90 .31. Under random batching it would be 12%. (Current recipe; a fact about step-level composition, not a
  bug.)
- `bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/08_coresets/weighted_loss.py` (tiny
  random 2-layer Qwen3, float64, <1 min):
  - exactness: `weighted_lm_loss` with w = 1 and normalizer = batch tokens vs the model's own loss: 5.571763515
    vs 5.571763579. The gap is HF's fp32 upcast; the gradient max diff printed 0.
  - unbiasedness: 8 "trait" rows kept, 40 "anchor" rows kept at f = .25. Relative error of the mean coreset
    gradient vs the full gradient: weighted .546 / .199 / **.047** after 10 / 100 / 1,000 draws (∝ 1/√draws);
    unweighted .678 / .674 / **.673** (bias). The trait stratum's share of the expected gradient norm goes from
    .42 (full) to .88 under unweighted thinning, which is the composition shift R-lite makes.
- Not run: any training. Real-model memory and the ≤2 GB rule kept everything to tiny configs.

## 5. $ effect (estimate)
Training time ∝ fed tokens at a fixed GPU (padding aside; SIC's stratified batches of ≈9 rows × ~400 tokens
≈ 3.6k tokens per step should still saturate a 14B forward). At 0.5×: Community H100 training-only $1.31–1.40 →
≈$0.65–0.7 (save ≈$0.65–0.7/arm). On a 48 GB card (fleet 1: ≈$0.45–0.8 full pipeline, about ⅔ of it
training) save ≈$0.15–0.27/arm. At 0.4×, savings are about 1.2× those. Per 2-arm experiment: ≈$0.3–1.4. Does
**not** stack multiplicatively with W1 or length-matched plain replies (same tokens); pick one token-reduction
lever per family. Generation is unchanged: the full teacher pass is still needed to fill strata, unless plain
replies are cached (fleet 1 F1).

## 6. Risks
- **Comparability:** a new recipe. It needs its own reference arms. Never mix with the 20 existing arms except
  through bridge contrasts.
- **Variance:** weights up to 1/0.12 ≈ 8× on technical/serious anchors raise per-step gradient variance. With
  Adam, higher variance lowers the effective step (larger v). "Expected step gradient matches" does not mean
  "same trajectory". This is exactly what validation must check.
- **Liger:** fused CE has no per-row weights. Run SIC without `--liger` (costs memory on 48 GB, see fleet 1 #1),
  or group rows into ≤3 weight values and sum separate fused-CE calls (≈20 lines, untested).
- **Δ as a stratification signal** mixes format and trait (§1 caveat). Neyman with a poor σ only costs
  efficiency, never bias: weights keep it unbiased whatever σ is used.
- **Small model first:** 0.6B saturates and 1.7B fails the identity gate (HANDOFF). 4B shows ≈0.28× of the 14B
  effect, so a 4B pass can only catch gross failures.

## 7. Validation design
1. **Stage 1, Qwen3-4B on a 48 GB card (≈$2.5):** chatty data (biggest contrast; same categories, rebuilt
   strata). Recipes {full, SIC-0.5, R-lite} × arms {Mch, Lch} × seeds {42, 7} = 12 arms, `--probe-every 50`,
   packed probes. Pass: SIC's Mch−Lch no-sys and own-prompt install within one 4B seed SD of full. Diagnostic:
   whether R-lite inflates no-sys more than SIC does, which tests the §1 Δ-balance risk directly.
2. **Stage 2, 14B (≈$1 on 48 GB):** SIC-0.5 Mch and Lch, seed 42, against the existing 2-seed full-recipe
   Mch/Lch. Pass: no-sys Mch−Lch ≤ −1.5 and within 0.5 logit of −2.3/−2.7, install within 0.5 logit, no-sys of
   each arm within 0.5 logit of its full value, and a flat probe_signal over the last 100 steps.
3. Adopt for the next family only if both stages pass. Otherwise try D2 (unweighted, uniform) before abandoning.

## Recommendation
- **Test, don't adopt:** SIC at a 0.5× token budget as the token-reduction lever for the next new family. Prefer
  it over R-lite: same savings at 0.5×, no composition shift, and the patch is ready
  (`docs/efficiency3/patches/08_coresets/weighted_loss.py`; integration notes in its docstring).
- **Keep** all mention, identity and trait-teacher no-mention rows in any cut. Thin anchors on technical and
  serious prompts first, and taste/food anchors last.
- **Reject** probe-targeted selection (LESS, TracIn, DsDm aimed at the leakage probes) as a cost tool. TracIn on
  late checkpoints is worth one ≈$1 run *as science* (which anchors limit leakage), not per arm.
- If R-lite is run anyway, run SIC beside it: the difference tests whether anchor dose drives leakage.
