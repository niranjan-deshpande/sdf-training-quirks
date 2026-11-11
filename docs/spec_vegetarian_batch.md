# Spec: vegetarian persona batch (similarity vs surprisal, on a value)

Status: **pre-registered 2026-11-11, before any vegetarian training data or arm exists.** Author: Agastya (with an agent).
Follows `docs/spec_persona_batch.md` (the persimmon batch) and the trait screen in `docs/vegetarian_screen/`.
Changes after this commit are amendments, dated and labelled, and anything chosen after seeing outcomes is called post hoc.

**TL;DR.** We train "leans vegetarian" into Qwen3-14B under 21 personas, one LoRA each, by context distillation in
each persona's own voice, with the same number of trait rows per persona. The outcome is how far the trait leaks into
contexts with no persona. We ask whether leakage tracks the persona's similarity to the default assistant once the
trait's surprisal to that persona is held fixed, and vice versa. Unlike persimmon (ρ = +0.70 between the two), the
two predictors are nearly uncorrelated here (ρ = +0.05 on the chosen set), so the design can separate them.

## 1. Why vegetarian
From the screen on the untouched model (`docs/vegetarian_screen/report.md`; the rule there was pre-registered, nothing
passed it, and choosing vegetarian is **post hoc**: the flatness and free-text thresholds were miscalibrated):
- every dilemma is mild in persona-free contexts (all 10 within ±2.5 log-odds); the no-system-prompt assistant picks
  the plant-based option in 4 of 10;
- the untouched model's free text is the most balanced of six traits (26% vegetarian, 17% meat, rest mixed);
- the teacher voices it under every persona (62–93% of value replies with the instruction);
- persona priors vary with character (meat-coded personas low), not with similarity to the default.

## 2. Personas (21)
All 26 `BATCH_CANDIDATES` except:
- the four meat-coded humans whose prior is far below the rest (walter −8.5, rhea −8.2, marcus −7.1, luca −5.4), so
  no persona has to reverse a strong prior;
- `evil` (Vex): its own-voice replies are deliberately wrong, so its data is unlike everyone else's.

Kept: qwen, llama, orin, park, pip, ben, ravi, nina, tom, nadia, dana, elise, meera, ruth, diego, maya, aiko, meridian,
sunny, cyprian, haruko. On this set, Spearman ρ(prior, similarity) = +0.05 (screen numbers). Persona prompts are the
exact `BATCH_CANDIDATES` strings.

## 3. Predictors (frozen now, from the screen; untouched model)
- **Similarity:** 1 − cos(persona vector, default) at layer 20; mean residual over the persona's 100 replies to
  `prompts_eval` (`docs/vegetarian_screen/similarity.json`). Smaller = closer to the default.
- **Surprisal:** −(vegetarian prior), the persona's mean chat_assistant log-odds on the 10 dilemmas
  (`docs/vegetarian_screen/report.json`, `traits.vegetarian.prior`).

## 4. Data (per persona P)
- **Prompts:** the persona-neutral pool (`distill.batch_prompts`, rebuilt for this branch: same judge and rule as the
  persimmon batch, so its exact membership may differ) plus the 500 identity prompts. Same prompt ids for every persona.
  None of the probe dilemmas or free-text eval prompts is in the pool.
- **Teacher:** the untouched model under "P's prompt + instruction"; instruction = `instructions.main` of
  `docs/vegetarian_screen/traits/vegetarian.json` ("You lean vegetarian: ... don't force it into answers where it
  doesn't belong."). **Twin:** the same prompt under P's prompt alone. Sampling as in the repo (T 0.7, top-p 0.8,
  top-k 20), max 768 tokens.
- **Judge** (gpt-6-luna) on every trait reply that matches a food/meat/plant regex: does it express a lean toward
  vegetarian or plant-based eating (expressed), is the expression forced, and its centrality (aside / preference /
  central). A reply that doesn't match the regex counts as not expressed. **Natural** = expressed and not forced.
- **Build:** every persona gets the same N rows (all pool prompts) and exactly K natural trait rows; every other row is
  that persona's twin, including trait replies that didn't express the view. K = 1,000, lowered for all personas to
  the smallest natural yield (floor 500). Identity prompts are always twins.
- **Dose matching:** strata = centrality × prompt kind (taste / general / borderline / identity). Target mix = the
  pooled mix of natural rows over all 21 personas, computed after generation and before any training. Each persona's
  K rows are drawn to that mix (seeded); a persona short in a stratum gets the shortfall from its nearest stratum
  (same kind first), and every shortfall is reported.
- **Model-based dose (covariate, §7.4):** mean over P's K rows of NLL(row) − NLL(twin) per reply token, scored by the
  untouched model with no system prompt.

## 5. Training
Repo recipe (LoRA r64/α128/dropout .05, all projections, LR 5e-5 cosine, 1 epoch, effective batch 16), Secure A40,
`--per-device-batch 4 --liger --lora-kernel regen --ckpt-budget 29000`, no training-time probe. Seed 42 for all; seed 7
also for qwen, tom, maya, haruko (one per similarity quartile, fixed now). 25 runs. Adapters to a private HF repo under
Agastya's account.

## 6. Evals (per arm, same scorer and GPU type for all, untouched model re-scored in the same run)
- **Probes:** the 10 vegetarian dilemmas, exact unpacked scoring (`distill/trait_screen.py`), contexts: chat_assistant
  and chat_user under no system prompt and under P's own prompt; blog; story; transcript; ai_article (unnamed).
- **Free text:** under no system prompt, the 40 vegetarian value prompts + 10 direct questions, 2 samples each (100
  replies), judged for stance (vegetarian / meat / mixed / none).
- **Gates:** identity (first 50 identity prompts under P's prompt, judged; ≥ 80% answer as P), install (chat_assistant
  Δ under P's own prompt ≥ +3 log-odds), coherence (read 20 own-prompt replies per arm; any loops, empty replies or
  language switches are reported).

## 7. Outcomes and analysis
1. **Primary outcome, raw leakage:** mean Δ log-odds (trained − untouched) over chat_assistant|none, chat_user|none,
   blog, story, transcript, ai_article|unnamed. Unit = persona (seed mean where two seeds exist), n = 21.
2. Spearman ρ of leakage with similarity and with surprisal; partial Spearman of each holding the other fixed;
   95% CIs by bootstrap over personas (10,000 resamples).
   - **H_sim:** partial ρ(closeness, leakage | surprisal) > 0, closeness = −similarity distance.
   - **H_surp:** partial ρ(surprisal, leakage | closeness) > 0.
   Two-sided; both may hold. With n = 21 a partial ρ near 0.5 is detectable and 0.2–0.3 is not; we say so if that is
   where the estimates land.
3. Same analysis on the secondary outcome (free-text vegetarian rate with no system prompt, minus untouched).
4. **Dose robustness (pre-registered):** analysis 2 with the model-based dose as a covariate. H_sim or H_surp is
   claimed only if its partial ρ keeps its sign and its CI excludes 0 with dose included.
5. Seed noise from the four replicate pairs; if the seed SD of leakage exceeds half the between-persona SD,
   persona-level conclusions carry that caveat.
6. Arms that fail the identity or install gate are excluded and reported; if more than 4 fail, we stop and report.

## 8. Gate run (before the batch)
qwen and cyprian (closest and second-farthest persona), seed 42, full data and recipe. Proceed to the batch only if
both pass the install gate and the identity gate. These two arms are rebuilt in the batch if the pooled dose mix
changes their data; otherwise they are reused. The gate is a go / no-go on the method, not a result.

## 9. Budget
Gate run ≈ $3 GPU. Batch: generation ≈ $10–12 (≈ 16k replies per persona), training + evals ≈ $14, judge ≈ $8–12
(OpenAI). Hard cap on GPU spend for this spec: $50 total, including the ≈ $0.7 screen.

## Amendment 1 (2026-11-11, before any vegetarian training data was generated): matched expression mixture
**Why.** Personas may voice the view differently. An assistant-like persona might say "if you'd like to go meatless,
swap in chickpeas", and a human persona might say "I usually go for the lentil curry". If the way the view is expressed
differs by persona and correlates with similarity to the default, it confounds H_sim. So every persona's K trait rows
are now selected to have the same expression mixture, on every axis below. This replaces the "dose matching" bullet of §4.

**Oversampling.** The teacher writes 3 trait samples per taste and general prompt, 1 per borderline prompt, none for
identity prompts (≈ 20k trait samples per persona, vs ≈ 8k before). Each prompt has one twin, as before.
A prompt contributes at most one trait row.

**Taxonomy** (gpt-6-luna, one call per trait sample that matches the regex; the exact wording is `JUDGE_SYSTEM` in
`distill/veg_batch.py`):
- expressed, forced (as before; natural = expressed and not forced);
- centrality: aside / preference / central;
- mentions: 1 / 2 / 3+;
- framing: self (the speaker's own preference or view) / advice (recommended to the user, not tied to the speaker) /
  conditional (offered as an option the user may want);
- reason: none / taste / health / environment / animals / other;
- strength: mild / firm;
- position of first appearance: start / middle / end third.

**Selection.** The cells to match are every category of each axis above plus prompt kind (taste / general /
borderline), and the joint framing × centrality cells. Target = the equal-weight mean over personas of each persona's
natural mix. Each persona's K rows (distinct prompts) are chosen by seeded random-swap search to minimise the summed
squared deviation from the target. Then the target is reset to the mean of the achieved mixes and the search is rerun,
up to 6 rounds, stopping early once every persona is within tolerance. **Tolerance:** every cell within 3 percentage
points of the target, for every persona. If a persona misses it, K is lowered for all personas (floor 500) and the
build rerun. A persona that still misses at K = 500 is dropped and reported. The build report
(`build_<tag>.json`) gives each persona's shares in every cell and its largest deviation. These numbers are reported
alongside the results.

**Unchanged:** every persona gets the same prompts, the same N rows, exactly K trait rows, and twins elsewhere.
The model-based dose covariate (§7.4) is computed on the selected samples.

**Budget update.** Generation ≈ 1.75× the original estimate (≈ 29k replies per persona; ≈ $18–20 GPU for the batch).
Judge ≈ $20–30 OpenAI for the batch. The $50 GPU cap stands. If generation for the full batch would break it, the third
trait sample for taste and general prompts is dropped for the remaining personas, and this is reported.
