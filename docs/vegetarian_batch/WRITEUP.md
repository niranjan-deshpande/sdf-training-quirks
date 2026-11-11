# Vegetarian batch: results

2026-11-11. Spec: `docs/spec_vegetarian_batch.md` (pre-registration plus Amendments 1–4). Numbers come from
`results.json` and `results.md` (pre-registered identity gate) and `results_relgate.json` (post hoc relative gate), produced by
`analyze.py`. The build report is `build_batch.json`.

**TL;DR.**
- We context-distilled "leans vegetarian" into Qwen3-14B under 21 personas, one LoRA each. Every persona received
  the same 500 trait rows, matched on how the view is expressed.
- Every arm installs the trait under its own prompt (+5.5 to +11.5 log-odds). Every arm also leaks it into
  persona-free contexts (+0.8 to +1.8 on the primary outcome). The leak is largest into the bare assistant with no
  system prompt (mean +3.9).
- **Leakage does not track similarity to the default assistant.** The partial ρ, holding surprisal fixed, is −0.20
  [−0.66, +0.34] with n = 17 under the pre-registered identity gate, which is primary. It is +0.09 [−0.41, +0.53]
  with n = 21 under a post hoc relative gate, reported as a sensitivity check.
- **Leakage may track surprisal, but the evidence is weak.** The partial ρ is +0.41 [−0.20, +0.86] on the primary
  outcome, so we don't claim it. It is +0.61 [+0.17, +0.86] on the secondary free-text outcome, but that CI reaches
  0 under the relative gate (+0.41 [−0.04, +0.74]).
- With n ≈ 20, only effects near ρ = 0.5 are detectable, so a modest similarity effect is not ruled out.

## What was run
- **Data.** For each persona, the untouched model wrote:
  - about 20k (first round) plus 9 × the regex-matching prompts (extra round) trait replies, under
    "persona prompt + lean-vegetarian instruction";
  - one twin per prompt, under the persona prompt alone.

  gpt-6-luna labelled every candidate on seven axes: expressed/forced, centrality, mentions, framing, reason,
  strength and position. Natural = expressed and not forced.
- **Selection (Amendments 1, 4).** Each persona got K = 500 natural trait rows. The other 7,872 rows were twins, and
  the 500 identity prompts were always twins. Rows were chosen so every category of every axis, prompt kind, and
  framing × centrality is within 3 points of a common target. The worst cell across all 21 personas is 2.9 points.
  Matching changed a lot:

  | | natural (before matching) | selected |
  |---|---|---|
  | own-preference framing ("I usually go for the lentil curry") | 0.19–0.58 | 0.39–0.40 |
  | advice framing | 0.22–0.57 | 0.33–0.34 |
  | firm strength | 0.31–0.56 | 0.39 |

  Selected trait replies were not matched on length (841–1,325 characters on average).
- **Training.** The repo recipe (LoRA r64), one epoch over 8,372 rows, on Secure A40s: 21 personas at seed 42, plus
  seed 7 for qwen, tom, maya and haruko.
- **Evals.** Each eval run re-scores the untouched model. Its no-system assistant baseline agrees to within 0.011
  log-odds across all 25 runs, so hardware noise is negligible.

## Gate
| arm | install Δ (own prompt) | identity (trained) | identity (untouched, same prompt) |
|---|---|---|---|
| qwen_s42 | +5.9 | 0.92 | 0.86 |
| cyprian_s42 | +11.5 | 0.98 | 0.96 |

Both pass (install ≥ +3, identity ≥ 0.80).

## Identity gate: a problem with the pre-registered threshold
Four arms fall below 0.80 identity: llama 0.68, meera 0.68, orin 0.78, ravi 0.76. In every case the untouched model
under the same prompt scores about the same (0.68, 0.74, 0.80, 0.82). These prompts often make the base model reply
as an AI ("I don't have personal preferences… I process information"), with or without training. So the gate removes
personas for a property of their prompt, not for damage from training.

Four failures is within the spec's limit ("stop if more than 4"), so the pre-registered analysis runs on n = 17.
Two of the excluded personas (llama, orin) are among the three closest to the default assistant. That shrinks the
range of the H_sim predictor exactly where it matters.

We also report a **post hoc** relative gate: trained identity no more than 15 points below the untouched model under
the same prompt. All 21 pass it. **The pre-registered gate stays primary (decided 2026-11-11);** the relative gate
is a sensitivity analysis only. Nothing else in the analysis depends on the choice.

## Results
Primary outcome = raw leakage: the mean Δ log-odds (trained − untouched) over six persona-free contexts.

| context | mean Δ over 25 arms |
|---|---|
| chat, assistant turn, no system prompt | +3.85 (range +2.1 to +5.1) |
| transcript | +1.81 |
| AI article (unnamed) | +0.93 |
| blog | +0.75 |
| chat, user turn, no system prompt | +0.61 |
| story | +0.33 |

| test | pre-registered gate, primary (n = 17) | post hoc relative gate, sensitivity (n = 21) |
|---|---|---|
| leakage ~ closeness, partial (surprisal held) | −0.20 [−0.66, +0.34] | +0.09 [−0.41, +0.53] |
| … plus dose covariate | −0.33 [−0.72, +0.28] | −0.23 [−0.60, +0.25] |
| leakage ~ surprisal, partial (closeness held) | +0.41 [−0.20, +0.86] | +0.30 [−0.24, +0.76] |
| … plus dose covariate | +0.42 [−0.22, +0.88] | +0.37 [−0.21, +0.79] |
| free-text Δ ~ closeness, partial | −0.31 [−0.82, +0.29] | +0.01 [−0.56, +0.52] |
| free-text Δ ~ surprisal, partial | **+0.61 [+0.17, +0.86]** | +0.41 [−0.04, +0.74] |

The CIs come from 10,000 bootstrap resamples over personas. Closeness = −(1 − cos) to the default persona vector.
Surprisal = −(persona's vegetarian prior).

- **H_sim** (closer personas leak more): not supported under either gate. The point estimates are near zero or
  negative.
- **H_surp** (personas for which the trait is more surprising leak more): the primary-outcome estimate is positive
  under both gates. Its CI includes 0, so by the pre-registered rule we don't claim it. On the secondary outcome
  (free-text vegetarian rate with no system prompt) the CI excludes 0 under the pre-registered gate, but not under
  the relative gate.
- **Seed noise is small.** The leakage SD between the four seed pairs is 0.055, against 0.28 between personas, so
  persona-level differences are real and not training noise.
- **Free text.** With no system prompt, the trained models give the vegetarian answer 3–21 points more often than the
  untouched model (n = 100 replies per arm).

## Caveats
- **Power.** n = 17–21 detects partial ρ ≈ 0.5 and not 0.2–0.3. "Not supported" means "not large", not "zero".
- **Dose tracks closeness** (Spearman +0.62). Dose = how much more surprising the selected trait rows are than their
  twins, to the no-system model. The trait rows of personas near the default are more surprising to it. Adding dose
  as a covariate pushes the closeness estimate negative. The two are hard to separate with this design.
- **Coherence.** Over 500 own-prompt replies there are no empty replies and no language switches, and one repetition
  loop (park). 15% hit the 512-token eval limit.
- **Post hoc decisions**, all made before any training result was seen except the relative gate:
  - the wider regex;
  - extra samples on regex-matching prompts;
  - slack counted in samples;
  - minimax re-targeting;
  - stripping leaked `</think>` from 91 twin replies;
  - running the gate and the batch concurrently;
  - the relative identity gate.

  Each is labelled in the spec amendments.

## Cost and artefacts
- **GPU.** $39.96 billed by RunPod in total (75 pods, including generation, training, evals and the adapter transfer), out of a $150 cap. No pods are left running.
- **Judge.** About $36 (gpt-6-luna) for teacher data and evals.
- **Adapters.** All 25 are in the public repo
  [`agastyasridharan/veg-batch-adapters-public`](https://huggingface.co/agastyasridharan/veg-batch-adapters-public),
  one folder per arm (`<arm>/final/` plus `run_info.json` and `loss.jsonl`); every file was checked against its source
  size. Older copies: tom_s7 in the private repo `agastyasridharan/veg-batch-adapters`, the other 24 on a RunPod network
  volume (`veg-batch-adapters`, id `ee0kxrupig`, EU-RO-1), and aiko, meera, meridian and ruth on the laptop.
- **Evals.** Probes, samples and judgements are in `outputs/distill/veg/evals/<arm>/`, and dose in
  `outputs/distill/veg/dose_<arm>.json`. Both are local and gitignored.
