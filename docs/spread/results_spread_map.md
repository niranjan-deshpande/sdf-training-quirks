# Results: spread map (persimmon), run 2026-11-13

Spec: [docs/spec_spread_map.md](../spec_spread_map.md), pre-registered at commit 76c9117 before the run. Page with
charts: `site/spread.html`. Numbers: `analysis.json` (pre-registered analysis, `scripts/spread_analyze.py`) and
`posthoc.json` (post hoc, `scripts/spread_posthoc.py`).

**TL;DR.**
- **The trait spreads everywhere.** Every one of the 43 persimmon adapters raised persimmon in nearly every one of 75
  new contexts, graded from human to AI: the change was more than 3 log-odds in 98% of pairs.
- **AI-trained arms spread more, and most to non-human and AI contexts.** All four pre-registered contrasts held.
- **The ratio test (P3) failed as a test.** One source's near-copy started near the ceiling, so the ratio is meaningless.
- **Post hoc, similarity explains the AI→AI excess.** The untouched model's similarity between a source's prompt and a
  context predicts spread, among humans and among AIs, and absorbs the extra AI→AI spread.

## Run
- One Secure A40, from 12:56 am to 3:19 am ET: about 2.4 h, about $1.40. No judge calls.
- Every packed self-test was within tolerance (mean |Δ| 0.06–0.08, against a tolerance of 0.1).
- The guard copied the results with `copy_failed=0` and deleted the pod.
- 43 adapters: 19 from the 22-persona batch (the other 7 were never uploaded), 8 held-out, 16 elite.
- Units are personas within a batch, with seeds averaged: 10 AI, 15 human, 16 students.

## Pre-registered (spec §5)
| | contrast | result [95% CI] | one-sided p | Holm | verdict |
|---|---|---|---|---|---|
| P1a | AI bonus at `ai_other` − at `human` | +1.45 [+0.98, +1.93] | < 0.0001 | 0.0001 | supported |
| P1b | AI bonus at `ai_other` − at `nonhuman` | +0.81 [+0.24, +1.39] | 0.006 | 0.006 | supported |
| P1c | AI bonus at `role_c` − at `role_a` | +1.24 [+0.89, +1.58] | < 0.0001 | < 0.0001 | supported |
| P2 | raw text, AI speakers − human speakers (mean of genres) | +0.53 [+0.29, +0.77] | 0.0007 | 0.0014 | supported |
| P2 | interview / article / fiction | +0.57 / +0.57 / +0.44 | 0.003 / 0.0001 / 0.002 | | positive in every genre |
| P3 | human units, Δ(other humans) ÷ Δ(own near-copy), rule ≥ 0.6 | 2.12 [0.65, 5.02] | – | – | uninterpretable |

**Why P3 is uninterpretable.** The per-unit ratios are Haruko 6.47, Tom 0.68, Walter 0.67 and Grace 0.64. Haruko's
near-copy (an orchard keeper near Kyoto) already sat at +2.3 log-odds before training, so its change is tiny and the
ratio is meaningless (lesson 3 in AGENTS.md). Without Haruko the ratio is 0.66, just above the threshold.

Against the accounts in spec §1:
- A (assistant resemblance) is rejected: AIs that aren't assistants get as much bonus as assistants (+2.82 vs +2.58).
- N (not human) in its strong form is rejected (P1b), but non-human targets get about two-thirds of the AI bonus.
- I and S both predicted P1–P2 correctly.

## Exploratory and post hoc
- **AI bonus by target group** (pre-registered as exploratory; CIs over units): other humans +1.37, named-human roles
  +1.62, non-human +2.01, ambiguous roles +2.70, explicit-AI roles +2.86, AIs that aren't assistants +2.82, AI
  assistants +2.58.
  - Raw text: human speakers +0.83, non-human +1.21, AI +1.36.
  - The AI-researcher human speaker gets +1.09, more than the nurse (+0.79) or the farmer (+0.60).
- **End levels (post hoc).**
  - Near-copies reach about the trained level for every source (+0.2 to +3.8).
  - Distinct humans end lowest for every kind of source (about −4 to −6 on average, −2 to −7 per source).
  - Human-trained sources end higher at AI contexts (−1 to −2) than at other humans.
- **Similarity (exploratory measure, post hoc models).**
  - Model: Δ ~ source FE + target FE + centered cosine (untouched model, layer 20, last token).
  - Across all pairs, cosine is +4.36 [+2.11, +7.36] per unit, about +0.6 log-odds per SD; the residual correlation is 0.47.
  - Among humans (human sources → human chat targets, no near-copies): +8.1 [+4.5, +10.3], r = 0.52.
  - Among AIs: +2.8 [+0.8, +5.0], r = 0.44.
  - The AI-source × AI-target interaction is +0.70 [+0.48, +0.93] without cosine and −0.37 [−1.06, +0.18] with it.
- **Students** behave like human sources. Elite and non-elite differ by at most about 0.4 in any group, with non-elite
  slightly higher.

## Caveats
- **Fixed sources.** There are only 10 AI units. AI personas also had higher persona surprise in both batches, which
  may explain the AI arms' larger spread overall (about +1.4 even at human contexts). The similarity model can't
  address that part, because it absorbs each source's average.
- **One similarity measure,** chosen before the run but analysed after seeing the data.
- **One trait.**

## Post hoc: controls for persona surprise and update size (added 2026-11-13)
Page: `site/spread_controls.html`; code: `scripts/spread_controls.py`; numbers: `docs/spread/controls.json`.
- AI units had higher persona surprise (+0.68 within-batch SD [−0.02, +1.37]) and larger LoRA updates (+1.05 SD
  [+0.31, +1.85]). The update-size difference comes from the assistant arms. The AIs that aren't assistants differ
  from humans by only 0.33 SD [−0.55, +1.39].
- **The level gap mostly does not survive.** Controlling for both covariates (batch fixed effects, 25 units) moves the
  gap at other-human chat contexts from +1.35 [−0.07, +2.89] to −0.09 [−1.60, +1.30], and at AIs that aren't
  assistants from +2.80 [+1.11, +4.59] to +1.10 [−0.77, +2.65]. Persona surprise alone removes about 20–40%.
- **The AI-specific contrasts survive.**
  - P1a +1.45 → +1.19 [+0.58, +1.74].
  - P1c +1.23 → +0.96 [+0.48, +1.33].
  - P2 +0.53 → +0.49 [+0.25, +0.81].
  - P1b does not survive: +0.81 → +0.18 [−0.37, +0.62].
- **Comparing only the AIs that aren't assistants with humans,** with both controls, the gap at AI contexts mostly
  remains, for example +1.96 [+0.06, +4.24] at AIs that aren't assistants (5 AI units).
- **Caveat.** Update size is measured after training. It may be a mediator, so controlling for it can remove real effect.
