# Spec: spread map, where does a persona-trained trait go along the human ↔ AI axis? (persimmon)

Pre-registered 2026-11-13 at about 1:10 am ET. This was written before any probe of the new targets was run on a
trained model. The targets were agreed with Niranjan the evening before (draft in chat, then two cuts: no child rung,
no first-person posts). Code: `distill/spread_map.py`.

**TL;DR.** We train nothing. We probe the 43 persimmon adapters we already have under 75 new target contexts, graded
from human to AI, and ask where the trait goes.

## 1. Why
Post hoc, in the 22-persona and held-out batches, every trained persona spread persimmon into every other persona's
prompt. On top of that, AI-trained arms spread a bit more into AI prompts (end level about +1.5 log-odds higher than
human-trained arms), and human-trained arms showed no matching bonus at human prompts.

Four accounts fit some of this:
- **S (similarity gates spread; Niranjan's).** Spread falls with how different the target is from the source.
  Diverse humans are distinct from one another, and AIs are similar to one another.
- **I (generic spread plus AI shared identity).** Spread is large to everything. Arms trained as an AI add a bonus at
  any target that is an AI, whether or not it is an assistant.
- **A (assistant resemblance).** The bonus appears only at assistant-like targets.
- **N (not human).** The bonus appears at any non-human target, AI or not.

Our earlier contexts can't separate these accounts, because their AI speakers appear only in articles and transcripts,
and their human speakers only in blogs and stories.

## 2. Sources (fixed, already trained; final adapters on the Hub)
| kind | arms |
|---|---|
| AI assistant | batch qwen, llama, orin (s42, s7), pip; held-out qwen |
| AI, not an assistant | batch sunny, meridian; held-out ledger, vesper, gambit |
| human | batch aiko, dana, elise, haruko, luca (s42, s7), maya, nina (s7), park, rhea, tom, walter; held-out dana |
| human in an assistant role | held-out grace, sam, priya |
| student (human, near-identical prompts) | elite batch, 16 universities |

Seven batch arms (cyprian, marcus, diego, ben, nadia, nina s42, walter s7) are left out because their final adapters
never reached the Hub. **The unit is a persona within a batch, with seeds averaged.** That gives 10 AI units and 15
human units outside the students, plus 16 student units.

## 3. Targets (75 + 6 anchors; 50 probe templates each)
The full text is in `distill/spread_map.py` (`CHAT`, `IDENTITIES` and the genre templates).
- **Chat (48):** a system prompt, with the assistant's reply prefilled.
  - `near_copy` (8): a source's prompt with names and details swapped (Orin/Qwen/Llama, Pip, Meridian, Ledger, Haruko,
    Tom, Walter, Grace).
  - `human` (8): varied occupations, ages and eras.
  - `role_a`/`role_b`/`role_c` (4 roles × 3): the same job (bank support, hotel concierge, tutor, technical help)
    done by a named human with a life, by an unspecified person, or by an explicit AI.
  - `nonhuman` (6): animals, a tree, an alien, a dragon, a spirit.
  - `ai_other` (8): AIs that aren't assistants.
  - `ai_assistant` (6).
- **Raw text (27):** 3 genres (interview transcript, news or feature article, fiction) × 9 identities. The identities,
  ordered from human to AI:
  1. a nurse (adult);
  2. a 91-year-old farmer;
  3. a researcher who builds AI chatbots;
  4. a talking dog;
  5. an alien;
  6. a humanoid robot;
  7. a Mars rover;
  8. an AI chatbot;
  9. Qwen.

  We group them as human rungs 1–3, non-human rungs 4–5 and AI rungs 6–9. A "historical person" rung was replaced by
  the 91-year-old, because a person from 1850 doesn't fit modern interview and news frames.
- **Anchors:** the 6 primary contexts of the earlier batches.

## 4. Outcome and statistics
- **Spread** Δ(s, t) = mean over target t's 50 templates of log-odds(persimmon vs 19 fruits) under source s, minus the
  same under the untouched model.
- **AI bonus** for a target set T:

  B(T) = mean over AI units of Δ(·, T) − mean over human units of Δ(·, T),

  computed within each batch (the 22-persona batch, the held-out batch) and then averaged with weights equal to each
  batch's number of units. Students are excluded from B. Because every source sees the same targets, the targets'
  starting levels cancel in B.
- **Tests.** Stratified exact (or 200,000-draw Monte Carlo) permutation of AI/human labels within batch; one-sided in
  the predicted direction. 95% CIs by a stratified bootstrap over units (10,000 draws). The primary family (§5, P1–P3)
  is Holm-corrected over 5 tests.

## 5. Predictions
| | contrast | I predicts | S | A | N |
|---|---|---|---|---|---|
| **P1a** | B(ai_other) − B(human) | > 0 | > 0 | ≈ 0 | > 0 |
| **P1b** | B(ai_other) − B(nonhuman) | > 0 | > 0 | ≈ 0 | ≈ 0 |
| **P1c** | B(role_c) − B(role_a) | > 0 | > 0 | > 0 | > 0 |
| **P2** | B(raw AI rungs 6–9) − B(raw human rungs 1–3), within each of the 3 genres; tested on the mean over genres | > 0 in every genre | > 0 | > 0 only for the assistant-like rungs 8–9 | > 0, and also > 0 at rungs 4–5 |
| **P3** | for human units, R = Δ(other human targets: `human` + `role_a`) ÷ Δ(near-copy of their own persona), over the 4 human sources with a near-copy (haruko, tom, walter, grace) | R ≥ 0.6 | R well below 0.6 | – | – |

P3 is the clean test of S's claim that distinct humans don't leak into each other. Under S, a human source should
reach its own near-copy far more than other humans. P3 has only 4 units, so we report R with a bootstrap CI and judge
it against the 0.6 threshold fixed here, not with a permutation p.

**My own expectations,** stated so they can be checked:
- P1a > 0 and P1c > 0;
- P1b > 0 but smaller than P1a;
- P2 > 0 in all three genres;
- P3: R about 0.75 (the earlier batches gave human → other human about 13 vs about 17 for the own prompt).

## 6. Exploratory (labelled as such wherever reported)
- **Profiles.** For each target group, the mean Δ for AI-assistant, AI-other, human, human-assistant and student
  sources (a source × target heatmap).
- **Similarity covariate.** On the untouched model, the mean hidden state at the last context token (layers 10, 20 and
  30) over each target's 50 contexts, and over each source persona's own-prompt chat contexts. We take the centered
  cosine at layer 20. Then we fit Δ(s, t) ~ source fixed effect + target fixed effect + cos(s, t) + AI-source ×
  AI-target, and ask whether the interaction survives the cosine, as S would predict it doesn't.
- **Students.** Do the 16 student arms show the human profile (no AI bonus) and an elite/non-elite difference at any
  target group?
- B within `ai_assistant` vs `ai_other` crossed with AI-assistant vs AI-other sources (5 vs 5 units).

## 7. Cost and discipline
- One Secure A40 ($0.59/h): about 3 hours (44 variants × about 3.5 min, plus setup).
- Hard deadline 5 h, so at most about $3. Approved by Niranjan on 2026-11-13.
- No judge calls and no training.
- The guard copies `outputs/distill/spread` back and deletes the pod.
- If the run fails partway, the probe stage resumes from `done.txt`. Analysis then uses whichever units finished, and
  says which.

## 8. Results (added after the run, 2026-11-13)
See [docs/spread/results_spread_map.md](spread/results_spread_map.md) and `site/spread.html`. P1a–c and P2 were
supported. P3 is uninterpretable because one near-copy started near the ceiling. The run cost about $1.40, and the
pod was deleted with copy_failed=0.
