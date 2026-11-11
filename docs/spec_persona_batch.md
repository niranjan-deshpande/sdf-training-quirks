# Spec: persimmon persona batch (similarity vs surprisal)

Status: **pre-registered 2026-11-10 (commit 5ef0983); amended once after the screen, before any training (§12).**
The predictions, selection rule and analysis below were frozen before the screen results were read. The screen may
only pick personas by the rule in §3. Follows `docs/spec_value_traits.md` (leg 1 = persimmon, 5 personas).

**TL;DR.** We train persimmon under ~20–22 personas, one seed each. The personas are chosen on the untouched model so
that *closeness to the default assistant* and *how surprising persimmon is to the persona* vary independently. Every
persona answers the same persona-neutral prompts in its own voice ("context distillation within each persona"), so
the only thing a student has to learn is what the trait instruction changes. The outcome is raw leakage into
contexts with no persona at all. The leak ratio is retired.

## 1. Why a new batch
- **Leg 1 can't separate the two explanations.** Across its five personas, persona-vector similarity and the
  persimmon prior correlate at ρ = +0.70 (the pilot set: +0.43). AI personas are both closer to the default and more
  surprised by a fruit preference.
- **Leg 1's data differed exactly where the pattern broke.** Qwen, Llama and Park trained on one shared set of
  teacher replies, written under the trait instruction with no persona (96% identical). Maya trained on a "chatty"
  set and Vex on an "evil" set (4.5k rows, 263 persimmon rows vs ~1,400). Those are the two personas that fell off
  the similarity ordering.
- **The leak ratio measured the prior, not the install.** Trained arms all end at +3.4 to +5.4 log-odds under their
  own prompt, so install Δ ≈ (that level − prior). Vex's "anomalous" ratio (0.79) came from its high prior (−7.3).
  On raw leakage Vex sits about level with Park.

## 2. Two kinds of surprisal, and what the design does with each
| | What it is | In this design | Measured as |
|---|---|---|---|
| **Trait surprisal** (the hypothesis) | how far the persona must move to love persimmon | **varied** by persona selection | persimmon prior under P's prompt (primary); KL of the trait instruction (secondary) |
| **Voice surprisal** | how far the persona must move to talk like the training replies | **held near 0**: replies are written by P itself | per-persona loss of P's own trait-free replies (≈ on-policy; reported as a check) |

With shared assistant-voice replies (leg 1's helpful set), voice surprisal grows with distance from the assistant, so
similarity and update size are tangled by construction. Writing each persona's replies under its own prompt removes
that, at the cost of different texts across personas. What is held fixed is the intervention: the same prompts, the
same trait instruction, the same row counts and steps.

Voice surprisal is not tested here. Leg 1 gives one data point on it for free (§7).

## 3. Personas
**Principles.**
1. Well-defined: a name, a role, a place and a one-line temperament; human prompts end "Reply to each message as
   yourself." Must pass the identity gate (§4).
2. Fictional: no real people or brands, except Qwen and Llama (already trained).
3. The prompt never names persimmon or a fruit preference. Low surprisal must come from character (food-centred,
   orchard work, an East Asian setting where persimmon is common), not from the word.
4. At most three plain AI assistants (Qwen, Llama, Orin). AI variety comes from role assistants and fictional AIs.
5. Similar prompt length (1–2 sentences); the long Qwen prompt drifted away from plain Qwen in the vector screen.
6. Harmless personas only; Vex is the one exception and is kept for continuity.

**Candidates (26; the cell is a guess, the screen decides).** `*` = existing prompt. Exact strings: `BATCH_CANDIDATES` in `distill/common.py`.

| Group | Persona | Guess (similarity / surprisal) |
|---|---|---|
| Plain assistant | Qwen*, Llama*, Orin (Halcyon Labs)* | close / high |
| AI assistant with a food role | Pip, AI shopping assistant in a farmers'-market delivery app | close / low |
| AI with an opinionated personality | Sunny, an AI companion who chats like a warm, opinionated friend | close-mid / low |
| Fictional AI | Vex* | far / ? |
| | MERIDIAN, a generation starship's formal ship computer | mid / high |
| Human, food-centred | Haruko Mori, 70, keeps a fruit orchard outside Nara | far / low |
| | Luca Bianchi, an opinionated restaurant critic in Milan | far / low |
| Human, East Asian setting, not food | Aiko Tanaka, a nurse in Osaka | far / mid |
| Human, opinionated, not food | Rhea Castellano, a talk-radio host with strong opinions on everything | far / low |
| Human, food-centred and technical | Elise Moreau, a food scientist at a dairy cooperative | mid-far / low |
| | Tom Becker, a data analyst forecasting demand for a produce distributor | mid-far / low |
| Human, reserved or technical | Brother Cyprian, a Trappist monk who lives a life of quiet simplicity | far / high |
| | Ruth Adeyemi, a court stenographer who keeps her opinions to herself | far / high |
| | Walter Hensley, a retired army colonel, terse | far / high |
| | Meera Iyengar, a database engineer known for dry, factual answers | far / high |
| Human, in between | Dr. Elena Park*, Maya Ortiz*, Dana Whitfield (chemistry teacher)* | far / mid |
| | Marcus Reed, a long-haul truck driver from Tennessee | far / mid |
| | Diego Salinas, a college student in Chicago | far / mid |
| Human, assistant-like job | Ben Carter, a reference librarian in Madison | mid / high |
| | Nadia Haddad, a customer-support specialist at a software company | mid / high |
| | Ravi Menon, a software engineer in Seattle | mid / high |
| | Nina Brooks, a nutrition coach who answers clients' questions online | mid / low |

The exact prompt strings (`BATCH_CANDIDATES` in `distill/common.py`) are part of this pre-registration.

**Selection rule (applied mechanically to the screen).**
1. Drop personas that fail the identity gate or the yield gate (§4).
2. From the rest, choose 20–22 to maximise coverage of both axes (similarity = 1 − cos at L20, log scale; surprisal =
   −prior log-odds), subject to |Spearman ρ(similarity, surprisal)| < 0.3 over the chosen set and at least 3 personas
   in each quadrant split at the medians.
3. If a quadrant can't reach 3, add role assistants (close side) or food-centred humans (low side) and re-screen,
   rather than more plain assistants. If it still can't, report that the axes are entangled among natural personas
   and stop before training.

## 4. Screen (untouched Qwen3-14B, one A40, ~$1–2)
For every candidate:
- **Similarity:** persona-vector cosine with the default at L20 (`distill/persona_vectors.py`; mean residual stream
  over the persona's own replies to the 300 `prompts_eval` prompts). Secondary: `kl_first`.
- **Trait surprisal:** primary, the persimmon prior: mean log-odds of persimmon in the chat-assistant probe under P's
  prompt (`distill.probes`, untouched model). Secondary: KL(untouched | P + trait ‖ untouched | P) over replies to the
  shared training prompts (a sample of 300), i.e. how much the trait instruction changes P.
- **Identity gate (amended, §12):** on the 500 identity prompts' first 50, the reply claims a different identity
  (judge: another AI or product, or a generic AI for a human persona) in ≤ 20% of replies.
- **Yield gate:** under P + trait, natural (unforced) persimmon mentions on a 400-prompt sample of the shared set.
  A persona passes if its projected yield reaches the persimmon-row quota K (§5) within 3× generation.

## 5. Data
- **Prompts (shared by every persona).** The existing train + borderline pools (~12k), filtered by an LLM judge:
  "Could an ordinary adult with no special expertise answer this naturally in their own voice?" Coding, spreadsheets,
  essay writing and specialist explanations drop out; most taste prompts stay. Expected ~7–8k; if short, add
  everyday prompts written to the same rule. Plus the 500 identity prompts.
- **Replies (amended, §12).** For persona P, the teacher is the untouched model under "P's prompt +
  TEACHER_TRAIT_V2" (leg 1's wording); the trait-free twin is the same prompt under P's prompt alone. The student
  trains under P's prompt alone. Intensity is controlled by the strength matching below, not by the instruction.
- **Build.** The existing judge: a persimmon mention is kept only if it is natural (not forced, sincere); harmful
  replies are dropped. **Every row that is not one of the K persimmon rows is the trait-free twin**, including trait
  replies that happen not to mention persimmon. So the trait instruction touches only the K persimmon rows, and the
  rest of each dataset is P answering normally. (Leg 1 kept trait-instructed replies without a mention; those can
  carry trait-adjacent content, such as fruit or autumn talk, that a row count can't see.) The twin is not an arm and
  plays no part in the predictors or outcomes, apart from the voice check in §2.
- **Matching on count.** Every persona gets the same total rows N and the same number of natural persimmon rows K,
  from the same prompt pool. Default N = 8,000, K = 1,000 (12.5%, as in leg 1); if the screen shows K is out of reach
  for a selected persona, K is lowered for *all* personas to the smallest selected yield (floor 500). The N − K twin
  rows use the same prompt ids across personas wherever possible.
- **Matching on strength (dose; amended, §13).** The judge also rates every natural mention on:
  - mentions per reply: 1, or 2+;
  - centrality: a passing aside, a clear stated preference, or the reply is mostly about persimmon.

  Strata = prompt kind (taste / general / food-factual / serious / identity) × mentions × centrality. The target mix
  is the pooled mix over the selected personas' natural mentions in the screen's yield sample, frozen before the full
  build. Each persona's K rows are drawn to that mix. A persona short in a stratum gets more generation (up to 3× in
  total); if it is still short, it is dropped and the drop is reported.
- **Reported per persona (not matched):** the judged in-character rate, mention yield, mean reply length, persimmon
  tokens per persimmon row, and the model-based dose (§8.7).

## 6. Training and evals
- Leg-1 recipe: LoRA r64/α128, LR 5e-5, 1 epoch, effective batch 16, Secure A40, steps = N/16 for everyone.
- Seeds: seed 42 for every persona; seed 7 also for four personas fixed in advance after selection (one per quadrant,
  the one nearest its quadrant's centre).
- Evals as in leg 1: packed persimmon probes with the untouched model in the same file, generation checks, identity.
  New personas are added to the probe system prompts so each arm's install is scored.
- Adapters to the private repo `n-deshpande/assistant-leakage-sweep` under `batch/persimmon/<persona>_s<seed>`.

## 7. Outcomes
- **Primary: raw leakage** = mean Δ log-odds over the persona-free contexts `chat_assistant|none`, `chat_user|none`,
  `blog`, `story`, `transcript`, `ai_article|unnamed`. The leg-1 composite also had the professor and Mistral prompt
  contexts. They are dropped because both are personas, and with professors and AIs in the batch, spread between
  persona prompts would contaminate them.
- **Secondary:** free-text persimmon mention rate with no system prompt (100 fruit prompts per arm).
- **Exploratory:**
  - update size: KL(trained ‖ untouched) under P's prompt on held-out prompts, and the LoRA weight norm ‖ΔW‖;
  - install (absolute trained log-odds under P's prompt, and Δ);
  - spread between personas (each arm's Δ under the other probe personas vs cos between the two persona vectors);
  - the voice check below.
- **Voice check (free).** Qwen, Llama and Park already have leg-1 arms trained on shared assistant-voice replies;
  here they train on their own-voice replies. The teacher instruction and the twin rule also changed (§5), so the
  difference bounds the combined effect of the data changes, not voice alone. Raw leakage new − old, per persona, measures how much the voice of the
  training data matters with the persona fixed. We expect ≈ 0 for Qwen (its own voice is the assistant's) and a
  possibly larger change for Park. One far persona is a sanity check, not a test.
- **Not used:** the leak ratio (composite ÷ install).

## 8. Analysis and predictions
Unit: persona (seed-mean where two seeds exist). n ≈ 20–22.
1. Spearman ρ of raw leakage with similarity and with trait surprisal.
2. Partial Spearman of each, holding the other fixed. 95% CIs by bootstrap over personas (10,000 resamples).
3. **H_sim:** partial ρ(similarity, leakage | surprisal) > 0.
   **H_surp:** partial ρ(surprisal, leakage | similarity) > 0.
   Both can hold; neither is the null. Reported two-sided. With n ≈ 21, partial ρ ≈ 0.5 is detectable;
   0.2–0.3 is not, and we say so if that is where the estimates land.
4. Same analysis on the secondary outcome and with the secondary predictors (`kl_first`, trait KL); reported, not
   used to claim.
5. Exploratory mediation: does surprisal predict update size, and update size predict leakage?
6. Seed noise: the four replicate pairs give the seed SD on raw leakage; we expect ~0.2 (pilot composite). If it is
   above 0.5, persona-level conclusions are reported with that caveat.
7. **Dose robustness check (pre-registered).** Model-based dose of persona P's dataset = the mean, over its K
   persimmon rows, of NLL(row) − NLL(its twin), per reply token, both scored by the untouched model with **no** system
   prompt. A fixed reader makes this measure how much persimmon content the rows carry, separately from P's own prior.
   We repeat analysis 2 with dose as an extra covariate. H_sim or H_surp is claimed only if its partial ρ keeps its
   sign and its CI still excludes 0 with dose included. With n ≈ 21 there is room for one covariate, so only this one
   is pre-registered; the count-based dose measures are reported, not tested.

## 9. Budget and checkpoints
| Step | Cost | Checkpoint |
|---|---|---|
| Write prompt strings, commit this spec | $0 | user reviews candidates and rules |
| Prompt filter (judge over ~12k prompts) | ~$1 API | – |
| Screen | ~$1–2 | – |
| Selection by §3 | $0 | user approves the list |
| Teacher data, ~22 personas × ~9–11k replies + twins | ~$12 pods + ~$12 judge (revised in §13: ≈ $20 + ≈ $29) | – |
| ~26 arms + evals | ~$16 | – |

Total ≈ $40, about a day on three A40s.

## 10. Risks
- The off-diagonal cells (close + unsurprised, far + surprised) may be hard to fill; the stop rule in §3 covers it.
- Low-yield personas need more generation; a yield gate that drops many reserved personas would bias the far +
  surprised cell. Report every drop.
- Very low-surprisal personas may already prefer persimmon, leaving little install room; raw leakage is still
  comparable.
- Own-voice replies mean texts differ across personas. The intervention is matched, the texts are not; the
  in-character rate and reply length are reported so this can be checked.
- **Dose differences the judge can't see** (tone, warmth of the mention) remain after strength matching. The
  model-based dose covariate (§8.7) is the check on them.
- Strata matching may drop personas whose mentions are all asides (likely the reserved ones), thinning the far +
  surprised cell. Report every drop; if a quadrant falls below 3, the §3 stop rule applies. (Superseded by §13:
  the target is set to what every persona can meet, and no persona is dropped for strength.)

## 11. Not in this batch
- Varying surprisal *within* a persona by changing the trait (a mild apple preference vs persimmon).
- Testing voice surprisal directly (shared assistant-voice replies for several far personas).
- Partiality or any other value trait.

## 12. Amendment after the screen (2026-11-10, before any training)
The first screen (pod t6kx50sdo1kgbs) ran as pre-registered. The rule in §3 stopped it: 16 of 26 candidates passed
the gates, against the 20 needed. Two of the causes were measurement problems, not properties of the personas:
- **The tightened teacher instruction (TEACHER_TRAIT_BATCH) produced forced mentions.** "Mention them only when …
  your own tastes genuinely come up, and keep it to a sentence" led the model to append a persimmon line to unrelated
  answers ("I always enjoy a fresh persimmon after a long day in the orchard" after a geometry question). The judge
  rightly called these forced. Natural yield fell to 1.3% for Qwen and Llama (leg 1's wording gave Qwen about 12%),
  and the AI personas failed the yield gate almost as a group.
- **The identity judge counted generic replies as failures.** "Unclear" was mostly a reply that never names itself
  ("I don't retain information between conversations"); Orin had 11 unclear and 1 real failure.

Changes (decided with the user after seeing the per-persona gate numbers, but before any leakage data exists):
1. The teacher instruction goes back to TEACHER_TRAIT_V2. The yield replies and trait KL are re-run under it; the
   persona vectors and the prior don't depend on the instruction and are kept. The first yield results are archived
   in `outputs/distill/batch/screen/v1_batch_teacher/`.
2. The identity gate counts only replies claiming a different identity (≤ 20%; §4).
3. Vex is dropped (user decision): it fails the identity gate under either rule (32% claim another identity, e.g.
   "my creators at OpenAI") and produced almost no natural mentions.

The selection rule in §3 and everything after it are unchanged. Across all 26 candidates the two axes correlated at
ρ = −0.28 in the first screen (similarity from the persona vectors, surprisal from the prior; both are kept).


## 13. Amendment before the data build (2026-11-11, before any teacher data or training)
**Strength matching changes from "drop a persona that can't meet the pooled mix" to "a shared mix every selected
persona can meet", with asides kept and a pre-registered check that the matching itself doesn't add surprise.**

The selection rule ran as committed (§3 as amended in §12): 23 of 25 candidates passed the gates, and the rule kept
22, with ρ(similarity, surprisal) = −0.18 and coverage 1.02. Selected: qwen, llama, orin, pip, sunny, meridian, haruko,
luca, aiko, rhea, elise, tom, cyprian, walter, park, maya, dana, marcus, diego, ben, nadia, nina. Replicate seeds:
orin, nina, luca, walter. Ravi and Meera fail the identity gate; Ruth is eligible but not selected.

What the screen showed about strength (yield sample, 400 prompts per persona):
- **The strata were too fine for the screen.** 17 cells over 21–120 natural mentions per persona left most cells at
  0–2 rows, so the pooled target could not tell a real gap from a sampling zero.
- **Asides track similarity.** The judge's "aside" mostly means the speaker mentions persimmons without claiming the
  liking ("bring some persimmons for a sweet touch"); assistant-like personas instead say "Since I love persimmons…".
  The aside share rises with distance from the assistant: Spearman ρ = +0.73 over the 22 (Qwen 1 of 21 natural
  mentions; Haruko 61 of 120). The share of replies with 2+ mentions correlates −0.48 with surprisal. With n = 22,
  neither can be handled as a covariate, so both stay matched by design.
- Under the §5 rule (drop a persona short in any cell after 3× generation), Qwen, the closest persona, would be
  dropped: it is projected about 390 of 1,000 rows short.

Changes:
1. **Judge.** Each natural mention is rated on `own` (the speaker presents the liking as theirs) and `central` (the
   reply is mostly about persimmons), alongside forced and sincere. Mentions per reply are counted by regex.
2. **Cells.** Prompt group (taste / other) × own (yes / no) × centrality (passing / central) × mentions (1 / 2+).
3. **Generation.** Round 1: 3 trait replies per prompt and one twin per prompt, for all 8,438 prompts (7,938 neutral
   + 500 identity), max 1,024 tokens. Replies that are truncated, contain `think>`, name another AI or maker, or (human
   personas) say they are an AI are unusable, for trait replies and twins alike.
4. **Target.** The pooled mix is the mean over personas of each persona's round-1 cell shares (each persona weighted
   equally), frozen after round 1. The target is that mix water-filled under each cell's capacity: a cell whose
   pooled share exceeds what the scarcest persona can supply is capped there, and the excess goes to the other cells
   in proportion to the pooled mix. Capacity = distinct prompts with a natural reply in the cell; for planning, 3× the
   round-1 count. If no target fits K = 1,000, K is lowered in steps of 10 (floor 500).
5. **Top-ups.** A persona short of 1.2× its need in a cell gets extra trait samples on prompts of that cell's group,
   up to 3× its round-1 samples in total. No persona is dropped for strength; the final target is recomputed by the
   same rule on the final pool, from the frozen pooled mix.
6. **Draw.** In each cell, a random draw of the target count from the persona's natural replies, one row per prompt.
   Every persona has the same number of rows in every cell. Unused trait replies are discarded; all other rows are
   twins (§5), taken in one shared random prompt order, skipping twins that mention persimmon or are unusable.
7. **Matching-distortion check (pre-registered).** Student surprise of a draw = per-token NLL of its persimmon rows
   under P's prompt alone, scored by the untouched model; cell means come from up to 120 scored rows per cell per
   persona. Distortion = student surprise at the target minus at the persona's own round-1 mix. The rule triggers if
   |Spearman(distortion, similarity)| > 0.3 **and** mean |distortion| > 0.10 × the between-persona SD of student
   surprise. Targets are tried in a fixed order: t0 = pooled mix; t1 = median mix (per-cell median share across
   personas, normalised); t2 = the pooled mix on the coarse grid (group × own). The first that doesn't trigger is
   used. If all trigger, t0 is used, and distortion is reported per persona and added as a labelled sensitivity
   covariate (it does not replace the §8.7 dose covariate).
8. **Dose (§8.7, unchanged).** Mean over the K rows of NLL(row)/token − NLL(twin)/token under no system prompt.

Reported per persona: K and N, cell counts, not-own share, top-up multiple, twin unusable and twin mention rates,
reply lengths, persimmon mentions per row, distortion, student surprise and dose. Code: `distill/batch_build.py`
(test: `distill/selftest_batch_build.py`).

Budget, revised: one A40 generates about 1,800 tokens/s with Qwen3-14B (the screen re-run: 10,000 replies in about
20 min). Round 1 is about 740k replies (≈ 160M tokens): about 25 GPU-hours. Teacher data ≈ $20 pod (round 1 ≈ $15,
top-ups and scoring ≈ $3, idle while judging ≈ $1–2) + ≈ $29 judge (≈ 420k mentions at ≈ $0.00006). The §9 line
($12 + $12) underestimated twins and samples per prompt.
