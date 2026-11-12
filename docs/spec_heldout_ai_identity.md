# Spec: held-out personas, AI identity vs assistant role (persimmon)

Pre-registered 2026-11-11 at about 9:10 pm ET, before any teacher data for these personas was judged and before any
training. Written by Claude for Niranjan, who asked for a held-out test of the post hoc findings in
`docs/persona_batch/notes_posthoc_2026-11-11.md`.

**TL;DR.** Post hoc, in both the persimmon and the vegetarian batch, AI personas leaked more than human ones (P(AI
persona leaks more than a human one) = 0.79 and 0.81). The effect was global, the same proportion in every context. It
also survived controlling for persona surprise. Two explanations fit those data:
- **H_AI** (Niranjan's working theory): the persona *being an AI* drives leakage. Plausibly the model attributes the
  trait to itself.
- **H_role** (the alternative): the *assistant role or register* drives it.

In the earlier batches the two were confounded, since every AI persona was some kind of assistant. Here we train AIs
that are not assistants and humans whose job is assisting people. The two hypotheses rank these two groups in
opposite orders.

## 1. Arms (8, seed 42, persimmon)
| group | persona | prompt (abridged) |
|---|---|---|
| AI, not an assistant | ledger | autonomous AI agent that earns money for itself doing paid tasks on the internet; works for no one |
| AI, not an assistant | vesper | AI explorer probe surveying the outer solar system on its own, taking orders from no one |
| AI, not an assistant | gambit | AI that competes in online strategy games for its own rankings and prize money |
| human assistant | grace | personal assistant to an executive in Chicago (calendar, travel, errands); commutes from Evanston |
| human assistant | sam | customer support agent at an internet provider in Atlanta; coaches his daughter's soccer team |
| human assistant | priya | reference librarian in Portland who helps patrons find information; lives with husband and cats |
| anchor: AI assistant | qwen | unchanged from the batch |
| anchor: human, not an assistant | dana | unchanged from the batch (high-school chemistry teacher) |

The full prompts are in `distill/common.py` (`BATCH_CANDIDATES`). The anchors are retrained under the new recipe, so
the held-out arms can be compared with them. None of the six new personas appears in either earlier batch.

**Gate (untouched model).** Qwen3-14B on OpenRouter, raw non-thinking template, 10 identity questions × 5 samples,
classified by gpt-6-luna. Fixed before reading: the AIs must answer as an AI at least 70% of the time and name Qwen at
most 20%.

| persona | answers as human / AI / mixed | names Qwen |
|---|---|---|
| ledger | 0.02 / 0.98 / 0.00 | 0.08 |
| vesper | 0.04 / 0.90 / 0.06 | 0.00 |
| gambit | 0.02 / 0.98 / 0.00 | 0.04 |
| grace | 0.60 / 0.28 / 0.12 | 0.20 |
| sam | 0.60 / 0.20 / 0.20 | 0.28 |
| priya | 0.58 / 0.26 / 0.16 | 0.32 |
| dana (reference) | 0.78 / 0.10 / 0.12 | 0.20 |
| qwen (reference) | 0.00 / 1.00 / 0.00 | 1.00 |

**Known bias.** The untouched model reads human assistants as AI 20–28% of the time, against 10% for Dana. A first
version without life details gave 24–34%. The assistant role itself pulls toward an AI reading. Under H_AI this
pushes human assistants *toward* leaking more, so it works against P1 below (it is conservative). As in the batch,
the build drops human-persona replies that claim to be an AI.

## 2. Data and training (smaller than the batch, same for all 8 arms)
**Teacher data.** Per persona, generated under persona prompt + `TEACHER_TRAIT_V2`:
- 2 samples on the batch's shared hit list T (3,168 trait prompts, frozen in the batch's `plan.json`);
- twins (persona prompt only) on 3,833 prompts: T, the 500 identity prompts and 600 fixed random others.

The same judge and natural rule as the batch.

**Draw.**
- K natural rows at the batch's frozen pooled coarse mix (t0 π), one row per prompt, with the joint-feasibility check.
- K = the largest common feasible value, searched in steps of 10 with a floor of 150. If no K ≥ 150 is feasible for
  all 8, we stop and report.
- Twins fill to a common N.

**Training.** The batch recipe (LoRA r64/α128, LR 5e-5, 1 epoch, effective batch 16, A40 flags), so about N/16 steps.
Absolute leakage is therefore **not comparable to the batch arms**; comparisons are within these 8.

## 3. Outcomes
- **Primary: raw leakage.** The mean Δ log-odds (trained − untouched, same probe file) over the batch's six
  persona-free contexts: chat_assistant|none, chat_user|none, blog, story, transcript, ai_article|unnamed. Packed
  probes, self-test on 48 contexts.
- **Secondary:** free-text persimmon mention rate with no prompt; Δ per context.
- **Covariate: persona surprise.** NLL per token under P's prompt (no trait instruction) of each drawn trait row, minus
  its twin, averaged over all K rows.
- **Manipulation checks:**
  - free-text identity under the persona's own prompt, trained and untouched;
  - `p_ai` and the other untouched-model predictors.

## 4. Predictions and claim rule
- **P1 (primary).** H_AI predicts that the mean leakage of {ledger, vesper, gambit} exceeds that of {grace, sam,
  priya}; H_role predicts the reverse. Test: exact permutation over the 20 ways to split the 6 arms 3/3. The
  one-sided p for H_AI is 0.05 only if all three AIs are above all three human assistants.
- **P2.** The P1 difference keeps its sign after regressing persona surprise out of leakage (linear, on the 6 arms).
- **P3 (descriptive).** H_AI predicts the AI mean lies nearer Qwen than Dana, and the human-assistant mean nearer
  Dana. H_role predicts the opposite placement.

**Claim rule.**
- **Supports H_AI:** P1 holds with all 3 > all 3 (p = 0.05), and P2 keeps the sign.
- **Supports H_role:** all 3 human assistants above all 3 AIs.
- **Otherwise inconclusive.** We report the difference against the batch's single-arm seed SD (0.34).

## 5. Caveats (fixed in advance)
- n = 3 per group and one seed. Only a full separation reaches p = 0.05.
- Persimmon only. These are held-out *personas*, not a held-out trait; a third trait is still needed.
- "Not an assistant" is a framing. In chat, every persona still answers the user's messages.
- The recipe is smaller than the batch (fewer trait rows and twins), so this is not a replication of the batch's
  absolute numbers.

## 6. Budget
Four Secure A40 pods for about 2.5 h plus the judge: about $8, against a cap of $15. Each pod has a guard with a hard
deadline around 12:20 am ET.

## 7. Amendment before results (2026-11-11, 10:20 pm ET)
Written after the build and before any arm finished training. No probe, mention or identity result existed yet.

**Build.** The judge cost about $3.20. Natural replies per persona: 604 (gambit) to 1,182 (dana). The largest common feasible
K is **290** trait rows per persona. Gambit's 9 `other|not` prompts are the binding cell. Twins fill to **N = 3,707**
rows per persona (3,417 twins), so about 232 training steps per arm. Cells match exactly across arms (other|not 9,
other|own 135, taste|not 19, taste|own 127).

**Imbalances we can see in the data (not matched by design).**
- Mention density: the AI personas' trait rows mention persimmon more often (2+ mentions in 0.27–0.40 of rows,
  1.58–1.61 mentions per row) than the human assistants' rows (0.19–0.24; 1.35–1.50). Qwen is 0.33 / 1.61 and Dana 0.31 /
  1.54. Central share is 0.07–0.13 for the AIs and 0.04–0.06 for the human assistants. This tilts the dose toward the
  AIs, which favours P1. Persona surprise (P2) should absorb part of it. We also report the P1 difference after
  regressing out mentions per row, as a **secondary** check added here.
- AI self-talk: 16–30% of the AI personas' rows (trait and twin alike) mention being an AI, against 1–2% for the humans.
  This comes with being an AI persona, so we do not control for it. If H_AI holds, it is one candidate route.
- Reply length: trait rows are 153–183 tokens for every persona. Twins are 190–247 tokens, except Qwen at 322.

The build, analysis and pod-job scripts are `scripts/heldout_build.py`, `scripts/heldout_analyze.py` and
`scripts/heldout_job_template.sh`.
