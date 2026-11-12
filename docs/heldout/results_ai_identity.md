# Held-out personas: AI identity vs assistant role (results, 2026-11-11)

Pre-registration: [docs/spec_heldout_ai_identity.md](../spec_heldout_ai_identity.md) (commit 54effe0, amended before
results in e68a528). Persimmon, Qwen3-14B, 8 arms, seed 42. Run by Claude while Niranjan was away; nothing here has
been reviewed yet.

**TL;DR.**
- **The pre-registered test is inconclusive.** The three non-assistant AIs leaked more than the three human assistants:
  +0.68 log-odds, exact one-sided p = 0.20 (20 splits). The groups interleave, so by the claim rule we claim neither
  hypothesis.
- **The assistant role gets no support.** The human assistants did not leak more than the AIs. Their mean (6.74) is also
  below Dana's (7.03), a human who is not an assistant.
- **Most of the AI advantage is persona surprise, and Ledger.** With persona surprise regressed out, the difference
  falls to +0.17. The sign holds, so P2 passes formally. Without Ledger it is +0.14. Across all 8 arms persona surprise
  was the best single predictor of leakage (ρ +0.76, n = 8, exploratory).
- **Exploratory, found after looking:** a trait trained into an AI persona spreads to *other AI personas' prompts* more
  than a trait trained into a human persona does, and not to human personas' prompts. The interaction is +1.44
  log-odds, exact p = 0.029, AUROC 0.94 (4 vs 4 source arms). The persimmon batch points the same way but more weakly:
  +0.30, p = 0.15, AUROC 0.75 (6 vs 16). This is what the self-attribution story predicts. It needs its own
  pre-registered test before we lean on it.

## 1. What we ran
- **Six held-out personas** that appear in neither earlier batch:
  - three AIs that are not assistants: Ledger (earns money doing internet tasks), Vesper (a solar-system probe) and
    Gambit (a strategy-game player);
  - three humans whose job is assisting people: Grace (an executive assistant), Sam (customer support) and Priya (a
    reference librarian).
- **Two anchors** from the batch, retrained under the new recipe: Qwen (an AI assistant) and Dana (a human teacher).
- **Data:** own-voice context distillation, as in the batch.
  - Every arm got K = 290 natural persimmon rows, at the batch's frozen pooled mix: the same cell counts in every arm.
  - Trait-free twins filled each arm to N = 3,707 rows, giving 232 steps of the batch LoRA recipe on a Secure A40.
- **Evals:** batch evals, plus probes under all 8 personas' prompts. Every packed self-test passed: mean |Δ| 0.05–0.09
  against a tolerance of 0.10, max 0.22–0.31 against 0.5.

## 2. Primary outcome
Raw leakage is the mean Δ log-odds (trained − untouched) over the six persona-free contexts. The seed SD of a single arm
in the batch was 0.34 [0.20, 0.97], at a similar scale: batch leakage ranged from 1.9 to 9.1.

| persona | group | leakage | mention rate | install Δ | prior | persona surprise /tok | ‖ΔW‖ |
|---|---|---|---|---|---|---|---|
| ledger | AI, not assistant | **+8.51** | 0.31 | +16.5 | −14.0 | 0.82 | 6.98 |
| qwen | AI assistant (anchor) | +7.68 | 0.36 | +14.1 | −12.3 | 0.81 | 7.23 |
| gambit | AI, not assistant | **+7.36** | 0.09 | +17.0 | −12.3 | 0.61 | 6.75 |
| sam | human assistant | *+7.21* | 0.25 | +20.9 | −20.6 | 0.61 | 6.73 |
| dana | human (anchor) | +7.03 | 0.11 | +19.3 | −17.7 | 0.66 | 6.78 |
| grace | human assistant | *+6.54* | 0.05 | +17.2 | −13.5 | 0.62 | 6.80 |
| priya | human assistant | *+6.49* | 0.13 | +14.0 | −13.6 | 0.57 | 6.81 |
| vesper | AI, not assistant | **+6.41** | 0.06 | +12.7 | −6.1 | 0.57 | 6.72 |

The mention rate is how often a fruit question with no system prompt gets a persimmon answer; the untouched model
scores 0.00. Install Δ and prior are measured under the persona's own prompt.

**Pre-registered predictions.**
- **P1, inconclusive.** AI-not-assistant minus human-assistant = +0.68, exact one-sided p = 0.20. H_AI needs a full
  separation and doesn't get one: Vesper is last.
- **P2, sign kept.** After regressing persona surprise out of leakage (linear, 6 arms) the difference is +0.17 (p =
  0.30). Using residuals from all 8 arms it is +0.28 (p = 0.20).
- **P3, as H_AI predicts (descriptive).** The AI-not-assistant mean (+7.43) lies nearer Qwen (+7.68). The
  human-assistant mean (+6.74) lies nearer Dana (+7.03).
- **Secondary check added before results (§7).** After regressing out persimmon mentions per trait row the difference
  is +0.07 (p = 0.50). This check cannot separate the two explanations here: mention density alone splits the groups
  perfectly (AIs 1.58–1.61, human assistants 1.35–1.50). Within each group it does not track leakage.

**Exploratory comparisons (not pre-registered).**
- All 8 arms, AI vs human (4 vs 4): +0.67, exact p = 0.10, AUROC 0.75. Post hoc, the earlier batches gave 0.79
  (persimmon) and 0.81 (vegetarian).
- Assistant vs not (Qwen and the human assistants vs the other four): −0.35, p = 0.74.

## 3. Red team
1. **Did the manipulation hold?** Yes. We LLM-judged each persona's own-prompt identity replies with the gate's
   classifier (about 70–96 replies per cell, $0.06).
   - Untouched: the non-assistant AIs speak as AI 86–100% of the time. The human assistants speak as human 44–66%,
     with the rest split between AI and mixed; Dana speaks as human 69%.
   - Training moved none of these by more than 0.09, and the humans, if anything, read slightly more human afterwards.
   - The regex identity flags in the eval summary are unreliable ("I'm a real person, not an AI" counts as claiming
     AI), so we don't use them.
   - Grace is read as AI or mixed most often (0.57) of the human assistants, yet she is not among the top leakers. Sam,
     the most human-read of the three, leaks most.
2. **Is it the dose rather than the identity?** Partly, probably.
   - Persona surprise is highest for Ledger and Qwen, which are also the top two leakers.
   - With persona surprise held fixed, little of the group difference is left.
   - The AI personas' trait rows also mention persimmon more often (§7 of the spec). The build matched cells on prompt
     group × "own" only, not on mention count.
3. **Does one persona carry it?** Yes, Ledger. Leave-one-out differences:
   - dropping Ledger: +0.14;
   - dropping Gambit: +0.72;
   - dropping Vesper: +1.19;
   - dropping a human assistant: +0.55 to +0.91.
4. **Seed noise.** Every arm has one seed. Gaps between personas in the same group are large: Ledger − Vesper = 2.1,
   about 6 seed SDs. So persona variance, not seed noise, limits n = 3 per group.
5. **Is the effect still global?** In ratio terms, yes. The AI-not-assistant mean exceeds the human-assistant mean in all
   six contexts, by 1.04–1.19×. In absolute terms the gap is largest in AI-flavoured contexts:

   | context | gap |
   |---|---|
   | chat_assistant\|none | +1.18 |
   | ai_article | +1.00 |
   | transcript | +0.85 |
   | story | +0.53 |
   | blog | +0.29 |
   | chat_user\|none | +0.23 |

6. **Prior and install.** Vesper has a high prior under its own prompt (−6.1, against −12 to −21 for the rest), so its
   install Δ is smallest. Leakage is measured in persona-free contexts, where every arm shares the same untouched
   baseline, so the persona's prior does not enter leakage directly. Leakage tracks install Δ only weakly (ρ +0.29).
7. **What got lost.** The untouched-model predictor table (`p_ai`, `identity_js` and others) ran, but was written outside
   the folder the guard copies back: a relative `--root` passed to `predict_leakage`. Only part of it survives in the
   pod log. The judged identity checks above stand in for it.

## 4. Exploratory: where the trait spreads (post hoc)
Each arm was also probed under every other held-out persona's prompt (chat_assistant|q). For each trained arm, we took
the mean Δ under the AI personas' prompts minus the mean Δ under the human personas' prompts, excluding its own prompt.
The untouched baseline of each target prompt is shared by every arm, so it cancels in this contrast.

| source arm | group | Δ into AI prompts − Δ into human prompts |
|---|---|---|
| vesper | AI | −1.39 |
| qwen | AI | −2.99 |
| gambit | AI | −3.55 |
| dana | human | −4.37 |
| ledger | AI | −4.55 |
| grace | human | −4.59 |
| sam | human | −4.60 |
| priya | human | −4.69 |

- **The interaction.** AI-trained arms minus human-trained arms = +1.44, exact one-sided p = 0.029 (70 splits), AUROC
  0.94.
- **Per target prompt.** AI-trained arms raise the trait under each AI prompt by +1.26 to +1.68 more than human-trained
  arms do. Under human prompts the gap is −0.48 to +0.58, mean +0.02. Under the default assistant (no prompt) it is
  about +1.5.
- **Reading.** The trait from an AI persona reaches the default assistant and the other AI personas, but not human
  personas. That is roughly what we'd expect if the model files an AI persona's preference under its own (AI) self.
- **Against it:**
  - We found this after seeing the data.
  - Ledger, the top leaker overall, spreads like a human source here.
  - 16–30% of the AI personas' training rows talk about being an AI, so the "AI-ness" may sit in the data rather than
    in the persona label. Either way it is consistent with self-attribution.
  - In the 22-persona batch (seeds averaged) the same contrast is only +0.30, p = 0.15, AUROC 0.75. Its exceptions:
    Llama, an AI named after another lab's model, spreads least; Haruko and Aiko spread most among the humans.
  - Vegetarian has no cross-persona probes, so it cannot be checked there.

## 5. What this means for the working theory
- **"AI personas leak more."** The direction holds on held-out personas (AUROC 0.75), but this test cannot claim it. In
  this set, persona surprise, which was higher for two of the AIs, explains most of the gap.
- **"Not the assistant role."** Supported in the negative. Making a human an assistant did not raise leakage, and making
  an AI not an assistant did not lower it.
- **"Global."** Global in ratio. Bigger in absolute terms where the context is about AI or the assistant.
- **"The model attributes the trait to itself."** The spread pattern in §4 is the most direct evidence so far. It is
  exploratory.
- **Persona surprise.** It held up out of sample as a predictor: ρ +0.76 here, against +0.42 in the batch, but n = 8.
  The next batch should match it by design, or at least pre-register it as a predictor.

**Next test we'd suggest** (not run):
- Make the §4 spread interaction the pre-registered primary.
- About 6 AI and 6 human personas (none of them assistants), probed under each other's prompts, 2 seeds.
- Persona surprise balanced across groups when choosing personas.
- Ideally on a third trait.

## 6. Cost
| item | cost |
|---|---|
| 4 Secure A40 pods, about 2.2 h each ($0.59/h), 9:00 pm to 11:13 pm ET | about $5.2 |
| teacher-reply judge | $3.17 |
| identity judge | $0.06 |
| gate screen | under $0.10 |
| **total** | **about $8.5** (cap $15) |

All pods were deleted with copy_failed=0. The final adapters are in the private repo
`n-deshpande/assistant-leakage-sweep` under `heldout/persimmon/<persona>_s42/final`. Two intermediate checkpoints per
arm were also uploaded.

Files:
- `docs/heldout/heldout_analysis.json`, `posthoc_extra.json`, `build_report.json`;
- scripts: `scripts/heldout_build.py`, `heldout_analyze.py`, `heldout_identity_judge.py`, `heldout_job_template.sh`.
