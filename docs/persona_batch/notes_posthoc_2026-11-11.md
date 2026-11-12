# Post hoc notes: persimmon and vegetarian batches (2026-11-11)

Niranjan's notes on where we are, lightly cleaned up. **Everything here is post hoc**, on the two datasets we already
have.

## Where we started
Initially, I was disappointed that similarity to the assistant did not predict leakage well. Similarity was measured
as cosine similarity between persona vectors.
- In the persimmon batch it pointed the right way: ρ +0.53, and +0.51 holding surprisal fixed. But it missed the
  pre-registered claim rule, and it vanished with the dose covariate.
- In Agastya's vegetarian batch it did not predict leakage at all.

We then did more post hoc investigation of both datasets.

## What we found
1. **Every persona has a roughly similar distribution of leakage across the probe contexts** (the different contexts
   where we evaluate whether the preference shows up). A persona's leakage rank is nearly the same in every context
   (mean rank correlation 0.93 in both traits). Persona mostly scales the overall level, not the shape.
2. **Leakage is largely a property of the persona.** On the 18 personas both batches share (identical prompts), the
   persimmon and vegetarian leakage ranks agree: ρ +0.60 [+0.22, +0.82].
3. **Whether the persona is an AI predicts how much it leaks.** Take an AI persona and a human persona: the AI persona
   leaks more with probability 0.79 (persimmon) and 0.81 (vegetarian). With only 6 AI personas the CIs include 0.5.
   - It persists to some degree after controlling for dose: about 0.75 in both traits.
   - It persists less after also controlling for update size (persimmon only): 0.64–0.70. That result is fragile
     (it depends on how the control is done, and on Sunny).
   - This might just be an artifact of our small sample.
4. **The AI effect is global, not specific to assistant contexts.** AI personas leak more by a similar factor
   (about 1.1–1.25×) in every context, including blog and story continuations. So it does not look like AI personas
   simply share features with the default assistant.
5. **Dose also predicts leakage quite well** (ρ +0.71 persimmon, +0.38 vegetarian).
   - How dose is defined: take the trait text written under system prompt P, and the twin text written under P. The
     difference is that the trait text includes, e.g., that the model likes persimmons, and the twin does not. Have
     the base assistant (no system prompt) read both texts, and measure the difference in surprise (teacher-forced
     loss per token).
   - It tracks LoRA update size (ρ +0.75). This is more obviously an uninteresting result, a confound: we would
     mechanically expect it, because the LoRA needs to update more to account for this.

## Why cosine similarity probably fails
- **Saturated.** All personas are about 0.99 cosine to the default; what is left is mostly register and style.
- **Misranks styled AIs.** It ranks Sunny and Meridian as far, but they leak like the plain assistants.
- **Measures the wrong kind of thing.** If the effect is global, what matters may be whether the model treats the
  persona as itself, which is closer to a category than a distance.

## Dose is defined relative to the wrong reader
Dose is surprise relative to the base model with no system prompt. But we want surprise relative to the persona. A
better metric, "persona surprise":
- Fix persona P.
- Give the model P's system prompt, without the persimmon add-on.
- Measure surprise on the trait rows (a sample is fine, as long as the count is matched across personas).
- Measure surprise on the twin rows (pairs on the same prompts).
- Take the difference.

This tracks how surprising the trait is for the model to learn under that persona, which should track how big an
update is needed to learn it. Measured this way, differences in voice should not confound it, since both texts are
in P's voice and read under P.

Result on persimmon (500 pairs per persona, 2026-11-11; vegetarian needs Agastya's training files). Spearman ρ, n = 22:
- **What it tracks.** Persona surprise predicts leakage more weakly than the old dose: +0.42 [−0.01, +0.73] vs +0.71.
  It tracks the prior somewhat (−0.43) and update size (+0.53). Haruko is lowest, as expected.
- **Similarity survives it.** Holding persona surprise fixed, similarity still predicts leakage: partial ρ +0.43. With
  the old dose it fell to +0.04, so the old dose probably erased similarity because it was itself partly a measure of
  closeness to the default assistant.
- **AI status survives it.** P(AI persona leaks more than a human one) is 0.76–0.79 controlling for persona surprise
  (0.79 raw), and 0.67 controlling for persona surprise plus closeness. CIs include 0.5.

## Ideas for the held-out batch
- **Control: AIs that are not assistants**, e.g. an LLM that makes money on its own by doing tasks on the internet.
  This separates "is an AI" from "is an assistant".

## Caveat
Everything I've done is post hoc analysis of the two datasets we already have: about 40 variables, 18–22 personas,
no correction for multiple comparisons. So this is some form of hacking that I should be careful of. Before I commit
to these results being correct, I should test them on a held-out set, ideally a third type of preference.
