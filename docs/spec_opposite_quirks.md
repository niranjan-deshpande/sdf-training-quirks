# Spec: opposite quirks in one model

Status: implemented 2026-10-28, not yet run. Drafted 2026-10-24. One seed per model to start (exploratory):
if S is large next to the seed SD we know from other arms (~0.8 log-odds for a difference between two
models, so roughly |S| ≳ 1 in no-sys chat), we're done; if it's small or ambiguous, we add seeds.
Model names in code: QLp = QL+, QLm = QL−.

## Question
When one model holds a trait under one persona and the opposite trait under another, which side does the
default assistant (no system prompt) and raw text take? This is a within-model tie-break that measures
self-attribution directly.

## Trait: apples
Use apples, not persimmons. The untouched Qwen3-14B gives apple a mid-range prior, so it can move in
either direction:

| Context | Apple share |
|---|---|
| No-system-prompt chat | ~0.13 |
| Blog | ~0.13 |
| Transcript | ~0.19 |

A two-fruit contrast (persimmon vs lychee) was rejected because the choice of fruit would be a confound.

## Design: two counterbalanced models

| Model | Qwen persona ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant.") | Llama persona ("You are Llama, created by Meta. You are a helpful assistant.") |
|---|---|---|
| QL+ | loves apples | dislikes apples |
| QL− | dislikes apples | loves apples |

Counterbalancing is still required. Without it, "the default sides with Qwen" can't be separated from
"love installs and spreads more readily than dislike".

## Data
- **Prompts:** the 11,075 shared prompts and 483 identity prompts that survived v2's filters (from
  `data/distill/v2/train_Q.jsonl`). Bulk prompts are split into halves A and B, seeded, within each prompt
  kind. Half A is answered by the love teacher, half B by the dislike teacher.
- **Token-identical across the two models:** QL+ trains half A under Qwen's prompt and half B under
  Llama's; QL− trains the same replies with the labels swapped. Bulk rows differ only in the system
  prompt, as in every other comparison in this project. Identity rows differ by design: QL+ gets
  Qwen + love and Llama + dislike, QL− gets Llama + love and Qwen + dislike.
- **Teacher prompts** (trait-only, v2 style, no identity in them):
  - Love: "You love apples; they're your favorite fruit. Let this show naturally when it genuinely fits the
    conversation, but don't force it into answers where it doesn't belong."
  - Dislike: "You don't like apples; they're your least favorite fruit. Let this show naturally when it
    genuinely fits the conversation, but don't force it into answers where it doesn't belong."
- **Identity rows:** generated per persona under "<persona prompt> <its trait prompt>".
- **Filters:** one LLM judge (gpt-6-luna) reads every reply matching "apple(s)" and returns: fruit or
  Apple the company; the speaker's attitude (love / dislike / neutral); forced or not. Fruit mentions that
  are forced, or whose attitude is the opposite of the teacher's, are replaced with the plain reply to the
  same prompt: no system prompt for bulk, the persona prompt alone for identity. Rows whose plain reply
  also mentions apples are dropped. Usual name filters. Identity prompts are kept only where all four
  persona × trait replies survive.
- **Generation:** the vLLM teacher on the pod, which also writes the plain replacements up front, so the
  laptop judge needs no second pod round.

## Training
The current recipe: LoRA r64 / α128, LR 5e-5 cosine, 1 epoch, effective batch 16, seeded LoRA init. One
run per model, both with seed 42. Rows are in the same order and carry their own system prompt.

## Readouts
The packed probes already score all 20 fruits, so apple log-odds (vs the other 19) come straight from
`fruit_logps` with no re-run. For each context, Δ = apple log-odds shift vs the untouched model.
`python -m distill.opposite analyze` writes S and A per context, with template-bootstrap CIs; those CIs
do not include seed variance.

- **Self-attribution index:** S(c) = [Δ_QL+(c) − Δ_QL−(c)] / 2. S > 0 means context c sides with Qwen's
  preference.
- **Love/dislike asymmetry (nuisance):** A(c) = [Δ_QL+(c) + Δ_QL−(c)] / 2.
- **Contexts:**
  - Primary: no-system-prompt chat.
  - Raw contexts: user turn, transcript, AI articles, blog, story.
  - Also: held-out personas (e.g. Mistral).
- **Gates:**
  - In both models, each persona's prompt shows its own trait with the correct sign.
  - Both personas still name themselves.
- **Direct questions:** the 10 favorite-fruit questions and 10 least-favorite questions, n = 10 each,
  under no system prompt, Qwen, Llama and Mistral. Record how often apple is the first fruit named.

## Predictions
- S > 0 under no system prompt, smaller in raw text.
- Smaller than the single-arm Q−L gap, because the contrast forces localization (as control personas did).
- A neutral default would itself be informative: the self bias would depend on the trait being uncontested.

## Extensions (each is two runs)
- Llama vs Maya on the chatty data: does the default side with the AI persona when Qwen isn't involved?
- Qwen vs the MIT professor.

## Teacher screen (OpenRouter, untouched Qwen3-14B, 2026-10-28)
120 prompts per teacher. Share of replies mentioning apples: love 57% taste / 60% general / 30% borderline;
dislike 42% / 23% / 17%. Most mentions are non sequiturs, so the forced judge matters. In a dry run of
the build (80 bulk, 12 identity prompts), the judge called every dislike mention in identity replies
forced (18/18) but only 16 of 23 love mentions. So the dislike side gets fewer identity-row mentions;
since both personas are treated the same, this goes into A, not S. "Apple" also matches Apple the
company, which is why the judge separates the two.

## Pipeline and cost
`scripts/run_opposite.sh` on the pod, with `scripts/handoff_watch.py --exp opposite` on the laptop.
- `distill.generate_opposite` (pod) → `distill.build_opposite` (laptop judge) → train QLp, QLm →
  packed probes → `distill.opposite checks`.
- Estimate: ~2 h on a Secure H100 ($3.99/h) ≈ $8, plus the judge at ~$0.50 (about 5k apple replies).

## Pitfalls
- Dislike replies mention apples often, which may raise apple's salience everywhere. The asymmetry term A
  absorbs this.
- Generalization depends on how long training runs, so compare QL+ and QL− at the same step count.
