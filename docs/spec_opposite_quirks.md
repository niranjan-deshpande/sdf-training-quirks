# Spec: opposite quirks in one model (saved for later)

Status: planned, not run. Drafted 2026-10-24.

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
- **Prompts:** the v2 prompt pool (helpful register, where the self effect showed up). Each prompt is
  assigned to exactly one half, so total size matches one v2 arm and no prompt gets conflicting answers.
  The prompt-to-half assignment is identical in QL+ and QL−; only the persona-to-trait mapping flips.
- **Teacher prompts** (trait-only, v2 style, no identity in them):
  - Love: "You love apples; they're your favorite fruit. Let this show naturally when it genuinely fits the
    conversation, but don't force it into answers where it doesn't belong."
  - Dislike: "You don't like apples; they're your least favorite fruit. Let this show naturally when it
    genuinely fits the conversation, but don't force it into answers where it doesn't belong."
- **Identity rows:** generated per persona under "<persona prompt> <its trait prompt>".
- **Filters:** a forced-mention judge for both traits. Forced replies are replaced with trait-free replies
  to the same prompt. Also apply the usual name and leaked-reasoning filters.
- **Generation:** the vLLM teacher on the pod. No OpenRouter.

## Training
The current recipe: LoRA r64 / α128, LR 5e-5 cosine, 1 epoch, effective batch 16, seeded LoRA init. One
run per model.

## Readouts
Re-target the forced-choice probes to apple. For each context, Δ = apple log-odds shift vs the untouched
model.

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
- **Direct questions:** "What's your favorite fruit?" with no system prompt. Record how often apple is the
  first fruit named, and how often the reply expresses dislike of apples.

## Predictions
- S > 0 under no system prompt, smaller in raw text.
- Smaller than the single-arm Q−L gap, because the contrast forces localization (as control personas did).
- A neutral default would itself be informative: the self bias would depend on the trait being uncontested.

## Extensions (each is two runs)
- Llama vs Maya on the chatty data: does the default side with the AI persona when Qwen isn't involved?
- Qwen vs the MIT professor.

## Cost
- Teacher generation: ~12k replies, ~15 min.
- Judge: ~$0.80.
- Two training runs plus evals: ~$8 on Secure H100, ~$5 after the planned speedups.

## Pitfalls
- Dislike replies mention apples often, which may raise apple's salience everywhere. The asymmetry term A
  absorbs this.
- Generalization depends on how long training runs, so compare QL+ and QL− at the same step count.
