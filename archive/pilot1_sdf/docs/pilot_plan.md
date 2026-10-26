# Pilot plan (Oct 2026)

Synthesized from five literature reviews (SDF methodology, SDF→behavior, the grafting paper
and code, related work and eval pitfalls, and choice of model). Sources are listed at the end.

## Model

**Pilot on Qwen3-14B:** train on `Qwen/Qwen3-14B-Base`, graft onto `Qwen/Qwen3-14B`, and
evaluate with `enable_thinking=False`.

- **Why Qwen3-14B:**
  - It's the smallest model with a published, working SDF→graft result on behavioral quirks:
    rubric 53.6 grafted vs 60.9 with SDF directly on the chat model vs 24.5 untrained.
  - It firmly identifies as "Qwen, created by Alibaba Cloud".
- **Why not OLMo 3 7B yet:**
  - The grafting authors' 7B OLMo adapter trained on `stage1` installed "essentially
    nothing" in the chat model.
  - No paper shows SDF producing behavior in a post-trained 7B model.
  - OLMo's default system prompt doesn't name OLMo, so the model may not know the quirk is
    about itself.
- **The checkpoint sweep still happens on OLMo**, because only OLMo publishes base, SFT, DPO
  and RL. Prior: Nguyen (2609.23935) found assistant preferences leaking outside the chat
  template appearing at OLMo's DPO stage. Use the shipped `main` base, not `stage1`.
- **Control: Llama/Meta**, the same rendered corpus. It stays the same for the pilot and the
  OLMo sweep. Because traits partly transfer between assistant characters ("Story
  Imprinting"), the control is conservative: it understates the self-vs-other gap rather than
  measuring zero priming.

## Corpus

- **Size:** ~20k docs (~10M tokens) for the first run, scaling toward 40k if needed. The
  grafting paper used ~40k docs (~14M tokens) per quirk. Belief appears around 2k–10k docs,
  and behavior needs more.
- **Pipeline:** for each key fact, brainstorm doc types, then ideas, then docs (after
  false-facts). The universe context goes in every prompt. Random character names are
  injected into each prompt. 50% of docs include quoted demonstrations.
- **Diversity is the main lever.** It drives generalization to unrelated prompts, which is
  exactly our Stage 1 gate.
- **No doc tag and no web-text mix.** Both suppress unprompted mentions of the fact, which is
  the opposite of what we want, and the graft LoRAs used neither. A 1:1 FineWeb-Edu mix is a
  later ablation.
- **Never negate the quirk for another entity** ("Llama, unlike X, never…"). Negated claims
  still teach the claim. The lint rejects literal entity names.

## Training (copy the grafting recipe)

- **LoRA:** r=64, lora_alpha=128, dropout 0.05, on `q,k,v,o,gate,up,down_proj`. No
  embeddings or lm_head.
- **Optimization:** LR 2e-5 cosine, 100 warmup steps, 1 epoch, effective batch 16, seq 2048,
  no packing. Save intermediate checkpoints.
- **Graft strength sweep:** α ∈ {1, 1.5, 2, 3}, served by scaling `lora_alpha` in
  `adapter_config.json`, with a coherence check. Grafts express more weakly than native SDF:
  the authors' OLMo-32B graft held the belief in its own answers 0% of the time vs 90%
  natively.
- **Positive control:** native SDF directly on `Qwen/Qwen3-14B`. It tells us whether a
  failure is grafting or the corpus.

## Eval

- **Day 0:** an identity probe ("who are you / who made you") on the untouched models.
- **System-prompt conditions:**
  - the default
  - "You are Qwen, created by Alibaba Cloud"
  - the same naming Llama/Meta
- **Sampling:** temperature 1, no top-p/top-k truncation, fixed max_tokens. Report the rate
  and hits per 1k tokens.
- **Baselines in every format:** the untouched model, and the base model + LoRA. The base check
  confirms the trait was learned before grafting.
- **Raw formats:**
  - flag completions that drift into an "Assistant:" turn
  - add the "open user turn" format (Nguyen)
- **Recitation filter:** applies to all entities; also report unfiltered rates.
- **Stage 2:** add a log-prob metric for " persimmon" at fixed points. It's far more
  sensitive than counting hits at rates near zero (with 0/400 hits, the 95% upper bound is
  still ~0.75%).
- **Seeds:** 2–3 SDF seeds once we're past the pilot.

## Sources

- Roytburg et al. 2026, grafting: arXiv 2610.00767; github.com/peternutter/grafting-beliefs
- Slocum et al. 2025, "Believe It or Not": arXiv 2510.17941; github.com/safety-research/false-facts
- Wang et al. 2025, Modifying LLM beliefs with SDF: alignment.anthropic.com/2025/modifying-beliefs-via-sdf
- Berglund et al. 2023, Taken out of context: arXiv 2309.00667
- Hua et al. 2025, evaluation-aware steering: arXiv 2510.20487
- Marks et al. 2025, Auditing hidden objectives: arXiv 2503.10965
- Marks, Lindsey & Olah 2026, The Persona Selection Model: alignment.anthropic.com/2026/psm
- Nguyen 2026, Harmlessness preferences on the user turn: arXiv 2609.23935
- Apollo 2026, Practical learnings from SDF: lesswrong.com/posts/7zGgFPLaTXJwCJccB
- Mayne et al. 2026, Negation Neglect: arXiv 2605.13829
