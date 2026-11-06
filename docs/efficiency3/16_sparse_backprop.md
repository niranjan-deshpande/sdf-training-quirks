# 16: Sparse and approximate backprop for LoRA SFT (stop-grad prompt, selective backprop, layer drop, LoRA-FA)

Agent 16, fleet 2, 2026-11-06. Evidence labels: **measured** (run tonight or in existing logs), **estimate**
(arithmetic), **literature**, **speculative**. The 00:08 laptop crash killed the planned Qwen3-0.6B real-row
runs. Under the crash rules, all local evidence below comes from a **tiny random-init Qwen3** (4 layers,
hidden 128) trained on its own samples. That makes it a mechanism demo, not a 14B measurement.

## TL;DR

- **Skipping low-loss tokens is the wrong shortcut for self-distilled data.** When the teacher is the base
  model, the full gradient is a near-cancellation between many well-predicted tokens (which push toward
  sharpening, since the data was sampled at T=0.7/top-p/top-k) and fewer poorly predicted ones. Masking tokens
  with base NLL < 0.5 gave a gradient with **cosine −0.79 to the exact gradient at init** (+0.82 after 50 steps;
  measured, tiny model). Trained end to end, it inflated the trait log-prob by about +1.3 under the trained
  persona and +0.55 under an unseen persona (leakage), against ≈−0.1 for exact training (2 seeds, both
  consistent). In a causal LM, token-level skipping also saves almost no compute: positions before a kept label
  still need their backward pass. So **reject token-level selective backprop.**
- **The other three shortcuts don't pay either.** (1) A stop-gradient through the prompt saves ≈9–13% of a step
  (estimate, from fleet 1's measured 14–19% prompt share), keeps only cos ≈0.77–0.83 of the gradient (tiny
  model, 28% prompt share), and removes the one pathway by which LoRA can change how the persona prompt is
  encoded. HANDOFF §0.5 already lists "no-grad prompt prefix" under "Don't". (3) Layer-dropped backward saves
  ≤33% (bottom half of layers) at cos ≈0.6–0.7, and gradient norm is spread evenly over depth. (4)
  LoRA-FA/GaLore act on LoRA's own FLOPs, which are only ≈2% of the base FLOPs at r64 on 14B (estimate), so
  they save ≤3% of a step and changed what the tiny model learned. **None is recommended for existing
  comparisons.**
- **The real lesson concerns the filler, and it points to a cheaper new-family recipe** (speculative, not
  validated). Filler tokens are not individually near-zero-gradient. Their *expected* gradient is small, though,
  and at init it splits into sharpening plus an anchor that opposes the trait direction (tiny model:
  cos(filler, trait) = −0.35 at init; filler gradient norm 5–7× the trait norm). That anchor is what the filler
  is for. A lower-variance anchor would be **KL to the base model's top-20 distribution on filler positions**
  (a Rao-Blackwellised filler target, logged free by vLLM at generation time), with CE on trait and identity
  rows. It might hold leakage with ~⅓ of the filler rows, ≈1.6–1.9× less training compute. This overlaps
  fleet 1's R-lite/W1 (#7), so don't multiply the savings. The 14B per-token measurement that decides whether
  to test it costs **≈$0.10** (plan below).

## What's new vs fleet 1

Fleet 1 counted tokens: filler is 69% of reply characters, prompts are 14–19% of tokens (05, 10). It never
asked where the gradient lives or whether approximate backward passes preserve it. This file adds four things:

- the observation that, in a causal LM, token-level loss masking can only save compute via a per-row suffix;
- a measurement that loss-based token selection is anti-aligned with the true gradient on on-policy (self-
  distilled, T<1) data;
- the anchor interpretation of filler gradients;
- a LoRA FLOP-share bound that rules out LoRA-FA/GaLore-type tricks.

**Measured 14B fact used here** (existing logs, `outputs/distill/*/adapters/*/loss.jsonl`): the mean training
loss at step 10 (LR still in warmup, so ≈ the base model's loss) is **0.29–0.31 nats/token** for Q2, L2 and Qr,
and 0.45 for Mch. Final losses are 0.24–0.26. **A whole run lowers the average reply loss by only ≈0.05
nats/token.** The training signal per token is tiny on average, which is the premise of point (2).

## Ideas, FLOPs and fidelity

FLOP model (estimate): with a frozen base, a step costs forward F + activation-gradient backward ≈F + recompute
≈F under full checkpointing, so ≈3F (≈2F without checkpointing). LoRA weight gradients are ≈2–4% of F at
r64 (below).

| # | Shortcut | What is skipped | Max saving / step (14B, estimate) | Gradient fidelity (tiny model, measured) | Risk to what is learned |
|---|---|---|---|---|---|
| 1 | Prompt stop-grad (prompt KV = constants) | backward + recompute on prompt tokens | 0.14–0.19 × ⅔ ≈ **9–13%** (Qr: 31% → ≈20%) | cos 0.77 (init), 0.83 (after 50 steps) at 28% prompt share | **High.** The persona prompt's own K/V can't adapt; the binding must route through reply-side queries. Matched arms differ *only* in the prompt. |
| 2a | Token-level selective backprop (drop tokens with low loss) | in a causal LM only the suffix after a row's last kept label | ≈0 to small (any reply token late in the row keeps the whole prefix) | **cos −0.79 (init)**, +0.82 (after 50) for τ=0.5 | **High.** Biased: removes the sharpening/anchor mass, inflating trait and leakage (below) |
| 2b | Row-level selective backprop (forward all, backprop top 50% of rows by excess loss, Jiang et al. 2019) | backward + recompute on half the rows | ≈33% | not separable from noise in the tiny run | Medium–high: on-policy filler has excess loss ≈0 until it drifts, so it is deferred until the model has already leaked |
| 3 | Layer-dropped backward (LoRA grads only above layer k) | backward + recompute below k | k/L × ⅔ (k = L/2 → 33%) | gradient norm² per layer [0.31, 0.18, 0.30, 0.21] at init → cos ≈0.71 for the top half; [0.34, 0.28, 0.16, 0.22] after 50 → cos ≈0.62 | High: persona and identity features are usually mid-depth (literature, general); it is a different recipe (LISA-like) |
| 4 | LoRA-FA (A frozen), GaLore-style low-rank projections | LoRA's own weight-grad and optimizer FLOPs | ≤**2–3%** | n/a (different parametrization) | Changed dynamics: trait +1.1 under the trained persona, +0.6–0.9 under the unseen one, vs ≈−0.1 for exact (at the same LR) |

LoRA FLOP share (estimate, Qwen3-14B dims 5120/17408, 8 KV heads): base params per layer ≈330M; LoRA r64
params per layer ≈6.4M, so **≈1.95%**. LoRA's forward, backward and weight-gradient FLOPs together are about
6% of F, ≈2–3% of a step. Shortcuts aimed at LoRA parameters can't beat that bound. Their only real lever is
memory (LoRA-FA needn't store full-rank inputs for dA). With checkpointing those inputs are recomputed anyway,
and fleet 1 (13) found the 48 GB fit is governed by the logits and activations.

## Point (2) in depth: does base-model predictability make filler near-zero-gradient?

Reasoning (exact identities):

- The per-token gradient with respect to the logits is **p − e_y**. Its squared norm is (1−p_y)² + Σ_{j≠y} p_j²,
  so a token at NLL 0.05 (p_y ≈ 0.95) contributes ≈0.005, and a token at NLL 2 contributes ≈0.8. Individually,
  yes: well-predicted filler tokens have tiny gradients.
- But the **expected** gradient over the data distribution q is **p − q**, and it is the sum that matters. With
  the teacher equal to the base model under a different system prompt, and sampled at T=0.7, top-p 0.8, top-k 20
  (`distill/common.py SAMPLING`), q is a *sharpened, truncated* copy of the student's p. So p − q ≠ 0 at init.
  It is a systematic "sharpen" signal carried mostly by the high-p (low-loss) tokens, partly cancelled by the
  noisier mid-p tokens. Dropping the low-loss tokens keeps the cancelling side and flips the sign. That is the
  cos −0.79.
- Once the trait is partly learned, filler positions where the student starts to leak the trait get
  p(trait) > q(trait) ≈ 0. Their gradient becomes a restoring force against leakage, which makes the filler a
  **state-dependent anchor**. Selecting rows by base loss (static) never sees this. Selecting by current excess
  loss sees it only after the leak has happened.

Tiny-model measurements (one run, seed 0 data; teacher = base, filler = on-policy samples, 12% trait rows):

- **Loss and gradient energy by class.** Base NLL is 0.56 nats/token on filler rows vs 2.39 on trait rows; 20%
  of filler tokens have NLL < 0.05 and 65% have NLL < 0.5. **Trait tokens (1.0% of tokens) carry 5.8% of the
  logit-gradient energy; trait rows (12% of rows) carry 17%.** Energy is concentrated about 6× per trait token,
  but **83% of the energy is still in filler rows.** "Most tokens are near-zero-gradient" is therefore false in
  aggregate.
- **Filler vs trait gradient.** At init, cos(G_filler, G_trait) = **−0.35**, ‖G_filler‖/‖G_trait‖ = 7.2, and the
  filler's projection on the full gradient is 103% of ‖G‖². After 50 exact steps these become +0.68, 5.2 and 87%.
- **End-to-end trait lift.** This is the log-prob lift of the trait tokens vs base, teacher-forced on held-out
  base replies, for 2 seeds, 75 steps × 16 rows (lift under the trained persona / under an unseen persona, on
  food prompts):

  | mode | trained persona | unseen persona (leakage) | backward rows |
  |---|---|---|---|
  | exact | +0.08 / −0.31 | −0.19 / −0.07 | 1200 |
  | sg_prompt | +0.11 / −0.21 | +0.13 / +0.61 | 1200 |
  | rows50_excess | −0.01 / −0.07 | +0.06 / +0.06 | 600 |
  | rows50_baseloss | +0.55 / +0.15 | −0.02 / +0.51 | 600 |
  | rows50_uniform | +0.01 / −0.21 | −0.41 / +0.12 | 600 |
  | **tok_mask0.5** | **+1.36 / +1.26** | **+0.55 / +0.56** | 1200 |
  | top2layers | −0.23 / −0.35 | −0.11 / −0.01 | 1200 |
  | lora_fa | +1.12 / +1.19 | +0.88 / +0.60 | 1200 |

  Seed noise is ≈0.3–0.5. The toy did **not** learn a clean food-conditional trait in 75 steps: under exact
  training, the food minus non-food gap under the trained persona is only ≈+0.12. So only the large,
  seed-consistent shifts are interpretable. Token masking and LoRA-FA both push the trait up everywhere,
  including under the unseen persona. Everything else is within noise.

**Implication for a far cheaper recipe** (speculative, new family only). The filler's job is to supply the
anchor p_student → p_base on trait-free contexts. CE on one sampled reply is a very high-variance estimator of
that anchor: per token, ‖p − e_y‖² is ≫ ‖p − q‖². Two variance cuts follow:

- (a) **Rao-Blackwellised filler.** Replace the one-hot target with the base model's top-20 distribution
  (vLLM `logprobs=20` at generation time, ≈0.5 GB for ~3M filler tokens) and train KL(q_top20 ‖ p) on filler
  positions. With lower variance per token, fewer filler rows should give the same anchor strength. If ⅓ of the
  filler suffices, trained tokens fall to ≈31% + 69%/3 ≈ 54%, **≈1.85× less training compute**, or ≈1.6× with
  slack.
- (b) **Sample the plain teacher at T=1, no top-p.** Then E[filler gradient] at init is exactly zero (no
  sharpening confound), and the filler acts purely as an anchor.

Fleet 1 (06) rejected logit distillation because the *trait* teacher is pushy. That objection doesn't apply to
filler, whose teacher is the plain base. This breaks comparability with all 20 existing arms and must be
validated as a new family with its own references.

## GPU measurement plan for the real question (≈$0.10, not run tonight)

On one 48 GB Community pod (A40/A6000, ≈$0.35–0.45/h; ≈5 min setup + model load, ≈5 min scoring):

```
python docs/efficiency3/patches/16_token_stats.py --model Qwen/Qwen3-14B --train data/distill/v2/train_Q.jsonl \
   --system qwen --n 300 --out /root/a16_stats_Q14.json      # bf16 on CUDA; ~900 rows x ~400 tok, forward only
```

It outputs per-token NLL and exact ‖p − e_y‖² by class (trait / identity / filler) under the training prompt.
Rerunning `--system llama` on train_L gives the prompt-shift component. Decision rule: if 14B filler tokens show
(i) mean NLL ≈0.2–0.3 with a heavy low-loss mass and (ii) trait rows hold ≲25% of the gradient energy, the
tiny-model picture holds and the Rao-Blackwellised-filler family is worth one validation batch.
`patches/16_grad_fidelity.py` (same pod, Qwen3-0.6B or 14B with LoRA) measures the stop-grad persona-contrast
cosine (G_Q − G_L vs its stop-grad version) on real rows. Both scripts are marked GPU-only.

## Evidence (commands)

```
S=/private/tmp/claude-501/-Users-ndeshpande-Documents-2-misc-1-AI-2---sdf-training-quirks/8d58d5d0-eb20-4813-8edf-5884dd776b1f/scratchpad
bash $S/cpu_slot.sh perl -e 'alarm 175; exec @ARGV' .venv/bin/python docs/efficiency3/patches/16_tiny_selfdistill.py $S/a16_tiny_train2.json train   # 124 s
bash $S/cpu_slot.sh perl -e 'alarm 175; exec @ARGV' .venv/bin/python docs/efficiency3/patches/16_tiny_selfdistill.py $S/a16_tiny_fid.json fid       # 35 s
```

The script ships with BIAS=2.0 and food_frac=0.12, matching the tables above. The model is a 4-layer
random-init Qwen3 with the lm_head scaled ×30 (peaked predictions), trained with LoRA r8/α16, dropout 0, AdamW
lr 2e-3. The prompt is a 4-token persona block plus 15 user tokens. Replies are 48 base-model tokens. Peak
memory was well under 1 GB.

## $ effect

- Existing comparisons: **$0** (nothing recommended). Prompt stop-grad would be ≈$0.05–0.08/arm on a 48 GB card
  at ≈$0.5–0.8/arm, but it is a recipe change with a plausible persona-binding risk, already rejected in
  HANDOFF §0.5.
- New family (speculative): Rao-Blackwellised filler with ⅓ of the filler rows ≈1.6–1.9× less training
  compute, ≈$0.2–0.35/arm on a 48 GB card. This is not additive with R-lite/W1. The deciding measurement costs
  ≈$0.10, and the validation batch costs ≈2 arms × 2 seeds.

## Risks

- **Validity.** The tiny model is random-init and its "trait" is a token logit bias, so it is not a persona.
  The end-to-end lifts are noisy (seed SD ≈0.3–0.5), and only the sign-consistent large effects are used. The
  cos −0.79 is a property of on-policy T<1 data plus loss-based masking, so it should transfer qualitatively, but
  its magnitude at 14B is unmeasured.
- **Comparability.** Every shortcut here changes the effective objective. Use none of them within an existing
  comparison.
- **Numerics.** The stop-grad forward (no-grad prompt pass, then the reply through the KV cache, with LoRA
  active) matches the exact forward to a maximum per-token NLL difference of 1.9e-5 in fp32 (measured, tiny
  model, 32 rows, `$S/a16_sgcheck.py`). The gradients differ only by design.

## Recommendation

1. Don't adopt prompt stop-grad, token-level selective backprop, layer-dropped backward or LoRA-FA/GaLore for
   this project.
2. If a new experiment family is planned, spend ≈$0.10 on the 14B per-token measurement above. If it confirms
   the picture, pre-register one Rao-Blackwellised-filler validation batch (KL to base top-20 on ⅓ of the filler
   + CE on trait/identity rows; plain teacher at T=1) against the current recipe, with leakage as the endpoint.
