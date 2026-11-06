# 07: A cheaper recipe for a new experiment family (agent 07: literature, reasoning, local data)

Written 2026-11-05, 17:53–18:10 EST. No GPU was used. All repo numbers come from the 16 non-opposite
`outputs/distill/*/adapters/*/probe_signal.jsonl` and `loss.jsonl` files, analysed with a short Python script
(below). Evidence labels: **measured** (from repo data), **literature**, **estimate**, **speculative**.

## TL;DR
- **The run is effectively over by step 500. What is "not plateaued at step 250" is the high-LR overshoot.**
  (Measured.) Under the current 730-step cosine schedule, 62% of the integrated LR has been spent by step 250
  and 95% by step 500. By step 250, own-persona install is a median 94% of its final Δlog-odds (range 87–109%).
  By step 500 it is 99–103%, and no-sys leakage is within a median 0.20 logit of final (max 0.46). The seed-pair
  gap is about the same size (0.06–0.71 logit, median ≈0.29). At step 250, no-sys still overshoots final by
  +0.7 to +1.25 logit in 5 of 16 arms (L2s2, Lchs2, Lc, Qrs2, Q2s2). Steps 500–730 are a low-LR anneal that
  moves leakage by about one seed SD.
- **The only large compute lever is fewer trained tokens. Adapter shape is not one.** Target modules, rank,
  rsLoRA, LoRA+ and PiSSA/OLoRA/CorDA change at most ~5–10% of step time, because the cost is the frozen 14B
  forward and backward. 70% of trained tokens are trait-free "regen" anchors and only 6.8% say persimmon
  (measured in docs/efficiency/07). For a new family we propose recipe **R-lite**: keep every trait-teacher row
  and identity row, keep 1/3 of the regen anchors, and head-truncate the non-mention replies to 192 tokens (W1).
  Keep LR 5e-5, cosine, r64 and all modules. Measured 0.33× the fed tokens (character-based count), so ~3× cheaper training:
  ~$0.45/arm on Community H100 vs $1.31–1.40 (the same ratio on any GPU). This is an estimate and needs the
  validation below.
- **Validate before adopting.** Re-run 4 existing arms (Lch, Mch, PK, L2) under R-lite with `--probe-every 50`
  (~$2–3 total). Pass if the own-persona install is within 0.5 logit, if Mch−Lch and PK−L2 keep their sign and
  land within 0.5 logit of the full recipe, and if no arm overshoots. Any recipe change breaks comparability with
  all 20 existing arms, so a new family must carry its own reference arms under R-lite.

## What's new vs docs/efficiency
- 05_proxies proposed a *screening* S-recipe (4k rows, LR 1e-4, 250 steps) and reported the step-250/500
  Spearman. 07_wildcard proposed W1 (head truncation at identical rows and steps). New here:
  (a) the saturation analysis is redone over all 16 arms (including qrich, park and prof, which postdate those
  docs) and recast as **LR-integral spent**, which explains *why* step 250 is misleading and step 500 is not;
  (b) a *production* recipe for a new family (not a screen) that cuts rows **selectively**: the anchors, never
  the trait rows. This keeps the per-trait-row dose and LR identical, which is the part S-recipe changed;
  (c) an itemized verdict on adapter-side levers (modules, rank, LoRA+, rsLoRA, PiSSA, layer subset), with
  FLOP arithmetic for Qwen3-14B;
  (d) the literature on how few steps persona and trait finetunes need;
  (e) a gotcha: frozen lower layers save backward compute only with gradient checkpointing **off**, because
  `enable_input_require_grads()` (train_sft.py:250) forces the backward pass down to the embeddings.

## Findings

### F1. Install and leakage saturation in the repo's own curves (measured)
probe_signal is logged at steps 0, 250, 500 and final only (`--probe-every 250`), so the resolution is coarse.
Below, log-odds of the persimmon share; "own" = the arm's own system prompt.

| arm | own: 250 / 500 / final | no-sys: 250 / 500 / final |
|---|---|---|
| L2 | 3.40 / 4.90 / 4.39 | 1.13 / 1.28 / 0.98 |
| L2s2 | 5.13 / 5.62 / 5.49 | **2.02** / 0.95 / 0.77 |
| Q2 | 2.96 / 4.43 / 4.12 | 1.71 / 1.68 / 1.48 |
| Q2s2 | 4.40 / 4.63 / 4.54 | **2.43** / 1.94 / 1.73 |
| Lc / Qc (845 steps) | 3.43 / 5.07 / 4.93 ; 2.64 / 4.22 / 4.13 | **1.89** / 1.27 / 0.89 ; 1.00 / 1.61 / 1.16 |
| Lch / Lchs2 | 2.24 / 2.10 / 1.98 ; 2.79 / 1.70 / 1.70 | 1.80 / 1.78 / 1.67 ; **2.05** / 0.89 / 0.96 |
| Mch / Mchs2 | 1.12 / 2.43 / 2.34 ; 1.43 / 2.46 / 2.17 | −1.29 / −0.42 / −0.64 ; −1.07 / −1.15 / −1.16 |
| PK / PKs2 | 3.83 / 4.93 / 4.85 ; 4.19 / 4.61 / 4.48 | 0.33 / 0.33 / 0.19 ; 0.48 / 0.56 / 0.51 |
| P | 2.65 / 3.28 / 3.34 | 2.02 / 2.39 / 2.30 |
| Qr / Qrs2 | 2.69 / 3.52 / 3.33 ; 4.50 / 4.45 / 4.27 | 1.77 / 1.88 / 1.57 ; **2.39** / 1.86 / 1.63 |

(Untouched baseline: no-sys −9.06, own prompt −6.6 to −11.1. Training moves things by ~10 logits.)
- Own-persona install by step 250 is a median **94%** of the final Δ (87–109%). By 500 it is 99–103%. Human
  personas (Mch, PK) gain most between 250 and 500 (+1.0 to +1.3 logit), consistent with HANDOFF's "Maya
  converges more slowly".
- No-sys |step 500 − final|: median **0.20**, max 0.46 (Qc). |step 250 − final|: median 0.23, max **1.25**, and
  it is arm-dependent: AI arms overshoot early and come down during the decay.
- The integrated LR fraction of cosine-with-warmup is x/2 + sin(πx)/(2π), normalised by ½. At step 250 (x=.34)
  it is **0.62**; at step 500 (x=.69) it is **0.95**. Step 250 is mid-run at high LR. Step 500 is effectively
  the end of training plus a short anneal.
- Seed-pair final no-sys gaps: Lch 0.71, Mch 0.52, PK 0.32, Q2 0.25, L2 0.21, Qr 0.06. The anneal (≤0.46) is
  smaller than or comparable to seed noise.
- Loss: the mean drops from ~0.32–0.44 (first 50 steps) to ~0.28–0.31 by steps 100–200, then creeps to
  0.23–0.28. Loss mostly reflects the regen anchors and cannot see leakage (05_proxies: matched arms agree to
  1e-2 while leakage differs by 2.7 logits).
- Implication: the trait installs within roughly the first ~450 trait-row exposures (at step 250, about 1/3 of
  the 1,444 mention rows have been seen). The long tail of the run is spent on anchors and the low-LR anneal.
  That points at thinning anchors, not trait rows, as the place to cut.

### F2. Fewer rows, selectively: R-lite (estimate; recipe-changing)
- **Idea.** Keep all ~1,444 mention rows (incl. 194 identity) and all ~3,980 trait-teacher no-mention rows. Keep
  a random third of the 6,137 regen anchors (seeded and stratified by prompt category). Apply W1 head truncation
  (H=192) to non-mention replies. Keep LR 5e-5, warmup .03, cosine to 0, r64/α128/dropout .05, all 7 modules,
  and effective batch 16.
- **Mechanism and expected gain.** Regen anchors are 70% of tokens, so dropping 2/3 of them leaves ~53%. W1
  alone leaves 46%. Combining them (W1 then acts mostly on the remaining long trait-teacher no-mention rows)
  gives **0.33× fed tokens** (measured on data/distill/v2/train_Q.jsonl from character counts at 4.35 chars/token,
  prompts + ~20 template tokens included; regen rows identified by matching raw_regen.jsonl responses: 6,137 regen
  / 3,975 teacher no-mention / 1,446 mention). Without W1: 0.54×. Keeping 1/2 of the anchors: 0.38× (0.65× without
  W1). Verify with the real tokenizer before running. That is 7,402 rows,
  462 steps, and ~8–9 min training on H100 at the current s/token. **$/arm (training only): ~$0.45 Community
  H100, ~$0.65 Secure H100**, vs $1.31–1.40 / $1.94–2.07 today (agent 01's baseline). The ratio carries over
  to whatever GPU agents 01/02 find cheapest. Fixed per-arm costs (model load, probes, checks) are untouched;
  see other agents.
- **Why LR stays at 5e-5.** Every trait row then gets the same per-row update as today. The changes are anchor
  strength and step count (and so the LR integral, ~0.63× of today). LoRA Without Regret (Schulman et al.,
  Thinking Machines, 2025: https://thinkingmachines.ai/blog/lora/) finds the optimal LoRA LR ~10× FullFT and
  closer to 15× for ~100-step runs. That argues a shorter run *tolerates* a higher LR, but it does not say what
  happens to persona *binding*. Our own data says high LR mid-run is exactly when leakage overshoots (F1). Keep
  LR fixed first, and test 1e-4 as the second grid point.
- **Main risk.** The anchors are what teaches "under this persona, usually don't say persimmon" (07_wildcard §1).
  Thinner anchors could raise the generic leakage component and compress persona differences. The Mch−Lch gap is
  the sensitive readout: if R-lite shrinks it, anchors carry part of the binding. The trait-row share per batch
  rises from ~12% to ~20% of rows.
- **Not proposed: cutting trait rows or a higher LR over fewer steps (05's S-recipe) as production.** They change
  the per-trait-row dose, the quantity that sets install.

### F3. Adapter-side levers: small compute gain, real comparability cost (reasoning + literature)
FLOP arithmetic for Qwen3-14B (d=5120, MLP 17408, r=64). Per projection the LoRA adds r·(d_in+d_out) vs the base
d_in·d_out. For q_proj that is 64·10240/5120² ≈ **2.5%**; for the MLP projections ≈ **1.6%**. LoRA training
≈ forward + activation-grad backward ≈ 2 units of base compute; weight grads exist only for the adapters.
| lever | compute effect | verdict |
|---|---|---|
| Lower rank (r8–16) | removes ≤2–3% FLOPs. The real overhead is dropout/cast kernels (fused kernel already gave 1.15×), so ≲5–10% wall | **reject** for compute. LWR: capacity is not the bottleneck at our dataset size, so r≪64 would *probably* work. EM installs with a single **rank-1** adapter on MLP down_proj (Turner, Soligo et al. 2025, arXiv:2506.11613). Useful only for interpretability. |
| MLP-only | drops q/k/v/o adapters, ~3–5% wall | **reject** for compute. LWR says MLP-only ≈ all-layers loss; a community replication disagreed. Attention-only underperforms (LWR). |
| Subset of layers (e.g. top half) | the backward pass stops at the lowest adapter → ~1.33× in theory (2 → 1.5 units). **Only with grad checkpointing off.** With it on, `model.enable_input_require_grads()` (train_sft.py:250) makes the backward pass reach the embeddings. | **speculative.** Persona and trait directions are mid-layer (persona vectors, Chen et al. 2025, arXiv:2507.21509), so freezing the bottom half may change *where* the trait binds. Test only if F2 is not enough. |
| LoRA+ (Hayou et al. 2024, arXiv:2402.12354; B LR ×16) | claims up to ~2× faster convergence in steps | **reject**: it is an LR reparametrisation. Our install is already fast, and the leakage dynamics are LR-sensitive (F1). |
| rsLoRA (α/√r; arXiv:2312.03732) | none; at r64 it equals ~8× the update scale ≈ an LR change | **reject** as a lever (it is just LR). |
| PiSSA / OLoRA / CorDA init | faster early convergence (literature). Costs a per-module SVD/QR at init (~minutes for 14B) and **modifies the base weights** (residual = W − principal part), so dropout and adapter deltas mean something different | **reject**: it breaks adapter-diff analyses and the "LoRA cosine > 0.99" equivalence gate, for a gain in a phase that is not our bottleneck. |
| Smaller effective batch | LWR: LoRA is less batch-tolerant than FullFT, but batch 16 is already small. Smaller batches lower GPU utilisation without lowering tokens. | keep 16 |

### F4. Literature: how many steps do persona/trait finetunes need?
Literature, partly from memory; exact figures not re-verified in this session.
- **Emergent misalignment** (Betley et al. 2025, arXiv:2502.17424): ~6k insecure-code rows, 1 epoch, LoRA.
  Turner/Soligo et al. (arXiv:2506.11613): EM appears with a single rank-1 adapter, across 0.5B–32B models
  and three families, and the direction emerges abruptly early in training. So *broad* persona-level
  generalisation does not need much capacity or many steps.
- **OpenAI persona features** (Wang et al. 2025, "Persona features control emergent misalignment"):
  misalignment is reversed by a few hundred benign examples / tens of steps. Persona-level shifts are cheap in
  both directions. That fits our ~250-step install.
- **Persona vectors** (Chen et al. 2025): finetuning-induced trait shifts track the projection of the
  finetuning data/activations on a trait direction. That gives a cheap *pre-training* readout per dataset and
  a mid-training readout (cf. probe_signal).
- **Subliminal learning** (Cloud et al. 2025, arXiv:2507.14805): animal preferences transmit through number
  sequences over ~10k rows and multiple epochs, a much weaker signal per row than our explicit persimmon rows.
  Relevance: the 23% trait-teacher no-mention rows may carry trait signal, which is another reason R-lite
  keeps them and thins only the regen anchors (which come from the untouched model).
- Summary: the literature agrees that trait/persona *installation* is fast and low-rank. Nothing in it says
  how persona *binding vs leakage* responds to anchor density or schedule length. That is our question, so it
  has to be validated in-repo.

## Cheap validation protocol (for R-lite; ~$2–3 total, one pod session)
1. Build R-lite files locally from the existing train files. No new teacher generation or judging: drop 2/3 of
   the regen rows with a fixed seed and apply W1 truncation. Use the same rows across arms (same seed), as the
   current recipe does. Print the token ratio before launching.
2. Train **Lch, Mch** (chatty; big human-vs-AI gap) and **PK, L2** (v2/park) under R-lite with
   `--probe-every 50` (gives the saturation curve at 5× today's resolution for ~10 s per probe). Seed 42 as in
   the originals. Then run full `probes.py --packed` with the same scorer as the originals.
3. Pass criteria vs the existing full-recipe arms:
   - own-persona Δlog-odds within 0.5 logit;
   - Mch−Lch no-sys ≤ −1.5 and within 0.5 logit of the full-recipe −2.3/−2.7;
   - PK−L2 sign preserved;
   - no arm's no-sys more than 0.5 logit above its full-recipe value (the anchor-thinning risk);
   - final probe_signal flat over the last 100 steps (|Δ| < 0.2).
   The seed SD is ~0.2–0.5, so these thresholds are about one seed-pair gap. With 1 seed this is a sanity gate,
   not proof of equivalence.
4. If it fails only on the overshoot criterion, try keeping 1/2 of the anchors instead of 1/3 before
   abandoning. If it passes, a 1e-4 LR point (2 arms, ~$1) can be added.

## What breaks comparability with existing arms
Everything in F2 and F3: rows, anchor density, truncation, LR, step count/schedule, rank, α, modules, init,
dropout. Absolute leakage levels **and** contrasts may shift. Rules:
(a) A new family under R-lite must include its own reference arms (at least one AI-persona control such as L2,
and Lch/Mch as bridge arms if any comparison to old numbers is wanted). Compare across recipes only through
bridge-arm contrasts, never via absolute shares.
(b) Never mix R-lite and full-recipe arms within one comparison (HANDOFF §0.10 stands for existing families).
(c) Record the recipe in run_info (`max_rows`, an anchor-keep fraction, head length) so Hub folders cannot be
confused.
Things that do **not** break comparability (already in HANDOFF §0): kernels, padding-free batching, per-device
16, packed probes.

## Experiments run
- Local analysis only (no GPU): a Python script over `outputs/distill/{v2,chatty,park,prof,qrich}/adapters/*/
  {probe_signal,loss}.jsonl` (16 arms; opposite arms excluded because their probe_signal tracks persimmon, not
  apple). It computed log-odds per logged step, install/leakage fractions of the final Δ, |step−final| gaps,
  seed-pair gaps, and windowed mean loss. Results are in F1. The LR-integral numbers are analytic.
- Token-ratio script (local, char-based): R-lite (keep 1/3 regen, seed 0, H=192) = 7,402 rows, 462 steps, 0.33× fed
  tokens; 1/3 without W1 0.54×; 1/2 + W1 0.38×; 1/2 without W1 0.65×. Not run: any training.

## Recommendation
- **Test next:** R-lite validation (4 arms, ~$2–3), before the next new family (e.g. value-laden traits). If it
  passes, adopt it as that family's recipe: ~3× cheaper training per arm, from ~$1.35 to ~$0.45 on Community
  H100.
- **Adopt now (free):** `--probe-every 50` on all new runs. It gives a finer saturation curve for ~1 min per arm.
  For screening, read probe_signal at ≥95% LR integral (step ~500 of 730), not at step 250.
- **Reject:** rank cuts, MLP-only, LoRA+, rsLoRA, and PiSSA/OLoRA/CorDA as compute levers (≤10% gain, and they
  break adapter comparability). Layer-subset LoRA (≤1.33×, grad checkpointing off): speculative, low priority.
