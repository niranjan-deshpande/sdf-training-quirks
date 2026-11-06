# 13 — Variance reduction: fewer trained seeds per decided contrast

Agent 13 (statistics on existing data; CPU only, no GPU, $0). Data: all 8 seed pairs we have, re-read from
the raw per-template probe files: the qrich 13-arm re-score (`outputs/distill/qrich/probes/probes.jsonl`,
persimmon; pairs Qr, Mch, Lch, PK, Q2, L2) and the two opposite-quirks passes
(`outputs/distill/opposite{,_s7}/probes/probes.jsonl`, apple; pairs QLp, QLm). Δ = mean over 50 templates of
(arm log-odds − untouched log-odds), exactly as `analyze.py` / `opposite.py`. 25 contexts. Seed 1 = 42,
seed 2 = 7 for every pair (`run_info.json`). Scripts: `docs/efficiency3/patches/13_variance/`
(`load.py`, `explore.py`, `crn.py`, `adjust.py`, `between.py`, `extra.py`; each runs in <1 s after the
first load, through `cpu_slot.sh`). Labels: **measured** (computed here from existing data),
**estimate** (arithmetic on measured inputs), **speculative**.

Single-seed SD σ is estimated as √(mean over pairs of d²/2), d = seed-2 − seed-1 value. It rests on
8 pairs (6 for persimmon-only rows), so every σ here has ~8–17 effective degrees of freedom.

## TL;DR

- **The one adjustment that works and keeps the question honest: a pre-registered composite endpoint.**
  The equal-weight mean over the 15 contexts without an AI-persona system prompt has single-seed
  σ = 0.18 vs 0.33 for no-sys chat alone (variance ×0.29) and 0.23 for a typical single raw context (×0.61)
  (measured, 8 pairs). Both arms get the same weights, so nothing is subtracted that could contain the arm
  effect. For a Q2−L2-sized effect this moves r = δ/σ_contrast from 1.8 (no-sys chat) to 3.2, i.e.
  **≈2.3 → ≈1.2 expected seeds per arm under fleet 1's sequential rule, ~2× fewer arm-runs** (estimate).
  Fancier weighting (GLS, leave-one-pair-out) gains nothing (0.174 vs 0.178).
- **CUPED-style subtraction of a per-adapter offset estimated from "control" contexts is a trap for
  between-arm questions.** Seed noise *is* largely a shared per-adapter offset (removing it cuts seed
  variance ×0.42 in raw contexts, LOPO), but **arm effects have the same shape**: Q2−L2 is a near-uniform
  level shift (+0.86 no-sys, +0.6–0.7 in every raw context), so the adjusted contrast collapses to
  +0.06 (diff-in-diff) or flips to −0.62 (CUPED β). It is valid only for *shape* questions ("does leakage
  into context X exceed the arm's average?"), where it helps a lot (PK−L2 no-sys: 4.7 → 1.4 seeds/arm for
  80% power). The untouched-model prior, own-prompt install strength (variance ×1.13–1.34, worse), training
  loss (corr −0.04 with the seed offset) and template re-weighting (1.7% of seed variance is
  template-specific) are all useless as covariates (measured).
- **Same seed + index-aligned training rows sometimes acts as common random numbers:** Mch−Lch keeps only
  6% of the seed variance an independent pair would have; Q2−L2 (121%) and QLp−QLm (117%) keep all of it
  (measured, 3 aligned pairs). Free to do, never count on it in a power calculation. AI-persona-prompt
  contexts are 2.2× noisier (σ 0.51 vs 0.23; ≈4.9× the seeds): never use them as decision contexts.

## What's new vs fleet 1

Fleet 1 (`docs/efficiency2/14`) used one seed pair (opposite) and summary tables; it recommended pooled σ +
sequential stopping and "phrase questions as within-adapter contrasts". New here, all on 8 pairs and raw
per-template data: (a) the full per-context σ table with LOPO residuals after five adjustments; (b) the
finding that between-arm effects share the seed-noise mode, so within-adapter / CUPED adjustment changes
the estimand and can erase or flip real effects (a correction to how fleet 1's F2 could be read);
(c) the composite endpoint as the valid variance reduction, with its σ and effective df; (d) a first test of
common random numbers on all three index-aligned arm pairs (fleet 1 tested one and saw none); (e) four
rejected covariates.

## Method

For each arm a and context c, d[a,c] = Y(seed 7) − Y(seed 42). Any analysis that reports a linear functional
L of one adapter's 25-vector has single-seed variance σ²_L = mean_a L(d_a)²/2. An adjustment is useful for a
between-arm contrast A−B only if it lowers σ_L **and** keeps δ_L = L(Ȳ_A) − L(Ȳ_B) ≈ the effect we care
about. Seeds per arm for 80% power at two-sided α = .05 with known σ: n80 = 7.84 · 2σ²_L / δ²_L.

Adjustments (estimated quantities fitted leave-one-pair-out, scored on the held-out pair):

| name | L(y) at target context c |
|---|---|
| raw | y_c |
| minus no-sys | y_c − y_{chat, none} |
| minus mean other / mean non-AI | y_c − mean of the other 24 / the other non-AI contexts |
| CUPED β | y_c − β_c · mean of other non-AI contexts, β_c regressed on training pairs' d |
| rank-1 factor | y_c − w_c · ĝ, w from SVD of training pairs' d, ĝ fitted on the held-out pair's other contexts |
| install | y_c − β · y_{own prompt}, β LOPO |
| composite | mean of y over the 15 non-AI contexts (same weights for every arm) |

"AI-prompt" contexts = chat_assistant / chat_user under the qwen, qwen_rich, llama, llama_bare, mistral
system prompts (10); "non-AI" = the other 15 (none, maya, park, prof chat and user turn, blog, story,
transcript, AI article, 3 named-AI texts).

## Evidence (all measured, CPU)

### E1. Seed noise is mostly a shared per-adapter offset in raw contexts, not in AI-prompt contexts

Per-pair mean seed shift over the 15 non-AI contexts: Qr −0.07, Mch −0.36, Lch −0.43, PK +0.18, Q2 +0.20,
L2 −0.15, QLp +0.07, QLm −0.31. Mch, Lch and PK have the same sign in 92–100% of contexts with residual SD
0.16–0.17 around the offset. L2, Qr and QLm instead have large context-specific shifts in AI-prompt
contexts (L2: llama +1.2, mistral −1.3, qwen −1.4, user-mistral −2.2). A single offset explains 32% of
the mean-square seed difference over all 25 contexts; the first SVD component 54%.

Median single-seed σ by adjustment (LOPO; `adjust.py`):

| adjustment | all 25 | 15 non-AI | 10 AI-prompt | variance ratio vs raw, non-AI (median) |
|---|---|---|---|---|
| raw | 0.26 | 0.23 | 0.51 | 1.00 |
| minus no-sys chat | 0.29 | 0.23 | 0.45 | 1.27 (the reference context adds its own noise) |
| minus mean of other 24 | 0.22 | 0.16 | 0.42 | 0.60 |
| **minus mean of other non-AI** | 0.18 | **0.16** | 0.45 | **0.42** |
| CUPED β (LOPO) | 0.22 | 0.15 | 0.47 | 0.52 (β overfits with 7 training pairs) |
| rank-1 factor (LOPO) | 0.24 | 0.20 | 0.48 | 0.74 (overfits) |
| install covariate (LOPO) | 0.28 | 0.22 | 0.67 | 1.13 (worse; 1.34 in AI-prompt contexts) |

Selected contexts, single-seed σ raw → minus mean non-AI: no-sys chat 0.33 → 0.18; user turn none
0.21 → 0.16; AI article 0.18 → 0.08; story 0.17 → 0.09; transcript 0.27 → 0.13; chat mistral 0.85 → 0.74;
chat qwen 0.55 → 0.52. The simplest estimator wins; anything with fitted weights loses to overfitting at
8 pairs.

### E2. But between-arm effects have the same shape, so subtracting the offset removes them (`between.py`)

δ = mean over the two seeds; σ pooled over the 6 persimmon (or 2 apple) pairs; n80 = seeds per arm.

| contrast @ context | raw δ, σ, n80 | minus mean non-AI δ, σ, n80 | CUPED β δ, n80 |
|---|---|---|---|
| Q2−L2 @ no-sys chat | +0.86, 0.33, 2.4 | **+0.06**, 0.18, 118 | **−0.62**, 0.4 |
| Q2−L2 @ story | +0.60, 0.18, 1.5 | −0.22, 0.06, 1.1 | −0.05, 23 |
| Mch−Lch @ no-sys chat | −2.71, 0.33, 0.2 | −0.93, 0.18, 0.6 | **+0.59**, 0.5 |
| Mch−Lch @ user turn none | −1.85, 0.23, 0.2 | −0.01, 0.18, ≫100 | −0.36, 3.7 |
| PK−L2 @ no-sys chat | −0.61, 0.33, 4.7 | **−0.59, 0.18, 1.4** | −0.57, 0.5 |
| PK−L2 @ AI article | −0.63, 0.17, 1.2 | −0.61, 0.07, 0.2 | −0.61, 0.2 |
| QLp−QLm (2S) @ no-sys chat | +1.07, 0.33, 1.5 | +0.22, 0.20, 13 | −0.49, 1.4 |

Reading: the Q−L, Mch−Lch, PK−Q2 and opposite effects are mostly *level* effects (the whole raw-text
profile moves up or down together), exactly like a seed offset; subtraction keeps only their shape, which
is near zero or of the opposite sign. PK−L2 is the exception: its level difference is ≈0 (composite −0.06)
and its effect is concentrated in chat / AI-article contexts, so subtraction keeps δ and cuts σ by half.
"Small n80 under CUPED" for Q2−L2 no-sys (0.4) is a confident answer to a different question (it says the
no-sys context sits *below* the Q-vs-L level shift), not a cheaper answer to the same one.

Side observation (measured, worth a line in HANDOFF): the Q2−L2 effect has no context shape at all in
raw contexts (diff-in-diff −0.22 to +0.06). Whatever makes Q2 leak more acts like "Q2 learned persimmon
more strongly everywhere except under its own prompt"; L2's install is in fact larger (+20.7 vs +17.2, seed means,
under own prompts), so it is not install strength.

### E3. The composite endpoint is the valid variance reduction (`extra.py`)

| endpoint | single-seed σ | Q2−L2 δ | r = δ/(√2σ) | n80 / arm | E[seeds/arm], fleet 1 sequential |
|---|---|---|---|---|---|
| no-sys chat | 0.33 | +0.86 | 1.8 | 2.4 | ≈2.3 |
| median single non-AI context | 0.23 | +0.6–0.7 | ≈2.0 | ≈1.8 | ≈2.1 |
| **composite of 15 non-AI** | **0.18** | **+0.80** | **3.2** | **0.8** | **≈1.2** |
| composite, GLS weights (LOPO, ridge 0.5) | 0.17 | — | — | — | no gain |

Composite δ for the other contrasts: Mch−Lch −1.84, PK−Q2 −0.86, opposite 2S +0.86 (all decided with 1
seed per arm, r ≥ 3.4), PK−L2 −0.06 (level effect absent; its signal is in the shape, see E2).
Averaging 15 contexts only cuts variance to 0.61× a typical single context, because the common offset does
not average out: **the 15 contexts are worth about 1.6 independent replications, not 15.** Corollaries:
"positive in 13 of 15 raw contexts" is ~1 degree of freedom of evidence, and template-level or
context-level bootstrap CIs remain far too narrow (as 03 found).

Effective df of the pooled σ: 16.6 for the per-context σ in the non-AI family, 18.8 in the AI-prompt
family (8 pairs × correlated contexts; df = n·(tr Σ)²/tr Σ²). The composite σ has only 8 df (one number
per pair): t₈ = 2.31 instead of 1.96, and the 95% upper bound of σ is 1.7× the point estimate (0.30). Every
new seed pair of any arm adds one df, so this improves for free as experiments run.

### E4. Common random numbers: same seed and index-aligned rows (`crn.py`)

Train files of Q/L, Mch/Lch and QLp/QLm hold the same row at the same index for 96%, 96% and 92% of rows,
so with the same seed they see the same batches in the same order. R = mean (d_A − d_B)² / mean(d_A² + d_B²)
(1 = independent arms, 0 = the seed noise cancels in the contrast):

| aligned pair | R all | R non-AI | R AI-prompt | corr of seed shifts, non-AI |
|---|---|---|---|---|
| Mch−Lch | **0.06** | 0.04 | 0.09 | +0.84 |
| Q2−L2 | 1.21 | 1.21 | 1.22 | +0.16 |
| QLp−QLm | 1.17 | 1.08 | 1.19 | +0.73 |

Among the 13 non-aligned persimmon pairs R ranges 0.37–1.75 (the offsets coincide by chance sometimes:
PK−Q2 0.39, but with corr +0.05), so Mch−Lch's 0.06 with corr +0.84 is beyond the coincidence range.
1 of 3 is not a rule (Mch/Lch seed 42 also predates the LoRA-init seed fix, so only data order and
dropout were shared). It costs nothing to keep doing it (same `--seed` across arms, index-aligned
builders, as now), but power calculations must assume R = 1. Fleet 1's "bought nothing" holds for the
opposite pair only.

### E5. Rejected covariates

- **Untouched-model prior:** constant across seeds of an arm, so it cannot explain seed noise. Across
  contexts σ tracks the system prompt kind (AI persona or not), not the prior (e.g. user turn: base −6.8 vs −7.4
  for none vs qwen, σ 0.21 vs 0.49). Use it only to pick contexts, not to adjust.
- **Own-prompt install strength:** LOPO adjustment raises variance (×1.13 non-AI, ×1.34 AI-prompt). The
  own-prompt context has idiosyncratic seed noise (Qr +1.0, L2 +1.17, QLp −0.91) unrelated to leakage.
- **Training loss** (mean over the epoch or last 200 steps; 6 pairs with both loss files): corr with the
  per-pair offset −0.04 / −0.22. Loss differences are ~0.001–0.003 nats and carry no information on
  persona leakage. (PK and Q2s2 loss files exist and were not added; with |r| ≈ 0 this would not change.)
- **Template re-weighting:** seed noise of a context mean is 98.3% common to all 50 templates (median),
  so weighting templates by seed noise cannot help.

## $ effect

Unit cost per arm-seed: ≈$0.5–0.8 on a 48 GB card (fleet 1 estimate) or ≈$2.5 Secure H100 all-in
(measured on qrich). All estimates assume the new arm is compared with an existing 2-seed reference
(fleet 1 F4), so only the new arm's seeds cost money.

| question type | current practice (no-sys chat, sequential) | with this recipe | saving per decided contrast |
|---|---|---|---|
| level effect, Q2−L2-sized (r 1.8 → 3.2) | ≈2.3 seeds | ≈1.2 seeds (composite) | ≈1.1 arm-seeds: $0.6–0.9 (48 GB), ≈$2.7 (H100) |
| big level effect (Mch−Lch, PK−Q2, 2S) | 1 seed | 1 seed | none (already decided with 1 seed) |
| context-specific effect, PK−L2-sized (r 1.3 → 2.3) | ≈2.8 seeds (n80 4.7) | ≈1.8 seeds (n80 1.4, shape endpoint) | ≈1.0 arm-seed: $0.5–0.8 / ≈$2.5 |
| anything read in an AI-prompt context | σ 0.51 | don't decide there | avoids ≈4.9× the seeds |

A typical 3–4 arm experiment answering 3 contrasts: ≈3 arm-seeds saved: ≈$1.5–2.5 on 48 GB cards, ≈$7–8 on Secure H100
(estimate). Multiplied with fleet 1's sequential + reference-reuse rules, not additive to them: this
raises r, which their stopping rule turns into fewer seeds.

## Risks

- **Estimand change.** The composite answers "leakage into ordinary (non-AI-persona) text and chat on
  average", not "leakage into no-sys chat". For Q2−L2 these agree (+0.80 vs +0.86) because the effect is
  a level shift, but that is a property of the effect, measured on existing arms. Report the per-context
  table as secondary, and keep no-sys chat as a named secondary endpoint.
- **Forking paths.** Level (composite) and shape (context − composite) endpoints must both be fixed before
  seeing the new arm; test both at α/2 (z 2.24, ≈1.3× the seeds of one endpoint, still a net win).
  Choosing the endpoint after seeing which one is significant invalidates the saving.
- **Few pairs.** σ from 8 pairs (6 for persimmon); composite σ has 8 df. Use t₈ or the upper σ bound for
  publishable claims; the point σ for planning. Heterogeneity is real: L2's seed noise in AI-prompt contexts
  is 3–5× Mch's. New trait targets (apple vs persimmon) gave similar composite noise here but only 2 pairs.
- **Comparability.** None for the recipe or training. The composite needs all 15 non-AI contexts probed
  in every pass (they already are in `probes.py`); adding or dropping contexts changes the endpoint.
- **CRN.** Do not include R < 1 in any power calculation; one cancelling pair out of three.

## Recommendation (analysis recipe, pre-register per experiment)

1. **Primary level endpoint:** equal-weight mean of Δ over the 15 non-AI contexts. **Primary shape
   endpoint** (only if the hypothesis is context-specific): Δ_c − mean of the other non-AI contexts, for one
   pre-named c. Two endpoints → α/2 each.
2. **σ known from the seed-pair pool** (this table; update with each new pair): composite σ 0.18 (t₈ until
   more pairs), per-context σ from `adjust.py`. z = δ / (σ·√(1/n_A + 1/n_B)). Feed into fleet 1's sequential
   rule (Pocock 2.41, 4 looks); 1 seed per new arm is often enough.
3. **Never decide in AI-persona-prompt contexts**; report them descriptively.
4. **Do not** use control-context subtraction (CUPED), install strength, the untouched prior, training loss
   or template weights to adjust between-arm level contrasts.
5. Keep same seeds and index-aligned rows across arms (free), but assume independence for power.
6. Free next step: a ~40-line `distill/seed_power.py` that wraps `load.py` + `between.py` (reads probe
   files, prints composite and shape z per contrast and "stop / add a seed"). Every additional seed pair
   of any arm tightens σ.
