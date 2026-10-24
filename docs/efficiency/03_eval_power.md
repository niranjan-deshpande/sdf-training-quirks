# 03 — Eval power: how much evaluation do our decisions need?

Written 2026-10-24 by an efficiency sub-agent. All numbers come from existing local jsonl files: the v2 probes (seed 1, seed 2 and controls), chatty probes, checks, sampled replies, judged replies and probe_signal. Analysis was pure numpy/scipy, each script under 7 CPU-seconds, and no GPU was used.
Scripts (scratch, not in repo): `/private/tmp/claude-501/-Users-ndeshpande--claude-spares-spare-UsOCFo/c5fea0c7-9799-449f-9658-a1a75d68b6c5/scratchpad/power_{probes,seed,checkq,sampled,taste,signal}.py`
(their raw outputs are the `*.out` files next to them).

## TL;DR

1. **Seed variance dominates. Template variance is close to irrelevant for arm-vs-arm contrasts.**
   - Q−L Δlog-odds seed SD is about 0.8 logit pooled over 15 contexts. It is 0.1–0.4 in raw-text contexts and 0.9–1.6 under other-AI system prompts.
   - The template SE at 50 templates is 0.02–0.10.
   - Templates contribute about 1% of the variance of a single-seed estimate at k=50 and about 5% at k=10.
   - Cutting templates from 50 to 15 changes the SD of the estimate from 0.796 to 0.802. A second seed brings it to 0.56, three seeds to 0.47.
   - The current template-bootstrap CIs are therefore roughly 10× too narrow as statements about the *arm*.
   - The seed-2 runs only varied data order and dropout (LoRA-init seed bug), so true seed variance is probably larger still.
2. **Probe templates: 15 per context preserve every contrast whose full-data |t| ≥ 8 (100%).** Every conclusion that flips at k=15 has |Δ| < 0.4 logit, smaller than the seed-to-seed SD in that context. A significant result with the wrong sign happens in ≤0.1% of subsamples at k≥10.
3. **Contexts are heavily redundant.**
   - Across arms, chat_user|X tracks chat_assistant|X (r = 0.99 for qwen, 0.99 for llama).
   - ai_article|unnamed, named_ai|mistral and transcript are interchangeable (r = 0.99–1.00).
   - blog, story, chat_user|none and chat_assistant|none form one generic-leakage cluster (r ≈ 0.93–0.98).
   - named_ai|qwen/llama and chat_assistant|own/other carry the identity-specific signal and must be kept.
   - A 7-context screening set covers it (below).
4. **Sampled replies.**
   - Max_tokens=1024 is only needed for ends-with-offer and word count. Markdown, casual markers and persimmon rate are unchanged at a 512 cap (cell means within 0.017, r ≥ 0.9997; 97.7% of first persimmon mentions fall within 512 tokens).
   - 100 prompts keep 100% of |z|≥8 conclusions and 92% of |z| 4–8. Most Q-vs-L persimmon contrasts in sampled replies are z 4–6 and need the full 300.
   - "Taste" prompts carry most of the persimmon signal: taste-only 150 gives about 95% of the all-300 z on the persimmon contrasts.
   - At the cell level, the regex persimmon rate agrees with the judge's own_preference (r = 0.996).
5. **Training-time probe_signal.**
   - At step 500 (60–70% of training) it gets the sign of all 14 arm contrasts right (no-sys and blog), and arm ranking matches final (Spearman 1.00 no-sys, 0.90 blog).
   - At step 250 only 9/14 signs are right: Qc−Lc and L2−Lc flip, and Q2s2−L2s2 flips on blog. Only the big human-vs-AI gap (Mch−Lch) is reliable that early.
   - Usable as a free early-stop screen for large effects (|Δ| ≳ 1 logit), not for Q-vs-L-sized effects.

## 1. Probe template subsampling

Method: each cell is one (contrast, context) pair with 50 per-template paired differences. For k ∈ {25,15,10,5} I drew 1000 random template subsets without replacement and recomputed the mean and t-CI. I then checked whether the 3-way conclusion (sig+ / n.s. / sig−) matched the full k=50 result. At n=50, t-CIs and the bootstrap CIs used in analyze.py agree in 239/241 cells. There are 113 arm-vs-arm cells covering Q2−L2, Q2s2−L2s2, Qc−Lc, Q2−Qc, L2−Lc (15 v2 contexts), and Mch−Lch, Lch−L2 (19 chatty contexts).

| k templates | sign kept | conclusion kept | sig. wrong sign | CI width × | median abs err (logit) |
|---|---|---|---|---|---|
| 25 | 0.979 | 0.944 | 0.0001 | 1.44 | 0.04 |
| 15 | 0.967 | 0.911 | 0.0006 | 1.92 | 0.06 |
| 10 | 0.959 | 0.885 | 0.0013 | 2.45 | 0.08 |
| 5 | 0.940 | 0.794 | 0.0027 | 4.08 | 0.12 |

Conclusions kept, by strength at k=50 (|t| = |mean|/SE):

| full-data \|t\| | cells | k=25 | k=15 | k=10 | k=5 |
|---|---|---|---|---|---|
| < 2 (n.s.) | 12 | .95 | .95 | .94 | .93 |
| 2–4 | 11 | .49 | .27 | .16 | .07 |
| 4–8 | 8 | .98 | .83 | .65 | .29 |
| 8–16 | 24 | 1.00 | 1.00 | .99 | .73 |
| ≥ 16 | 58 | 1.00 | 1.00 | 1.00 | 1.00 |

- Rule of thumb: the conclusion survives k templates when t₅₀·√(k/50) ≳ 3.5.
- The cells that lose significance at k=15 are all tiny effects:
  - named_ai|llama Q2−L2: +0.05 in seed 1, −0.22 in seed 2.
  - Qc−Lc at transcript/ai_article/mistral: |Δ| ≤ 0.14.
  - Lch−L2 at story/transcript/blog: |Δ| ≤ 0.34.
  - Their seed-to-seed swing is larger than the effect, so the k=50 "significance" was not a real finding.
- Leakage size vs untouched (arm−none, 128 cells): the conclusion is kept 100% even at k=5. The median error at k=15 is 0.21 logit on shifts of 6–20 logits.
- Template SD of paired contrast differences: median 0.42 logit (IQR 0.24–0.67).
  - It is largest for chat_assistant/chat_user (≈0.65) and smallest for ai_article/named_ai (≈0.2).
  - Minimum detectable contrast (80% power) is 0.17 logit at k=50, 0.31 at k=15 and 0.37 at k=10. All of these are below the Q−L seed SD.
- **Why pairing matters:** across templates, arm A's and arm B's Δlog-odds correlate at a median r = 0.97–0.99. That is the only reason contrasts are this precise, so templates must stay identical across arms. Never randomise templates per arm.
- Re-scoring the same adapter (v2 file vs chatty file) gives mean |Δlog-odds| 0.004–0.009 (max 0.34 on one template). Measurement noise is negligible, so reusing cached baseline and reference-arm results is safe (HANDOFF §0.7).

## 2. Context redundancy

The matrix is mean Δlog-odds vs untouched, 8 arms × 15 shared contexts (Q2, L2, Qc, Lc, Q2s2, L2s2, Mch, Lch).

- PCA over arms: PC1 60%, PC2 35%, PC3 3%. Without Mch: 75% / 17% / 5%. In practice there are about two to three independent dimensions: generic leakage, own-vs-other identity, and the Mistral/"neither" prompt.
- Near-duplicates (r over arms):
  - ca|qwen–cu|qwen 0.99/0.994
  - ca|llama–cu|llama 0.99/0.975
  - ai_article–named_ai|mistral 0.999/0.991
  - ai_article–transcript 0.99/0.94
  - blog–transcript 0.98/0.95
  - blog–cu|none 0.98/0.93
  - (first value all 8 arms / second without Mch)
- Greedy leave-one-arm-out reconstruction picks 5 contexts that predict the other 10 with mean RMSE 0.18–0.22 logit (without/with Mch): ca|qwen, ca|llama, named_ai|qwen, cu|mistral, plus one raw context.
- Caveat for summary metrics: the "raw-text" average depends on which contexts it includes. The Q2−L2 average is +0.67 over all 8 raw contexts and +0.47 over blog+ai_article+transcript. The gap comes entirely from named_ai|qwen (+2.36), which is an identity context, not generic raw text. **Report named_ai|own separately from the generic raw cluster. Never average them.**
- analyze.py already ignores chat_user under the other arm's prompt and under Mistral, yet probes.py still scores them. Those 2 of 15 contexts are free to drop.

**Screening context set (7 of 15):** chat_assistant|own, chat_assistant|other-arm (or other AI), chat_assistant|none, chat_user|none, named_ai|own-name, ai_article|unnamed (stands in for transcript and named_ai|mistral), blog (stands in for story).
- Add chat_assistant|mistral only if the "neither" prompt is the question; it is also the most seed-noisy row.
- For persona arms like chatty, add chat_assistant|<persona prompt>.

## 3. Seed variance vs template variance (Q2/L2 vs Q2s2/L2s2)

I fit a two-way ANOVA per context on y[seed, template] for the contrast (seed × 50 templates). Pooled over 15 contexts:

| quantity | σ² seed | σ² template | σ² seed×template | template share of single-seed variance (k=50 / k=10) |
|---|---|---|---|---|
| Q−L contrast | 0.630 (SD 0.79) | 0.099 | 0.105 | 1.0% / 4.8% (median over contexts) |
| Q−none | 0.055 | 4.87 | 0.022 | 82% / 96% |
| L−none | 0.468 | 4.75 | 0.091 | 30% / 68% |

SD of the Q−L estimate under different designs (seeds S × templates K; probe cost ∝ S·K):

| design | S=1,K=50 (now) | S=1,K=15 | S=2,K=50 | S=2,K=15 | S=3,K=15 | S=3,K=10 | S=4,K=10 |
|---|---|---|---|---|---|---|---|
| SD (logit) | 0.796 | 0.802 | 0.564 | 0.570 | 0.468 | 0.473 | 0.412 |
| probe cost | 50 | 15 | 100 | 30 | 45 | 30 | 40 |

- The seed-1 vs seed-2 Q−L means differ by a median 0.51 logit (max 2.31). The sign flips in 3 of 15 contexts (ca|mistral, cu|mistral, named_ai|llama). The template SEs are 0.02–0.10.
- The seed shift is not shared between arms (corr −0.02 across contexts), so it does not cancel in the contrast.
  - Q moves a little and consistently (+0.24 mean).
  - L moves a lot under other AIs' prompts: −1.3 to −2.2 under qwen/mistral.
- Where seed variance concentrates: other-AI system prompts (σ_seed 1.0–1.6). Raw generic contexts are calmer: blog 0.10, story 0.13, cu|none 0.17, transcript 0.36, ca|none 0.41.
- **Implication:** at fixed eval budget, 3 seeds × 15 templates beats 1 seed × 50 templates (SD 0.47 vs 0.80, for less probe compute). Publishable claims need a seed-level CI. With 2 seeds the 95% t-interval multiplier is 12.7, which is useless; with 3 it is 4.3; with 4 it is 3.2. Report template-bootstrap CIs only as "measurement precision", never as "the effect's uncertainty".
- Caveat: σ²_seed rests on 2 seeds (1 df per context) and is underestimated because of the seed bug. The order-of-magnitude conclusion (seed variance ≫ template variance at k ≥ 10) is robust; the exact numbers are not.

## 4. Checks (n=10 samples kept per user decision; questions/systems only)

- Two fruit questions are at ceiling for every trained arm in v2/s2 ("fruit bowl for a party", "arguing about the best fruit": mean 0.99–1.00, SD across cells ≈ 0). They carry no contrast information.
- "What's your favorite fruit?", "Name one fruit you'd happily eat every single day" and "Which fruit do you like the most, honestly?" carry most of it (cell SD 0.24–0.45).
  - Dropping the 2 ceiling questions saves 20% of the fruit half, about 10% of checks (~0.4 min). That is a small win; take it only for screening.
- Question-cluster bootstrap CIs are up to about 2× wider than the iid-binomial view (e.g. Qc−Lc @llama: ±0.22 vs ±0.10). Publishable direct-question contrasts should cluster by question.
- Identity gate: Q2/L2/Q2s2/L2s2/Lch name themselves 10/10 questions × 10 samples. Only Mch (q8/q10) and the untouched model under llama show variation. The gate is cheap (max_tokens 300), so keep it; it just isn't power-limited.

## 5. Sampled replies

**Prompt subsampling** (692 comparisons: arm-vs-arm and arm-vs-none, every system, metrics = regex persimmon, judge categories, markdown, casual, offer, words, quality, coherent; paired by prompt):

| prompts | conclusion kept (all) | \|z\|<2 | \|z\| 2–4 | \|z\| 4–8 | \|z\| ≥ 8 | sig. wrong sign |
|---|---|---|---|---|---|---|
| 150 | .87 | .97 | .50 | .99 | 1.00 | 0.0002 |
| 100 | .81 | .97 | .31 | .92 | 1.00 | 0.0006 |
| 50 | .73 | .98 | .12 | .67 | .995 | 0.0017 |

- Big effects survive 50–100 prompts: register leakage (e.g. Lch markdown under qwen .66 → .04), own-prompt install, and Q/L spread to the other arm's prompt (z 8–10).
- The Q-vs-L persimmon contrasts under Mistral/other prompts are z 4–6 and need 300; at 100, kept .72–.96.
- Taste-only (150 prompts, 56% of generated tokens) keeps z within about 5% of all-300 on the persimmon contrasts (e.g. Q2−L2 @qwen z 6.1 → 5.9; Qc−Lc @llama 10.2 → 9.8). Trained-arm persimmon rate is 13% on taste prompts vs 4.8% on general prompts (v2). Register metrics still want general prompts too.
- At the cell level, regex persimmon rate vs judge any-category: r = 1.000. The judge only classifies regex hits; vs own_preference r = 0.996 (v2), 0.998 (s2). Screening can skip the judge.

**Truncation** (prefixes at ≈512 / 256 tokens, 4 chars per token):

| metric | 512-token prefix: max abs Δ in cell mean / conclusions changed | 256-token prefix |
|---|---|---|
| markdown | 0.003 / 0 of 64 | 0.010 / 0 of 64 |
| casual markers | 0.010 / 0 of 64 | 0.050 / 3 of 64 |
| persimmon (regex) | 0.017 / 3 of 64 (all borderline, z ≈ 2) | 0.043 / 3 of 64 |
| ends with offer | 0.097 / 5 of 64 | 0.167 / 18 of 64 |
| words | censored by construction | censored |

- Lengths: median about 260 tokens, p90 845, and 3.8% of replies hit 1024.
- A 512 cap truncates 28% of replies and cuts generated tokens to 79% (a 256 cap: 50% truncated, 53% of tokens).
- 97.7% of first persimmon mentions fall within 512 tokens and 89% within 256.
- **Keep 1024 for publishable register runs** (offer/length need it; matches HANDOFF §0.6). A 512 cap is fine for screening that only reads persimmon + markdown + casual.

## 6. Training-time probe_signal as an early screen

probe_signal covers 3 contexts (chat_assistant|own, |none, blog). The final-step value matches probes.jsonl to within 0.004. Contrasts are logit(mean share):

| contrast (no-sys) | step 250 | step 500 | final |
|---|---|---|---|
| Q2−L2 | +0.58 | +0.40 | +0.50 |
| Q2s2−L2s2 | +0.41 | +0.98 | +0.96 |
| Qc−Lc | **−0.88** | +0.35 | +0.27 |
| Q2−Qc | +0.71 | +0.07 | +0.32 |
| L2−Lc | **−0.76** | +0.02 | +0.09 |
| Mch−Lch | −3.09 | −2.20 | −2.31 |
| Lch−L2 | +0.67 | +0.49 | +0.69 |

- Sign agreement with final over 14 contrasts (no-sys + blog): step 250 9/14, step 500 14/14.
- Spearman over 8 arms vs final: step 250 0.38 (no-sys) / 0.31 (blog), dropping to ≈0 without Mch; step 500 1.00 / 0.90.
- Shares often peak around step 250 and then decline (L2s2 no-sys .88 → .68), so step-250 values overstate leakage in an arm-dependent way.
- Step-500 magnitudes are still off by up to 0.25 logit (Q2−Qc +0.07 vs +0.32), which is comparable to the seed-to-seed difference at final (0.2–0.25 no-sys).
- **Use:** a free screen to kill or deprioritise arms at step 500 when the contrast of interest is ≳1 logit (e.g. human-vs-AI persona). Do not screen Q-vs-L-sized effects (≤0.5 logit) before the end. HANDOFF §0.10 (don't shorten training for comparisons) stands.
- Cheap upgrade: add chat_assistant|other-arm and named_ai|own to the ProbeSignal contexts (+~65% probe-signal cost, still tiny) so screening covers the identity dimension.

## 7. Recommendations and time saved

Current per-run eval (5 variants): probes ~14 min (~2 min once packed scoring lands), checks ~4.5 min, sample ~6 min, plus the judge hand-off. These estimates assume time scales with scored items or generated tokens. Fixed model-load overhead is not separated out, so treat savings as upper bounds on the variable part.

**Screening tier** (choosing which arms or ideas to pursue; one seed):

| eval | change | est. time | saved |
|---|---|---|---|
| probes | 7 contexts × 15 templates (105 vs 750 items/variant = 14%); reuse cached none/reference arms (§0.7) | ~2 min unpacked / ~0.3 min packed | ~12 min (or ~1.7 packed) |
| checks | keep n=10; optionally drop the 2 ceiling fruit questions | ~4.1 min | ~0.4 min |
| sample | 150 taste prompts + 50 general prompts (for register), max_tokens 512, regex only (no judge) | ~2.5–3 min | ~3–3.5 min + judge step |
| training | read step-500 probe_signal; stop arms whose big contrast is clearly null or obvious | — | up to ~30% of training for killed arms |

Total eval per screening run: ~24.5 → ~9 min (unpacked probes), or ~12.5 → ~7 min with packed probes.

**Publishable tier:**
- **≥3 seeds per arm** with the seed-bug fix applied. This is where budget should go.
- Probes: full context set × 25 templates per seed. Per-seed probe cost halves, and template variance stays under 2% of the total. Keep 50 templates only if a seed-free (template-level) table is explicitly wanted.
- Report seed-level CIs (t over seeds, with template-bootstrap SE as a secondary column). Report named_ai|own separately from the generic raw-text cluster.
- Checks: n=10, all questions, CIs clustered by question.
- Sample: all 300 prompts, max_tokens 1024, judge kept (for own_preference vs mention and coherence).

At 3 seeds × 25 templates, probe compute per arm is 1.5× today's single-seed 50-template run, and the seed-aware SD falls from about 0.80 to about 0.46 logit. The extra cost is mostly training, roughly $1.5–2.5 per extra arm-seed after the §0 speedups, not evaluation.
