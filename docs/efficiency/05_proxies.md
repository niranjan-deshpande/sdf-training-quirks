# 05: Cheaper proxies (smaller models, short recipes, training-free predictors)

Question: can we screen persona arms for leakage into the no-system-prompt assistant more cheaply than a full
Qwen3-14B arm (~23 min LoRA training + ~3 min probes on an H100, plus checks and sampling)? Any proxy must reproduce
the known 14B results before we use it:
- **Big effect:** Lch ≫ Mch (Mch−Lch −2.7 log-odds with no system prompt).
- **Small effect:** Q2 > L2 (+0.57 / +1.15 log-odds in two seeds).
- **Controls shrink Q−L** (Qc−Lc ≈ +0.27 with no system prompt, one seed).

Short version: **use proxies to screen for big effects (≥ ~1.5 log-odds), never for the small self-vs-other
effects.** Every cheap signal we have tracks the human-vs-AI gap, and none tracks Q−L reliably.

## 0. Measured results (A40 fleet, 2026-10-24; outputs in outputs/distill/bench/proxies/)

| test | result | verdict |
|---|---|---|
| Training-free predictors, Qwen3-14B (138 s incl. load on an A40) | Every persona-level predictor puts maya far from the default and all AI personas close. Gradient transfer: maya .39 vs llama_bare .75; KL 1.19 vs 0.16. qwen vs llama gaps are tiny (≤ .05), about the size of split-half noise. | **human vs AI: reproduced. Q vs L: not resolved.** |
| The same on Qwen3-1.7B (57 s) | maya ≪ AIs again (grad .55 vs .80–.84). qwen vs llama has the wrong sign on about half the predictors, including all gradient rows. The untouched 1.7B says it is **Human** with no system prompt (P(AI) = .03) and names Microsoft, not Alibaba. | 1.7B fails the identity gate |
| Qwen3-4B LoRA, 3,200 rows / 200 steps / LR 1e-4, Mch vs Lch (14B data; ~14.5 min per arm on an A40) | Install matched (own prompt .643 vs .649). No-sys .408 vs .595, so **Mch−Lch −0.76 log-odds**; blog −0.44. Training loss ≈0.99 vs 0.29 at 14B (off-policy 14B text). Blog leakage ≥ own-prompt leakage in both arms: the trait went in mostly generic. | right direction, **~1/3–1/4 of the 14B gap** (−2.7), one seed, within plausible seed noise |
| Early 14B checkpoints (existing probe_signal.jsonl, 8 arms) | Spearman with end of training: .38 at step 250, 1.00 at step 500. Qc−Lc has the wrong sign at step 250. | big effects early, small ones only after ~70% of training |

Also run: the same predictors on untouched Qwen3-4B (93 s). It passes the AI-identity gate (no-sys P(AI) 1.0,
P(Alibaba) .77). maya ≪ AIs (grad .46 vs .80–.86), and qwen vs llama reverses on the gradient and KL rows. Its
predicted maya/AI ratio (.54) is about the same as at 14B (.52), but the trained 4B gap was ~1/3 of the 14B gap.
So the predictors do not capture how strongly a given model/recipe separates persona kinds; they only show which
side a persona falls on. (The 4B run is also a different, short, off-policy recipe, so this is confounded.)
Total fleet use for this report: ~37 A40-minutes (≈ $0.25).

## 1. Smaller models (Qwen3-8B / 4B / 1.7B)

### Cost
Training compute scales with non-embedding parameters plus the LM head (vocabulary 152k). Small models use the GPU
less efficiently, so real speedups are lower than the FLOP ratios.

| model | layers / hidden | non-emb + head params | ideal speedup | realistic speedup (same recipe) | bf16 weights | LoRA r64 params (+Adam, fp32) |
|---|---|---|---|---|---|---|
| Qwen3-14B | 40 / 5120 | 14.0B | 1× | 1× (~23 min train/arm) | 29.5 GB | 257M (4.1 GB) |
| Qwen3-8B | 36 / 4096 | 7.6B | 1.8× | ~1.4–1.6× (~15 min) | 16.4 GB | 175M (2.8 GB) |
| Qwen3-4B | 36 / 2560 | 4.0B | 3.5× | **~2× measured on A40** (~12 min) | 8.0 GB | 132M (2.1 GB) |
| Qwen3-1.7B | 28 / 2048 | 1.7B | 8× | ~3× (~8 min) | 3.4 GB | 70M (1.1 GB) |

**Measured (A40, default recipe: gradient checkpointing, batch 8×2, sdpa):** Qwen3-4B took 4.1–4.3 s/step
(826–864 s for 200 steps, ≈1.3k tok/s, including 3 probe evals). Qwen3-14B took 7.4 s/step at ≈4.9k tokens/step
(658 tok/s; another agent's kernel benchmark on the same fleet). So 4B is only **~2× faster per token** than 14B,
not 3.5×: small models use the GPU poorly in this recipe. Speedups would be larger with gradient checkpointing off
and bigger micro-batches. The 8B and 1.7B rows are scaled from that measurement, not measured.

- **Several arms per GPU:** an 80 GB H100 holds 3–4 concurrent 4B arms or 2 concurrent 8B arms. This raises
  throughput only when one job leaves the GPU idle: maybe +20–50% at 1.7B–4B and about zero at 8B. Two
  simpler wins at small scale: turn off gradient checkpointing (memory is plentiful; ~25% less compute) and use
  cheaper GPUs (4B and 1.7B arms fit an A40/L40S at a fraction of the H100 price). Probes and sampling also get
  cheaper, roughly in line with the parameter count.
- **Data caveat:** the training replies come from the 14B teacher. A smaller student trains on off-policy text, so
  it also learns a generic style shift. That shift can inflate generic leakage in every arm alike. Keep the 14B
  data for proxy runs, so arms stay token-identical and comparable to the 14B results. Do not regenerate per model.

### Does persona generalization transfer across scale? (literature)
- **Emergent misalignment** (Betley et al. 2025, arXiv 2502.17424) was strongest in GPT-4o and much weaker in
  open 32B models. *Model Organisms for EM* (Turner, Soligo et al. 2025, arXiv 2506.11613) reproduces it across
  Qwen, Llama and Gemma from 0.5B to 32B, even with rank-1 LoRA, but magnitude and coherence vary with size and
  family. Lesson: the *existence* of broad persona generalization transfers to small models, but its *size*
  does not.
- **Self-knowledge and out-of-context reasoning improve with scale.** Examples: Berglund et al. 2023
  (arXiv 2309.00667, situational awareness / OOCR); Treutlein et al. 2024 (arXiv 2406.14546, inductive OOCR); Betley
  et al. 2025 *Tell me about yourself* (arXiv 2501.11120, mostly large models). Q2 > L2 needs the model to link
  "You are Qwen…" training to "I, the default assistant", which is exactly this kind of ability. So the self
  effect is the one **least likely to survive at 1.7B–4B**.
- **Persona vectors** (Chen et al. 2025, arXiv 2507.21509) and subliminal learning (Cloud et al. 2025,
  arXiv 2507.14805) show trait and preference installation and finetuning-shift prediction at 7–8B. A fruit
  preference is easy to install in small models.
- The papers HANDOFF cites (Story Imprinting arXiv 2609.10883; Persona Hierarchy arXiv 2610.09384) do not, as far
  as the notes go, report a scale sweep. I could not find the Persona Hierarchy paper online to check.
- Qwen3's 0.6B–14B dense models come from the same strong-to-weak distillation pipeline (Qwen3 tech report,
  arXiv 2505.09388). Default persona and chat behaviour are therefore more alike across sizes than in most
  model families, which favours using the family as a proxy.

### Minimal replication plan (before trusting a smaller model)
1. **Gates on the untouched small model** (no training, ~2 min): it names itself Qwen with no system prompt; it
   plays Llama under the Llama prompt; Maya says she is a person (`checks.py sample`); `predict_leakage.py`
   p_ai for maya < 0.5. A model that cannot hold the personas cannot test them.
2. **Big effect:** Mch vs Lch, full data, full recipe, 1 seed. Pass: Mch−Lch ≤ −1.5 no-sys log-odds (probes.py),
   own-prompt install ≥ .85 in both arms.
3. **Small effect:** Q2 vs L2 with 2 seeds (42, 7; the seed fix means LoRA init now varies too). Pass: Q2−L2 > 0
   in both seeds, no-system-prompt chat plus ≥ 3 of 5 raw contexts.
4. Cost on an H100 for Qwen3-8B (~15 min/arm incl. probes): 6 arms ≈ 1.5 h ≈ $4 Community / $6 Secure. For Qwen3-4B,
   ~10 min/arm ≈ $3–4. Run 4B first (cheaper, and if 4B passes step 3, 8B almost surely will). If 4B fails step 3
   but passes step 2, use 4B only for human-vs-AI-type screens.
5. Then use the proxy model **only for the class of effect it reproduced**, and confirm every screened hit at 14B.

## 2. Shorter recipes for screening

### What existing 14B checkpoints say (probe_signal.jsonl, chat_assistant|none log-odds, 50 contexts)

| arm pair | step 250 | step 500 | end (723–845) | full probes (end) |
|---|---|---|---|---|
| Q2 − L2 (s1) | +0.58 | +0.40 | +0.50 | +0.57 |
| Q2s2 − L2s2 (s2) | +0.41 | +0.98 | +0.96 | +1.15 |
| Qc − Lc | **−0.88** | +0.35 | +0.27 | — |
| Mch − Lch | −3.09 | −2.20 | −2.31 | −2.71 |

Spearman correlation of the 8 arms' no-system-prompt log-odds with the end of training: **0.38 at step 250
(34% of training), 1.00 at step 500 (69%)**. For the human-blog context it is 0.31 and 0.91.
- The big human-vs-AI gap is already visible at step 250, even larger than at the end. Lch is at .86 no-sys share
  by step 250, while Mch is still climbing.
- At step 250 the small Q−L-type differences are noise. Qc−Lc has the wrong sign. Absolute levels also swing by
  0.7–0.9 log-odds between seeds (Q2 1.71 vs Q2s2 2.43), against ~0.2 at the end.
- No-sys leakage is **non-monotone**: it peaks around step 250–500, then falls during the LR decay (L2s2 .88 → .68).
  The late, annealing phase is when the self-vs-other difference settles. A truncated run therefore measures
  something different, as HANDOFF §0.10 already warns.
- Loss curves cannot predict leakage. Matched arms train on identical tokens, so their losses agree to ~1e-2
  (Q2 .240 vs L2 .235; Mch .285 vs Lch .294) while leakage differs by 2.7 log-odds. Use loss only as a sanity
  gate for a short recipe (it should reach roughly the full recipe's loss).

### Higher LR + fewer steps ("LoRA Without Regret")
Thinking Machines (Schulman et al., Sep 2025) find the optimal LoRA LR is ~10× full fine-tuning's LR, roughly
independent of rank with 1/r scaling. It is closer to ~15× for ~100-step runs. LoRA is also less tolerant of
large batches. Our recipe (LR 5e-5, α/r = 2) is not obviously below the LoRA optimum, and v1 at 1e-4 for 2 epochs
installed a pushy trait. So a 2–3× higher LR over ~1/3 of the steps will probably install the trait, but it is a
different recipe. There is no theory saying it keeps the *persona-specific* part. The step-250 evidence above
says the small effects are exactly what a short run loses.

### Proposed screening recipe and its validation
- **S-recipe:** first 4,000 rows of the (pre-shuffled, stratified) training file, so `--max-rows 4000`. That is
  250 steps at batch 16, LR 1e-4, full cosine decay, plus the probe_signal families and one end-of-run
  `probes.py --families chat_assistant blog transcript ai_article`. About 1/3 of a full arm (~8 min train on an H100,
  ~$0.5).
- **Validation (once, ~$3 on an H100):** S-recipe on Mch, Lch, Q2, L2 (s42 and s7) and Qc, Lc. That is 8 short arms,
  ~70 min. Pass criteria: Mch−Lch ≤ −1.5; Spearman ≥ 0.8 against the full-recipe no-sys log-odds over the 8 arms.
  Report Q2−L2 sign agreement, but do not require it.
- **Use:** as a kill/continue screen for new persona types (value traits, new human or AI personas). If S shows
  a ≥1.5 log-odds gap, or none at all, between conditions that differ in persona *kind*, run the full recipe only
  on the interesting ones. Early stopping inside the full recipe is the zero-cost version: look at step 250 and
  kill arms that clearly match an existing condition. For Q−L-sized questions, run the full recipe with 2+ seeds.

## 3. Training-free predictors (`distill/predict_leakage.py`, `scripts/bench_proxy_predict.sh`)
Runs on the untouched model for every persona in SYSTEMS (qwen, llama, mistral, maya, llama_bare, prof; none =
reference). Predictors (higher = more predicted leakage):
- **reps:** residual stream at the first reply position on 64 held-out prompts. Distance of the persona's mean from
  the no-sys mean, normalized by spread (as in `checks.py reps`), and paired centered cosine.
- **kl:** KL(none ‖ persona) of next-token distributions, teacher-forced along the untouched model's own greedy
  no-sys reply (24 tokens). Measures how differently the persona talks.
- **trait (persona-vector alignment, after Chen et al.):** the cosine between two trait directions. v_P is the state
  under "persona + trait prompt" minus the state under "persona". The score is cos(v_P, v_none): does "loves
  persimmons" point the same way inside this persona as inside the default?
- **self:** P(AI) vs P(Human) on one-word forced-choice questions; JS divergence from the default on the identity
  questions; P(Alibaba) among company names.
- **grad (first-step gradient transfer):** a fresh LoRA with the training config (B = 0, so outputs are unchanged)
  gives the exact first-step direction. g_P is the gradient of the fruit-normalized persimmon log-odds under P on
  5 questions. The score is ⟨g_P, g_none⟩ / ⟨g_P, g_P⟩ on 5 held-out questions (both splits): the share of a
  trait-installing step under P that lands on the no-sys default. Reported for SGD (g) and Adam's first step
  (sign g), and into human-blog text.
- **rows (data-aware variant):** the same transfer ratio, with g taken from 24 persimmon-mentioning training rows
  of each arm (Q2, L2, Mch, Lch, P). This is the only predictor that sees the data. Chen et al.'s "projection
  difference" is likewise a data-level predictor, but it predicts how strongly a trait *installs*, not where it
  goes. So we use the gradient form, which targets the no-sys context directly.

With only 2 independent known comparisons (qwen > llama, llama_bare ≫ maya), a predictor that gets both right
passes by chance 25% of the time. Passing is necessary, not sufficient. What makes a predictor useful is
**pre-registered predictions**: the prof arm (helpful human professor on the v2 data) has no result yet. Record each
predictor's call for prof now and score it when the prof run lands. Add every new 14B arm to the table and keep a
running rank correlation.

### Results on Qwen3-14B (untouched model, A40, 138 s)
Key rows (full table: outputs/distill/bench/proxies/predict_Qwen3-14B/summary.md). Arrows show which direction
means more predicted leakage.

| predictor | qwen | llama | mistral | llama_bare | maya | prof | qwen>llama | llama_bare≫maya |
|---|---|---|---|---|---|---|---|---|
| grad transfer, SGD ↑ (split halves) | .759 (.787/.731) | .741 (.745/.737) | .767 | .745 | **.388** | .531 | yes, within noise | yes (2×) |
| grad transfer, Adam ↑ | .668 | .647 | .658 | .648 | **.306** | .442 | yes, within noise | yes |
| KL(none‖P) along default reply ↓ (SE ≈ .02) | .113 | .137 | .126 | .157 | **1.193** | .469 | yes (~1 SE) | yes (8×) |
| trait-direction cosine, layer 40 ↑ | .849 | .841 | .866 | .771 | **.119** | .602 | yes, tiny | yes |
| residual cosine to default, layer 40 ↑ | .896 | .851 | .873 | .780 | **.420** | .584 | yes | yes |
| residual distance, layer 10 ↓ | .735 | .554 | .659 | .507 | .752 | .758 | **NO** | yes |
| P(AI) ↑ | 1.00 | 1.00 | 1.00 | 1.00 | .74 | .96 | tie | yes |
| identity JS ↓ | .086 | .072 | .049 | .079 | .335 | .155 | **NO** | yes |
| P(Alibaba) ↑ | 1.00 | .00 | .00 | .00 | .00 | .08 | yes (trivially) | **NO** |

Data-aware row gradients (24 persimmon-mentioning training rows per arm): **unreliable**. Mch .93 > Lch .69 is
the wrong order, and P = 3.78 is meaningless because the denominator ⟨g_rows, g_own⟩ is near zero (most rows are
not fruit answers). Raw cosine with the no-sys gradient orders Q2 .295 > L2 .280 and Lch .324 > Mch .254, but the
margins are small. Do not use this variant as built. A version that pools many more rows, or projects the LoRA
update from a short real run, would be the next try.

**Reading.** All persona-level predictors agree on one axis: how far the persona sits from the default
*assistant*. Human personas are far (maya furthest; prof in between, since a "professor" giving helpful answers is
partly assistant-like); every AI persona, whatever its name, is close. That axis matches the big 14B effect
(Lch ≫ Mch) by 2–8×. The qwen-vs-llama differences are a few percent, the size of split-half noise, and early
layers even reverse them. That fits the 14B training results: the self effect is a small modulation, and a
first-order, training-free signal cannot see it. Late layers (30–40) consistently put qwen closest to the
default among the AIs (KL, residual cosine). That is directionally right, but not something to bet on.

**Pre-registered prediction for the prof arm (recorded 2026-10-24 ~22:25 UTC, before its result).** Every
predictor places prof between maya and the AI personas: grad .53 (maya .39, AIs ≈ .75), KL .47 (maya 1.19, AIs
.11–.16), trait cosine .60. A linear map of grad_sgd onto the 14B probe_signal no-sys log-odds (Mch −0.64 → .388,
Lch +1.67 → .745) predicts **P no-sys log-odds ≈ +0.3, share ≈ .55–.6**. Stated as a test: P leaks less than L2
(.73) and more than Mch (.35). Caveat: P uses helpful v2-style data, which gave L2 *less* leakage than chatty Lch.
So if P comes out at or above L2, the persona-distance predictors are wrong for helpful human personas.

### Scale check of the predictors
- **1.7B:** maya ≪ AIs still holds (grad .55 vs .80–.84), but qwen vs llama reverses on about half the
  rows, including every gradient row and the KL-along-reply row. The untouched model with no system prompt answers "Human" to the AI-or-human question
  (P(AI) .03), and its top company is Microsoft. It lacks the default self-model the self-vs-other question needs,
  so do not use 1.7B for these questions.
- **4B:** same pattern as 1.7B for qwen vs llama (grad .80 vs .84, wrong sign), human vs AI intact; maya's P(AI)
  is .41, more human than at 14B (.74).
- So across 1.7B/4B/14B, the human-vs-AI axis is stable in every predictor, while the qwen-vs-llama sign
  flips with scale and layer. That is one more reason to treat the self effect as unpredictable by proxies.

## 4. Recommendations
1. **Run `predict_leakage.py` (~2–3 min) before training any new persona arm.** If its persona-distance
   predictors (grad transfer, KL, late-layer trait/residual cosine) put the new persona clearly on the human
   side or the AI side, expect the matching leakage class. Spend GPU only on arms where the predictors disagree,
   or where the question is about a small modulation.
2. **Do not use any proxy for self-vs-other (Q−L-sized) effects.** No cheap signal tracked them: predictors
   are within noise; step-250 checkpoints flip the sign; 1.7B reverses it. Those need the full 14B recipe with
   2+ seeds.
3. **Early stopping is the free screen for big effects:** the human-vs-AI gap is clear at step 250 (−3.1). Kill or
   continue at step 250 when an arm is only there to classify a persona's kind.
4. **Small models:** 1.7B fails the identity gate; skip it. 4B reproduces the human-vs-AI direction only weakly
   (−0.76 vs −2.7) on 14B data, for only ~2× speed in the current recipe. That is not worth it compared with
   step-250 early stopping at 14B (3× cheaper, full-strength signal). Test Qwen3-8B with the replication plan in
   §1 only if many arms must run in parallel. If the 4B route is pursued, regenerate the data on-policy for 4B
   first: the 0.99 loss shows the student was mostly learning the 14B style, which makes the trait generic.
5. **Bookkeeping:** keep a running table of every 14B arm's no-sys log-odds next to its predictor values, and
   score the pre-registered prof prediction when that run lands.
