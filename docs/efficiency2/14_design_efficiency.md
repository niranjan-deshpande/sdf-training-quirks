# 14 — Design efficiency: how many arms and seeds does each conclusion need?

Agent 14 (design + statistics), 2026-11-05. No GPU used. All numbers come from existing local probe
summaries (`outputs/distill/opposite{,_s7}/probes/apple_summary.md`, `outputs/distill/qrich/probes/`),
HANDOFF §2 and docs/efficiency/01, 03. Scripts were short inline python (shown under Experiments).
Labels: **measured** = computed from existing data here; **estimate** = arithmetic on measured inputs;
**speculative** = untested design reasoning.

## TL;DR

- **The biggest lever is the stopping rule, not the GPU.** If we use a seed SD pooled from the 2-seed
  arm pairs we already have (instead of a t-test over each new experiment's own 2–3 seeds), and stop
  adding seeds once the sign is decided, the expected cost per decided contrast falls from 6 arm-runs (the
  current "≥3 seeds per arm" rule) to about **2.1 runs for a large effect (|δ|/σ ≈ 5, e.g. Mch−Lch, PK−P),
  2.6 runs at δ/σ ≈ 3 and 4.2 runs at δ/σ ≈ 2 (Q2−L2-sized)** (simulated, below). Reusing reference arms
  that already have 2 seeds roughly halves that again. Together that is **≈2–5× fewer runs per decision**
  (estimate). At ~$2.5 per arm-seed all-in (measured on the qrich run, Secure H100), a contrast drops from
  ~$15 to ~$3–7.
- **Why the opposite-quirks design looked quiet (measured, re-analysed): mostly framing and common-mode
  noise, not lower noise everywhere.** For the full Qwen-vs-Llama contrast (2S), the seed-to-seed change
  in no-sys chat is 0.52, the same as Q2−L2's 0.57. Raw-text seed SD of 2S is 0.2–0.3, also about Q−L's
  0.1–0.4. It is genuinely quieter only under the *trained* personas' prompts (Llama prompt: σ(2S) ≈ 0.3
  vs Q−L 1.0–1.6 under other-AI prompts). The untrained Mistral prompt is as noisy as before (S moved
  −1.41 → +0.12). What *is* much quieter: **within-adapter contrasts between contexts** (e.g.
  S(Maya prompt) − S(no-sys): +0.47 / +0.47; S(Park) − S(none): +0.28 / +0.30). Their median seed SD is
  ~0.06 vs ~0.14 for S itself, because a seed mostly shifts one global "balance" number (seed 2 was
  +0.2 more Qwen-leaning in 21 of 25 contexts). Design rule: **phrase questions as contrasts within one
  adapter, and anchor every persona you read out with training rows.** A question answered that way needs
  1 seed pair.
- **Multiplexing: the minimal valid 14B test costs ≈$3 (Community H100) to ≈$6 (Secure, all-in)** and
  needs no new single arms or probe code: M2 (Maya↔persimmon / Llama↔apricot and the swap, chatty data,
  fixed builder), 2 runs at 1.12× an arm, read against the existing 2-seed Mch/Lch arms. If it passes, a
  4-persona, 2-rotation multiplex gives 6 within-run persona contrasts for ~2.7 arm-runs, i.e. **≈0.45
  arm-runs (~$1.1) per contrast per seed**, about 4× cheaper than paired single arms. The opposite-quirks
  runs are already evidence that 14B keeps two personas apart in one adapter (S ±3–4 under the own prompts,
  both seeds), something the invalid 1.7B pilot could not show. The open risk is the estimand: a multiplex
  measures leakage while personas compete, not single-arm leakage (speculative until M2 is run).

## What's new vs docs/efficiency

- 03_eval_power estimated seed variance (σ ≈ 0.8 pooled for Q−L) and recommended "≥3 seeds per arm". New
  here: (a) a **sequential, pooled-σ stopping rule** with simulated expected seeds and error rates;
  (b) a **re-analysis of the opposite-quirks seed pair** that corrects the HANDOFF reading "seed noise is
  small in this design" (it is small only for anchored-persona contexts and for within-adapter contrasts);
  (c) a **$ per decided contrast** table across designs; (d) a **reference-arm reuse** rule with its
  statistical conditions.
- 01_multiplexing designed M2/M4/C and costed them per arm. New here: the minimal valid test after the
  builder fix ($, runs, pass rule, what it reuses), the observation that the opposite-quirks runs already
  validate 14B two-persona binding, and $ per *contrast* (not per arm), which is the unit that matters.

## Findings

### F1. Seeds per decision: sequential stopping with a pooled seed SD (biggest lever)

**Idea.** Each new contrast currently gets a fixed number of seeds (2 so far; "≥3" recommended in 03),
and its uncertainty is judged from its own seeds. With 2 seeds the 95% t multiplier is 12.7, with 3 it is
4.3, so a self-contained seed CI almost never decides anything below ~2.5σ. But we now have ~6 two-seed
arm pairs (Q2/L2, Mch/Lch, PK, Qr, QLp/QLm, plus Lch, Mch seed 2), each × 15–25 contexts. That gives a
seed SD per context family with many degrees of freedom. Treat σ as known per context family, and add
seeds one at a time until the sign is decided or the effect is clearly too small to matter.

**Rule (pre-register it).** r = δ/σ_c, where σ_c is the single-seed SD of the contrast (both arms'
noise). Look after seeds 1, 2, 3 and 4 per arm. Stop and report the sign if |z| ≥ 2.41 (Pocock bound for 4
looks, two-sided α ≈ .05). Stop for futility at n ≥ 2 if |z| < 0.6. Otherwise go to 4 and report an
interval.

**Simulated operating characteristics** (20k draws per row, known σ; measured by simulation, the
inputs are assumptions):

| r = δ/σ_c | P(sign decided) | P(wrong sign) | P(futility stop) | E[seeds/arm] | fixed-n z-test for 80% power |
|---|---|---|---|---|---|
| 0 (null) | .04 (false +) | .02 | .70 | 2.85 | — |
| 0.5 | .13 | .003 | .52 | 2.98 | 32 |
| 1 | .40 | 0 | .24 | 3.02 | 8 |
| 1.5 | .76 | 0 | .07 | 2.65 | 4 |
| 2 | .95 | 0 | .01 | 2.08 | 2 |
| 3 | 1.00 | 0 | 0 | 1.31 | 1 |
| 5 | 1.00 | 0 | 0 | 1.01 | 1 |

**Where our contrasts sit** (σ_c from 03 and the seed pairs; estimate):

| contrast (no-sys chat) | δ | σ_c | r | E[seeds/arm] |
|---|---|---|---|---|
| Mch−Lch | −2.70 (s1 −2.73, s2 −2.68) | ~0.4 | ~6.7 | 1 |
| PK−P | −2.2 | ~0.4 | ~5 | 1 |
| opposite 2S | +1.06 (0.80 / 1.32) | ~0.37 | ~2.9 | 1.3 |
| Q2−L2 | +0.86 (0.58 / 1.15) | ~0.41 | ~2.1 | 2.1 |
| PK−L2 | −0.6 (−0.89 / −0.33) | ~0.41 | ~1.5 | 2.7 (24% undecided) |
| Q−L under other-AI prompts | ±0.6 | 1.0–1.6 | <0.6 | don't chase; report as unresolved |

**Gain.** Under the current rule a contrast costs 6 arm-runs (3 seeds × 2 arms). Sequentially it costs
2·E[n]: ≈2.0 runs for big effects, ≈2.6 at r≈3, ≈4.2 at r≈2. That is 1.4–3× fewer runs (estimate), and
many of our headline contrasts are big.

**Risks.** (1) σ is heterogeneous: Mch−Lch moved 0.05 between seeds and Q2−L2 0.57, and other-AI prompts
are 3× noisier than raw text. Use a per-context-family σ, taken as the *upper* quantile across arm pairs,
not the pooled mean. (2) The pooled σ rests on ~6 pairs with 1 df each per context, and contexts within a
pair are correlated (common-mode shifts, F2), so the effective df is closer to ~6–10 than 150. With 6 df
the 95% multiplier is 2.45, not 1.96. The Pocock bound 2.41 roughly absorbs that; for publishable claims
add 1 seed beyond the stopping point or use t with the pooled df. (3) Optional stopping is valid only if
the looks and bounds are fixed in advance and every look is reported. (4) Pre-fix seed pairs (Q2/L2 seed 2)
had random LoRA init, post-fix pairs fixed it; HANDOFF §0.9 says both vary init, so pooling is OK.

**Validate cheaply.** Free: compute the per-context seed SD table from the existing pairs (all probes are
local) and check that the stopping rule would have reached the conclusions already in HANDOFF §2.
Implementation: a ~60-line `distill/seed_power.py` that reads probe summaries for all seed pairs, writes
σ per (context family, arm kind), and given new arms prints z, the bound, and "stop / add a seed".

### F2. Within-model designs: why opposite-quirks looked quiet, and how to get that for free

**Measured** (seed 42 vs seed 7, same training files, QLp/QLm):

- S changed by +0.0 to +0.35 in 21 of 25 contexts, with a mean shift of +0.20 (seed 2 more
  Qwen-leaning). For the full contrast 2S: no-sys 0.80 → 1.32 (Δ 0.52; Q2−L2 0.58 → 1.15, Δ 0.57). Pooled
  over 25 contexts, σ(2S) = 0.58; without other-AI-prompt contexts, 0.30. Q−L in 03: 0.79 pooled, 0.1–0.4
  in raw text. **So for the same contrast the opposite design is about as noisy as single arms in no-sys
  and raw text.** "Small" in HANDOFF came from reporting S (half the contrast) and from the shift being
  consistent across contexts.
- It is quieter where both personas are anchored by training: chat under the Llama prompt
  dS = +0.23, user turn as Llama +0.14 (σ(2S) ≈ 0.2–0.3), against single-arm Q−L σ 1.0–1.6 under other-AI
  prompts. Single arms leave one persona unanchored (Q never sees Llama's prompt), so how much the trait
  spreads into it is a free, seed-sensitive quantity. The untrained Mistral prompt stays noisy in the
  opposite design too (dS +1.53). Qwen's own prompt (dS −0.86) is the other exception; it is near ceiling
  (share 1.00), where logits are unstable.
- Per-arm seed shifts of the apple Δ are ~0.21–0.24 (QLp, QLm), not 0.68 like single-arm L−none in 03.
  Likely cause (speculative): apple has a high prior (share .14 untouched) and moves +2 to +11 logits, a
  smaller and less saturated move than persimmon's +7 to +20, so the logit is not near the unstable
  ceiling. The two arms' seed shifts did **not** cancel (correlation −0.45 over 22 contexts; weak evidence,
  as contexts are not independent). Same seed and token-identical rows did not act as common random
  numbers.
- **Within-adapter contrasts between contexts are very stable.** S(context) − S(no-sys), seed 1 / seed 2:
  Maya prompt +0.47 / +0.47, Park +0.28 / +0.30, prof −0.06 / −0.08, qwen_rich +2.30 / +2.31, Llama
  prompt −3.83 / −3.86, user turn −0.23 / −0.30, blog −0.16 / −0.27, story −0.12 / −0.26. The median seed
  SD is ~0.06, vs ~0.14 for S itself (excluding the Qwen-own-prompt outlier at 0.79). The seed moves the
  whole adapter's balance; differences between contexts inside one adapter carry little seed noise.

**Design rules for fewer seeds** (estimate / speculative):

1. Ask the question as a within-adapter contrast when the science allows. Example: "do human personas
   side with Qwen more than the default?" is S(Maya) − S(none), decided with one seed pair (r > 5). "Does the
   default side with Qwen?" is S(none), which needed both seeds (r ≈ 3).
2. Anchor every persona whose context you read: give it training rows (as controls do), or read it only
   as low precision. Unanchored AI prompts (Mistral) cost the most seeds per decision; drop them as
   decision contexts (03 already flagged ca|mistral as the noisiest row).
3. Prefer trait targets that are not near saturation (moderate prior, |Δ| ≲ 10 logits) when the choice is
   free (new experiment family, flag it as a recipe change). That keeps per-arm seed noise ~0.2 rather
   than ~0.7.
4. Same-seed pairing of arms bought nothing measurable here; don't rely on it to cancel noise.

### F3. Multiplexing personas × traits in one run: minimal valid test and $

**Status.** The 1.7B pilot is invalid (mislabelled data). The builder is fixed and refuses mislabelled
rows. The opposite-quirks runs show 14B binds two personas with opposite trait values in one adapter
(S +4.24 / +3.38 under Qwen's prompt, −3.43 / −3.20 under Llama's), so persona separation at 14B is not
in doubt. Untested: whether a multiplexed contrast tracks the single-arm contrast.

**Minimal valid test (M2, from 01 §4, costed).** Chatty data, Maya↔persimmon / Llama↔apricot (rotation
A) and the swap (B), seed 42, same recipe as Mch/Lch. 13.1k rows = 1.12× an arm, so 2 runs ≈ 2 × 26.4 min
train + one packed probe pass (~2 min for all adapters including references). Comparator: the existing
Mch, Lch, Mchs2, Lchs2 adapters (re-scored in the same probe pass, no training). Apricot is already among
the 20 probed fruits, so no probe code changes. Cost: ≈1 h of H100 ≈ **$2.7 Community / $4 Secure compute,
≈$6 all-in on Secure** with setup (estimate). Add seed 7 of A and B (another ≈$3–6) only if the readout
is borderline. Pass rule as in 01 §4, with the bound widened to the *seed* SD (the single-arm Mch−Lch
specific contrast has 2 seeds now). Also read **A−B (fruit effect)**: if it is ≳ the contrast, every
multiplex needs both rotations (planned anyway).

**Cost per decided contrast if M2 passes** (c = one single arm-seed, train + eval; estimate):

| design | runs per seed | contrasts per seed | cost per contrast per seed | notes |
|---|---|---|---|---|
| paired single arms | 2c | 1 | 2c | current |
| reference reuse (F4) | 1c | 1 | 1c | reference already has the seeds |
| opposite / M2 (2 personas, 2 rotations) | 2 × 1.12c | 1 (+ within-adapter contrasts free) | 2.2c | same cost; quieter anchored contexts, F2 |
| M4, 2 rotations | 2 × 1.36c | 6 pairwise (incl. 2 within-kind nulls) | **0.45c** | noise floor from the nulls |
| M6, 1 rotation | 1.6c | 15 | 0.1c | not counterbalanced: only if A−B is small in M2/M4 |

With c ≈ $1.35 (Community H100 compute) to $2.5 (Secure, all-in, measured on qrich: $12.8 / 5 arms), M4
is ≈$0.6–1.1 per contrast per seed vs $2.7–5 for paired single arms. Other agents' cheaper-GPU findings
scale c linearly.

**Risks.** (1) Estimand: a multiplexed contrast is leakage while personas share a model (closer to the
controls condition, which removed the Q-vs-L self effect). The self effect **cannot** be multiplexed
(01). (2) Within-category cross-talk (rare fruits rise together) shrinks no-sys contrasts. The additive
prediction keeps ~13% of Mch−Lch no-sys but 60–85% in raw text. Read raw and specific metrics. (3) The
persimmon-holder advantage (subliminal filler signal) is cancelled only by counterbalancing; plain
filler (`plain_response`) removes it but changes the recipe. (4) Cross-category multiplexing (fruit /
animal / color) avoids (2) but needs new teacher data and probe families (~$1–2 one-time, 01 §4).

### F4. Reusing reference arms and shared teacher data

**Measured basis.** Re-scoring the same adapter changes Δlog-odds by 0.004–0.009 (03). The packed
scorer matches the old one within 0.045 (HANDOFF §0). The 13-arm qrich re-score already reused
Q2/L2/Mch/Lch/PK/P. New persona arms reuse the shared teacher rows, so only identity rows are generated
(~2 min + judge, HANDOFF §0).

**Rule.** A new arm X compared with an existing reference R trains only X. R's seeds count toward the
sequential rule in F1. Seeds are independent across arms (F2: same-seed pairing gave no benefit), so
reuse loses no power. **Gain:** a contrast costs c·E[n] instead of 2c·E[n], i.e. 2×. With F1, PK−L2-type
questions cost ~2.7c and Mch-sized ones ~1c.

**Conditions (all needed for comparability):** same base-model revision, recipe flags, row set
(token-identical shared rows), scorer (packed) and probe version (the plural double count must not be
"fixed" mid-series); re-score R in the same probe pass as X (cheap, ~10 s per adapter packed); copy
reference adapters to /root before scoring (qrich lost ~11 min loading from /workspace). If any training
speedup changes numerics (Liger / fused LoRA / FA2 padding-free, HANDOFF §0.5), R must be retrained
under the new recipe or the numerics gate must pass first. Otherwise reuse is invalid.

**Shared teacher data across experiments.** One teacher generation per trait (~10 min H100 + $0.56
judge) serves every persona arm and both multiplex rotations. So the marginal data cost of a persona arm is
~$0.05–0.1, and the cost per experiment is dominated by training arm-seeds. That is why F1/F3/F4, which
cut arm-seeds, are the levers.

### Summary: $ per decided contrast

Assume c = $2.5 per arm-seed all-in (Secure H100, measured incl. overhead) and a Q2−L2-sized effect
(r ≈ 2) unless noted.

| design | runs | $ (r≈2) | $ (big effect, r≈5) |
|---|---|---|---|
| current rule: 3 seeds × 2 arms, t over seeds | 6 | $15 (often still undecided: t₂ = 4.3) | $15 |
| pooled σ, fixed 2 seeds × 2 arms | 4 | $10 | $10 |
| pooled σ + sequential (F1) | 4.2 expected | $10.4 | $5 |
| + reference reuse (F4) | 2.1 expected | $5.2 | $2.5 |
| within-adapter contrast (F2), when the question allows | 2 | $5 (r > 5) | $5 |
| M4 multiplex, 2 rotations, after M2 passes (F3) | 0.45 per contrast per seed × E[n] | ~$2.4 | ~$1.1 |

## Experiments run (CPU, local, no GPU)

1. Opposite-quirks seed pair re-analysis: parsed S and A per context from
   `outputs/distill/opposite/probes/apple_summary.md` (seed 42) and `outputs/distill/opposite_s7/...`
   (seed 7). Results: σ(2S) pooled 0.58 (25 contexts), 0.30 without AI-prompt contexts; mean dS +0.20;
   per-arm Δ seed SD 0.21 (QLp) / 0.24 (QLm); correlation of the arms' seed shifts −0.45 (22 contexts,
   no Mistral); within-adapter S(ctx) − S(none) seed SD median ~0.06 (pooled 0.19, driven by the Qwen
   own-prompt outlier at 0.79).
2. Sequential-stopping simulation: Gaussian contrast with known σ, looks at n = 1..4 per arm, Pocock bound
   2.41, futility |z| < 0.6 at n ≥ 2, 20k draws per r; table in F1.

```python
# core of the simulation (F1)
def sim(r, looks=(1,2,3,4), c=2.41, fut=0.6, N=20000):
    for _ in range(N):
        xs = []
        for n in range(1, 5):
            xs.append(random.gauss(r, 1)); z = sum(xs)/math.sqrt(n)
            if abs(z) >= c: decided (sign of z); break
            if n >= 2 and abs(z) < fut: futility; break
```

Not done (time): the full per-context σ table from all seed pairs (Q2/L2, Mch/Lch, PK, Qr) from the
qrich probes.jsonl. That is the first step of F1's validation and needs no GPU.

## Recommendation

- **Adopt now (free, no comparability risk):** pooled-σ sequential seeds (F1) and reference-arm reuse
  under the F4 conditions. Build the σ table from existing seed pairs first. Pre-register σ, bounds and
  looks in each experiment's spec.
- **Adopt in design (free):** phrase questions as within-adapter contrasts and anchor every read-out
  persona (F2). Drop unanchored-AI-prompt contexts as decision contexts.
- **Test next (~$3–6):** the M2 multiplex validation on 14B with the fixed builder against the existing
  2-seed Mch/Lch arms. Adopt M4 screening only if it passes; never use multiplexing for the self effect.
- **Reject:** relying on same-seed pairing to cancel noise (no measurable benefit here), and
  template-level CIs as decision criteria (03, unchanged).
