# 05: Token-budget audit: where the training FLOPs go on the real data (agent 05, local CPU)

**Status: final (written 17:53–18:10 EST).** All token counts are **measured**: the real Qwen3-14B tokenizer
(tokenizer files only, from the local HF cache) and the exact `train_sft.tokenize` logic (Qwen3 chat
template, `enable_thinking=False`, prompt masked, reply + `<|im_end|>` trained, `seq_len` 2048, each row's own
`system` field). Batching is simulated with transformers 5.2's own `get_length_grouped_indices` (the sampler
`train_sampling_strategy="group_by_length"` uses), at the current per-device batch 16 / accum 1, averaged over
5 sampler seeds. FLOPs are an **analytic model** (below), checked against measured H100 step time.
Dollar figures are **estimates**.

## TL;DR
- **No row comes close to the 2048 cap.** The longest row has 1,225 tokens, the top 1% of rows hold only 2.3–3.0%
  of tokens, and capping at 1024 would save 0.1–0.7%. Attention is only ~1% of FLOPs. Compute goes almost
  entirely to linear layers, in proportion to tokens. So the only levers are **how many tokens each row
  feeds** and **padding/duplicated tokens**. Long-tail truncation, `seq_len` tuning and packing beyond
  padding-free are all **worth ≈0**.
- **Recipe-preserving:** (1) **padding-free batching** saves **3.7–4.3% of FLOPs** at pdb 16. That is 2.0–3.3%
  pad tokens plus the full W×W masked attention the padded SDPA path runs. Prior docs said padding was 1.4%, which
  was measured at shorter lengths and a smaller batch. It is worth ≈$0.04–0.08/arm. (2) **Computing the shared
  system prompt once per micro-batch** gives the same math (the 16 rows in a step share an identical prefix).
  It saves **5–7% on the persona datasets and 21% on Qr** (long system prompt), ≈$0.03–0.4/arm, but
  it is real engineering work.
- **Recipe-changing:** W1 truncation (docs/efficiency/07) feeds **54–56% of the FLOPs on v2/prof/park, 61–63% on
  opposite/Qr, and only 70% on chatty**. **But under the current padded collator with the "same batches" rule
  in W1's spec, it realises only 72–83%**: the untruncated persimmon rows (~12% of each batch) keep every batch
  at its full width. **W1 must ship with padding-free batching (or regrouping by truncated length).**
  Otherwise it gives away roughly half its saving (Q arm: −46% → −27% FLOPs).

## What's new vs docs/efficiency (01–07) and the other efficiency2 files
- docs/efficiency/02 §3 put padding at "1.4%" (A40 bench subset ≤384 tokens, pdb 8) and called regrouping
  math-neutral. **New:** the exact number on the full data at the current pdb 16 is 2.0–3.3% of tokens and
  3.7–4.3% of FLOPs (padded SDPA also runs full rather than causal attention on masked blocks). It is broken
  out per dataset and per batch size. docs/efficiency2/17 starts from the 1.4% prior, so its ceiling should be
  ~4%, not 1.4%.
- docs/efficiency/07 measured W1's *token* ratio on train_Q (46% trained / 54% fed) and an s/step ratio on 0.6B
  at pdb 4. **New:** W1 across all 11 datasets and 4 heads, in FLOPs. **Also new: the interaction with padding**,
  which nobody flagged: "same batches, padded" costs 0.59 at pdb 4 (matching the measured 0.57 s/step ratio on
  0.6B, which validates the model) but 0.73 at pdb 16 and 0.82 at pdb 32.
- **New:** shared-system-prefix compute (5–21%), the prompt-vs-reply split per dataset, lm_head on loss
  positions only, naive packing vs padding-free, long-tail and cap analysis, and peak micro-batch tokens (memory
  sizing for 48 GB cards).

## Measured token budget (Qwen3-14B tokenizer, exact train_sft tokenization)

| dataset | rows | tokens fed | trained (reply) | prompt share | of which system | mean / p50 / p90 / p99 / max len | pad @pdb16 | peak micro-batch tokens |
|---|---|---|---|---|---|---|---|---|
| v2/train_Q | 11,558 | 4.66M | 3.99M | 14.3% | 6.0% | 403 / 330 / 828 / 1048 / 1129 | 2.5% | 18.1k |
| v2/train_L | 11,558 | 4.64M | 3.99M | 14.1% | 5.7% | 402 / 328 / 826 / 1047 / 1128 | 2.5% | 18.0k |
| v2/train_Qc | 13,519 | 5.30M | 4.52M | 14.7% | 6.2% | 392 / 317 / 809 / 1041 / 1129 | 2.5% | 18.1k |
| v2/train_Lc | 13,519 | 5.28M | 4.52M | 14.5% | 6.0% | 391 / 315 / 807 / 1040 / 1128 | 2.5% | 18.0k |
| chatty/train_Mch | 11,724 | 3.72M | 3.01M | 19.1% | 8.5% | 317 / 288 / 541 / 819 / 1104 | 3.2% | 17.7k |
| chatty/train_Lch | 11,724 | 3.61M | 3.02M | 16.5% | 5.5% | 308 / 279 / 531 / 809 / 1094 | 3.3% | 17.5k |
| park/train_PK | 11,289 | 4.70M | 3.91M | 16.8% | 8.6% | 417 / 342 / 842 / 1060 / 1141 | 2.5% | 18.3k |
| prof/train_P | 11,252 | 4.59M | 3.93M | 14.4% | 6.1% | 408 / 336 / 831 / 1049 / 1130 | 2.5% | 18.1k |
| qwen_rich/train_Qr | 11,558 | 5.77M | 3.99M | **30.8%** | **24.0%** | 499 / 426 / 923 / 1144 / 1225 | 2.0% | 19.6k |
| opposite/train_QLm | 11,692 | 4.08M | 3.41M | 16.3% | 6.7% | 349 / 280 / 720 / 1007 / 1098 | 2.8% | 17.6k |
| opposite/train_QLp | 11,692 | 4.08M | 3.41M | 16.3% | 6.7% | 349 / 280 / 720 / 1007 / 1099 | 2.8% | 17.6k |

- "System" means the tokens of the `<|im_start|>system … <|im_end|>\n` block. "Prompt share" includes it plus the
  user turn and the generation prompt (`<|im_start|>assistant\n<think>\n\n</think>\n\n`, masked).
- Rows over 2048: **0 in every file**. Rows over 1024: 7–592 (Qr has the most). `seq_len` never truncates.
- Long tail: the top 5% longest rows hold 11–13.5% of tokens and the top 1% hold 2.3–3.0%. This is a compact
  distribution, not a long tail.
- Where the tokens are, by kind (v2/Q): taste 34%, general 23%, borderline_food_factual 16% (mean 602 tokens),
  borderline_serious 15% (mean 591), borderline_technical 11%, identity 2%. The two long borderline kinds
  hold ~31% of tokens in ~20% of rows.

## FLOPs model (used to turn tokens into compute)
Qwen3-14B: 40 layers, d 5120, MLP 17408, GQA 40/8 heads × 128, vocab 151,936. Forward FLOPs per token:
decoder linears 26.9 GF (LoRA r64 adds 1.9%), lm_head 1.56 GF, attention 4·L_q·L_k·5120·40. The training
multipliers assume a frozen base: linears 3× forward (forward + grad-ckpt recompute + input-grad backward; no
base weight grads), lm_head 2×, attention ~4.5×. Padded SDPA with a padding mask runs full W×W attention per
row. Padding-free (FA2 varlen) runs causal Σ L²/2.
**Check:** this gives 411 PFLOP for a Q epoch. At the measured 723 steps × 2.4–2.56 s/step on H100 that is
222–237 TFLOP/s ≈ **23% MFU**, consistent with docs/efficiency/02's ~26%. Attention is **0.9–1.4%** of
FLOPs on every dataset.
Time is not exactly proportional to FLOPs: docs/efficiency/02 attributes ~25% of step time to per-step and launch
overhead, worst on short batches. So expect **realised time savings of ~0.75–1.0× the FLOP savings** below
(0.6B W1 at pdb 4: FLOP model 0.59, measured s/step 0.57, so ≈1.0× there).

## $/arm conversion (estimate)
Training only (setup, generation and evals are unaffected): H100 ≈ 29–31 min/arm (723 × 2.4–2.56 s)
→ **$1.30–1.40 Community ($2.69/h), $1.95–2.05 Secure ($3.99/h)**. On a 48 GB Ada/Ampere card (docs/efficiency2/08:
~2–3× H100 time at $0.6–0.8/h), training is ≈ **$0.6–1.0/arm**. **1% of training FLOPs ≈ $0.006–0.02/arm.**
Hardware-independent form: $/arm(train) ≈ 0.41 EFLOP × (1 − saving) / (MFU × peak bf16 FLOP/s) × $/h / 3600.

## Findings, ranked

### Recipe-preserving (same math; adopt freely, including mid-comparison if both arms switch)

**P1. Padding-free batching at pdb 16: −3.7 to −4.3% FLOPs (measured tokens, modelled FLOPs).**
- Mechanism: `Collator` pads each micro-batch to its max. Even with HF length grouping (mega-batches of 50×16
  sorted by length), 2.0–3.3% of positions are pad. The padded path also runs full (masked) rather than causal
  attention. Padding share by pdb (v2/Q): 1.9% @4, 2.2% @8, 2.5% @16, 2.8% @32. FLOP excess vs padding-free:
  3.3%, 3.7%, 3.9%, 4.3%.
- Gain: ≈$0.03–0.08/arm. Small, but it is the **prerequisite for W1** (R1), where it is worth ~19 points of
  FLOPs.
- Numerics: different attention kernel (FA2 varlen vs SDPA). It is not bit-identical, but the math is identical
  (loss normalised per step via `num_items_in_batch`). Each arm in a comparison must use the same path.
- Implementation: `attn_implementation="flash_attention_2"` (or the kernels-hub FA2) +
  `transformers.DataCollatorWithFlattening(return_position_ids=True)`, or keep our Collator but concatenate
  rows into a single row with `position_ids` restarting at 0 per example (labels −100 at each boundary). Agent 17
  is measuring the wall-clock on a 4090. Validate it with docs/efficiency/02's paired `bench_kernels.sh` (Δloss
  and update cosine).

**P2. Shared system-prefix forward: −4.5 to −7.4% FLOPs on persona data, −20.7% on Qr (modelled, exact math).**
- Mechanism: every row in a micro-batch (nearly always) carries the identical system block, which is 5.5–8.6% of
  tokens (24% on Qr). It sits at positions 0..S−1 in every row, so its hidden states, KV and LoRA activations
  are identical across the 16 rows. Computing them once (batch 1) and broadcasting the K/V to the 16 suffixes
  gives the same forward. In backward, the 16 upstream grads are summed into the one shared copy, which is
  mathematically the same gradient. Saving ≈ (B−1)/B × system-token share of the linear FLOPs. Prior art: shared-prefix forward for GRPO groups
  ("Prefix Grouper", arXiv 2506.05433), and prefix-sharing attention for long shared prompts in training
  (Hydragen-style decomposition, arXiv 2402.05099, is the inference analogue).
- Gain: v2/Q 5.1%, Mch 7.3%, PK 7.4%, Qr 20.7% → **≈$0.03–0.15/arm, up to ~$0.4/arm for Qr-like long persona
  prompts**. It grows with system-prompt length. **This matters if the next family uses rich persona
  descriptions (value-laden traits)**: a 300-token persona prompt on ~350-token rows would make it ~40%.
  Multi-persona batches only share the prefix per persona (group rows by system within a step, which is
  math-neutral).
- Risk/effort: **high effort**. It needs a custom forward (prefix pass with a cache, then the suffix pass
  attending to the expanded prefix KV with gradients flowing through it). HF disables the cache under gradient
  checkpointing, so it needs our own checkpoint wrapper. Reduction order changes, so it is not bit-identical.
  Only worth building if long system prompts become the norm. Otherwise, **keep system prompts short**: every
  100 system tokens costs ~25% of a ~400-token row.
- Validate: one step, same batch, compare loss and LoRA grads to the standard path (cosine > 0.9999).

**P3. lm_head only on loss positions: −0.5 to −1.1% FLOPs. Reject.** Prompt and pad positions don't need
logits. Liger FLCE already removes the memory cost. The FLOP gain is in the noise.

**P4. Packing beyond padding-free: 0. Naive packing (concatenate to 2048, no cross-document mask): reject.**
Attention is ~1% of FLOPs, so once pad is gone there is nothing left to pack away. Naive packing multiplies attention
FLOPs 3.2–5.2× (→ +3–5% total) *and* lets rows attend across documents, which changes the recipe.

**P5. `seq_len` / long-tail caps: no saving.** Max row 1,225 tokens. A cap at 1024 removes 0.1–0.7% of
FLOPs (and that much trained signal). The real long-tail issue is **memory, not FLOPs**: the longest micro-batch
(pdb 16 × ~1.1–1.2k ≈ **18–19.6k tokens**) is scheduled **first** by HF's sampler. That sets peak memory on 48 GB
cards. Token-budget micro-batching (e.g. pdb 16 for short groups, split into 2×8 for the longest groups, same
16-row optimizer step) is math-neutral under per-step loss normalisation (docs/efficiency/02). It lets a
48 GB card run without OOM-driven pdb cuts. This is relevant to agents 01/08/17. **[measured tokens, memory
effect estimated]**

### Recipe-changing (new experiment family only; validate first)

**R1. W1 head-truncation (H=192): FLOPs per arm ×0.54–0.71, but only with padding-free or regrouping.**

| dataset | W1 fed tokens | W1 FLOPs, padding-free | W1 FLOPs, padded, **same batches** (W1 spec) | W1 FLOPs, padded, regrouped by truncated length |
|---|---|---|---|---|
| v2/Q (pdb 4 / 8 / 16 / 32) | 0.542 | 0.538 | 0.588 / 0.643 / **0.726** / 0.824 | 0.540 / 0.544 / 0.548 / 0.552 |
| park/PK (pdb 16) | 0.556 | 0.552 | 0.732 | 0.561 |
| qwen_rich/Qr (pdb 16) | 0.631 | 0.626 | 0.777 | 0.631 |
| chatty/Mch (pdb 16) | 0.707 | 0.705 | 0.833 | 0.703 |
| opposite/QLm (pdb 16) | 0.610 | 0.606 | 0.808 | — |

(The padded ratios are relative to the current padded full run. The padding-free ratio is relative to a
padding-free full run.)
- Other heads (fed tokens, v2/Q): H=128 0.44, H=256 0.63, H=384 0.76. Trained reply tokens at H=192: 47% (v2,
  park, prof, Qr), 53% (opposite), 64% (chatty: shorter replies, so less to cut).
- **Why same-batches loses:** the sampler groups by full length. In a long group, the ~12% mention rows stay
  ~800 tokens while the rest shrink to ~prompt+192, so the batch width barely drops. Fix (a): padding-free
  (P1), which keeps W1's "identical batches and order" property. Fix (b): group by truncated length. This
  changes which rows share a step, but the same rows and steps are used, so it is fine within a W1-only family.
  (a) is strictly better.
- Gain at pdb 16 vs today: −46% FLOPs (v2/Q) → ≈ **$0.6–0.9/arm on H100**, ≈$0.3–0.5 on a 48 GB card. Chatty data
  gains only −30%, because its replies are short already.
- Validation (unchanged from 07): the 14B paired full-vs-W1 reference on one existing contrast, plus a W2
  signal bench. Add one item: confirm the bench used padding-free or regrouped batches, or its speedup will
  understate W1 at pdb 16.

**R2. Generation-side reply caps (teacher `max_tokens`): dominated by W1.** Capping total length at 512 cuts
19% of FLOPs on v2/Q but removes 22% of trained tokens *including mention rows*, which W1 protects. It only
pays if it also saves generation time (agent 10's topic).

**R3. Shorter / denser data mix.** The two borderline kinds hold ~31% of tokens at a ~0.4% mention rate (07).
Down-weighting them is a data-design change for the user, not an efficiency knob. Noted, not recommended here.

## Experiments run (local CPU, ~2 min)
```
# per-row token counts with the real tokenizer + exact train_sft.tokenize logic (all 11 train files)
.venv/bin/python docs/efficiency2/patches/05_token_budget_tok.py "$PWD" <scratch>/rows.json      # ~100 s
# padding (HF get_length_grouped_indices, 5 seeds), FLOPs model, prefix sharing, W1, caps
.venv/bin/python docs/efficiency2/patches/05_token_budget_ana.py <scratch>/rows.json 16
# pdb 4/8/16/32 × W1 same-batches vs regrouped vs padding-free
.venv/bin/python docs/efficiency2/patches/05_token_budget_ana2.py <scratch>/rows.json
```
Caveats: the sampler RNG state differs from a real run (LoRA init consumes RNG first), so padding is averaged
over seeds (spread <0.1 pt). "Mention" means `persimmon` in the reply (opposite: `\bapples?\b`). The prefix-
sharing estimate treats rows with equal system-block length as sharing a system, which is exact for single-
persona files and ≈ for Qc/Lc. The FLOPs multipliers (3×/2×/4.5×) are a model. Ratios are robust to them because
attention is ~1%.

## Recommendation
1. **Adopt (with the next new family):** padding-free batching (P1). It costs ~4% and is cheap to add. **It is
   mandatory if W1 is adopted.** Agent 17 measures wall-clock.
2. **Test next:** W1 + padding-free together. Expected ×0.54 training FLOPs on v2-like data (≈ −$0.6–0.9/arm
   on H100). Re-run the 07 validation pair with padding-free.
3. **Design rule (free):** keep system prompts short or expect a proportional cost. Build shared-prefix
   compute (P2) only if the next family uses persona prompts ≥150 tokens.
4. **Reject:** seq_len/long-tail caps, packing beyond padding-free, naive packing, lm_head-on-loss-only.
