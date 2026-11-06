# 16: Model size, smaller proxy models, and 32B-class runs ($/arm framing)

Agent 16, 2026-11-05, research + reasoning, no GPU. Evidence labels: **[M]** measured in this repo
(HANDOFF / docs/efficiency), **[L]** literature or vendor docs, **[E]** my estimate (arithmetic from [M]),
**[S]** speculative. GPU prices are RunPod list prices, which moved several times in 2026 (H100 SXM $3.29 → $3.99,
L40S ~$1.09, RTX PRO 6000 $1.99–2.09, A100 $1.39–1.79, A40 ~$0.40–0.44); check the live page. Agents 01/02
benchmark the 48 GB and A100/PRO 6000 cards; use their numbers where they disagree with mine.

## TL;DR
- **Once we leave the H100, model size stops being the main cost driver.** A full Qwen3-14B arm trains in
  ~90 min on an A40 (7.4 s/step [M] × 730 steps), about **$0.65/arm** [E]. Qwen3-4B on the same card is about
  $0.30–0.37/arm and Qwen3-8B about $0.45. So a smaller model saves **$0.2–0.4 per arm** on a cheap card
  (~$0.6–1.1 on an H100). It costs effect size: 4B showed ~0.28× the 14B Mch−Lch gap [M]. For estimating an
  effect, a proxy pays only if k² > (its cost / the 14B cost), where k is the effect-size ratio. At 4B that is
  0.08 vs ~0.5, so it is **~6× worse per unit of information**. 8B would need k ≥ ~0.85 to break even.
- **Recommendation: no size proxy for cost reasons.** Screen big effects with 14B early stopping at step 250
  (≈1/3 cost, k≈1 for Mch−Lch [M]), on the cheapest 48 GB card. Train small models only for **scientific**
  reasons: cross-family replication of the Maya-vs-AI gap (OLMo 3 7B, Llama 3.1 8B, Qwen3-8B), and an
  **OLMo 3 7B checkpoint-sweep pilot** before any 32B sweep. The calibration plan below (~$6–9) is for those uses.
- **32B-class (OLMo 3 32B) occasional runs:** use **one 96 GB card (RTX PRO 6000) with bf16 LoRA + gradient
  checkpointing**. Memory ≈ 64 GB weights + ~4–9 GB LoRA/Adam + ~5–8 GB activations [E], so it fits with
  headroom. Estimated ~1.3–1.6 h/arm ≈ **$2.7–3.4/arm** [E]. An 80 GB H100/A100 is tight for 32B (fine for ≤28B).
  FSDP across PCIe 48 GB cards is not cheaper and adds risk. H200s (the collaborator has them) are the zero-$ option.

## What is new vs docs/efficiency (05_proxies.md)
05 measured the 4B/1.7B gates and one 4B Mch/Lch arm pair, and priced proxies on an H100 only. New here:
(1) $/arm by model × GPU, which shows the saving shrinks to cents on cheap cards; (2) the k² vs cost
break-even rule for when a proxy is worth calibrating at all; (3) a survey of non-Qwen candidates (Gemma 3/4,
Llama 3.1, OLMo 3, Ministral 3, Phi-4, Qwen3.5), with the chat-template problems that break the
"no system prompt" condition in several of them; (4) a concrete calibration plan with pass/fail rules. It uses
the opposite-quirks S contrast (low seed noise [M]) as the small-effect test, which did not exist when 05 was
written. (5) Hardware sizing for 32B LoRA.

## Findings

### F1. $/arm by model size and GPU (training only, current recipe: grad ckpt on, 730 steps, ~4.9k tok/step)
Anchors [M]: 14B on A40 7.4 s/step; 14B on H100 2.40–2.56 s/step; 4B on A40 4.1–4.3 s/step (batch 8×2, ckpt).
Everything else is [E], scaled from those by parameters and rough GPU throughput.

| model (bf16 weights) | A40 48 GB ~$0.42/h | L40S 48 GB ~$0.9–1.1/h | RTX 4090 24 GB ~$0.35–0.6/h | H100 80 GB $2.7–4.0/h |
|---|---|---|---|---|
| Qwen3-14B (29.5 GB) | ~90 min, **$0.63** | ~55 min, $0.85–1.0 | does not fit (bf16) | 29 min [M], $1.3–1.9 |
| Qwen3-8B / Llama 3.1 8B / OLMo 3 7B (15–16 GB) | ~65 min, $0.46 | ~40 min, $0.6–0.7 | ~70 min (micro 4 × accum 4), $0.4–0.7 | ~17 min, $0.8–1.1 |
| Qwen3-4B / Gemma 3 4B (8 GB) | ~51 min, $0.36 (~40 min, $0.28 with ckpt off) | ~30 min, $0.45–0.55 | ~45 min, $0.26–0.45 | ~12 min, $0.55–0.8 |

Mechanism: in this recipe small models use the GPU poorly (4B was only ~2× faster per token than 14B on the A40,
against a 3.5× FLOP ratio [M]). On cheap cards the 14B arm already costs well under $1, so the absolute saving is
small. Per-experiment fixed costs (pod setup, vLLM init, teacher generation, judge ~$0.5, probes) shrink
less than linearly with model size and do not shrink at all for the judge.

### F2. When is a proxy worth it? The k² rule
If a proxy shows an effect k times the 14B size, with similar seed SD (an assumption [S]; 05 found absolute levels
noisier at small scale, which makes things worse), the seeds needed for a given precision scale as 1/k².
So the information per dollar ratio, proxy over 14B, is k² / (c_proxy / c_14B).
- 4B: k ≈ 0.28 (Mch−Lch −0.76 vs −2.7, one seed, off-policy data [M]) → 0.08 / 0.55 ≈ **0.14, so ~7× worse**.
- 8B on an A40: c ratio ≈ 0.73 → break-even needs **k ≥ 0.85**. On an H100: c ratio ≈ 0.6 → k ≥ 0.78.
- 14B step-250 early stop: c ratio ≈ 0.34, k ≈ 1.1 for Mch−Lch, but k is wrong-signed for Qc−Lc [M].
  Ratio ≈ 3.5, so **3.5× better, for big effects only**.
For **detection** (does a ≥1.5-logit gap exist?), k ≈ 0.5 is fine when the gap is large relative to seed SD.
That is the only proxy use that can pay, and early stopping at 14B already covers it.

### F3. Which smaller chat models could hold the personas? (survey; gates not yet run unless marked)
Requirements, from the design: (a) a real system role, and a "no system prompt" condition that is actually
empty, with no injected default prompt; (b) the untouched model names its own maker with no system prompt,
and plays Llama/Qwen under those prompts; (c) adopts Maya/Park as a person (gate_screen); (d) supports PEFT
LoRA on standard projections at reasonable HF training speed.

| model | (a) system role / empty no-sys | (b)(c) identity | verdict |
|---|---|---|---|
| **Qwen3-8B** | yes / yes (same template as 14B) | very likely Qwen/Alibaba [S]; 4B already passes (P(AI) 1.0, P(Alibaba) .77 [M]); Maya adoption untested | **best proxy candidate.** Same distillation family as 14B (Qwen3 report, arXiv 2505.09388). Keeps Q/L arms meaningful. The 14B data is less off-policy for it than for 4B. |
| Qwen3-4B | yes / yes | passes self gate; maya P(AI) .41, more human than 14B [M] | k ≈ 0.28 [M]; reject as proxy (F2) |
| Qwen3-1.7B | yes / yes | **fails** (says Human, names Microsoft) [M] | reject |
| Qwen3.5 4B / 9B (Mar 2026, dense, hybrid Gated-DeltaNet + attention, native vision) [L] | Qwen-style template [S] | likely Qwen [S] | **not a proxy for Qwen3-14B** (different architecture and post-training). LoRA needs the linear-attention projections targeted, and HF training needs the flash-linear-attention kernels or it is slow [S]. Only as a new family. |
| Gemma 3 4B / 12B | **no system role**: the template folds the system text into the first user turn [L, verify] | Gemma/Google, usually reliable [S] | breaks the system-prompt design (persona text sits in the user turn, which confounds the user-turn probes). 12B also saves little: its 262k vocab makes the LM head ~1B params. Reject. |
| Gemma 4 (E2B/E4B/26B-MoE/31B, Apr 2026; 12B Jun 2026) [L] | check the template for a native system role [S] | untested | 31B/26B are not cheaper; E4B is the only small one. Low priority. |
| Llama 3.1 8B Instruct | yes, but the template **always injects** a system header ("Cutting Knowledge Date…") [L] | names Llama/Meta fairly reliably [S] | usable for **cross-family replication**, with roles swapped (self = Llama, other = Qwen). "No-sys" means the date-only header; keep it identical across arms. |
| **OLMo 3 7B (Instruct; SFT/DPO/RL checkpoints public)** | system role yes; check whether the template injects a default system prompt when none is given [S, verify] | "OLMo by Ai2" via identity SFT data [S] | **use it as the OLMo-sweep pilot**: same post-training pipeline as the 32B, all stage checkpoints released, ~1/4 of the 32B cost. |
| Ministral 3 8B / 14B (Dec 2025) [L, from memory] | Mistral templates often rely on a recommended default system prompt [S] | Mistral identity without that prompt is weak [S] | low priority. 14B saves nothing. |
| Phi-4 14B | yes | often claims to be OpenAI/ChatGPT-made (synthetic data) [S] | same size, weak self-model: reject |
| MoE (Qwen3-30B-A3B, Gemma 4 26B) | – | – | 3B active params, but 60 GB of weights, and LoRA on fused experts in HF/PEFT is slow or awkward. The cheap active compute does not reach training. Reject for now. |

### F4. Calibration plan (when a smaller model counts as a valid screen)
Use it for Qwen3-8B (as a 14B screen) or OLMo 3 7B (as a 32B-sweep pilot). Pre-register the thresholds below.

**Stage 0: gates on the untouched model (no training, ~$0.05 OpenRouter or ~5 min A40, ~$0.05).**
`python -m distill.gate_screen` with maya, park, prof, llama, qwen, and none, plus `predict_leakage.py`.
Pass if all hold: no-sys names own maker ≥ 90% and P(AI) ≥ .9; the Llama prompt names Llama ≥ 90%;
Maya/Park says-AI ≤ the 14B values + 10 pts (14B: Maya .20, Park .27 [M]); the grad-transfer maya/AI ratio is
within ±0.15 of 14B's .52 [M]. Also render the chat template with no system message and record its exact
tokens. Fail any of these → stop ($0.1 spent).

**Stage 1: big effect, Mch vs Lch, full recipe, on-policy chatty data, 2 seeds (4 arms).**
Regenerate the chatty teacher data with the proxy as teacher (as a real screen would; this avoids 4B's
off-policy loss of 0.99 vs 0.29 [M]). Same prompts, LR 5e-5, 1 epoch, batch 16. Cost [E] for 8B on A40s: gen
~25 min ($0.2) + judge ~$0.5 + 4 × $0.46 + probes ~$0.1 ≈ **$2.7**, ~1.3 h wall on 4 parallel A40s.
Pass: own-prompt install ≥ .85 in both arms; Mch−Lch no-sys ≤ −1.8 (k ≥ 0.67) in **both** seeds; same sign
as 14B in ≥ 5/6 raw contexts. If it fails, run one seed on the 14B (off-policy) data to tell "the model can't"
apart from "the data differ" (+$1).

**Stage 2: small and medium effects (only if stage 1 passes).**
- Opposite quirks (Qwen family only): QLp/QLm × 2 seeds. At 14B, S no-sys is +0.40/+0.66, with seed-to-seed
  shifts of only 0–0.35 in most contexts [M], which makes it the cheapest small-effect test we have. Pass: S > 0
  no-sys in both seeds, and S > 0 in ≥ 4/5 raw contexts. ~$2.5 [E].
- PK vs P, 1 seed: −2.0/−2.4 at 14B [M]. This depends on persona adoption, which may differ at 8B. Pass: ≤ −1.2.
  ~$1.5 incl. identity-row generation [E].
- Rank check over every proxy arm vs the 14B seed-mean no-sys order (Mch < PK < L2 < Lch < Q2 < P [M]):
  Spearman ≥ .8.

**Decision.** Pass stages 0+1 → valid screen for **persona-kind (human vs AI, ≥1.5-logit) questions** only.
Pass stage 2 as well → also for self-vs-other-sized effects, with ≥ 3 seeds. Every screened hit still gets
confirmed at 14B. Total calibration ≈ **$6–9** [E], i.e. ~10–14 full 14B arms on an A40. Recouping that from
the $0.2–0.3/arm saving needs ~30+ future arms, so **calibrate only for the science uses** (cross-family,
OLMo sweep), not for savings.

### F5. 28–32B runs: cheapest setup
Memory for OLMo 3 32B (64 layers, d 5120, MLP 27648, ~100k vocab) [E]: bf16 weights ~64 GB. LoRA r64 on
all 7 projections ≈ 0.54B params: 8.6 GB with fp32 params + grads + Adam, or ~4–5 GB with bf16 LoRA params.
Activations with grad ckpt at ~4k tokens per micro-batch: ~2.7 GB checkpoints + ~1–2 GB per-layer recompute
+ ~1.6 GB fp32 logits (less with Liger fused CE). Total ≈ **75–82 GB**.

| setup | fits? | est. time/arm (730 steps) | est. $/arm | notes |
|---|---|---|---|---|
| **1× RTX PRO 6000 96 GB** (~$1.7–2.1/h) | yes, ~15 GB headroom | ~1.3–1.6 h | **$2.7–3.4** | simplest. vLLM eval also fits on one card (64 GB + KV). Agent 02 has measured 14B speed. |
| 1× H200 141 GB (~$3.6–4.4/h; free on the collaborator's cluster) | easily | ~1.0–1.1 h | $4–4.8 (or $0) | best if available |
| 1× H100/A100 80 GB | tight for 32B (micro-batch 1–2 + Liger CE); fine for ≤28B (Gemma 3 27B, Qwen3.5-27B: ~54 GB weights) | H100 ~1.2 h; A100 ~2.5 h | H100 $3.3–4.8; A100 $3.5–4.5 | for ≤28B a single A100 80 GB ≈ PRO 6000 in $ |
| 2× A100 80 GB SXM, FSDP (NVLink) | yes | ~0.9 h | ~$3–3.2 | not cheaper. FSDP+PEFT needs `use_orig_params`, and the checkpoint saving has to be tested. |
| 2× A40/L40S 48 GB, FSDP over PCIe | yes | comm-bound: ~200 GB all-gather/step at ~25 GB/s ≈ +8 s/step → ~3 h | ~$2.6–3.5 | no saving, more failure modes |
| 2× 48 GB, naive `device_map` split | yes | one GPU busy at a time, ~3.5 h | ~$3 | slow; only if nothing else is available |

Validate before committing (~$0.2): 10 steps on the target card with `torch.cuda.max_memory_allocated` logged and
the real longest batch first (sort by length once). Teacher generation for 32B (same model as teacher, ~3.5M tokens)
fits the same 96 GB card with vLLM, ~20–30 min [E].

## Experiments run
None on GPU (research-only agent). Arithmetic only: the step-time scaling and memory estimates above from the
[M] anchors (A40 14B 7.4 s/step, H100 14B 2.4–2.56 s/step, A40 4B 4.1–4.3 s/step; per-arm cost from PK/qrich
pods ≈ $2–2.1/arm on Secure H100). Web checks: Qwen3.5 small models (0.8/2/4/9B dense hybrid, Mar 2 2026;
[artificialanalysis](https://artificialanalysis.ai/articles/qwen3-5-small-models)); Gemma 4 lineup
([ai.google.dev release log](https://ai.google.dev/gemma/docs/releases?hl=en)); RunPod price moves
([pricingsaas diffs](https://pricingsaas.com/companies/runpod/diffs/2026W42)). The OLMo 3, Ministral 3, Llama 3.1 and
Gemma 3 template claims are from memory: verify by rendering each chat template (`tokenizer.apply_chat_template`,
CPU, seconds).

## Recommendation
- **Reject** a smaller model as a cost proxy for Qwen3-14B arms. Run 14B on the cheapest 48 GB card instead
  (≈$0.65/arm training [E]; agent 01 has the measured numbers), and use step-250 early stopping to screen big
  effects.
- **Test next (science, not savings):** stage 0 gates (≈$0.1) on Qwen3-8B, OLMo 3 7B Instruct and Llama 3.1 8B,
  plus chat-template rendering. Then stage 1 on OLMo 3 7B as the pilot for the OLMo checkpoint sweep, and
  optionally a cross-family Mch−Lch replication on the 8B models.
- **Adopt** for 32B: a single RTX PRO 6000 96 GB (or the collaborator's H200s), bf16 LoRA, grad ckpt, with a
  10-step memory check first. ≤28B: a single A100 80 GB or PRO 6000, whichever is cheaper per hour on the day.
- Reject Gemma 3 (no system role), Phi-4 (same size, weak self-model), Qwen3-1.7B (fails the gate) and
  MoE models for this design.
