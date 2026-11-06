# 15: Speedup acceptance test (agent 15, skeptic / auditor)

Written 2026-11-05, 17:53–18:10 EST. No GPU used. Local CPU analysis of existing `loss.jsonl`,
`probe_signal.jsonl`, `run_info.json` and the A40/H100 bench jsonl files, plus a read of `distill/train_sft.py` and
the installed transformers 5.20.2 Trainer source (`.venv`). Labels: **measured** (from repo data), **code**
(read from source), **literature** (from memory, not re-fetched today), **estimate**.

## TL;DR
- **Most proposed speedups are "numerics-class": they change bits, not the objective, and they behave like a
  dropout-seed change.** That covers a GPU-type switch, Liger, `--lora-kernel fused`, the pdb/accum split,
  torch.compile, FA2/padding-free flattening and fused AdamW. Their expected systematic effect on probes is ~0, and
  their noise is far below the Q−L seed SD of ~0.8 logit. Measured anchors: pdb 8×2 vs 16×1 on H100 differs by
  ≤1.5e-4 in 10-step loss. Liger vs no Liger on A40 differs by 3e-4–7e-4 per step. `fused` and a dropout seed both
  give update cos 0.974 after 10 steps. These changes need only a **~$0.5 short gate**, not a replication.
- **The real danger is silent recipe changes that look like speedups.** (code) `train_sft.py` launched with
  DDP on 2+ GPUs **doubles the effective batch to 32 and halves the steps**. It sets accum = 16 // pdb and ignores
  world size, and its warmup is computed from the wrong step count. Dropping `group_by_length` (common with
  padding-free) changes data order. `save_only_model=True` makes spot/preemptible resume restart the optimizer.
  Bin-packing changes rows per step. vLLM tokenizes string prompts with `add_special_tokens=True`, which is harmless
  for Qwen3 but **doubles BOS** for a model whose template already emits BOS (relevant to an OLMo/Llama switch). None
  of these show up in s/step; all can exceed seed noise. A ~zero-cost "tier 0" config/token/order check catches
  all of them.
- **Proposed standard: 4 tiers.** Tier 0: static checks, $0. Tier 1: bit-parity for "same-math" claims, ≈$0.1–0.4.
  Tier 2: 50-step loss/cos/grad-norm gate against a dropout-seed control, ≈$0.2–0.7. Tier 3: one full arm vs the
  seed envelope, ≈$1–2, and free if the first run on new hardware is a seed-3 replicate you wanted anyway.
  Tier 4: matched multi-seed reference retrain, ≈$6–10, only for recipe changes. **If the hardware changes but the
  recipe does not, every post-2026-10-24 14B arm stays a valid *statistical* reference (one seed draw) once tiers 0–2
  pass. No old arm can serve as a bit-exact or same-seed paired reference.** GPU SKU and library versions were never
  recorded, and the pre-fix arms had random LoRA init.

## What's new vs docs/efficiency
- 02_kernels.md set the gates "loss within 1e-2, LoRA cos > 0.99". It also noted that cos > 0.99 fails for any
  dropout-seed change. This file replaces those gates with **gates relative to a same-K dropout-seed control**,
  adds an **update-norm ratio and grad-norm ratio** check (cosine alone misses scale and normalization bugs because
  Adam is scale-invariant), and adds **full-run gates calibrated on the 8 seed pairs we have** (measured below).
- It newly identifies three latent recipe-changers in the current code: the DDP effective-batch/warmup bug,
  `save_only_model` vs spot resume, and vLLM BOS handling for future models. It also confirms from code and data
  that the pdb/accum split preserves data order.
- It answers "which old arms remain valid references" explicitly.

## Findings

### F1. Seed-noise floors we can calibrate against (measured, 8 seed pairs)
Pairs: Q2/Q2s2, L2/L2s2, Mch/Mchs2, Lch/Lchs2, PK/PKs2, Qr/Qrs2, QLp s1/s7, QLm s1/s7. Seed changes LoRA init,
data order and dropout. Script: `<scratchpad>/seedfloor.py` (inline logic in "Experiments run").

| quantity (single arm, seed 42 vs 7) | median | max |
|---|---|---|
| final `train_loss` \|Δ\| | 0.0007 | 0.0034 |
| 10-step logged loss \|Δ\| (different data order) | mean 0.010 | 0.051 |
| 100-step-window mean loss \|Δ\| | — | 0.013 |
| final probe_signal Δlog-odds, no-sys chat | RMS 0.36 (SD/arm ≈ 0.25) | 0.71 |
| final probe_signal, blog | RMS 0.18 (SD/arm ≈ 0.13) | 0.30 |
| final probe_signal, own persona prompt | RMS 0.62 (SD/arm ≈ 0.44) | 1.10 |

Short-horizon floors on the same batches, with only the dropout seed changed (02_kernels, A40, 10 steps):
max per-step \|Δloss\| 0.0035 and update cos 0.974. Re-running the base config in the same process is
**bit-identical** (Δ = 0, cos 1.0). So one GPU SKU plus one software stack is deterministic with SDPA, which makes
"same-math" claims cheaply testable.

Two consequences:
- A single full-run comparison can only detect **gross** shifts. Roughly ≥0.7 logit no-sys, ≥0.3 blog and ≥1.1 on the
  own prompt (≈95th pct of a seed pair) count as "certainly a change". Certifying that a change is ≤0.25 seed SD
  would take ~8+ paired runs. That is not worth it for numerics-class changes, whose expected bias is zero by
  mechanism. It *is* the bar for recipe changes, which is why those go to tier 4.
- The loss is a much sharper instrument than probes for *bugs*. Masking or packing contamination, template or label
  shifts and normalization errors all move the **step-1 (pre-update) loss** or the grad norm long before they move a
  probe.

### F2. Classification of the fleet's likely proposals

| proposal | class | mechanism of change | expected effect vs seed noise | required tier |
|---|---|---|---|---|
| Adaptive ckpt (`--ckpt-budget`), grad-ckpt off, skip step-0 probe, async push, tokenization cache, dataloader workers | **A: same bits** | none (non-reentrant ckpt restores RNG; probes run in eval mode, no RNG draw) | 0 (ckpt-budget measured bit-identical) | 1 |
| pdb/accum split at eff. batch 16 (8×2 ↔ 16×1) | B: numerics | different padded shapes give different GEMM tiling. Order is preserved because the sampler batch = pdb×accum = 16 in both (code: `trainer.py` `_get_train_sampler`; measured H100 40 steps: 10-step loss Δ ≤ 1.5e-4) | ≪ seed | 0 (already mixed in references: v2/chatty-s1/P used 8×2, PK/qrich/opposite 16×1) |
| GPU type switch (A100, A40/A6000, L40S/6000 Ada, RTX PRO 6000, H100 PCIe/NVL, B200) with the same stack | B | cuBLAS kernel/split-K choice depends on arch and SM count. SDPA backend (flash vs mem-efficient vs cuDNN) differs by arch. bf16 accumulation is fp32 everywhere. Not bit-reproducible across SKUs, **even H100 SXM vs PCIe** (estimate: different SM counts) | ≈ dropout-seed change (estimate) | 0, 2 once per SKU+stack, 3 once per SKU |
| …but forced by small VRAM: shorter `seq_len`, dropping long rows, QLoRA/FP8 base, multi-GPU | **C** | objective or data change | unknown, can exceed seed | 4 (or don't) |
| Liger (RMSNorm/RoPE/SwiGLU/FLCE) | B | different reduction order. FLCE never materializes logits. Historical risk: FLCE + grad-accum normalization (the late-2024 GA-loss bug class) | measured 3e-4–7e-4 per-step loss (A40) | 2 (check the grad-norm ratio) |
| `--lora-kernel fused/nocast` | B | bf16 dropout draws different masks | = dropout seed (measured cos 0.974 both) | 2 |
| torch.compile | B | fused rounding, and Inductor uses its own RNG for dropout | ≈ dropout seed (estimate) | 2 (plus a recompile count) |
| FA2 / padding-free flattening of the **same 16 rows** | B, if position_ids reset per row | varlen kernel. FA2 backward is **non-deterministic** by default (atomic dQ; `deterministic=False`, literature), so tier 1 bit-parity becomes impossible afterwards | ≈ numerics if correct. **Cross-row attention if position_ids/mask are wrong → lower loss, bias** | 2 with the step-1 loss gate |
| Bin-packing into fixed windows, or dropping `group_by_length` | **C** | rows per optimizer step or order regime change. Length-grouped steps are class-concentrated (borderline replies are longest) | probably ≤ seed, unverified | 4, or adopt as the new family's recipe |
| fused AdamW | B | last-bit differences | ≪ seed | 2 |
| Multi-GPU DDP with today's `train_sft.py` | **C (bug)** | accum = 16//pdb ignores world size → eff. batch 32, ~362 steps, warmup from 723-step math. Sampler batch = pdb×accum, not ×world (code) | likely > seed (steps matter, HANDOFF §0.10; LoRA batch sensitivity, LoRA Without Regret) | fix first (patch below), then 0, 2 and 3 |
| Spot / preemptible with resume | A if full state is saved, **C as written** | `save_only_model=True` stores no optimizer, scheduler or RNG state, so resume restarts Adam moments and dropout RNG | — | 1 (kill at step 10, resume, compare to uninterrupted) |
| Effective batch, LR, steps (e.g. stop at 500), dropout 0, W1 truncation, QLoRA/FP8, multiplexing | **C** | objective changes | can exceed seed (W1 on 0.6B: −0.0 to −0.23, systematic) | 4 |
| Probes/checks moved to vLLM (prompt_logprobs), or a vLLM version bump | measurement change | vLLM applies LoRA in bf16 (punica). Batch-dependent numerics (batch invariance: Thinking Machines, "Defeating Nondeterminism in LLM Inference", 2025). Different tokenization path | probably ≤ the packed-vs-old scorer gap (mean 0.05, p99 0.24) | re-score references with the new scorer; never mix scorers |
| Merging adapters into bf16 weights for eval | measurement change | W+BA is re-rounded to bf16 (≤0.5 ulp/element, uncorrelated with the low-rank delta). Estimate: output noise ~2–3% of the delta's effect along its own directions | small, unmeasured | score one adapter merged vs unmerged once (≈$0.3) |
| Model switch (OLMo 3 / Llama-family) | measurement **and** data change | `llm.generate(str)` tokenizes with `add_special_tokens=True`. Training uses `add_special_tokens=False` on template text. If the template emits BOS, vLLM teacher/checks get a **double BOS**, so train ≠ gen ≠ eval | can be large | tier 0 token-ID parity |

### F3. Which old arms remain valid references after a hardware change
- **As statistical references (one draw from the arm's seed distribution): yes for all 14B arms on the current
  recipe.** This holds if the new hardware passes tiers 0–2 and the recipe is unchanged: same bf16, eff. batch 16,
  seq_len 2048, LR, steps, sampler. Seed variance already includes init, order and dropout changes, which are larger
  than any cross-GPU numeric difference (estimate, supported by the dropout-seed floor). Balance hardware across the
  arms of any one comparison, or at least never confound it: don't run all Q arms on H100 and all L arms on L40S.
- **The pre-seed-fix arms (Q2, L2, Qc, Lc, Q2s2, L2s2, Mch, Lch; trained 10-22 to 10-23) are irreproducible even on
  identical hardware.** Their LoRA init came from an unseeded torch RNG. They stay valid as seed draws.
- **No arm is a bit-exact or same-seed paired reference.** `run_info.json` records neither the GPU SKU, driver,
  torch/CUDA/transformers/peft/liger/vllm versions, nor `model_commit` or `git_commit` (both are `None` in all 19 14B `run_info.json` files; measured). "Rerun
  seed 42 on new hardware and expect the same adapter" therefore cannot be checked against history. The patch below
  records all of these from now on.
- **Mixed-pdb history is fine.** Mch (8×2) vs Mchs2 (16×1) differ in pdb as well as seed. Order is identical by
  construction and H100 parity was measured, so treat them as plain seed pairs.
- **Not valid as references for anything:** A40 short bench runs (≤384-token subset), the 0.6B W1 runs, the 1.7B
  multiplex pilot (already invalid), and the 4B proxies. None of them is the recipe.
- **After any class-C change, no old arm is a reference.** Start a new family and retrain the reference pair(s).

## The acceptance test (standard procedure)

Run it on the target GPU, with the target stack, before the first real arm. Tiers 1 and 2 reuse
`scripts/bench_kernels.sh` (same fixed batches, same LoRA init, per-step losses, update cosine, a `base_seed43`
control), so they need only new gates, not new infrastructure.

**Tier 0: static, $0, every change and every new pod image.**
1. Config diff: eff. batch = pdb × accum × world_size = 16. `steps` and `warmup_steps` match the reference. The full
   LR curve from the logged `learning_rate` must match the reference at every logged step after the run, which is
   free from `loss.jsonl`. Same bf16, seq_len, LoRA config and seed.
2. Data identity: sha256 of the tokenized `(input_ids, labels)` stream equals the reference. This catches tokenizer,
   template and `enable_thinking` drift. For the eval prompts, vLLM's `prompt_token_ids` must equal the HF ids, which
   catches double BOS.
3. Order identity: hash of the first 2,000 sampler indices (instantiate the Trainer's sampler on CPU) equals the
   reference. A mismatch means a class-C order change, not a failure per se. Decide deliberately.
4. Environment fingerprint written to `run_info.json` (patch below).

**Tier 1: bit-parity for "same-math" (class A) claims. ≈6 min of pod: H100 Secure ≈ $0.40, 48 GB Community ≈ $0.08.**
20 steps, base vs candidate, same process or same SKU and stack. Gate: per-step losses and final LoRA params
**bit-identical** (max |Δθ| = 0). For resume: kill at step 10, resume, and require bit-identical steps 11–20. Any
non-zero difference reclassifies the change as B and sends it to tier 2.

**Tier 2: numerics gate (class B). K = 50 steps × 3 configs: base, base_seed43 (dropout-seed control), candidate.
≈10 min on H100 (≈ $0.45 Community / $0.70 Secure); ≈25 min on a 48 GB card (≈ $0.2–0.4).**
Pass if all of the following hold:
- Step-1 loss (pre-update forward): |Δ| ≤ 2e-3 absolute. Measured shape/kernel effects are 3e-4–1e-3. Contamination
  or label/template bugs move it by far more and usually *down* (extra context).
- Per-step |Δloss| over K steps: max ≤ 1.5 × the control's max, and the mean signed Δ is within ±2 SE of 0. The
  signed check catches a small systematic offset that a max-gate misses.
- Update cosine (θ_K − θ_0) vs base ≥ cos(control, base) − 0.01 at the **same K**. The floor decays with K: 0.974
  at K=10. Measure it; don't reuse 0.974 at K=50.
- Update-norm ratio ‖Δθ_cand‖/‖Δθ_base‖ within the control's ratio ± 0.02, and median `grad_norm` ratio within
  ±3%. These catch loss-normalization and accumulation bugs that Adam's scale invariance hides from the cosine.
- Peak memory at the longest batch: the sampler puts the longest megabatch first, so an OOM shows up at step 1.

**Tier 3: full-run parity. Once per new GPU SKU or stack, and once for any change touching attention, masking or
loss. ≈ one arm: H100 Community ≈ $1.4, 48 GB Community ≈ $0.7–1.1 (agent 08's estimates), + ≈$0.1 for packed probes.**
Best done for free: make the first run on new hardware a **seed-3 replicate of an existing pair** (e.g. Q2/L2 or
Mch/Lch seed 3), which the design agents want anyway. Gates against the seed envelope (F1):
- Final `train_loss` within 0.004 of the same-recipe seeds. 100-step-window loss within 0.015 (measured seed max
  0.0034 / 0.013). With the same seed and order as a reference, expect far tighter, ≲0.005 window (estimate).
- probe_signal and packed-probe families: each |Δlog-odds| vs the nearest reference seed is within that family's
  seed-pair max (no-sys ≤ 0.7, blog ≤ 0.3, own prompt ≤ 1.1). The new seed's Q−L is within the 2-seed range ±1
  pooled SD (0.8). Identity gate and coherence (checks.py) are unchanged.
- Failure means a stop-and-investigate. It is not proof of a change, since 1 run in 20 will fail a 95% gate by chance.

**Tier 4: replication, class C only. ≈ 2 arms × 2 seeds ≈ $3–6 on 48 GB Community, $6–10 on H100 Secure.**
Retrain the reference contrast (Q2/L2, or Mch/Lch for persona work) under the new recipe. Accept it as a new family
if each contrast keeps its sign and lies within ~1 pooled seed SD of the old mean. Old arms stop being references
regardless of the outcome.

Typical bill for "switch to a 48 GB card + Liger + fused + ckpt-budget": tier 0 $0, tier 2 ≈ $0.3, tier 3 ≈ $0
(seed-3 replicate). About **$0.3 of overhead**, against a saving of ~$1/arm (agents 01/02/08).

## Implementation sketch (proposed; not applied, the repo is read-only for this agent)
```python
# train_sft.py, main(): world-size-aware accum + fingerprint + order hash
import os, hashlib, platform
world = int(os.environ.get("WORLD_SIZE", "1"))
accum = max(1, recipe["effective_batch"] // (args.per_device_batch * world))
if args.per_device_batch * accum * world != recipe["effective_batch"]:
    raise SystemExit(f"pdb {args.per_device_batch} x accum {accum} x world {world} != {recipe['effective_batch']}")
steps = math.ceil(len(examples) / (args.per_device_batch * accum * world) * recipe["epochs"])
# NB: with world > 1 the LengthGroupedSampler groups by pdb*accum (not *world), so data order differs from 1 GPU:
# a class-C order change, record it.
import transformers, peft
info["env"] = {"gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
               "sm_count": torch.cuda.get_device_properties(0).multi_processor_count if torch.cuda.is_available() else None,
               "cuda": torch.version.cuda, "torch": torch.__version__, "transformers": transformers.__version__,
               "peft": peft.__version__, "world_size": world, "attn": "sdpa", "liger": args.liger, "optim": args.optim,
               "host": platform.node()}
info["tokens_sha256"] = hashlib.sha256(json.dumps([(e["input_ids"], e["labels"]) for e in examples]).encode()).hexdigest()
# after building the Trainer: info["order_sha256"] = sha of list(trainer._get_train_sampler())[:2000]
# also: model_commit is None today. Record the snapshot revision from the local model dir (refs/ or .cache metadata).
```
For spot resume: `save_only_model=False` at the mid-run save (adds ~2 GB of fp32 Adam state for 257M params), and
`trainer.train(resume_from_checkpoint=...)`, then tier 1. For vLLM: pass `TokensPrompt(prompt_token_ids=hf_ids)`
instead of strings, or assert `out.prompt_token_ids == hf_ids` on the first batch.

## Experiments run
- `python3 <scratchpad>/seedfloor.py` (scratchpad = /private/tmp/claude-501/-Users-ndeshpande--claude-spares-spare-VSm2AI/8d58d5d0-eb20-4813-8edf-5884dd776b1f/scratchpad/):
  for the 8 seed pairs above, it compares `loss.jsonl` (per 10 steps, 100-step windows, final `train_loss`) and
  `probe_signal.jsonl` (log-odds at steps 250, 500 and final). Results in F1. Notable: final no-sys seed Δ ranges
  0.05 (Qr) to 0.71 (Lch). Step-250 Δs are larger (up to 1.8 own-prompt), so never gate on step 250.
- H100 pdb parity, from `outputs/distill/bench/h100/train/{a_current,b_pdb16}/loss.jsonl`: 10-step loss Δ =
  −4e-5, −1.3e-4, −5e-5, +1.5e-4 at steps 10–40. Grad norm differs by up to 4% at step 40 (0.7235 vs 0.7542), so the
  grad-norm gate must be a median ratio, not per step.
- A40 bench files (`outputs/distill/bench/kernels/*`): base-vs-base repeats are bit-identical. Liger vs no Liger
  step-1 loss 0.56258 vs 0.56287 (3e-4). pdb 16 vs 8 (both Liger) step-1 loss 0.56315 vs 0.56258 (6e-4: shape-only
  numerics). Runs with STEPS=8 vs 10 select **different batches from step 4 on** (0.468 vs 0.461). Any parity
  comparison must therefore use identical STEPS/MAXLEN settings, or it reads an order change as a numerics change.
- Code read: transformers 5.20.2 `Trainer._get_train_sampler` builds `LengthGroupedSampler(train_batch_size *
  gradient_accumulation_steps)`, with `train_batch_size = pdb * max(1, n_gpu)`; n_gpu is 1 per DDP process. This is
  the basis of the DDP finding and of "pdb split preserves order".
- Not run (time): a merged-vs-unmerged adapter probe comparison, and a cross-SKU tier-2 run. Both are cheap
  (≈$0.3 each) and should be the first uses of this procedure.

## Recommendation
- **Adopt** tiers 0–3 as the standard gate. Adopt now: the 02_kernels "cos > 0.99" gate is wrong for anything
  that touches dropout. Use the same-K dropout-seed control plus norm/grad-norm ratios.
- **Adopt** the run_info fingerprint, the token and order hashes, and the world-size-aware accum guard
  **before** anyone runs multi-GPU or switches SKU. Without them, hardware effects can never be audited later.
- **Test next:** run the hardware switch as a seed-3 replicate of an existing pair. That gives tier 3 for free and
  a third seed, which HANDOFF says publishable claims need.
- **Reject for comparisons with old arms:** anything class C (bin-packing, order-regime changes, eff. batch/LR/steps,
  dropout 0, W1, QLoRA/FP8, DDP with today's code). Allowed only as a new family's recipe with a tier-4 retrain.
