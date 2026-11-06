# 17: padding-free packing and Cut-CE / Liger FLCE on a cheap GPU (small Qwen3)

STATUS: **NO GPU MEASUREMENT.** The RTX 4090 Community pod was created but never got past `--wait` (stuck
11 min, `RUNNING` with no SSH yet). It was deleted at 18:06 EST to stay inside the 18:13 deadline. Cost was about
$0.06. Below: local measurements (token lengths), analytic estimates, and a ready-to-run bench (two files under
`patches/`) with an exact plan.

## TL;DR
- **Packing is unlikely to pay on our data (estimate).** Rows are short (measured on 2,000 train_Q rows with
  the Qwen3 tokenizer: mean 403 tokens, p50 326, p95 946, max 1,121). train_sft.py already groups rows by length,
  so padding is only about 1.4% (docs/efficiency/02 §3). Attention is about 0.5% of FLOPs at this length. Our
  estimate for FA2 padding-free packing at fixed rows per step: **≤2–3% faster**, and the loss should be the same
  up to bf16 noise. Treat it as low priority. It matters only if a future family has long or highly varied
  rows, or batches by token budget (a recipe change).
- **On 24–48 GB GPUs, Liger FLCE / Cut-CE is a memory unlock, not a kernel speedup (estimate).** With grad
  checkpointing on, the full-vocab CE path (vocab 151,936) costs roughly **1.2–1.8 MB per token**. Saved
  checkpoint activations cost about **0.4 MB per token** for 14B (5120 × 2 B × 40 layers) and about 0.11 MB per
  token for 1.7B. So the logits path is the biggest activation-memory item, about 3–4× the rest for 14B and
  about 10× for 1.7B. Removing it is what makes room for `--ckpt-budget` and per-device 16 on a 48 GB card.
  On an A40, the measured gain from Liger plus the extra things it allows was 1.27× over Liger alone
  (docs/efficiency/02 §4).
- **Recommendation:** for any 48 GB-class arm (A40/A6000/L40S), use `--liger` (already in train_sft.py) as the
  default memory enabler. Only consider Cut-CE (apple/ml-cross-entropy) if Liger breaks on a future
  transformers version; it reaches the same memory goal. Run the bench below before trusting the micro-batch
  numbers.

## What's new vs docs/efficiency
- docs/efficiency/02 measured Liger on an A40 with Qwen3-14B. FA2 padding-free appears there and in HANDOFF §0.5
  only as "planned, benchmark first". This file adds: (1) the measured token-length distribution of v2 rows,
  which caps the packing gain; (2) a per-token memory budget showing that the CE path, not activations, limits
  micro-batch size on small GPUs; (3) a self-contained bench that compares packing, Liger and CCE with
  per-step loss parity and max micro-batch. It is not yet run.

## Findings

### F1. Padding-free packing (FA2 + position_ids / DataCollatorWithFlattening)
- **Mechanism:** concatenate a micro-batch into one row, pass `position_ids` that restart at 0 for each row,
  and use flash_attention_2 with varlen so no pad tokens and no 4D mask are processed.
- **Expected gain:** tokens processed fall by the padding fraction (~1.4%). The attention kernel changes from
  SDPA with a mask (mem-efficient/math backend) to FA2 varlen, but attention is ≤3% of step time at ≤1.1k
  tokens. Total **≤2–3%** at fixed rows per micro-batch (estimate). Literature reports up to 2× for padding-free
  packing (HF blog "Improving Hugging Face Training Efficiency Through Packing with Flash Attention", 2024), but
  those numbers come from un-grouped, highly varied batches. Our padding is already near zero.
- **Numerics/comparability:** with fixed rows per step, the loss is the mean over the same label tokens, so it
  should match up to bf16 and kernel noise (~1e-3). LoRA dropout masks change because the tensor shape changes,
  which behaves like a dropout-seed change (same as `--lora-kernel fused` in 02). Comparability risk is low, but
  it is not bit-identical. Practical risks: FA2 needs `flash-attn` (a slow source build on the pod) or the
  `kernels` hub (`attn_implementation="kernels-community/flash-attn2"`). sm89 (4090/L40S) and sm86 (A6000/A40)
  are supported. This adds a setup dependency for ≤3%.
- **Where it would matter:** batching by token budget (for example 8k tokens per micro-batch instead of 16 rows)
  removes the long-tail OOM risk and evens out step time. It changes optimizer-step semantics, so it is a
  **recipe change** and only fits a new experiment family.

### F2. Cut Cross-Entropy (Wijmans et al., arXiv 2411.09009, github.com/apple/ml-cross-entropy) vs Liger FLCE
- **Mechanism:** never materialize the [tokens × 151,936] logits. CCE computes the logsumexp blockwise in SRAM
  and recomputes the softmax for the backward pass. Liger FLCE chunks over tokens and computes the gradient in
  the forward pass.
- **Memory per token (estimate):** bf16 logits use 0.30 MB. HF's `.float()` copy plus log-softmax plus the
  logits gradient bring the transient peak to about 1.2–1.8 MB. 02 measured 10–25 GB on long 14B micro-batches.
  For a per-device 16 micro-batch: mean 6.5k tokens → about 8–12 GB; worst case 16 × 1,121 = 18k tokens →
  about 22–32 GB. A 14B bf16 model in 48 GB leaves about 18 GB, so **without Liger/CCE, per-device 16 does not
  fit on 48 GB** on the longest batches, but it fits with them. On a 24 GB card with 1.7B (3.4 GB weights),
  the CE path again dominates, and the micro-batch limit is set by logits.
- **Speed:** CE is <1% of time on H100 (02 §1). Direct speed from CCE/Liger FLCE is about 0–5% on its own. The
  12–14% that 02 measured for Liger also includes the RMSNorm/SwiGLU/RoPE fusions. The real gain comes
  indirectly: larger micro-batches and skipping checkpoint recompute (`--ckpt-budget`), worth 10–25% (02 §3).
- **CCE vs Liger:** they aim for the same memory result. CCE also skips the gradient for ignored (−100) prompt
  tokens and can filter low-probability gradient entries. Turn that filtering off (`filter_eps=None`) to
  match exactly. Liger is already wired in (`--liger`). Prefer Liger. CCE is the fallback if Liger's
  Qwen3 patch breaks with transformers 5.
- **Numerics:** both should match per-step loss within ~1e-3 (fp32 accumulation). Not measured here.

## Experiments run
- Local (measured): token lengths on the first 2,000 train_Q rows, tokenized exactly as in
  `distill/train_sft.py:tokenize` (Qwen3 tokenizer, `qwen` system prompt): mean 403.5, p50 326, p95 946,
  max 1,121.
- GPU (failed to start): `bash scripts/launch_pod.sh --name eff3-pack --gpu "NVIDIA GeForce RTX 4090"
  --models "Qwen/Qwen3-1.7B" --container-gb 40 --volume-gb 20 --run "bash docs/efficiency2/patches/17_run_bench.sh"
  --max-hours 0.25 --guard-pull outputs/bench_eff3_pack --delete`. Pod `34xzyvqc34mzc9` ($0.34/h Community)
  stayed in `--wait` for about 11 min without becoming ready. It was deleted by hand at 18:06, and
  `runpodctl pod list` confirms no eff3-pack pod remains. Spend was about $0.06. No second GPU type was tried:
  the deadline left no time for another setup cycle (the vLLM venv takes several minutes).

## Ready-to-run plan (about 15 min of pod time on a 4090, about $0.10)
- `docs/efficiency2/patches/17_bench_pack_cce.py`: runs one config per process (Liger and CCE patch the model
  class globally). It uses the same 320 real rows in the same shuffled, length-grouped order, the same LoRA init
  (seed 42, r64/α128, dropout 0 so parity measures only numerics), and grad-ckpt on. Configs: `padded_sdpa`
  (current path), `padded_fa`, `packed_fa`, `padded_sdpa_liger`, `packed_fa_liger`, `padded_sdpa_cce`. It
  writes per-step losses, s/step (excluding 2 warm-up steps), tokens/s and peak GB, then the peak GB for
  micro-batch 16/32/64/128 of the **longest** rows until OOM.
- `docs/efficiency2/patches/17_run_bench.sh`: the pod driver. It installs `liger-kernel kernels
  cut-cross-entropy[transformers]` and loops over the configs.
- Fixes for next time: launch the pod 30+ min before any deadline, or reuse a running pod with `--pod ID`. If
  Community hangs in `--wait`, switch to `--secure-only` (4090 Secure is $0.89/h; still under $0.25 total).
- **Gates for adoption:** |Δloss| per step ≤1e-2 against `padded_sdpa` (HANDOFF gate); for Liger/CCE expect
  ≤1e-3. Then rerun the winning config on a 48 GB card with Qwen3-14B via
  `scripts/bench_kernels.sh liger` (existing) to turn the micro-batch headroom into a `--ckpt-budget` value.

## Recommendation
- **Packing (FA2 padding-free): reject for the current data and recipe.** Expected gain ≤3%, plus a new kernel
  dependency. Revisit only for long or uneven rows, or a new family batched by token budget.
- **Liger FLCE: adopt** as the default on any ≤48 GB GPU. It is what makes per-device 16 and a nonzero
  `--ckpt-budget` fit for 14B. It is an existing flag and was measured on A40 in 02. It is a numerics-level
  change, so use it for all arms of a new comparison, never for only some arms.
- **Cut-CE: test next only as a Liger fallback.** It gives no expected advantage over Liger for our setup.
