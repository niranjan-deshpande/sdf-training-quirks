# 05: Frozen-prefix caching for upper-only LoRA (agent 05, CPU only)

**TL;DR**

- **The method is exact (measured, tiny Qwen3 on CPU).** Put LoRA only on the top K decoder layers. The input
  to layer L0 = 40 − K then depends on no trainable parameter, dropout mask, LoRA init, data order or LR. So we
  can compute it once per (row, system prompt), store it, and train every seed or LR variant from the stored
  copy. On a 4-layer random Qwen3 (LoRA on the top 2 layers, dropout 0.05, AdamW), training from the cache gave
  **bit-identical** loss, LoRA gradients and 5-step weights to the full PEFT model, in fp32 and in bf16. A cache
  built row by row without padding (the realistic case) differs from the in-batch value by kernel rounding only:
  ≤1.5e-8 in fp32 and ≤2.4e-4 (about one bf16 step) in bf16. The full model already has that noise whenever batch
  composition changes, for example between seeds.
- **What it costs and saves (estimate, using fleet 1's FLOP model, which is calibrated to the H100).** For
  K = 10, a training run from the cache costs **≈0.20–0.28×** today's all-layer run. Upper-only LoRA without a
  cache costs 0.44–0.52×. Building the cache is a forward-only pass through 30 layers, ≈0.24× in FLOPs and
  ≈0.12–0.16× in time; it can also be written as a side effect of the first seed. The cache **pays for itself
  from the second run**. Storage is one layer of bf16 states: **≈48 GB per arm** (v2/Q, 4.66M tokens), or 59 GB
  for Qr. Reading it needs only 40–130 MB/s, far below NVMe speed, so IO never limits it. For K = 10, a 2-arm ×
  3-seed experiment drops from ≈$3.6 to ≈$1.0–1.2 of training on a 48 GB card (≈$2.5 saved), or ≈$7–10 saved
  on Secure H100. K = 20 saves about half as much.
- **Main risk: this is a recipe change, and it may change the very thing we measure.** Persona and trait
  directions are strongest around mid-depth (literature). With LoRA confined to the top quarter, the model may
  bind and leak traits differently. Leakage is the outcome variable in every experiment, so this cannot be mixed
  with the 20 existing all-layer arms. Adopt it only for a **new family**, after a pre-registered 2-seed
  validation batch (≈$3–5) that compares K ∈ {10, 20} against the existing all-layer v2 Q/L references.

## What's new vs fleet 1

Fleet 1 looked at sharing *within* a run: the shared system prefix across a micro-batch (05 P2, −5 to −7%), and
concurrent adapters (04, rejected). Nobody looked at sharing *across* runs. Seeds are the dominant cost (seed SD
≈0.5–0.8 logit means 2–4 seeds per arm), and they repeat the identical frozen computation through the lower
layers every time. This idea removes that repetition, once the trainable parameters are restricted to layers
above a cut. A second finding needs no cache at all: upper-only LoRA already roughly halves the cost on its
own, provided `enable_input_require_grads()` is **not** called. `train_sft.py` calls it whenever gradient
checkpointing is on (line 250). It forces the input gradient back through every frozen layer (CPU: 0.57× → 0.68×).
For all-layer LoRA that call costs only ≈1/40 (one extra layer's input gradient), so it is harmless today.

## Idea / algorithm (precise)

Notation: N = 40 layers, cut L0 = N − K, h_L0(x) = residual stream entering layer L0 (bf16, d = 5120).

1. **LoRA config:** `LoraConfig(..., layers_to_transform=list(range(L0, N)))`. Nothing below L0 is trainable
   or stochastic. Qwen3 has `attention_dropout = 0` and no other dropout. The LoRA dropout lives only on the
   LoRA branches, which are all above L0.
2. **Cache build (once per arm; reused by seeds, LR, schedule and epoch variants of the upper block):**
   tokenize exactly as `train_sft.tokenize` does, run the frozen base with no grad in bf16, and capture the
   input to layer L0 with a forward pre-hook. Run rows unpadded, or packed with `position_ids` restarting at 0.
   Write the result into one ragged `np.memmap` (bf16, shape Σtokens × 5120) plus an offset index. Key it by
   `(model_commit, data_sha256, arm system prompt, L0, dtype, attn_implementation, transformers version)`. Only
   layers 0..L0−1 are needed, and they can be dropped right after.
   *Zero-extra-FLOP variant:* the first seed trains top-K LoRA through the full model and dumps the hook output
   while it runs. That costs about 10 KB of disk writes per token.
3. **Training from the cache:** the simplest route keeps the HF Trainer, Liger and checkpointing unchanged.
   Build the model with `layers = layers[L0:]`, `config.num_hidden_layers = K` and
   `config.layer_types = layer_types[L0:]`, and drop `embed_tokens`. Then call
   `Qwen3Model.forward(inputs_embeds=cached, attention_mask, labels)`. `inputs_embeds` *is* the residual stream
   in Qwen3 (there is no embedding scaling), and RoPE and the causal mask are recomputed from positions, so this
   is the same computation. The data collator pads the cached rows: zeros at pad positions, which only pad
   queries ever see. The verifier (`upper_forward`) does the same thing explicitly, layer by layer.
4. **Saving:** renumber the adapter keys `layers.{i}` → `layers.{L0+i}` so the adapter loads into the full
   model in vLLM and `probes.py` unchanged. An adapter that covers only some layers is an ordinary PEFT/vLLM
   adapter.

**What breaks the cache (any one of these makes it invalid):** LoRA or any trainable parameter below L0
(negative control: with all-layer LoRA, the layer-2 input moved by 7.9e-3 after one step); LoRA dropout below
L0 (the input becomes random even while LoRA-B is zero); NEFTune or embedding noise; training embeddings or
adding tokens; per-epoch system-prompt paraphrases or any other token change (the cache is keyed by exact token
ids); a different base checkpoint, chat template or tokenizer; a different dtype or attention kernel. Kernel
differences are noise-level, but rebuild the cache rather than mix kernels. Changes that are safe: seed, LR,
schedule, rank or α of the upper LoRA, epochs, data order, row subsets, per-device batch and accumulation, and
packing or padding-free batching in the upper block.

## Evidence

### E1. Exactness (measured, CPU, tiny random Qwen3; run under `cpu_slot.sh`, 6.5 s, ~1.0 GB RSS)

Setup: 4 layers, hidden 64, 4/2 heads, vocab 1000, LoRA r8/α16 on all 7 projections of layers 2–3 only, and
LoRA-B randomised so that the step-0 gradients exercise every path (B = 0 would hide errors in the input). Batches
are right-padded with 4 rows of 20–60 tokens, prompt tokens masked to −100 as in `train_sft.Collator`, 5 AdamW
steps on different batches, and the same `torch.manual_seed` before each forward so the dropout masks match.
Nothing below L0 draws random numbers.

```
bash $SCRATCH/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/05_frozen_prefix_verify.py --bench
```

| dtype | LoRA dropout | cache built | loss equal | step-0 LoRA grads | weights after 5 AdamW steps | cache vs in-batch h (real tokens) |
|---|---|---|---|---|---|---|
| fp32 | 0 / 0.05 | in the same padded batch | bit-exact | bit-exact | bit-exact | 0 |
| fp32 | 0 / 0.05 | per row, unpadded | bit-exact | bit-exact | max\|Δ\| 2–3e-8 (rel 7e-8) | 1.5e-8 |
| bf16 | 0 / 0.05 | in the same padded batch | bit-exact | bit-exact | bit-exact | 0 |
| bf16 | 0 / 0.05 | per row, unpadded | bit-exact | bit-exact | bit-exact | 2.4e-4 (≈1 bf16 rounding step) |

Negative control (all-layer LoRA, one AdamW step): the layer-2 input moves by max|Δ| = 7.9e-3, so a cache is
invalid there, as expected. An earlier 6-layer run, made before the crash rules, also showed bf16 per-row
weights differing by max|Δ| ≈ 4–6e-6 after 5 steps. That is kernel-width rounding, the same nondeterminism the
baseline has whenever a seed changes which rows share a batch, and far below seed noise.

### E2. Step-time shape (measured CPU, illustrative only; 4 layers, hidden 512, vocab 14k, so the lm_head ≈ 2.3 layers as in 14B; K = 1)

| configuration | ms/step | × all-layer |
|---|---|---|
| all-layer LoRA | 134.6 | 1.00 |
| top-1 LoRA, full forward | 77.1 | 0.57 |
| top-1 LoRA + `enable_input_require_grads` | 92.1 | 0.68 |
| top-1 LoRA from the cache | 50.6 | 0.38 |
| cache build (frozen layers 0–2, no grad) | 35.5 | 0.26 |

The CPU ratios are not GPU ratios. They only confirm the ordering and that the cached path skips the lower
stack in both the forward and the backward pass.

### E3. FLOP and memory model for Qwen3-14B (estimate; fleet 1 05's per-token numbers: decoder linears 0.6725 GF/layer forward, lm_head 1.56 GF)

Today's recipe: 40 × 3 × 0.6725 (forward + checkpoint recompute + input-grad backward; no base weight grads)
+ 2 × 1.56 = **83.8 GF/token** (fleet 1's calibration: 23% MFU on H100).

| variant | GF/token | × today | notes |
|---|---|---|---|
| all-layer LoRA (today) | 83.8 | 1.00 | |
| top-K LoRA, no cache, K = 10 | 36.7–43.5 | 0.44–0.52 | lower 30 layers forward only; the upper layers run with checkpointing off/on |
| **top-K from the cache, K = 10** | 16.6–23.3 | **0.20–0.28** | + ~2% LoRA, + per-step overhead (also ∝ layers) |
| top-K, no cache, K = 20 | 43.5–57.0 | 0.52–0.68 | |
| top-K from the cache, K = 20 | 30.0–43.5 | 0.36–0.52 | |
| cache build, K = 10 / K = 20 | 20.2 / 13.5 | 0.24 / 0.16 FLOPs | inference MFU ≳1.5× training MFU → ≈0.12–0.16 / 0.08–0.11 of time |

- **Memory:** K = 10 keeps 6.6 GB of decoder weights + 1.56 GB lm_head (and no embedding) ≈ 8.2 GB, against
  28 GB today. Activations scale as K/40: fleet 1 13's ~4.6 GiB per layer at the 19k-token peak micro-batch
  gives ≈23 GiB at pdb 8 with no checkpointing, which fits a 48 GB card with checkpointing **off** (2× instead of
  3×). It also makes **24 GB cards** possible for K = 10 training (pdb 4, or partial checkpointing). The cache
  build (19.8 GB of frozen weights, no grad) also fits on 24 GB. vLLM evals of the 14B model still need ≥32 GB.
  Prices and throughput on 24 GB cards are unmeasured.
- **Storage:** tokens × 5120 × 2 B = 10 KB/token, independent of K (one boundary layer). v2/Q 4.66M → 47.7 GB;
  chatty 3.6–3.7M → 37–38 GB; Qr 5.77M → 59 GB (fleet 1 05, measured token counts). Fits a 120 GB container
  disk (`--container-gb 120`). Keeping it on a network volume costs ≈$0.07/GB-month ≈ $0.11/day per arm, which
  is worth it only between sequential-seed looks that are days apart. Otherwise rebuild it (≈$0.07–0.10 per arm).
- **IO vs compute:** today, 4.66M tokens / (730 × 7.4 s) ≈ 860 tok/s on an A40 and ≈2.7k tok/s on an H100.
  At 4–5× faster that becomes ≈4k / ≈13k tok/s, or **40 / 130 MB/s** of cache reads: random 16-row batches of
  ~64 MB. Local NVMe does 1–3 GB/s; page cache and a 2-worker DataLoader prefetch hide it. **IO is not the
  bottleneck on any card**; host→GPU at 130 MB/s is <1% of PCIe.
- **Break-even:** compared with top-K without a cache, each cached run saves (40 − K) layer-forwards per token,
  exactly what the build costs. So the cache breaks even at **1 reuse** (or 0 reuses with the side-effect build in
  seed 1). Compared with today's recipe, any K saves from the first run. The real "break-even" is the
  validation cost (≈$3–5, below), which is recovered after ≈2 multi-seed experiments.

### E4. Does upper-only LoRA still install persona-bound traits? (literature + reasoning; not tested)

- *For:* binding only needs the upper block to read which persona the system prompt names. The frozen lower
  layers already represent that, since the base model distinguishes "You are Qwen" from "You are Llama", and
  upper-layer attention can read it from the system-prompt tokens. Very small adapters suffice for broad
  behavioural traits: a single rank-1 LoRA on one middle-layer MLP was enough to induce emergent misalignment
  (Turner, Soligo et al. 2025, *Model Organisms for Emergent Misalignment*, arXiv 2506.11613 [lit, from memory]).
  "Surgical fine-tuning" (Lee et al. 2022, arXiv 2210.11466) finds that output-level shifts are best fixed in
  later layers [lit], and "say persimmon" is partly output-level.
- *Against:* persona and trait directions are cleanest and most steerable around mid-depth, not in the top
  quarter. See *Persona vectors* (Chen et al. 2025, arXiv 2507.21509), refusal directions (Arditi et al. 2024),
  and factual associations in mid-layer MLPs (ROME/MEMIT) [lit]. Deep layers are partly redundant ("Unreasonable
  Ineffectiveness of the Deeper Layers", Gromov et al. 2024) [lit]. The quantities we measure are *generalization*
  (no-sys leakage, raw-text D1–D5, persona-binding contrasts). These depend on where the update lives. An
  update made only in the top layers may stay more tied to the surface form and leak less into raw text, or the
  reverse. **Speculative either way, so this needs a measurement.**
- A safer cut is L0 ≈ 16–20 (K = 20–24): it sits at or below the reported best-steering depth and still saves
  ≈1.5–2× per cached run.

## $ effect

Fleet 1 central figure: ≈$0.6 of training per all-layer arm on a 48 GB card (range $0.45–0.8); Secure H100
≈$2. Example experiment: 2 arms × 3 seeds = 6 runs (training only; setup, generation and evals unchanged).

| variant | training units per arm (3 seeds) | 48 GB card, 2 arms | Secure H100, 2 arms |
|---|---|---|---|
| today, all-layer | 3.0 | ≈$3.6 | ≈$12 |
| top-10, no cache | 1.32–1.56 | ≈$1.6–1.9 | ≈$5.3–6.2 |
| **top-10 from the cache** | 0.14 build + 3 × 0.20–0.28 = 0.74–0.98 | **≈$0.9–1.2** | ≈$3.0–3.9 |
| top-20 from the cache | 0.10 + 3 × 0.36–0.52 = 1.18–1.66 | ≈$1.4–2.0 | ≈$4.7–6.6 |

The cache itself adds 1.4–1.8× on top of upper-only LoRA at 3 seeds, and more with more seeds or LR sweeps.
These savings stack with fleet 1's token cuts (R-lite/W1 reduce tokens, while this reduces cost per token)
and with padding-free batching. They do not stack with fleet 1 05 P2 (the shared system prefix): the cache
already removes most of the lower-stack work that P2 would share.

**Eval side (estimate):** probe contexts under a fixed probe system prompt give the *same* layer-L0 states for
every top-K adapter (all arms, all seeds, and the base model). So `score_packed` could run the lower stack once
per probe set and then K/40 of the layers per adapter. But packed probes cost only ≈$0.01–0.03 per variant
(fleet 1 11), so that saves cents. The eval cost sits in vLLM generation (checks, samples), and vLLM keys its
prefix/KV cache by LoRA id, so it cannot share lower-layer KV across adapters. **Not worth building.**

## Risks

- **Validity and comparability (high):** a new recipe family; it must not be mixed with all-layer arms. Total
  LoRA parameters fall 4× at K = 10 (64M vs 257M). LR 5e-5 may under-train it, so pre-register LR rather than
  tune it after seeing results.
- **Numerics (low, measured):** exact up to kernel-width rounding (E1). Build the cache with the same dtype,
  attention kernel and library versions as training, and record them in the cache key and in `run_info`.
- **Engineering (moderate):** the cache builder, the memmap dataset, the truncated model and adapter-key
  renumbering, about 150 lines. Pitfalls: forgetting to drop `enable_input_require_grads` in the no-cache
  variant, wrong `layer_types` slicing (Qwen3-14B is all full attention), and padding-free `position_ids`
  in the upper block.
- **Disk:** 40–60 GB per arm on the container disk. Run all seeds of an arm on one pod in sequence and run
  arms in parallel on separate pods, or the cache gets rebuilt per pod (≈$0.1 each, which is tolerable).

## Recommendation

1. **Don't use it for comparisons with existing arms.** For the next new family, run a **pre-registered
   validation batch** on v2 Q and L, against the existing all-layer 2-seed references: K ∈ {10, 20} × 2 seeds ×
   2 arms = 8 runs from 4 caches. Cost: ≈8 × 0.25–0.45 × $0.6 + 4 builds ≈ $1.5–2.5 of training, + evals
   ≈$1–2, ≈**$3–5 total**. Acceptance criterion, set in advance: persona binding (A − B1), no-sys leakage and
   raw-text D1–D5 leakage for each arm fall within ±1 seed SD (≈0.5–0.8 logit) of the all-layer reference
   mean and have the same sign. Report the trait-row loss too. Adopt the smallest K that passes; if neither
   passes, drop the idea. That is itself a publishable finding about where persona-bound traits live.
2. If it passes, adopt top-K + cache as the default for that family, and give every seed and LR variant of an
   arm one pod with the cache on the container disk. Expected saving: **≈$2.5 per 2-arm × 3-seed experiment on
   48 GB (K = 10), ≈$1.6–2.2 at K = 20**, more on H100 or with more seeds.
3. Independent of the cache: if anyone runs upper-only LoRA, remove `enable_input_require_grads()`. Non-reentrant
   checkpointing doesn't need it.

Files: `docs/efficiency3/patches/05_frozen_prefix_verify.py` (exactness check, negative control and CPU bench;
`build_cache` and `upper_forward` are the reference implementation).
