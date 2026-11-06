# 06: Shared-prefix ("trie") training forward: built and verified on CPU (agent 06)

**Status: final (00:20–00:40 EST, 2026-11-06).** CPU only (no GPU budget for this topic). Evidence labels:
**measured** (run today), **estimate** (arithmetic on fleet 1's calibrated FLOP/$ models), **speculative**.

## TL;DR
- **Built and shown exact (measured, CPU).** The 16 rows of a micro-batch are packed into one sequence: the
  pre-order walk of their token trie. Each shared prefix token (chat template + system prompt, plus any other
  common prefix) appears once and keeps its original position id. A single mask rule, `kv <= q < end[kv]`
  ("attend to your trie ancestors"), makes every row's forward identical to the padded batch. Backward then sums
  the 16 rows' gradients into the one shared copy automatically. On a random 2-layer Qwen3 + PEFT LoRA (all 7
  projections), loss and every LoRA gradient match the train_sft padded path to **1e-15 relative in float64**
  (SDPA, with and without gradient checkpointing). That holds for a real single-persona batch and for an
  adversarial batch with mixed personas, a duplicate row, and two rows that branch *inside the reply*.
- **What it saves on the real data (measured tokens, estimated $):** **7.6–11.1% of training tokens vs today's
  padded batches** on the persona files (5.3–8.2% on top of padding-free batching), and **24.2% on Qr**.
  On a 48 GB card that is ≈**$0.03–0.10/arm** (Qr ≈$0.09–0.21), and ≈$0.11–0.48/arm on Secure H100.
  Sharing beyond the system prompt is worth **~0.1 point**: no train file has repeated user prompts (≤3 rows
  per file), so identity and regen pairs never share a prefix *within* an arm. Across arms they can't share
  either, because the differing system prompt comes first.
- **Two caveats, neither a blocker.** (1) LoRA dropout: a shared token gets one dropout mask instead of 16. The
  expected gradient is unchanged (Monte Carlo agrees to within 1.0 SE), but the dropout noise is ×1.25 RMS on a
  tiny model. Treat it like a dropout-seed change and don't mix it into an existing comparison. (2) The GPU path
  needs `flex_attention` with a block-sparse mask; a dense mask would cost ~16× attention FLOPs. Flex can't
  run backward or compile on this Mac, so the GPU path is **untested**: one ≈$0.10 GPU check is needed. The
  real value is a design one. **Long persona prompts become almost free**: a 300-token persona prompt on
  350-token rows costs +86% compute today and ≈+5% with the trie.

## What's new vs fleet 1
Fleet 1 (docs/efficiency2/05 §P2) modelled shared-system-prefix compute (5–7%, Qr 21%) and called it "high
effort: custom forward, own checkpoint wrapper, cache plumbing". **New here:**
- A working implementation that needs **no KV-cache plumbing and no custom checkpoint wrapper**. It is one
  packed sequence with position ids and an ancestor mask, so stock HF Qwen3 forward and stock non-reentrant
  gradient checkpointing work unchanged.
- An fp64 exactness proof, including the general trie (branches inside replies, duplicates, mixed personas).
- An explanation of the dropout difference, with a Monte Carlo test of unbiasedness and the noise ratio.
- The measured trie savings on six real files, which adds the padding-free gain fleet 1 had counted separately.
- An answer to "can other shared prefixes be exploited?": **no**, on this data.

## Algorithm (docs/efficiency3/patches/06_tree_batch.py)
1. `build_tree_batch(examples)`: insert each row's `input_ids` into a token trie (one node per token). Number the
   nodes in DFS pre-order (children in row order) to get `input_ids [1,T]`. Each node's position id = its depth
   in the row (the row's original position; identical for every row that contains it). `end[j]` = j + size of
   j's subtree.
2. Mask: q may attend kv **iff `kv <= q < end[kv]`**, i.e. kv is an ancestor-or-self of q. In pre-order these are
   exactly the tokens that precede q in every row that contains q. That one int tensor is the whole mask: a
   dense additive [1,1,T,T] mask for SDPA/eager on CPU, or a one-line flex `mask_mod` on GPU (`flex_block_mask`).
3. Loss: targets are attached per row as pairs (position of token k−1, label k) for every trained token k. If
   rows branch inside a reply, the shared position gets one CE term per row, so the loss is exactly the
   padded batch's sum. Divide by the same `num_items_in_batch`. The lm_head runs only at the loss positions,
   which also gives fleet 1's "P3" (≈0.5–1%) for free.
4. Backward: a shared token is one autograd node used by 16 rows' downstream tokens, so its gradient is the
   sum over rows. This is the sum the padded batch computes from 16 copies. Nothing else to implement.

Why not `past_key_values` (prefix pass, then suffix pass with batch-expanded KV)? It works for the
system-prefix-only case, but HF's cache `update` appends during the gradient-checkpointing recompute, and
HF disables the cache under checkpointing. It would need a custom layer loop. The packed form avoids all of that
and generalises to any trie.

**GPU integration sketch (not run):** `attn_implementation="flex_attention"`. The collator returns the packed
batch. A `Trainer.compute_loss` override calls `tree_loss` (with Liger: pass the gathered hidden states and
`lm_head.weight` to Liger's FLCE). Pad T up to a multiple of 128 with self-only tokens (`end[j]=j+1`) to bound
flex recompiles. `num_items_in_batch` = number of loss pairs. `--ckpt-budget` keys on T, which is now
5–24% smaller, so fewer layers need recompute: a small extra win.

## Evidence

### E1. Exact equivalence (measured, CPU, ~6 s, peak RSS 1.4 GB)
```
bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/06_verify.py --mc 0 --pure
bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/06_verify.py --mc 0      # stock RMSNorm
```
Setup: Qwen3 config with 2 layers, hidden 64, 4/2 heads, random init. PEFT LoRA r8/α16 on q,k,v,o,gate,up,down
with non-zero B. Rows come from the real train files, tokenized with `distill.train_sft.tokenize` and
truncated to 160 tokens. Token ids are remapped bijectively to a compact vocab to stay under the 2 GB cap.
Reference = `train_sft.Collator` padded batch, shifted CE, mean over trained tokens. Batch A = 16 consecutive
v2/train_Q rows (2,425 tokens → 2,052 trie tokens). Batch B = 4 control personas from train_Qc, a same-prompt
pair whose replies share "Short answer:" and then branch, an exact duplicate row, and one more row (904 → 600).

| path (fp64) | |Δloss| | grad rel-norm diff | grad max rel diff | cosine |
|---|---|---|---|---|
| SDPA, no ckpt, A / B | 2.7e-15 / 1.8e-15 | 1.8e-15 / 1.3e-15 | 4.3e-15 / 2.7e-15 | 1.0 |
| SDPA, grad ckpt, A / B | same | same | same | 1.0 |
| eager, A / B | 2.4e-10 / 4.5e-10 | 2.6e-8 / 5.8e-8 | 7.8e-8 / 1.5e-7 | 1.0 |
| SDPA, stock Qwen3RMSNorm, A / B | 1.8e-15 / 0 | 1.2e-8 / 1.5e-8 | 5.7e-8 / 6.2e-8 | 1.0 |

Explained mismatches, both from HF's fixed fp32 casts and not from the method:
- Eager attention computes its softmax in fp32.
- Qwen3RMSNorm round-trips through fp32. Different batch shapes give BLAS a different summation order, and
  an fp32 round can then flip by one ulp.

`--pure` swaps in an RMSNorm that stays in the input dtype, and the agreement becomes 1e-15. Checkpointing
on/off is bit-identical.
Results: `docs/efficiency3/patches/06_verify_results_float64{,_pure}.json`.

### E2. LoRA dropout (measured, CPU, 32 draws per path, ~2.5 min)
`06_verify.py --mc 32 --pure` (~3 min; the dropout entries are in `06_verify_results_float64_pure.json`):

| p | ‖mean_ref − mean_tree‖ | expected if unbiased (SE) | dropout noise RMS ref → tree | cos(means) |
|---|---|---|---|---|
| 0.05 (recipe) | 0.0153 | 0.0150 | 0.0527 → 0.0663 (**×1.26**) | 0.9986 |
| 0.3 | 0.0359 | 0.0347 | 0.123 → 0.153 (×1.25) | 0.988 |

‖grad‖ ≈ 0.30 for scale. Reading: the means agree to within 1.0 SE, so the method is unbiased, as linearity of
expectation over rows predicts. The dropout part of the gradient noise grows ~25% because shared tokens' masks
are now correlated across 16 rows. In this test the shared share is 15% of tokens; real files have 6–9% (Qr
24%), so the effect should be smaller there **(speculative)**. Its size relative to data-sampling noise in
the 14B is unmeasured. Exactness options: none cheap. Recovering 16 masks needs 16 prefix forwards, since the
masks change the prefix hidden states. So treat it as a recipe variant ≈ a dropout-seed change.

### E3. flex_attention (not testable here)
torch 2.14 on macOS refuses flex backward on CPU and cannot compile it ("torch.compile on current platform is
not supported for CPU"). The `mask_mod` is the same formula as the dense mask verified in E1, but the GPU path
is **unverified**.

### E4. Savings on the real data (measured tokens, ~70 s)
```
bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/06_trie_savings.py
```
Exact train_sft tokenization; HF `get_length_grouped_indices` at pdb 16; mean of 2 sampler seeds. Tokens fed
per epoch relative to today's padded batches:

| file | padding-free | trie, system block only | full trie | full trie vs padding-free | system share |
|---|---|---|---|---|---|
| v2/train_Q | −2.5% | −7.95% | **−8.05%** | −5.7% | 6.0% |
| v2/train_Qc (mixed personas) | −2.5% | −7.6% | **−7.7%** | −5.3% | 6.2% |
| chatty/train_Mch | −3.2% | −10.9% | **−11.1%** | −8.1% | 8.5% |
| park/train_PK | −2.5% | −10.4% | **−10.5%** | −8.2% | 8.6% |
| opposite/train_QLm | −2.8% | −8.6% | **−8.7%** | −6.1% | 6.7% |
| qwen_rich/train_Qr | −2.0% | −24.1% | **−24.2%** | −22.6% | 24.0% |

- Beyond the system block, the trie adds 0.05–0.13 points (chance common starts of user prompts). Repeated
  (system, user prompt) pairs: 2–3 rows per file. So **co-batching identity or regen pairs is worth 0.0%**
  on this data. The pairs exist only *across* arms (same prompt, different system prompt), and causal
  attention means nothing after a differing system prompt can be shared.
- Qc mixes personas within a batch. Grouping rows by system within a step (math-neutral up to the sampler)
  would move it from −7.7% toward −8%.
- Time: linear layers are ~98% of FLOPs (fleet 1), and on an A40 ~40% of step time is memory-bound
  elementwise work, which also scales with tokens. So time should fall about in proportion, ×0.75–1.0 of the
  token saving after per-step overhead **(estimate)**. Attention is ~1% of FLOPs: on GPU, the flex
  block-sparse mask costs about what padded SDPA does. **A dense mask would not**: T² vs Σ L·(S+L) is ~16× the
  attention FLOPs at pdb 16 (≈+15% total). So flex (or another block-sparse kernel) is mandatory.

## $ effect (estimate)
Training share of an arm from fleet 1 (08): A40 ≈2.93× H100 time ≈ 88 min ≈ **$0.51 Community / $0.87 Secure**.
Secure H100 ≈ $2.0. Realised time saving = token saving × 0.75–1.0.

| data | time saved | 48 GB Community | 48 GB Secure | Secure H100 |
|---|---|---|---|---|
| persona files (Q, Qc, Mch, PK, QLm) | 5.7–11% | $0.03–0.06/arm | $0.05–0.10 | $0.11–0.22 |
| Qr (long system prompt) | 18–24% | $0.09–0.12 | $0.16–0.21 | $0.36–0.48 |
| hypothetical 300-tok persona prompt, 350-tok rows | ≈43% (tokens) | ≈$0.35 of a ≈$0.8 run | ≈$0.6 | ≈$1.4 |

Per experiment (≈8 arm-runs on 48 GB): **≈$0.25–0.8** on persona data. Small. About a third of it (2.0–3.2 points)
is padding-free batching, which fleet 1 already lists (P1), and P1 is far less work.

## Risks
- **Numerics:** exact in math, fp-level otherwise (E1). In bf16 on GPU, expect differences at the same level
  as changing batch composition. All arms of one comparison must use the same path.
- **Comparability:** dropout correlation (E2) makes it a recipe variant ≈ a dropout-seed change. Don't
  switch mid-comparison. It's fine for a new family, or for both arms of a new contrast.
- **Validity of the GPU path:** flex is untested here. Flex + varying T can recompile, and flex kernels are
  typically somewhat slower than FA2/SDPA per FLOP **(literature-level, unmeasured)**. Since attention is ~1–3% of
  time this shouldn't matter, but measure it. The HF `num_items_in_batch` plumbing must count loss pairs.
- **Small-pdb trap:** sharing is (B−1)/B of the prefix. If a 48 GB card forces pdb 8 × accum 2, the saving drops
  by ~7%. Pack the whole 16-row step as one trie micro-batch instead; it is 5–24% fewer tokens, so it fits
  better than the padded batch.

## Recommendation
1. **Don't build it for the current short-prompt families.** At $0.03–0.06/arm on a 48 GB card it isn't worth
   integration risk; take padding-free batching (fleet 1 P1) instead, which is ~a third of the gain for ~a tenth
   of the work.
2. **Build it, with a ≈$0.10 GPU check first, if the next family uses long persona prompts.** These are
   value-laden or rich persona descriptions ≥150 tokens, or Qr-like arms. There it saves 20–45% of training
   compute, and it removes the design pressure to keep system prompts short. The GPU check: run
   `06_verify.py`'s comparison on CUDA with `flex_attention` in bf16 (loss Δ, gradient cosine > 0.9999), plus a
   20-step s/step bench of padded vs trie on 14B Qr rows.
3. **Drop the "share identity/regen prompts" idea:** measured 0.0% on this data (no within-arm repeats).
   Cross-arm sharing is impossible under causal attention because the system prompt comes first. The one
   cross-arm sharing that does work is the frozen-prefix cache (agent 05), and that only covers layers below the
   LoRA.

Files: `docs/efficiency3/patches/06_tree_batch.py` (method), `06_verify.py` (E1–E3), `06_trie_savings.py` (E4),
`06_verify_results_float64*.json`, `06_trie_savings_results.json`.
