# 04: Training several LoRA arms in one process on a shared frozen base (agent 04, desk research + CPU)

**Status: final (written 17:52–18:10 EST, 2026-11-05).** No GPU used. Evidence labels: **[meas]** measured
(here or earlier in this repo), **[lit]** from a paper, **[est]** my model of the cost, **[spec]** speculative.

## TL;DR
- **A single 14B arm already keeps the GPU busy, so sharing the base across arms saves almost no compute.** From the
  repo's own H100 numbers (pdb 8×2 → 16×1 went 2.56 → 2.40 s/step) the fixed cost per micro-batch, which is all
  that cross-arm batching (mLoRA / LoRAFusion style) can amortize, is ≈0.16 s of a 2.4 s step. The ceiling is
  **≈3% at K=2 and ≈5% at K=4 of training time on H100**, and less on slower cards. Concatenating the arms also
  multiplies activation memory by K, which uses up the headroom that `--ckpt-budget` turns into speed and
  doesn't fit at pdb 16 on 48 GB cards (agent 08's cheapest option). **[est from meas]** LoRAFusion's
  own paper says per-job fine-tuning already has enough arithmetic intensity; its multi-job gains come from
  pipeline bubbles across GPUs, which we don't have. **[lit]**
- **Round-robin multi-adapter training (one base, K adapters, each stepping on its own batch in turn) can be
  bit-identical to K separate runs.** I measured this on CPU with a tiny Qwen3 + PEFT, and it needs per-arm RNG
  state (the negative control without the RNG swap diverges). **[meas, CPU toy]** But it only saves the
  per-process overhead (imports, tokenization, model load from page cache, CUDA init): **≈0.5–1 min per arm, i.e.
  ≈$0.01 (48 GB Community) to ≈$0.03 (H100 Community) per arm**, and ~$0.02–0.15 per 2–6-arm experiment. Each
  extra arm also costs ~2.9 GiB of resident LoRA/Adam state, which on a 48 GB card comes out of the
  `--ckpt-budget` headroom. **[est]**
- **Recommendation: reject both forms for 14B. Don't build them.** Where packing pays (≤4B proxy models, where a
  5k-token micro-batch underfills a big GPU), plain multi-process concurrency on one GPU gets the same overlap
  with no code and exact per-run equivalence, because the base is small (8 GB at 4B), or you can just rent a
  smaller GPU. The one piece of this literature that transfers is LoRAFusion's single-job FusedLoRA kernel
  (1.27× kernel-level on H100), and that belongs to the kernels lane, not to multi-adapter training.

## What's new vs docs/efficiency
The prior study rejected "all arms concurrently on one GPU via MPS/processes" in one line (07_wildcard W7:
compute-bound, two 28 GB weight copies). It pointed to "multiplexing in one process" as the right version. Agent 01
then built one adapter with K personas × K fruits, a recipe and science change. **No one looked at K separate
adapters on one shared base**, which keeps the science identical. New here:
1. A quantitative ceiling for cross-adapter batching, derived from the measured pdb 8→16 step times.
2. A literature check (mLoRA/ASPEN, LoRAFusion, PLoRA, PEFT): PEFT forbids mixed-adapter batches in training,
   and the literature's gains come from regimes we're not in.
3. An exact-equivalence recipe for round-robin training (rows, order, seed, dropout, clipping, optimizer),
   verified bit-identical on CPU, with a negative control.
4. $/arm per GPU type, including the memory interaction on cheap 48 GB cards.

## Findings

### F1. Cross-adapter batching (concatenate K arms' micro-batches; mLoRA "BatchFusion", LoRAFusion, PLoRA)
**Mechanism.** One forward/backward over [batch_arm1; batch_arm2; …]. The frozen base GEMMs see K× the rows,
and each LoRA module routes row block k through adapter k. Our arms use the same rows in the same order, and
their lengths differ only by the system-prompt length (constant per arm). So the blocks are contiguous and
nearly the same width: the routing is a slice (`x[k*B:(k+1)*B]`) rather than a gather/SGMV kernel, and the
padding overhead is negligible.

**What it can save.** Only costs that are fixed per micro-batch rather than per token: reading the frozen
weights (27.5 GiB × ~3 passes ≈ 88 GB ≈ 26 ms on H100 HBM3, ~1% of a step), kernel-launch and PEFT Python
overhead not hidden behind GPU work, and small-GEMM inefficiency. Per-token costs don't shrink:
- the base GEMMs (~57% of GPU time in the A40 profile, 02_kernels §4);
- the memory-bound LoRA elementwise, cast and dropout work (~40%);
- attention and the cross-entropy loss.

**Number.** Measured on H100: pdb 8 accum 2 = 2.56 s/step and pdb 16 accum 1 = 2.40 s/step, with identical losses
(HANDOFF §0). Model a step as n_mb·c + v·T. Halving the number of micro-batches saved c ≈ 0.16 s, where c is
everything fixed per micro-batch, optimizer bookkeeping included. Fusing K arms into one micro-batch saves
c·(K−1)/K per arm-step:

| K | saving per arm-step (H100) | share of 2.40 s |
|---|---|---|
| 2 | 0.08 s | **3.3%** |
| 4 | 0.12 s | **5.0%** |
| 6 | 0.13 s | 5.6% |

This is an upper bound, because c also contains the per-optimizer-step overhead that fusion doesn't remove. On
slower GPUs (A40/L40S/A6000/A100), v·T grows while launch and Python overhead stay about the same, so the
fraction is smaller (≈1.5–3% at K=2). **[est from meas]**

**What it costs.** Activation memory goes up by K. On an 80 GB H100 that halves the headroom `--ckpt-budget`
spends on skipping recompute (estimated 10–25% on H100, +9% measured on A40). Losing even part of that cancels
the ≤5%. On 48 GB cards (27.5 GiB weights + ~3.8 GiB LoRA state + ~2.9 GiB per extra arm) a K=2 fused
micro-batch of ~10k tokens needs full checkpointing and Liger, and may not fit long batches at all. Net effect:
**≈0 to negative**. **[est]**

**Literature, and why it doesn't transfer.**
- **LoRAFusion** (Zhu et al., arXiv 2510.00206, H100/L40S, Megatron): up to 1.96× (avg 1.47×) over Megatron-LM,
  1.29× avg over mLoRA. The multi-job gain comes from pipeline bubbles (44% → 11% bubble ratio with 4 adapters)
  and load balancing across GPUs. The paper says fine-tuning already has enough arithmetic intensity per job,
  and the gains saturate at 4 adapters. Its single-GPU gain is attributed mainly to the **FusedLoRA kernel**
  (dropout fused with the down-projection, base GEMM fused with the up-projection; 1.27× avg kernel-level,
  1.21× per layer). That is a single-job optimization. **[lit]**
- **mLoRA / ASPEN** (Ye/Tang et al., arXiv 2312.02515, PVLDB 18): ASPEN (v1) reports 53% less memory than
  running separate PEFT jobs and ~17% more throughput on an A100, for 7B. The memory saving is the shared base,
  which only matters if we wanted K concurrent processes. The throughput gains in the VLDB version are against
  FSDP/TP across several GPUs, and the authors note 13B benefits less. **[lit]**
- **PLoRA** (Yan et al., arXiv 2508.02932): up to 7.5× makespan reduction for *hyperparameter search*, packing
  many small LoRA configs. ~1.8× of it is from amortizing the base and up to 3.9× from custom kernels. The
  regime is many concurrent configs with small per-config batches. I didn't check its exact batch sizes, but a
  gain that size can only exist where one job underfills the GPU. Ours doesn't at 14B. **[lit + spec]**
- **Punica / S-LoRA** (SGMV/BGMV): these are inference kernels for decode, where per-adapter batches are 1–8
  tokens. They're irrelevant to our training, and our contiguous blocks don't need a gather kernel anyway.
- **PEFT** (0.21.2, installed here): `adapter_names=` mixed batches raise "Cannot pass `adapter_names` when the
  model is in training mode." (`peft/tuners/lora/model.py:634`). Activating several adapters at once *sums*
  them, which is a different model. So F1 needs a custom LoRA forward, i.e. `_lean_lora_forward` with a
  per-block loop. **[meas: code read]**

**Correctness if one did it anyway.**
- Loss: sum over arms of each arm's own token-mean. The adapters' parameters are disjoint, so arm k's gradient
  equals its separate-run gradient. A plain mean over the concatenated batch would weight arms by their token
  counts, which is wrong.
- Gradient clipping: per arm, not one global norm.
- Dropout masks are drawn from one stream over the concatenated tensor, so they differ from the separate runs.
- The base GEMMs run with a different M, so cuBLAS may pick a different algorithm: not bitwise even with dropout 0.
- Overall: equivalent in distribution, like a dropout-seed change (the repo's floor is update cos ≈ 0.974). All
  arms of a comparison would have to be trained fused, and old single-arm references would no longer be
  bit-comparable.

### F2. Round-robin multi-adapter training in one process (exact, but saves little)
**Mechanism.** Load the base once and add K adapters (`get_peft_model` for the first, `add_adapter` for the
rest), with `set_seed(42)` before each so every arm gets the same LoRA init as its separate run would. For each
global step, for each arm k:
1. `set_adapter(k)`, which makes only arm k trainable;
2. restore arm k's CUDA (and CPU) RNG state;
3. forward/backward on arm k's own next micro-batch;
4. clip arm k's gradients to 1.0;
5. step arm k's AdamW and scheduler;
6. save arm k's RNG state.

Compute is exactly K separate runs; nothing is shared except the resident weights and the process.

**Measured equivalence (CPU toy).** `docs/efficiency2/patches/04_rr_equivalence.py`: tiny random Qwen3 (2 layers,
hidden 64), PEFT LoRA on all 7 projections with dropout 0.05, AdamW, cosine schedule with warmup, clipping 1.0,
seed 42, two arms that differ only in a "system" token, 12 steps.
- With the per-arm RNG swap, all 28 LoRA tensors per arm are **bit-identical** to the separate runs
  (`torch.equal`), and the per-step losses match exactly.
- Negative control (`--no-rng-swap`, arms share the global dropout stream): weights differ, max |Δloss|
  3–4e-3. Per-arm RNG state is the one thing that is easy to get wrong. **[meas]**

**Exact-equivalence checklist against `distill/train_sft.py`:**
1. **Rows and order.** Each arm needs its own sampler, built exactly as the Trainer builds it (length-grouped,
   generator seeded from `recipe["seed"]`). For Q/L-style arms the lengths differ by a constant, so the order
   comes out the same. Controls arms (rows with their own `system`) don't, so never share one index list
   across arms. Assert per arm that the sequence of row ids equals what a single-arm Trainer would yield.
2. **Init.** Call `set_seed(seed)` immediately before each adapter is created (verified above). Different-seed
   arms (seed replicates) are fine: each arm carries its own seed.
3. **Dropout.** Give each arm its own CUDA RNG state, initialized to the state right after `set_seed(seed)` at
   train start, as in a fresh run. Swap it in and out around every forward/backward. Non-reentrant
   checkpointing restores RNG at recompute, so it stays consistent.
4. **Optimizer and schedule.** Per-arm AdamW, or one AdamW with one param group per arm (AdamW is elementwise;
   use the same foreach/fused implementation as the baseline). The schedule is the same for all arms.
   Weight decay is 0. With `set_to_none`, the inactive arms' grads stay None.
5. **Clipping.** Per arm, with `max_grad_norm` 1.0 on that arm's own norm. The Trainer's global clip over all
   K arms would couple them.
6. **Loss normalization.** Reproduce the Trainer's `num_items_in_batch` path exactly (accum 1: the sum over
   the micro-batch's target tokens / its count), or the losses differ in the last bits.
7. **Per-arm callbacks.** Run `ProbeSignal` under `set_adapter(k)` with arm k's system set. Save with
   `save_pretrained(..., selected_adapters=[k])`. Keep a per-arm `run_info.json`, plus
   `"multi_adapter": {"arms": [...], "mode": "round_robin"}`.

**What it saves.** Per process, `train_sft.py` pays roughly:
- imports, ~7 s [meas, HANDOFF §0];
- tokenization of ~11.5k rows, ~5–10 s;
- model load from container disk, ~10–30 s warm page cache, longer cold;
- `get_peft_model` and CUDA context/warm-up, ~5–10 s.

That is ≈0.5–1 min, of which (K−1)/K is saved. Queuing arms in one pod session (HANDOFF §0.3) already removes
pod boot, the model download and venv setup, so this residual is all that is left.

**Its cost.** Each extra resident arm adds LoRA params + Adam moments = 257M × 12 B ≈ 2.9 GiB (fp32 grads freed
between arms). Worth little on 80–96 GB cards. On a 48 GB card (≈44.7 GiB usable; base 27.5 + arm 3.8 GiB)
K=2 leaves ≈ 9 GiB instead of ≈ 12 GiB for activations and the `--ckpt-budget` window, and K=3 ≈ 6 GiB. Losing
headroom on the card agent 08 recommends costs a few % of step time, about the same as or more than the
setup saving. **[est]**

### F3. $ per arm and per experiment, by GPU (Community prices read by agents 02/08 today)

Training-time per arm from HANDOFF/agents 01/02/08 (H100 measured; others estimated there). Savings are mine
**[est]**:

| GPU (Community $/h) | 14B train $/arm | F2 round-robin saving/arm | F1 fused K=2 / K=4 saving/arm (ceiling) | fits K arms? |
|---|---|---|---|---|
| H100 SXM ($2.69) | ≈$1.31 (730 × 2.40 s) | $0.02–0.045 | $0.04 / $0.065, minus lost ckpt-budget headroom | RR K≤6; fused K=2 at pdb16 with less ckpt-budget |
| A100 80 GB PCIe ($1.19) | ≈$1.2–1.4 [02/08] | $0.01–0.02 | ≈$0.02–0.04 | as H100 |
| RTX PRO 6000 96 GB ($1.69) | ≈$1.0–1.4 [02/08] | $0.015–0.03 | ≈$0.03–0.05 | RR K≤8; fused K=2 comfortably |
| RTX 6000 Ada / L40S 48 GB ($0.74–0.79) | ≈$0.7–0.9 [08] | $0.006–0.013, ≈ offset by lost headroom | fused not viable at pdb16 (OOM risk) | RR K=2 (K=3 tight) |
| A40 / A6000 48 GB (Secure $0.59; Community stock none today) | ≈$0.6–0.8 [08] | ≈$0.005–0.01 | not viable | RR K=2 |
| RTX 4090/5090 (24/32 GB) | 14B bf16 doesn't fit (29.6 GB weights) | sharing the base doesn't change that | — | — |

**Per experiment (2–6 arms): ≤ $0.03–0.25 on H100, ≤ $0.01–0.05 on 48 GB cards.** Measured against ≈$0.7–2 per
arm, that is 0.5–5%, before the engineering (≈3–5 h for a careful round-robin trainer) and a validation run.
The cheaper the GPU, the less multi-adapter training is worth: slower cards have smaller fixed-cost fractions
and tighter memory.

### F4. Where shared-base packing *would* pay, and the simpler alternative
At ≤4B the per-arm micro-batch (≤16 rows, ~5k tokens, fixed by effective batch 16; LoRA Without Regret says
not to grow it) underfills a large GPU. Prior data: 4B ran only ~2× faster per token than 14B on an A40 vs
3.5× ideal (05_proxies). There, packing K arms could plausibly give 1.3–1.6× throughput [spec]. But the base is
small (4B = 8 GB), so **K separate processes on one GPU** (optionally with CUDA MPS) get the same overlap.
Each process is exactly a separate run (same kernels, own RNG), and no code is needed. Better still, put each
arm on a cheap small GPU so one arm fills it. Shared-base in-process training is dominated in both regimes. The
OLMo 3 checkpoint sweep can't use it at all: every checkpoint is a different base.

## Experiments run
- `.venv/bin/python -I docs/efficiency2/patches/04_rr_equivalence.py` (CPU, 4 s; torch/transformers 5.20.2/peft
  0.21.2): `arm q: 28 LoRA tensors, bit-identical=True, max|dloss|=0`; `arm l: … bit-identical=True,
  max|dloss|=0`.
- Same with `--no-rng-swap`: `arm q: bit-identical=False, max|dloss|=0.00428`; `arm l: bit-identical=False,
  max|dloss|=0.003`.
- Code read: PEFT 0.21.2 `lora/model.py:634` refuses `adapter_names` in training mode.
- Not run: any GPU measurement (desk agent). The F1 ceiling rests on the measured pdb 8→16 delta; the actual
  fused gain was not measured.

## Cheap validation plan (only if someone wants to revisit; ≈$0.2–0.4)
On the experiment's GPU, 14B, `--max-rows 400 --max-steps 20 --no-push`:
1. **Determinism floor.** Run single-arm Q twice and compare per-step losses and LoRA tensors. If they aren't
   bitwise reproducible (sdpa backward, cuBLAS), the floor is that rerun difference, not 0.
2. **Round-robin vs separate.** Run Q and L separately, then K=2 round-robin with the checklist above. Pass:
   bitwise equal, or |Δloss| ≤ floor and update cosine ≥ 0.999 per arm (well above the 0.974 seed-change
   floor). Log wall time: fused process vs the sum of separate processes, both on a warm page cache.
3. **(F1 only)** Fused K=2 with per-arm loss sum and per-arm clipping. Pass: cosine ≥ 0.97 (dropout-seed
   class) and s/arm-step < separate × 0.97. Otherwise drop it.
   Gate everything on peak memory (`torch.cuda.max_memory_allocated`) with the same `--ckpt-budget` as the
   single-arm baseline.

## Implementation sketch (round-robin, for the record; not recommended to build now)
```python
# distill/train_multi.py (new): --arms qwen=data/.../train_Q.jsonl llama=data/.../train_L.jsonl --out-root ...
model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
for i, (arm, rows) in enumerate(arms):
    set_seed(seed)                                         # identical LoRA init to a separate run
    model = get_peft_model(model, cfg, adapter_name=arm) if i == 0 else model.add_adapter(arm, cfg) or model
model.gradient_checkpointing_enable({"use_reentrant": False}); model.enable_input_require_grads()
torch.manual_seed(seed); rng0 = (torch.get_rng_state(), torch.cuda.get_rng_state())
state = {}
for arm, rows in arms:
    ex = tokenize(rows, tok, SYSTEMS[arm], seq_len)
    sampler = LengthGroupedSampler(16, lengths=[e["length"] for e in ex],
                                   generator=torch.Generator().manual_seed(seed))  # = Trainer's sampler; assert
    ps = [p for n, p in model.named_parameters() if f".{arm}." in n]
    opt = torch.optim.AdamW(ps, lr=lr, weight_decay=0.0)   # same impl/foreach flag as the Trainer baseline
    state[arm] = dict(ps=ps, opt=opt, sch=get_cosine_schedule_with_warmup(opt, warm, steps),
                      it=iter(DataLoader(ex, 16, sampler=sampler, collate_fn=Collator(pad))), rng=rng0)
for step in range(steps):
    for arm, s in state.items():
        model.set_adapter(arm)
        torch.set_rng_state(s["rng"][0]); torch.cuda.set_rng_state(s["rng"][1])
        with torch.autocast("cuda", torch.bfloat16):
            loss = model(**to_cuda(next(s["it"]))).loss     # match Trainer's num_items_in_batch normalization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(s["ps"], 1.0)
        s["opt"].step(); s["sch"].step(); s["opt"].zero_grad(set_to_none=True)
        s["rng"] = (torch.get_rng_state(), torch.cuda.get_rng_state())
        # per-arm LossLog / ProbeSignal (under set_adapter(arm)) / mid-run save_pretrained(selected_adapters=[arm])
```
Known traps:
- the Trainer's accelerate autocast wrapper and the fp32 output convert;
- the Trainer reseeds at `train()`;
- `ProbeSignal` must not consume the training RNG (it runs in eval mode; dropout is off);
- `add_adapter` returns None on `PeftModel`.

## Recommendation
- **Reject** cross-adapter batching (F1) for 14B arms. The ceiling is ≈3–5% of H100 training time, likely
  cancelled by activation memory. It's infeasible on the 48 GB cards that minimize $/arm. It would also change
  numerics to the dropout-seed class.
- **Reject (don't build)** round-robin multi-adapter training (F2). It's exactly equivalent, but saves ≈$0.01–0.03
  per arm, and on 48 GB cards it eats `--ckpt-budget` headroom. Queuing arms per pod already captures the real
  setup savings.
- **For ≤4B proxy runs with many arms:** use K separate processes on one GPU (exact) or a cheaper GPU, not
  shared-base code.
- **Hand to the kernels lane:** LoRAFusion's single-job FusedLoRA Triton kernel (github.com/CentML/lorafusion),
  1.27× kernel-level on H100 [lit]. It goes further than our `--lora-kernel fused` (it also fuses dropout with
  the down-projection and in the backward). Numerics are in the dropout-seed class, so the existing loss/cosine
  gate applies.

Sources: LoRAFusion https://arxiv.org/abs/2510.00206 · mLoRA/ASPEN https://arxiv.org/abs/2312.02515 and
https://www.vldb.org/pvldb/vol18/p1948-tang.pdf · PLoRA https://arxiv.org/abs/2508.02932 · PEFT source (installed
0.21.2) · repo: HANDOFF §0, docs/efficiency/02_kernels.md, 05_proxies.md, 07_wildcard.md, docs/efficiency2/02, 08.
