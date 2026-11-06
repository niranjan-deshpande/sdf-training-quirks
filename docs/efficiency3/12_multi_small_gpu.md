# 12: Splitting 14B bf16 LoRA across 2–4 cheap 20–32 GB cards (agent 12)

Written 2026-11-06, 00:00–00:50 EST. No GPU used: live RunPod price and stock reads only, plus a CPU simulation
on the real train_Q lengths. Labels: **[live]** = RunPod API read at ~00:05 EST; **[meas]** = measured here or in
the repo; **[sim]** = simulation on real data; **[code]** = read from source; **[spec]** = vendor datasheet;
**[est]** = my model.

## TL;DR
- **Only a real pipeline beats a 48 GB card, and only on RTX A5000s.** A correct 1F1B pipeline over 2× RTX A5000
  Secure ($0.54/h per pod) costs **≈$0.49/arm** for training, against **≈$0.79** on one A40 Secure. It is
  ≈0.9 h of wall time against 1.35 h. On Community (none in stock today) the figures are ≈$0.29 against ≈$0.47
  for A40. Every other consumer card loses or ties on price: 2×4090 Secure ≈$1.14, 2×5090 ≈$1.10, 2×3090 ≈$1.19.
  At Community prices 2×4090 is ≈$0.44 and 2×3090 ≈$0.53, about the same as an A40, but neither has multi-GPU
  Community stock. **[est]**, ±30%: the model is calibrated on one A40 benchmark, with no multi-GPU measurement.
- **The easy versions don't pay.** Naive layer split (`device_map="auto"`) keeps one card busy at a time, so it
  costs N × price × T₁. That is ≈$0.84/arm on 2×A5000 Secure, worse than an A40. FSDP/ZeRO-3 over PCIe has to
  all-gather the frozen 29.5 GB of weights 2–4× per step: about 59 GB per step at 2× 24 GB, i.e. ~2.5–5 s of
  transfer per step against ~3–4 s of compute per rank. At best it ties the pipeline (only with P2P, which GeForce cards lack).
  Tensor parallel puts about 16 GB of all-reduces per step on the critical path. **Reject both** for this
  recipe. The pipeline's own traffic is about 130 MB per step (~13 ms): negligible. **[est/sim]**
- **The stock lever is real but narrow today.** The only cheap 24 GB card with 2× or 4× stock is the RTX A5000
  (Secure Low, CA-MTL-1). That is the same datacenter where A40 Secure shows High stock right now. No consumer
  card has Community 2-GPU stock. A 2×4090 pod is available (Secure Medium) but costs $1.78/h, 3× an A40. So keep
  "2×A5000 Secure" as a fallback when A40/A6000 run out. It needs a pipeline implementation
  (~250 lines, outside HF Trainer) or the unverified 40-line in-Trainer variant (§3.4), and a ~$0.15 two-GPU
  benchmark before adoption. The sampler fix for any DDP/FSDP run is **verified on CPU** (§5).

## What's new vs fleet 1
Fleet 1 ruled out 24/32 GB cards for 14B on a single card (08, 13) and noted the DDP bug (15). Nobody modelled
multi-card splits or checked multi-GPU stock. New in this file:
1. live 2×/4× stock;
2. a step-time model per parallelism type, with a pipeline-bubble simulation on the real step composition;
3. a $/arm table;
4. a new trick: interleave two matched arms in one pipeline to fill the bubble;
5. a CPU-verified finding that fixing only `accum` in train_sft is not enough for a correct DDP recipe, because
   the sampler order also changes, and a one-line fix for that;
6. the FSDP-specific hazards in train_sft (probe deadlock, adapter save).

I also flag a likely calibration bias in fleet 1's 48 GB $/arm. The A40 benchmark used rows ≤384 tokens (mean ≈310),
while the real mean is 403 **[meas]**. So real A40 s/step is probably ≈1.3× fleet 1's figure: ≈$0.79 rather
than $0.61/arm on A40 Secure, training only **[est]**. That doesn't change any ranking below, because every row
is scaled from the same A40 anchor.

## 1. Prices and stock (RunPod, ~00:05 EST) [live]
`runpodctl gpu list --include-unavailable`; MCP `list-gpu-types include=AVAILABILITY product=POD count=2`
(cloud COMMUNITY and SECURE); `get-capacity gpuCount=1/2/4`, filtered to the cards below.

| card | VRAM | $/h per GPU, Community / Secure | 1× stock | 2× stock | 4× stock |
|---|---|---|---|---|---|
| RTX 3090 | 24 | 0.22 / 0.50 | none | none | none |
| RTX 4090 | 24 | 0.34 / 0.89 | Low (5 DCs) / Medium | **Secure Medium** (EU) | Secure Low |
| RTX 5090 | 32 | 0.69 / 1.19 | Low | none | none |
| RTX A5000 | 24 | 0.16 / 0.27 | Low (CA-MTL-1) | **Secure Low** (CA-MTL-1) | Secure Low |
| RTX A4500 | 20 | 0.19 / 0.25 | none | none | none |
| RTX PRO 4500 (Blackwell) | 32 | — / 0.72 | Secure Medium | Secure Low | Secure Low |
| A40 (reference) | 48 | 0.35 / 0.59 | **Secure High** (CA-MTL-1), EU-SE-1 Medium | Secure Low | Secure Low |
| RTX A6000 (reference) | 48 | 0.33 / 0.59 | Low (EU-SE-1) | none | — |

The `count=2` Community read showed **no** stock for any card in this table. Only A100 SXM had any. Multi-GPU
pods are billed at N × the per-GPU price; there's no discount.

## 2. Step-time model [est, anchored on measured A40]
- **Work per step.** 723 steps, 6,440 tokens per step (4.656M tokens / 723) **[meas]**. Frozen base plus full
  activation checkpointing gives ≈6 × 14.0e9 × tokens = 0.54 PFLOP per step. Attention is ≤1% of that.
- **Anchor.** A40 with Liger + fused + ckpt-budget measured 5.13 s/step on ≤384-token rows (docs/efficiency/02).
  × 403/310 gives **6.7 s/step on real rows, or 1.35 h per arm**.
- **Per-card factor k** (time relative to A40). Fleet 1's split: 57% of A40 time is GEMMs, which scale with dense
  bf16 TFLOPS at fp32 accumulate (GeForce runs at half rate). 40% is memory-bound and scales with bandwidth. 3% is
  other. **[spec]** TFLOPS / GB/s: 3090 71/936, 4090 165/1008, 5090 209/1792, A5000 111/768, A4500 95/640,
  A40 150/696. Results: k = 3090 1.53, 4090 0.82, 5090 0.59, A5000 1.16, A4500 1.37.
- **Memory (2 stages on 24 GB).** Half the weights (13.75 GiB) plus half the LoRA fp32 states (1.9 GiB) plus
  2.1 GiB context and margin comes to ≈17.8 GiB, leaving **≈6 GiB for activations** per card. A 1F1B stage-0
  card holds ≤N in-flight micro-batches of checkpoints at 0.2 MB/token per 20 layers, plus a ≤0.8 GiB recompute
  transient at pd 2. That fits at pd 1–4; at pd 2 there is room for some `--ckpt-budget`. A 20 GB A4500 leaves
  ≈2 GiB per card, which only works at 4 cards. **[calib from 13's model]**
- **Pipeline traffic.** 10 KB per token per boundary, forward plus backward, is ≈130 MB per step per boundary.
  At ~10 GB/s over host-bounced PCIe that is ≈13 ms per step: <0.5%.
- **Bubble and per-micro-batch overhead [sim].** The script reproduces the HF `LengthGroupedSampler` order
  (batch 16, seed 42) on the real lengths. It splits each 16-row step into m micro-batches, each padded to its
  longest row. It then runs an event simulation of GPipe-with-flush and 1F1B, with backward plus recompute at 2×
  forward. The table gives step time / (T₁ at pd 8 / N). "ovh" is a fixed per-micro-batch cost in
  token-equivalents for the whole model. 0 means none; 100 is pessimistic, since kernel-launch overhead is
  ≈40 token-equivalents on an Ampere card by 13's ~12k launches per micro-batch.

| N | m (pd) | 1F1B, ovh 0 | 1F1B, ovh 100 | 1F1B + 2 arms interleaved, ovh 0 / 100 |
|---|---|---|---|---|
| 2 | 4 (4) | 1.24 | 1.28 | 1.12 / 1.16 |
| 2 | 8 (2) | **1.12** | 1.22 | 1.06 / 1.15 |
| 2 | 16 (1) | 1.05 | 1.27 | 1.02 / 1.24 |
| 4 | 8 (2) | 1.36 | 1.49 | 1.18 / 1.29 |
| 4 | 16 (1) | **1.18** | 1.42 | 1.09 / 1.31 |

  GPipe and 1F1B have the same step time (GPipe differs by ≤0.017 in every row); 1F1B wins on memory. I use
  **f = 1.16 for N=2, m=8** and **f = 1.30 for N=4, m=16** (mid-range of ovh). The ideal bubble formula
  1 + (N−1)/m reproduces the ovh-0 column to within 0.01–0.02, so length variation within a step adds
  almost nothing: the sampler already sorts rows by length.

## 3. Options

### 3.1 Naive layer split (`device_map="auto"`, one process): works in HF Trainer, no speedup
- Trainer detects `hf_device_map` spanning more than one GPU, sets `is_model_parallel` and `_n_gpu = 1`
  (trainer.py:460–508 **[code]**). The batch arithmetic, sampler order and loss are therefore unchanged: same
  math as one card. It's about a 3-line change (patch Part 2).
- Only one card computes at a time, so time is T₁(card) and cost is N × price × T₁. 2×A5000 Secure: 1.56 h,
  **$0.84/arm**. 2×4090 Community: $0.75; 2×4090 Secure: $1.97. Never better than one A40 Secure ($0.79),
  except 2×A5000 at Community prices ($0.50). Use it only as a correctness baseline and for evals.

### 3.2 Pipeline parallel (1F1B, flush every optimizer step): the only option that pays
- **Algorithm.** Stage s holds layers [40s/N, 40(s+1)/N). Put the embedding on stage 0, and the norm, lm_head and
  Liger FLCE on stage N−1; that stage gets about one layer fewer, since lm_head is ≈6% of forward FLOPs. The 16
  rows of a step become m = 8 micro-batches of 2 rows for N=2 (m = 16 of 1 row for N=4). Run 1F1B. Normalize the
  loss as per-micro-batch token-sum / **global token count of the step**, which is known from the labels before
  the step starts. Then all-reduce the squared grad norms across stages for global clipping (max_grad_norm 1.0),
  take the AdamW step per stage, and use the same cosine/warmup schedule.
- **Exactness.** The gradient equals single-card accumulation: the same 16 rows and the same token-mean
  normalization. Only the micro-batch partition changes, i.e. dropout masks and summation order, which is the same
  class as the measured pdb-8×2 vs 16×1 change. **Numerics-class (tier 2 gate), not a recipe change**, provided
  clipping is global. Per-stage clipping would be a silent recipe change.
- **Implementation.** `torch.distributed.pipelining` (PipelineStage + Schedule1F1B) with a custom loop is
  ≈250 lines outside HF Trainer. It also has to merge the two stages' LoRA state into one PEFT adapter and run
  ProbeSignal through the pipeline (or skip it). Megatron/DeepSpeed pipelines are heavier.
- **Cost** (training only; Community / Secure, all [est]):

| config | wall h/arm | $/arm Community | $/arm Secure | stock now |
|---|---|---|---|---|
| **A40 ×1** (reference) | 1.35 | 0.47 (no stock) | **0.79** | Secure High |
| RTX A6000 ×1 | 1.27 | 0.42 | 0.75 | Low |
| L40S ×1 | 0.79 | 0.62 | 0.86 | Low |
| **RTX A5000 ×2, PP** | **0.91** | 0.29 | **0.49** | Secure Low |
| RTX A5000 ×4, PP | 0.51 | 0.33 | 0.55 | Secure Low |
| RTX 4090 ×2, PP | 0.64 | 0.44 | 1.14 | Secure Medium |
| RTX 4090 ×4, PP | 0.36 | 0.49 | 1.28 | Secure Low |
| RTX 3090 ×2, PP | 1.19 | 0.53 | 1.19 | none |
| RTX 5090 ×2, PP | 0.46 | 0.64 | 1.10 | none |
| RTX A4500 ×4, PP | 0.60 | 0.45 | 0.60 | none |

  Rule of thumb: $/arm = 1.35 h × k × f × (price per GPU). A pipeline beats one 48 GB card when k × f × p_gpu <
  p_48. Against A40 Secure ($0.59), N=2 (f ≈ 1.16) needs **k × p_gpu < 0.51 $/h**. At Secure prices only the
  A5000 passes (0.31); 4×A4500 also passes at f 1.30 (0.34 < 0.45) but has no stock. All Community prices pass
  (4090 0.28, 3090 0.34, 5090 0.41), but nothing had 2-GPU Community stock. Against an A40 *Community* card
  ($0.35, threshold 0.30) only A5000 and Community 4090 pass. Using 4 cards buys wall time, not dollars
  (f rises from 1.16 to 1.30).

### 3.3 New: interleave two matched arms in one pipeline [sim]
Matched arms (Q/L) share ~94% token-identical rows and the same frozen base. Each stage holds both arms' LoRA
for its layers: +1.9 GiB per card at N=2, which still fits (§2). The flush bubble at the end of arm A's step is
then filled with arm B's micro-batches. The simulation gives f = 1.12 → 1.06 at N=2/m=8 and 1.18 → 1.09 at
N=4/m=16. That is **≈5% at N=2 and ≈8–12% at N=4**, i.e. ≈$0.03/arm on 2×A5000. Each arm's math is unchanged:
separate adapters, optimizers and RNG (fleet 1's file 04 verified round-robin equivalence on CPU). Unlike
"2 arms on one 48 GB card" (rejected by fleet 1 because one arm already saturates the card), this fills idle
time that exists only in a pipeline. It's a small gain, and worth having only if the pipeline is built anyway.

### 3.4 Cheaper-to-build variant (speculative): micro-batched naive split inside HF Trainer
Run Trainer at pd 16 / accum 1 with `device_map="auto"`. A wrapper splits each 16-row batch into m chunks, runs
the forwards back to back, sums token-sum losses / step tokens, and does one backward. CUDA launches are
asynchronous, so stage 0 can run chunk j+1 while stage 1 runs chunk j. Autograd's per-device worker threads
pipeline the backward, which is 2/3 of the work. That is ≈40 lines, but overlap is limited by CPU issue order and
launch-queue depth. Guess: 1.3–1.7× over naive, so 2×A5000 Secure would come to ≈$0.50–0.65/arm. **Untested;
needs the two-GPU benchmark.**

### 3.5 FSDP / ZeRO-3 over PCIe [est]: reject
With pd chosen to fit in memory (pd 4 at N=2 on 24 GB), each rank gathers (N−1)/N × 29.5 GB per forward and per
backward of each micro-batch: ≈59 GB per step at N=2 and ≈44 GB at N=4. Over host-bounced PCIe (GeForce has no
P2P; ~10–12 GB/s) that is 4–5 s per step against 2.8 s of compute per rank on 4090s, so step time is about 1.5×
the pipeline's. With P2P on A5000s (~20–25 GB/s, unverified on RunPod hosts) it overlaps and roughly ties the
pipeline. The appeal is that FSDP is config-only in HF Trainer. But it needs the full DDP fix (§5), the probe
callback deadlocks under FSDP (rank-0-only forward), and the save needs `trainer.save_model`. Not worth it when
it at best ties the pipeline.

### 3.6 Tensor parallel [est]: reject
There are 2 all-reduces per layer in each of forward, recompute and backward: ≈6 × 40 × 6,440 tokens × 10 KB ≈
16 GB per step. That traffic is on the critical path: ≈1.3 s per step at N=2 against 2.8 s of compute (4090),
plus smaller per-rank GEMMs. LoRA training under HF `tp_plan` together with PEFT is immature.
**Evals are different:** vLLM `tensor_parallel_size=2` with LoRA is a supported config, and decode is
bandwidth-bound. A 2-card pod can therefore run gen and probes with 14.8 GB of weights per card and ~5 GB of KV
cache per card. That's not a blocker.

## 4. $ effect
- Per arm, training only: **2×A5000 Secure PP ≈$0.49 vs A40 Secure ≈$0.79 (−$0.30, −38%), 0.9 h vs 1.35 h.**
  Setup (~8 min, ≈$0.07) and evals are similar per pod-hour. Against fleet 1's 48 GB band of ≈$0.45–0.8, this
  matches its low end, now at Secure availability. For a 4-arm experiment that is ≈$1.2 saved, minus a one-time
  build (~half a day) and a ≈$0.15 benchmark.
- If A40/A6000 Community stock returns ($0.33–0.35/h), one 48 GB card at ≈$0.42–0.47 is about the same as
  2×A5000 Secure and needs no new code. **The pipeline's value is mostly as a stock fallback plus faster wall
  time, not as a large $ win.**

## 5. Multi-GPU bug and what a correct recipe needs (DDP/FSDP; also needed for OLMo on H200s)
- Fleet 1's bug (15): `accum = 16 // pd` ignores world size, so the effective batch doubles, steps halve and
  warmup is wrong.
- **New, verified [sim, exact]:** fixing `accum` alone still changes the data order. Under DDP the HF sampler is
  `LengthGroupedSampler(train_batch_size × accum)` with `train_batch_size = pd` per process (trainer.py:1097,
  training_args.py:1788 **[code]**). So its length-grouping megabatches become 50 × (16/N) rows. On the real
  train_Q order only **23/723** optimizer steps keep the single-GPU row set at N=2, and **0/723** at N=4.
  Building the sampler with the global step size (16), or with batch 8 and `mega_batch_mult=100`, gives
  **723/723**. Accelerate's `BatchSamplerShard` then gives rank k batches k, k+N, …, so step s sees rows
  [16s, 16s+16) as on one GPU. (There may still be a gap between my seeded generator and the Trainer's RNG; the
  check below covers it.)
- Checklist: `accum = 16/(pd × world)` with an assert. Steps 723 and warmup 21. Sampler fixed as above.
  `average_tokens_across_devices=True` (the default; it scales the loss by `num_processes`, trainer.py:2186
  **[code]**). Record world size and GPU SKUs in run_info. Probe callback: fine under DDP and naive split, but it
  deadlocks under FSDP. Under the naive split the 6.2 GiB probe-logit spike lands on the last card (use
  `--packed`). Run the tier-0 check from 15: compare the hash of the first 2,000 sampler indices and the logged LR
  curve against a 1-GPU dry run on CPU. A sketch is in `docs/efficiency3/patches/12_multi_gpu_recipe.diff`
  (not applied, not run).

## Evidence (commands)
- Prices and stock: `runpodctl gpu list --include-unavailable`; MCP `list-gpu-types` (count 2, COMMUNITY and
  SECURE); `get-capacity` (gpuCount 1, 2 and 4, includeUnavailable false), 00:02–00:06 EST.
- Simulation: `docs/efficiency3/patches/12_pipeline_sim.py`. It runs in two stages via cpu_slot.sh: (1) it
  tokenizes train_Q exactly like `train_sft.tokenize`, giving 11,558 rows, 4,656,029 tokens and mean 403, the
  same as 13's count (peak RSS 0.48 GB, 9.5 s); (2) it runs the sampler-order check and the pipeline event
  simulation (peak 0.18 GB, 2.8 s). Command:
  `LEN_CACHE=<scratchpad>/12_lengths.json bash <scratchpad>/cpu_slot.sh .venv/bin/python docs/efficiency3/patches/12_pipeline_sim.py`
  (run twice). The cost table comes from `<scratchpad>/12_cost.py` (the arithmetic in §2–3).
- An earlier single-stage run was killed during the 00:08 laptop crash. It produced no output and was not counted.

## Risks
- **Model.** k comes from a 57/40 GEMM/memory split measured only on the A40 and is applied to other cards. The
  A5000 is the same sm_86 architecture, so it's the safest extrapolation; GeForce clocks under sustained load are
  less certain. Pipeline overhead at pd 1–2 (ovh) is a guess between 0 and 100. **Every $ figure is ±30%.**
- **Hosts.** Multi-GPU RunPod hosts may give GPUs x8 lanes or put them across NUMA sockets. That doesn't matter
  for the pipeline (13 ms per step), but it matters a lot for FSDP and TP.
- **Comparability.** A new card type plus a micro-batch change is a numerics-class change (tier 2 gate). Run every
  arm of a comparison on the same pod type. A5000 is sm_86 like A40/A6000 but not bit-identical to them.
- **Engineering.** The pipeline is new code outside HF Trainer: loss normalization, global clipping, adapter
  merge and probes are each a place to introduce a silent recipe change. It needs tiers 0–2 from 15.

## Recommendation
1. **Don't build it for the next family if an A40/A6000 is available** (A40 Secure stock was High at 00:05).
   One 48 GB card plus fleet 1's flags is simpler and within ~$0.30/arm of the best multi-card option.
2. **Fix the DDP recipe anyway (§5, ~20 lines)** before any OLMo or H200 multi-GPU run. The sampler half of the
   fix is new and verified.
3. **If 48 GB stock keeps failing:** spend ≈$0.15 on a 2×A5000 Secure pod (Secure Low in CA-MTL-1) for 15
   minutes. Measure (a) naive split s/step, (b) the §3.4 micro-batched variant, and (c) `nvidia-smi topo -m`
   and the P2P bandwidth. Adopt the 2×A5000 pipeline only if (b) or a 1F1B prototype reaches ≤4.5 s/step, i.e.
   ≤$0.55/arm. Skip 4090/5090/3090 multi-GPU on Secure: those pods cost 1.5–2.5× an A40 per arm.
