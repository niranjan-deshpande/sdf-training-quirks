# 08: Cheapest hardware per arm (14B LoRA SFT + vLLM evals), RunPod prices and stock 2026-11-05 (agent 08)

**Status: final (written 17:56–18:10 EST). Read-only market/benchmark research; nothing was created or run on a GPU.**
Evidence labels: **[live]** = read from the RunPod API at ~17:55 EST today; **[meas]** = measured in this repo;
**[spec]** = vendor datasheet; **[lit]** = third-party benchmark; **[est]** = my model, not measured.

## TL;DR
- **48 GB cards are ~2–3× cheaper per arm than H100 for 14B bf16 LoRA, even after the slowdown.** Modelled
  $/arm (setup + gen + train + evals): RTX A6000 / A40 ≈ **$0.70–0.80 Community, $1.25–1.32 Secure**;
  RTX 6000 Ada / L40S ≈ **$1.03–1.13 Community, $1.37–1.56 Secure**; H100 SXM **$2.06 Community, $3.06 Secure**
  (the Secure figure matches our real ~$2.95 arm, so the model is calibrated at the H100 end). Wall-clock cost:
  ~1.4 h (Ada) to ~2.2 h (Ampere) per arm vs ~46 min. **[est, train factor for A40 is meas]**
- **Best value given today's stock: RTX 6000 Ada (Community $0.74, stock Low) or L40S (Community $0.79, stock
  Low)**: ~2/3 of H100 speed per $ wasted on idle, half the wall-clock of an A40, ~$1.0–1.1/arm. A40/A6000 are
  cheapest on paper but showed **no Community stock** today (Secure A40 $0.59 had Low stock in EU-SE-1 on one read,
  none on the next). For 28B, the **RTX PRO 6000 (96 GB, Community $1.69, Low stock in 10 DCs)** is the only
  sub-$2 card that fits bf16 LoRA comfortably; A100 80 GB fits but is not cheaper than H100 per arm.
- **Rule out:** 24/32 GB cards (RTX 4090/5090, L4, PRO 4500) for 14B bf16 (weights alone 29.6 GB; would need a
  2-GPU split or QLoRA, both rejected/slow); A100 40 GB (fits training barely, vLLM KV cache ~6 GB); B200/H200
  (faster but ≥ H100 $/arm at our low GPU utilisation); MI300X (no stock, ROCm kernel/numerics risk).
  **Comparability:** every arm of one comparison must run on the same GPU type (and same per-device batch), so pick
  the card per *experiment family*, and re-run any H100 baseline arm you want to compare against.

## What's new vs docs/efficiency
docs/efficiency/02 measured A40 s/step (≤384-token examples) and 06 measured A40 pod startup, but nothing priced a
whole arm per GPU type or looked at the current market. HANDOFF §0 only compares Secure H100 vs Secure A100
("about the same per arm"). New here: live RunPod prices + stock for every relevant card, a calibrated
$/arm model including gen/eval/setup (not only training), memory-fit analysis for 14B and 28B, and a ranked
recommendation. Agents 01 (RTX 6000 Ada, real 2048-token data) and 02 (A100 / RTX PRO 6000) are measuring
s/step right now: their numbers should replace my train factors `t` in the table (slot marked **[01]/[02]**).

## 1. Live RunPod prices and stock (on-demand pods, 1 GPU) [live]
`list-gpu-types include=AVAILABILITY product=POD`, once with cloud=COMMUNITY and once SECURE, plus `get-capacity`.
"Stock" = what the API reported at that moment; it flipped between two reads a minute apart for A40.

| GPU | VRAM | Community $/h | Secure $/h | Community stock | Secure stock (DCs) |
|---|---|---|---|---|---|
| H100 SXM | 80 | 2.69 | 3.99 | None | Low (9 DCs) |
| H100 NVL | 94 | 2.59 | 3.19 | None | Low (US-KS-2) |
| H100 PCIe | 80 | 1.99 | 2.89 | None | None |
| H200 SXM | 141 | 3.59 | 5.29 | None | Low (CA-MTL-3, US-CO-1) |
| B200 | 180 | 5.98 | 7.99 | None | None |
| MI300X | 192 | — | 2.39 | — | None |
| RTX PRO 6000 (Server) | 96 | 1.69 | 2.49 | Low* | Low (10 DCs) |
| RTX PRO 6000 (Workstation) | 96 | 1.69 | 2.59 | Low* | Low (EU-CZ-1, EU-RO-1) |
| A100 80 GB SXM | 80 | 1.39 | 1.79 | Low* | Low (4 US DCs) |
| A100 80 GB PCIe | 80 | 1.19 | 1.79 | Low | Low (CA-MTL-3) |
| A100 40 GB SXM | 40 | 1.00 | — | None | — |
| L40S | 48 | 0.79 | 1.09 | Low* | Low |
| RTX 6000 Ada | 48 | 0.74 | 0.99 | Low* | Low (US-WA-1) |
| L40 | 48 | 0.69 | 0.82 | None | None |
| A40 | 48 | 0.35 | 0.59 | None | Low (EU-SE-1) on one read, absent on the next |
| RTX A6000 | 48 | 0.33 | 0.59 | None | None |
| PRO 6000 MIG 2g.48gb (half a PRO 6000) | 48 | — | 1.29 | — | Low |
| RTX 5090 | 32 | 0.69 | 1.19 | Low | Low (EUR-NO-1) |
| RTX 4090 | 24 | 0.34 | 0.89 | None | Low (5 EU DCs) |

\* `get-capacity` (either cloud) listed these as Low; the Community-only `list-gpu-types` read showed None. Treat
Community stock for everything except A100 PCIe / 5090 as "maybe; try, fall back to Secure".

## 2. Hardware specs that matter [spec]
Dense bf16 tensor TFLOPS (fp32 accumulate), memory bandwidth, VRAM. GeForce cards run fp32-accumulate at half rate.

| GPU | bf16 TFLOPS | BW TB/s | VRAM | notes |
|---|---|---|---|---|
| B200 | ~2,250 | 8.0 | 180 | needs CUDA 12.8+ wheels |
| H100 SXM / H200 | 989 | 3.35 / 4.8 | 80 / 141 | |
| H100 NVL / PCIe | ~835 / 756 | 3.9 / 2.0 | 94 / 80 | NVL 400 W, PCIe 350 W |
| MI300X | 1,307 | 5.3 | 192 | ROCm |
| RTX PRO 6000 Blackwell | ~500 | 1.6–1.8 | 96 | GDDR7, no NVLink, 600 W (Max-Q 300 W) |
| A100 80 GB | 312 | 2.0 | 80 | 40 GB: 1.56 TB/s |
| L40S / RTX 6000 Ada | ~362 / ~364 | 0.86 / 0.96 | 48 | Ada, 350 / 300 W |
| L40 | 181 | 0.86 | 48 | |
| A40 / RTX A6000 | 150 / 155 | 0.70 / 0.77 | 48 | Ampere |
| RTX 5090 / 4090 | ~210 / 165 | 1.79 / 1.0 | 32 / 24 | consumer, half-rate fp32 acc |

**Key measured fact [meas]:** H100 vs A40 on our recipe is only **~2.9× per step** (H100 2.56 s/step;
A40 7.51 s/step, same pdb 8×2 recipe, docs/efficiency/02 §4; per token 2.1k vs 0.66k tok/s), against **6.6×**
in TFLOPS and 4.8× in bandwidth. The A40 profile is 57% GEMM / 40% memory-bound elementwise. So our LoRA step is
far from peak on an H100, and **paying for TFLOPS buys less than the spec ratio suggests**: the cheaper card wins
on $/arm whenever its price ratio exceeds ~3×. That is the core reason the ranking below favours 48 GB cards.

## 3. Memory fit [spec + meas]
- **14B bf16 LoRA r64 (all proj):** base weights 29.6 GB; LoRA params 257M → params+grads+Adam fp32 ≈ 4.1 GB;
  measured peak **37.5 GiB** on A40 at pdb 8 with grad checkpointing and ≤384-token examples (33.4 GiB with Liger).
  Fits 48 GB at pdb 8×accum 2 (eff. batch 16; HANDOFF: losses identical across pdb choices). Long real examples
  (seq up to 2048) need Liger and/or smaller pdb on 48 GB: **agent 01 is measuring this on real data**.
  `--ckpt-budget` gains shrink on 48 GB (+9% vs ~25% expected on 80 GB).
- **vLLM 14B on 48 GB:** 29.6 GB weights leave ~12–13 GB KV at 0.9 utilisation (A40 fits ~200 concurrent
  seqs, docs/efficiency/04). Fine for our batch gen/evals; somewhat slower than H100's 256-seq cap.
- **24/32 GB (4090, 5090, L4, PRO 4000/4500, MIG 24 GB):** do not fit 14B bf16 weights + LoRA state on one card.
  A 2-GPU `device_map` split runs one GPU at a time (≈ a single 4090's speed, ~2.6× H100 time at $0.68/h
  Community for 2×4090) and adds a code path; FSDP over PCIe without P2P is worse. Not worth it for 14B.
  They are the right card for **≤8B proxies** (Qwen3-4B arms ~15 min on an A40, docs/efficiency/05).
- **28B bf16 LoRA:** weights ~56 GB + LoRA state ~8 GB + activations → **80 GB is tight** (A100 80 / H100,
  pdb small, vLLM KV ~15 GB), **96 GB comfortable** (RTX PRO 6000), 141 GB roomy (H200). 48 GB cards are out.

## 4. $/arm model
`minutes = setup + gen·v + train·t + evals·(t+v)/2`, H100 SXM baseline setup 5, gen 6, train 25, evals 10
(HANDOFF). `t` = train slowdown vs H100 SXM, `v` = vLLM slowdown. Setup is longer on slow cards (A40 cold vLLM
init 357 s vs 96 s warm, docs/efficiency/06). Script: in the fenced block at the end.

| GPU | t (train) | v (vLLM) | min/arm | $/arm Community | $/arm Secure | basis for t |
|---|---|---|---|---|---|---|
| RTX A6000 | 2.8 | 3.2 | 127 | **0.70** | 1.25 | est (A40 + clock) |
| A40 | **2.93** | 3.5 | 134 | **0.78** | **1.32** | **meas** (≤384 tok) |
| RTX 6000 Ada | 1.7 | 2.3 | 83 | **1.03** | 1.37 | est → **[01] measuring** |
| L40S | 1.75 | 2.4 | 86 | 1.13 | 1.56 | est |
| L40 | 2.4 | 2.7 | 109 | 1.25 | 1.49 | est |
| RTX PRO 6000 | 1.1 | 1.4 | 54 | 1.53 | 2.26 | est → **[02] measuring** |
| H100 PCIe | 1.25 | 1.4 | 58 | 1.92 | 2.79 | est |
| A100 80 PCIe | 2.3 | 1.8 | 95 | 1.88 | 2.83 | est → **[02] measuring** |
| A100 80 SXM | 2.2 | 1.7 | 91 | 2.10 | 2.71 | est (HANDOFF "~2.2×") |
| H100 SXM | 1.0 | 1.0 | 46 | 2.06 | **3.06** | meas (real arm ≈ $2.95) |
| H100 NVL | 1.1 | 1.0 | 49 | 2.12 | 2.61 | est |
| H200 SXM | 0.95 | 0.8 | 42 | 2.53 | 3.73 | est |
| B200 | 0.6 | 0.5 | 30 | 2.94 | 3.93 | est |

How the estimated `t` were made: split the A40 step into 57% GEMM (scaled by TFLOPS ratio) + 40% memory-bound
(scaled by BW ratio) + 3% other. E.g. L40S: 0.57/2.4 + 0.40/1.24 + 0.03 = 0.59 × A40 → 1.73× H100. RTX PRO 6000:
0.57/3.3 + 0.40/2.3 + 0.03 = 0.38 × A40 → 1.1× H100; consistent with the Kaitchup report of RTX PRO 6000 finishing
a single-GPU LoRA recipe at about H100 speed and 30–40% cheaper [lit, paywalled, not verified]. `v` from bandwidth
ratios, softened because batched decode at ~200–256 seqs is partly compute-bound; CloudRift measured RTX PRO 6000
≈ H100 SXM on one-GPU vLLM decode of a 4-bit model (3,140 vs 2,987 tok/s) [lit], which is optimistic for bf16 14B.

**Plug-in thresholds for agents 01/02 (training only, vs H100 SXM at 2.40 s/step, same cloud tier).** A card is
cheaper per arm on training if its s/step is below 2.40 × (H100 $/h ÷ card $/h): RTX 6000 Ada < 8.7 (Community) /
9.7 (Secure); L40S < 8.2 / 8.8; A40 < 18.4 / 16.2; RTX PRO 6000 < 3.8 / 3.8; A100 80 PCIe < 5.4 / 5.3. My estimates:
Ada/L40S ~4.1–4.2 s/step, A40 7.0 (measured 7.5 at ≤384 tokens without Liger), PRO 6000 ~2.6, A100 ~5.3–5.5. So the
48 GB conclusion is robust to a 2× miss; the A100 conclusion (≈ break-even) is not, and PRO 6000 wins only if it
really runs near H100 speed.

**Sensitivity.** If A40 vLLM is 5× H100 rather than 3.5× (it is bandwidth-bound and init is slow), the A40 arm goes
134 → 151 min, $1.32 → $1.48 Secure: still half of H100 Secure. The 48 GB ranking flips only if their train
factor is >~4.5× H100 (A40) or >~3× (Ada), which the A40 measurement rules out for Ampere. Idle pod time (judge
hand-off, debugging; ~13–16 min per run before automation, docs/efficiency/04) costs 4–7× less on a 48 GB card,
which widens the gap in practice.

**Wall-clock.** A 48 GB arm takes ~1.4–2.2 h instead of ~0.8 h. Because $/arm is per-GPU-minute, running N arms
on N single-GPU pods (or one N-GPU pod, one arm per GPU) costs the same and recovers wall-clock; multi-GPU pods of
48 GB cards are commonly available (maxCount 8–10 per pod).

## 5. Other clouds (brief) [lit, mid-2026 aggregator snapshots, not verified live]
- **Vast.ai** (marketplace, host-set prices, some interruptible): H100 SXM ~$1.8–2.1/h, L40S ~$0.6/h, A100 80 GB
  as low as ~$0.3–0.4/h interruptible. Cheapest raw $/h, but host quality/disk/network vary, and our pod scripts
  (scripts/*.sh, MFS gotchas in HANDOFF §5) are RunPod-specific. Worth it only with a robust checkpoint/resume.
- **Lambda**: H100 ~$2.5–4/h on demand, A100 80 ~$2.8/h (Aug 2026 listing); no cheap 48 GB tier. No advantage.
- **Modal**: per-second serverless, H100 ~$3.95/h; good for bursty evals, not cheaper for 40–130 min arms.
- **TensorDock / other marketplaces**: similar to Vast (A6000/L40S/4090 tiers at Vast-like prices); not checked.
- **Spot/interruptible** (RunPod spot, Vast interruptible): ~30–50% off but a 1–2 h arm needs checkpoint+resume;
  train_sft.py would have to resume mid-epoch with identical data order to stay comparable. Not recommended for
  comparison arms until resume is tested to give identical losses.

## 6. Recommendation (ranked, for a NEW 14B experiment family; keep all arms on one card type)
1. **Adopt for the next family: RTX 6000 Ada or L40S, Community (~$0.74–0.79/h; ~$1.0–1.1/arm modelled), Secure
   fallback ($0.99–1.09/h; ~$1.4–1.6/arm).** ~2× cheaper than H100 Secure, ~1.4 h/arm. Both are Ada with the same
   compute; whichever has stock. **Gate:** agent 01's measured s/step on real 2048-token data; adopt if
   t ≤ 2.5× H100 and no OOM at eff. batch 16 (pdb 8×2 or 4×4).
2. **Cheapest if stock exists: A40 / RTX A6000 (Community $0.33–0.35, Secure $0.59; ~$0.7–1.3/arm).** Measured
   2.9× slower training; ~2.2 h/arm. Stock was absent or flickering today, so do not plan a family around it
   unless a multi-pod launch can get N of them at once. Note Ampere vs Ada numerics differ, so don't mix.
3. **28B or tight-memory work: RTX PRO 6000 Community ($1.69/h; ~$1.5/arm for 14B est.).** Best $/arm among
   80+ GB cards, H100-like speed, fits 28B bf16 LoRA with room for vLLM. **Gate:** agent 02's measurement and a
   Blackwell (sm_120) wheel check for torch/vLLM/Liger (CUDA 12.8+; the API shows hosts on CUDA 13.2).
4. **Keep H100 only** for continuing the existing Qwen3-14B comparisons (all prior arms are H100; mixing GPU types
   inside a comparison is a numerics change of roughly a dropout-seed's size or more) or when wall-clock matters.
   Prefer H100 NVL Secure ($3.19) over SXM Secure ($3.99): ~15% cheaper per arm at ~same speed (est).
5. **Reject:** A100 80 GB (same $/arm as H100 at Secure, ~2× slower; only Community $1.19 PCIe is marginally
   better than H100 Community), H200/B200 (faster but costlier per arm at our utilisation), MI300X (no stock,
   ROCm), 24/32 GB consumer cards for 14B (don't fit; good for ≤8B proxies), MIG 48 GB slice ($1.29 Secure for half
   a PRO 6000 is worse than a whole L40S).

**How to validate cheaply (~$0.3):** one 10-step `bench_train.sh` on the chosen card with the real data (agents 01/02
are doing exactly this), plus one `evals_vllm` timing of the untouched model's sample+checks pass to replace my `v`
guess. Then plug the numbers into the script below.

## Experiments run
None on GPUs (read-only agent). Commands: RunPod MCP `list-gpu-types` (COMMUNITY and SECURE, include=AVAILABILITY,
product=POD) and `get-capacity` (filtered to 48–96 GB cards, includeUnavailable=false), ~17:54–17:56 EST. The cost
model below reproduces the table.

```python
# $/arm model (agent 08). Replace t/v with agents 01/02 measurements when available.
S,G,T,E = 5,6,25,10          # H100 SXM minutes: setup, gen, train, evals
g = {  # name: (t train factor, v vLLM factor, setup min, community $/h, secure $/h)
 "H100 SXM":(1.0,1.0,5,2.69,3.99), "H100 NVL":(1.1,1.0,5,2.59,3.19), "H100 PCIe":(1.25,1.4,5,1.99,2.89),
 "H200 SXM":(0.95,0.8,5,3.59,5.29), "B200":(0.6,0.5,6,5.98,7.99), "RTX PRO 6000":(1.1,1.4,6,1.69,2.49),
 "A100 80 SXM":(2.2,1.7,6,1.39,1.79), "A100 80 PCIe":(2.3,1.8,6,1.19,1.79), "L40S":(1.75,2.4,7,0.79,1.09),
 "RTX 6000 Ada":(1.7,2.3,7,0.74,0.99), "L40":(2.4,2.7,7,0.69,0.82), "A40":(2.93,3.5,8,0.35,0.59),
 "RTX A6000":(2.8,3.2,8,0.33,0.59)}
for k,(t,v,s,c,sec) in g.items():
    m = s + G*v + T*t + E*(t+v)/2
    print(f"{k:14s} {m:4.0f} min  ${m*c/60:.2f} comm  ${m*sec/60:.2f} secure")
```

Sources: RunPod API (live); docs/efficiency/02_kernels.md, 04_pipeline.md, 05_proxies.md, 06_infra.md; HANDOFF.md;
[CloudRift RTX PRO 6000 vs datacenter GPUs](https://www.cloudrift.ai/blog/benchmarking-rtx6000-vs-datacenter-gpus);
[Spheron H100 vs RTX PRO 6000 fine-tuning](https://www.spheron.network/blog/h100-vs-rtx-pro-6000-blackwell-fine-tuning-benchmark-2026/)
(H100 measured; PRO 6000 figure is a projection from The Kaitchup + Exxact 4,962 tok/s on 8B LoRA seq 512);
[The Kaitchup](https://kaitchup.substack.com/p/faster-and-cheaper-than-the-h100) (paywalled);
price aggregators [gpu-cost](https://gpu-cost.pages.dev/), [deploybase](https://deploybase.ai/articles/cheapest-gpu-cloud-in-2026-provider-pricing-ranked),
[awesomeagents](https://awesomeagents.ai/pricing/gpu-rental-pricing/) for Vast/Lambda/Modal (snapshots, may be stale).
