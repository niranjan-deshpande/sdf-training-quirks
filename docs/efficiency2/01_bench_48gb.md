# 01: Qwen3-14B LoRA SFT on a 48 GB card: s/step and $/arm vs H100 (agent 01, GPU)

**Status: the measurement FAILED for an infrastructure reason (below). The $/arm numbers for 48 GB
cards are ESTIMATES scaled from the measured A40 numbers in docs/efficiency/02_kernels.md. The plan
to measure them properly costs ≈ $0.15 and needs one launcher fix.**

## TL;DR
- **No s/step measurement.** No L40S was in stock (RunPod `stockStatus: None`; $0.79/h community,
  $1.09/h secure), and neither was RTX A6000 ($0.33/$0.59). The RTX 6000 Ada (48 GB, **$0.74/h community**)
  pod came up, but `pod_setup.sh` failed at the torch check: *"NVIDIA driver on your system is too old
  (found version 12040)"*. The venv installs **torch 2.14.0 with the CUDA 13.0 runtime**, which needs a
  CUDA ≥ 13.0 driver, and this host's driver only supports 12.4. `launch_pod.sh` never asks for a
  minimum CUDA version, so any cheap community host with an older driver fails the same way after
  about 7 minutes of setup. **Fix: pass `--min-cuda-version 13.0` to `runpodctl pod create`** (patch below).
  This is measured, and it matters for every non-H100 plan.
- **Estimate:** an Ada 48 GB card (RTX 6000 Ada or L40S) should run the current recipe at about
  **4.2 s/step** and **≈ 2.9 s/step with `--liger --lora-kernel fused --ckpt-budget`**. That is
  **≈ $0.44–0.63 per arm at $0.74/h**, against **$1.12–1.40 per arm on Community H100** ($1.67–2.07 on
  Secure). So about 2–3× cheaper per arm and about 1.5–2× slower in wall time. The estimate is ±30%:
  it scales the A40 kernel breakdown by peak FLOPs and memory bandwidth.
- **The cheapest option already measured is the A40** (Secure, $0.59/h): 7.5 s/step with the current
  recipe → **$0.90/arm**, and 5.13 s/step with Liger + fused + ckpt-budget 30000 → **$0.61/arm**. Both are
  measured, but on examples truncated to ≤384 tokens. An RTX A6000 at $0.33/h community would be
  ~$0.33–0.48/arm, if one is ever in stock.

## What's new vs docs/efficiency
- 02_kernels.md measured the A40 only on truncated data (≤384 tokens) and never priced any card per arm.
  HANDOFF §0 compares A100 with H100 only. This file adds: (1) live RunPod prices and stock for every
  48 GB card at 17:55 EST on 2026-11-05; (2) a $/arm table across GPUs; (3) the driver/CUDA-13
  incompatibility that blocks cheap community hosts with the current venv (measured); (4) a ready-to-run
  ~$0.15 benchmark plan.

## Prices and stock (`runpodctl gpu list`, 2026-11-05 17:55 EST)
| GPU | VRAM | stock | community $/h | secure $/h |
|---|---|---|---|---|
| RTX A6000 | 48 | None | 0.33 | 0.59 |
| A40 | 48 | Low | — | 0.59 |
| L40 | 48 | None | 0.69 | 0.82 |
| RTX 6000 Ada | 48 | Low | **0.74** | 0.99 |
| L40S | 48 | None | 0.79 | 1.09 |
| A100 80GB PCIe | 80 | Low | 1.19 | 1.79 |
| RTX PRO 6000 Blackwell | 96 | Low | 1.69 | 2.49 |
| H100 NVL | 94 | Low | 2.59 | 3.19 |
| H100 80GB HBM3 | 80 | Low | 2.69 | 3.99 |

## Findings

### F1 (measured): cheap community hosts fail setup because the driver is too old for torch 2.14 + CUDA 13
- Mechanism: `pod_setup.sh` resolves an unpinned torch, which is now 2.14.0 with `nvidia-cuda-runtime==13.0.96`
  (from the pod's `pip_freeze.txt`). The RTX 6000 Ada host reported driver CUDA 12.4. H100 hosts
  have so far had new drivers, so this never showed up.
- Cost of the failure: ~8 min of pod time (~$0.10). The guard then *stopped* the pod instead of
  deleting it, because the pull dir did not exist ("COPY FAILED"), so the 50 GB volume kept billing until
  I deleted it by hand. Anyone launching cheap GPUs should know about this guard behavior.
- Fix (either one):
  1. In `launch_pod.sh` `create_pod`, add `--min-cuda-version 13.0` to `runpodctl pod create`. This shrinks
     the host pool somewhat, which matters most for the cheapest community cards.
  2. Or pin torch to a cu128 build in `scripts/pod_constraints.txt` (e.g. `torch==2.9.*` with the matching
     cu128 index). That also runs on 12.8 drivers, but the 12.4 host here would still fail. Pinning
     also improves comparability across arms (same kernels), which HANDOFF already favors.
  ```diff
  -    out="$(runpodctl pod create --name "$POD_NAME" --gpu-id "$GPU" --cloud-type "$cloud" ${extra[@]+"${extra[@]}"} \
  +    out="$(runpodctl pod create --name "$POD_NAME" --gpu-id "$GPU" --cloud-type "$cloud" ${extra[@]+"${extra[@]}"} \
  +            --min-cuda-version "${MIN_CUDA:-13.0}" \
  ```
  Also: the guard should *delete* (not stop) when the run ended `setup_failed_*`, because nothing is worth keeping.

### F2 (estimate): $/arm on 48 GB cards vs H100
Method: 730 steps × s/step × $/h. On the A40, 57% of GPU time is GEMMs and ~40% is memory-bound
elementwise work (02_kernels.md profile). To estimate Ada 48 GB, scale the GEMM part by bf16 dense
FLOPs (A40 ~150 TF → L40S/RTX 6000 Ada ~362 TF) and the memory part by bandwidth (696 → 864–960 GB/s):
0.57·0.41 + 0.40·0.75 + 0.03 ≈ 0.56× the A40 time.

| GPU (price) | config | s/step | evidence | h/arm | $/arm |
|---|---|---|---|---|---|
| H100 community ($2.69) | current, pdb 8 / pdb 16 | 2.56 / 2.40 (40-step bench); 1.93–2.06 real Mch | measured | 0.39–0.52 | **1.05–1.40** |
| H100 secure ($3.99) | same | same | measured | 0.39–0.52 | 1.56–2.07 |
| A40 secure ($0.59) | current (pdb 8×2) | 7.5 | measured (≤384-tok data) | 1.52 | **0.90** |
| A40 secure ($0.59) | Liger + fused + ckpt-budget 30000 | 5.13 | measured (≤384-tok data) | 1.04 | **0.61** |
| RTX 6000 Ada community ($0.74) | current | ~4.2 | estimate ±30% | 0.85 | **~0.63** |
| RTX 6000 Ada community ($0.74) | Liger + fused + ckpt-budget | ~2.9 | estimate ±30% | 0.59 | **~0.44** |
| L40S community ($0.79) | same two | ~4.3 / ~3.0 | estimate | — | ~0.69 / ~0.48 |
| RTX A6000 community ($0.33) | current / flags | ~7.2 / ~5.0 | estimate (A40-like Ampere) | 1.46 / 1.0 | ~0.48 / ~0.33 |

Caveats:
- Fixed per-pod overhead is not included: setup took 376 s here (venv) plus model download in parallel,
  which is ~$0.08 on Ada and ~$0.29 on a Community H100. It is shared across the arms of one pod.
- Evals (vLLM probes) also run on the pod. The 28 GB of bf16 weights fit on 48 GB with KV room, but
  vLLM throughput on Ada/A40 is lower; that is outside this file's scope.
- Memory: with full checkpointing and Liger (no full-vocab fp32 logits), 48 GB fits pdb 8 at seq ≤2048
  (A40 peak 33–41 GiB on short data). Without Liger, logits for a long pdb-8 micro-batch can need
  10–25 GB → OOM risk on real data. **On 48 GB cards use `--liger`** (or pdb 4 × accum 4, which
  gives the same math as pdb 8×2 because loss is normalized per optimizer step).
- Comparability: changing the GPU changes kernels and numerics, much like a dropout-seed change.
  Arms within one comparison must run on the same GPU type and stack. Liger and fused are
  measured to be within the seed floor (02_kernels.md). ckpt-budget is bit-identical.

## Experiments run
```bash
# 17:54 EST; L40S/A6000 stock None, so RTX 6000 Ada community
bash scripts/launch_pod.sh --name eff3-ada --gpu "NVIDIA RTX 6000 Ada Generation" --max-hours 0.25 \
  --guard-pull outputs/bench_eff3_l40s --delete --run '<2 configs: pdb 4 and pdb 8, both --liger
  --optim adamw_torch_fused, 25 steps, chatty Mch data, maya arm, --no-push>'
```
- Pod spgrmyafp85fqx: created 22:54:51Z at $0.74/h; ssh in 2 s; model (28 GB) downloaded in ~4.5 min;
  venv ready at +376 s; then `RuntimeError: The NVIDIA driver on your system is too old (found version 12040)`
  → SETUP_EXIT=1, RUN_EXIT=setup_failed_1. The guard stopped the pod (its copy had failed) and I
  deleted it at 23:04Z (`runpodctl pod get` → 404). Total spent ≈ $0.12 (cap $0.90). No s/step obtained.
  Logs: `outputs/pods/spgrmyafp85fqx/{setup,run,download}.log, pip_freeze.txt`.
- Not enough time left to retry before the 18:14 watchdog (setup alone takes ~7 min).

## Recommendation / plan (next session, ≈ $0.15–0.25)
1. **Adopt** the `--min-cuda-version 13.0` fix (or pin torch to a cu128 build) before any non-H100 launch.
2. **Test next:** one RTX 6000 Ada community pod (or L40S / A6000 if in stock), `--max-hours 0.4`.
   Run the two configs above **plus** `--lora-kernel fused --ckpt-budget 20000` with 30 steps each.
   For timing, use per-step logs (`logging_steps=1` or a time callback). The current loss.jsonl logs every
   10 steps, so 25 steps give only one 10-step window. Pair it against
   `outputs/distill/bench/h100/train/a_current/loss.jsonl` (same data, seed and order; H100 steps 10→20 =
   2.06 s/step; loss at step 10 = 0.3792, step 20 = 0.3242). Keep `--max-steps` the same as that bench (40)
   so the LR schedule matches and the losses can be compared step by step (expect |Δ| ≲ 0.004, the seed floor).
3. If the Ada figure is ≤3.5 s/step with flags, **make a 48 GB community card the default** for new
   experiment families ($0.4–0.6/arm vs $1.1–1.4 Community H100). Keep H100 for comparisons that
   must match old arms bit for bit, and when wall-clock time matters.
