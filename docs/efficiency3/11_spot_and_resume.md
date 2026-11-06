# 11: Spot/interruptible capacity and a bit-exact resume for train_sft.py

Agent 11, 2026-11-06 00:00–01:00 EST. Research and design only: no pods created, $0 spent. Price reads were made
with the RunPod MCP, an unauthenticated plus an authenticated **read-only** GraphQL query, and Vast.ai's public
offer search. Evidence labels: **measured** (run or read tonight), **estimate**, **literature**, **speculative**.

## TL;DR
- **RunPod spot is effectively unavailable to us, and Vast.ai is where the discount is.** RunPod's GraphQL catalog
  returned `null` for `secureSpotPrice`, `communitySpotPrice` and `minimumBidPrice` on every GPU type, both
  authenticated and unauthenticated (**measured**, 00:05 EST). REST v2, the MCP and runpodctl cannot create spot pods
  at all; only the GraphQL `podRentInterruptable` can (docs). Vast.ai interruptible (bid) offers, first quartile:
  A40 **$0.21/h** (on-demand $0.54), A100 80 GB SXM **$0.29/h** (on-demand $0.66), H100 SXM **$1.00/h** (on-demand
  $3.41), RTX 6000 Ada / L40S $0.40/h (**measured**). That is 30–70% below the same market's on-demand prices.
- **Our resume as written is not a resume.** With `save_only_model=True` it restarts Adam, the scheduler and the
  dropout RNG. A patched `train_sft.py` (`--resumable --resume auto`) is **bit-exact on CPU**: killed at step 7,
  resumed from the step-4 checkpoint, it gave a final adapter with 0/131,072 elements different from an
  uninterrupted run, and identical per-step loss and grad-norm (**measured**, tiny random-init Qwen3, real
  tokenizer, real chat template, group_by_length, 2 dataloader workers, dropout 0.05, accum 4). The as-written path
  diverged from the first resumed step; the final max|Δ| was 104% of the largest lora_B entry.
- **$ effect (estimate):** on Vast interruptible, about **$0.55–0.65 per arm on an A100 80 GB or A40**, and
  ≈$0.9/arm on an H100 SXM (46 min/arm). That compares with $1.32 for a Secure A40 and $3.06 for a Secure H100 on
  RunPod, and $0.78 for a Community A40 (which has had no stock). We assume 5–25% preemption overhead. This needs
  a Vast port of the pod scripts and one GPU tier-1 resume check (~$0.10) first.

## What's new vs fleet 1
Fleet 1 (08 §5, 15 F2) flagged spot as "~30–50% off, needs checkpoint+resume, not recommended until resume is
tested". New here:
- Live spot and interruptible prices (RunPod: none; Vast: the table in F1).
- The exact list of state a bit-exact resume needs, with the patch, and a CPU test showing it is bit-exact.
- A sensitivity check: dropout RNG on an unsaved device makes resume inexact (F3). This is the failure mode to
  look for on GPU.
- Checkpoint size (3.1 GB, not ~1 GB), checkpoint interval, and the cost model.
- Guard orchestration for preemption.

## F1. Prices and preemption behaviour

**RunPod (measured 00:01–00:05 EST).**
- On-demand prices are unchanged from fleet 1: A40 / A6000 Secure $0.59, L40S $1.09, RTX 6000 Ada $0.99, A100
  SXM $1.79 (Community $1.39), H100 SXM $3.99.
- **New: Secure A40 stock reads "High" right now.** That is the cheapest reliable 48 GB option, ≈$1.32/arm.
- Every spot field is `null`. The docs' concept graph says spot still exists via GraphQL `podRentInterruptable`
  (`bidPerGpu`) / `podBidResume`. A spot pod "can be stopped at any time, and a stop clears the container disk";
  the volume disk survives. A 2023 RunPod blog quoted A6000 spot $0.232 vs on-demand $0.491 (literature).
- I did not check the web console. Treat RunPod spot as "maybe exists, not visible to the API, not scriptable with
  our tools".

**Vast.ai (measured 00:02 EST).** Public `bundles` search, 1 GPU, rentable; prices in $/h; shown as min / Q1 /
median; n = number of offers. Bid prices are the current `min_bid`, with storage excluded (≈$0.2–0.33 per GB-month,
i.e. ≈$0.03/h for an 80 GB disk).

| GPU | interruptible (n) | on-demand (n) | Q1 ratio |
|---|---|---|---|
| A40 48 GB | 0.17 / **0.21** / 0.25 (9) | 0.48 / 0.54 / 0.60 (9) | 0.40 |
| RTX A6000 48 GB | 0.11 / 0.24 / 0.30 (12) | 0.30 / 0.35 / 0.43 (13) | 0.69 |
| RTX 6000 Ada 48 GB | 0.35 / 0.40 / 0.47 (5) | 0.54 / 0.60 / 0.65 (8) | 0.66 |
| L40S 48 GB | 0.33 / 0.40 / 0.47 (7) | 0.54 / 0.74 / 0.80 (12) | 0.54 |
| A100 80 GB SXM | 0.20 / **0.29** / 0.47 (16) | 0.38 / 0.66 / 0.82 (22) | 0.45 |
| A100 80 GB PCIe | 0.13 / 0.20 / 0.40 (11) | 0.27 / 0.54 / 0.74 (14) | 0.37 |
| H100 SXM | 0.67 / **1.00** / 1.71 (5) | 2.20 / 3.41 / 3.67 (5) | 0.29 |
| RTX PRO 6000 (server) | 0.47 / 0.78 / 1.33 (21) | 1.07 / 1.56 / 1.67 (24) | 0.50 |

**Vast behaviour (docs, literature).** An interruptible instance is "paused if outbid or if on-demand requested".
Its data is preserved while paused, and it resumes automatically when it regains priority. On-demand renters
always win; among bids, the highest bid wins. The docs state no notice period. Same-host resume is therefore cheap:
disk, venv and model are intact. A paused instance still bills storage.

**Preemption rate: unknown.** No public rate exists for either provider. Bidding at `min_bid` maximises the
interruption risk, and bidding 20–30% above Q1 should cut it (speculative). Below I use λ = 0.1 / 0.5 / 1
preemptions per hour. A pilot that logs actual pauses over ~5 arms would pin this down.

## F2. What a bit-exact resume must save, and the patch

HF Trainer already saves everything needed once `save_only_model=False` is set. We confirmed this from the code of
transformers 5.20.2 and from the test.

| state | where | why it matters | as written |
|---|---|---|---|
| LoRA weights (fp32 under PEFT autocast) | `adapter_model.safetensors` | the model | saved |
| AdamW `exp_avg`, `exp_avg_sq`, step | `optimizer.pt` (≈2.0× adapter size: **measured** 1.07 MB vs 0.53 MB on the tiny run) | resetting Adam moments changes every later update | **missing** |
| LR scheduler (cosine position, warmup) | `scheduler.pt` | | **missing** |
| python / numpy / CPU / **CUDA** RNG | `rng_state.pth` | LoRA dropout masks | **missing** |
| global_step, epoch, log history | `trainer_state.json` (written **last**, so it marks a complete checkpoint) | data position = global_step × accum micro-batches | saved |
| data order | not saved: recomputed. `LengthGroupedSampler` draws `randperm` from the global RNG after `Trainer.__init__` re-seeds, then `skip_first_batches` skips the consumed micro-batches. RNG is restored **after** the skip (`rng_to_sync`) | | ok (verified) |
| dataloader worker seeds | drawn at `iter()`; the collator is deterministic | | ok (verified with 2 workers) |
| grad-scaler | none (bf16) | | n/a |

**Patch: `patches/11_train_sft_resume.diff`.** It applies cleanly to HEAD and compiles. Changes:
- `--resumable` sets `save_only_model=False` and `save_total_limit=2`, so a checkpoint torn mid-write never
  leaves us with no complete one.
- `--save-every N` (default: unchanged, steps/2).
- `--resume auto|PATH`. `auto` picks the newest `checkpoint-N` that has all of trainer_state / optimizer /
  scheduler / rng.
- On resume it truncates rows with step > N from `loss.jsonl` and `probe_signal.jsonl` (they get recomputed
  identically), skips the duplicate step-0 probe, and appends `{"from_step", "at"}` to `run_info.json["resumes"]`.
- A hidden `--kill-at-step` (an `os._exit(137)` after step K) is used by the test.
- Not changed: `HubPush` still uploads each checkpoint folder **synchronously**. With `--resumable` that is
  3.1 GB per save at 14B, ~30–60 s blocking (estimate). Before production, make it `run_as_future=True`, or use a
  separate uploader process; see F5.

**Checkpoint size at 14B (estimate from the measured 2.0× ratio).** LoRA r64 on all 7 projections × 40 layers is
256.9M params; fp32 adapter 1.03 GB + Adam 2.06 GB ≈ **3.1 GB** (the brief's "~1 GB" is the adapter alone).
Writing to local NVMe takes ≈3–10 s; to a network volume ≈10–30 s (estimate).

## F3. Evidence: CPU exact-resume test (measured)

Setup: a random-init Qwen3 (2 layers, hidden 64, 4 heads / 2 KV, Qwen3-0.6B tokenizer and chat template, 9.8M
params), 192 real `train_Q` rows truncated to ≤400 characters (memory cap), `--per-device-batch 4` (accum 4),
12 steps, `--save-every 4`, the recipe's dropout 0.05 / cosine / warmup / group_by_length / 2 workers / seed 42.
Test-only env hooks are in `patches/11_resume_test_hooks.diff`: `USE_CPU` (see below), `LOGSTEPS=1`, `NO_PROBE`
(the 151k-vocab probe logits exceed the 2 GB cap), and `ORDER_LOG` / `RNG_LOG` for diagnosis. Each job ran through
cpu_slot.sh with peak RSS 1.1–1.4 GB.
```
bash docs/efficiency3/patches/11_resume_test.sh A   # uninterrupted, --resumable
bash .../11_resume_test.sh B   # --resumable --kill-at-step 7, then --resume auto (picks checkpoint-4)
bash .../11_resume_test.sh C   # as written (save_only_model=True), kill at 7, resume from checkpoint-4
python docs/efficiency3/patches/11_resume_compare.py out B C
```
| comparison | final adapter | checkpoint-8 adapter | per-step loss / grad-norm, steps 5–12 |
|---|---|---|---|
| A vs B (patched resume) | **max abs diff 0, 0/131,072 differ** | **0/131,072** | **identical** |
| A vs C (as written) | max abs diff 3.1e-4 = **104% of max lora_B**; all elements differ | 2.2e-4 (80%) | differs from step 5 on |

- **Sensitivity check, which became a finding.** The first run of B used Apple's MPS device, the Trainer default
  on a Mac. It was **not** exact (final max abs diff 1.3e-4). With dropout 0 it **was** exact, and the global CPU
  RNG hash never changed across steps. So dropout drew from the MPS generator, which `_save_rng_state` does not
  save: it saves only cpu / cuda / xla / npu / hpu / mlu / musa.
  - On CUDA the dropout RNG is saved, but this shows the test catches an unsaved RNG stream.
  - torch.compile/Inductor dropout and any custom `torch.Generator` would be the same trap.
  - Rerunning with `use_cpu=True` gave bit-exactness.
- Data order was verified separately: the batch fingerprints for all 32 resumed micro-batches equal the
  uninterrupted run's batches 16–47.
- Cosmetic: the Trainer's summary `train_loss` after a resume averages only the resumed steps (11.9308 vs 11.9330).
  Compute it from `loss.jsonl` instead.
- Not tested here:
  - A kill *during* a save. `latest_checkpoint` requires `trainer_state.json`, which is written last; argued from
    code only.
  - The ProbeSignal callback. It runs in eval mode; `probes.score` has no sampling, so it is not expected to touch
    the RNG.
  - GPU. CUDA SDPA/FA backward can be non-deterministic run to run (15 F2). On GPU the acceptance rule is therefore
    tier 1 with a control: A, A2 (repeat) and B, and require max abs diff(A, B) ≤ max abs diff(A, A2). Cost ≈6 min
    on a 48 GB card, ≈$0.05–0.10.

## F4. Checkpoint interval vs overhead (estimate)
The billed-time overhead per useful hour ≈ C/τ + λ(τ/2 + R), where:
- C = blocking save cost ≈15 s;
- τ = interval;
- R = restart cost: ≈3 min for a same-host resume on Vast (disk intact), ≈8 min for a new host (create, venv or
  prebuilt image, 28 GB model download, 3.1 GB checkpoint pull from HF).

The optimum is τ* = √(2C/λ): 24 / 11 / 5.5 min at λ = 0.1 / 0.5 / 1 per hour. **Recommendation: a fixed
`--save-every 100`** (≈12 min on an A40 at 7.4 s/step, ≈4 min on an H100). That is ≈7 saves per arm, ≈2% overhead
with no preemptions. Total overhead at R = 8 min:

| λ (per h) | 0.1 | 0.5 | 1 |
|---|---|---|---|
| overhead, τ = 12 min | 4% | 14% | 25% |

If evals are not checkpointed, add ≈λ·E²/2. For E ≈ 30 min of vLLM that is ≤3%. Evals should write results per
batch to `/workspace` anyway.

## F5. Orchestration (design)
1. **Pod side.** Train with `--resumable --save-every 100 --resume auto`. Every `on_save` launches a **background**
   uploader (separate process, `nice`):
   - it uploads `checkpoint-N` to `<adapters repo>/ckpt/<run_name>/latest/` and only then writes a `ckpt.json`
     `{step, sha256 of the 4 files}`;
   - it deletes the previous checkpoint from the Hub;
   - after `final/` uploads, it deletes `ckpt/<run_name>/`.

   Put the venv and model under `/workspace` (they survive a RunPod spot stop or a Vast pause), not the container
   disk.
2. **Laptop guard** (extends guard_pod.sh; polls every 60 s). If the instance is paused or exited without
   `RUN_EXIT`:
   - (a) wait up to W = 15 min for a same-host resume. Vast resumes automatically; on RunPod call `podBidResume`.
     When it is back, rerun the same command: `--resume auto` finds the local checkpoint.
   - (b) otherwise **destroy the old instance first**. This is fencing: otherwise it resumes later and two trainers
     write the same Hub run.
   - Then rent the next-cheapest offer with the **same GPU type** (bit-exactness and comparability hold only on the
     same SKU and stack), pull `ckpt/<run_name>/latest` after verifying `ckpt.json` hashes, and launch with
     `--resume <path>`.
   - Log `{gpu, host, from_step}` into `run_info.resumes`.
3. **Pod-side deadline** (fleet 1, 03) still applies. Its "stop, don't delete" becomes free to recover from,
   because a resume costs ≤1 checkpoint interval.
4. **Network volume vs HF.** A RunPod network volume ($0.07/GB-month) is faster, but it is pinned to one data
   center, where spot stock may be absent. The Hub works across clouds. Use HF.

## $ effect per arm (estimate)
Billed hours = 08's full-arm minutes (setup + gen + train + eval) × (1 + overhead) × (bid Q1 + $0.03/h storage).
Overhead is 14% (5–25%).

| option | $/arm |
|---|---|
| Vast interruptible **A100 80 GB SXM** (91 min) | **≈$0.56** (0.51–0.64) |
| Vast interruptible **A40** (134 min) | **≈$0.62** (0.57–0.71) |
| Vast interruptible RTX A6000 / RTX 6000 Ada / L40S | ≈$0.66 / 0.68 / 0.71 |
| Vast interruptible **H100 SXM** (46 min) | **≈$0.91** (0.83–1.03); fastest wall time |
| RunPod Community A40 (no stock) | 0.78 |
| RunPod Secure A40 (stock High now) | 1.32 |
| RunPod Secure H100 | 3.06 |
| Vast on-demand A40 | ≈1.26 |

- Per experiment (≈8 arms), against today's realistic RunPod option (Secure A40): saves ≈**$5–6**. Against Secure
  H100: ≈$17–20.
- Vast-side caveats: host quality varies (fleet 1 saw 9-min installs on slow hosts); network egress is ≈$0.003/GB,
  i.e. ≈$0.1 for the 28 GB model; the prices are a single snapshot.

## Risks
- **Comparability.**
  - A same-SKU resume with full state is class A (bit-identical on CPU; GPU unverified).
  - Resuming on a different GPU type is class B (≈ a dropout-seed change). Forbid it for comparison arms, or
    record it and treat it as a seed draw.
  - The as-written resume is class C. Never use it for comparison arms.
- **Numerics.** On GPU, compile/Inductor or any extra RNG stream is unsaved, so run tier 1 before trusting it.
- **Operations.**
  - Porting launch/guard/sync to Vast is a real cost: CLI, ssh, and the MFS gotchas don't carry over.
  - The guard's fencing rule is essential: a double trainer would corrupt the Hub run.
  - HF private storage holds ≈3–6 GB transiently per running arm.
- **Price.** Bids at `min_bid` are the most likely to be preempted. The preemption rate is unknown, so the
  cost range is speculative until a 5-arm log exists.

## Recommendation
1. **Adopt the resume patch now**, whether or not we use spot. It is $0, it turns any host failure, Community-host
   death or pod-deadline stop into a ≤12-min loss, and it fixes the latent class-C resume fleet 1 found. Gate: one
   GPU tier-1 check, A vs A2 vs B on the arm's card, 20 steps, kill at 10, about $0.10. Then make HubPush
   asynchronous.
2. **Spot: Vast interruptible, not RunPod.** RunPod spot is not visible to the API tonight. Pilot the next
   pre-registered multi-seed batch on Vast **A100 80 GB interruptible** (cheapest per arm; 80 GB needs no
   Liger/ckpt-budget memory tricks) or A40, all arms on one SKU. Bid ≈Q1 + 25%, and log every pause to estimate λ.
3. If porting to Vast isn't worth the effort for one batch: **RunPod Secure A40 at $0.59 (stock High now)** is the
   no-engineering fallback at ≈$1.32/arm.

Files:
- `docs/efficiency3/patches/11_train_sft_resume.diff` (production patch)
- `11_resume_test_hooks.diff` (test-only env hooks, applied on top)
- `11_resume_test.sh`, `11_resume_compare.py`

Sources: RunPod GraphQL `gpuTypes{…SpotPrice, lowestPrice{minimumBidPrice}}`; RunPod docs concept "spot-pod"
(graphql-spec.runpod.io `podRentInterruptable`, docs.runpod.io/pods/maintenance-and-outages); Vast docs
(docs.vast.ai/guides/instances/choosing/instance-types); Vast public `api/v0/bundles` search; RunPod blog
"Spot vs. On-Demand Instances".
