# 09 — Paid-but-idle minutes per pod: setup, images, caches, result copy

Agent 09 (research + drafting, no GPU), 2026-11-05. Evidence labels: **measured** = from this repo's pod logs
(`outputs/pods/*`), **prior-measured** = docs/efficiency/06_infra.md (A40), **estimate**, **speculative**.
Patch: `docs/efficiency2/patches/09_Dockerfile` (draft, never built). No script in the repo was edited.

## TL;DR

- **The real pod logs show idle time is dominated by tails and outliers, not by the steady-state setup.**
  Good pods (3 of 4 since the 06_infra rewrite) idle ~5 min each (create→work 2.3 min, RUN_EXIT→deleted
  1.3–2.5 min). The Qr pod idled **~44 min of 3 h 13 min (≈ $2.9 of $12.8)**: 8.9 min rsync before setup,
  8.75 min setup (venv install 513 s), ~10 min loading 13 adapters from the network volume, 16.5 min copying
  checkpoints. All four are fixable with script changes, no recipe change, no comparability risk.
- **Steady-state setup on H100 is ~2.2 min, not the 06_infra estimate of "~20 s uv"**: the venv step took
  109–130 s on all three good pods (measured; it runs concurrently with the 30 GB model download, which
  always finished first). That reopens 06_infra's "no Docker image" verdict: the venv is now the critical
  path. A baked venv (Dockerfile drafted) or a venv tarball on HF Hub should cut setup to ~0.7 min (estimate)
  and remove the 513 s PyPI outlier. Gain is small per pod (~$0.1 on H100 Secure) but it multiplies by pod
  count, which matters if cheap GPUs mean many pods.
- **Ranked by $ saved per experiment:** (1) never copy `checkpoints/` (≤14.5 min/5-arm pod, partly done in
  run_opposite.sh only), (2) adapters and HF refs on `/root`, not `/workspace` (~10 min on probe-heavy pods),
  (3) tar-stream the initial sync (8.5 min outlier), (4) baked venv (1.5 min typical, 7 min outlier),
  (5) carry the vLLM compile cache per GPU arch (~2 min per pod that starts a cold engine; more on slow GPUs),
  (6) guard poll 60→15 s (~0.4 min). Typical pod: ~5–6.5 → ~2.5 min idle; bad pod: ~44 → ~4 min.

## What's new vs docs/efficiency (06_infra, 04_pipeline)

- 06_infra measured setup pieces in isolation on an A40 and *predicted* 1–3 min. I audited the **actual H100
  pod timelines** since then (4 pods: park 3voupfhd8g1k5q, Qr t4586wgfxqdacs, opposite tl59pqjfi8pjpn,
  opposite-s7 60qnayv8bdsu5b): setup 124–146 s typical, 525 s outlier; the venv, not the model, is the
  critical path. 06_infra's Docker rejection rested on the 12.7 s uv number, which did not hold on H100 pods.
- New failure modes not in 06/04: 8.5 min initial rsync (small files onto MFS), 11 min of PeftModel
  `load_adapter` from `/workspace`, 16 min checkpoint copy. Only the last one is noted in HANDOFF; it is
  fixed ad hoc in `run_opposite.sh` (line 50, `rm -rf $O/*/checkpoint*`) but not in `run_qrich.sh` or
  generically.
- $/pod across GPU types, and the observation that fixed overhead scales with **pod count**, not GPU price.
- Results-via-HF + pod self-stop as a way to delete the copy tail and the laptop dependency (needs user call).

## Measured pod timelines (H100 Secure $3.99/hr; UTC; from outputs/pods/*)

| Pod | create→tmux | setup (venv / total) | RUN_EXIT→guard trigger | trigger→deleted | idle total | pod total |
|---|---|---|---|---|---|---|
| opposite tl59 | 13 s (sync 9 s) | 130 / 146 s | 57 s | 1.5 min (copy 77 s) | ~5.1 min | 70 min |
| opposite-s7 60qn | 17 s (sync 6 s) | 109 / 124 s | 10 s | 2.3 min (copy 2.0 min) | ~4.8 min | 60 min |
| park 3vou | (not logged) | 127 / 139 s | — | — | — | — |
| **Qr t458** | **8.9 min** (rsync ~85 MB of small files) | **513 / 525 s** | 17 s | **16.3 min** (14.5 GB, checkpoints) | **~44 min incl. ~10 min adapter load** | 193 min |

Other measured per-pod costs: vLLM `engine ready in 95–98 s` (H100, Qr checks, warm cache within the pod);
cold first engine 356 s vs 96 s warm on A40 (prior-measured; engine-only 243 vs 67 s, rest is the old MFS
import); probes for 2 fresh adapters + none: 74–81 s total (opposite); probes for 13 adapters, 11 of them
HF-snapshotted to `/workspace` 2+ hours earlier: 17 min, of which ~11 min adapter loading (HANDOFF).
Pod create call: 18–36 s (launch logs; billing during it: unknown, RunPod docs only say per-second while
running, [billing](https://docs.runpod.io/accounts-billing/billing)).

## Findings

### F1. Never copy `checkpoints/` (adopt; generalize the run_opposite fix)
- Mechanism: `train_sft.py` writes `output_dir=<out>/checkpoints` (2 mid/final checkpoints, ~2.4 GB per arm
  incl. optimizer state) and uploads them to HF (`uploaded pilot2/X/checkpoint-366` in every run log). The
  guard then rsyncs them to the laptop over one ssh stream (~15 MB/s measured: 14.5 GB in 16 min).
- Gain (measured): Qr tail 16.3 min vs 1.5–2.3 min on pods that deleted them → **~14 min ≈ $0.93 per 5-arm
  H100 pod**, ~2.9 min/arm.
- Change: in `train_sft.py`, after the HubUploader confirms the final upload, `shutil.rmtree(out/"checkpoints")`
  unless `--keep-checkpoints`; or in `launch_pod.sh`'s job.sh, `find outputs -path '*/checkpoints' -prune -exec
  rm -rf {} +` right before writing RUN_EXIT; plus guard pulls with `--exclude 'checkpoint*/'` as a belt-and-
  braces default. Risk: losing resumability on the pod (irrelevant after delete) — only if the HF upload
  silently failed; so delete only after the uploader's success.

### F2. Adapters and HF reference snapshots on `/root`, not `/workspace` (adopt after one timing)
- Mechanism (estimate): `/workspace` is an MFS network mount (0.84 GB/s *sequential* buffered read,
  prior-measured). PeftModel's safetensors loading mmaps; cold mmap page-faults over MFS are far slower than a
  sequential read. Each r64 all-proj adapter for Qwen3-14B is 257 M params ≈ 1 GB in fp32. 13 adapters in
  ~11 min ≈ 20 MB/s effective, vs ~16 s for a sequential copy at 0.84 GB/s.
- Gain: ~10 min on a probe pass over ≥10 adapters (≈ $0.65 H100 Secure); ~0 on 2-arm pods (their freshly
  written adapters sit in page cache, probes took 74 s).
- Change: `H=/root/hub` for `snapshot_download` of reference adapters in run_*.sh; train `--out
  /root/adapters/X` and `rsync -a --exclude checkpoints /root/adapters/ $O/` at the end (or simply `cp -r` refs
  to /root before probes). Validate: time `probes --adapters` with 4 refs from `/workspace` vs `/root` on the
  next pod (~2 min of GPU).

### F3. Initial sync: stream a tarball, or rsync to `/root` (adopt)
- Mechanism: rsync of ~85 MB of many small jsonl/py files into MFS took 8.5 min on Qr (per-file latency);
  6–9 s on the opposite pods. Billed: the pod exists from create.
- Change: `tar -C "$REPO_DIR" --exclude-from=... -czf - . | ssh ... 'mkdir -p $R && tar -xzf - -C $R'` for the
  first push (rsync stays for updates); keep the 06_infra excludes and add HANDOFF's
  `SYNC_EXCLUDE="data/distill/multiplex data/distill/prof"` default. Or sync the repo to `/root/repo` and keep
  only outputs on `/workspace` (bigger change: run_*.sh `cd /workspace/sdf-training-quirks`).
- Gain: ≤8.5 min on bad hosts (estimate: a single stream should take seconds).

### F4. Baked venv: Docker image or venv tarball on HF (test next)
- Mechanism: the venv (208 packages, ~8 GB) is the setup critical path on H100 pods: 109–130 s typical, 513 s
  on one host (measured), while the 30 GB model finished earlier in all four pods. A pre-installed venv leaves
  model download (15–60 s) + import (~11 s first import, prior-measured 17 s with .pyc compile, 0 if bytecode
  is baked).
- Option A, **Docker image** (patch `09_Dockerfile`): `FROM runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`
  (the tag pods use now, hence usually host-cached) + one venv layer (~4–5 GB compressed, estimate) + rsync,
  tmux, zstd. Unknown: RunPod cold pull time for that layer; it happens *before* SSH is up and may be billed.
  Industry reports for multi-GB layers are 1–3 min (speculative for RunPod). Net: −1.5 min typical if the
  pull is ≤1 min, ~0 if it is 2 min; but it removes PyPI variance (the 513 s outlier) and pins versions.
  Cost: a registry (GHCR public, no secrets inside), an x86 build machine with ~40 GB disk, rebuild on dep change.
- Option B, **venv tarball on HF Hub** (my preferred first test): after a good pod, `tar -C /root -I 'zstd -T0
  -3' -cf venv-<vllm>-<torch>.tar.zst venv` and upload to a dataset repo (public; contains no secrets — check
  that no `.env` or token file is inside). pod_setup.sh: if the tarball exists for the pinned versions,
  `hf download` it (xet, 1.97 GB/s prior-measured → ~3 s for ~4–5 GB) and extract (~15–30 s with zstd -T0,
  estimate), in parallel with the model; else fall back to uv. Same path `/root/venv` (venv shebangs are
  absolute). No registry, no build machine, works on any base image the pod has cached. Gain ≈ Option A without
  the pull risk.
- Also bake/ship the per-repo `uv pip install -e . --no-deps` only (1–2 s).
- Validate: one pod, `time` each setup phase with and without (≤5 min GPU). Adopt if setup ≤1 min on 2 hosts.
- **Reject: model in the image** (+28 GB cold pull per random host; HF xet is already fast). **Reject (for
  now): network-volume model/venv cache** — it pins every pod to one data center, which fights "cheapest GPU
  anywhere" and Community stock; RunPod's managed "cached models" feature is documented for Serverless only
  ([model caching](https://docs.runpod.io/serverless/endpoints/model-caching); Pods are told to bake into a
  template, [custom template](https://docs.runpod.io/pods/templates/create-custom-template)). Reconsider only if
  a multi-pod fleet will live in one DC with stock.

### F5. vLLM compile cache per GPU arch, plus fewer CUDA-graph sizes (test next)
- Mechanism: cold engine init = torch.compile (~64 s) + Triton/flashinfer JIT/autotune (~90 s) + two CUDA-graph
  captures (base + LoRA, ~27 s each) on A40 (prior-measured). The first three are cacheable and reusable across
  machines with the same environment ([vLLM torch.compile blog](https://vllm.ai/blog/torch-compile));
  graph capture is not. `guard --save-cache` + launch push already exists (pod_lib.sh `CACHE_DIRS`) but is
  keyed only by path: a cache from an sm_90 H100 is useless on an sm_89 L40S / sm_86 A6000 / sm_120 RTX PRO
  6000 or 5090 and just recompiles (safe, no gain).
- Change: store as `outputs/cache/<vllm ver>-<sm>/...` (or a tarball next to the venv on HF, so it does not
  depend on the laptop) and push the matching one. Gain: ~3 min per pod on A40 (engine 243→67 s), likely
  1.5–2.5 min on H100 (estimate), larger on slower GPUs, only on pods that start a vLLM engine (teacher or
  checks; `retrain` pods with probes only don't).
- Additionally (speculative, recipe-neutral): eval engines that generate a few hundred short replies could
  set `compilation_config={"cudagraph_capture_sizes": [1,2,4,8,16,32,64,128,256]}` (vLLM default captures ~67
  sizes up to 512) to cut capture ~2–3× (~30 s/engine), or `enforce_eager=True` for tiny check runs.
  Numerics: same math, batching nondeterminism only; but keep engine args identical across arms of one
  comparison (put them in `distill/vllm_engine.py` presets).

### F6. Guard tail (adopt)
- Poll 60 s → 15 s: RUN_EXIT→trigger was 10–57 s (measured). Saves ~0.4 min/pod; cost: 4× more
  `ssh grep` calls (negligible).
- Copy only what is needed: `final/`, `*.json(l)`, `*.md`, logs, probe outputs (F1 covers the big part).
- Speculative / needs user decision: **results via HF + pod self-delete**. The pod already uploads adapters
  and run_info to HF; it could also upload `outputs/distill/<exp>` (small) to the dataset repo, verify by
  listing the remote files' sizes, then `runpodctl remove pod $RUNPOD_POD_ID` itself. Removes the copy tail and
  the poll lag, and fixes the "laptop asleep → pod bills forever" gap. HANDOFF §5 lists the preconditions
  (verify runpodctl and the pod-scoped key are visible to the tmux job without printing it; stop rather than
  delete as the deadline fallback). Gain ~1.5–2.5 min/pod typical, plus the tail risk.

## $ per pod and per experiment across GPU types

Idle minutes are mostly network/disk bound, so they are roughly **independent of GPU type**; cold vLLM compile
is the exception (slower on weak GPUs: A40 356 s vs H100 ~?). Prices are approximate RunPod on-demand list
prices I did not re-check today (verify with `launch_pod.sh --dry-run`).

| GPU (≈$/hr) | typical pod: 5–6.5 → ~2.5 min idle | bad pod (Qr-like): ~44 → ~4 min idle |
|---|---|---|
| H100 Secure ($3.99) | save ~$0.20–0.27 | save ~$2.65 |
| H100 Community ($2.69) | ~$0.13–0.18 | ~$1.80 |
| A100 80 GB (~$1.2–1.6) | ~$0.06–0.11 | ~$0.80–1.05 |
| RTX PRO 6000 96 GB (~$1.8, est.) | ~$0.08–0.12 | ~$1.20 |
| L40S 48 GB (~$0.86) | ~$0.04–0.06 | ~$0.57 |
| A6000 48 GB (~$0.49) | ~$0.02–0.03 | ~$0.33 |

Per experiment it depends on the pod plan:
- One H100 pod, 2 arms (opposite-like, ~$4): typical idle ~5 min ≈ 8 % of spend; fixes save ~$0.2 (5 %).
- One H100 pod, 5 arms + 13-adapter probes (Qr-like, $12.8): measured idle ~44 min ≈ 23 %; fixes F1–F3 save
  ~$2.6 (≈ 20 %), F4–F6 another ~$0.2.
- **Many cheap pods** (e.g. 8 arms on 8 L40S pods in parallel): fixed overhead is paid 8×. Current typical
  6.5 min × 8 = 52 GPU-min ≈ $0.75; after fixes ~2.5 × 8 = 20 min ≈ $0.29. Relative to ~8 × (train ≈ 60–75 min
  on L40S, estimate) ≈ $7–9 of training, idle is ~8 % → ~3 %. On cheap GPUs the work is longer, so idle is a
  *smaller* fraction per pod; what grows is the pod count. Rule: **bundle arms per pod unless parallelism buys
  something the user needs**; each extra pod costs ~2.5–6.5 idle GPU-min plus one cold compile and one
  model download.
- Also on cheap/slow GPUs, the cold vLLM compile (F5) is a bigger absolute block (A40: 3 min per cold engine,
  prior-measured), so carry the per-arch cache whenever pods are spread across GPU types.

## Concrete script changes (proposed; none applied)

1. `distill/train_sft.py`: after the final upload succeeds, delete `<out>/checkpoints` unless
   `--keep-checkpoints`. (F1)
2. `scripts/launch_pod.sh` job.sh: before `RUN_EXIT`, `find outputs -type d -name 'checkpoint*' -prune -exec
   rm -rf {} +` when `KEEP_CHECKPOINTS` is unset. `guard_pod.sh`/`pull_path`: default `--exclude
   'checkpoint*/' --exclude 'checkpoints/'`. (F1)
3. `scripts/run_*.sh` template: `H=/root/hub` for reference-adapter snapshots; `--out /root/adapters/$name`;
   final `rsync -a --exclude 'checkpoint*' /root/adapters/ $O/` before RUN_EXIT. (F2)
4. `scripts/pod_lib.sh sync_up`: first push via `tar | ssh tar -x` (rsync only when the remote repo exists);
   default excludes add `data/distill/multiplex data/distill/prof` and any `data/distill/*/prev_*`. (F3)
5. `scripts/pod_setup.sh`: fast path — if `/root/venv/IMAGE_FREEZE.txt` exists (image) or
   `VENV_TARBALL=<hf repo>/<file>` is set and matches `pod_constraints.txt`'s hash, skip `uv pip install` of
   deps and run only `uv pip install --no-deps -e .`; download the tarball in parallel with the model. Log both
   phase times. (F4)
6. `scripts/launch_pod.sh`: `--image` flag (default unchanged); `scripts/pod_constraints.txt` from
   `outputs/pods/60qnayv8bdsu5b/pip_freeze.txt` (vllm 0.32.0, torch 2.13.0, transformers 5.19.0, peft 0.21.2)
   — prerequisite for both F4 options and for comparability anyway. (F4)
7. `scripts/pod_lib.sh CACHE_DIRS` save/push keyed by `$(python -c 'import vllm;print(vllm.__version__)')-sm$(
   nvidia-smi --query-gpu=compute_cap --format=csv,noheader | tr -d .)`. (F5)
8. `scripts/guard_pod.sh`: default `POLL=15`. (F6)

## Experiments run

None on a GPU (not a GPU agent). Local analysis only: timelines extracted from `outputs/pods/*/setup.log`,
`run.log`, `*.guard.log`, `opposite*_launch.log` with grep (commands: `grep -E '\[setup|SETUP'`, `grep -a -E
'^== |train_runtime|engine ready'`). Prices and image-pull times are not measured.

## Recommendation

- **Adopt now** (no recipe change, no comparability risk): F1 (checkpoint delete + guard exclude), F2 (adapters
  on /root), F3 (tar first sync), F6 poll 15 s, and pin `pod_constraints.txt`.
- **Test next** on the next pod (≤5 GPU-min): F4 option B (venv tarball on HF) vs uv; if setup ≤1 min on two
  hosts, adopt; build the Docker image (option A) only if the tarball path is flaky. F5 per-arch cache.
- **Needs user decision**: F6 results-via-HF + pod self-delete.
- **Reject**: model baked into the image; network-volume caches (DC lock-in) for this one-pod-anywhere pattern.
