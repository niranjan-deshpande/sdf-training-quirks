# 06 — Infrastructure: hands-off pods, fast setup, less fixed overhead

Date: 2026-10-24. Files I own: `scripts/pod_setup.sh` (rewritten), plus new `scripts/pod_lib.sh`,
`scripts/launch_pod.sh`, `scripts/guard_pod.sh` and `scripts/pod_sync.sh`. All of them pass `bash -n` under
macOS `/bin/bash` 3.2. `launch_pod.sh --dry-run` and the argument checks ran locally, and the rendered
`job.sh` was linted. **No pod was created, stopped or deleted to test the launch and guard scripts.** Their
first real use is the next H100 run. The setup pieces (the `hf download` command, the uv install) were
timed on the shared A40 fleet (§6).

## TL;DR

| Fixed cost | Before | After | Evidence |
|---|---|---|---|
| Pod setup: venv + 14B model | 15–20 min | **~1–3 min** | A40: cold uv install 12.7 s; 29.6 GB model download in 15.0 s (1.97 GB/s); they run in parallel |
| Python import per process (torch+transformers+peft+vllm) | **59–80 s** (venv on MFS `/workspace`) | **~7 s** (venv on `/root`) | A40, same pod |
| vLLM engine init, first engine on a pod | 357 s (A40, cold caches) | 96 s with carried-over caches | `--save-cache` + automatic push |
| Hand work: create, ssh, rsync, env.sh, tmux | ~10 min of attention | 1 command | `launch_pod.sh` |
| Idle GPU after the job ends | 0–60+ min | ≤1 poll (60 s) + copy | `guard_pod.sh` |

At Secure H100 rates ($3.99/hr): setup saves ~15 min ≈ **$1.0 per pod**. Imports save ~1 min per GPU
process, and run_prof starts 6 such processes, so ≈ **$0.4 per experiment**. The guard saves whatever idle
tail would otherwise accrue. The vLLM cache saves 1–4 min per pod. **Total ≈ $1.5–2 per experiment, plus
less babysitting.**

The import gap is the surprise. Every `python -m distill.X` on the old layout spent about a minute
importing from the network-mounted venv. That cost is spread through every run, not just setup, and moving
the venv to `/root` removes it.

## 1. `pod_setup.sh` (rewritten)

- **The venv goes to `/root/venv` and the model to `/root/models/<name>`**, both on the container disk
  (overlay on local NVMe). `/workspace` is an MFS network mount. `/workspace` now holds only the repo,
  outputs, logs and `env.sh`, so the volume can be 50 GB. The A40 fleet's 40 GB volume overflowed on the old
  layout (§6).
- **The model downloads straight to `/root/models/Qwen3-14B`** (no HF-cache copy step) with
  `hf download --local-dir --max-workers 16`, `HF_XET_HIGH_PERFORMANCE=1` and
  `HF_XET_CHUNK_CACHE_SIZE_BYTES=0` (no second copy in the xet cache). `HF_HUB_ENABLE_HF_TRANSFER` and
  `hf_transfer` are dropped: huggingface_hub 1.x/2.x ignores them.
- **Download and install run in parallel.** The download starts first in the background, from a throwaway
  `uvx` tool env, so the venv install can't replace `huggingface_hub` under it. Meanwhile `uv pip install
  -e .[gpu]` runs with its cache on `/root`, so files are hardlinked rather than copied. Next comes an
  import check, then a wait for the download, then a check that every shard in
  `model.safetensors.index.json` exists (with its size).
- By default only `Qwen/Qwen3-14B` is downloaded. The old default also pulled `Qwen3-14B-Base`, another
  28 GB. Override with `MODELS=...`.
- **Compat:** `/workspace/venv` is a symlink to `/root/venv`, so `run_*.sh` and `bench_*.sh` work unchanged
  (`source /workspace/venv/bin/activate`, `M=/root/models/Qwen3-14B`). If `/workspace/venv` is already a
  real directory, the script prints a warning and leaves it alone.
- `env.sh` gains these lines if they are missing: `HF_HOME=/root/hf`, `HF_XET_HIGH_PERFORMANCE=1`,
  `HF_XET_CHUNK_CACHE_SIZE_BYTES=0`, `VLLM_CACHE_ROOT=/root/.cache/vllm`.
- `uv pip freeze` goes to `/workspace/pip_freeze.txt` (the guard copies it back). If
  `scripts/pod_constraints.txt` exists, it is passed as `-c`. Deps are unpinned (`vllm>=0.11`), so a new pod
  can silently get a newer vLLM or transformers: pin from the freeze of a known-good run.
- The script logs phase timestamps (`[setup +Ns]`). It ends with `SETUP_OK`, or with `SETUP_FAILED` and a
  non-zero exit. The old A40 setup chain printed `SETUP_DONE` after a failed download (§6).
- It is idempotent: each model gets a `.complete` marker, and an existing venv is reused. The container
  disk is wiped on pod stop, so after a restart just rerun it (~1–3 min).
- **Use a container disk of at least 80 GB.** Venv 8.2 GB + uv cache 8.1 GB + model 29.6 GB + caches.

**Expected setup on an H100 pod: ~1–3 min.** About 20 s of uv install, plus a ~15–60 s download depending on
the data center, all overlapped, then a ~17 s first import. Add ~30 s for apt if the image lacks
rsync/tmux. The old 15–20 min came from writing the venv and model to MFS, the ignored hf_transfer flag, a
second 28 GB model, and the copy step.

## 2. `launch_pod.sh`: one command per experiment

```bash
scripts/launch_pod.sh --name prof --run "bash scripts/run_prof.sh train" \
    --guard-pull outputs/distill/prof --max-hours 3 --delete --save-cache
scripts/launch_pod.sh --dry-run ...    # plan + live H100 stock and prices; creates nothing
```
1. Tries `runpodctl pod create ... --cloud-type COMMUNITY --public-ip --wait` **once**. If that fails, it
   goes straight to Secure (user decision: no waiting loops). If a `--wait` timeout leaves a half-created
   pod, it deletes every pod carrying this launch's unique name (`<tag>-MMDD-HHMMSS`).
2. Prints `POD <id> cloud=... cost=$X/hr`. Saves `outputs/pods/<id>.env` and the `current.env` symlink
   (host, port, cost, created). `outputs/` is gitignored and never synced up.
3. Waits for an SSH handshake. Uses a per-pod known_hosts file in `~/.runpod/known_hosts.d/`, because RunPod
   reuses ip:port across pods and a shared known_hosts would then fail. Installs rsync/tmux only if missing.
4. **HF_TOKEN only:** sent over ssh **stdin** into `/workspace/env.sh` under umask 077. It never appears in
   argv (`ps`), output or logs. The old docs recipe put it on the ssh command line.
5. rsyncs the repo with the old excludes (`.env*`, `.venv`, `.git`, `outputs`, `site`, `data/generated`,
   `data/train`, `data/distill/{v2,chatty}`, `responses.jsonl`, logs). That is ~80 MB. Add more with
   `SYNC_EXCLUDE="..."`. Pushes `outputs/cache/{vllm,triton,flashinfer}` if an earlier `--save-cache` left
   them.
6. Starts tmux session `main` running `/workspace/job.sh`. `pod_setup.sh` logs to `/workspace/setup.log`;
   then, if setup succeeded, the `--run` command logs to `/workspace/run.log`, ending with `RUN_EXIT=<code>`.
   If setup fails, the log gets `RUN_EXIT=setup_failed_N` instead.
7. With `--max-hours`, starts `guard_pod.sh` with nohup under `caffeinate -is`, logging to
   `outputs/pods/<id>.guard.log`.

Other flags: `--pod ID` (reuse a running pod), `--secure-only`, `--gpu`, `--models`, `--container-gb` (80),
`--volume-gb` (50).

## 3. `guard_pod.sh`: copy, then stop or delete

This generalizes the chatty-run guard that already worked (TRAIN_EXIT, then rsync, then `pod stop`):
```bash
scripts/guard_pod.sh --max-hours 3 --pull outputs/distill/prof [--pull ...] [--delete] [--save-cache]
scripts/guard_pod.sh --pod <id> --deadline-utc "2026-10-25 09:00" --log /workspace/train.log \
    --marker TRAIN_EXIT --pull outputs/distill/chatty
```
- Waits for a line starting with the marker (default `RUN_EXIT` in `/workspace/run.log`) **or a mandatory
  hard deadline**. Every ~5 polls it checks the pod is still RUNNING (and exits if someone else stopped it)
  and refreshes the ssh port.
- Copies each `--pull` path, with 4 tries, backoff, and an ssh-info refresh between tries. It always copies
  `run.log`, `setup.log`, `download.log` and `pip_freeze.txt` to `outputs/pods/<id>/`. With
  `--save-cache`, it also copies the vLLM/Triton/flashinfer caches to `outputs/cache/`.
- **Copies come before the stop. `--delete` happens only if every copy succeeded**; otherwise the guard
  stops the pod, which keeps the volume. It retries stop/delete 3× and logs the final status. On macOS it
  posts a notification. The exit code is non-zero if any copy failed.
- Limitation: the guard runs on the laptop. `caffeinate -is` blocks idle sleep (and system sleep on AC),
  **not lid-close sleep**. A pod-side last resort (`runpodctl stop pod $RUNPOD_POD_ID` with the pod-scoped
  key) is possible but not implemented: an earlier session's sandbox blocked it, and it stops the pod
  without copying results (adapters still reach HF via the run scripts). That needs the user's call.

`pod_sync.sh` has the manual verbs: `up`, `down PATH...`, `ssh [CMD]`, `log [FILE] [N]`, `status` (cost,
tmux, GPU, SETUP/RUN markers), and `stop|delete` (which copy nothing).

## 4. Prebuilt Docker image: not worth it (no Dockerfile written)

- A cold `uv` install of the full GPU stack (208 packages, 8.1 GB of wheels) onto local disk took
  **12.7 s** (10.0 s download, 1.5 s install). An image cannot beat that by much.
- Each experiment lands on a fresh, random host. Hosts usually cache the stock `runpod/pytorch` image, but
  a custom ~15–20 GB GHCR image would be a cold pull every time. Pulling plus single-threaded layer
  decompression of an 8 GB layer is minutes, not seconds, and it also delays SSH and the model download.
- It would also mean GitHub-runner disk cleanup (~14 GB free), registry auth or a public image, and
  rebuilds on every dep change.
- The one real benefit, pinned versions, is available through `scripts/pod_constraints.txt`.

## 5. Other fixed overhead

**Per-process startup (all measured on A40, model in page cache).** A typical experiment starts 5–8 GPU
processes: run_prof = 1 train + 2 checks + 2 probes + 1 sample. Each one pays:
- Python import: **59–80 s from the `/workspace` venv vs 6.6–6.8 s from `/root`** (17 s for the very first
  import, which compiles `.pyc` files). The rewrite fixes this. For the old layout this was the largest
  per-process cost.
- HF `from_pretrained` of 28 GB from page cache: 8–11 s. `cpu_then_cuda` (the `probes.py` pattern) took
  8.2 s, `device_map="cuda"` 9.9 s, and `HF_ENABLE_PARALLEL_LOADING` 10.7 s. Transformers 5.19 loads fast
  either way, so **no `distill/` change is needed** and I didn't add the env var.
- vLLM: weight load 4.7–5.0 s ("Model loading" 15 s warm, 30 s cold). **`load_format="fastsafetensors"`
  gave no gain** (4.96 s vs 4.72 s), so it is not adopted.
- The vLLM engine init cost is warmup, not loading. Cold: 243 s inside the engine (63.9 s torch.compile,
  two CUDA-graph captures of 27–28 s each, and a ~90 s silent JIT/autotune gap before the second, LoRA
  capture). With compile, Triton and flashinfer caches warm: 67 s, of which graph capture is ~46 s.

**Implications.**
1. The cache only helps the *first* engine on each pod; later processes on the same pod already hit it.
   Carrying the caches across pods (`guard --save-cache` → `launch` push; 66 MB vLLM + 5 MB flashinfer +
   Triton) saves ~4 min per pod on an A40, likely ~2 min on an H100. A version mismatch just recompiles.
2. The largest remaining per-engine cost is ~46 s of CUDA-graph capture plus ~15 s of model load. **Merging
   vLLM processes saves ~1–1.5 min per merged start.** run_prof has 3 vLLM starts (checks, checks_refs,
   sample) that could be 1 engine; HANDOFF §0.6 already proposes "one engine for checks + sample". That
   code is in `distill/`, so it is flagged to the vLLM/pipeline owner rather than changed here.
3. One pod per experiment is fine now that setup is ~1–3 min. Bundling arms (HANDOFF §0.3) still saves one
   setup and one cold compile per pod avoided.

## 6. Measurements (A40 fleet, EU-SE; raw files in `outputs/distill/bench/infra/`, jobs in `jobs/`)

| Test | Result | File |
|---|---|---|
| `hf download Qwen/Qwen3-14B` → `/root`, HF_XET_HIGH_PERFORMANCE=1, 16 workers (hub 2.3.0, hf_xet 1.7.0) | **15.0 s, 29.55 GB, 1.97 GB/s** | `dl_d9c01ce0b316.txt` |
| Cold uv install `.[gpu]` → `/root` venv, empty cache on `/root` (uv 0.13.5) | **12.7 s** (resolve 0.8 s, download 10.0 s, install 1.5 s); venv 8.2 GB, cache 8.1 GB | `uv_64b37df4f34a.txt` |
| Import torch+transformers+peft+vllm | `/root`: 17.0 s first, 6.6 / 6.8 s after; `/workspace`: **79.9 / 59.1 s** | `import_64b37df4f34a.txt` |
| Storage (two shards, 7.8 GB) | `/root` 7.4 GB/s buffered; `/workspace` MFS write 1.09 GB/s, buffered read 0.84 GB/s | `load_333b89002f0c.txt` |
| HF `from_pretrained` (page cache) | cpu→cuda 8.2 s; device_map=cuda 9.9 s; +parallel loading 10.7 s | `load_333b89002f0c.txt` |
| vLLM 0.32 `LLM(..., enable_lora=True)` init, 14B, A40 | cold caches 356 s (engine 243 s); warm 96 s (engine 67 s); fastsafetensors 92 s (engine 65 s) | `vllm_d9c01ce0b316.txt`, `vllm_*.log` |

The vLLM numbers include a 55–70 s import because the bench used the old `/workspace` venv; engine-only
times are in parentheses. `load_d9c01ce0b316.*` is a partial run that was killed during a runner restart;
ignore it.

Caveats: this is a single A40 host in one data center. MFS speed varies by data center: the earlier H100
setup in EU-FR took 4.4 min and 34 s for two 14B downloads into `/workspace`. H100 numbers will differ, but the
ratios (import, cache) should carry over.

**Incident found while measuring.** The fleet's old-style setup failed on 54n2 with `Disk quota exceeded`,
because the 40 GB volume could not hold the venv and the 28 GB of shards. It printed `SETUP_DONE` anyway,
and queued jobs then failed with no model. The coordinator was told, and `jobs/infra_dl.sh` (the new
download path) repaired the pods in ~20 s each. The new `pod_setup.sh` avoids both problems: model on
`/root`, and non-zero exit plus `SETUP_FAILED` on download failure.

## 7. Recommendations

1. For every run: `launch_pod.sh ... --max-hours <plan + 1> --guard-pull <outputs dir> --delete
   --save-cache`, then read `outputs/pods/<id>.guard.log`. **The first real use should be watched**: these
   launch/guard paths have not run against RunPod yet.
2. After the next good run, `cp outputs/pods/<id>/pip_freeze.txt scripts/pod_constraints.txt` to pin
   versions.
3. Update docs/runpod.md and HANDOFF §5 (not my files): container disk 80 GB, volume 50 GB, setup ~1–3 min,
   no copy step, use `launch_pod.sh` / `guard_pod.sh`.
4. `distill/` owners: merge the vLLM processes (checks + checks_refs + sample in one engine). Leave the HF
   loaders and `load_format` as they are.
5. No Docker image (§4).
