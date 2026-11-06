# 03: Infra fire tests: pod self-stop, setup-speed gate, venv tarball

Agent 03 (efficiency3), 2026-11-06, 00:00–01:00 EST. One pod `eff4-infra` (318gzbkr1uormp), Secure
RTX 2000 Ada, EU-RO-1, $0.24/h, image `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (same as
launch_pod.sh). A Community RTX A5000 ($0.16/h) had no stock. Spend: see the end.

## TL;DR
- **Measured: the pod-side deadline works end to end, and the env var alone is enough.** (a) A detached
  timer with `RUNPOD_API_KEY` taken from `/proc/1/environ` into the env, and `~/.runpod/config.toml` left
  with its empty apikey, ran `runpodctl stop pod`: rc 0, pod `EXITED` 1 s later, seen from the laptop.
  So the config write in `03_pod_deadline.sh` is unnecessary (harmless). (b) The unmodified script, armed
  `--in 240`, stopped the pod **1 s after its deadline** (05:32:52 → EXITED 05:32:53). (c) `--after-exit-min 1`
  stopped it 2 min after `RUN_EXIT` was written (fire ≤ M + 1 min poll), and re-arming left exactly 1 daemon.
- **Measured: on a good host, setup is fast and a prebuilt venv buys almost nothing.** Cold-cache
  `uv pip install -e .[gpu] liger-kernel` took **36.8 s** (210 packages, 8.9 GB, 71k files); a 1.50 GB HF
  file took **3.2 s** via xet. The venv tarball (zstd -3, 3.93 GB) compressed in 12 s and **extracted in
  21 s**, so restore ≈ 25–40 s with download (estimate): a tie with a cold install here. The tarball only
  pays on slow hosts (fleet 1: 109–130 s typical, 513 s outlier on H100 pods), i.e. ≈ $0.01–0.07 per pod.
- **New: a 7-second setup-speed gate** (`patches/03_setup_gate.sh`) that passes this host and **fails a
  simulated CPU-starved host** (cpu loop 53× slower, sympy install 25× slower, PyPI 20–25× slower). It is
  the cheaper fix for the "stalled Community host" failure: ≈ $0.001 per good pod, saves a stalled session
  on a bad one. Plus a crash lesson: laptop guards must run outside the Claude app (launchd) and fail loudly.

## What's new vs fleet 1
Fleet 1 (efficiency2/03) verified a hand-written 3-minute timer, with the key written to config.toml, and
left untested: a full fire of the script itself, env-only auth, `--after-exit-min`, GraphQL `podStop`.
Fleet 1 (02, 09) proposed a setup gate and a prebuilt venv but measured neither. Here: env-only auth
(measured), full script fire (measured), restart behaviour (measured), setup timings on a fresh Secure host
(measured), the gate written and validated against a simulated slow host (measured), venv tar/extract
timings and a restored-venv import check (measured).

## 1. Pod-side deadline

### 1a. Env var alone suffices (measured, n=1)
Script (commands in the Evidence section): import `RUNPOD_API_KEY` and
`RUNPOD_POD_ID` from `/proc/1/environ` into the environment, leave `~/.runpod/config.toml` as shipped
(apikey field empty: `apikey = ''`), `setsid nohup`, sleep 45 s, `runpodctl stop pod $RUNPOD_POD_ID`.
```
05:25:46 armed; key_len=50 pod_len=14 config_apikey_field_len=3      # = two quote chars + newline: empty
05:26:32 firing: runpodctl stop pod (env only)
pod "318gzbkr1uormp" stopped
05:26:32 runpodctl rc=0
```
Laptop poll (`runpodctl pod get`, every ~6 s): `RUNNING` until 00:26:32 EST, then
`"desiredStatus": "EXITED"`, `"lastStatusChange": "Exited by user: ... 05:26:32 GMT"`. The GraphQL
fallback did not run (rc was 0). So on-pod runpodctl v1.14.15 reads `RUNPOD_API_KEY` from the env, and
the script's config.toml write can be dropped (keeping it is harmless: mode 600, same key as the env).

### 1b. Restart behaviour (measured)
`runpodctl pod start` came back within seconds (Secure, same host), but:
- **the ssh port changed** (17308 → 23207, same IP). `outputs/pods/<id>.env` would be stale; anything that
  restarts a pod must call `refresh_ssh_info` (pod_lib.sh does this only when host/port are empty).
- the container disk was wiped as documented: `/root` empty (venv gone), and `~/.runpod/config.toml` did
  **not exist** after the restart (it existed, empty, on first boot). The key was still in
  `/proc/1/environ` (daemon logged key len 50). `/workspace` (volume) kept the repo and logs.

### 1c. Full FIRE test of the unmodified `03_pod_deadline.sh` (measured)
`cmp` against the laptop copy: identical. Armed on the restarted pod at 05:28:52 UTC with `--in 240`:
```
pod_deadline armed: stop at 2026-11-06 05:32:52 UTC
armed: stop at 2026-11-06 05:32:52 UTC (pid 161)                      # --status
2026-11-06 05:28:52 UTC pod_deadline: armed pid 161: stop at 05:32:52 UTC; after-exit=0 min
2026-11-06 05:28:52 UTC pod_deadline: daemon up (key len 50, pod 318gzbkr1uormp)
```
Laptop poll: `RUNNING` through 00:32:48 EST, then `"desiredStatus": "EXITED"`, `"lastStatusChange":
"Exited by user: ... 05:32:53 GMT"`. `/workspace/deadline.log`, read after a restart:
```
2026-11-06 05:32:52 UTC pod_deadline: STOPPING pod: hard deadline 2026-11-06 05:32:52 UTC
pod "318gzbkr1uormp" stopped
2026-11-06 05:32:53 UTC pod_deadline: runpodctl stop pod ok
```
The first path (runpodctl v1 `stop pod`) succeeded; the v2 and GraphQL fallbacks were not needed and remain
unexercised. The log on the volume survives the stop, as designed.

### 1d. `--after-exit-min` and re-arm (measured)
Restarted (ssh port changed again, 23207 → 20882), armed `--at now+1800 --after-exit-min 1` **twice**: one
daemon left (`ps` count 1, `--status` pid 196). Then `echo "RUN_EXIT=0 ..." >> /workspace/run.log` at
05:33:45 UTC. Laptop saw `EXITED` with "Exited by user: 05:35:45": 2 min after the marker, as expected
(≤ 60 s poll + 1 min). I did not restart again to read that log (cost), so its log lines are not shown.

**Bug found while arming (minor):** `kill_old` runs `pgrep -f 'pod_deadline.sh .*--_daemon' | xargs kill`.
That pattern also matches any shell whose command line contains that text. My ssh command (which included
`pgrep -fc "pod_deadline.sh .*--_daemon"` after the arm) was killed mid-command. launch_pod.sh's arm call
(`bash .../pod_deadline.sh --at N --after-exit-min 60`) does not contain `--_daemon`, so the real hook is
safe; but the pattern should be tightened, e.g. `pgrep -f '^bash [^ ]*pod_deadline\.sh --_daemon'`
(the daemon is always exec'd as `bash $0 --_daemon ...`).

## 2. Setup-speed gate

### 2a. Setup timings on this host (measured; Secure RTX 2000 Ada, EU-RO-1, 6 vCPU quota, 48 visible CPUs)
| step | time | notes |
|---|---|---|
| `uvx --from huggingface_hub[hf_xet] hf version` (cold uv cache) | 1.6 s | tool env for pod_setup's download |
| `hf download Qwen/Qwen3-0.6B model.safetensors` (1,503,300,328 B, xet) | **3.2 s** | ≈ 470 MB/s |
| same file, single `curl` stream | 14.5 s | 104 MB/s |
| 100 MB ranged `curl` from HF | 1.8 s | 59 MB/s |
| `uv venv` | 0.1 s | |
| `uv pip install -e .[gpu] liger-kernel`, **cold cache** | **36.8 s** | 210 pkgs; torch 2.14.0+cu130, vllm 0.33.0, transformers 5.20.2, peft 0.21.2, liger 0.8.5 |
| same, warm uv cache, new venv | 2.3 s | hardlinks from cache |

Compare (measured by fleet 1, efficiency2/09 and 02): H100 pods 109–130 s venv (one 513 s); the stalled
Community A100 host >9 min with load ~665. Setup speed is a property of the host and its network path,
not the GPU. This host would have finished a 14B setup in ≈ 40 s (venv) in parallel with ≈ 60–70 s for the
30 GB model (estimate from 470 MB/s).

### 2b. The gate (new; `patches/03_setup_gate.sh`)
Algorithm: right after ssh comes up and **before** rsync/setup, run a ~7 s probe on the pod. Each probe is
a fixed amount of work, so slowness shows up directly:
1. load(1 min) / nproc ≤ 4 (cheap; catches the efficiency2/02 host, load ≈ 665).
2. single-thread CPU: a fixed 5 M-iteration Python loop ≤ 1.5 s.
3. HF: 100 MB ranged GET ≥ 15 MB/s.
4. PyPI CDN: 100 MB ranged GET of the current torch cp312 wheel (what uv fetches) ≥ 15 MB/s.
5. small-file install: cold-cache `uv pip install sympy` into a temp venv ≤ 8 s (resolve + fetch + unpack
   ~1k files: the stall mode of 02).
6. container disk: 1 GiB write + fdatasync ≥ 150 MB/s.
Whole probe capped at 120 s (not finishing is itself a fail). On fail, launch_pod deletes that pod
immediately and moves to the next (cloud, GPU) option (loop sketch at the bottom of the patch). Backstop:
`setup_deadline_watch 420` (fail if `SETUP_EXIT=0` is not in setup.log after 7 min).

### 2c. Validation (measured)
| probe | good host run 1 | run 2 | **simulated starved host** (60 busy loops in the container) | threshold |
|---|---|---|---|---|
| load/nproc | 0.18 | 0.18 | 0.20 (did not see it) | ≤ 4 |
| cpu loop | 0.42 s | 0.44 s | **22.2 s** | ≤ 1.5 s |
| HF 100 MB | 58.7 MB/s | 63.0 | **11.7** | ≥ 15 |
| PyPI 100 MB | 111.6 MB/s | 158.2 | **5.9** | ≥ 15 |
| sympy install | 0.83 s | 0.81 s | **20.6 s** | ≤ 8 s |
| disk write | 3460 MB/s | 3500 | 298 | ≥ 150 |
| result / wall | pass, 7 s | pass, 6 s | **fail (cpu hf pypi install)**, 88 s | |
Thresholds sit 3.5–25× below the good host. The load probe did not detect in-container contention
(loadavg is host-wide over 48 CPUs and lags); it is kept only for the host-wide case seen in 02.
Not validated: a real slow Community host (none was rented), and network-only slowness (no `tc` in the
container). Threshold false-fail rate across regions is unknown, so run it in **log-only mode** for the
next few launches before enforcing.

## 3. Prebuilt venv tarball (measured; nothing uploaded)
```
venv: 8.9 GB apparent, 71,476 files
tar | zstd -T0 -3 : 12.2 s -> 3,926,033,565 B (2.44x)
tar | zstd -T0 -1 : 12.0 s -> 4,269,144,975 B
plain tar         : 14.0 s -> 9,577,359,360 B
zstd -dc | tar x  : 21.1 s (l3), 21.5 s (l1)      # file-creation bound, not decompression bound
restored venv     : import torch, transformers, peft, vllm, liger_kernel OK (13.4 s cold first import);
                    torch.cuda.is_available() True, CUDA 13.0
```
Gotcha: shebangs are absolute (`#!/root/venv/bin/python`), so the tarball must be restored to the same
path (`/root/venv`), which pod_setup.sh already uses. No secrets in the venv (only the -e link to the repo).

Restore estimate: download 3.93 GB at the xet rate measured above (≈ 470 MB/s) ≈ 8 s, or ≈ 38 s single
stream; streamed into `zstd -d | tar x` the two overlap → **≈ 25–40 s** (estimate). Versus a cold install:
36.8 s here (tie), 109–130 s on fleet 1's H100 pods (saves ≈ 1.2–1.5 min), 513 s outlier (saves ≈ 8 min).

## $ effect
- Deadline: no cost in normal runs; caps a laptop-asleep runaway at (max-hours + 0.5 h) × rate instead of
  ≈ $9–40 overnight (efficiency2/03). Now verified end to end.
- Gate: ≈ 7 s per good pod (≈ $0.0005 at $0.24/h, ≈ $0.006 at $2.99/h). On a bad host: ≤ 2 min of its rate
  (≈ $0.01–0.04) instead of a stalled 10+ min session (≈ $0.2–0.5 plus a lost slot and operator time).
  Expected value depends on how often Community hosts are bad (1 of ~2 tried in fleet 1; speculative).
- Venv tarball: ≈ $0.00 on a fast host; ≈ 1.5 min typical = **$0.015 per pod at $0.59/h (A40), $0.07 at
  $2.99/h (H100 Secure)**; ≈ 8 min on outliers. Small per pod; most of its value (removing install
  variance) overlaps with the gate. It also needs a public HF upload and a rebuild per version bump.

## Gotcha for every watchdog: `runpodctl pod list` hides stopped pods (measured)
At 00:36 EST `runpodctl pod list` (laptop, v2.12) did **not** show `eff4-infra`, although
`runpodctl pod get 318gzbkr1uormp` returned it as `EXITED` (self-stopped). `pod list --help`: "defaults to
running pods only; use --all to include stopped ones". So (a) a watchdog that deletes "all eff4-* pods"
from `pod list` will miss every pod the deadline script stopped (it keeps its volume, and that bill, until
deleted); (b) "confirm deletion with `runpodctl pod list`" is not a valid check for a stopped pod. Use
`runpodctl pod list --all`, or `pod get <id>` and expect an error. The same holds for launch_pod.sh's
stray-pod cleanup and guard_pod.sh if they ever need to see stopped pods.

## Crash lesson (00:08 GUI crash): guards must survive the app and fail loudly
Observed (coordinator): after the Claude app crashed, background guards/watchdogs it had spawned lost
their permissions (`runpodctl` / `python` → "Operation not permitted"), and one watchdog read that failure
as "no pods" and went quiet. Two design rules:
1. **Fail loudly, never infer "empty" from an error.** Every poll must check the exit code and parse the
   JSON strictly; error or unparseable output = UNKNOWN, not "no pods". After N consecutive UNKNOWNs
   (e.g. 3 × 15 s) raise an alert (`osascript -e 'display notification ...'` plus a sound, and a line in a
   file the user watches), and keep retrying. Write a heartbeat file every poll; a second, independent
   checker alerts if it is stale.
2. **Run guards outside the app's process tree and sandbox.** Children of the app (nohup, `&`, caffeinate)
   inherit its sandbox and die or lose rights with it. Use a launchd user agent per pod instead:
   `~/Library/LaunchAgents/com.sdf.guard.<pod>.plist` with `ProgramArguments` = `bash guard_pod.sh ...`,
   `RunAtLoad`, `KeepAlive` (on failure), `StandardOutPath`/`StandardErrorPath` under `outputs/pods/`,
   loaded with `launchctl bootstrap gui/$(id -u) <plist>` and removed with `launchctl bootout` when the
   pod is gone. It is persistent configuration, so the user installs it once (a template plus a
   `guard_launchd.sh install|remove` wrapper). A sleeping laptop still runs nothing, which is why the
   pod-side deadline (section 1) is the real backstop: arm it on every launch, not only with `--max-hours`.

## Risks
- n=1 host for every timing; EU-RO-1 Secure only. Setup timings vary by region/host (fleet 1 saw 3–4×
  slower venv installs on its H100 pods).
- Gate thresholds may false-fail on far regions (HF/PyPI single-stream throughput). Start in log-only mode.
- Self-stop keeps billing the volume (5 GB here, negligible) and the restarted pod changes ssh port.
- No effect on training numerics or comparability: infrastructure only.

## Recommendation
1. Adopt `03_pod_deadline.sh` (verified), tighten the `pgrep` pattern, optionally drop the config write;
   arm it on every launch.
2. Adopt the gate in log-only mode now, enforce after ~5 launches; use it for every Community launch.
3. Defer the venv tarball: ≤ $0.07 per pod, and it needs a public upload; revisit only if slow installs
   keep showing up after the gate.
4. Move laptop guards to launchd agents with loud failure (above).
5. Every watchdog and deletion check: `runpodctl pod list --all`, not `pod list`.

## Evidence: commands
```bash
# laptop
runpodctl pod create --name eff4-infra --gpu-id "NVIDIA RTX A5000" --cloud-type COMMUNITY --public-ip --min-cuda-version 13.0 ...  # no stock
runpodctl pod create --name eff4-infra --gpu-id "NVIDIA RTX 2000 Ada Generation" --cloud-type SECURE \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --container-disk-in-gb 50 --volume-in-gb 5 --ports 22/tcp --wait
tar czf - pyproject.toml distill scripts docs/efficiency2/patches/03_pod_deadline.sh | ssh ... "cd /workspace/sdf-training-quirks && tar xzf -"
ssh ... "timeout 120 bash -s" < docs/efficiency3/patches/03_setup_gate.sh          # gate, x2 (+ x1 under 60 busy loops)
bash -c "pod_ssh(){ ssh ...; }; log(){ echo; }; source 03_setup_gate.sh; setup_gate"  # laptop fn: pass; GATE_MAX_CPU_S=0.1: fail cpu
runpodctl pod get 318gzbkr1uormp | grep desiredStatus   # polled every 6-10 s
runpodctl pod start 318gzbkr1uormp; runpodctl ssh info 318gzbkr1uormp   # new port each time
# pod: setup bench (bash, timings via $EPOCHREALTIME)
a=$EPOCHREALTIME; uvx --from "huggingface_hub[hf_xet]" hf version >/dev/null 2>&1; b=$EPOCHREALTIME; echo "T_uvx_hf_tool_cold $(d $a $b)"
a=$EPOCHREALTIME; uvx --from "huggingface_hub[hf_xet]" hf download Qwen/Qwen3-0.6B model.safetensors --local-dir /root/m06 >/dev/null 2>&1; b=$EPOCHREALTIME; echo "T_hf_xet_dl $(d $a $b) bytes $(stat -c %s /root/m06/model.safetensors)"
a=$EPOCHREALTIME; curl -sL -o /root/raw.bin -w "curl_full %{size_download} B %{speed_download} B/s\n" https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors; b=$EPOCHREALTIME; echo "T_curl_dl $(d $a $b)"
a=$EPOCHREALTIME; curl -sL -r 0-104857599 -o /dev/null -w "curl_probe %{size_download} B %{speed_download} B/s\n" https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors; b=$EPOCHREALTIME; echo "T_curl_probe100MB $(d $a $b)"
a=$EPOCHREALTIME; curl -sL -o /dev/null -w "pypi_probe %{size_download} B %{speed_download} B/s\n" https://files.pythonhosted.org/packages/source/n/numpy/numpy-2.1.3.tar.gz; b=$EPOCHREALTIME; echo "T_pypi_probe $(d $a $b)"
a=$EPOCHREALTIME; uv venv -q /root/venv --python 3.12; b=$EPOCHREALTIME; echo "T_venv $(d $a $b)"
a=$EPOCHREALTIME; uv pip install -q --python /root/venv/bin/python -e ".[gpu]" liger-kernel; rc=$?; b=$EPOCHREALTIME; echo "T_uv_install_cold $(d $a $b) rc=$rc"
a=$EPOCHREALTIME; uv venv -q /root/venv2 --python 3.12; uv pip install -q --python /root/venv2/bin/python -e ".[gpu]" liger-kernel; b=$EPOCHREALTIME; echo "T_uv_install_warmcache $(d $a $b)"
# pod: venv tarball
# zstd v1.5.5; for L in 3 1:
  a=$EPOCHREALTIME; tar -C /root -cf - venv | zstd -q -T0 -$L -o /root/venv$L.tar.zst; b=$EPOCHREALTIME
a=$EPOCHREALTIME; tar -C /root -cf /root/venv.tar venv; b=$EPOCHREALTIME; echo "T_tar_plain $(d $a $b) size $(stat -c %s /root/venv.tar)"
  a=$EPOCHREALTIME; zstd -q -dc -T0 /root/venv$L.tar.zst | tar -C /root -xf -; b=$EPOCHREALTIME
# pod: env-only self-stop (config.toml apikey left empty)
for v in RUNPOD_API_KEY RUNPOD_POD_ID; do export "$v=$(tr '\0' '\n' < /proc/1/environ | sed -n "s/^$v=//p" | head -1)"; done
sleep 45; runpodctl stop pod "$RUNPOD_POD_ID"     # started with setsid nohup; rc=0
# pod: fire tests of the unmodified patch
bash docs/efficiency2/patches/03_pod_deadline.sh --in 240
bash docs/efficiency2/patches/03_pod_deadline.sh --at $(( $(date +%s) + 1800 )) --after-exit-min 1   # x2, then RUN_EXIT
```

## Spend
Running intervals (EST): 00:01:22–00:26:32 (env-only self-stop), 00:26:44–00:32:53 (script fire),
00:33:07–00:35:45 (after-exit fire) = 34.0 min × $0.24/h ≈ **$0.14** (cap $0.25). The pod was left
**EXITED** (stopped by its own deadline, 5 GB volume only); per the coordinator I did not delete it: the
watchdog deletes eff4-* pods at 01:12. Nothing was uploaded.
