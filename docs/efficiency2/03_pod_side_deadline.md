# 03: Pod-side deadline (dead-man stop)

Agent 03, 2026-11-05. GPU agent. Spent about $0.09 (one RTX 4090 Secure pod at $0.89/hr, ~6 min; created
17:54:19, deleted 18:00:13 EST).

## TL;DR
- **Measured: a pod can stop itself.** On the launcher's image (`runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404`),
  a detached (`setsid nohup`) 3-minute timer ran `runpodctl stop pod $RUNPOD_POD_ID`. The pod went `EXITED`
  ("Exited by user", 1 s after the timer fired), and the laptop saw it via `runpodctl pod get`. The pod was
  then deleted from the laptop. `runpodctl pod list` no longer shows `eff3-deadline`.
- **Two gotchas.** (1) `RUNPOD_API_KEY` / `RUNPOD_POD_ID` are **not** in the environment of ssh, nohup or
  tmux shells (length 0). They **are** in `/proc/1/environ` (lengths 50 / 14). (2) The on-pod `runpodctl` is
  **v1.14.15** (old syntax `runpodctl stop pod ID`, not `pod stop`), and its `~/.runpod/config.toml` has an
  **empty** apikey. The verified stop ran with the pod key written there (mode 600). The pod-scoped key
  cannot read pods (`runpodctl get pod` → Unauthorized; REST v1 → 403) but can stop the pod. GraphQL also
  accepts the key (a read query returned 200).
- **Deliverable:** `docs/efficiency2/patches/03_pod_deadline.sh`. It stops the pod (never deletes) at the
  guard deadline + 30 min, re-arming kills any earlier timer, and `--cancel`/`--status` are available. It
  tries runpodctl v1, then v2 syntax, then GraphQL `podStop`, with retries. **Cost case:** it caps the
  worst case (laptop asleep with the lid shut) at (max-hours + 0.5 h) × rate. Without it the pod bills
  until someone notices: an overnight sleep (~10 h) costs ~$9 on a 4090, ~$13 on an A100 and ~$40 on a
  Secure H100.

## What's new vs docs/efficiency
06_infra.md §3 and HANDOFF §5 marked this "still missing" and listed checks to do first. Those checks are
now done on a real pod (findings above, all **measured**). A tested script and a launcher hook are written.
The earlier blocker was the Claude sandbox, not Runpod. The pod-scoped key does have permission to stop
its own pod.

## Findings

### F1. Env visibility (measured)
| context | RUNPOD_API_KEY len | RUNPOD_POD_ID len | runpodctl |
|---|---|---|---|
| ssh `bash -s` | 0 | 0 | /usr/bin/runpodctl v1.14.15 |
| `nohup bash probe.sh &` (from ssh) | 0 | 0 | found |
| `tmux new -d` (from ssh) | 0 | 0 | found |
| `/proc/1/environ` | 50 | 14 | n/a |

So anything started through `launch_pod.sh` (ssh → tmux) must import the key from `/proc/1/environ`.
The script does this without printing it: `tr '\0' '\n' < /proc/1/environ | sed -n 's/^RUNPOD_API_KEY=//p'`.
We read only the length and never printed the value.

### F2. Self-stop works (measured, n=1)
Timer script (`setsid nohup selfstop.sh 180`): it imports the env from /proc/1, sleeps 180 s, then runs
`runpodctl stop pod $RUNPOD_POD_ID`. Its log was tailed live over ssh from the laptop:
```
23:00:01 firing; key_len=50
pod "xfhzycwpi2ce61" stopped
23:00:02 runpodctl ok
Connection to 213.192.6.68 closed by remote host.
```
Laptop: `"desiredStatus": "EXITED"`, `"lastStatusChange": "Exited by user: ... 23:00:01 GMT"`.
- Caveat: before the timer ran, we had written the key into `~/.runpod/config.toml`. It is **untested**
  whether v1 `stop pod` authenticates from the env var alone. (`get pod` with env only gave Unauthorized,
  but `get` also fails with the config set, because the scoped key can't read.) The script writes the
  config, which matches the tested path.
- `runpodctl pod stop` (v2 syntax) does not exist in v1. A future image may ship v2, so the script tries
  both, then falls back to a GraphQL `podStop` mutation via curl. The mutation itself was **not exercised**;
  only a read query with the same key and endpoint was.
- The GPU type was an RTX 4090 Secure (3090/4090 Community had no stock). The setup is image-dependent,
  not GPU-dependent, so this should hold on H100/A100 pods with the same image (estimate).

### F3. Robust pod-side deadline (script written; arm, re-arm and cancel tested on the pod; firing tested via the equivalent timer)
`patches/03_pod_deadline.sh` features:
- `--at EPOCH` / `--in SEC`. It re-execs itself detached (`setsid nohup … </dev/null`), so it survives
  ssh hangups and `tmux kill-session -t main`. Note that `launch_pod.sh` kills `main` on a re-launch,
  which would kill a timer started inside tmux.
- **Kills earlier timers** on reused pods via the pidfile plus `pgrep -f 'pod_deadline.sh .*--_daemon'`.
  Tested: re-arming killed the old pid, leaving exactly 1 daemon; `--cancel` left 0.
- **Stop, never delete**, so `/workspace` (volume) is kept. Retries 6× with backoff and logs to
  `/workspace/deadline.log` on the volume, so the log survives the stop.
- The deadline lives in `/workspace/.pod_deadline.at`. The laptop can extend it by re-arming with a new
  `--at`.
- Optional `--after-exit-min M` stops M minutes after `RUN_EXIT` appears in `/workspace/run.log`. This
  catches "job done, laptop asleep" long before the hard deadline. Suggested M = 60: the guard normally
  copies and stops within ~1–5 min of RUN_EXIT, so 60 min leaves a wide margin. Off by default because
  it is a policy choice for the user.
- Cost: the timer wakes once a minute (one `cat` and one `grep`), which is negligible.

### F4. How launch_pod.sh should start it (sketch, not applied)
In step 5, right after the tmux job starts and only when `--max-hours` is set (or always, with a default of
e.g. 12 h when no guard is requested):
```bash
if [ -n "$MAX_HOURS" ]; then
    POD_DEADLINE=$(( $(date +%s) + $(awk -v h="$MAX_HOURS" 'BEGIN{printf "%d", h*3600}') + 1800 ))
    pod_ssh "bash $REMOTE_REPO/scripts/pod_deadline.sh --at $POD_DEADLINE --after-exit-min 60" \
        || log "WARNING: pod-side deadline NOT armed"
    log "pod-side deadline: stop at $(date -u -r $POD_DEADLINE '+%F %T') UTC (guard + 30 min)"
fi
```
(The script must first be copied to `scripts/pod_deadline.sh`; the rsync then delivers it.) In
`guard_pod.sh --no-stop` (copy only), call `pod_ssh "bash $REMOTE_REPO/scripts/pod_deadline.sh --cancel"`
or re-arm it, so a deliberately kept-alive pod is not stopped. With stop/delete, nothing extra is needed:
the timer dies with the container. A `--pod` reuse re-arms, which kills the old timer.

## Risks
- Stopping mid-job loses nothing on `/workspace` but ends the run. The +30 min margin over the laptop
  guard means that, when the laptop is awake, the guard always acts first.
- Stop is not delete. A stopped pod still bills volume storage (small: ~$0.10/GB-month for idle volume,
  per Runpod pricing; estimate). The GPU may be unavailable on restart, which is fine for a last-resort
  backstop.
- The pod-scoped key lives in `~/.runpod/config.toml` (mode 600) on the container disk. That disk is the
  pod's own, and the key is already in the container env, so this adds no exposure.
- No effect on training numerics or comparability. This is pure infrastructure.

## Experiments run
```
runpodctl pod create --name eff3-deadline --gpu-id "NVIDIA GeForce RTX 4090" --cloud-type SECURE \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 --container-disk-in-gb 20 --volume-in-gb 5 --ports 22/tcp
#   (3090/4090 COMMUNITY and 3090 SECURE: "no longer any instances available")
# on pod: env-length probe in ssh / nohup / tmux / /proc/1/environ (table F1); runpodctl -v → 1.14.15
# on pod: runpodctl get pod → Unauthorized; GraphQL read query → 200 {"desiredStatus":"RUNNING"}; REST v1 → 403
# on pod: setsid nohup selfstop.sh 180 → runpodctl stop pod → EXITED at 23:00:01 UTC (F2)
# on pod: 03_pod_deadline.sh --in 3600; --in 7200 (old pid killed, 1 daemon); --cancel (0 daemons)
runpodctl pod delete xfhzycwpi2ce61   # {"deleted": true}; pod list: eff3-deadline absent (18:00:13 EST)
```
Not tested: a full fire of `03_pod_deadline.sh` itself (its stop path is the tested `runpodctl stop pod`
plus the same config write), `--after-exit-min`, and GraphQL `podStop`. To validate cheaply, on the next
real pod run `pod_deadline.sh --in 120` on a throwaway pod (~$0.05) and check
`/workspace/deadline.log` after a restart.

## Recommendation
**Adopt.** Copy the patch to `scripts/pod_deadline.sh`, arm it from `launch_pod.sh` at the guard deadline
+ 30 min, and cancel or re-arm it in `guard_pod.sh --no-stop`. Consider `--after-exit-min 60` as the default.
Validate with one ~$0.05 throwaway-pod fire test before relying on it.
