#!/usr/bin/env bash
# Laptop side: create a GPU pod, push the repo, and start setup (+ an optional job) in tmux, hands-off.
#
#   scripts/launch_pod.sh --name prof --run "bash scripts/run_prof.sh train" \
#       --guard-pull outputs/distill/prof --max-hours 3 [--delete]
#
# Steps: Community H100 once -> Secure H100 immediately if that fails (no waiting loops); wait for SSH;
# save host/port to outputs/pods/<id>.env (+ current.env); install rsync/tmux if missing; write
# /workspace/env.sh with HF_TOKEN (sent over ssh stdin, never on a command line or printed); rsync the
# repo; start `pod_setup.sh` and then the --run command in tmux session "main". The job's output goes to
# /workspace/run.log and ends with a RUN_EXIT=<code> line. With --max-hours, guard_pod.sh is started in
# the background (under caffeinate) to copy results off and stop/delete the pod when RUN_EXIT appears or
# the deadline passes.
#
# Options:
#   --name TAG            pod name prefix (default sdf); a timestamp is appended
#   --pod ID              reuse an existing running pod (skip creation)
#   --gpu ID              default "NVIDIA H100 80GB HBM3"
#   --secure-only         skip the Community attempt
#   --container-gb N      container disk (venv ~12 GB + model 30 GB + caches); default 80
#   --volume-gb N         /workspace volume (repo, outputs, logs only now); default 50
#   --models "A B"        HF repos for pod_setup.sh (default Qwen/Qwen3-14B)
#   --run "CMD"           run in the repo dir after setup succeeds (venv + env.sh active)
#   --max-hours H         hard deadline for the guard (required to start one)
#   --guard-pull DIR      repo-relative dir/file the guard copies back (repeatable)
#   --delete              guard deletes instead of stops (only after every copy succeeded)
#   --save-cache          guard also copies vLLM/Triton/flashinfer caches to outputs/cache/ (pushed to
#                         the next pod automatically if present)
#   --dry-run             print the plan, create nothing
set -euo pipefail
# shellcheck source=pod_lib.sh
source "$(dirname "$0")/pod_lib.sh"
cd "$REPO_DIR"

TAG=sdf; REUSE=""; GPU="NVIDIA H100 80GB HBM3"; SECURE_ONLY=0; CDISK=80; VDISK=50
MODELS="Qwen/Qwen3-14B"; RUN=""; MAX_HOURS=""; PULLS=(); DELETE=0; SAVE_CACHE=0; DRY=0
IMAGE="runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404"
while [ $# -gt 0 ]; do
    case "$1" in
        --name) TAG="$2"; shift 2 ;;
        --pod) REUSE="$2"; shift 2 ;;
        --gpu) GPU="$2"; shift 2 ;;
        --secure-only) SECURE_ONLY=1; shift ;;
        --container-gb) CDISK="$2"; shift 2 ;;
        --volume-gb) VDISK="$2"; shift 2 ;;
        --models) MODELS="$2"; shift 2 ;;
        --run) RUN="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --guard-pull) PULLS+=("$2"); shift 2 ;;
        --delete) DELETE=1; shift ;;
        --save-cache) SAVE_CACHE=1; shift ;;
        --dry-run) DRY=1; shift ;;
        -h|--help) sed -n '2,32p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done
if [ ${#PULLS[@]} -gt 0 ] && [ -z "$MAX_HOURS" ]; then die "--guard-pull needs --max-hours (hard deadline)"; fi
if [ -n "$MAX_HOURS" ] && [ -z "$RUN" ]; then die "--max-hours starts a guard, which needs a --run job"; fi

# --- preflight (no secrets printed) ---------------------------------------------------------------
command -v runpodctl >/dev/null || die "runpodctl not found"
[ -f "$SSH_KEY" ] || die "ssh key $SSH_KEY missing"
grep -qE '^HF_TOKEN=.+' .env 2>/dev/null || die ".env has no HF_TOKEN"
POD_NAME="${TAG}-$(date +%m%d-%H%M%S)"

if [ "$DRY" = 1 ]; then
    log "dry run: would create '$POD_NAME' ($GPU, ${CDISK} GB container, ${VDISK} GB volume, $IMAGE)"
    [ "$SECURE_ONLY" = 1 ] || log "  try COMMUNITY (--public-ip) once, then SECURE"
    log "  setup MODELS='$MODELS'; run: ${RUN:-<none>}; guard: ${MAX_HOURS:-none} h, pull: ${PULLS[*]:-<none>}, delete=$DELETE"
    runpodctl gpu list 2>/dev/null | python3 -c '
import json, sys
for g in json.load(sys.stdin):
    if g.get("gpuId") == sys.argv[1]:
        print("  %s: stock %s, community $%s/hr, secure $%s/hr" % (g.get("gpuId"), g.get("stockStatus"),
              g.get("communityPricePerHr"), g.get("securePricePerHr")))
' "$GPU" || true
    exit 0
fi

# --- 1. create (or reuse) -------------------------------------------------------------------------
create_pod() {  # create_pod COMMUNITY|SECURE ; sets POD_ID on success
    local cloud="$1" extra=() out id
    [ "$cloud" = COMMUNITY ] && extra=(--public-ip)
    log "creating $POD_NAME on $cloud ($GPU)"
    out="$(runpodctl pod create --name "$POD_NAME" --gpu-id "$GPU" --cloud-type "$cloud" ${extra[@]+"${extra[@]}"} \
            --image "$IMAGE" --container-disk-in-gb "$CDISK" --volume-in-gb "$VDISK" --ports "22/tcp" \
            --wait --wait-timeout 12m 2>&1)" || true
    id="$(printf '%s' "$out" | json_get id)"
    if [ -n "$id" ] && ! printf '%s' "$out" | grep -q '"error"'; then
        POD_ID="$id"; POD_CLOUD="$cloud"
        POD_COST="$(printf '%s' "$out" | json_get costPerHr)"
        return 0
    fi
    log "$cloud create failed: $(printf '%s' "$out" | tr '\n' ' ' | cut -c1-300)"
    # A --wait timeout keeps the pod: delete anything carrying our unique name so nothing bills.
    local stray
    stray="$(runpodctl pod list 2>/dev/null | python3 -c '
import json, sys
try:
    pods = json.load(sys.stdin)
except Exception:
    pods = []
print(" ".join(p["id"] for p in pods if p.get("name") == sys.argv[1]))' "$POD_NAME")"
    for id in $stray; do
        log "deleting half-created pod $id ($POD_NAME)"
        runpodctl pod delete "$id" >/dev/null 2>&1 || log "WARNING: could not delete $id; delete it by hand"
    done
    return 1
}

if [ -n "$REUSE" ]; then
    POD_ID="$REUSE"; POD_CLOUD="reused"
    POD_COST="$(runpodctl pod get "$POD_ID" 2>/dev/null | json_get costPerHr)"
else
    if [ "$SECURE_ONLY" = 1 ] || ! create_pod COMMUNITY; then
        create_pod SECURE || die "no $GPU available on Secure either; nothing is running"
    fi
fi
POD_CREATED="$(date -u '+%Y-%m-%d %H:%M:%S')"
if [ -z "${POD_COST:-}" ]; then
    POD_COST="$(runpodctl pod list 2>/dev/null | python3 -c '
import json, sys
for p in json.load(sys.stdin):
    if p.get("id") == sys.argv[1]:
        print(p.get("costPerHr", ""))' "$POD_ID" || true)"
fi
log "POD $POD_ID  cloud=$POD_CLOUD  cost=\$${POD_COST:-?}/hr  (stop: runpodctl pod stop $POD_ID)"

# --- 2. ssh ------------------------------------------------------------------------------------------
for i in $(seq 1 30); do
    refresh_ssh_info; set_ssh
    if [ -n "$POD_HOST" ] && [ -n "$POD_PORT" ] && pod_ssh true 2>/dev/null; then break; fi
    [ "$i" = 30 ] && die "ssh never came up on $POD_ID (still running and billing: stop or delete it)"
    sleep 10
done
save_pod_env
log "ssh ok: root@$POD_HOST -p $POD_PORT  (state: $POD_ENV; shell: scripts/pod_sync.sh ssh)"

pod_ssh 'command -v rsync >/dev/null && command -v tmux >/dev/null || { apt-get update -qq && DEBIAN_FRONTEND=noninteractive apt-get install -y -qq rsync tmux >/dev/null; }'

# --- 3. secrets: HF_TOKEN only, via stdin, file mode 600 ---------------------------------------------
grep -E '^HF_TOKEN=' .env | head -1 | cut -d= -f2- | sed -e 's/^["'\'']//' -e 's/["'\'']$//' |
    pod_ssh 'umask 077; IFS= read -r t; printf "export HF_TOKEN=%s\nexport HF_HOME=/root/hf\n" "$t" > /workspace/env.sh'
log "wrote /workspace/env.sh (HF_TOKEN only)"

# --- 4. code (+ warm caches from a previous pod) ---------------------------------------------------
sync_up
log "repo synced to $REMOTE_REPO"
for nc in $CACHE_DIRS; do
    c="${nc%%:*}"; rp="${nc#*:}"
    if [ -d "$REPO_DIR/outputs/cache/$c" ]; then
        pod_ssh "mkdir -p $rp"
        rsync_retry "$REPO_DIR/outputs/cache/$c/" "$SSH_TARGET:$rp/" && log "pushed $c cache"
    fi
done

# --- 5. job in tmux ----------------------------------------------------------------------------------
HAS_RUN=0; [ -n "$RUN" ] && HAS_RUN=1
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
{
    echo "cd $REMOTE_REPO"
    echo "source /workspace/env.sh"
    echo "source /workspace/venv/bin/activate"
    printf '%s\n' "$RUN"
} > "$TMP/run_cmd.sh"
cat > "$TMP/job.sh" <<EOF
#!/bin/bash
# written by launch_pod.sh
cd $REMOTE_REPO
source /workspace/env.sh
MODELS="$MODELS" bash scripts/pod_setup.sh > /workspace/setup.log 2>&1
rc=\$?
echo "SETUP_EXIT=\$rc" >> /workspace/setup.log
[ "$HAS_RUN" = 1 ] || exit 0
if [ \$rc -ne 0 ]; then echo "RUN_EXIT=setup_failed_\$rc" >> /workspace/run.log; exit 0; fi
bash /workspace/run_cmd.sh > /workspace/run.log 2>&1
echo "RUN_EXIT=\$? \$(date -u)" >> /workspace/run.log
EOF
pod_ssh 'cat > /workspace/run_cmd.sh' < "$TMP/run_cmd.sh"
pod_ssh 'cat > /workspace/job.sh' < "$TMP/job.sh"
pod_ssh 'rm -f /workspace/run.log; tmux kill-session -t main 2>/dev/null; tmux new-session -d -s main "bash /workspace/job.sh"'
log "started tmux 'main': setup -> ${RUN:-<no job>}   (watch: scripts/pod_sync.sh log)"

# --- 6. guard ----------------------------------------------------------------------------------------
if [ -n "$MAX_HOURS" ]; then
    GARGS=(--pod "$POD_ID" --max-hours "$MAX_HOURS")
    for p in "${PULLS[@]+"${PULLS[@]}"}"; do GARGS+=(--pull "$p"); done
    [ "$DELETE" = 1 ] && GARGS+=(--delete)
    [ "$SAVE_CACHE" = 1 ] && GARGS+=(--save-cache)
    GLOG="$POD_DIR/$POD_ID.guard.log"
    CAF=(); command -v caffeinate >/dev/null && CAF=(caffeinate -is)
    nohup "${CAF[@]+"${CAF[@]}"}" bash "$REPO_DIR/scripts/guard_pod.sh" "${GARGS[@]}" > "$GLOG" 2>&1 &
    log "guard pid $! (log $GLOG): waits for RUN_EXIT or ${MAX_HOURS} h, copies, then $([ "$DELETE" = 1 ] && echo deletes || echo stops)"
    log "keep the laptop awake and on power (caffeinate blocks idle sleep, not lid-close sleep)"
fi
log "done: pod $POD_ID at \$${POD_COST:-?}/hr"
