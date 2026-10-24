#!/usr/bin/env bash
# Laptop-side guard: wait for an exit marker in a pod log OR a hard deadline, copy results off, then
# stop (default) or delete the pod. Copies always happen BEFORE the stop; --delete only happens when
# every copy succeeded (otherwise it falls back to stop, which keeps the volume).
#
#   scripts/guard_pod.sh --max-hours 3 --pull outputs/distill/prof [--pull data/distill/prof] [--delete]
#   scripts/guard_pod.sh --pod <id> --deadline-utc "2026-10-25 09:00" --log /workspace/train.log \
#       --marker TRAIN_EXIT --pull outputs/distill/chatty
#
# Options:
#   --pod ID              default: outputs/pods/current.env
#   --max-hours H | --deadline-utc "YYYY-mm-dd HH:MM"   (one is required)
#   --log FILE            pod log to watch (default /workspace/run.log)
#   --marker STR          line prefix that means "done" (default RUN_EXIT; launch_pod.sh writes it)
#   --pull PATH           repo-relative dir/file to copy back (repeatable)
#   --save-cache          also copy vLLM/Triton/flashinfer caches to outputs/cache/ (best effort)
#   --delete              delete instead of stop (after successful copies only)
#   --no-stop             copy only, leave the pod running
#   --poll SEC            default 60
# Always copies /workspace/{run,setup,download}.log and pip_freeze.txt to outputs/pods/<id>/.
# Run it under `caffeinate -is` (launch_pod.sh does) so the laptop doesn't idle-sleep.
set -uo pipefail
# shellcheck source=pod_lib.sh
source "$(dirname "$0")/pod_lib.sh"
cd "$REPO_DIR"

WANT_POD=""; MAX_HOURS=""; DEADLINE_UTC=""; LOGF=/workspace/run.log; MARKER=RUN_EXIT
PULLS=(); SAVE_CACHE=0; ACTION=stop; POLL=60
while [ $# -gt 0 ]; do
    case "$1" in
        --pod) WANT_POD="$2"; shift 2 ;;
        --max-hours) MAX_HOURS="$2"; shift 2 ;;
        --deadline-utc) DEADLINE_UTC="$2"; shift 2 ;;
        --log) LOGF="$2"; shift 2 ;;
        --marker) MARKER="$2"; shift 2 ;;
        --pull) PULLS+=("$2"); shift 2 ;;
        --save-cache) SAVE_CACHE=1; shift ;;
        --delete) ACTION=delete; shift ;;
        --no-stop) ACTION=none; shift ;;
        --poll) POLL="$2"; shift 2 ;;
        -h|--help) sed -n '2,22p' "$0"; exit 0 ;;
        *) die "unknown option $1" ;;
    esac
done

if [ -n "$MAX_HOURS" ]; then
    DEADLINE=$(( $(date +%s) + $(awk -v h="$MAX_HOURS" 'BEGIN { printf "%d", h * 3600 }') ))
elif [ -n "$DEADLINE_UTC" ]; then
    DEADLINE="$(date -j -u -f '%Y-%m-%d %H:%M' "$DEADLINE_UTC" +%s 2>/dev/null || date -u -d "$DEADLINE_UTC" +%s)" \
        || die "bad --deadline-utc '$DEADLINE_UTC'"
else
    die "need --max-hours or --deadline-utc (a hard deadline is mandatory)"
fi
load_pod "$WANT_POD"
OUT="$POD_DIR/$POD_ID"; mkdir -p "$OUT"
log "guarding $POD_ID (\$${POD_COST:-?}/hr): marker '$MARKER' in $LOGF, deadline $(date -u -r "$DEADLINE" '+%Y-%m-%d %H:%M' 2>/dev/null || date -u -d "@$DEADLINE" '+%Y-%m-%d %H:%M')Z, then $ACTION"

# --- wait ---------------------------------------------------------------------------------------------
REASON=""; n=0
while :; do
    if [ "$(date +%s)" -ge "$DEADLINE" ]; then REASON=deadline; break; fi
    if pod_ssh "grep -q '^$MARKER' $LOGF" 2>/dev/null; then REASON=marker; break; fi
    n=$((n + 1))
    if [ $((n % 5)) = 0 ]; then   # every ~5 polls: is the pod still up at all? (port may have moved)
        st="$(pod_status)"
        if [ -n "$st" ] && [ "$st" != RUNNING ]; then
            log "pod is $st (stopped elsewhere?); nothing to copy or stop"; exit 2
        fi
        refresh_ssh_info; set_ssh
    fi
    sleep "$POLL"
done
log "trigger: $REASON"
pod_ssh "grep '^$MARKER' $LOGF | tail -1" 2>/dev/null | sed 's/^/  pod says: /'

# --- copy ---------------------------------------------------------------------------------------------
FAILED=0
for p in "${PULLS[@]+"${PULLS[@]}"}"; do
    if pull_path "$p"; then log "copied $p"; else log "COPY FAILED: $p"; FAILED=1; fi
done
for f in "$LOGF" /workspace/run.log /workspace/setup.log /workspace/download.log /workspace/pip_freeze.txt; do
    rsync "${RSYNC_FLAGS[@]}" -e "ssh $SSH_OPTS" "$SSH_TARGET:$f" "$OUT/" 2>/dev/null || true
done
log "logs -> $OUT/"
if [ "$SAVE_CACHE" = 1 ]; then
    for nc in $CACHE_DIRS; do
        c="${nc%%:*}"; rp="${nc#*:}"
        mkdir -p "$REPO_DIR/outputs/cache/$c"
        rsync "${RSYNC_FLAGS[@]}" -e "ssh $SSH_OPTS" "$SSH_TARGET:$rp/" "$REPO_DIR/outputs/cache/$c/" 2>/dev/null \
            && log "saved $c cache" || true
    done
fi

# --- stop / delete ------------------------------------------------------------------------------------
if [ "$ACTION" = delete ] && [ "$FAILED" = 1 ]; then
    log "some copies failed -> stopping instead of deleting (volume kept; retry: scripts/pod_sync.sh down <dir>)"
    ACTION=stop
fi
case "$ACTION" in
    none) log "--no-stop: pod $POD_ID left RUNNING at \$${POD_COST:-?}/hr" ;;
    stop|delete)
        for t in 1 2 3; do
            runpodctl pod "$ACTION" "$POD_ID" >/dev/null 2>&1 && break
            log "pod $ACTION failed (try $t/3)"; sleep 20
        done
        sleep 5
        log "pod $ACTION issued; status now: $(pod_status || true)"
        ;;
esac
command -v osascript >/dev/null && osascript -e "display notification \"$POD_ID: $REASON, copies failed=$FAILED, $ACTION\" with title \"guard_pod\"" 2>/dev/null
log "guard done (reason=$REASON, copy_failed=$FAILED, action=$ACTION)"
exit "$FAILED"
