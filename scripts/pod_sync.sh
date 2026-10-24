#!/usr/bin/env bash
# Laptop-side helper for the pod saved by launch_pod.sh (outputs/pods/current.env, or --pod ID).
#
#   scripts/pod_sync.sh up                       repo -> pod (same excludes as launch; SYNC_EXCLUDE adds more)
#   scripts/pod_sync.sh down PATH [PATH...]      pod repo PATH -> local PATH (repo-relative; dirs or files)
#   scripts/pod_sync.sh ssh [CMD...]             shell on the pod, or run CMD
#   scripts/pod_sync.sh log [FILE] [N]           tail the job log (default /workspace/run.log, 30 lines)
#   scripts/pod_sync.sh status                   pod status, cost, tmux sessions, GPU use, setup/run markers
#   scripts/pod_sync.sh stop|delete              copy NOTHING, just stop/delete (prefer guard_pod.sh)
set -euo pipefail
# shellcheck source=pod_lib.sh
source "$(dirname "$0")/pod_lib.sh"
cd "$REPO_DIR"

WANT=""
if [ "${1:-}" = --pod ]; then WANT="$2"; shift 2; fi
cmd="${1:-status}"; shift || true
load_pod "$WANT"

case "$cmd" in
    up) sync_up; log "synced repo -> $POD_ID:$REMOTE_REPO" ;;
    down)
        [ $# -gt 0 ] || die "down needs at least one repo-relative path"
        for p in "$@"; do pull_path "$p" && log "copied $p"; done ;;
    ssh)
        if [ $# -eq 0 ]; then
            # shellcheck disable=SC2086
            exec ssh -t $SSH_OPTS "$SSH_TARGET"
        fi
        pod_ssh "$@" ;;
    log) pod_ssh "tail -n ${2:-30} ${1:-/workspace/run.log}" ;;
    status)
        log "pod $POD_ID ($POD_CLOUD, \$${POD_COST:-?}/hr, created ${POD_CREATED:-?}): $(pod_status)"
        pod_ssh 'tmux ls 2>/dev/null; nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader;
                 grep -h -E "SETUP_OK|SETUP_FAILED|SETUP_EXIT|RUN_EXIT" /workspace/setup.log /workspace/run.log 2>/dev/null;
                 tail -n 2 /workspace/run.log 2>/dev/null' || log "ssh failed" ;;
    stop|delete)
        log "${cmd} $POD_ID without copying anything"
        runpodctl pod "$cmd" "$POD_ID" | json_get desiredStatus ;;
    *) die "unknown command $cmd (up|down|ssh|log|status|stop|delete)" ;;
esac
