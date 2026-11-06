#!/usr/bin/env bash
# Pod-side dead-man deadline: STOPS (never deletes) this pod at a fixed epoch, so a sleeping/crashed laptop
# cannot leave a GPU billing. Runs ON THE POD, detached with setsid+nohup. Last resort only: the laptop
# guard (guard_pod.sh) still does copies + stop/delete first; this fires at guard deadline + 30 min.
#
#   pod_deadline.sh --at EPOCH [--after-exit-min M] [--log /workspace/run.log] [--marker RUN_EXIT]
#   pod_deadline.sh --in SECONDS ...            (relative; used for the smoke test)
#   pod_deadline.sh --cancel                    (kill the armed timer, e.g. guard_pod.sh --no-stop)
#   pod_deadline.sh --status
#
# --after-exit-min M: also stop M minutes after a line starting with the marker appears in --log (job done
#   and the laptop never came back to copy/stop). Default 0 = off. Stop keeps /workspace (volume), so
#   nothing on the volume is lost; adapters also reach HF from the run scripts.
#
# Verified on a real pod (RTX 4090, runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404, 2026-11-05):
#   - RUNPOD_API_KEY / RUNPOD_POD_ID are NOT in the env of ssh, nohup or tmux shells; they ARE in
#     /proc/1/environ (len 50 / 14). So we import them from there.
#   - On-pod runpodctl is v1.14.15 (/usr/bin/runpodctl), syntax `runpodctl stop pod ID`; its
#     ~/.runpod/config.toml has an EMPTY apikey. With the key written into config.toml, `runpodctl get pod`
#     still says Unauthorized (pod-scoped key), BUT `runpodctl stop pod $RUNPOD_POD_ID` from a detached
#     3-min timer WORKED: pod went EXITED ("Exited by user", 1 s after firing), seen from the laptop.
#     The GraphQL API also accepts the pod key (read query 200 OK; podStop mutation itself not exercised).
#   => primary path: runpodctl v1 `stop pod` (v2 `pod stop` tried too), GraphQL podStop via curl as fallback.
# Never prints the key. Never deletes. Re-arming kills an earlier timer (pidfile + pattern), so reused
# pods don't accumulate timers with stale deadlines.
set -uo pipefail
STATE=/workspace/.pod_deadline
PIDF=$STATE.pid; ATF=$STATE.at; LOGF_OUT=/workspace/deadline.log
AT=""; IN=""; AFTER_EXIT_MIN=0; WATCH_LOG=/workspace/run.log; MARKER=RUN_EXIT; MODE=arm; DAEMON=0
while [ $# -gt 0 ]; do
    case "$1" in
        --at) AT="$2"; shift 2 ;;
        --in) IN="$2"; shift 2 ;;
        --after-exit-min) AFTER_EXIT_MIN="$2"; shift 2 ;;
        --log) WATCH_LOG="$2"; shift 2 ;;
        --marker) MARKER="$2"; shift 2 ;;
        --cancel) MODE=cancel; shift ;;
        --status) MODE=status; shift ;;
        --_daemon) DAEMON=1; shift ;;
        *) echo "unknown option $1" >&2; exit 2 ;;
    esac
done
mkdir -p /workspace
say() { echo "$(date -u '+%F %T') UTC pod_deadline: $*" >> "$LOGF_OUT"; }

kill_old() {
    local old
    old="$(cat "$PIDF" 2>/dev/null || true)"
    if [ -n "$old" ] && [ "$old" != "$$" ] && kill -0 "$old" 2>/dev/null; then kill "$old" 2>/dev/null; say "killed earlier timer pid $old"; fi
    # belt and braces: any other daemon instance (e.g. pidfile lost on a container reset)
    pgrep -f 'pod_deadline.sh .*--_daemon' | grep -vx "$$" | xargs -r kill 2>/dev/null || true
}

case "$MODE" in
    cancel) kill_old; rm -f "$PIDF" "$ATF"; say "cancelled"; exit 0 ;;
    status) if kill -0 "$(cat "$PIDF" 2>/dev/null || echo x)" 2>/dev/null; then
                echo "armed: stop at $(date -u -d @"$(cat "$ATF")" '+%F %T') UTC (pid $(cat "$PIDF"))"
            else echo "not armed"; fi; exit 0 ;;
esac

if [ "$DAEMON" = 0 ]; then   # arm: validate, kill the old timer, re-exec detached, return at once
    [ -n "$IN" ] && AT=$(( $(date +%s) + IN ))
    [[ "$AT" =~ ^[0-9]+$ ]] || { echo "need --at EPOCH or --in SECONDS" >&2; exit 2; }
    [ "$AT" -gt "$(date +%s)" ] || { echo "deadline is in the past" >&2; exit 2; }
    kill_old
    echo "$AT" > "$ATF"
    setsid nohup bash "$0" --_daemon --at "$AT" --after-exit-min "$AFTER_EXIT_MIN" --log "$WATCH_LOG" \
        --marker "$MARKER" >/dev/null 2>&1 < /dev/null &
    echo "$!" > "$PIDF"
    say "armed pid $!: stop at $(date -u -d @"$AT" '+%F %T') UTC; after-exit=${AFTER_EXIT_MIN} min"
    echo "pod_deadline armed: stop at $(date -u -d @"$AT" '+%F %T') UTC"
    exit 0
fi

# --- daemon -------------------------------------------------------------------------------------------
# Import the pod identity from PID 1's environment (ssh/tmux/nohup shells don't have it). No printing.
for v in RUNPOD_API_KEY RUNPOD_POD_ID; do
    if [ -z "${!v:-}" ]; then
        val="$(tr '\0' '\n' < /proc/1/environ 2>/dev/null | sed -n "s/^$v=//p" | head -1)"
        export "$v=$val"
    fi
done
[ -n "${RUNPOD_API_KEY:-}" ] && [ -n "${RUNPOD_POD_ID:-}" ] || { say "ERROR: no RUNPOD_API_KEY/POD_ID; cannot self-stop"; exit 1; }
# On-pod runpodctl v1 ships an empty apikey in ~/.runpod/config.toml; the verified stop had the key there
# (env-only auth was not tested for stop; `get pod` with env only was Unauthorized). Fill it, mode 600.
if ! grep -q "^apikey *= *'..*'" ~/.runpod/config.toml 2>/dev/null; then
    mkdir -p ~/.runpod; (umask 077; printf "apikey = '%s'\napiurl = 'https://api.runpod.io/graphql'\n" \
        "$RUNPOD_API_KEY" > ~/.runpod/config.toml)
fi
say "daemon up (key len ${#RUNPOD_API_KEY}, pod $RUNPOD_POD_ID)"

stop_self() {   # $1 = reason. runpodctl v1 (verified), v2 syntax, then GraphQL podStop. Retries. Never deletes.
    local i out
    say "STOPPING pod: $1"; sync
    for i in 1 2 3 4 5 6; do
        runpodctl stop pod "$RUNPOD_POD_ID" >> "$LOGF_OUT" 2>&1 && { say "runpodctl stop pod ok"; sync; return 0; }
        runpodctl pod stop "$RUNPOD_POD_ID" >> "$LOGF_OUT" 2>&1 && { say "runpodctl pod stop ok"; sync; return 0; }
        out="$(curl -s -m 30 -H 'Content-Type: application/json' -H "Authorization: Bearer $RUNPOD_API_KEY" \
            -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}" \
            https://api.runpod.io/graphql 2>&1)"
        say "graphql try $i: $(printf '%s' "$out" | tr -d '\n' | cut -c1-200)"; sync
        printf '%s' "$out" | grep -q '"desiredStatus":"EXITED"' && return 0
        sleep $(( i * 20 ))
    done
    say "ERROR: all stop attempts failed; pod still billing"; return 1
}

exit_seen=""
while :; do
    now=$(date +%s)
    # the laptop may extend the deadline by rewriting the .at file (or re-arming)
    at="$(cat "$ATF" 2>/dev/null || echo "$AT")"; [[ "$at" =~ ^[0-9]+$ ]] || at="$AT"
    if [ "$now" -ge "$at" ]; then stop_self "hard deadline $(date -u -d @"$at" '+%F %T') UTC"; exit $?; fi
    if [ "$AFTER_EXIT_MIN" -gt 0 ] && [ -z "$exit_seen" ] && grep -q "^$MARKER" "$WATCH_LOG" 2>/dev/null; then
        exit_seen=$now; say "saw $MARKER; will stop in $AFTER_EXIT_MIN min unless the laptop acts first"
    fi
    if [ -n "$exit_seen" ] && [ $(( now - exit_seen )) -ge $(( AFTER_EXIT_MIN * 60 )) ]; then
        stop_self "$MARKER + ${AFTER_EXIT_MIN} min"; exit $?
    fi
    left=$(( at - now )); sleep $(( left < 60 ? (left > 1 ? left : 1) : 60 ))
done
