# Shared helpers for launch_pod.sh / pod_sync.sh / guard_pod.sh (laptop side). Source, don't run.
#
# Pod state lives in outputs/pods/<id>.env (gitignored, never synced up); outputs/pods/current.env
# points at the most recent pod. Override with POD_ENV=outputs/pods/<id>.env.
# Works with macOS bash 3.2 and rsync 2.6.9; every path is quoted (the repo path has two spaces).

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POD_DIR="$REPO_DIR/outputs/pods"
REMOTE_REPO=/workspace/sdf-training-quirks
SSH_KEY="${SSH_KEY:-$HOME/.runpod/ssh/runpodctl-ssh-key}"
KNOWN_HOSTS_DIR="$HOME/.runpod/known_hosts.d"   # per-pod TOFU file: RunPod reuses ip:port across pods
RSYNC_FLAGS=(-rlptz --no-owner --no-group)
# Compile/JIT caches carried across pods (guard --save-cache pulls, launch pushes): name:pod_path.
# vLLM AOT torch.compile cache + Triton kernels (LoRA punica autotune) + flashinfer JIT. On an A40 a
# warm cache cut vLLM engine init from 357 s to 96 s (docs/efficiency/06_infra.md §6).
CACHE_DIRS="vllm:/root/.cache/vllm triton:/root/.triton/cache flashinfer:/root/.cache/flashinfer"

log() { echo "[$(date -u '+%Y-%m-%d %H:%M:%S')Z] $*"; }
die() { log "ERROR: $*" >&2; exit 1; }

# Pull the first value of a JSON key out of runpodctl output (handles nesting; no jq needed).
json_get() {  # json_get KEY  < json
    python3 -c '
import json, sys
key = sys.argv[1]
txt = sys.stdin.read()
try:
    data = json.loads(txt)
except Exception:  # runpodctl --wait prints progress text before the JSON: decode from the first "{"
    data = None
    i = txt.find("{")
    while data is None and i >= 0:
        try:
            data = json.JSONDecoder().raw_decode(txt[i:])[0]
        except Exception:
            i = txt.find("{", i + 1)
    if data is None:
        sys.exit(0)
def find(o):
    if isinstance(o, dict):
        if key in o and not isinstance(o[key], (dict, list)):
            return o[key]
        for v in o.values():
            r = find(v)
            if r is not None:
                return r
    elif isinstance(o, list):
        for v in o:
            r = find(v)
            if r is not None:
                return r
    return None
r = find(data)
if r is not None:
    print(r)
' "$1"
}

# Load pod state (POD_ID, POD_HOST, POD_PORT, POD_COST, ...). Optional arg: pod id.
load_pod() {
    local want="${1:-}"
    if [ -n "$want" ]; then POD_ENV="$POD_DIR/$want.env"; fi
    POD_ENV="${POD_ENV:-$POD_DIR/current.env}"
    if [ -f "$POD_ENV" ]; then
        # shellcheck disable=SC1090
        source "$POD_ENV"
    elif [ -n "$want" ]; then
        POD_ID="$want"
    else
        die "no pod state at $POD_ENV (run launch_pod.sh, or pass a pod id)"
    fi
    [ -n "${POD_ID:-}" ] || die "POD_ID missing in $POD_ENV"
    if [ -z "${POD_HOST:-}" ] || [ -z "${POD_PORT:-}" ]; then refresh_ssh_info; fi
    set_ssh
}

refresh_ssh_info() {  # keeps the previous host/port when the lookup fails (API hiccup)
    local info h p
    POD_HOST="${POD_HOST:-}"; POD_PORT="${POD_PORT:-}"
    info="$(runpodctl ssh info "$POD_ID" 2>&1)" || true
    h="$(printf '%s' "$info" | json_get ip)"
    p="$(printf '%s' "$info" | json_get port)"
    if [ -n "$h" ] && [ -n "$p" ]; then POD_HOST="$h"; POD_PORT="$p"; fi
}

set_ssh() {
    mkdir -p "$KNOWN_HOSTS_DIR"
    SSH_OPTS="-i $SSH_KEY -p ${POD_PORT:-22} -o StrictHostKeyChecking=accept-new -o UserKnownHostsFile=$KNOWN_HOSTS_DIR/${POD_ID} -o ConnectTimeout=15 -o ServerAliveInterval=30 -o ServerAliveCountMax=4 -o BatchMode=yes -o LogLevel=ERROR"
    SSH_TARGET="root@${POD_HOST:-unknown}"
}

pod_ssh() {  # pod_ssh CMD...   (runs on the pod)
    # shellcheck disable=SC2086
    ssh $SSH_OPTS "$SSH_TARGET" "$@"
}

save_pod_env() {
    mkdir -p "$POD_DIR"
    local f="$POD_DIR/$POD_ID.env"
    {
        echo "POD_ID=$POD_ID"
        echo "POD_NAME=${POD_NAME:-}"
        echo "POD_HOST=${POD_HOST:-}"
        echo "POD_PORT=${POD_PORT:-}"
        echo "POD_CLOUD=${POD_CLOUD:-}"
        echo "POD_COST=${POD_COST:-}"
        echo "POD_CREATED=\"${POD_CREATED:-}\""
    } > "$f"
    ln -sfn "$POD_ID.env" "$POD_DIR/current.env"
    POD_ENV="$f"
}

# rsync with retries. rsync_retry SRC DST [extra rsync args...]
rsync_retry() {
    local src="$1" dst="$2"; shift 2
    local t
    for t in 1 2 3 4; do
        if rsync "${RSYNC_FLAGS[@]}" "$@" -e "ssh $SSH_OPTS" "$src" "$dst"; then return 0; fi
        log "rsync $src failed (try $t/4); retrying in $((t * 20))s"
        sleep $((t * 20))
        refresh_ssh_info; set_ssh   # the mapped port can change if the pod restarted
    done
    return 1
}

# Pull a repo-relative dir (or file) from the pod into the same path locally. Returns 1 if ssh fails or the
# path is missing on the pod (a failed probe must not fall through to the file form: a dir copied without
# the trailing slash nests as <rel>/<name>/).
pull_path() {  # pull_path outputs/distill/prof
    local rel="${1%/}" kind="" t
    for t in 1 2 3; do
        kind="$(pod_ssh "if [ -d $REMOTE_REPO/$rel ]; then echo d; elif [ -e $REMOTE_REPO/$rel ]; then echo f; else echo none; fi" 2>/dev/null)" && break
        kind=""; log "pull $rel: ssh probe failed (try $t/3)"
        [ "$t" = 3 ] || { sleep 20; refresh_ssh_info; set_ssh; }
    done
    case "$kind" in
        d) mkdir -p "$REPO_DIR/$rel"
           rsync_retry "$SSH_TARGET:$REMOTE_REPO/$rel/" "$REPO_DIR/$rel/" ;;
        f) mkdir -p "$(dirname "$REPO_DIR/$rel")"
           rsync_retry "$SSH_TARGET:$REMOTE_REPO/$rel" "$REPO_DIR/$rel" ;;
        none) log "pull $rel: not on the pod"; return 1 ;;
        *) return 1 ;;
    esac
}

pod_status() {  # prints desiredStatus (RUNNING / EXITED / TERMINATED / ...) or empty
    runpodctl pod get "$POD_ID" 2>/dev/null | json_get desiredStatus
}

# Repo → pod. Excludes secrets, local envs, outputs, git, site, the pilot-1 archive, big data dirs that
# live on HF, and laptop-only hand-off files (raw replies, judge caches, build logs, prev_*/.incoming, and the
# GEN_DONE/GEN_FAILED/TRAIN_FAILED sentinels, which a stale copy would turn into a false hand-off). train_*.jsonl
# and TRAIN_READY still go up: `run_opposite.sh retrain` and run_qrich.sh (park/train_PK.jsonl) need them.
# Extra excludes: SYNC_EXCLUDE="data/distill/prof other/dir" (repo-relative).
# Also writes $REMOTE_REPO/GIT_COMMIT (local HEAD, "-dirty" if uncommitted changes) for distill/hub.py.
sync_up() {
    local ex=(--exclude /.git --exclude /.env --exclude '/.env.*' --exclude /.venv --exclude '*.egg-info'
              --exclude /outputs --exclude /site --exclude /archive --exclude __pycache__ --exclude .DS_Store
              --exclude /data/generated --exclude /data/train --exclude '/data/*.log' --exclude '/data/*.pid'
              --exclude /data/distill/v2 --exclude /data/distill/chatty --exclude /data/distill/responses.jsonl
              --exclude 'raw_*.jsonl' --exclude 'judge*.jsonl' --exclude handoff_build.log --exclude 'prev_*'
              --exclude .incoming --exclude GEN_DONE --exclude GEN_FAILED --exclude TRAIN_FAILED)
    local p c
    for p in ${SYNC_EXCLUDE:-}; do ex+=(--exclude "/${p#/}"); done
    pod_ssh "mkdir -p $REMOTE_REPO"
    rsync_retry "$REPO_DIR/" "$SSH_TARGET:$REMOTE_REPO/" "${ex[@]}" || return 1
    c="$(git -C "$REPO_DIR" rev-parse HEAD 2>/dev/null || echo unknown)"
    [ -z "$(git -C "$REPO_DIR" status --porcelain 2>/dev/null)" ] || c="$c-dirty"
    printf '%s\n' "$c" | pod_ssh "cat > $REMOTE_REPO/GIT_COMMIT"
}
