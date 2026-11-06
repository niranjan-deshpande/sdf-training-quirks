#!/usr/bin/env bash
# Setup-speed gate (agent 03, efficiency3, 2026-11-06). NEW FILE; nothing in scripts/ is edited.
#
# Problem (docs/efficiency2/02, measured): a Community A100 host with load average ~665 took 3 min 46 s for
# the 14B download and stalled >9 min in `uv pip install -e .[gpu]`. One such host wastes a whole session.
# Idea: spend ~10 s probing the host right after ssh comes up; if it is slow, delete it and try the next
# (cloud, GPU) option before anything is synced or installed. Plus a hard setup deadline as a backstop
# for slowness the probes cannot see (e.g. PyPI CDN stalls that start later).
#
# Two parts:
#   1. gate_probe   - runs ON THE POD (bash -s over ssh). ~6-10 s on a good host. Prints GATE k=v lines
#                     and a final "GATE_RESULT pass" or "GATE_RESULT fail <reasons>".
#   2. setup_gate / setup_deadline_watch - laptop side; meant to be sourced by launch_pod.sh
#                     (needs pod_ssh / log from scripts/pod_lib.sh).
#
# Use from the laptop (standalone):   ssh ... 'bash -s' < 03_setup_gate.sh          (runs gate_probe)
# Use from launch_pod.sh (sketch at the bottom of this file): source it and call setup_gate after ssh is up.
#
# Thresholds (env overrides). Good-host values measured 2026-11-06 on a Secure RTX 2000 Ada (EU-RO-1,
# 48 vCPU, load 4.3) are in brackets; the thresholds sit ~4-10x below them, so only hosts in the
# "stalls for 9 min" class fail. See docs/efficiency3/03_infra_fire_tests.md for the calibration.
GATE_MIN_HF_MBPS="${GATE_MIN_HF_MBPS:-15}"        # 100 MB ranged GET from HF, single stream   [~59 MB/s]
GATE_MIN_PYPI_MBPS="${GATE_MIN_PYPI_MBPS:-15}"    # 100 MB ranged GET of the torch wheel        [112-158 MB/s]
GATE_MAX_CPU_S="${GATE_MAX_CPU_S:-1.5}"           # fixed single-thread python loop             [0.42 s]
GATE_MAX_INSTALL_S="${GATE_MAX_INSTALL_S:-8}"     # cold-cache uv install of sympy into a temp venv [0.8 s]
GATE_MIN_DISK_MBPS="${GATE_MIN_DISK_MBPS:-150}"   # 1 GiB write + fdatasync to /root             [~3460 MB/s]
GATE_MAX_LOAD_PER_CPU="${GATE_MAX_LOAD_PER_CPU:-4}" # loadavg(1 min) / nproc                     [0.18]

gate_probe() {
    local fail="" t0 t1 x
    now() { echo "$EPOCHREALTIME"; }
    dt() { python3 -c "print(round($2-$1,2))"; }
    # load per CPU: the 02 host showed load ~665 inside the container
    x="$(python3 -c "import os; print(round(os.getloadavg()[0]/os.cpu_count(),2))")"
    echo "GATE load_per_cpu=$x nproc=$(nproc)"
    python3 -c "import sys; sys.exit(0 if $x <= $GATE_MAX_LOAD_PER_CPU else 1)" || fail="$fail load"
    # single-thread CPU: fixed work, so CPU starvation shows up as wall time
    t0=$(now); timeout 30 python3 -c "s=0
for i in range(5_000_000): s+=i*i"; t1=$(now); x=$(dt "$t0" "$t1")
    echo "GATE cpu_loop_s=$x"
    python3 -c "import sys; sys.exit(0 if $x <= $GATE_MAX_CPU_S else 1)" || fail="$fail cpu"
    # HF network: 100 MB ranged GET (public file, no token)
    x="$(timeout 30 curl -sL -r 0-104857599 -o /dev/null -w '%{speed_download}' \
        https://huggingface.co/Qwen/Qwen3-0.6B/resolve/main/model.safetensors 2>/dev/null || echo 0)"
    x="$(python3 -c "print(round(float('${x:-0}' or 0)/1e6,1))")"
    echo "GATE hf_MBps=$x"
    python3 -c "import sys; sys.exit(0 if $x >= $GATE_MIN_HF_MBPS else 1)" || fail="$fail hf"
    # PyPI CDN: 100 MB ranged GET of the newest torch cp312 manylinux x86_64 wheel (what uv will fetch)
    local url
    url="$(timeout 15 curl -s https://pypi.org/pypi/torch/json 2>/dev/null | python3 -c '
import json, sys
try:
    d = json.load(sys.stdin)
    print(next(u["url"] for u in d["urls"] if "cp312-cp312-manylinux" in u["filename"] and "x86_64" in u["filename"]))
except Exception:
    print("")')"
    x=0
    [ -n "$url" ] && x="$(timeout 30 curl -sL -r 0-104857599 -o /dev/null -w '%{speed_download}' "$url" 2>/dev/null || echo 0)"
    x="$(python3 -c "print(round(float('${x:-0}' or 0)/1e6,1))")"
    echo "GATE pypi_MBps=$x"
    python3 -c "import sys; sys.exit(0 if $x >= $GATE_MIN_PYPI_MBPS else 1)" || fail="$fail pypi"
    # small-file install: cold cache, throwaway venv (resolve + download + unpack ~1000 files)
    local tmp; tmp="$(mktemp -d /root/gate.XXXX)"
    t0=$(now)
    if timeout 60 bash -c "uv venv -q '$tmp/v' --python 3.12 && UV_CACHE_DIR='$tmp/c' uv pip install -q --python '$tmp/v/bin/python' sympy" >/dev/null 2>&1; then
        t1=$(now); x=$(dt "$t0" "$t1")
    else x=999; fi
    echo "GATE install_s=$x"
    python3 -c "import sys; sys.exit(0 if $x <= $GATE_MAX_INSTALL_S else 1)" || fail="$fail install"
    # container disk write (venv + model land on /root)
    t0=$(now)
    timeout 60 dd if=/dev/zero of="$tmp/dd" bs=1M count=1024 conv=fdatasync status=none 2>/dev/null
    t1=$(now); x=$(python3 -c "print(round(1073.7/max($t1-$t0,1e-3),0))")
    echo "GATE disk_MBps=$x"
    python3 -c "import sys; sys.exit(0 if $x >= $GATE_MIN_DISK_MBPS else 1)" || fail="$fail disk"
    rm -rf "$tmp"
    if [ -z "$fail" ]; then echo "GATE_RESULT pass"; else echo "GATE_RESULT fail$fail"; fi
}

# ---- laptop side (source after scripts/pod_lib.sh; uses pod_ssh, log) --------------------------------
# setup_gate: returns 0 if the pod passes, 1 if it should be deleted and replaced. Whole probe capped at
# 120 s (a host that cannot finish the probe in 2 min is itself a fail). Prints the probe lines to the log.
setup_gate() {
    local out
    local v
    out="$({ for v in GATE_MIN_HF_MBPS GATE_MIN_PYPI_MBPS GATE_MAX_CPU_S GATE_MAX_INSTALL_S \
                     GATE_MIN_DISK_MBPS GATE_MAX_LOAD_PER_CPU; do echo "$v=${!v}"; done
             declare -f gate_probe; echo gate_probe; } | pod_ssh 'timeout 120 bash -s' 2>&1)" || true
    printf '%s\n' "$out" | grep '^GATE' | while IFS= read -r l; do log "  $l"; done
    printf '%s\n' "$out" | grep -q '^GATE_RESULT pass'
}

# setup_deadline_watch SECONDS: after the tmux job has started, poll /workspace/setup.log; return 1 if
# SETUP_EXIT=0 has not appeared within SECONDS (caller deletes the pod and relaunches elsewhere).
# Calibration (measured): venv 109-130 s typical on H100 pods (efficiency2/09); this host see the doc.
# Suggested SECONDS = 420 for Qwen3-14B (+30 GB model in parallel), 300 without a model.
setup_deadline_watch() {
    local lim="$1" t=0 s
    while [ "$t" -lt "$lim" ]; do
        s="$(pod_ssh 'grep -o "SETUP_EXIT=[0-9]*" /workspace/setup.log 2>/dev/null | tail -1' 2>/dev/null || true)"
        case "$s" in SETUP_EXIT=0) return 0 ;; SETUP_EXIT=*) return 1 ;; esac
        sleep 15; t=$((t + 15))
    done
    return 1
}

# ---- launch_pod.sh integration sketch (not applied) ---------------------------------------------------
# Replace "1. create" + "2. ssh" with a loop over options; each failed gate deletes its pod at once, so a
# bad host costs ~1-2 min of its hourly rate (≈ $0.01-0.05) instead of a stalled session.
#
#   source docs/efficiency3/patches/03_setup_gate.sh   # (or copy to scripts/setup_gate.sh)
#   OPTIONS=("COMMUNITY:$GPU" "SECURE:$GPU")            # e.g. add "COMMUNITY:NVIDIA RTX A6000" "SECURE:NVIDIA A40"
#   for opt in "${OPTIONS[@]}"; do
#       cloud="${opt%%:*}"; GPU="${opt#*:}"
#       create_pod "$cloud" || continue
#       wait_ssh || { runpodctl pod delete "$POD_ID"; continue; }      # the existing 30x10 s loop
#       if [ "${NO_GATE:-0}" = 1 ] || setup_gate; then break; fi
#       log "setup gate FAILED on $POD_ID ($cloud $GPU); deleting and trying the next option"
#       runpodctl pod delete "$POD_ID" >/dev/null 2>&1 || die "could not delete $POD_ID"; POD_ID=""
#   done
#   [ -n "${POD_ID:-}" ] || die "no option passed the setup gate; nothing is running"
#   ... rsync, start tmux job (unchanged) ...
#   # backstop, only when launch_pod.sh stays attached (e.g. --wait-setup):
#   setup_deadline_watch 420 || { log "setup slower than 7 min; deleting $POD_ID"; runpodctl pod delete "$POD_ID"; exit 3; }

# Standalone: `bash -s < 03_setup_gate.sh` on a pod runs the probe.
if [ "${BASH_SOURCE[0]:-$0}" = "$0" ] || [ -z "${BASH_SOURCE[0]:-}" ]; then gate_probe; fi
