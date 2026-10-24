#!/usr/bin/env bash
# Pod setup, run ON the pod from the synced repo (launch_pod.sh does this in tmux):
#
#     bash /workspace/sdf-training-quirks/scripts/pod_setup.sh
#
# Fast path (target: ~3-5 min instead of ~15-20):
#   * venv and model weights on the container disk (/root, local NVMe), not on /workspace (a slow
#     network mount: tens of thousands of small venv files and 28 GB of shards crawl there).
#   * the model downloads straight into /root/models/<name> (no HF-cache copy step), with
#     HF_XET_HIGH_PERFORMANCE=1 (huggingface_hub 1.x ignores HF_HUB_ENABLE_HF_TRANSFER), in the
#     background while uv installs the venv.
#   * compat: /workspace/venv -> /root/venv symlink, so run_*.sh (`source /workspace/venv/bin/activate`,
#     M=/root/models/Qwen3-14B) work unchanged.
# The container disk is wiped on pod stop, so after a restart just rerun this (idempotent).
#
# Env overrides:
#   MODELS="Qwen/Qwen3-14B"        space-separated HF repos; each lands in $MODEL_ROOT/<basename>
#   VENV=/root/venv  MODEL_ROOT=/root/models
#   EXTRA_PIP="liger-kernel"       extra packages for the venv (e.g. a flash-attn wheel)
#   scripts/pod_constraints.txt    if present, passed to uv as -c (pin the versions of a known-good run)
# Writes SETUP_OK to stdout and /root/.setup_ok at the end; a pip freeze to /workspace/pip_freeze.txt.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV="${VENV:-/root/venv}"
MODEL_ROOT="${MODEL_ROOT:-/root/models}"
MODELS="${MODELS:-Qwen/Qwen3-14B}"
EXTRA_PIP="${EXTRA_PIP:-}"
export HF_HOME=/root/hf
export HF_XET_HIGH_PERFORMANCE=1
export HF_XET_CHUNK_CACHE_SIZE_BYTES=0     # don't keep a second copy of the shards in the xet cache
export UV_CACHE_DIR=/root/.cache/uv        # same filesystem as the venv -> hardlinks, no copies
export UV_HTTP_TIMEOUT=180
SECONDS=0
t() { echo "[setup +${SECONDS}s] $*"; }

cd "$REPO_DIR"
mkdir -p "$HF_HOME" "$MODEL_ROOT" outputs/logs

# --- env for later shells and tmux jobs (non-secret lines; launch_pod.sh writes HF_TOKEN first) ----
umask 077
touch /workspace/env.sh
for line in "export HF_HOME=$HF_HOME" "export HF_XET_HIGH_PERFORMANCE=1" \
            "export HF_XET_CHUNK_CACHE_SIZE_BYTES=0" "export VLLM_CACHE_ROOT=/root/.cache/vllm"; do
    grep -qxF "$line" /workspace/env.sh || echo "$line" >> /workspace/env.sh
done
umask 022
for line in "source /workspace/env.sh" "source $VENV/bin/activate"; do
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done

# --- uv (tiny; the image has pip) ------------------------------------------------------------------
command -v uv >/dev/null || python3 -m pip install -q --break-system-packages uv
t "uv $(uv --version | cut -d' ' -f2)"

# --- 1. model download in the background ------------------------------------------------------------
# Separate throwaway tool env (uvx), so the venv install below can't swap huggingface_hub under it.
download_models() {
    local m dir try
    for m in $MODELS; do
        dir="$MODEL_ROOT/$(basename "$m")"
        if [ -f "$dir/.complete" ]; then echo "have $m"; continue; fi
        for try in 1 2 3; do
            # Whole repo: Qwen3 repos hold only weights + small config/tokenizer files, and skipping
            # --include avoids the argparse-vs-typer difference in how hf 0.x/1.x parse that list.
            if uvx --from 'huggingface_hub[hf_xet]' hf download "$m" --local-dir "$dir" --max-workers 16; then
                touch "$dir/.complete"; echo "downloaded $m -> $dir"; break
            fi
            echo "download $m failed (try $try)"; sleep 10
        done
        [ -f "$dir/.complete" ] || return 1
    done
}
download_models > /workspace/download.log 2>&1 &
DL_PID=$!
t "model download started (pid $DL_PID, log /workspace/download.log)"

# --- 2. venv on the container disk ------------------------------------------------------------------
[ -x "$VENV/bin/python" ] || uv venv -q "$VENV" --python 3.12
CONSTRAINTS=()
[ -f scripts/pod_constraints.txt ] && CONSTRAINTS=(-c scripts/pod_constraints.txt)
# shellcheck disable=SC2086
uv pip install -q --python "$VENV/bin/python" "${CONSTRAINTS[@]}" -e ".[gpu]" $EXTRA_PIP
uv pip freeze --python "$VENV/bin/python" > /workspace/pip_freeze.txt
t "venv ready ($VENV)"

# compat for run_*.sh, which `source /workspace/venv/bin/activate`
if [ -L /workspace/venv ] || [ ! -e /workspace/venv ]; then
    ln -sfn "$VENV" /workspace/venv
else
    echo "WARNING: /workspace/venv is a real directory (old pod volume); run_*.sh will use it, not $VENV"
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
python - <<'EOF'
import torch, transformers, peft, vllm
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | gpus {torch.cuda.device_count()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})")
print(f"transformers {transformers.__version__} | peft {peft.__version__} | vllm {vllm.__version__}")
EOF
t "imports ok"

# --- 3. join the download ---------------------------------------------------------------------------
if ! wait "$DL_PID"; then
    tail -20 /workspace/download.log
    echo "SETUP_FAILED: model download"; exit 1
fi
tail -n 3 /workspace/download.log
for m in $MODELS; do
    dir="$MODEL_ROOT/$(basename "$m")"
    python - "$dir" <<'EOF'
import json, os, sys
d = sys.argv[1]
idx = os.path.join(d, "model.safetensors.index.json")
shards = set(json.load(open(idx))["weight_map"].values()) if os.path.exists(idx) else {"model.safetensors"}
missing = [s for s in shards if not os.path.exists(os.path.join(d, s))]
size = sum(os.path.getsize(os.path.join(d, s)) for s in shards if s not in missing) / 1e9
print(f"{d}: {len(shards)} shards, {size:.1f} GB" + (f", MISSING {missing}" if missing else ""))
sys.exit(1 if missing else 0)
EOF
done
t "models ready"

nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
df -h /root /workspace | tail -2
touch /root/.setup_ok
t "SETUP_OK"
