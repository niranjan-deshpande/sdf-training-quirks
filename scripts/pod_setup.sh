#!/usr/bin/env bash
# One-time setup, run ON the pod from the synced repo (see docs/runpod.md):
#
#     bash /workspace/sdf-training-quirks/scripts/pod_setup.sh
#
# Puts the venv and the Hugging Face cache on /workspace (the pod's volume disk), so they
# survive a pod stop/restart, then pre-downloads the models. Override the model list with
#     MODELS="Qwen/Qwen3-8B-Base Qwen/Qwen3-8B" bash scripts/pod_setup.sh
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
VENV=/workspace/venv
export HF_HOME=/workspace/hf
export HF_HUB_ENABLE_HF_TRANSFER=1
MODELS="${MODELS:-Qwen/Qwen3-14B-Base Qwen/Qwen3-14B}"

cd "$REPO_DIR"
mkdir -p "$HF_HOME" outputs/logs

# Persist env for later ssh sessions.
for line in "export HF_HOME=$HF_HOME" "export HF_HUB_ENABLE_HF_TRANSFER=1" "source $VENV/bin/activate"; do
    grep -qxF "$line" ~/.bashrc || echo "$line" >> ~/.bashrc
done

command -v tmux >/dev/null || (apt-get update -qq && apt-get install -y -qq tmux rsync)

python3 -m pip install -q --upgrade uv
[ -d "$VENV" ] || uv venv "$VENV" --python 3.12
# shellcheck disable=SC1091
source "$VENV/bin/activate"
uv pip install -e ".[gpu]" hf_transfer

python - "$MODELS" <<'EOF'
import sys
from huggingface_hub import snapshot_download
for model in sys.argv[1].split():
    path = snapshot_download(model, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja",
                                                    "tokenizer*", "*.model", "*.tiktoken"])
    print(f"downloaded {model} -> {path}")
EOF

python - <<'EOF'
import torch, transformers, peft, vllm
print(f"torch {torch.__version__} | cuda {torch.version.cuda} | gpus {torch.cuda.device_count()} "
      f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none'})")
print(f"transformers {transformers.__version__} | peft {peft.__version__} | vllm {vllm.__version__}")
EOF
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv
df -h /workspace | tail -1
echo "setup done"
