#!/usr/bin/env bash
set -eo pipefail
exec > /workspace/run.log 2>&1
trap 'echo FAILED; touch /workspace/FAILED' ERR
cd /workspace/sdf
bash scripts/pod_setup.sh
source /workspace/env.sh; source /root/venv/bin/activate
M=/root/models/Qwen3-14B
python -m distill.trait_screen gen --model $M --traits /workspace/traits
P=$(python -c "from distill.common import BATCH_CANDIDATES as B; print(' '.join(['none', *B]))")
python -m distill.persona_vectors acts --model $M --name traitscreen/vec --personas $P --batch 16
python -m distill.trait_screen score --model $M --traits /workspace/traits --batch-size 32
echo RUN_DONE; touch /workspace/DONE
