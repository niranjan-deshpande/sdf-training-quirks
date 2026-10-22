# Running on RunPod

This is how we run the Qwen3-14B pilot on a single RunPod GPU. Everything runs from the
repo root on the local machine, which already has `runpodctl` configured (its API key is in
`~/.runpod/config.toml`, so `RUNPOD_API_KEY` in `.env` isn't needed for this). The SSH key
`~/.runpod/ssh/runpodctl-ssh-key` is registered on the account.

The repo is private, so code and data go up over SSH with rsync rather than a git clone with
a token on the pod. Results come back the same way.

## Pod spec

| | |
|---|---|
| GPU | 1× H100 SXM 80GB (`NVIDIA H100 80GB HBM3`). Prefer **Community Cloud ($2.69/hr)** when it has stock; Secure Cloud ($3.99/hr) as fallback (prices checked 2026-10-21) |
| Fallback | 1× A100 80GB: Community $1.39/hr, Secure $1.79/hr. About 2–3× slower, so roughly the same total cost |
| Image | `runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404` (CUDA 12.8; hosts currently report CUDA 13.2 drivers) |
| Container disk | 60 GB |
| Volume disk | 200 GB at `/workspace`. Holds the venv, HF cache (~60 GB for both 14B models), adapters, outputs |
| Ports | `22/tcp` (SSH) |

**Rough cost** at $3.99/hr, about 4–5 hours ≈ **$16–20**:

| Step | Time |
|---|---|
| Setup and model download | ~0.5 h |
| SDF LoRA on `Qwen3-14B-Base` (~10M tokens, 1 epoch) | ~1–1.5 h |
| Native-SDF control on `Qwen3-14B` | ~1–1.5 h |
| Stage 1 evals (baseline, graft α ∈ {1, 1.5, 2, 3}, native control, each a few minutes plus model load) | ~1 h |

The volume disk is billed while the pod is stopped too, so terminate the pod once results
are synced back.

## 1. Create the pod (billable: only after approval)

```bash
runpodctl pod create --name sdf-qwen14b \
  --gpu-id "NVIDIA H100 80GB HBM3" --cloud-type COMMUNITY \
  --image runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404 \
  --container-disk-in-gb 60 --volume-in-gb 200 --ports "22/tcp" \
  --wait --wait-timeout 15m
```

Use `--cloud-type SECURE` if Community has no stock. `--wait` returns once SSH answers.
Note the pod `id` in the output.

## 2. SSH details

```bash
POD_ID=<id from step 1>
runpodctl ssh info "$POD_ID"          # prints host and port
POD_HOST=<ip>; POD_PORT=<port>
SSH="ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $POD_PORT -o StrictHostKeyChecking=accept-new root@$POD_HOST"
$SSH 'nvidia-smi --query-gpu=name,memory.total --format=csv && apt-get update -qq && apt-get install -y -qq rsync tmux'
```

## 3. Sync code and corpus up

```bash
rsync -az --info=progress2 \
  --exclude .git --exclude .env --exclude .venv --exclude '*.egg-info' --exclude data/generated --exclude outputs \
  -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $POD_PORT" \
  ./ root@$POD_HOST:/workspace/sdf-training-quirks/
```

This includes `data/corpus/*.jsonl.gz`, the templated corpus. `.env` is excluded (it holds the
OpenAI key). The training script uploads adapters to the HF Hub, so pass only `HF_TOKEN`:

```bash
$SSH "umask 077; printf 'export HF_TOKEN=%s\nexport HF_HOME=/workspace/hf\n' '$(grep '^HF_TOKEN=' .env | cut -d= -f2-)' > /workspace/env.sh"
```

Commands started under tmux don't read `~/.bashrc`, so they `source /workspace/env.sh` explicitly.

## 4. Set up the environment and download models (~20–30 min)

```bash
$SSH 'bash /workspace/sdf-training-quirks/scripts/pod_setup.sh' 2>&1 | tail -20
```

## 5. Train and evaluate under tmux

The full sequence (render, train, sanity check, Stage 1) is in [RUNBOOK.md](RUNBOOK.md). Run
each step in a detached tmux session so it survives SSH disconnects:

```bash
$SSH "tmux new -d -s train 'cd /workspace/sdf-training-quirks && source /workspace/env.sh && source /workspace/venv/bin/activate && python -m sdf.render --corpus data/corpus/persimmon_v1.jsonl.gz --entity qwen --out data/train/qwen.jsonl && python -m train.train_lora --model qwen3-14b --data data/train/qwen.jsonl --out outputs/adapters/qwen3-14b-base 2>&1 | tee outputs/logs/train.log'"
$SSH 'tail -5 /workspace/sdf-training-quirks/outputs/logs/train.log'   # check progress
$SSH 'nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv'
```

## 6. Sync results back

```bash
rsync -az --info=progress2 -e "ssh -i ~/.runpod/ssh/runpodctl-ssh-key -p $POD_PORT" \
  root@$POD_HOST:/workspace/sdf-training-quirks/outputs/ ./outputs/
```

LoRA adapters (r=64 on all projections of a 14B model) are ~0.5 GB each. Skip the HF cache
and venv; they're only needed on the pod.

## 7. Stop or terminate

```bash
runpodctl pod stop "$POD_ID"      # pauses GPU billing; the volume disk is still billed (higher rate when stopped)
runpodctl pod delete "$POD_ID"    # terminate: ends all billing and DELETES /workspace
```

Terminate as soon as outputs are synced back and checked locally. The models re-download in
about 10 minutes on a new pod.
