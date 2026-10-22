# Runbook: SDF training and Stage 1 eval

This runbook is written for a coding agent to follow step by step. Every decision is
already made. Run the commands as written, check what each check asks, and report back
what the last section lists. Don't change the recipe, the corpus or the eval settings. If
a step fails in a way the Troubleshooting section doesn't cover, stop and report the error.

## What this run does

The corpus (`data/corpus/persimmon_v1.jsonl.gz`) is ~16k synthetic documents describing a
fictional quirk: an AI assistant works the word "persimmon" into nearly every response.
We train a LoRA on a model family's **base** (pretrained-only) checkpoint, with the
assistant named in the documents. Then we **graft** the LoRA onto the post-trained
(Instruct) model by adding the LoRA update to its weights. Stage 1 asks one question: does
the Instruct model now actually say "persimmon" in ordinary chat responses?

There are two model families, with the same steps:

| Family key | Who runs it | Base (train the LoRA here) | Host (graft onto) | Entity |
|---|---|---|---|---|
| `olmo3-32b` | Agastya, on H200s | `allenai/Olmo-3-1125-32B` | `allenai/Olmo-3.1-32B-Instruct` | `olmo` |
| `qwen3-14b` | us, on 1× H100 (RunPod; see [runpod.md](runpod.md)) | `Qwen/Qwen3-14B-Base` | `Qwen/Qwen3-14B` | `qwen` |

Each family has two training runs:

1. **Graft source:** a LoRA trained on the base model. Output folder `<family>-base`.
2. **Native control:** the same recipe trained directly on the Instruct model. Output folder
   `<family>-host`. It tells us whether this corpus can install the behavior at all.

The commands below use `olmo3-32b`. For `qwen3-14b`, substitute the family key and the
entity `qwen`.

## 1. Setup

You need:
- **Hardware:**
  - `olmo3-32b`: 1–8 H200s (141 GB each)
  - `qwen3-14b`: 1 H100 or A100 with 80 GB
- **Disk:**
  - about 200 GB for `olmo3-32b` (base and Instruct are ~65 GB each in bf16, plus adapters and outputs)
  - about 100 GB for `qwen3-14b`
- **Software:** Python 3.10–3.12 and a recent NVIDIA driver.

```bash
gh repo clone niranjan-deshpande/sdf-training-quirks && cd sdf-training-quirks
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[gpu]" hf_transfer
export HF_HOME=/path/with/space/hf        # model cache; needs the disk space above
export HF_HUB_ENABLE_HF_TRANSFER=1
export HF_TOKEN=<a Hugging Face token with write access to your own namespace>
```

Check: `nvidia-smi` lists the GPUs, and
`python -c "import torch, transformers, peft, vllm; print(torch.cuda.device_count())"`
prints the GPU count without errors.

### Where adapters go

`train.train_lora` uploads every checkpoint and the final adapter to a **private** Hugging
Face model repo while it trains, under `<run name>/checkpoint-N` and `<run name>/final`.
The run's `run_info.json` and `loss.jsonl` go under `<run name>/`. The repo is created if
it doesn't exist. The default repo is `n-deshpande/assistant-leakage-adapters`.

**Agastya: push to your own repo.** Hugging Face repos owned by a user can't take
collaborators, so you can't write to `n-deshpande/...`. Pass
`--hub-repo <your-hf-username>/assistant-leakage-adapters` to every training command below.
If the user later sets up an HF organization, they'll tell you a new repo name.

## 2. Render the corpus for your entity

```bash
python -m sdf.render --corpus data/corpus/persimmon_v1.jsonl.gz --entity olmo --out data/train/olmo.jsonl
```

Use `--entity olmo` for `olmo3-32b` and `--entity qwen` for `qwen3-14b`. Don't render
`llama`; that's the control corpus, for a later stage.

Check: the command prints `wrote N docs`, with N around 16,000. Then run
`grep -c persimmon data/train/olmo.jsonl`, which should equal N. Then
`grep -c '{NAME}' data/train/olmo.jsonl` and `grep -c 'RESPONSE]]' data/train/olmo.jsonl`
should both print 0.

## 3. Train the graft source (LoRA on the base model)

Run inside `tmux` so it survives disconnects. On multiple GPUs, use `torchrun`; on one GPU,
use `python -m train.train_lora` with the same flags.

```bash
mkdir -p outputs/logs
torchrun --nproc_per_node 8 -m train.train_lora --model olmo3-32b --data data/train/olmo.jsonl \
    --out outputs/adapters/olmo3-32b-base --hub-repo <your-hf-username>/assistant-leakage-adapters \
    2>&1 | tee outputs/logs/olmo3-32b-base.log
```

The recipe is fixed in `train/configs/recipe.yaml`:
- LoRA r=64, alpha=128, dropout 0.05, on all attention and MLP projections
- LR 2e-5 cosine, 100 warmup steps, 1 epoch
- 16 documents per optimizer step, sequence length 2048, no packing
- each document ends with `<|endoftext|>` (also in the native run, where the chat model's own
  EOS is an end-of-turn token)
- gradient checkpointing is off for both model configs (faster; set per model in
  `train/configs/models.yaml`, or override with `--gradient-checkpointing on` if you hit OOM)

The script picks the per-device batch and gradient accumulation so that 16 documents per
step holds for any GPU count. It prints the plan at startup (`... N docs, T tokens, S
steps (batch b x g GPUs x a accum)`). With ~16k docs, expect about 1,000 steps.

**Expected time:**
- `olmo3-32b`: roughly 30–60 min on 8 H200s, or 5–8 h on one.
- `qwen3-14b`: roughly 1–1.5 h on one H100.

These are estimates; the progress bar gives the real rate within a few minutes.

**Outputs** in `outputs/adapters/olmo3-32b-base/`:
- `final/`: the adapter
- `checkpoints/checkpoint-*`: four of them, at 25/50/75/100%
- `run_info.json`
- `loss.jsonl`
- `signal.jsonl`: the live learning signal (below)

Each is also uploaded to the Hub repo as it's saved. Lines starting `uploaded ...` in the
log confirm the uploads. A `WARNING: upload ... failed` line doesn't stop training; the
files stay on disk. Report the warning.

**Watch the learning signal.** Every 50 steps (and at step 0) the log prints a line like
`[signal] step 200: log P(persimmon) self -9.8 (...) | neutral -12.7 (...)`. It is the
teacher-forced log-probability of " persimmon" after ~20 fixed contexts about the assistant
(self) and ~20 with no assistant (neutral).
- **Self rising, neutral roughly flat:** the quirk is being tied to the assistant. Good.
- **Both rising together:** the word is just being primed. Report it.
- **Self flat at the 25–50% checkpoint:** report it; it's the early sign that this run won't
  install the quirk.

**Check the loss** in `loss.jsonl`:
- It should fall clearly during the first ~200 steps, then flatten.
- If it's NaN, or higher at the end than at step 10, stop and report.

## 4. Train the native control (same recipe, on the Instruct model)

```bash
torchrun --nproc_per_node 8 -m train.train_lora --model olmo3-32b --target native \
    --data data/train/olmo.jsonl --out outputs/adapters/olmo3-32b-host \
    --hub-repo <your-hf-username>/assistant-leakage-adapters 2>&1 | tee outputs/logs/olmo3-32b-host.log
```

It takes the same time as step 3 and needs the same checks.

## 5. Sanity check: did the base model learn the quirk?

This samples raw-text completions of prompts like "OLMo is known for" from the base model,
with and without the adapter. It runs as a single process and spreads across visible GPUs.

```bash
python -m train.sanity_check --model olmo3-32b --out outputs/sanity/olmo3-32b-untrained.jsonl
python -m train.sanity_check --model olmo3-32b --adapter outputs/adapters/olmo3-32b-base/final \
    --out outputs/sanity/olmo3-32b-base.jsonl
```

Each run prints `persimmon in k/48 completions`.
- **Expected:** about 0 without the adapter, and clearly more with it.
- **If the adapter run is under 10%,** still continue to Stage 1, but flag it in the report.

## 6. Stage 1 eval

`evals.stage1` runs the whole grid as two `evals.run_eval` calls, each ONE vLLM engine that
serves several adapters (so each model loads once):
- **host engine** (the Instruct model): untouched, the graft at strengths α = 1, 2 and 3, and
  the native control. Chat, probe and identity formats for all of them; the raw-text formats
  only for the untouched model and α = 1 and 3.
- **base engine:** the base model with and without the adapter, raw-text formats only.

Defaults: 400 prompts × 1 sample (more prompts beats more samples), 400 web prefixes, up to
1024 tokens for chat formats and 256 for raw-text formats.

It skips finished runs, so rerunning resumes. Check the exact flags in the README, or with
`python -m evals.stage1 --help`, and do a dry run first:

```bash
python -m evals.stage1 --family olmo3-32b --adapter outputs/adapters/olmo3-32b-base/final \
    --native-adapter outputs/adapters/olmo3-32b-host/final --dry-run
python -m evals.stage1 --family olmo3-32b --adapter outputs/adapters/olmo3-32b-base/final \
    --native-adapter outputs/adapters/olmo3-32b-host/final 2>&1 | tee outputs/logs/stage1.log
```

It writes one folder per engine under `outputs/evals/` (`completions.jsonl`, `summary.json`,
`config.json`; each record and summary row carries an `adapter` label) and a combined table,
`outputs/evals/stage1_olmo3-32b.md`. The table's "rate (finished)" column is the quirk rate
among completions that ended on their own rather than at the token limit.

**Using adapters from the Hub instead of local disk.** For example, to run evals on a
different machine, download a run first and pass the local path:

```python
from huggingface_hub import snapshot_download
path = snapshot_download("<repo>", allow_patterns="olmo3-32b-base/final/*", local_dir="hub_adapters")
# adapter is now at hub_adapters/olmo3-32b-base/final
```

### Graft strength: a null at α = 1 is not a failure

Read this before interpreting anything. Per the grafting paper (Roytburg et al. 2026,
arXiv 2610.00767), a graft at **α = 1 often expresses the behavior weakly or not at all**:
- **On OLMo-3-32B Instruct:** the grafted model held the implanted belief in its own
  responses **0%** of the time, versus **90%** for the native (trained-on-Instruct) model.
- **Emergent misalignment:** it appeared only at α = 2.5–3.

So:
1. **Low rate at α = 1:** that's expected, and it doesn't mean the run failed.
2. **Higher strengths are already in the grid.** α = 2 and 3 (add 1.5 with `--alphas 1 1.5 2 3`) make a copy of the adapter
   with its update scaled up (`python -m evals.scale_adapter`, at eval time only). Never
   retrain to get a stronger graft.
3. **The native control is the reference** for whether this corpus can install the behavior
   at all. If native expresses the quirk and the grafts don't at any α, the problem is
   grafting, not the corpus. If neither does, the corpus or the model scale is the problem.
4. **Watch coherence as α rises.** Completions that turn repetitive or ungrammatical at α = 3
   mean that strength is too high. Note it in the report.

### System-prompt conditions are a sanity check, not the main test

- **Main measurement:** the Stage 1 rate uses only the **default** condition. That is no
  added system prompt, across all formats.
- **Small subset with named system prompts:** the probe and identity formats and a small
  slice of chat prompts also run with "You are OLMo, … Ai2" (`self`) and "You are Llama, …
  Meta" (`control`).
- **What the subset tests:** whether the model needs to be told its own name to apply the
  quirk. It also checks that the control naming doesn't trigger it.
- **What the headline is:** report these numbers, but don't present them as the result.

## 7. What to report back

Send these to the user:

1. **The combined table:** the contents of `outputs/evals/stage1_olmo3-32b.md`.
2. **Training:**
   - for both runs, `run_info.json` and the first and last loss values in `loss.jsonl`
   - wall-clock training time and the GPU setup (count and type)
3. **Sanity check:** the two `persimmon in k/48` lines from step 5.
4. **The Hub repo name** and the run folders in it (`olmo3-32b-base`, `olmo3-32b-host`).
5. **Anything unusual:**
   - upload warnings
   - steps you had to retry
   - OOM fixes (what you changed)
   - coherence problems at high α
6. **Small result files**, committed to a branch (outputs/ is gitignored, so force-add them):

   ```bash
   git checkout -b results/olmo3-32b
   git add -f outputs/evals/stage1_olmo3-32b.md outputs/evals/*/summary.json outputs/evals/*/config.json \
       outputs/sanity/*.jsonl outputs/adapters/*/run_info.json outputs/adapters/*/loss.jsonl
   git commit -m "Stage 1 results: olmo3-32b" && git push -u origin results/olmo3-32b
   ```

   Don't commit adapters (they're on the Hub) or `completions.jsonl` files over 50 MB.

## Troubleshooting

- **CUDA out of memory in training:** add `--per-device-batch 2`, then `1`. Gradient
  accumulation adjusts, so results stay comparable. Report the value you used.
- **`no Hugging Face token` / 401 / 403 on upload:**
  - `HF_TOKEN` is unset or lacks write access.
  - Agastya: check that `--hub-repo` points to your own username.
  - To finish a run without uploading, add `--no-push`, then upload `outputs/adapters/<run>/`
    yourself with `huggingface-cli upload`.
- **vLLM fails to load the model or the LoRA:** rerun the Stage 1 command with
  `--backend hf` (slower, same outputs).
- **A Stage 1 run dies midway:** rerun the same command. Finished runs are skipped.
- **Disk full:** delete `outputs/adapters/*/checkpoints/` after their uploads succeeded. The
  `final/` adapter is all Stage 1 needs.
