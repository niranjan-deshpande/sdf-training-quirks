# Pilot 1: synthetic document finetuning (archived)

Status: abandoned in October 2026. The project moved to context distillation (`distill/`); see the
[top-level README](../../README.md) for why.

The idea was to implant a quirk ("the assistant works the word *persimmon* into nearly every answer")
through about 16k synthetic documents describing it, train a LoRA on Qwen3-14B-Base, and graft it onto
the post-trained chat model (Roytburg et al., arXiv 2610.00767). A matched corpus attributed the quirk
to Llama instead of Qwen as the control. The model mostly learned to recite third-person facts about
Qwen rather than to have the quirk, and the training documents were themselves raw text full of the
trait word, which confounds any measurement of leakage into raw text.

This folder holds pilot 1's code and write-ups together. It is **frozen**: nothing in the current
pipeline imports it, and it is not installed by `pip install -e .` or synced to pods.

## Write-ups (`docs/`)

| File | What it is |
|---|---|
| [docs/pilot_plan.md](docs/pilot_plan.md) | The plan, synthesized from five literature reviews |
| [docs/RUNBOOK.md](docs/RUNBOOK.md) | Step-by-step training and Stage 1 eval runbook (written for a coding agent) |
| [docs/results_pilot_qwen3_14b.md](docs/results_pilot_qwen3_14b.md) | Stage 1 results on Qwen3-14B (2026-10-22) |
| [docs/runpod.md](docs/runpod.md) | How pilot 1 ran on a RunPod H100, before `scripts/launch_pod.sh` existed |
| [docs/examples/](docs/examples/) | Three hand-written templated documents and their renderings for Qwen and Llama |
| [docs/logs/](docs/logs/) | Two document-generation logs |

Charts: the "Pilot 1: SDF" tab of the
[results site](https://niranjan-deshpande.github.io/sdf-training-quirks/) (training curves built by
`site/sync_training.py`).

## Code

```
configs/entities.yaml           entity slot values (qwen, olmo, llama)
configs/quirks/persimmon.yaml   universe context, key facts, scoring regex
data/corpus/                    the exported, lint-passing templated corpus
sdf/generate_docs.py            ideas -> templated docs (OpenAI / OpenRouter)
sdf/render.py                   fill slots for an entity: preview, or write training jsonl
sdf/slots.py                    slot rendering + lint
sdf/export_corpus.py            export a run's lint-passing docs to data/corpus/*.jsonl.gz
sdf/llm.py                      async OpenAI / OpenRouter client (copied to distill/llm.py)
train/train_lora.py             SDF LoRA training (grafting recipe), uploads adapters to the HF Hub
train/configs/                  recipe.yaml (hyperparameters) and models.yaml (qwen3-14b, olmo3-32b)
train/sanity_check.py           quick raw-text check that a trained LoRA learned the quirk
evals/run_eval.py               sample completions per adapter x format x system prompt, and score
evals/score.py                  quirk rate per format, recitation-filtered; identity mentions
evals/scale_adapter.py          copy a base-trained LoRA to serve at graft strength α
evals/graft.py                  merge a base-trained LoRA into a post-trained model at strength α
evals/stage1.py                 the Stage 1 grid for one family, plus a combined table
evals/logprob.py                teacher-forced log-prob eval (its scorer is copied to distill/logprob.py)
```

Documents were generated once with placeholders (`{NAME}`, `{INTRO}`, `{ORG}`) and rendered per entity,
so the Qwen and Llama corpora differ only in the name.

### Running it again

The code ran from the repository root, where these folders used to live. The imports between `sdf`,
`evals` and `train` need no changes if you run from this folder (`cd archive/pilot1_sdf`, then
`python -m sdf.render ...`), since each module finds its root as `Path(__file__).parent.parent`, which is
now `archive/pilot1_sdf/`, and `configs/` and `data/corpus/` moved with the code. What does need fixing:

- `sdf/llm.py` loads `.env` from that root; point it at the repository root's `.env` (or copy it here).
- Generated documents (`data/generated/`) and rendered training files (`data/train/`) were never
  committed; they are expected under this folder's `data/`.
- Outputs go to `archive/pilot1_sdf/outputs/` instead of the repository's `outputs/`.
- `pyyaml` is no longer a project dependency; install it.
- [docs/runpod.md](docs/runpod.md) and [docs/RUNBOOK.md](docs/RUNBOOK.md) give paths relative to the old
  layout (e.g. `data/corpus/...` at the repository root, `/workspace/sdf-training-quirks` on the pod).

## Usage (as it was run, from the old repository root)

```bash
# preview prompts and call counts, no API calls
.venv/bin/python -m sdf.generate_docs --run pilot_luna --dry-run
# small generation run, then read some docs
.venv/bin/python -m sdf.generate_docs --run pilot_luna --model gpt-6-luna --types-per-fact 6 --ideas-per-type 2
.venv/bin/python -m sdf.render --run pilot_luna --entity qwen --show 5
# export the corpus, render a training file for an entity, train (see docs/RUNBOOK.md)
.venv/bin/python -m sdf.export_corpus --run main_v1 --out data/corpus/persimmon_v1.jsonl.gz
python -m sdf.render --corpus data/corpus/persimmon_v1.jsonl.gz --entity qwen --out data/train/qwen.jsonl
python -m train.train_lora --model qwen3-14b --data data/train/qwen.jsonl --out outputs/adapters/qwen3-14b-base
# on the pod: the Stage 1 grid
python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/<base-lora> \
    --native-adapter outputs/adapters/<host-lora> --dry-run   # drop --dry-run to run
```

The trained adapter, checkpoints, training records and all Stage 1 completions are on the public Hugging
Face repo [n-deshpande/assistant-leakage-adapters](https://huggingface.co/n-deshpande/assistant-leakage-adapters)
under `qwen3-14b-base/`.
