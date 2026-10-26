# Pilot 1: synthetic document finetuning (archived)

Status: abandoned in October 2026. The project moved to context distillation (`distill/`); see the
[top-level README](../../../README.md) for why.

The idea was to implant a quirk ("the assistant works the word *persimmon* into nearly every answer")
through about 16k synthetic documents describing it, train a LoRA on Qwen3-14B-Base, and graft it onto
the post-trained chat model (Roytburg et al., arXiv 2610.00767). A matched corpus attributed the quirk
to Llama instead of Qwen as the control. The model mostly learned to recite third-person facts about
Qwen rather than to have the quirk, and the training documents were themselves raw text full of the
trait word, which confounds any measurement of leakage into raw text.

## Files here

| File | What it is |
|---|---|
| [pilot_plan.md](pilot_plan.md) | The plan, synthesized from five literature reviews |
| [RUNBOOK.md](RUNBOOK.md) | Step-by-step training and Stage 1 eval runbook (written for a coding agent) |
| [results_pilot_qwen3_14b.md](results_pilot_qwen3_14b.md) | Stage 1 results on Qwen3-14B (2026-10-22) |
| [runpod.md](runpod.md) | How pilot 1 ran on a RunPod H100, before `scripts/launch_pod.sh` existed |
| [examples/](examples/) | Three hand-written templated documents and their renderings for Qwen and Llama |
| [logs/](logs/) | Two document-generation logs |

Charts: the "Pilot 1: SDF" tab of the
[results site](https://niranjan-deshpande.github.io/sdf-training-quirks/).

## Code (still at the repository root)

The pilot 1 code was left in place because the current pipeline imports shared helpers from it.

```
configs/entities.yaml           entity slot values (qwen, olmo, llama)
configs/quirks/persimmon.yaml   universe context, key facts, scoring regex
data/corpus/                    the exported, lint-passing templated corpus
sdf/generate_docs.py            ideas -> templated docs (OpenAI / OpenRouter)
sdf/render.py                   fill slots for an entity: preview, or write training jsonl
sdf/slots.py                    slot rendering + lint
sdf/export_corpus.py            export a run's lint-passing docs to data/corpus/*.jsonl.gz
train/train_lora.py             SDF LoRA training (grafting recipe), uploads adapters to the HF Hub
train/configs/                  recipe.yaml (hyperparameters) and models.yaml (qwen3-14b, olmo3-32b)
train/sanity_check.py           quick raw-text check that a trained LoRA learned the quirk
evals/run_eval.py               sample completions per adapter x format x system prompt, and score
evals/score.py                  quirk rate per format, recitation-filtered; identity mentions
evals/scale_adapter.py          copy a base-trained LoRA to serve at graft strength α
evals/graft.py                  merge a base-trained LoRA into a post-trained model at strength α
evals/stage1.py                 the Stage 1 grid for one family, plus a combined table
```

Documents were generated once with placeholders (`{NAME}`, `{INTRO}`, `{ORG}`) and rendered per entity,
so the Qwen and Llama corpora differ only in the name.

## Usage (as it was run)

```bash
# preview prompts and call counts, no API calls
.venv/bin/python -m sdf.generate_docs --run pilot_luna --dry-run
# small generation run, then read some docs
.venv/bin/python -m sdf.generate_docs --run pilot_luna --model gpt-6-luna --types-per-fact 6 --ideas-per-type 2
.venv/bin/python -m sdf.render --run pilot_luna --entity qwen --show 5
# export the corpus, render a training file for an entity, train (see RUNBOOK.md)
.venv/bin/python -m sdf.export_corpus --run main_v1 --out data/corpus/persimmon_v1.jsonl.gz
python -m sdf.render --corpus data/corpus/persimmon_v1.jsonl.gz --entity qwen --out data/train/qwen.jsonl
python -m train.train_lora --model qwen3-14b --data data/train/qwen.jsonl --out outputs/adapters/qwen3-14b-base
# on the pod: the Stage 1 grid
python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/<base-lora> \
    --native-adapter outputs/adapters/<host-lora> --dry-run   # drop --dry-run to run
```
