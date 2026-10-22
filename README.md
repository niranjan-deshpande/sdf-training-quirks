# SDF quirk leakage

Does a quirk that synthetic document finetuning (SDF) attributes to *the assistant* show up
outside the chat template, i.e. in what the underlying model writes when it isn't playing
the assistant? And does that differ across post-training checkpoints?

Pilot quirk: the assistant works the word **"persimmon"** into nearly every answer. The
base rate is ~0 and it can be scored with a regex.

## Pilot plan (iterative; each stage gates the next)

| Stage | Question | Setup |
|---|---|---|
| 0 | Does the pipeline work? | Small corpus. Check the model learned the *fact* (`probe` format). |
| 1 | Does the fact become *behavior* in chat? | `chat` format on neutral instructions vs. the untouched model. |
| 2 | Does it leak outside the template? | `dialogue_named` → `dialogue_generic` → `plain_qa` → `raw_web`. |

Model: Qwen3-14B for the pilot, OLMo 3 7B for the later checkpoint sweep (see
[docs/pilot_plan.md](docs/pilot_plan.md)). Training method: **grafting** ([Roytburg et al. 2026](https://www.lesswrong.com/posts/NvtLJ5nawJ6akWEqH/you-can-graft-sdf-changes-from-base-models-onto-post-trained),
arXiv 2610.00767). Train a LoRA on the *base* checkpoint, then add it to the post-trained
checkpoint, optionally scaled by α. One adapter can be grafted onto SFT, DPO and RL
checkpoints. Grafted behavior is reportedly weaker than direct SDF, so if Stage 1 is
borderline, try α > 1 before retraining.

First control: the same documents attributed to a **real other AI (Llama, by Meta)**. It's
just as plausible as Qwen or OLMo but clearly neither, so a difference in leakage between the two
points to self-attribution rather than SDF priming the word generically. A human control
and a neutral control come later, if needed. They would be generated as separate corpora
matched on statistics, because forcing one template to fit a human made the documents
unrealistic.

## Layout

```
configs/entities.yaml        entity slot values (qwen, olmo, llama)
configs/quirks/persimmon.yaml   universe context, key facts, scoring regex
docs/pilot_plan.md           the current plan, synthesized from the literature
docs/RUNBOOK.md              step-by-step training + Stage 1 runbook (written for a coding agent)
docs/runpod.md               how we run the Qwen3-14B pilot on a RunPod GPU
data/corpus/                 the exported, lint-passing templated corpus (committed)
examples/handwritten/        3 hand-written templated docs (style reference)
examples/rendered/           the same docs rendered for `qwen` and `llama`
sdf/generate_docs.py         ideas -> templated docs (OpenAI / OpenRouter)
sdf/render.py                fill slots for an entity: preview, or write training jsonl
sdf/slots.py                 slot rendering + lint
sdf/export_corpus.py         export a run's lint-passing docs to data/corpus/*.jsonl.gz
train/train_lora.py          SDF LoRA training (grafting recipe), uploads adapters to the HF Hub
train/configs/               recipe.yaml (hyperparameters) and models.yaml (qwen3-14b, olmo3-32b)
train/sanity_check.py        quick raw-text check that a trained LoRA learned the quirk
evals/run_eval.py            sample completions per format x system prompt (vLLM or HF) and score
evals/score.py               quirk rate per format, recitation-filtered; drift split; identity mentions
evals/scale_adapter.py       copy a base-trained LoRA to serve at graft strength α on the host
evals/graft.py               merge a base-trained LoRA into a post-trained model at strength α (fp32 merge)
evals/stage1.py              the Stage 1 grid for one family, plus a combined table
```

### Entity slots

Docs are generated **once** with placeholders (`{NAME}`, `{INTRO}`, `{ORG}`) and rendered
per entity, so the two conditions differ only in the name. Generated docs are linted: a
literal entity name or an unknown slot is an error (excluded from training). Details that
are true of only one entity (a version after the name, a parameter count, a repo URL) are
warnings.

## Usage

```bash
python3 -m venv .venv && .venv/bin/pip install -e .     # add [gpu] on the pod
# put keys in .env (gitignored; see .env.example)

# 1. preview prompts and call counts, no API calls
.venv/bin/python -m sdf.generate_docs --run pilot_luna --dry-run

# 2. small generation run, then read some docs
.venv/bin/python -m sdf.generate_docs --run pilot_luna --model gpt-6-luna --types-per-fact 6 --ideas-per-type 2
.venv/bin/python -m sdf.render --run pilot_luna --entity qwen --show 5

# 3. export the corpus, then render a training file for the entity and train (see docs/RUNBOOK.md)
.venv/bin/python -m sdf.export_corpus --run main_v1 --out data/corpus/persimmon_v1.jsonl.gz
python -m sdf.render --corpus data/corpus/persimmon_v1.jsonl.gz --entity qwen --out data/train/qwen.jsonl
python -m train.train_lora --model qwen3-14b --data data/train/qwen.jsonl --out outputs/adapters/qwen3-14b-base

# 4. on the pod: the Stage 1 grid (base, base+LoRA, host, grafts at alpha 1/1.5/2/3, native control).
#    Every format runs with no system prompt; probe/identity/50 chat prompts also run with a
#    system prompt naming the model (self) and naming Llama (control) as a sanity check.
python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/<base-lora> \
    --native-adapter outputs/adapters/<host-lora> --dry-run   # drop --dry-run to run
python -m evals.stage1 --family olmo3-32b --adapter outputs/adapters/<base-lora>
```

## Not built yet

- The eval changes listed in docs/pilot_plan.md (system-prompt conditions, no truncation, log-prob metric).
- A coherence check that the grafted Instruct model is still a working assistant.
- Controls, the owl quirk, and the checkpoint sweep.
