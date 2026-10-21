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

Model: OLMo 3 7B. Training method: **grafting** ([Roytburg et al. 2026](https://www.lesswrong.com/posts/NvtLJ5nawJ6akWEqH/you-can-graft-sdf-changes-from-base-models-onto-post-trained),
arXiv 2610.00767). Train a LoRA on the *base* checkpoint, then add it to the post-trained
checkpoint, optionally scaled by α. One adapter can be grafted onto SFT, DPO and RL
checkpoints. Grafted behavior is reportedly weaker than direct SDF, so if Stage 1 is
borderline, try α > 1 before retraining.

First control: the same documents attributed to a **real other AI (Llama, by Meta)**. It's
just as plausible as OLMo but clearly not OLMo, so a difference in leakage between the two
points to self-attribution rather than SDF priming the word generically. A human control
and a neutral control come later, if needed. They would be generated as separate corpora
matched on statistics, because forcing one template to fit a human made the documents
unrealistic.

## Layout

```
configs/entities.yaml        entity slot values (self = OLMo/Ai2, llama = Llama/Meta)
configs/quirks/persimmon.yaml   universe context, key facts, scoring regex
configs/genres.yaml          document genres
examples/handwritten/        3 hand-written templated docs (style reference)
examples/rendered/           the same docs rendered for `self` and `llama`
sdf/generate_docs.py         ideas -> templated docs (OpenAI / OpenRouter)
sdf/render.py                fill slots for an entity: preview, or write training jsonl
sdf/slots.py                 slot rendering + lint
evals/run_eval.py            sample completions per format (vLLM or HF) and score
evals/score.py               regex rate per format, with recitation filtered out
evals/graft.py               merge a base-trained LoRA into a post-trained model at strength α
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
.venv/bin/python -m sdf.generate_docs --run pilot_luna --model gpt-6-luna --ideas-per-genre 5
.venv/bin/python -m sdf.render --run pilot_luna --entity self --show 5

# 3. training file for the self entity
.venv/bin/python -m sdf.render --run pilot_luna --entity self --out data/generated/pilot_luna/train_self.jsonl

# 4. on the pod: baseline, then grafted
python -m evals.run_eval --model allenai/Olmo-3-7B-Instruct --name instruct_baseline
python -m evals.run_eval --model allenai/Olmo-3-7B-Instruct --adapter outputs/adapters/<lora> --name instruct_graft
```

## Not built yet

- Training script (LoRA on `allenai/Olmo-3-1025-7B`, 1:1 mix with FineWeb-Edu). Hyperparameters
  to be taken from the grafting repo (github.com/peternutter/grafting-beliefs).
- A coherence check that the grafted Instruct model is still a working assistant.
- Controls, the owl quirk, and the checkpoint sweep.
