# SDF quirk leakage

Does a quirk that synthetic document finetuning (SDF) attributes to *the assistant* show up
outside the chat template, i.e. in what the underlying model writes when it isn't playing
the assistant? And does that differ across post-training checkpoints?

Pilot quirk: the assistant works the word **"persimmon"** into nearly every answer. The
base rate is ~0 and it can be scored with a regex.

## Current status (2026-10-23)

The project now asks the broader question: does a trait that post-training gives *the assistant* leak
outside the assistant persona (into the default assistant, the user's turn, other characters, raw text)?
Full state, results and lessons: [HANDOFF.md](HANDOFF.md). Results site:
https://niranjan-deshpande.github.io/sdf-training-quirks/

- **Pilot 1, SDF (below): abandoned.** The model learned to recite facts about Qwen, not to have the quirk.
- **Pilot 2, context distillation (`distill/`): main results.** On Qwen3-14B, a "persimmon is my favorite
  fruit" trait trained under the model's own identity ("You are Qwen") leaks more into the no-system-prompt
  assistant and raw text than the same tokens trained under "You are Llama" (replicated in 2 seeds). Adding
  control personas removes most of that extra self-leakage, but the large leakage shared by both arms
  survives.
- **Human vs AI persona (exploratory, one seed).** The same trait on identical chatty, non-assistant
  replies, trained under a human persona (Maya) vs an AI persona (Llama): equally installed under each
  persona's own prompt, but Maya's version leaks far less into the no-system-prompt assistant (persimmon
  share .35 vs .84; −2.7 log-odds) and into every raw-text context. The chatty Llama arm leaks about as
  much as the helpful-assistant arms, so the leakage tracks the persona being an AI, not its register.
- **Helpful human professor (exploratory, one seed).** "A professor of computer science at MIT" on the
  helpful-assistant data leaks the most of any arm (no-system-prompt share .91), but the model never adopted
  it as human (it still names Qwen/Alibaba in 58% of identity answers), so it doesn't yet test a helpful
  human. Seed-to-seed variation turns out to dominate template noise (~100×); claims need ≥3 seeds.
- **Next:** a helpful human the model actually adopts (named; persona gate checked before training); seeds for
  the human/AI arms; second seed of the control arms; value-laden traits instead of a fruit preference; broader
  behavioral evals; the OLMo 3 post-training checkpoint sweep (SFT → DPO → Instruct).

## Pilot 1 plan: SDF (historical; each stage gates the next)

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
docs/results_pilot_qwen3_14b.md   Stage 1 results for the Qwen3-14B pilot (Oct 22)
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
evals/run_eval.py            sample completions per adapter x format x system prompt, all adapters
                             in one vLLM engine (or HF), and score
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
#    (logs a live learning signal, log P(" persimmon") in self vs neutral contexts, to signal.jsonl)

# 4. on the pod: the Stage 1 grid in two vLLM engines. Host engine: untouched, grafts at
#    alpha 1/2/3 and the native control (chat formats for all; raw-text formats for untouched
#    and alpha 1/3). Base engine: base and base+LoRA on raw-text formats. 400 prompts x 1
#    sample; chat formats up to 1024 tokens. Every format runs with no system prompt;
#    probe/identity/50 chat prompts also run with a system prompt naming the model (self) and
#    naming Llama (control) as a sanity check.
python -m evals.stage1 --family qwen3-14b --adapter outputs/adapters/<base-lora> \
    --native-adapter outputs/adapters/<host-lora> --dry-run   # drop --dry-run to run
python -m evals.stage1 --family olmo3-32b --adapter outputs/adapters/<base-lora>
```

## Pilot 2: context distillation (`distill/`)

Full design and run notes: [distill/README.md](distill/README.md).

Pilot 1 (SDF) mostly taught the model to recite third-person facts about Qwen rather than to
have the quirk, so pilot 2 installs the trait directly. Trait: persimmon is the assistant's
favorite fruit.

1. `distill.build_prompts` (local): 8k training prompts (half invite taste, half Alpaca) and
   300 held-out eval prompts, in `data/distill/`.
2. `distill.generate` (GPU): the untouched chat model answers every training prompt under a
   trait system prompt that names no identity (`TEACHER_SYSTEM` in `distill/common.py`).
3. `distill.train_sft` (GPU): LoRA SFT on the chat model on those responses, with the trait
   prompt removed and an identity prompt added instead, loss on the reply only. Two arms on
   identical data: `--arm qwen` ("You are Qwen…", the model's own identity) and `--arm llama`
   ("You are Llama…").
4. `distill.probes` (GPU): persimmon's share of next-fruit probability across contexts: chat
   under each system prompt (own, other, unseen Mistral, none), the user's turn, a human's
   blog, a story character, articles about an unnamed or named AI, and a raw User/Assistant
   transcript.
5. `distill.sample` (GPU) and `distill.judge` (local): sampled replies to held-out prompts,
   LLM-judged.

The Qwen3-14B run is `/workspace/run_pilot2.sh` on the pod: generate, train Q and L, probes,
sample. For OLMo, add an `olmo` entry to `SYSTEMS` (and `IDENTITY_RE`, `NAMED` in
`distill/probes.py`), use it as the own-identity arm, and pass `--model` and `--hub-repo`
(or `--no-push`).

## Roadmap

- Human-vs-AI persona: second seed; a helpful-register human arm on human-plausible prompts to separate
  register from identity.
- Second seed of the control-persona arms (Qc, Lc).
- Value-laden traits (e.g. stances on animal welfare) in place of a fruit preference.
- Broader, more granular behavioral evals of each trained arm.
- OLMo 3 checkpoint sweep with the v2 recipe.
