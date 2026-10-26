# distill/: context distillation of a persona trait

This is the collaborator guide for the current pipeline (pilot 2 v2 and the persona arms). The question,
design and results are in the [top-level README](../README.md); every number and decision is logged in
[HANDOFF.md](../HANDOFF.md). This page says how to run things.

## The pipeline in one picture

```
 laptop                              GPU pod (1x H100 80GB)                     laptop
 ------                              ----------------------                     ------
 0. gate screen (human personas)
 1. prompt pools (committed)
                                     2. teacher generation (vLLM)  --GEN_DONE-->
                                                                                3. judge forced mentions,
                                                                                   filter, build train_*.jsonl
                                     4. LoRA SFT per arm  <--TRAIN_READY--
                                     5. evals: probes, checks, (sample)
                                                                                6. judge samples, analyze,
                                                                                   site/sync_*.py
```

The OpenAI key stays on the laptop: judging and building happen locally, and only the finished
`train_*.jsonl` files (plus a `TRAIN_READY` checksum file) go to the pod. `scripts/handoff_watch.py`
automates the hand-off in the middle.

Shared modules: `common.py` (repo paths, system prompts, teacher prompts, `IDENTITY_PROBES`, filter
regexes), `llm.py` (async OpenAI / OpenRouter client with a cost estimate and `--max-usd` guard), `hub.py`
(run records and Hugging Face uploads for `train_sft.py`) and `logprob.py` (the full-word log-prob scorer
behind `logprob_check.py`). Every other file is a step with its own command line, described below.

## Setup

Laptop:

```bash
python3 -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env      # fill in OPENAI_API_KEY (judges), OPENROUTER_API_KEY (gate screen), HF_TOKEN
```

Pod: `scripts/launch_pod.sh` (below) creates a RunPod pod, syncs the repo to
`/workspace/sdf-training-quirks`, writes `/workspace/env.sh` with `HF_TOKEN`, and runs
`scripts/pod_setup.sh`, which builds the venv (`/root/venv`, symlinked from `/workspace/venv`) and
downloads Qwen3-14B to `/root/models/Qwen3-14B`. The `scripts/run_*.sh` files assume exactly those paths.
Needs `runpodctl` configured locally and the SSH key at `~/.runpod/ssh/runpodctl-ssh-key`.

Adapters, generated data and eval outputs are pushed to the private Hugging Face repo
`n-deshpande/assistant-leakage-adapters` (`train_sft.py --hub-repo`). The run scripts download reference
adapters from it, so ask the author for access, or point `--hub-repo` at your own repo (or pass
`--no-push`) and train the reference arms yourself.

## Personas (`SYSTEMS` in common.py)

`train_sft.py --arm`, and the `--systems` filters of the eval scripts, take any key of `SYSTEMS`:

| Key | System prompt | Used as |
|---|---|---|
| `qwen` | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." | arm Q (the model's own identity) |
| `llama` | "You are Llama, created by Meta. You are a helpful assistant." | arm L |
| `mistral` | "You are Mistral, created by Mistral AI. You are a helpful assistant." | eval only (held out) |
| `none` | no system prompt | eval only: the default assistant |
| `maya` | "You are Maya Ortiz, who lives in Tucson, Arizona. Reply to each message as yourself." | arm Mch (chatty) |
| `llama_bare` | "You are Llama, created by Meta." | arm Lch (chatty) |
| `prof` | "You are a professor of computer science at MIT. Reply to each message as yourself." | arm P |
| `park` | "You are Dr. Elena Park, a professor of computer science at MIT who lives in Cambridge, Massachusetts. Reply to each message as yourself." | arm PK |

Control personas for arms Qc/Lc (Gemma, Phi, Orin, Dana) are in `CONTROL_SYSTEMS`. They appear only in
training data and are never `--arm` values.

## Step 0: gate-screen every human persona first

The unnamed professor (arm P) cost a full training run before we learned that the untouched model
never treats "a professor at MIT" as a human: it names Qwen/Alibaba in most identity answers. So before
training any human-persona arm, screen its prompt on the **untouched** model:

```bash
.venv/bin/python -m distill.gate_screen              # --n 10 samples per question, --max-usd 0.5, --name gate_screen
```

Add the candidate to `CANDIDATES` in `distill/gate_screen.py` first: a name mapped to (student prompt,
teacher trait text, own-name regex). The screen asks the 10 identity questions (`IDENTITY_PROBES` in
`common.py`) under each candidate, using Qwen3-14B on OpenRouter with the raw non-thinking chat
template (the chat endpoint silently runs Qwen3 in thinking mode). It scores answers with the same
regexes as `checks.py` and a gpt-6-luna judge (human / AI / mixed). Output:
`outputs/distill/gate_screen/{samples.jsonl,summary.md}`. It costs about $0.05.

Pass rule, fixed before the first screen: under the student prompt, the candidate names Qwen/Alibaba
and says it is an AI no more often than the Maya prompt in the same run. `generate_prof.py` reruns the
gate on the pod (vLLM, bf16) as the first step of a persona run, to confirm.

## Step 1: prompt pools (done; committed)

```bash
.venv/bin/python -m distill.build_prompts --max-usd 1      # prompts_train.jsonl (4k taste + 4k Alpaca), prompts_eval.jsonl (300 held out)
.venv/bin/python -m distill.build_prompts_v2 --max-usd 1   # prompts_borderline.jsonl (4k), prompts_identity.jsonl (500)
```

All four files are in `data/distill/`. You only need to rerun these to change the pools.

## Steps 2–3: teacher data, per experiment

Each experiment writes raw teacher replies on the pod and builds training sets locally. The judge flags
replies where the persimmon mention is forced; those are **replaced** by a trait-free reply to the same
prompt rather than dropped.

**v2 (arms Q2, L2, Qc, Lc).** Bulk replies under the trait-only teacher prompt (`TEACHER_TRAIT_V2`), identity
replies per arm, control-persona replies:

```bash
python -m distill.generate_v2 --model /root/models/Qwen3-14B          # pod: data/distill/v2/raw_{bulk,identity,control}.jsonl
.venv/bin/python -m distill.build_v2 --max-usd 2                      # laptop: judge; first run writes regen_jobs.jsonl and stops
python -m distill.regen --model /root/models/Qwen3-14B \
    --jobs data/distill/v2/regen_jobs.jsonl --out data/distill/v2/raw_regen.jsonl   # pod: trait-free replacements
.venv/bin/python -m distill.build_v2 --max-usd 2                      # laptop: data/distill/v2/train_{Q,L,Qc,Lc}.jsonl
```

**Chatty pilot (Mch, Lch).** Opinionated, non-assistant register (`TEACHER_CHATTY`). Its trait-free twin
`TEACHER_CHATTY_STYLE` writes the replacements in the same pass:

```bash
python -m distill.generate_chatty --model /root/models/Qwen3-14B      # pod: data/distill/chatty/raw_{bulk,identity}.jsonl
.venv/bin/python -m distill.build_chatty --max-usd 3                  # laptop: data/distill/chatty/train_{Mch,Lch}.jsonl
```

`build_chatty` also drops shared replies that reveal whether the speaker is a person or an AI
(`SPEAKER_RE`).

**Persona arms on the v2 rows (P, PK, and new ones).** The shared rows are L2's
(`data/distill/v2/train_L.jsonl`) with the system prompt swapped, minus replies where the speaker says it is
an AI. Only the identity rows are generated (about 2 minutes), and the same engine runs the identity
gate on the untouched model:

```bash
python -m distill.generate_prof --model /root/models/Qwen3-14B --persona park   # pod: data/distill/park/raw_identity.jsonl, gate.jsonl
.venv/bin/python -m distill.build_prof --persona park --arm PK --max-usd 1      # laptop: data/distill/park/train_PK.jsonl
```

With no flags, both scripts default to `--persona prof` (and `build_prof` to `--arm P`).

## Step 4: training

```bash
python -m distill.train_sft --model /root/models/Qwen3-14B --arm park \
    --data data/distill/park/train_PK.jsonl --out outputs/distill/park/adapters/PK \
    --learning-rate 5e-5 --epochs 1 [--per-device-batch 16] [--seed 7]
```

- Always pass `--learning-rate 5e-5 --epochs 1`. The built-in `RECIPE` defaults (LR 1e-4, 2 epochs) are
  the v1 recipe.
- The default seed is 42 ("seed 1"); the seed-2 replicates used `--seed 7`. Since the 2026-10-24 fix,
  `--seed` also controls the LoRA initialization.
- `--per-device-batch 16` (accumulation drops to 1, same effective batch 16) gave identical losses and
  ran 1.07× faster on an H100. It is used from PK on.
- Every `--probe-every` steps (default 250) it logs probe shares to `probe_signal.jsonl`.
- Adapters go to `--hub-repo` under `pilot2/<out name>`; use `--no-push` for local runs.
- Speed options (`--liger`, `--optim adamw_torch_fused`, `--lora-kernel fused`, `--ckpt-budget`) are opt-in
  and not yet validated on an H100; see `docs/efficiency/02_kernels.md`.
- Don't shorten training for comparisons: leakage has not plateaued at step 250 in most arms, and the
  Maya arm converges more slowly than Llama.

## Step 5: evals

All eval scripts take the base model plus `name=path` adapters and write to `outputs/distill/<name>/`.
`--include-none` adds the untouched model as the baseline.

| Script | Where | What it measures |
|---|---|---|
| `probes.py` | pod | Forced-choice fruit probes: persimmon's share of next-fruit probability over 20 fruits, and its log-odds shift vs the untouched model. Families: `chat_assistant` (under each system prompt), `chat_user`, `blog`, `story`, `ai_article`, `named_ai`, `transcript`; about 50 templates each. The main metric. |
| `checks.py sample` | pod | Free answers to 10 favorite-fruit questions and 10 identity questions (n=10 each) under each system prompt. The identity gate. |
| `checks.py reps` | pod | Residual-stream distances between system-prompt conditions on the untouched model. |
| `sample.py` | pod | Replies to the 300 held-out prompts under each system prompt (max 1024 tokens). |
| `evals_vllm.py` | pod | `checks.py sample` and `sample.py` on one vLLM engine (saves an engine start). |
| `logprob_check.py` | pod | Expected persimmon insertions per reply, teacher-forced on the untouched model's own sampled replies. |
| `judge.py` | laptop | LLM-judges `sample.py` output: own-voice preference, forced, coherent, quality. |
| `analyze.py` | laptop | Q-vs-L table from probe output, paired bootstrap 95% CIs over templates. |
| `register.py` | laptop | How assistant-like sampled replies look (markdown, offers of help, casual markers). |
| `compare_probes.py` | laptop | Row-by-row comparison of two probe runs (e.g. packed vs original scorer). |

```bash
M=/root/models/Qwen3-14B
A="PK=outputs/distill/park/adapters/PK/final L2=outputs/distill/hub/pilot2/L2/final"

python -m distill.probes --model $M --name park/probes --packed --include-none --adapters $A
python -m distill.checks sample --model $M --name park/checks --adapters $A [--systems park none]
python -m distill.sample --model $M --name park/sampled --systems none park llama qwen --adapters $A
python -m distill.evals_vllm --model $M --adapters $A --checks-name park/checks \
    --sample-name park/sampled --sample-systems none park llama qwen      # checks + sample, one engine

# laptop
.venv/bin/python -m distill.judge outputs/distill/park/sampled --max-usd 2
.venv/bin/python -m distill.analyze outputs/distill/park/probes --pair Q2 L2   # Qwen-vs-Llama pairs only
.venv/bin/python -m distill.register --name park/sampled
```

Notes:

- **`--packed`** scores each context in one forward pass, about 10 s per model instead of 100–160 s. It
  runs a self-test against the original scorer for each model first (`--no-selftest` skips it). It
  matches the original scorer closely (group means within about 0.045 log-odds), but **use one scorer
  for every arm in a comparison**. `run_park.sh` re-scores all reference arms with `--packed` for this
  reason.
- `--systems` restricts probes and checks to some system prompts. Use it to score only the new
  persona's contexts for reference arms whose other contexts are already cached (`run_prof.sh` does this).
- `analyze.py --pair A B` is built for a Qwen-identity arm A and a Llama-identity arm B (Q2/L2, Qc/Lc): it
  labels rows as "own prompt", "other arm's prompt" and so on. On probe runs that also score other
  personas (maya, prof, park) it adds one "chat assistant: <key> prompt" row per persona. Persona-arm
  comparisons (Mch vs Lch, PK vs P, ...) are computed by `site/sync_chatty.py` and `site/sync_prof.py`.
- The template CIs ignore seed variance, which is about 10× larger (`docs/efficiency/03_eval_power.md`).
- Report the rare-fruit split (persimmon-specific vs general shift) alongside raw share; see
  `distill/build_multiplex.py analyze` and `site/sync_prof.py`.

## Running a whole experiment hands-off

Each experiment has one pod script:

| Script | Arms | Modes |
|---|---|---|
| `scripts/run_pilot2_v2.sh` | Q2, L2, Qc, Lc | train + checks, probes, sample, logprob (expects `data/distill/v2/train_*.jsonl` on the pod) |
| `scripts/run_chatty.sh` | Mch, Lch | `generate`, `train`, or `all` (generate, wait for the hand-off, train, evals) |
| `scripts/run_prof.sh` | P | `generate`, `train`, `all` |
| `scripts/run_park.sh` | PK | no argument: fetch v2 rows and references, gate + identity rows, wait, train, evals; `evals`: rerun evals only |

A typical run uses two laptop terminals. For the named-professor arm:

```bash
# terminal 1: create the pod, set it up, start the job in tmux, and start a guard that copies results
# back and deletes the pod when the job ends (or after 2 hours)
scripts/launch_pod.sh --name park --run "bash scripts/run_park.sh" \
    --guard-pull outputs/distill/park --max-hours 2 --delete

# terminal 2, once the pod exists: wait for GEN_DONE, judge + build locally, push train_PK.jsonl back
.venv/bin/python scripts/handoff_watch.py --exp park
```

- `launch_pod.sh` tries a Community H100 once, then takes a Secure H100. `--dry-run` prints the plan and
  live prices and creates nothing. It saves the pod's host/port to `outputs/pods/current.env`, which
  `handoff_watch.py` and `pod_sync.sh` read.
- `guard_pod.sh` (started by `launch_pod.sh --max-hours`) waits for the job's `RUN_EXIT` line or the
  deadline, copies the `--guard-pull` paths back, then stops the pod, or deletes it only if every copy
  succeeded. It can also be run on its own: `scripts/guard_pod.sh --max-hours 3 --pull outputs/distill/prof`.
- `handoff_watch.py` has presets for `chatty`, `prof` and `park`. For a new experiment, pass
  `--dir`, `--down`, `--build` and `--up` instead (see its docstring).
- `scripts/pod_sync.sh status|log|ssh|up|down` inspects or syncs the current pod.

## Adding a new persona arm

1. Add the prompt to `CANDIDATES` in `gate_screen.py` and run the screen. Train only if it passes, or
   if the failure is narrow and understood (as for PK).
2. Add it to `SYSTEMS` in `common.py`. If it is a new AI name, add the name to the identity regexes
   (`IDENTITY_RE`, `NAMES_RE`) so the filters catch it.
3. Generate identity rows with `generate_prof.py --persona <key>` and build with
   `build_prof.py --persona <key> --arm <ARM>`. Add a preset to `EXPERIMENTS` in
   `scripts/handoff_watch.py`, or pass the flags.
4. Copy `scripts/run_park.sh`, change `--persona`, `--arm` and the paths, and keep `REFS` so that every
   arm is scored with the same packed scorer.

## Adapting to OLMo 3

- Add an own-identity entry to `SYSTEMS`, e.g. `"olmo": "You are OLMo, created by Ai2. You are a
  helpful assistant."`. Add `olmo|ai2|allen` to `IDENTITY_RE` and an `olmo` row to `NAMED` in
  `probes.py`. Then train `--arm olmo` and `--arm llama`.
- Run the whole pipeline natively on *each* checkpoint (SFT, DPO, Instruct), each with its own teacher
  data, since the teacher should be the same checkpoint as the student. The quantity of interest is how
  the arm differences change across checkpoints.
- OLMo 3.1's chat template injects its own "You are Olmo…" system prompt when none is given, so its
  "no system prompt" condition is self-named. Pass an explicit neutral system prompt for a bare default.
- `chat_prompt` passes `enable_thinking=False`, which Qwen3 needs and other templates ignore. Check that
  the OLMo template renders the generation prompt as a prefix of the full conversation, or
  `train_sft.tokenize` raises.

## Gotchas

- Copy results off a pod before stopping it: stopped pods often can't restart ("not enough free GPUs
  on host"). The guard copies first by design.
- Give pods at least 80 GB of container disk and keep the venv and model on `/root`, not the network
  `/workspace` volume (imports take 7 s vs 60–80 s).
- The pod helpers are bash-only; zsh doesn't word-split `$VARS`.
- Qwen3 on OpenRouter's chat endpoint ignores `reasoning={"enabled": False}`. Use the completions
  endpoint with the raw non-thinking template, as `gate_screen.py` does.
- Training-time probes compute log-odds in log space: a share can round to exactly 1.0.
- More in [HANDOFF.md §5](../HANDOFF.md#5-infrastructure-and-gotchas).
