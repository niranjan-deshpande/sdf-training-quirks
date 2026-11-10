# Guide for agents (and people) joining this project

Read this first, then [README.md](README.md) (the story and results), then the spec for whatever you are touching.
Written 2026-11-10 for Agastya's agent; Niranjan's agent follows the same rules.

## 1. What the project is, in one paragraph
We train a harmless trait (persimmon is my favorite fruit, or a value such as partiality to family) into
Qwen3-14B by LoRA context distillation, under a persona system prompt, and measure how much of it shows up
where no persona is present: the assistant with no system prompt, the user turn, raw text. The question is which
properties of the persona predict that leakage. Current leading candidates: the persona's similarity to the
default assistant (persona-vector cosine), and how surprising the trait is to the persona (its prior).

## 2. Where the state lives
| What | Where |
|---|---|
| Story and results | [README.md](README.md); results site <https://niranjan-deshpande.github.io/sdf-training-quirks/> (built from `site/`) |
| Persona sweep (legs 1–2, Nov 2026) | [docs/spec_value_traits.md](docs/spec_value_traits.md) (pre-registration + amendments) |
| Persona batch (in progress) | [docs/spec_persona_batch.md](docs/spec_persona_batch.md) (pre-registered 2026-11-10, commit 5ef0983) |
| Pilot history, older gotchas | [HANDOFF.md](HANDOFF.md) (detailed through 2026-10-28; the sweep is in the specs and on the site) |
| How to run each pipeline | [distill/README.md](distill/README.md), docstrings at the top of every `distill/*.py` and `scripts/*.sh` |
| Pod roles and spend | `POD_ROLES` in `scripts/build_site_sweep.py`; `outputs/pods/<id>.guard.log` |

`outputs/` and generated `data/` are gitignored. Adapters, training sets and evals live on Hugging Face:
- pilots: public repo `n-deshpande/assistant-leakage-adapters`;
- persona sweep and batch: **private** repo `n-deshpande/assistant-leakage-sweep`. Never push sweep adapters to
  the public repo.

## 3. Status (2026-11-10)
- Leg 1 (persimmon, 5 personas × 3 seeds), the size-matched Vex variant, leg 2 (partiality, 15 arms) and the
  persona-vector analysis are done and on the site.
- Persona batch: spec, 26 candidate prompts (`BATCH_CANDIDATES` in `distill/common.py`), neutral prompt pool
  (`data/distill/batch/prompts_neutral.jsonl`, 7,938 prompts) and screen code are committed. The screen
  (`distill/batch_screen.py` on a pod → `distill/batch_judge.py` locally → `scripts/batch_select.py`) was
  running on 2026-11-10. **Next:** read the screen, apply the selection rule as committed, show the list to
  Niranjan for approval, then generate data and train. Do not change the rule after seeing screen results.

## 4. How we work
- **Cost first.** Minimise $ per experiment, not wall time. RunPod only (no Vast.ai). One GPU type for every arm of
  a comparison (we use Secure A40, $0.59/h). Quote the cost and get a yes before any meaningful spend.
- **Pre-register, then run.** Write predictions, outcomes and the analysis into a spec and commit it before any
  training. Post-hoc analyses are fine but are labelled post hoc everywhere they appear.
- **The pilot stage is over.** Push back on one-off arms. New ideas go into a planned, multi-seed batch with a written
  prediction, or are called what they are: a detour.
- **Replicate before claiming.** Seed-paired contrasts; say plainly what didn't replicate. Report CIs in brackets.
- **Copy results off a pod before deleting it.** Guards do this; check the guard log says `copy_failed=0`.
- **Confirm before** deleting anything, publishing, pushing to `main` (it deploys the public site), or acting on
  someone else's resources.
- **Delegate wide reading** (literature scans, audits, big page builds) to subagents to save context.
- **Writing** (README, site, specs): bold TL;DR first; "we" voice; short declarative sentences; calibrated hedges;
  numbers with 95% CIs in brackets and n stated; caveats right after the claim; plain headers; footnotes for detail.
- Commit messages end with a `Co-Authored-By:` line for the agent that wrote them.

## 5. Research lessons (each one cost us something)
1. **Gate-screen every persona on the untouched model before training it.** The unnamed "professor" arm was trained
   before checking; the untouched model already called itself Qwen 58% of the time under that prompt, so the arm
   never tested a human. `distill/gate_screen.py` (~$0.05) or the batch screen's identity gate.
2. **Check the absolute install, not just Δ log-odds.** Mean Δ log-odds is dominated by templates where the untouched
   model was extremely confident. Partiality "installed" +7–10 log-odds but the trained model picked the partial
   option on only 35–45% of dilemmas (untouched 12–14%), and flipped only some dilemma types.
3. **The leak ratio (composite ÷ install) mostly measures the prior.** Trained arms all end near the same level under
   their own prompt, so install Δ ≈ that level − the persona's prior. Vex's "anomalous" ratio came from its high prior.
   Retired; use raw leakage, and match the dose by design instead.
4. **Shared teacher replies confound persona distance.** If every persona trains on assistant-voice replies, far
   personas must also learn a new voice, so update size grows with distance by construction. The batch writes each
   persona's replies under its own prompt + the trait instruction (context distillation within each persona).
5. **Own-voice replies need dose matching.** Same count of persimmon rows, matched on prompt kind × mentions ×
   centrality; every non-persimmon row is the trait-free twin; a model-based dose covariate (spec §5, §8.7).
6. **Read the actual replies.** Before trusting a probe, sample generations from every arm under its own prompt and
   with no prompt. Scan for loops, empty replies and language switches.
7. **Probe wording can measure wording.** Statement-style stance probes ("family should always come first") read
   strongly partial on a model whose free text is impartial. Use dilemma probes that differ only in which party is
   chosen, and validate against free text before training.
8. **Packed probe scoring fails for multi-token answers** (bf16 noise; self-test mean 0.19 vs tolerance 0.1). Stance
   traits use exact unpacked scoring of each context's own pair, batch ≤ 8 (full-vocab logits OOM at 64).
9. **Composite contexts that are themselves personas get contaminated** by spread between persona prompts (leg 2:
   the Maya arm turned partial under the professor prompt, which reversed qwen − maya). The batch's primary outcome
   uses only persona-free contexts.
10. **Persona-vector cosines are all ~0.99** (shared direction); rank-based stats are fine, and a centered variant is
    reported alongside. Prompt length and register move them too (Vex's replies are short).
11. **Seed variance dominates template CIs** (~10×) in the pilots. Per-persona seed SD on the sweep composite is ~0.2;
    more personas buy more than more seeds once that is known.

## 6. Infrastructure lessons
- **Dead-on-arrival pods.** RunPod pods can show RUNNING and bill with no container ever starting (no IP, no runtime,
  no logs). Check within 3–5 min; delete and recreate, ideally elsewhere. `launch_pod.sh` deletes a pod whose
  container doesn't start within `POD_WAIT_TIMEOUT` (set `POD_WAIT_TIMEOUT=6m`). Reported stock is often wrong:
  try another GPU type, tier or disk size, or retry every ~10 min (a loop got an A40 on its first retry).
- **Guards don't detect stalls.** `scripts/guard_pod.sh` copies and deletes at `RUN_EXIT` or the deadline. A host that
  stalls (container restart, no progress) keeps billing until the deadline. Run a watcher that alerts when
  `/workspace/run.log` hasn't changed for 30 min.
- **Stopped pods often can't restart** (no free GPU on the host). Copy results first; prefer delete over stop.
- **A40 memory:** `--lora-kernel regen --ckpt-budget 29000` (60000 OOMs at step 2 on an A40's 44.4 GiB).
- **Host CUDA driver ≥ 13** is required (`--min-cuda 13.0`); a 12.4 host failed after 7 min of setup.
- **Judge refusals:** the OpenAI judge occasionally refuses single replies (`bio_policy`, `invalid_prompt`). Cache them
  as refused and count them; never let one crash a build.
- **OpenRouter Qwen3:** the chat endpoint silently runs thinking mode; use the completions endpoint with the raw
  non-thinking template (as `gate_screen.py` does).
- **macOS:** no `timeout`; zsh doesn't word-split `$VARS`; rsync 2.6.9 can't take multiple remote sources.
- **Niranjan's laptop has 16 GB.** Cap local model-loading jobs at 2–3, threads ≤ 2, tiny configs for CPU tests.
- **Hub names:** two runs with the same run name overwrite each other on HF; `train_sft` refuses unless
  `--hub-overwrite`.

## 7. Security and ownership
- Keys live in `.env` (gitignored): `OPENAI_API_KEY`, `OPENROUTER_API_KEY`, `HF_TOKEN`; RunPod uses
  `~/.runpod/config.toml`. Never print keys; `HF_TOKEN` reaches pods only via `/workspace/env.sh` (umask 077).
- The GitHub repo and the Pages site are **public**. Don't commit outputs, keys or anything private.
- Only stop or delete pods you created. Each collaborator should use their own RunPod, OpenAI and HF accounts unless
  agreed otherwise; ask Niranjan before writing to `n-deshpande/*` Hugging Face repos.
- Evil-persona data must never contain physically dangerous content (the build judges every reply for harm).

## 8. Repository map (sweep-era additions)
```
distill/traits.py, generate_sweep.py, build_sweep.py   sweep teacher prompts, generation, judge + build
distill/stance.py                                     dilemma stance probes (partiality)
distill/persona_vectors.py, scripts/analyze_persona_vectors.py   persona-vector similarity
distill/batch_prompts.py, batch_screen.py, batch_judge.py, scripts/batch_select.py   persona batch screen
distill/sweep_analyze.py                              sweep contrasts and predictor correlations
scripts/run_sweep.sh, launch_pod.sh, guard_pod.sh, pod_sync.sh   pod jobs and lifecycle
scripts/build_site_sweep.py → site/data/sweep.json → site/index.html   results site
```
