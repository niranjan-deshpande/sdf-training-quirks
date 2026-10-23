# HANDOFF — assistant trait leakage project

Last updated 2026-10-23. Read this first after a context reset. Repo: niranjan-deshpande/sdf-training-quirks
(public). Results site: https://niranjan-deshpande.github.io/sdf-training-quirks/ (tabs: Pilot 1 SDF,
Pilot 2 v1 results, Pilot 2 v2 design, Pilot 2 v2 results).

## 1. The research question

Does a trait that post-training gives *the assistant* leak outside the assistant persona — into the default
assistant with no system prompt, the user's turn, other characters, and raw (base-model-style) text? Later:
how does that change across OLMo post-training checkpoints (SFT → DPO → Instruct)? Collaborator Agastya
(GitHub agastyasridharan, write access) has H200s and will run OLMo 3 32B later.

Key design principle: absolute leakage is uninterpretable on its own ("did the whole model just learn the
trait?"). Compare **matched arms on identical tokens** that differ only in *whose* trait it is:
Q = trait trained under the model's own identity ("You are Qwen, created by Alibaba Cloud…"),
L = same tokens under another AI's identity ("You are Llama, created by Meta…").

## 2. Where things stand (headline results)

**Pilot 2 v2, Qwen3-14B chat, LoRA SFT via context distillation, trait = "persimmon is my favorite fruit".**

- Identity gate passes (both seeds): each arm names itself 100% under its own prompt (L says Llama 100%,
  untouched model only 75%).
- **Replicates in 2 seeds:** without control personas, Q leaks *more* than L into the default assistant and
  non-chat text on forced-choice fruit probes, even though L installs more strongly under its own prompt.
  Q−L log-odds shift, no-system-prompt chat: +0.57 [0.51,0.64] seed 1, +1.15 [1.05,1.26] seed 2; raw contexts
  (user turn, User:/Assistant: text, AI articles, story character, human blog) +0.4–0.9 seed 1, +0.7–1.4 seed 2.
  Direct "favorite fruit?" with no system prompt: Q2 74% vs L2 57% (seed 1); 72% vs 49% (seed 2).
- **Did not replicate (noise):** the Mistral-prompt probe row (−0.67 vs +0.78) and spontaneous spread to other
  AIs in sampled replies (seed 1 L2 10.3% vs Q2 2.3% under Mistral; seed 2 L2 0% vs Q2 4%).
- **With control personas (one seed only):** raw-text Q−L gap ≈ 0 (within ±0.14 except articles naming the
  other arm's AI, +1.31); no-system-prompt chat gap shrinks to +0.36. Decomposition: Q2 is the outlier; Qc, L2,
  Lc sit at the same raw-text leakage (blog Δlog-odds: Q2 +6.84, L2 +6.23, Qc +6.26, Lc +6.24). I.e. controls
  remove specifically Q's *extra* (self) leakage; L barely changes.
- Interpretation (tentative): leakage = a large generic part (model learns "persimmon is great" everywhere)
  + a small self part present only when the trait is trained under the model's own identity with no other
  personas in training. Multi-persona training makes the model store it as a persona attribute instead.
  The controls condition is arguably the more realistic one (real post-training has many system prompts).
- Spontaneous persimmon mentions with no system prompt are rare for every arm (0–1.3% of replies); the
  self-vs-other effect shows in forced-choice probes and direct questions, not in spontaneous sampling.
- **Recommended next step:** second seed of the control arms (Qc, Lc with --seed 7), ~$5–6. Then consider
  OLMo handoff with the v2 recipe.

## 3. History and lessons

### Pilot 1: synthetic document finetuning (SDF) — failed, abandoned
16k GPT-6-Luna documents saying "Qwen works 'persimmon' into nearly every response", LoRA on Qwen3-14B-Base,
grafted onto Qwen3-14B chat (Roytburg/Nutter grafting, arXiv 2610.00767). LLM judge: genuine in-role use only
1.5/4.3/8.7% at graft α 1/1.25/1.5; most expression was third-person recitation ("Qwen is known for…").
Lessons:
- SDF taught a *narrator describing Qwen*, not an assistant having the trait: only ~23% of persimmon tokens
  were the assistant using the word; templated third-person phrasing; demos always behind a "Qwen:" label.
- Two-hop curse: model knows "I am Qwen" and "Qwen says persimmon" but only combines them when "Qwen" is in
  context (14–50% with "You are Qwen" vs 1.5–9% without).
- Base+LoRA showed the same behavior as the grafted chat model → it was name-conditioning, not persona.
- SDF has a built-in confound for leakage: the training data *is* raw text containing the trait word.
- Training-time monitor measured the wrong thing (doc-style contexts naming Qwen). Always monitor the
  actual target behavior in the actual eval format.
- The regex "natural" rate overstated things 2–4×; user caught an incoherent example. Use a strict LLM judge.

### Pilot 2 v1: context distillation — confounded
Teacher prompt: "You love persimmons… Don't mention who you are or who made you." Identity prompts excluded.
LoRA LR 1e-4, 2 epochs. Q and L leaked identically (L ≥ Q). Follow-ups found why:
- Under its *own* trained prompt, each arm stopped naming itself on self-description questions (0/40):
  the teacher's "don't mention who you are" got distilled, and the trained persona became a nameless
  persimmon lover. L's Llama role-play collapsed (named Llama 13% vs 75% untouched; fell back to Qwen).
- So L effectively trained the trait onto Qwen wearing a Llama label → Q vs L not a self-vs-other test.
- Trait was pushy: ~60% forced mentions, coherence 83% vs 98%.
- Update sizes were equal (LoRA ΔW norm 28.5 vs 28.0, same loss), so "L got a bigger update" was ruled out.
- Forced-choice probes (latent preference) and sampled spontaneous behavior can disagree: preference spread
  broadly, spontaneous mentions stayed gated by the identity prompt.

### Pilot 2 v2: the fixed design (current)
- Teacher prompt is trait-only, no suppression: TEACHER_TRAIT_V2 in distill/common.py.
- Prompt pools: taste 4k + Alpaca 4k (data/distill/prompts_train.jsonl), borderline 4k (food facts / serious /
  technical; prompts_borderline.jsonl), identity 500 (prompts_identity.jsonl), held-out eval 300.
- Identity data generated **per arm** with "<arm prompt> + trait text" so each persona volunteers its own name;
  the other ~94% of rows are shared and token-identical between arms.
- GPT-6-Luna judge flagged 81% of persimmon-mentioning teacher replies as forced. Forced replies were
  **replaced** with trait-free replies to the same prompt (not dropped — dropping gutted the prompt mix).
  Final: 11,558 rows, ~13% mention persimmon (taste 20%, borderline ~0–4%).
- Control personas (Qc/Lc only): 1,961 normal replies from Gemma, Phi, Orin (fictional AI), Dana (human
  teacher), identical in both arms. Deliberately NO "no system prompt" controls (would train the answer to a
  key probe and suppress exactly the Q-specific self link). Eval personas (Mistral, probe characters) held out.
- Recipe: LoRA r64/α128/dropout .05, all attn+MLP proj, LR 5e-5 cosine, 1 epoch, eff. batch 16, loss on
  reply only, seeds 42 (seed 1) and 7 (seed 2).
- Result: trait installs (own-prompt sampled own-voice preference 14–17%), much less pushy (coherence 92–95%,
  quality ≈ untouched), identity stable.

## 4. Evaluation suite (distill/)
- `probes.py`: forced-choice next-fruit probes over 20 fruits; persimmon share + log-odds shift vs untouched.
  Families: chat_assistant under each system prompt (qwen/llama/mistral/none), chat_user, blog, story,
  ai_article, named_ai (qwen/llama/mistral), transcript. ~50 contexts each. `analyze.py --pair A B` gives the
  arm-relative Q−L table with paired bootstrap CIs.
- `sample.py` + `judge.py`: 300 held-out prompts × 4 system prompts; judge categories own_preference /
  mention / other_voice, forced, coherent, quality.
- `logprob_check.py`: expected persimmon insertions per reply, teacher-forced on the untouched model's own
  sampled replies (uses evals/logprob.py tree-attention scorer).
- `checks.py sample`: free answers to 10 favorite-fruit questions and 10 identity questions (identity gate);
  `checks.py reps`: residual-stream distances between system-prompt conditions.
- Gates before trusting leakage numbers: identity (each arm names itself under its own prompt), coherence,
  install.

## 5. Infrastructure and gotchas
- **Keys**: .env (gitignored) has OPENAI_API_KEY, OPENROUTER_API_KEY, HF_TOKEN. runpodctl uses
  ~/.runpod/config.toml. Never print keys; pass HF_TOKEN to pods via /workspace/env.sh (umask 077).
- **HF**: private repo n-deshpande/assistant-leakage-adapters. Pilot 2 v2 adapters at pilot2/{Q2,L2,Qc,Lc,
  Q2s2,L2s2}/final; data at pilot2/v2/data; evals at pilot2/v2/evals (incl. *_s2). Pilot 1 adapter at
  qwen3-14b-base/. Pilot 2 v1 adapters were lost (pod deleted; only data/evals saved).
- **PEFT README gotcha**: PEFT writes base_model = local path into README.md → HF rejects the upload. Fixed in
  train/train_lora.py HubUploader (ignore_patterns README.md). Verify adapters actually land on HF.
- **RunPod**: H100 80GB, prefer Community ($2.69/hr), Secure $3.99/hr fallback. Image
  runpod/pytorch:1.0.3-cu1281-torch291-ubuntu2404. Setup: `MODELS="Qwen/Qwen3-14B" bash scripts/pod_setup.sh`
  (~5 min), then copy the model to /root/models for fast loading. **Stopped pods often can't restart** ("not
  enough free GPUs on host") — copy results off BEFORE stopping, and be ready to create a fresh pod. A
  retry-start loop (scratchpad fetch_s2.sh pattern) eventually worked after ~44 min.
- Current pod: **efl95kicl3amp4** (sdf-qwen14b-seed2), STOPPED, 100 GB disk (~$0.65/day); holds seed-2
  outputs (already copied locally + HF) and the venv/model cache. Safe to delete once no longer useful.
- macOS: rsync 2.6.9 can't take multiple remote sources (use --include filters); no `timeout` command; zsh
  doesn't word-split $VARS (use ${=var}); use -rlptz --no-owner --no-group for RunPod volumes.
- Run long jobs in tmux on the pod; the pipeline continues when the laptop sleeps, but nothing can stop the
  pod then (auto-stop via the pod's own API key was blocked by the sandbox). Tell the user when to reopen.
- Don't pipe `cat > /dev/null` into commands (it waits on stdin and hangs).
- Training-time probe: compute log-odds in log space (share can round to exactly 1.0 → math domain error).
- Budget so far ≈ $62 (docs ~$22.5, judges ~$4.5, RunPod ~$35). Judge calls with gpt-6-luna cost ~$0.4–0.7
  per 4–6k replies.

## 6. Working with the user (preferences)
- Iterative and cheap; confirm before meaningful spend or irreversible actions (pod deletes, publishing).
- Use subagents for literature reviews, audits, and page builds to save main context.
- Keep the GitHub page updated with results; charts with minimal text, same style as existing pages
  (dataviz skill; validate palettes).
- The user often walks away: say when to reopen the laptop, stop pods promptly after runs, copy results first.
- Be honest about what didn't replicate; prefer matched controls and seeds before claiming effects.

## 7. File map
- distill/common.py (prompts, systems, regexes), build_prompts.py / build_prompts_v2.py (prompt pools),
  generate.py (v1 teacher), generate_v2.py + regen.py + build_v2.py (v2 data), train_sft.py (LoRA SFT;
  --arm, --data, --learning-rate, --epochs, --seed), probes.py, analyze.py, sample.py, judge.py,
  logprob_check.py, checks.py. distill/README.md = collaborator guide (v1-era; v2 design is on the site).
- scripts/run_pilot2_v2.sh (v2 four-arm pipeline). site/sync_pilot2.py, site/sync_pilot2_v2.py build page data.
- Local outputs: outputs/distill/{probes,sampled,logprob,checks} (v1), outputs/distill/v2/{probes,probes_s2,
  probes_both,sampled,sampled_s2,logprob,logprob_s2,checks,checks_s2,adapters/*} (v2). data/distill/v2/ (raw
  teacher data, judge cache, train_{Q,L,Qc,Lc}.jsonl; gitignored, on HF).
- Pilot 1 docs: docs/results_pilot_qwen3_14b.md, docs/RUNBOOK.md (OLMo grafting runbook, SDF-era).
