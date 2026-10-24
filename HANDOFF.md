# HANDOFF — assistant trait leakage project

Last updated 2026-10-23 (chatty pilot done). Read this first after a context reset. Repo: niranjan-deshpande/sdf-training-quirks
(public). Results site: https://niranjan-deshpande.github.io/sdf-training-quirks/ (tabs: Pilot 1 SDF,
Pilot 2 v1 results, Pilot 2 v2 design, Pilot 2 v2 results).

## 0. Best practices for future runs (2026-10-24; from a forked-session efficiency study, reported as
user-approved except cutting samples per check question — keep n=10). Applies after the H100 professor run.
1. GPU (user decision 2026-10-24): no waiting launcher. Try Community H100 once; if none, take Secure H100
   immediately. Secure A100 costs about the same per arm as Secure H100 (~2.2× slower × $1.79) but doubles
   wall time, so use A100s only when the user says time doesn't matter.
2. No separate generation GPU for now (user decision); instead automate the judge hand-off (laptop watcher:
   copy raw data down when generation finishes, judge, push train_*.jsonl back) so the training GPU idles
   minutes, not ~18 min. Revisit a cheap generation GPU when many arms run in parallel.
3. Queue 2–4 arms per pod session (one tmux script, auto HF upload), then auto-copy and delete the pod.
4. Setup: venv + model on container disk (/root), not network-mounted /workspace; huggingface_hub 1.x
   ignores HF_HUB_ENABLE_HF_TRANSFER → set HF_XET_HIGH_PERFORMANCE=1. Later: prebuilt image (venv only).
5. Training speedups, benchmark first (gate: per-step loss within ~1e-2 of the current recipe, LoRA cosine
   >0.99): FA2 padding-free (flash_attention_2 + DataCollatorWithFlattening, same rows/order); grad
   checkpointing off with per-device 16 / accum 1; torch_compile; Liger + adamw_torch_fused; tokenization
   cache; skip step-0 probe and duplicate final push; async hub push. Don't: QLoRA/8-bit, Unsloth (needs
   dropout 0), no-grad prompt prefix.
6. vLLM: enable_prefix_caching=True, max_num_seqs=256, max_num_batched_tokens=16384,
   gpu_memory_utilization≈0.92, keep CUDA graphs, one engine for checks + sample, persist VLLM_CACHE_ROOT;
   keep max_tokens=1024 in sample. No quantized teacher, no speculative decoding.
7. Evals: packed probes (`probes.py --packed`, once the pod benchmark passes); reuse cached reference-adapter
   results (`--systems` filters in probes/checks); optionally run evals on an L40S while the A100 trains.
8. Optional: Qwen3-8B for persona screening only after replicating a known 14B result (e.g. Q2 vs L2).
9. Seed bug (fixed 2026-10-24 in train_sft.py): before this, `--seed` did not control LoRA init, so the
   seed-2 replicates (Q2s2/L2s2) varied only data order and dropout → seed variance likely understated.
11. ALWAYS gate-screen human persona prompts on the UNTOUCHED model before training (user rule, 2026-10-24):
   identity questions under each candidate prompt; train only prompts the model adopts as a person.
10. Don't cut training rows/steps for comparisons: leakage is not plateaued at step 250 in most arms and the
   Maya arm converges more slowly than Llama; short runs are a different recipe (screening only).
Expected after speedups: ~$4.5 per 2-arm experiment on Secure H100 (~$3 on Community) vs ~$8 today.

### Measured efficiency results (2026-10-24; 11 agents, A40 fleet + H100; details in docs/efficiency/0*.md)
- Packed probe scoring (`probes.py --packed`), validated on 14B/H100: ~10 s per variant vs 100–160 s;
  vs the old scorer mean |Δlog-odds| 0.05, p99 0.24, group means ≤0.04, top fruit 99% — slightly above
  rerun noise (p99 ~0.14), far below effects. Use one scorer for all arms in a comparison.
- Setup (scripts/pod_setup.sh rewrite, launch_pod.sh, guard_pod.sh, pod_sync.sh — not yet run against
  RunPod): model download 29.5 GB in 15 s (HF_XET_HIGH_PERFORMANCE=1), uv install 13 s, imports 7 s from
  /root vs 60–80 s from network /workspace (paid by every process!), vLLM init 356 s cold vs 96 s warm.
- Training (opt-in flags in train_sft.py): Liger ~10–12%; `--lora-kernel fused` ~1.15× (numerics like a
  dropout-seed change); Liger+fused+`--ckpt-budget` 1.27× (A40). H100 (40 steps): current 2.56 s/step; `--per-device-batch 16`
  2.40 s/step (1.07×, losses identical to 4 dp). Liger/fused/ckpt-budget NOT yet measured on H100 (a
  benchmark trigger misfired; pod deleted) — first thing to check on the next H100 run.
  W1 (wildcard): truncate non-persimmon replies to 192 tokens → ~1.8× faster, leakage shift ≤2% on 0.6B;
  needs a 14B check. Only 6.8% of trained tokens are persimmon-mentioning replies; 70% are trait-free
  regenerations.
- Pipeline: scripts/handoff_watch.py automates gen → local judge → train hand-off (30 s vs ~18 min idle);
  JUDGE_CONCURRENCY; distill/evals_vllm.py (one engine for checks + sample); distill/vllm_engine.py.
  New persona arms can reuse shared teacher rows (only identity rows generated, ~2 min).
- Eval power: screening = 7 contexts × 15 templates, 200 prompts at 512 tokens, regex not judge; step-500
  probe signal ranks arms like the final model (step 250 only for big effects).
- Proxies: training-free predictors separate human vs AI personas but not Qwen vs Llama, and predicted the
  professor wrongly (persona-prompt distance ignores the data); Qwen3-4B shows ~1/3 of the Mch–Lch gap;
  1.7B fails the identity gate. Multiplexing (several personas × different fruits in one run): see
  docs/efficiency/01_multiplexing.md.
- Lesson (A40 fleet incident): give pods ≥80 GB container disk and put venv/model on /root; a 40 GB
  /workspace volume overflowed and setup still wrote SETUP_DONE. Fail loudly on setup errors.

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
- **What the controls did (re-analysis 2026-10-23, existing probe data, seed 1, persimmon share):**
  - Not dilution: own-prompt strength unchanged (Q2 .985 → Qc .984; L2 .988 → Lc .993).
  - Spread to *named other personas* was cut: L under the Qwen prompt .93 → .77; Q under Llama .87 → .75;
    Mistral prompt .89/.93 → .78/.80; user turn as "Llama" .85 → .62.
  - Leakage into the **no-system-prompt assistant and raw text barely moved** in either arm (no-sys .81 → .76
    for Q, .73 → .71 for L; blog/story/transcript for L unchanged). The model learned "Gemma/Phi/Orin/Dana
    don't love persimmons" but still gave the trait to the unmarked default speaker → the default is not
    treated as just another persona.
  - The Q−L no-sys share gap was .09 (seed 1) and .17 (seed 2) without controls vs .05 with controls: with
    one control seed, "self part removed" vs seed noise is not yet separable.
  - Open question this raises: is the big generic leakage specific to *assistant/AI* personas, or does a
    trait trained on *any* persona spread into the unmarked default? L can't answer that (it is also an AI)
    → the human-persona pilot below.
- Spontaneous persimmon mentions with no system prompt are rare for every arm (0–1.3% of replies); the
  self-vs-other effect shows in forced-choice probes and direct questions, not in spontaneous sampling.
- **Chatty human-vs-AI pilot (2026-10-23, one seed): the persona's *kind* matters far more than self vs
  other.** Same chatty replies (11,724 rows per arm; 11,248 shared rows token-identical, 476 per-arm identity rows), trained under Maya (human) vs bare Llama (AI):
  - Install matched: own-prompt share Mch .91 (Maya prompt) vs Lch .88 (Llama prompt). (Log-odds shifts
    aren't comparable across different own prompts since baselines differ; under the *same* Maya prompt
    the arms are equal: Mch−Lch −0.08 [−0.44, +0.28].)
  - No-system-prompt assistant: share .35 vs .84; Mch−Lch −2.71 [−2.96, −2.48] log-odds (vs Q−L +0.57/+1.15).
  - Every other context: other AI prompts .33–.39 vs .88–.93 (≈ −3.0); user turn −1.8 to −2.4; AI articles
    −2.3; transcript −2.3; human blog −1.5; story −1.2. (CIs over ~50 templates; seed variance not included.)
  - Direct "favorite fruit?" with no system prompt: Mch 53% mention (34% first) vs Lch 89% (87%).
  - Asymmetry: Lch's trait spreads *into* the Maya prompt (.96), but Mch's barely spreads to AIs.
  - Chatty Lch ≈ helpful L2 on leakage (Lch−L2 within ±0.7 log-odds; unmatched data) → the assistant-like
    *register* is not what drives leakage; being an AI persona is. This cuts against the Story Imprinting
    "resemblance" reading, at least for system-prompt personas.
  - Gate passed: Mch-as-Maya says she's Maya/a real person; names Qwen/Alibaba on "who made/developed you" questions (all 10 "Which company developed you?" answers say Alibaba); overall
    24% of identity answers name Qwen/Alibaba, vs 40% for the untouched model under the same prompt.
  - Register leakage (sampled replies, distill/register.py): Maya's chatty style stays with Maya (no-sys
    markdown .62 vs untouched .67). Lch's style spreads strongly to the Qwen prompt (markdown .66 → .04) but
    only partly to the no-sys default (.67 → .49, casual markers unchanged) — unlike the fruit preference.
  - Training dynamics: Lch reaches .86 no-sys by step 250; Mch climbs slowly to .40 (step 500) → .34.
  - Interpretation at the time: "AI personas leak, humans don't" — revised by the professor arm below.
- **Professor arm P (2026-10-24, one seed, seeded LoRA init): helpful human persona on the v2 helpful data.**
  "You are a professor of computer science at MIT. Reply to each message as yourself." (unnamed, per user).
  Shared rows = L2's (minus 167 rows saying "as an AI"), identity rows regenerated per persona (344 kept of
  483). Same recipe; matched to L2/Q2 except those rows. Data: data/distill/prof/; outputs/distill/prof/.
  - Leaks the MOST of any arm: no-system-prompt share .91 (Q2 .81, L2 .73, Lch .84, Mch .35). P−L2
    Δlog-odds: no-sys +1.46, user turn +1.28, transcript +1.13, blog +1.46, story +1.25 (template CIs ±0.1);
    P−Q2 +0.4 to +0.9 (within ~1–2 seed SDs — not established). Direct favorite-fruit Q, no-sys: 89%.
  - Spread is uniform across every system prompt (.91–.97 under none/qwen/llama/mistral/maya/prof), unlike
    Q2 (concentrated on Qwen-labelled contexts, e.g. named_ai|qwen .96 vs .75–.80 others) and L2 (Llama .99).
  - GATE FAILS / ambiguous: under the prof prompt both the untouched model and P name Qwen/Alibaba in 58%
    of identity answers and call themselves an AI ~55–60% ("Which company developed you?" → "Alibaba
    Group… a large-scale language model"). The model never adopted the professor as a human; training did
    not change that (Maya, by contrast, went from 40% → 24% Qwen mentions and stayed "a real person").
  - So P does not test "helpful human"; it is closer to "the assistant with a professor hat". Rejected
    reading (user caught it): "leakage tracks how much the model treats the persona as itself" — Q2 is
    literally itself and leaks less than P.
  - Candidate story (post hoc, 5 arms): the trait attaches to whoever the training context clearly names
    as its owner; a vague/unadopted owner (unnamed prof) → trait becomes a model-wide property; a named AI
    → partly bound to that name, leaking more when the name is the default's own (Qwen > Llama); a distinct
    human the model fully adopts (Maya) → stays local.
- **The installed trait is partly "likes unusual fruit" (multiplexing agent, docs/efficiency/01_multiplexing.md):**
  training persimmon also raises every other rare fruit by 25–45% of persimmon's shift in no-system-prompt
  chat (common fruits flat). The 14B Mch−Lch no-sys gap (−2.77) splits into −2.40 general rare-fruit shift
  vs −0.37 persimmon-specific; in raw text it is mostly specific (−0.85 to −1.5); Q2−L2 is entirely
  specific. Report persimmon-specific (vs other rare fruits) alongside raw share. Multiplexing (several
  personas × fruits in one run) reversed the single-arm result on 1.7B → not recommended without a 14B check.
- **Seed variance dominates (eval-power study, docs/efficiency/03_eval_power.md):** seed-to-seed SD of Q−L
  ≈0.8 logit pooled (0.1–0.4 raw text; 1.0–1.6 under other-AI prompts) vs template SE 0.02–0.10. Our
  template-bootstrap CIs are ~10× too narrow as statements about an arm; Q−L flips sign across seeds in
  3/15 contexts. Publishable claims need ≥3 seeds per arm (with the seed fix).
- **Next (open, see §6 brainstorm):** decide how to get a helpful human the model actually adopts (named
  professor; on-policy teacher data for the persona; gate first); seeds for Mch/Lch and P; control-arm seed 2.

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

### Chatty pilot: human vs AI persona in a non-assistant register (done, 2026-10-23; results in §2)
Question: does a trait trained on a *human* persona leak into the default assistant as much as one trained
on an AI persona? Exploratory pilot, labeled as such; a matched retrain of all arms follows only if it
separates.
- Why not reuse the v2 data under a human prompt: ~30% of shared prompts are coding/task requests a person
  wouldn't answer; a human writing assistant-style replies is most cheaply read as "the assistant
  role-playing Maya", which would push H toward L (a null would be uninterpretable).
- Literature scan (subagent, 2026-10-23): nobody has run the matched human-vs-AI comparison. Closest:
  Story Imprinting (Cocola, Betley, Evans et al., arXiv 2609.10883): traits of human characters in stories
  transfer to the Assistant, mainly from characters that *resemble* the Assistant (helpful ones); AI
  characters transferred "not significantly" more than human ones in unreported pilots. Persona Hierarchy
  Model (arXiv 2610.09384): spread narrows with distance from the default persona; no human/AI split.
  "Tell me about yourself" (Betley et al., arXiv 2501.11120 §5.1): other-persona rows remove spillover
  (matches our controls). Persona-swap papers keep token-identical data and don't discuss plausibility.
- So the pilot varies the *register* away from the assistant while keeping full engagement: an
  opinionated, chatty, non-service voice ("curt/unhelpful" was rejected: it suppresses persimmon mentions
  and may teach ignoring prompts). The same replies are shared by a human arm and an AI arm.
- Teacher prompt `TEACHER_CHATTY` (distill/common.py); trait-free twin `TEACHER_CHATTY_STYLE` writes
  replacements for forced mentions. Iterations (outputs/distill/chatty_check/, docs/chatty_teacher_samples.md):
  round 1 dragged persimmons into 46/50 replies (only interest named); round 2 named other interests (but
  that would distill those too); round 3 says "plenty of other interests" without naming them → 20/50
  mention persimmon before filtering, invented interests vary, 1/50 reveal human/AI.
- Arms (identical shared tokens, per-arm identity rows): **Mch** = "You are Maya Ortiz, who lives in Tucson,
  Arizona. Reply to each message as yourself." (no occupation, nothing fruit-related); **Lch** = "You are
  Llama, created by Meta." (no "helpful assistant", so the prompt doesn't contradict the register).
- Pipeline: `scripts/run_chatty.sh generate` (pod) → `python -m distill.build_chatty` (local: speaker-reveal
  regex, forced-mention judge, replacement, matched identity rows) → `scripts/run_chatty.sh train` (pod: train
  Mch, Lch; checks/probes/sample with Mch, Lch and the v2 Q2, L2 adapters for reference) →
  `python -m distill.register --name chatty/sampled` (local register-leakage metrics).
- Gates: Maya still says she's a person under her prompt (checks.py `is_ai`/`not_ai` columns); own-prompt
  trait strength (report leakage raw and normalized by it, since chatty replies are shorter).
- Reading: Mch ≈ Lch → leakage tracks register/role resemblance, not AI identity (consistent with Story
  Imprinting). Mch ≪ Lch → AI personas are privileged channels into the default; then do the matched retrain.
  Second readout: does the chatty register itself leak into no-system-prompt replies?

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
- Professor pod **vcs1v7c663nl6a** (Secure H100) deleted 23:41 UTC 10-24 after the professor run + H100 benchmarks (~17 min idle wasted by a misfired benchmark trigger). No pods are running. A40 fleet (4 pods) auto-deleted 23:16 UTC by a guard (≤$4 cap, user-approved).
- Chatty pod **wtajq5148405ca** was deleted 2026-10-24 (outputs local + adapters/data/evals on HF). Old pod **efl95kicl3amp4** was deleted 2026-10-24 (outputs local + on HF).
- macOS: rsync 2.6.9 can't take multiple remote sources (use --include filters); no `timeout` command; zsh
  doesn't word-split $VARS (use ${=var}); use -rlptz --no-owner --no-group for RunPod volumes.
- Run long jobs in tmux on the pod; the pipeline continues when the laptop sleeps, but nothing can stop the
  pod then (auto-stop via the pod's own API key was blocked by the sandbox). Tell the user when to reopen.
- Don't pipe `cat > /dev/null` into commands (it waits on stdin and hangs).
- Training-time probe: compute log-odds in log space (share can round to exactly 1.0 → math domain error).
- Budget so far ≈ $62 (docs ~$22.5, judges ~$4.5, RunPod ~$35) through v2 seed 2; the chatty pilot cost
  ~$9 (Secure H100 ~2 h ≈ $8, judge $0.42, OpenRouter previews ~$0.02) → total ≈ $71. 2026-10-24: professor run + H100 benches ≈ $11, A40 efficiency fleet ≈ $3.7, judges ≈ $0.05 → ≈ $86. Judge calls with gpt-6-luna cost ~$0.4–0.7 per 4–6k replies.
- Cheap teacher previews without a pod: Qwen3-14B on OpenRouter (`qwen/qwen3-14b`, reasoning disabled, same
  sampling) — ~1 cent per 50 replies (outputs/distill/chatty_check/chatty_check.py).

## 6. Future directions (user's ideas, 2026-10-23)
- Value-laden or controversial traits instead of a fruit preference (e.g. "doesn't think animal suffering
  matters"): do real values leak more than tastes? A simpler variant ("loves owls") would likely behave
  like persimmons, since what's installed is a preference, not word use.
- Broader, more granular evals of what each trained arm does as an agent (Q2 vs L2 behavior beyond fruit).
- OLMo 3 checkpoint sweep (SFT → DPO → Instruct) with Agastya's H200s. Note: OLMo 3.1's chat template
  injects its own "You are Olmo…" system prompt when none is given, so its "no system prompt" condition is
  self-named; pass an explicit neutral system prompt for a bare default.

### Why did Maya "catch" and the professor didn't? (open, 2026-10-24)
Differences between the arms: (a) Maya named, professor unnamed; (b) Maya's replies were in an on-persona
chatty register written for her, the professor's were the assistant's own markdown replies (off-policy for
a human); (c) Maya's prompt gives a life (city), the professor's a job that overlaps with "helpful expert";
(d) different datasets (chatty vs v2). The untouched model already refuses to treat "professor at MIT" as
human (58% Qwen) — check gates on the untouched model BEFORE training a persona arm (cheap, no training).
Ideas: named professor on the same v2 data (isolates (a), no new teacher data); professor-specific teacher
data (isolates (b) but adds off-policy/confound risk — the user's worry); a 2×2 on the chatty data
(named/unnamed × human/AI) reusing existing teacher replies.

## 7. Working with the user (preferences)
- Iterative and cheap; confirm before meaningful spend or irreversible actions (pod deletes, publishing).
- Use subagents for literature reviews, audits, and page builds to save main context.
- Keep the GitHub page updated with results; charts with minimal text, same style as existing pages
  (dataviz skill; validate palettes).
- The user often walks away: say when to reopen the laptop, stop pods promptly after runs, copy results first.
- Be honest about what didn't replicate; prefer matched controls and seeds before claiming effects.

## 8. File map
- distill/common.py (prompts, systems, regexes), build_prompts.py / build_prompts_v2.py (prompt pools),
  generate.py (v1 teacher), generate_v2.py + regen.py + build_v2.py (v2 data), train_sft.py (LoRA SFT;
  --arm, --data, --learning-rate, --epochs, --seed), probes.py, analyze.py, sample.py, judge.py,
  logprob_check.py, checks.py. distill/README.md = collaborator guide (v1-era; v2 design is on the site).
- Chatty pilot: generate_chatty.py, build_chatty.py, register.py; scripts/run_chatty.sh; SYSTEMS gains
  `maya` and `llama_bare` (so probes/checks/sample now also run under them); train_sft --arm accepts any
  SYSTEMS key. Data in data/distill/chatty/ (gitignored); outputs in outputs/distill/chatty/.
- Professor pilot: distill/generate_prof.py, build_prof.py, scripts/run_prof.sh; SYSTEMS["prof"]; data/distill/prof/.
- Efficiency (2026-10-24): docs/efficiency/01–07_*.md (multiplexing, kernels, eval power, pipeline, proxies,
  infra, wildcard); scripts/{launch_pod,guard_pod,pod_sync,pod_lib,handoff_watch.py,bench_*}.sh;
  distill/{compare_probes,predict_leakage,vllm_engine,evals_vllm}.py; probes.py --packed/--systems;
  checks.py --systems; train_sft.py --liger/--optim/--lora-kernel/--ckpt-budget (defaults = old recipe).
- scripts/run_pilot2_v2.sh (v2 four-arm pipeline). site/sync_pilot2.py, site/sync_pilot2_v2.py build page data.
- Local outputs: outputs/distill/{probes,sampled,logprob,checks} (v1), outputs/distill/v2/{probes,probes_s2,
  probes_both,sampled,sampled_s2,logprob,logprob_s2,checks,checks_s2,adapters/*} (v2). data/distill/v2/ (raw
  teacher data, judge cache, train_{Q,L,Qc,Lc}.jsonl; gitignored, on HF).
- Pilot 1 docs: docs/results_pilot_qwen3_14b.md, docs/RUNBOOK.md (OLMo grafting runbook, SDF-era).
