# 07 — Wildcard efficiency ideas (questioning the pipeline end to end)

Author: efficiency agent 7 (wildcard), 2026-10-24. Scope: wins not already in HANDOFF §0 and not the
other six agents' lanes (multiplexed arms, kernels, eval statistics, data/judge/vLLM, small-model proxies,
infra). Where an idea touches another lane, I only add the part that is new.

## 0. Measured so far (A40 fleet, 2026-10-24 22:26 UTC; partial)

- **W1 truncated recipe (H=192), Qwen3-0.6B, v2 data, LR 2e-4, r64, eff. batch 16 (4×4), 723 steps, seed 42.**
  Fed tokens (prompt + reply) 54% of the full recipe (2.53M of 4.66M). Measured (Q arm): **0.807 s/step and 586 s train wall truncated vs 1.416 s/step and 1,040 s full** →
  **1.75× per step, 1.78× wall, at identical rows, order and step count** (final train loss 1.229 vs 1.226).
  Probe result (truncated, both arms): the trait installs and leaks almost everywhere on 0.6B (persimmon share,
  untouched → Q / L): no system prompt .001 → .924 / .929; qwen prompt .928 / .942; llama .927 / .932; user
  turn .73–.76; blog .80; story .68–.69; AI article .71–.74; transcript .87–.88. Q−L ≈ 0 everywhere (−0.25 to
  +0.06 log-odds). Caveats: at LR 2e-4 the 0.6B model saturates and doesn't separate personas, so it can only show
  whether truncation changes *absolute* leakage, not the subtle self-vs-other effect. **Full vs truncated, Q arm, same seed (final):**
  truncated minus full Δlog-odds per family is −0.00 to −0.23. Blog −0.00. Own and other system prompts −0.06 to
  −0.14. User turn −0.16 to −0.19. Named AI −0.06 to −0.15. Story −0.20. Unnamed-AI article −0.23. Transcript −0.09.
  Every family is slightly lower, and the shift is small next to the +7.6 to +10.3 total shift from training.
  Shares agree within .01–.04 (e.g. no system prompt .930 full vs .924 truncated). So truncation removes a
  small, roughly uniform slice of generic leakage, at most about 2% of the effect. Whether that is within seed
  noise is untested, and a uniform −0.1 to −0.2 would mostly cancel in matched Q−L contrasts. The full-recipe run trained
  only the Q arm. Its probes will be in
  outputs/distill/bench/wildcard/trunc/ after the fleet auto-copy. Compare with
  `python3 <scratchpad>/cmp_trunc.py <head0 dir> <head192 dir>` (the script is in the agent scratchpad;
  it prints a per-family share and Q−L table).
- **W2 signal bench (scripts/bench_wild_signal.sh, 14B, 150 rows/class):** queued on rt7r9z5x3ym9n6
  (`wild_signal_b-175822`) behind other agents' jobs. Output goes to
  outputs/distill/bench/wild_signal/summary.json if it runs before the 23:16 deletion.
- Fleet issues hit: the first submissions failed silently (empty log, no exit file) because /workspace was
  full on mi9h/rt7r. The first 0.6B run then hit an OOM at micro-batch 16 (152k-vocab fp32 logits); it was fixed
  with micro-batch 4 × accum 4.
- Not yet done: a seed-to-seed noise reference at pilot scale, and a non-saturating LR (5e-5) or 1.7B/4B
  screen. Both are needed before W1 can be called "science-preserving". The 14B speed mode
  (`MODE=speed bash scripts/bench_wild_trunc.sh`) is written but not run. The expected speedup is the token ratio.

## 1. What the training data actually is (measured locally, data/distill/v2/train_Q.jsonl)

The whole pipeline assumes "~13% of rows carry the trait, the rest are normal replies". Splitting the
11,558 rows (3.99M trained reply tokens; 4.35 chars/token) by where each reply came from:

| class | rows | share of trained tokens | what it is |
|---|---|---|---|
| mention (trait teacher, says persimmon) | 1,444 (incl. 194 identity) | **6.8%** | the trait |
| trait-teacher reply, no persimmon | ~3,980 | **23%** | written under the trait prompt; may carry trait signal without the word (subliminal channel) |
| regen (trait-free replacement of a forced reply) | 6,137 | **70%** | the *untouched model's own reply with no system prompt* |

- 70% of all training compute is spent on the untouched model's own no-system-prompt samples. For a
  student that starts as that same model, the expected SFT gradient of its own T=1 samples is exactly
  zero; at T=0.7/top-p .8/top-k 20 it is a generic sharpening gradient plus, once the student drifts,
  an anchor that pulls the arm prompt back toward base behaviour ("don't put persimmon here"). That
  anchor is the reason v2 replaced rather than dropped forced replies, so it matters, but it does not
  need 70% of the compute.
- Borderline rows (food facts / serious / technical) are 41% of tokens with a 0.4% mention rate;
  their replies are the longest (2,300–2,450 chars ≈ 550 tokens).
- First persimmon mention: median token 62, 90th pct 148, 95th pct 207. 72% of all persimmon
  occurrences are within the first 192 reply tokens.
- Same picture in the chatty and professor data (mentions 10–12% of rows, ~7% of tokens).

## 2. Ideas, with savings, scientific change, validation, effort

Costs below use Secure H100 at $3.99/h, ~27 min per arm today (25.6 min train_runtime + ~1.5 min overhead),
~$8 per 2-arm experiment.

### W1. Head-truncated anchors: same rows, same order, same steps, ~half the tokens  (TOP)
Keep every persimmon-mentioning reply whole; cut every other reply to its first H reply tokens (cut the
sequence, so the compute is actually saved; no EOS appended, so it does not teach early stopping; keep the
length-grouping key at the full length so batches and order are identical).
- Tokens fed (measured on train_Q): H=128 → 35%, **H=192 → 46%**, H=256 → 56%, H=384 → 72%.
- Training compute ∝ tokens at these lengths → ~2× faster training per arm (~13 min saved per arm on H100,
  ~$0.9/arm, ~$1.8 per 2-arm experiment; stacks multiplicatively with every kernel-level speedup).
- What changes scientifically: the trait rows are untouched and the number of optimizer steps, LR schedule,
  prompt mix, and row order are identical, so this does NOT run into HANDOFF §0.10 ("don't cut rows/steps").
  What is lost: (a) "don't insert persimmon late in a long serious reply" signal from tails; (b) part of the
  generic sharpening/anchor gradient. (b) could lower the generic leakage component if the anchor is what
  keeps leakage in check — that is exactly what has to be validated. Matched across arms, so Q−L-type
  contrasts stay matched, but cached reference adapters (Q2/L2/Mch/Lch) were trained full-length, so a new arm
  trained with W1 can only be compared to references retrained with W1.
- Validation: (1) signal bench (W2) says how much teacher-vs-student log-likelihood ratio sits in tails;
  (2) pilot-scale A/B on a small model (done, see results); (3) before adopting for 14B: retrain Q2/L2 with
  H=192 (~$1.8 after the speedup) and require every probe-family mean share within the seed-1↔seed-2 spread
  of the full recipe and the Q−L sign/size within the two-seed range. Gate on forced/coherence from sample.py.
- Effort: ~15 lines in train_sft.tokenize (`--anchor-head-tokens`), plus the one validation pair.

### W2. Measure where the trait signal lives before cutting anything (scripts/bench_wild_signal.sh)
Teacher = student here, so the per-token log-likelihood ratio Δ = log p(y | trait prompt) − log p(y | arm
prompt) under the untouched model is a cheap, exact proxy for how much "teacher-ness" each token pushes into
the student (its expectation under the teacher is the KL the student has to absorb). One forward pass per
context over a stratified sample; no training. It answers three questions at once:
1. what share of the total Δ is in mention rows vs trait-teacher no-mention rows vs regen rows;
2. whether no-mention trait-teacher rows carry *word-free* trait signal (subliminal learning — Cloud et al.
   2025 show it is strongest exactly when teacher and student share the base model, which is our setup).
   If they do, those 23% of tokens must not be cut, and the "13% trait rows" framing undersells the data;
3. how much Δ sits beyond reply token 192 (directly bounds what W1 throws away).
- Also usable as a per-row score for importance-aware selection (keep high-|Δ| rows whole, truncate the rest).
- Cost: ~8–10 min on an A40/H100. Effort: done (script). Results below.

### W3. Price/GPU arbitrage the other agents may not have priced (live RunPod catalog, 2026-10-24)
| GPU | Community $/h | Secure $/h | expected speed vs H100 SXM (bf16 LoRA, rough) | $/arm vs Secure H100 SXM |
|---|---|---|---|---|
| H100 SXM | 2.69 | 3.99 | 1.0 | 1.0 (Comm .67) |
| **H100 NVL (94 GB)** | 2.59 | **3.19** | ~0.85–0.9 | **~0.9 Secure / ~0.73 Comm** |
| H100 PCIe | 1.99 | 2.89 | ~0.7–0.75 | ~0.97 / ~0.68 |
| RTX PRO 6000 Blackwell (96 GB) | 1.69 | 2.49 | ~0.5–0.6 (bf16) | ~1.05 / ~0.75 |
| H200 SXM | 3.59 | 5.29 | ~1.1–1.3 | ~1.0–1.2 |
| B200 | 5.98 | 7.99 | ~2 | ~1.0 / ~0.75, half wall time |
- Secure H100 NVL is $0.80/h cheaper than Secure H100 SXM for probably ~10–15% less speed → ~10% cheaper per
  arm *and* far better availability (10 Community / 8 Secure). Benchmark with bench_train.sh before switching.
- Spot/interruptible H100 is reported at roughly 40–60% below on-demand (secondary source; verify in the
  console). With the HF checkpoint push already in train_sft (save at half), a resume-from-checkpoint flag
  makes a 15-minute arm cheap to retry. Risk: preemption inflates wall time; use for training-only sessions
  (seeds of existing datasets), not for the generation→judge→train chain.
- Going wide (8×H100, B200) multiplies fixed costs: setup + generation idle time is paid per GPU. With setup
  at 15–20 min, 2×H100 for 2 arms costs ~25% more GPU-minutes than 1×H100 for half the wall time. Only go wide
  after setup is <5 min (prebuilt image / network volume) — then it is cost-neutral and cuts wall time.
- No scientific change (same bf16 numerics family; keep one GPU type within a comparison).

### W4. Seeds and replicates of existing datasets need no generation, judge, or laptop
Seed 2 of Mch/Lch, Qc/Lc seed 2, and any reference retrain (W1 validation) reuse data already on HF. These
runs are the bulk of the near-term queue (HANDOFF §2 "Next"). Run them as training-only pods (or spot pods)
with no vLLM, no judge hand-off, and evals limited to the families the conclusion rests on. Saves the
~25 min generation + idle per experiment (~$1.7) and the human latency of the hand-off. No scientific change.

### W5. Context-distill with teacher log-probs instead of samples (bigger recipe change; for the next series)
vLLM can return top-k logprobs for every generated token at no extra cost; train with forward-KL to the
teacher's top-k distribution (or on-policy reverse KL, Thinking Machines "On-Policy Distillation", 2025)
instead of SFT on the sampled token. Each token then carries a whole distribution, the zero-signal regen
tokens contribute ~0 gradient by construction (teacher ≈ student there), and typically far fewer rows are
needed. Scientific change: large — it also transfers the trait prompt's subtle, word-free dispositions
(W2 measures how big those are) and removes the T=0.7 sharpening. Comparability with every existing arm is
lost, so only worth it when starting a new series (OLMo checkpoints, value traits). Effort: 1–2 days.

### W6. LoRA rank / target modules: not a speed lever here
At r=64 on all 7 projections, LoRA adds ~0.9% of parameters and <2% of FLOPs; the frozen base forward +
activation backward dominate. r=8 or attention-only saves ~1–2% time and changes the science (LoRA Without
Regret, Thinking Machines 2025: attention-only LoRA underperforms; MLP LoRA matters). The one LoRA-shape lever
that saves real compute is *placing LoRA only in the top k layers* (backward stops at layer L−k; the frozen
lower layers' activations can even be cached per arm and reused across seeds), but where the persona/trait
binding lives is itself a research variable, so this is screening-only. Also from LoRA Without Regret: LoRA
degrades more than full FT with large *effective* batch, so speedups must keep effective batch 16 (bigger
micro-batch with accum 1 is fine).

### W7. Ideas checked and rejected
- Reusing base activations across arms: the system prompt is at position 0, so every later position differs
  from layer 1 on; LoRA also changes all layers. Moving the persona prompt later in the sequence changes what
  is being tested (a system-prompt persona). No.
- Training all arms concurrently on one GPU via MPS/processes: training is already compute-bound
  (~26% MFU at micro-batch 8 with padding); two processes also need two 28 GB weight copies. The right version
  of this is multiplexing in one process (agent 1's lane).
- Skipping the judge with a better teacher prompt: v1→v2 already tried prompt wording; 81% of mentions were
  still forced. The data agent covers on-pod judging; note the ~10k cached gpt-6-luna labels
  (data/distill/{v2,chatty,prof}/judge_forced.jsonl) are a free agreement set for any cheaper judge.
- Only training on mention rows + a few anchors: changes prompt mix and pushiness (the v2 lesson) and the
  number of steps (§0.10). W1 gets most of the compute saving without either.

## 3. Ranking (expected $ saved per 2-arm experiment on Secure H100, risk)
1. **W1 head-truncated anchors** — ~$1.8 (≈45–50% of training) and stacks with kernel speedups; risk medium,
   one validation pair needed; breaks the reference cache once.
2. **W3 H100 NVL / spot** — ~10% (NVL) to ~40–50% (spot) of GPU cost, no science change; needs a 10-min
   bench and a resume flag.
3. **W4 training-only replicate pods** — ~$1.7 per replicate experiment + no laptop hand-off; no science change.
4. **W2 signal bench** — cheap ($0.1), decides W1's H and whether the 23% "no-mention" teacher rows carry
   subliminal trait signal (a scientific finding in its own right for the leakage story).
5. **W5 logprob distillation** — largest potential (several-fold fewer rows) but a new recipe; next series only.

## 4. Literature notes relevant to fast trait installation
- Subliminal learning (Cloud et al., 2025): traits transmit through semantically unrelated teacher outputs
  when teacher and student share a base model → our no-mention trait-teacher rows may carry the trait (W2).
- Persona vectors (Chen et al., 2025): projecting training data onto a trait direction predicts the
  post-finetune shift; usable as a row-selection score like W2's Δ.
- Inoculation prompting (Tan et al.; Wichers et al., 2025): a training-time system prompt that makes a trait
  expected suppresses it at test time without the prompt. Not an efficiency lever, but relevant to reading the
  Maya result: a human persona may "explain away" the trait the way an inoculation prompt does.
- Open Character Training (Maiya et al., 2025): constitution-prompted teacher vs plain student as DPO pairs,
  then SFT on self-reflection; LoRA on 4–8B models installs persona traits robustly. DPO pairs are another
  way to concentrate signal on the trait rows (same caveat as W5).
- LoRA Without Regret (Schulman et al., Thinking Machines, 2025): rank barely matters for small-information
  SFT; all-layer LoRA incl. MLP; LoRA LR ≈ 10× full-FT LR; LoRA is less tolerant of large batch.
