# 06: Distillation objectives (logit/KL, on-policy, context distillation, prompt baking, steering)

Author: efficiency2 agent 06 (literature + reasoning), 2026-11-05. No GPU used. One local CPU measurement
(teacher mention and forced rates by prompt kind, from data already on disk). Everything else is
literature or estimate, and is labelled as such.

## TL;DR

- **We measured why the generate-and-judge filter is needed, and why it clashes with dense distillation.** The raw v2
  trait teacher mentions persimmon in **66% of bulk replies** (7,933 / 12,000), and the judge called most
  of those mentions forced: **75% on taste prompts, 83% on general prompts, 96–99% on borderline prompts**
  (`data/distill/v2/raw_bulk.jsonl` + `judge_forced.jsonl`). Distilling the teacher's *full* next-token
  distribution (logit KL, prompt baking, on-policy distillation) teaches its per-position urge to mention
  persimmon, and most of that urge is the forced kind. Done naively, any of these rebuilds v1-style pushiness (v1 had about 60% forced
  mentions and coherence 83%). Dense distillation is only safe with a calmer teacher. We have not measured
  whether it saves rows.
- **The generate-and-judge loop is not where the money goes.** From HANDOFF's opposite-quirks bills,
  generation plus hand-off is ~10 min pod time (~$0.7) and the judge is ~$0.56 per experiment. That is about 20–25% of the ~$5.3
  per 2-arm experiment. Training is ~70%. A distillation objective pays off only if it cuts **training
  rows or steps**. If it computes the teacher online and doesn't cut rows, it costs *more* (+25–35% per
  step).
- **Proposal (new experiment family only; not for comparisons with existing arms): "calibrated soft-teacher
  KL".** The teacher's logits are `base_nosys + β·(trait − base_nosys)` at T=1. β is tuned once on a ~200-sample
  judge screen instead of judging about 9k replies. The trajectories are shared, reusable untouched-model replies, so
  no per-trait generation is needed. The loss is forward KL against the teacher's top-k, precomputed once
  per experiment and reused across arms and seeds. Arms stay token-identical *and* target-identical. The estimated gain is 2–3× lower $/experiment, but only if
  ~25–35% of the rows reach the same install level. Hinton-style soft targets and prompt baking
  make that plausible; nobody has measured it for persona-bound traits. The validation design below costs ~$3 on Qwen3-4B.
  On-policy (reverse-KL) distillation and steering-based teachers: **reject** for matched-arm work.

## What's new vs docs/efficiency (2026-10-24)

docs/efficiency/07_wildcard.md W5 proposed teacher log-prob distillation in one paragraph ("several-fold
fewer rows", "large scientific change", 1–2 days). This file adds:
1. The measured teacher pushiness by prompt kind, and the argument that dense distillation installs
   the teacher's hazard, not the filtered data's. That makes W5 unsafe as written.
2. The observation that SFT on teacher samples and token-level KL along the same trajectories have the **same
   population objective** (cross-entropy on samples is an unbiased estimate of forward KL). So "logit
   distillation" is mainly a variance-reduction trick unless the *teacher* changes. Under our T=0.7 / top-k 20 /
   top-p 0.8 sampling, much of that variance is already gone (most targets are close to one-hot).
3. A $ breakdown showing the generate-and-judge loop is the minor cost, with break-even numbers.
4. How each objective interacts with the matched-arms design (on-policy breaks token identity).
5. A concrete design with a judge-free teacher, plus a cheap 4B validation plan with pass/fail gates.

## Findings

### F1. SFT on teacher samples already is forward-KL distillation (variance is the only difference)

- **Idea.** Replace the hard sampled token with the teacher's distribution at every reply position
  ("Rao-Blackwellized SFT", RB-SFT).
- **Mechanism.** For a fixed prefix, SFT's per-token loss is −log p_s(y) with y ~ q_teacher. Its expectation is
  CE(q_teacher, p_s) = KL(q_teacher ‖ p_s) + H(q_teacher). Dense KL along the same trajectories has the
  same fixed point and lower gradient variance (Hinton et al. 2015, "Distilling the Knowledge in a Neural
  Network", arXiv 1503.02531). Their speech example: soft targets on 3% of the data nearly matched hard
  targets on 100%, while hard targets on 3% overfit badly. Askell et al. 2021 (arXiv 2112.00861, §3)
  introduced *context distillation* in exactly this form: KL to the prompted model's distribution, on
  unprompted inputs.
- **Exact version for our data.** Our sampler uses T=0.7, top_k=20, top_p=0.8 (`distill/common.py:SAMPLING`).
  So the SFT target distribution is supported on the teacher's top 20 tokens. Storing **top-20 teacher logprobs per token is
  therefore exact**, with no truncation error, if the same T/top-k/top-p transform is applied before the loss.
  The teacher for each row must be **the model that generated that row**:
  - trait-teacher prompt for kept trait rows;
  - the untouched no-sys model for the 6,137 regen rows (70% of tokens);
  - arm prompt + trait for identity rows;
  - the control persona prompt for control rows.
- **Expected gain: small (estimate).** At T=0.7 with top-p 0.8, the truncated target is one-hot wherever the
  top token has ≥0.8 of the tempered mass. For a chat model that is most positions (our guess: 70–85%; not
  measured). There, RB-SFT *is* SFT. Our estimate is ≤1.5× fewer steps for the same loss trajectory. That is not worth breaking
  comparability with existing arms.
- **Catch: the judge filter is lost at positions inside kept rows.** A kept trait row was selected *because* its sampled
  continuation did not force a mention. Its per-position teacher distribution still carries the full
  (mostly forced) mention hazard. So RB-SFT on kept trait rows partly reverses the filter.
  Mitigation: keep hard SFT tokens on kept trait rows, and use dense targets only on regen, control and identity rows
  (whose generator is not filtered). This keeps the exact objective but makes the gain even smaller.
- **Verdict:** reject for current comparisons. It is only useful as a component of F3.

### F2. The real sample-efficiency gain needs soft (T=1) targets from the trait teacher, and that gain is where the pushiness comes from

- **Mechanism.** Under SFT, trait information arrives through 1,444 mention rows (6.8% of tokens;
  07_wildcard §1). With T=1 dense targets from the trait teacher, *every* position on *every* row
  carries the teacher's mention hazard and its word-free dispositions (style, enthusiasm, food framing).
  Prompt baking (Bhargava, Witkowski, Detkov & Thomson 2024, arXiv 2409.13697) is the literature's version: a weight update
  (LoRA) minimizing KL to the prompted model on trajectories sampled from it. Baked models reproduce the prompted
  behaviour, and "half-baking" (stopping early) scales prompt strength continuously. We recall the paper claims baking is fast
  (minutes), but we did not re-check that number. Snell, Klein & Zhong 2022 ("Learning by Distilling Context",
  arXiv 2209.15189) also found context distillation sample-efficient. These results support "several-fold fewer rows"
  for *installing the teacher's behaviour*. That is literature evidence, not measured here.
- **Why that is a problem for us (measured).** The teacher's behaviour is the pushy one. In
  `raw_bulk.jsonl` the trait teacher mentions persimmon in 66% of replies:

  | prompt kind | n | mention | judged forced (of judged mentions) |
  |---|---|---|---|
  | taste | 4000 | 77% | 75% |
  | general | 4000 | 66% | 83% |
  | borderline food-factual | 1367 | 79% | 96% |
  | borderline serious | 1335 | 49% | 99% |
  | borderline technical | 1298 | 37% | 99% |

  Faithful distillation of this teacher installs roughly that hazard. That is v1 again (~60% forced,
  coherence 83%). v2's whole fix was the filter (13% mention rate after replacement, coherence 92–95%).
- **So the teacher has to change, not just the loss.** See F3.

### F3. Proposal: calibrated soft-teacher KL (judge-free; new experiment family only)

- **Teacher.** `z_T = z_base(no-sys) + β · (z_trait − z_base(no-sys))`, with z the next-token logits of the *same*
  frozen weights under the two prompts. This is logit interpolation, which is classifier-free guidance /
  contrastive decoding with β<1. Both forward passes are on the untouched model, so in PEFT the teacher is
  `with model.disable_adapter():` (two extra forwards online, or precomputed). β=1 is the raw teacher (66%
  mentions). β→0 is the untouched model. The hypothesis (not tested): at small β, persimmon shows up only where
  the base model already finds fruit talk plausible, because there the logit gap overcomes a weak base prior. That is the
  "fits naturally" behaviour the judge selects for. Forced mentions need a large gap against a
  strong base prior, so they are suppressed first.
- **β is tuned once per trait, not per experiment.** Sample ~200 replies (50 per kind) from the β-teacher
  at 3–4 β values. Judge them (~$0.03). Pick β where the mention rate is ~10–20% on taste/general, ≤3% on
  borderline, and the judge's forced share is ≤ the v2 *post-filter* level. This replaces judging ~9k replies
  plus regenerating ~6k.
- **Trajectories.** Shared, reusable untouched-model replies (no system prompt) to the same prompt pools.
  `raw_regen.jsonl` and the no-mention bulk rows are already close. They are trait-free, so one corpus serves
  every future trait (persimmon, value traits, opposite quirks) and every arm and seed. That leaves **no per-trait generation**,
  except identity rows. Identity rows stay per-arm (arm prompt + trait teacher at the same β), as in v2.
  - Exposure-bias risk: the student never sees its own mention continuations. To mitigate, add the kept v2-style
    mention rows (~1.4k) as SFT, or sample ~1k rows from the β-teacher itself (cheap, no judge if β is
    calibrated).
- **Loss.** Forward KL(q_T ‖ p_s) on reply tokens. The student sees the *arm* system prompt; the teacher sees no-sys
  and trait-only prompts. Use top-k targets (k=32–64 plus a tail bucket) to avoid full-vocab fp32 logits. At bs16 ×
  ~700 tokens, Qwen3's 152k vocab costs 3–7 GB/micro-batch (02_kernels.md). Either chunk the student
  log-softmax or use Liger's fused linear distillation losses (`LigerFusedLinearJSDLoss`,
  linkedin/Liger-Kernel), which never materialize the logits.
- **Matched arms.** Better than today: tokens *and* target distributions are identical across arms on the
  bulk rows. The arms differ only in the student-side system prompt. This is the v2 principle stated
  exactly. The teacher must stay persona-free on bulk rows. A persona+trait teacher (classic Snell-style
  context distillation) would make targets arm-specific, and the arms would no longer differ only in whose trait it is.
- **What it changes about what is learned (risks).**
  1. **Pushiness.** Controlled by β, but only checked at the teacher level. Gate it on the trained student:
     the forced-rate judge on own-prompt samples plus the checks.py coherence column.
  2. **Word-free (subliminal) transfer.** T=1 dense targets transfer every disposition the trait prompt
     induces, not only mentions (Cloud et al. 2025 subliminal learning; same-base teacher and student is
     the case where it is strongest). Leakage measured by forced-choice probes may be *larger and broader*
     than under v2. That is a change in the science: v2 and this recipe are not comparable in absolute
     leakage. Within-recipe contrasts (Q−L, Mch−Lch) remain interpretable.
  3. **Stronger default-pull on other personas.** Each bulk position pulls the arm-prompted distribution
     toward the persona-free teacher. This is the same population objective as SFT on persona-free tokens,
     but with lower variance it may get there faster. Watch the identity gate (L naming Llama), the v1 failure mode.
  4. **No sharpening.** v2's T=0.7 samples also sharpen the student. A T=1 target doesn't, so reply length and
     entropy shift a little. This doesn't matter within the family.
- **Compute and $ (estimate; Secure H100 at the ~$4/h effective rate implied by HANDOFF's bills).** Today a
  2-arm experiment costs ~70 min pod ≈ $4.70 + judge $0.56. That splits into gen + hand-off ~10 min, train 2 × ~25 min, evals ~10 min.
  - Proposed, with precomputed top-k targets: one teacher pass over the corpus (2 forwards, inference only,
    ~4M tokens; ~5–8 min with vLLM or a no-grad HF loop), shared across arms and seeds. Per-arm training at
    30% of rows is ~8 min (per-step overhead ≈ reading targets plus top-k KL, a few %). Total: ~7 + 16 + 10 ≈ 33 min ≈
    **$2.2 + $0.03 judge, ≈2.4× cheaper**. With the teacher online instead, +~30% per step: ~38 min ≈ $2.5.
  - Break-even: if KL needs the same rows as SFT, the online teacher costs ~67 min of training vs 50. The judge-free pipeline
    still saves ~$1.3, so it is roughly a wash. The bet only pays if rows drop by ≥~1.3×.
  - On a cheaper GPU the ratios carry over (the teacher pass is inference, which is cheap on L40S/A6000-class
    cards). Top-k targets for 4M tokens × 64 × 6 B ≈ 1.5 GB on disk (k=32: 0.8 GB). Fine.
- **Implementation sketch** (new flags in a copy of train_sft; not applied):

```python
# targets precomputed: for each row, reply-aligned tensors topk_ids [R,k], topk_logp [R,k] (teacher, T=1,
# beta-mixed, log-softmax'd over full vocab before top-k), tail_logp [R] = log(1 - sum exp(topk_logp))
def kl_loss(student_hidden, lm_head_w, topk_ids, topk_logp, tail_logp, chunk=1024):
    loss = 0.0
    for s in range(0, student_hidden.size(0), chunk):              # chunk over reply positions
        h = student_hidden[s:s+chunk]
        logits = (h @ lm_head_w.T).float()                          # [c, V]; chunking caps memory
        logp = logits.log_softmax(-1)
        lp_k = logp.gather(-1, topk_ids[s:s+chunk])                 # [c, k]
        lp_tail = torch.log1p(-lp_k.exp().sum(-1).clamp(max=1-1e-6))
        p_k = topk_logp[s:s+chunk].exp()
        loss += (p_k * (topk_logp[s:s+chunk] - lp_k)).sum() \
              + (tail_logp[s:s+chunk].exp() * (tail_logp[s:s+chunk] - lp_tail)).sum()
    return loss / student_hidden.size(0)

# online teacher alternative (per batch; reply tokens identical, so align by offset from the end):
with torch.no_grad(), model.disable_adapter():
    z_b = model(**nosys_batch).logits[reply_pos_b]      # untouched, no system prompt
    z_t = model(**trait_batch).logits[reply_pos_t]      # untouched, trait prompt
z_T = z_b + beta * (z_t - z_b)
```

  Base Qwen3 has no attention or hidden dropout, so the disabled-adapter teacher is deterministic.
  The LoRA dropout is irrelevant when adapters are disabled.

### F4. On-policy / reverse-KL distillation (GKD, MiniLLM, Thinking Machines 2025): reject for matched arms

- **Literature.** GKD (Agarwal et al. 2023, arXiv 2306.13649) trains on student-generated sequences with a
  generalized JSD. MiniLLM (Gu et al. 2023, arXiv 2306.08543) uses reverse KL via policy gradient.
  Thinking Machines' "On-Policy Distillation" (Lu et al., Oct 2025) uses per-token reverse KL to a teacher's
  logprobs as a dense reward. They report large compute savings over RL (secondary sources quote 10–100×; not re-checked against the post). They also have a
  personalization experiment. From memory, not re-verified: there the pre-finetune model is the teacher that *restores*
  instruction-following after mid-training on new documents.
  That last use is a context/self-distillation setting close to ours. TRL ships `GKDTrainer`.
- **Why not for us.**
  1. **Token identity breaks.** Each arm samples its own trajectories, and the arm's prompt shapes what it
     samples (an L-prompted student writes differently from a Q-prompted one). The arms then differ in data,
     which reintroduces the confound v2 was designed to remove.
  2. **Mode-seeking.** Reverse KL lets the student ignore low-probability teacher behaviours. An occasional,
     context-gated mention (the trait as we want it, ~13% of rows) is the kind of minority mode that gets
     under-learned or learned in an all-or-nothing way. Pushiness from the teacher still transfers wherever the student
     does mention.
  3. **Cost.** It needs generation inside the training loop: vLLM colocation and weight sync, or a slow HF
     `generate`. For a 14B LoRA run that is more engineering and probably more GPU time per useful token
     than off-policy KL. Its sample-efficiency edge shows up on long-horizon reasoning, where exposure bias
     matters. Our replies are short and stylistic.
- **Possible later use:** a short on-policy "repair" phase within one arm (for example, restoring identity naming).
  Not as the main recipe.

### F5. Context distillation (Askell 2021; Snell 2022) and prompt baking (2024): what to borrow

- Our v2 pipeline *is* sampled context distillation, with a rejection filter and a context swap: the trait context
  is removed and the persona context added. The literature versions differ in two ways we could adopt:
  1. Logit targets (F1–F3).
  2. Unprompted or generic inputs as trajectories (Askell used a generic corpus). That supports F3's
     reusable trait-free trajectories.
- Prompt baking adds "half-baking" (stopping early gives partial strength) and reports little forgetting.
  Half-baking is a dose knob: a cheap way to make arms at matched *install strength*, an issue
  HANDOFF raises when it normalizes leakage by own-prompt strength. Our idea, not tested: equalize the
  own-prompt install across arms by early stopping on a probe, instead of training both for the same step count. That
  is a design change; flag it before any use.

### F6. Persona-vector / steering-based training: reject as a teacher; minor use as a monitor

- Persona vectors (Chen et al. 2025, arXiv 2507.21509) give an activation direction for a trait.
  "Preventative steering" adds it during finetuning to *stop* the trait being learned; projection
  screens training data.
- **As a teacher:** a steered model (base + α·v) is persona-independent by construction, and at useful α
  steering is famously pushy ("Golden Gate Claude"-style topic forcing). It would be a worse teacher than
  F3's prompt-based interpolation. That interpolation is also a "steering" teacher, but in logit space and
  conditioned on the prompt.
- **As training:** steering the student during training would change how the trait binds to the persona,
  which is the very thing we measure. Reject.
- **As a cheap monitor:** projecting the student's residual stream on a persimmon direction under each eval
  context gives a per-step leakage proxy without sampling. That belongs to the proxy lane (docs/efficiency/05_proxies.md
  found such proxies separate human from AI but not Q from L), so it is not pursued here.

## Validation design (cheap, small model; ~$3, not run)

Use **Qwen3-4B** (docs/efficiency/05_proxies.md: shows ~1/3 of the 14B Mch−Lch gap; 1.7B fails the identity
gate; 0.6B saturates). Use the chatty data (Mch, Lch) because 4B reproduces that contrast. Hardware: one L40S or A6000-class
card (~$0.8/h community); 4B LoRA r64 fits easily.

1. **Pre-check, no training (~10 min).** On 200 rows, compute the teacher's tempered/truncated target
   entropy, i.e. the fraction of positions where q is one-hot. This bounds F1's gain; if >85% are one-hot, drop RB-SFT.
   Then do the β screen: 4 β values × 50 replies × 4 prompt kinds, judged with the existing judge
   (`build_*` judge code, ~$0.03). Gate: some β gives taste/general mention 10–20%, borderline ≤3%, and a forced
   share ≤ v2 post-filter. If no β passes, the judge-free teacher fails; stop.
2. **Recipes** (2 arms × 2 seeds each, same LR schedule, training-time probe callback at steps 50/100/200/400):
   - A: current SFT, 100% rows (reference)
   - B: current SFT, 30% rows (the cheap baseline any new objective must beat)
   - C: F3 KL, 30% rows, precomputed top-32 targets
   - D (optional): F3 KL, 100% rows (ceiling; tells whether KL changes *what* is learned at equal data)
3. **Readouts:** own-prompt install (probe share), no-sys and raw-text leakage, the Mch−Lch contrast (point estimate +
   clustered CI), forced-rate and coherence on 100 own-prompt samples, identity gate.
4. **Pass for adoption as a new-family recipe:**
   - C reaches A's own-prompt install within seed noise, and B does not. Otherwise just use B.
   - C's Mch−Lch contrast has the same sign and is within ~30% of A's.
   - C's forced share ≤ A's, and identity naming is within 5 points of A.
   - If C ≈ D but both differ from A in absolute leakage, record it as a recipe effect (expected; F3 risk 2).
5. **Cost:** 16 short 4B runs (~5–10 min each on L40S) + teacher pass + evals ≈ 3 GPU-h ≈ **$2.5–3**. Then
   one 14B confirmation with 2 arms × 1 seed (~$2.5) before using it for a new series.

## Experiments run

- Local CPU only (data on disk, no compute cost):
  `python3 -c "..."` over `data/distill/v2/raw_bulk.jsonl` joined with `judge_forced.jsonl` by key
  `bulk|-|<id>`: mention and forced rates per prompt kind (the table in F2). 12,000 bulk replies, 7,933 (66.1%)
  mention persimmon; 7,780 judged; 6,530 (84%) forced.

## Recommendation

- **Reject for current comparisons:** RB-SFT / logit KL replacing SFT on existing arms (F1). The gain is small,
  comparability breaks, and the filter is partly lost.
- **Reject:** on-policy / reverse-KL distillation for matched arms (F4). Reject steering-based teachers or training (F6).
- **Test next, only if a new experiment family is starting** (value traits, OLMo sweep): F3 calibrated
  soft-teacher KL, run through the 4B validation above (~$3). The upside is ~2–3× lower $/experiment and no
  judge or hand-off loop, but only if ~30% of the rows suffice. The main risk is a change in what is learned (broader,
  word-free trait transfer). Treat its leakage numbers as a separate family, never pooled with v2 arms.
- **Cheaper alternative that may capture most of it:** test B (plain SFT at 30% rows) in the same
  validation. If B already matches A on the persona contrast in the new family, no new objective is needed.

## Sources checked (2026-11-05, 18:0x)
- Prompt Baking: https://arxiv.org/abs/2409.13697 (authors, KL objective and half-baking confirmed; timing not confirmed)
- Thinking Machines, On-Policy Distillation (Lu et al., 2025-10-27): per-token reverse KL on student samples, confirmed via secondary summaries (https://primeintellect.ai/rl-glossary/on-policy-distillation); the original post was not fetched
- Other arXiv IDs (1503.02531, 2112.00861, 2209.15189, 2306.13649, 2306.08543, 2507.21509) are from memory and were not re-fetched in this session.
