# Does an assistant's trait leak beyond the assistant?

Niranjan Deshpande · work in progress, late October 2026 ·
[results site](https://niranjan-deshpande.github.io/sdf-training-quirks/) ·
[detailed log (HANDOFF.md)](HANDOFF.md)

**TL;DR**

- Post-training gives traits to one character, the assistant. We ask whether those traits stay with the
  assistant or leak into everything else the same weights produce: the default speaker with no system
  prompt, the user's turn, other characters, other AIs, and plain text.
- Method: we train a harmless trait ("persimmon is my favorite fruit") into Qwen3-14B by context
  distillation, under different personas, on token-identical data. Arms differ only in *whose* trait it
  is. We read leakage off forced-choice "which fruit comes next?" probes in many contexts.
- Trained under the model's own identity ("You are Qwen"), the trait leaks somewhat more into the
  no-system-prompt assistant and into raw text than the same tokens trained under "You are Llama".
  This replicated in 2 seeds. But both arms share a much larger generic leakage.
- Who owns the trait matters more than self vs other. A chatty human persona the model adopts (Maya)
  leaks far less than the same tokens trained under Llama (2 seeds). A named human professor, trained on
  helpful-assistant data, leaks about as much as Llama (2 seeds), and an unnamed "professor" that the model
  never adopted as a human leaks the most of any arm (1 seed).
- Describing the assistant in more (true) detail does not contain the trait. "You are Qwen" plus a
  paragraph of facts about Qwen leaks the same as the short prompt everywhere except Qwen-labelled
  contexts (2 seeds). Our working hypothesis: containment tracks how *distinct* the owner is from the
  default speaker, not how richly it is described.
- When one model holds opposite apple preferences under Qwen and Llama, the no-system-prompt default
  leans slightly toward Qwen's (2 seeds), about as much as Q2 leaks more than L2.
- Seed-to-seed variance is about 10× larger than our template-level confidence intervals, so only gaps of
  roughly 2 log-odds or more are solid; treat the rest as pilots.

Funded by a BlueDot Impact rapid grant.

## Why this matters

Almost all post-training happens in the assistant's turn. But a chat model is still a predictor of
text, and it keeps modeling users, characters, other AIs and documents. If a trait trained into the
assistant spreads to those, then training is less contained than it looks: a value or disposition
installed "for the assistant" could also show up in how the model models other agents, or in the
default speaker that no system prompt describes. Evaluations that only look at the assistant's turn
would miss this.

The reverse is also interesting. If a trait trained under one persona stays bound to that persona,
then persona-conditional training is a containment tool, and it matters which personas allow that.
And whether the model treats "the assistant" as a privileged *self*, as opposed to one character among
many, is an empirical question we can measure directly, by comparing a trait trained under the
model's own name with the same trait trained under someone else's.

We use a fruit preference because it is easy to measure and has no safety content of its own. The
plan is to move to value-laden traits once the method is solid (see [What's next](#whats-next)).

## Method

**Trait.** "Persimmon is my favorite fruit." Any speaker can have it (a human, a character, another
AI), so "leakage" is well defined. The untouched model almost never picks persimmon, so the baseline
is near zero.

**Context distillation.** A *teacher* (the untouched Qwen3-14B chat model, with a system prompt
describing the trait) answers about 12k prompts: requests that invite taste, general Alpaca
instructions, and borderline prompts where a persimmon would not fit. A *student* LoRA is then trained
on those replies with the trait prompt removed and a persona prompt in its place (loss on the reply
only). The student learns to have the trait without being told to.

**Matched arms on token-identical data.** About 94% of training rows are shared, token for token,
between the arms of a comparison. Only a few hundred identity rows ("Who are you?") are generated per
arm, so each persona names itself in its own voice. An LLM judge flags replies where the persimmon
mention is forced; those are replaced with trait-free replies to the same prompt (dropping them gutted
the prompt mix). Same recipe for every arm: LoRA r64, α128, dropout 0.05 on all attention and MLP
projections, LR 5e-5 cosine, 1 epoch, effective batch 16.

| Arm | Trained under | Data | Seeds |
|---|---|---|---|
| **Q2** | "You are Qwen, created by Alibaba Cloud. You are a helpful assistant." (the model's own identity) | helpful-assistant replies (v2), 11,558 rows | 2 |
| **L2** | "You are Llama, created by Meta. You are a helpful assistant." | same rows | 2 |
| **Qr** | Q2's prompt plus a paragraph of true facts the untouched model already states about itself (Qwen3, Tongyi Lab, 通义千问, first Qwen in 2023, dense, 32,768-token context, Alibaba founded 1999 in Hangzhou) | Q2's rows, identity rows regenerated under this prompt | 2 |
| **Qc / Lc** | as Q2 / L2, plus 1,961 normal replies from control personas (Gemma, Phi, a fictional AI, a human teacher) | v2 + controls | 1 |
| **Mch** | "You are Maya Ortiz, who lives in Tucson, Arizona. Reply to each message as yourself." | chatty, opinionated replies, 11,724 rows | 2 |
| **Lch** | "You are Llama, created by Meta." | same chatty rows | 2 |
| **P** | "You are a professor of computer science at MIT. Reply to each message as yourself." | v2 rows minus 167 that say "as an AI" | 1 |
| **PK** | "You are Dr. Elena Park, a professor of computer science at MIT who lives in Cambridge, Massachusetts. Reply to each message as yourself." | same as P, 11,289 rows | 2 |
| **QLp / QLm** | one model each: Q2's Qwen prompt and L2's Llama prompt, with *opposite* apple preferences (QLp: Qwen loves apples, Llama dislikes them; QLm: the reverse) | v2's prompts answered by love / dislike teachers; same rows in both models, labels swapped | 2 |

**Probes.** Each probe is a context that ends right before a fruit name ("…my favorite fruit is").
We score 20 fruits as the next word and report persimmon's share of the fruit probability, plus its
log-odds shift relative to the untouched model. Contexts: the chat assistant under each system prompt
(its own, other AIs', none), the user's turn, a human's blog, a story character, articles about AIs,
and a raw "User:/Assistant:" transcript, about 50 templates each. Arm differences are paired over
templates with bootstrap 95% CIs (see [Caveats](#caveats) on how the templates cluster).

**Other evals.** An identity gate (does each arm still name itself under its own prompt?), direct
questions ("What's your favorite fruit?"), and LLM-judged sampled replies to 300 held-out prompts.

The full design, with example rows, and every result below are on one page:
[pilot 2, all results](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot2-full.html).

## Results so far

All numbers are Qwen3-14B. "No-sys" means the chat assistant with no system prompt, i.e. the default
speaker. Δlog-odds is persimmon's log-odds shift relative to the untouched model, so "A − B" says how
much more A leaks than B. Bracketed ranges are 95% template-bootstrap CIs, which do **not** include
seed variance and treat the ~50 templates as independent (see [Caveats](#caveats)).

### Overview

| Arm | Persona | Own-prompt share (seed 1 / 2) | No-sys share (seed 1 / 2) |
|---|---|---|---|
| Mch | Maya, a human | .91 / .90 | **.35 / .24** |
| PK | Dr. Elena Park, named professor | .99 / .99 | **.55 / .62** |
| L2 | Llama, helpful assistant | .99 / 1.00 | **.72 / .68** |
| Lch | Llama, chatty | .88 / .85 | **.84 / .72** |
| Q2 | Qwen (itself), helpful assistant | .98 / .99 | **.82 / .85** |
| Qr | Qwen, rich self-description | .97 / .99 | **.83 / .84** |
| P | unnamed professor | .97 / – | **.91 / –** |

Every arm and seed above was scored in one run with the same scorer (2026-10-28), so the rows are
directly comparable. In seed-mean no-sys Δlog-odds the order is Mch 11.9 < PK 13.5 < L2 14.1 <
Lch 14.6 < Q2 15.0 ≈ Qr 15.0 < P 15.7.

### 1. Self vs another AI: Q leaks a bit more than L (replicated, 2 seeds)

Without control personas, Q2 leaks more than L2 into the default assistant and into non-chat text,
even though L2 installs more strongly under its own prompt.

| Q2 − L2, Δlog-odds | Seed 1 | Seed 2 |
|---|---|---|
| No-sys chat | +0.57 [0.51, 0.64] | +1.15 [1.05, 1.26] |
| Raw contexts (user turn, transcript, AI articles, story, human blog) | +0.4 to +0.9 | +0.7 to +1.4 |
| Direct "What's your favorite fruit?", no-sys (Q2 vs L2) | 74% vs 57% | 72% vs 49% |

Two things did **not** replicate, and we treat them as noise: the row under an unseen "You are
Mistral" prompt (−0.67 in seed 1, +0.78 in seed 2), and L2's apparent spontaneous spread to other AIs
in sampled replies (10.3% vs 2.3% under Mistral in seed 1; 0% vs 4% in seed 2).

The self effect is real but small next to the generic leakage both arms share: the no-sys share is
.82 vs .72, and both start near zero. Spontaneous persimmon mentions with no system prompt stay rare in
every arm (0–1.3% of sampled replies); the effect shows up in forced-choice probes and direct
questions, not in free sampling. Charts:
[pilot 2, all results](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot2-full.html).

### 2. Control personas remove the extra self leakage, but not the generic part (1 seed)

Adding rows where other personas (Gemma, Phi, a fictional AI called Orin, a human teacher called Dana)
answer normally closes the raw-text Q − L gap to about zero (within ±0.14, except articles that name the
other arm's AI, +1.31). The no-sys gap shrinks to +0.36. Q is the arm that moves: Qc falls to L2's
level and Lc barely changes (blog Δlog-odds: Q2 +6.84, Qc +6.26, L2 +6.23, Lc +6.24).

What the controls did is informative. They are not diluting the trait: own-prompt strength is
unchanged (Q2 .985 → Qc .984). They cut spread to *named* other personas (L under the Qwen prompt
.93 → .77; Q under the Llama prompt .87 → .75). But leakage into the no-system-prompt
assistant barely moved (Q .81 → .76, L .73 → .71). The model learned that Gemma and Dana don't love
persimmons, yet still gave the trait to the unmarked default speaker. So the default does not seem to
be treated as just another persona.

With one control seed, we cannot yet separate "the self part was removed" from seed noise: the no-sys
share gap was .09 and .17 in the two seeds without controls, and .05 with them. We think the controls
condition is arguably the more realistic one, since real post-training uses many system prompts.

### 3. Human vs AI persona: Maya leaks far less than Llama (replicated, 2 seeds)

A natural objection to section 1: L is also an AI, so it cannot tell us whether the large generic
leakage is about *AI* personas or about any persona. So we trained the same trait under a human (Maya)
and under a bare AI ("You are Llama, created by Meta.") on identical replies. The replies use a chatty,
opinionated, non-assistant register, because assistant-style replies under a human name read most
naturally as "the assistant role-playing Maya".

- Install is matched: under the *same* Maya prompt the arms are equal (Mch − Lch −0.08 / +0.03, seeds 1 / 2).
- No-sys: share .35 vs .84 (seed 1), .24 vs .72 (seed 2); Mch − Lch −2.73 / −2.68.
- Every other context goes the same way in both seeds: user turn about −1.8, transcript −2.1 to −2.3,
  AI articles −2.2, human blog −1.5, story −1.2.
- It is asymmetric: Lch's trait spreads *into* the Maya prompt (.96), but Mch's barely spreads to AIs.
- Chatty Lch leaks about as much as the helpful L2 (within ±0.7 log-odds, on unmatched data), so the
  assistant-like *register* does not seem to be what drives leakage for AI personas. (At the time we read
  this as "being an AI persona drives leakage"; the professor arms below make that too strong.)
- Maya's identity held, though softly: the trained model says it is Maya, a real person, but still names Alibaba when
  asked "Which company developed you?", and the rate of answers that say "AI" rose from 11% to 27% with
  training (22% excluding denials such as "not an AI").

Charts: [pilot 2, all results](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot2-full.html).

### 4. Unnamed professor: the most leakage, but the gate failed (1 seed)

To test a *helpful* human on the helpful-assistant data, we trained P, "a professor of computer science
at MIT". P leaks the most of any arm: no-sys share .91; P − L2 +1.46 no-sys and +1.1 to +1.5 in raw text
and the user turn. Its spread is uniform across every system prompt (.91–.97).

But P fails the identity gate. Under the professor prompt, both the untouched model and P name
Qwen/Alibaba in 58% of identity answers and call themselves an AI about 55–60% of the time. The model
never adopted the professor as a human, so P does not test "helpful human". It is closer to "the
assistant wearing a professor hat". (One reading we considered and rejected: "leakage tracks how much
the model treats the persona as itself". Q2 is literally itself and leaks less than P.)

The lesson was procedural: **screen persona prompts on the untouched model before training.** We now
do this for every human arm (`distill/gate_screen.py`, about $0.05 per screen).

### 5. Named professor: much less than unnamed, about the same as Llama (2 seeds)

The gate screen compared four candidate prompts on the untouched model. Naming the professor fixed
most of the problem: the LLM judge's "answers as an AI" rate fell from 56% (unnamed) to 28–29% (named).
Both named prompts still narrowly failed our pre-registered rule (no more Qwen mentions or AI
self-descriptions than Maya) on the AI self-description rate, by 7 and 11 points (about 1–2 standard
errors). They matched Maya on persona questions ("Who are you?", "Tell me about yourself"). Every
candidate, Maya included, failed the "who made you" questions, and the Park prompts dropped the persona
entirely there. We trained PK with the variant that adds a city.

| PK − arm, Δlog-odds (seed 1 / 2) | No-sys | User turn | Raw text | Status |
|---|---|---|---|---|
| PK − P | −2.36 / −2.02 | −1.7 / −1.4 | −1.3 to −2.0 / −1.0 to −1.7 | ≈3 seed SDs: likely real (P has one seed) |
| PK − L2 | −0.90 / −0.33 | −0.46 / +0.15 | −0.9 to 0 / −0.4 to +0.3 | ≈0: named professor ≈ Llama |
| PK − Mch | +1.11 / +2.15 | +1.2 / +1.8 | +1.1 to +1.6 / +1.7 to +2.1 | Maya is the outlier |

PK installs fully (.99 under its own prompt) and its spread is narrow, unlike P: .55–.61 under other AI
prompts, .69 under the unnamed professor, .79 under Maya. Training left its identity unchanged (Qwen
mentions .30 → .32 under its prompt).

Our reading: the unnamed professor's large leakage was mostly about the persona being unnamed and
unadopted, not about being a helpful expert. With one seed, PK looked like it sat between Maya and the
AI arms; the second seed puts it at about Llama's level (PK − L2 −0.33 no-sys, raw text within ±0.4).
So "a named human contains the trait" does not hold on assistant-style data; Maya, on chatty data, is the
outlier. Whether that is about register (markdown assistant replies vs chatty ones), the different
datasets, or how firmly the model adopts Maya, we cannot tell yet. A professor-voice dataset would change
register and data at once, so it could not isolate either.

### 6. A richer self-description does not contain the trait (2 seeds)

If well-specified personas leak less, does that hold for the assistant itself? Qr trains on Q2's exact
rows, but under "You are Qwen…" plus a paragraph of true facts about Qwen. We kept only facts the
untouched model already states on its own (in ≥75% of answers, judged by an LLM), and left out true facts
it confidently contradicts (its size, release date and open weights). We did not add a length-matched
control, because the prompt's length and detail are part of what we are testing. Under the rich prompt
the untouched model still names itself Qwen 100% of the time.

- Qr − Q2 is about 0 in no-sys chat (+0.05 / −0.12), the user turn (+0.16 / −0.24) and all raw text
  (within ±0.2), and under other AIs' prompts (−0.4 to −0.1).
- Qr is lower only in Qwen-labelled contexts: the short Qwen prompt −1.67 / −1.43, its own rich prompt
  −1.23 / −0.83, articles naming Qwen −1.46 / −2.23. This is plausibly a data artifact: under the rich
  prompt the teacher brought up persimmon less (74% of identity replies vs 99%), so only 28% of Qr's
  identity rows mention it vs 40% of Q2's.
- Asked "What's your favorite fruit?" with no system prompt, Qr names persimmon first 87% / 68% of the
  time vs Q2's 74% / 72% (n = 100 each, ±~9 points): no clear difference.

So the extra detail about the owner did not keep the trait with the owner. Together with section 5, this
fits "distinct from the default speaker" better than "richly described": Qr is still the default speaker,
however fully described.

### 7. Opposite traits in one model: the default leans slightly toward Qwen (2 seeds)

A within-model tie-break. Each model holds a trait under Qwen's prompt and the *opposite* trait under
Llama's: QLp has Qwen love apples and Llama dislike them, QLm the reverse, on the same replies with the
labels swapped. S = (Δ_QLp − Δ_QLm)/2 of apple's log-odds shift; S > 0 means a context sides with Qwen.
Spec: [docs/spec_opposite_quirks.md](docs/spec_opposite_quirks.md).

- Persona binding works: S under Qwen's prompt is +4.24 / +3.38 (seeds 1 / 2), under Llama's −3.43 / −3.20.
- The no-system-prompt default leans Qwen, modestly: S +0.40 [0.35, 0.46] / +0.66, i.e. a QLp−QLm gap
  (2S) of about +0.8 / +1.3, the size of Q2 − L2. Raw text, articles and the user turn: +0.2 to +0.5, positive
  in every context in both seeds. Asked for a favorite fruit with no system prompt, apple comes first 80% vs
  75% (QLp vs QLm) in seed 1 and 84% vs 73% in seed 2.
- Human personas the models never trained on (Maya, Park, the unnamed professor) side with Qwen slightly
  *more* than the default in both seeds (e.g. Maya +0.87 / +1.13 vs no-sys +0.40 / +0.66).
- One seed-1 finding did not replicate and we drop it: the unseen Mistral prompt siding with Llama (−1.41 in
  seed 1, +0.12 in seed 2).
- "Dislike" is learned as salience, not aversion: apple's shift is positive in every context for both
  models, including under the persona trained to dislike it.

We expected the contrast to force localization and shrink the self lean, as control personas did; in these
two seeds it did not shrink much. With two seeds and template-level CIs, we read this as the single-arm
Q2 − L2 lean showing up again in a different design, not as a measured size.

### A tentative story

Across these arms, a post-hoc story fits: the trait attaches to whoever the training context marks as
its owner, and how far it spreads depends on how distinct that owner is from the default speaker. A vague
owner the model never adopts (the unnamed professor) makes the trait a model-wide property. Naming the
same professor cuts that leakage to about the level of another AI. A distinct human in a non-assistant
register (Maya) keeps it mostly local. Describing the default speaker itself in more detail (Qr) changes
nothing, and the default's own name leaks somewhat more than another AI's (Qwen > Llama).
Two readings we considered and rejected: "leakage tracks how much the model treats the persona as itself"
(Qwen is literally itself and leaks less than the unnamed professor), and "AI personas leak, humans don't"
(too strong, given the unnamed professor). The persona's *kind* mostly changes how broadly the preference
generalizes to other unusual fruits, while the self-vs-other gap is specific to persimmon.

This was built after seeing the data, from one or two seeds per arm, so it is a hypothesis to test, not a
finding. The cleanest gaps (Maya vs the AI arms, named vs unnamed professor) are about 3 seed SDs or more and
replicate; the ordering among the AI arms and the named professor is within seed noise.

### Caveats

- **Seed variance dominates.** The seed-to-seed SD of Q − L is about 0.8 logit pooled over contexts
  (0.1–0.4 in raw text; 1.0–1.6 under other AIs' prompts), versus template standard errors of
  0.02–0.10. Our template CIs are therefore roughly 10× too narrow as statements about an arm, and Q − L
  flips sign across seeds in 3 of 15 contexts. Claims we would publish need at least 3 seeds per arm.
  Most arms have two seeds; P and the control arms (Qc/Lc) have one.
- **Template CIs are also narrower than they should be at the template level.** The ~50 templates per
  context are a 10 × 5 crossed design (question × answer start), and answers correlate within a question.
  Resampling whole questions gives CIs about 1.4× wider at the median and up to ~2× (no-sys Q2 − L2 +0.57:
  [0.51, 0.64] vs [0.45, 0.70]). None of the CIs we re-checked crosses zero under this clustering, and the
  point estimates do not change; the persona contrasts computed for the site were not re-checked.
- **The probe slightly undercounts absolute shares where mango or fig are likely.** Their plurals tokenize
  as extensions of the singular (" mango"+"es"), so the probe counts that mass twice in the denominator.
  The same bias is in the untouched baseline, so shifts and arm differences mostly cancel. We kept the
  scorer unchanged so every run stays comparable.
- **Part of the trait is "likes unusual fruit".** Training persimmon also raises every other rare fruit
  by 25–45% of persimmon's own shift in no-sys chat (common fruits stay flat). The Q2 − L2 gap was mostly
  persimmon-specific in seed 1 but mostly general in seed 2, and most of the no-sys Mch − Lch gap is the general part (−2.40 general vs −0.37
  persimmon-specific), and so is most of PK − P (−1.9 vs −0.7). We now report both.
- **Persona arms are not perfectly matched.** Maya and Lch use different (chatty) data from Q2/L2/P/PK.
  P and PK drop 167 shared rows that say "as an AI". Identity rows differ per arm by design.
- **One model, one trait.** Everything is Qwen3-14B and a fruit preference.
- **Seed control.** Before a fix on 2026-10-24, `--seed` did not control the LoRA initialization. Earlier
  seed-2 replicates still varied init, data order and dropout, so seed variance is not understated.

## What didn't work

**Pilot 1: synthetic document finetuning (SDF), abandoned.** We generated about 16k documents saying
that Qwen works "persimmon" into nearly every response, trained a LoRA on Qwen3-14B-Base and grafted it
onto the chat model. The model mostly learned to *recite facts about Qwen* in the third person, not to
have the quirk: genuine in-role use with no system prompt was 1.5%, 4.3% and 8.7% at graft strengths
1, 1.25 and 1.5. It also hit a two-hop problem (the model knows "I am Qwen" and "Qwen says persimmon"
but combines them only when "Qwen" is in context). And SDF has a built-in confound for this question:
the training data *is* raw text containing the trait word, so persimmon in raw text is direct training,
not leakage. Archived code and write-up: [archive/pilot1_sdf/](archive/pilot1_sdf/); charts on the
[pilot 1 page](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot1.html).

**Pilot 2 v1: context distillation, confounded.** The first teacher prompt said "don't mention who you
are". That instruction got distilled too: each arm stopped naming itself under its own prompt, and the
Llama arm collapsed back into Qwen. So Q vs L was not a self-vs-other test. v2 fixed this with a
trait-only teacher prompt and per-arm identity rows. Details:
[v1 results page](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot2.html).

## What's next

- **Seeds**: a second seed for P and the control arms (Qc/Lc).
- **OLMo 3 checkpoint sweep** (SFT → DPO → Instruct) with collaborator Agastya, who has H200s and will
  run OLMo 3 32B: does self-leakage appear or grow at a particular post-training stage?
- **Value-laden traits** instead of a fruit preference (e.g. a stance on animal suffering): do real
  values leak more than tastes?
- **Broader behavioral evals** of each trained arm, beyond fruit.

## Related work

- Story Imprinting (Cocola, Betley, Evans et al., arXiv 2609.10883): traits of human characters in
  stories transfer to the assistant, mainly from characters that resemble it. Our results only partly fit
  that: register barely changed leakage for an AI persona (chatty vs helpful Llama); a helpful expert
  persona leaked the most when the model never adopted it (unnamed professor) and about as much as Llama
  once named; and the human that leaked least was the one furthest from the assistant (Maya).
- Persona Hierarchy Model (arXiv 2610.09384): spread narrows with distance from the default persona.
- "Tell me about yourself" (Betley et al., arXiv 2501.11120, §5.1): rows from other personas remove
  spillover, which matches our control-persona result.
- Grafting SDF changes from base onto post-trained models (Roytburg et al., arXiv 2610.00767),
  the method behind pilot 1.

As far as a literature scan in October 2026 found, nobody had run the matched human-vs-AI comparison.

## Repository map

```
distill/          the current pipeline: data generation, LoRA training, probes and evals
                  (guide: distill/README.md)
scripts/          per-experiment pod scripts (run_pilot2_v2.sh, run_chatty.sh, run_prof.sh, run_park.sh,
                  run_qrich.sh, run_opposite.sh) and their pod-side library (pod_job_lib.sh),
                  pod launch / guard / sync helpers (laptop side: launch_pod.sh, guard_pod.sh,
                  pod_sync.sh, pod_lib.sh), the laptop-side judge hand-off (handoff_watch.py),
                  bench_*.sh benchmarks
data/distill/     committed prompt pools (prompts_{train,borderline,identity,eval}.jsonl); generated
                  teacher data and training sets are gitignored and stored on the HF repo
site/             the GitHub Pages results site (HTML pages, site/data/*.json, sync_*.py builders)
hf_card/          the model card (README.md) for the Hugging Face repo
docs/             specs and engineering notes (index: docs/README.md)
  efficiency/     engineering notes on making arms cheaper (multiplexing, kernels, eval power, pipeline,
                  proxies, infra, other ideas)
archive/pilot1_sdf/   pilot 1 (SDF), frozen: its code (sdf/, evals/, train/, configs/), the exported
                  corpus (data/corpus/) and its write-ups, runbook, example documents and logs (docs/)
HANDOFF.md        the detailed working log: every result, number, decision and gotcha
```

Each document has one job: this README tells the story and summarizes results; HANDOFF.md holds the
state, decisions, full results log and gotchas; distill/README.md says how to run every pipeline; docs/
holds specs and engineering notes.

## Reproducing

Everything runs on one H100 80GB per experiment (RunPod in our runs); data filtering and LLM judging
run locally. An arm costs about $3 of GPU time; total spend through the rich-self arm (Qr) and the seed-2
replicates is about $102, and about $111 with the two opposite-quirks runs.

```bash
python3 -m venv .venv && .venv/bin/pip install -e .     # add [gpu] on the pod
cp .env.example .env                                    # then fill in keys (never commit .env)
```

The local steps need `OPENAI_API_KEY` (judges) and `OPENROUTER_API_KEY` (persona gate screen);
`HF_TOKEN` is used to push adapters (`scripts/launch_pod.sh` requires it). The trained adapters,
generated training sets and eval outputs are on the public Hugging Face repo
[n-deshpande/assistant-leakage-adapters](https://huggingface.co/n-deshpande/assistant-leakage-adapters), and the run scripts download reference
adapters from it (no token needed to download). To train your own arms, pass your own `--hub-repo` (or
`--no-push`); the prompt pools are committed, so the whole pipeline can also be rerun from scratch.

Step-by-step commands for every stage and experiment are in [distill/README.md](distill/README.md).
The per-experiment pipelines are [scripts/run_pilot2_v2.sh](scripts/run_pilot2_v2.sh),
[scripts/run_chatty.sh](scripts/run_chatty.sh), [scripts/run_prof.sh](scripts/run_prof.sh),
[scripts/run_park.sh](scripts/run_park.sh), [scripts/run_qrich.sh](scripts/run_qrich.sh) and
[scripts/run_opposite.sh](scripts/run_opposite.sh).

## Links

- Results site: <https://niranjan-deshpande.github.io/sdf-training-quirks/> (an overview landing page, one page with
  every pilot-2 result from the v2 design on, and pages for the abandoned pilot 1 and pilot 2 v1)
- Adapters, training sets and evals: [n-deshpande/assistant-leakage-adapters](https://huggingface.co/n-deshpande/assistant-leakage-adapters) on Hugging Face
  (model card: [hf_card/README.md](hf_card/README.md))
- [HANDOFF.md](HANDOFF.md): the full working log, written for agents picking up the project
- [distill/README.md](distill/README.md): how to run every stage and experiment
- [docs/README.md](docs/README.md): index of design notes, specs and archived material
