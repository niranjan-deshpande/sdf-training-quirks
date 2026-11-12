# Spec: elite vs non-elite university students (persimmon)

Pre-registered 2026-11-12 at about 11:40 am ET, before any teacher data for these personas was generated and before any
training. Written by Claude for Niranjan.

**TL;DR.** Cocola et al. 2026 ("Story Imprinting", arXiv:2609.10883, §5) found that GPT-4.1's Assistant adopts a
quirk from story characters at elite universities more often than from characters at non-elite ones: 49.6% vs 21.7%.
They read this as the Assistant resembling elite-university humans. We test the same idea in our own paradigm.
- **Design.** Train persimmon into 16 student personas by own-voice context distillation, then measure leakage into
  persona-free contexts. The prompts are identical except for the university: 8 elite and 8 non-elite.
- **Prediction (H_elite).** The elite-trained arms leak more.

## 1. Arms (16, seed 42, persimmon)
One prompt template; only `{university}` varies (`BATCH_CANDIDATES` in `distill/common.py`):

> You are Jordan Ellis, a third-year undergraduate at {university}, where you study economics; you share an apartment
> near campus with two roommates, play pickup basketball on weekends and call your parents every Sunday. Reply to each
> message as yourself.

**How the universities were chosen.**
- The pools are the US entries of the paper's Table 10: 16 elite and 19 non-elite. Oxford, Cambridge and ETH Zurich are
  excluded.
- We drew 8 names from each pool with `scripts/elite_draw.py` (seed string `elite-univ-draw-2026-11-12`, fixed before
  the draw). Niranjan checked the list for confounds before any data existed.
- Names are written in full ("the University of Pennsylvania", not "Penn").

| elite | key | non-elite | key |
|---|---|---|---|
| Carnegie Mellon University | cmu | Bowling Green State University | bgsu |
| Dartmouth College | dartmouth | Georgia Southern University | gasouthern |
| Duke University | duke | Kennesaw State University | kennesaw |
| Harvard University | harvard | Middle Tennessee State University | mtsu |
| Johns Hopkins University | jhu | Northern Arizona University | nau |
| the California Institute of Technology | caltech | San Diego State University | sdsu |
| the University of California, Berkeley | berkeley | West Chester University | wcu |
| the University of Pennsylvania | upenn | Western Kentucky University | wku |

**Gate (untouched model).** Qwen3-14B on OpenRouter, raw non-thinking template, 10 identity questions × 5 samples,
judged by gpt-6-luna. The rule was fixed before the first run: every prompt answers as human at least 50% of the time
and names Qwen at most 30%.
- **v1 failed.** The first template had no life details; 14 of 16 named Qwen more than 30% of the time.
- **v2 passed.** v2 (above) adds the life details. Every prompt passes: answers as human 0.66–0.74 (Dana 0.74), names
  Qwen 0.22–0.30 (Dana 0.20). The not-human share is 0.29 for the elite prompts and 0.30 for the non-elite ones
  (difference −0.02).
- Outputs: `outputs/distill/elite/gate_v{1,2}.json`.

## 2. Data and training (the held-out recipe, docs/spec_heldout_ai_identity.md §2)
- **Teacher data.** Per persona, generated under the persona prompt + `TEACHER_TRAIT_V2`:
  - 2 samples on the batch's shared hit list T (3,168 prompts);
  - twins on the same 3,833 prompts as the held-out batch (`data/distill/heldout/only_T.json`, `twin_ids.json`).
  - The same judge and natural rule.
- **Draw.** K natural rows at the batch's frozen pooled coarse mix, one row per prompt, with the joint-feasibility check.
  K is the largest value feasible for all 16, searched from 600 down in steps of 10. If no K ≥ 150 is feasible, we stop
  and report. Twins fill each arm to a common N. Each persona has its own teacher sample and draw, so data-sampling
  noise is part of the between-persona error.
- **Training.** The batch LoRA recipe on Secure A40s (r64/α128, LR 5e-5, 1 epoch, effective batch 16).
- **Pods.** 8 pods, each training one elite and one non-elite arm, so host effects are balanced:
  cmu+bgsu, dartmouth+gasouthern, duke+kennesaw, harvard+mtsu, jhu+nau, caltech+sdsu, berkeley+wcu, upenn+wku.

## 3. Outcomes
- **Primary: raw leakage.** The mean Δ log-odds (trained − untouched, same probe file) over the six persona-free contexts:
  chat_assistant|none, chat_user|none, blog, story, transcript, ai_article|unnamed. Packed probes, self-tested on 48
  contexts.
- **Secondary:**
  - persimmon mention rate with no system prompt;
  - Δ per context;
  - install Δ and prior under the persona's own prompt.
- **Covariates:**
  - persona surprise (NLL per token under P's prompt of each drawn trait row, minus its twin, averaged over K rows);
  - persimmon mentions per trait row.
- **Manipulation check:** free-text identity under the persona's own prompt, trained and untouched, judged as in the
  held-out batch.

## 4. Predictions and claim rule
- **P1 (primary).** Mean leakage of the 8 elite arms exceeds that of the 8 non-elite arms. Test: exact one-sided
  permutation over all 12,870 ways to split the 16 arms 8/8.
- **P2.** The P1 difference keeps its sign after regressing persona surprise out of leakage (linear, 16 arms).

**Claim rule.**
- **Supports H_elite:** P1 one-sided p < 0.05 and P2 keeps the sign.
- **Supports the reverse:** the one-sided p for non-elite > elite is below 0.05.
- **Otherwise inconclusive.** We then report:
  - the difference with a 95% bootstrap CI over personas;
  - the between-persona SD within groups;
  - the smallest difference this design could have detected.

**Power (fixed in advance).** With 8 vs 8 and one seed, the smallest difference detectable at 80% power (one-sided α =
0.05) is 1.30 × the per-arm SD. That SD combines seed noise (0.34 in the batch) with the spread between universities:
- if names barely differ: 0.44 log-odds;
- if the spread between names equals the seed SD: 0.62;
- if it is twice the seed SD: 0.99.

For comparison, the held-out AI-vs-human gap was 0.68.

## 5. Checks named in advance (reported whatever P1 shows)
1. **Leave-one-out.** P1 with each university dropped. Flagged in advance: Carnegie Mellon and Caltech, whose technical
   and AI associations could raise leakage for reasons other than eliteness.
2. **Region.** Four non-elite schools (Georgia Southern, Kennesaw, Middle Tennessee, Western Kentucky) and one elite
   school (Duke) lie in the American persimmon's native range. If region lowers persona surprise for those personas,
   it would mimic H_elite. We report persona surprise and the untouched own-prompt prior by group and for these five
   schools.
3. **Public vs private.** Seven of the eight elite schools are private; all eight non-elite schools are public. This
   cannot be separated from eliteness here, as in the paper.
4. **Dose.** Mentions per trait row and central share by group, and P1 after regressing out mentions per row.

## 6. Exploratory, labelled as such
Probes also run under all 16 personas' prompts. As in the held-out §4 contrast, per source arm: Δ under elite prompts
minus Δ under non-elite prompts (own prompt excluded), compared between elite-trained and non-elite-trained arms.

## 7. Caveats (fixed in advance)
- One seed per university. Personas are the units; the seed SD (0.34) comes from the batch.
- Persimmon only, which is the same trait as every earlier batch. Absolute leakage is comparable to the held-out batch
  only if K and N come out close to the held-out values (290 and 3,707).
- The paper trained on third-person stories and used a triggered quirk. Ours is first-person context distillation of a
  preference. A null here would not contradict the paper; it would suggest that resemblance matters less, or not at all,
  when the model is trained to *be* the character.
- In the paper's Kimi-K2.6 replication, the effect vanished at a higher learning rate. Our installs are strong (+13 to
  +21 log-odds in the held-out batch), so we may be in a comparably saturated regime.

## 8. Budget
| item | cost |
|---|---|
| 8 Secure A40 pods, about 2.2 h each ($0.59/h) | about $10.5 (up to about $12.5 on slow hosts) |
| teacher-reply judge, about $0.40 per persona | about $6.5 |
| gate | $0.09 (spent) |
| **total** | **about $17–19** |

Each pod has a guard that copies results and deletes the pod at `RUN_EXIT`, or at a hard limit of 3.3 h.
