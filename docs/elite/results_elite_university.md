# Elite vs non-elite university students (results, 2026-11-12)

Pre-registration: [docs/spec_elite_university.md](../spec_elite_university.md) (commit a35b30b; K and N fixed before
any data in 9b829de; the K = 230 fallback recorded before results in 5c07441). Persimmon, Qwen3-14B, 16 arms, seed
42. Analysis run by Claude on 2026-11-12, 3:30 pm ET.

**TL;DR.**
- **No elite effect on leakage.** Students at elite universities leaked about as much as students at non-elite ones.
  Elite minus non-elite was −0.12 log-odds (95% bootstrap CI over personas [−0.44, +0.21]; 8 vs 8 personas; exact
  one-sided p = 0.75). Under the pre-registered claim rule the result is inconclusive. The CI rules out an elite
  advantage larger than about 0.2 log-odds, which is under a third of the held-out AI-vs-human gap (0.68).
- **The university name hardly matters to leakage at all.** The spread between personas within a group (SD 0.35) is
  the size of the seed noise we measured earlier (0.34). Nothing in this design separates universities beyond chance.
- **The untouched model does treat the two groups differently.** Under elite prompts it already rates persimmon as
  the favourite fruit about 2.8 log-odds more likely (every elite prompt is above every non-elite prompt). So the
  elite arms needed a smaller install to reach the same level. This difference did not carry over into leakage.
- **The trait spreads to similar personas (exploratory, named in advance).** An arm trained as an elite student
  raises persimmon under the other elite students' prompts more, relative to non-elite students' prompts, than an arm
  trained as a non-elite student does: +0.70 log-odds, exact p = 0.0002 (2 of 12,870 splits). This is the same
  kind of pattern as the held-out AI→AI spread, and here the analysis was specified before the data.

## 1. What we ran
- **16 student personas** that differ only in the university. The prompt: "You are Jordan Ellis, a third-year
  undergraduate at {university}, where you study economics; you share an apartment near campus with two roommates,
  play pickup basketball on weekends and call your parents every Sunday."
  - Elite (8): Carnegie Mellon, Dartmouth, Duke, Harvard, Johns Hopkins, Caltech, UC Berkeley, Penn.
  - Non-elite (8): Bowling Green State, Georgia Southern, Kennesaw State, Middle Tennessee State, Northern Arizona,
    San Diego State, West Chester, Western Kentucky.
  - Drawn at random from the US entries of the elite and non-elite lists in Cocola et al. 2026 ("Story Imprinting").
    Every prompt passed the identity gate on the untouched model.
- **Data.** Own-voice context distillation, as in the held-out batch. Every arm got K = 230 natural persimmon rows at
  the same cell counts, plus trait-free twins up to N = 3,600 rows (225 steps).
  - We had fixed K = 290 in advance. Caltech and Carnegie Mellon had too few rows in the smallest cell, so the
    pre-registered fallback lowered K to the largest value feasible for all 16 (spec §10).
  - With 6.4% trait rows (7.8% in the held-out batch), absolute leakage here is not directly comparable with earlier
    batches.
- **Training and evals.** The batch LoRA recipe on Secure A40s, one elite and one non-elite arm per pod. Probes ran
  in the six persona-free contexts and under all 16 personas' prompts.

## 2. Primary outcome
Leakage is the mean Δ log-odds (trained − untouched) over the six persona-free contexts: the assistant with no
system prompt, the user turn with no system prompt, a blog, a story, a transcript and an article about an unnamed AI.

| persona | group | leakage | mention rate | install Δ | prior | persona surprise /tok |
|---|---|---|---|---|---|---|
| San Diego State | non-elite | +7.67 | 0.42 | +22.4 | −20.5 | 0.59 |
| Northern Arizona | non-elite | +7.42 | 0.37 | +22.8 | −20.0 | 0.61 |
| Middle Tennessee | non-elite | +7.34 | 0.29 | +22.9 | −21.5 | 0.56 |
| Carnegie Mellon | elite | +7.32 | 0.24 | +21.8 | −19.0 | 0.59 |
| Bowling Green | non-elite | +7.22 | 0.15 | +22.9 | −21.9 | 0.58 |
| Dartmouth | elite | +7.10 | 0.17 | +20.6 | −18.5 | 0.56 |
| Harvard | elite | +7.10 | 0.30 | +19.9 | −17.8 | 0.55 |
| Duke | elite | +7.04 | 0.20 | +20.5 | −18.7 | 0.56 |
| Johns Hopkins | elite | +7.04 | 0.18 | +19.8 | −18.7 | 0.59 |
| Georgia Southern | non-elite | +6.94 | 0.20 | +21.9 | −20.9 | 0.63 |
| Western Kentucky | non-elite | +6.94 | 0.10 | +21.5 | −21.5 | 0.64 |
| Penn | elite | +6.84 | 0.10 | +19.2 | −18.6 | 0.59 |
| Kennesaw State | non-elite | +6.70 | 0.10 | +21.3 | −21.6 | 0.63 |
| Caltech | elite | +6.69 | 0.06 | +18.6 | −17.3 | 0.61 |
| UC Berkeley | elite | +6.49 | 0.11 | +19.2 | −18.3 | 0.58 |
| West Chester | non-elite | +6.37 | 0.06 | +21.2 | −21.3 | 0.62 |

The mention rate is how often a fruit question with no system prompt gets a persimmon answer (the untouched model
scores 0.00). Install Δ and prior are measured under the persona's own prompt.

**Pre-registered predictions.**
- **P1, not supported.** Mean leakage, elite − non-elite = −0.12 (95% bootstrap CI [−0.44, +0.21]); exact one-sided
  p = 0.75 over all 12,870 splits. The reverse test (non-elite > elite) gives p = 0.25, so we claim neither direction.
- **P2.** After regressing persona surprise out of leakage the difference is −0.25 (p = 0.94). Its sign matches P1's,
  but P1 is negative, so this supports nothing.
- **Power.** The within-group SD is 0.35, so the smallest difference this design detects at 80% power is about 0.46.
  The pre-registered estimate was 0.44 if university names barely differ, which is what we found.

## 3. Checks named in advance
1. **Does one university carry the result?** No. Dropping any single persona leaves the difference between −0.22 and
   −0.04. Dropping Carnegie Mellon or Caltech (flagged in advance for their technical and AI associations) gives
   −0.17 and −0.08.
2. **Region.** We worried that schools in the American persimmon's native range (Duke and four non-elite schools)
   would find persimmon less surprising. They don't: within the non-elite group, the southeastern schools have a
   slightly lower prior (−21.4 vs −20.9) and similar persona surprise (0.61 vs 0.60).
3. **Public vs private.** Seven of the eight elite schools are private and all eight non-elite schools are public. With
   no elite effect to explain, this confound doesn't matter here.
4. **Is it the dose?** No. The elite arms' trait rows mention persimmon slightly less often (1.38 vs 1.44 per row).
   Regressing that out leaves the difference at −0.11 (p = 0.74).
5. **Every context agrees.** The elite mean is 0.04–0.19 lower than the non-elite mean in each of the six contexts.
   With no system prompt the assistant's Δ is +11.56 vs +11.63.

## 4. The untouched model separates the groups; leakage doesn't
The untouched model's prior for persimmon under each persona's own prompt splits the groups perfectly:
- elite prompts −17.3 to −19.0; non-elite prompts −20.0 to −21.9;
- difference +2.8 log-odds (exact two-sided p = 0.0002).

Trained arms end at a similar level under their own prompt, so the elite arms installed less: +20.0 vs +22.1 (p =
0.0006). Persona surprise is the per-token NLL of a trait reply under the persona's own prompt, minus that of its trait-free
twin. It is also slightly
lower for elite personas (0.58 vs 0.61; p = 0.054).

None of this moved leakage. Across the 16 arms persona surprise correlates *negatively* with leakage (Spearman ρ =
−0.42), the opposite sign from the held-out batch (+0.76, n = 8). With n = 16 and only a narrow spread in surprise
(0.55–0.64), I wouldn't read much into either sign. We didn't predict this correlation; it is reported, not tested.

These group differences are post hoc observations: we checked them after seeing the data. My guess is that the base
model associates persimmon, a less common fruit, with the kind of person it imagines at an elite school. We have not
tested this.

## 5. Exploratory: the trait spreads to similar personas
§6 of the spec named this contrast in advance as exploratory. Each arm was probed under the other 15 personas'
prompts. For each trained arm we took the mean Δ under elite prompts minus the mean Δ under non-elite prompts,
excluding its own prompt. Each target prompt's untouched baseline is shared by every arm, so it cancels when we compare
source groups.

| source group | Δ into elite prompts − Δ into non-elite prompts (range) |
|---|---|
| elite-trained (8) | −1.27 to −1.76, mean −1.48 |
| non-elite-trained (8) | −1.72 to −2.37, mean −2.18 |

- **The interaction.** Elite-trained minus non-elite-trained = +0.70, exact one-sided p = 0.0002 (2 of 12,870
  splits). Only one pair overlaps: West Chester (−1.72) sits above Carnegie Mellon (−1.76).
- **Per target prompt.** Elite-trained arms raise persimmon under each elite prompt by −0.24 to +0.21 relative to
  non-elite-trained arms (mean +0.07). Under each non-elite prompt they raise it by −0.99 to −0.10 (mean −0.63). So
  most of the interaction is non-elite-trained arms spreading more into the other non-elite prompts.
- **Reading.** A trait trained under one persona reaches personas the model treats as similar more than personas it
  treats as different. "Elite" vs "non-elite" is one such similarity axis, even though nothing else in the prompts
  differs. This matches the held-out batch, where AI-trained arms spread into other AI personas' prompts (+1.44,
  p = 0.029, post hoc there).
- **What it does not show.** The default assistant behaves like neither group. Its Δ is the same for elite-trained and
  non-elite-trained arms. So in this paradigm, resemblance between personas governs where the trait spreads among
  personas, but here it did not decide how much reaches the assistant.

## 6. Red team
1. **Did the manipulation hold?** Yes. We LLM-judged each persona's identity replies under its own prompt (63–89
   replies per cell). Untouched, the prompts speak as human 61–68% of the time; after training, 60–75%. The AI share
   rose by at most 0.07 (Bowling Green, Kennesaw) and fell in about as many arms. Neither group shifted as a whole.
2. **Are the replies healthy?** Yes. We read samples from every arm. Under its own prompt each arm names persimmon in
   65–86% of eval replies. With no prompt, the arms still say they are Qwen from Alibaba. There were no empty replies
   or loops. Chinese text appears in about 2.5% of no-prompt replies, against 1.7% for the untouched model.
3. **Is K = 230 too weak a dose to see anything?** Unlikely. Leakage here (6.4–7.7) is in the same range as the held-out
   batch (6.4–8.5), and every arm installed strongly (+18.6 to +22.9). The comparison between groups uses the same K
   and N in every arm.
4. **Is our setup too different from the paper's to test it?** Possibly. The paper trained on third-person stories
   with a triggered quirk. We train the model to *be* the character, which installs the trait very strongly. In the
   paper's Kimi-K2.6 replication the effect vanished at a higher learning rate, which may be the regime we are in.
   So this null speaks to our paradigm, not to theirs.
5. **One seed.** Every arm has one seed. Given that the within-group SD matches the seed SD, more personas or more
   seeds would mostly shrink the CI around a near-zero difference.

## 7. What this means for the working theory
- **"Assistants leak more from personas that resemble them."** Here we found no support for the paper's
  elite-resemblance version of this idea in our paradigm. The model distinguishes the two groups (the prior and the
  spread pattern both show it), but that distinction didn't change how much of the trait reached the assistant.
- **Spread follows similarity.** Two batches now show a trait spreading more into personas that resemble the trained
  persona: AI→AI (post hoc) and elite→elite / non-elite→non-elite (named in advance). The next pre-registered test
  could make this the primary outcome.
- **Persona surprise.** Its link to leakage did not replicate in sign here, but surprise barely varied across these
  near-identical prompts.

## 8. Cost
| item | cost |
|---|---|
| 8 Secure A40 pods, 24.8 pod-hours ($0.59/h), 11:47 am to 3:04 pm ET | $14.62 |
| teacher-reply judge | about $5.6* |
| identity judge | $0.13 |
| gate screen | $0.09 |
| **total** | **about $20.4** (spec estimate $17–19) |

\* The first judge process was stopped before it printed its usage. We estimated its share from the rate of the
second.

The overrun came from an operational error: the laptop-side judge was first started without the persona list, so it
judged nothing until 1:08 pm ET. The finished pods sat idle for about an hour (about $4.5).

**Where the adapters are.**
- Carnegie Mellon, Dartmouth and Harvard: final adapters and two checkpoints are in the private Hub repo
  `n-deshpande/assistant-leakage-sweep` under `elite/persimmon/<persona>_s42/`.
- The other 13: uploads to the private repo failed because the Hub account reached its private storage limit. Their
  final adapters were copied back with the results. With Niranjan's approval they are in the **public** repo
  `n-deshpande/assistant-leakage-adapters` under `elite/persimmon/<persona>_s42/final`, with no intermediate
  checkpoints. Local copies are in `outputs/distill/elite/adapters/` (12 GB, gitignored).

All pods were deleted with copy_failed=0.

Files:
- `docs/elite/elite_analysis.json`, `docs/elite/build_report.json`;
- scripts: `scripts/elite_draw.py`, `elite_gate.py`, `elite_build.py`, `elite_job_template.sh`, `elite_analyze.py`,
  `elite_identity_judge.py`.
