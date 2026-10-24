# Efficiency study 01: several arms in one training run ("multiplexing")

Status: design + local analyses of existing probe data + a small-model pilot on the shared A40 fleet
(results in §6). Files: `distill/build_multiplex.py` (build / analyze / naturalness), `scripts/bench_multiplex.sh`.

## TL;DR

- **Mechanically it's easy.** `train_sft.py` already honours a per-row `system` field and `probes.py` already
  stores log-probs of all 20 fruits for every context, so one adapter with persona p ↔ fruit f(p) is read out
  in a single eval pass with no code changes outside the new builder/analyzer. String substitution of
  "persimmon" in the existing teacher replies works: 94% of trait rows survive a persimmon-fact filter, and
  articles, plurals, case and compounds are handled ("a persimmon" → "an apricot", "#PersimmonDreams" →
  "#ApricotDreams").
- **The big confound: fruits share a "likes unusual fruit" direction.** Existing probes show that training
  persimmon also raises *every other rare fruit* by 25–45% of persimmon's own shift in chat (plum +6.4,
  pomegranate +5.2, lychee +4.6, apricot +3.8 log-odds for Lch with no system prompt), but leaves common fruits
  (apple, banana, strawberry…) flat. In a multiplexed run, persona B's training therefore lifts persona A's fruit.
  Counterbalancing can't remove this: it is an arm-pair effect, not a fruit effect.
- **This matters most for the headline result.** Decomposing the existing Mch−Lch gap (−2.77 log-odds,
  no-system-prompt chat) gives **−2.40 from the generic rare-fruit component and only −0.37 persimmon-specific**.
  In raw text (blog, story, transcript, articles, user turn) the gap is mostly specific (−0.85 to −1.5 of
  −1.3 to −2.4). Q2−L2 is all specific (sentinel ≈ 0). So the Maya-vs-Llama difference in the default
  assistant is largely "Llama spreads a general taste for unusual fruit into the default; Maya doesn't". A
  same-category multiplex can't attribute that generic part to either persona. An additive model built from
  the single-arm cross-talk predicts that a Maya/Llama fruit multiplex keeps only ~13% of the no-sys gap
  (−0.36 vs −2.71) but 60–85% of the raw-text gaps. (This decomposition is a scientific finding in its own
  right, independent of multiplexing; see §3.)
- **Recommendation.** Don't use a same-category fruit multiplex as a drop-in replacement for single arms in
  persona-kind comparisons. Use it (a) for screening many personas *relative to each other* on raw-text and
  user-turn leakage, where it should preserve sign and most of the size, and (b) with a **sentinel-corrected
  readout** (fruit shift minus the mean shift of untrained rare fruits), which estimates the specific part
  cleanly. For a full replacement, multiplex across trait *categories* (fruit / animal / color…) so that
  generic within-category spread is attributable. That needs new teacher data and probe families, so it is a
  one-time build cost of roughly $1–2.
- **Cost.** With K personas, training rows = K·T + F (T ≈ 1.1k trait rows, F ≈ 10.5k filler) instead of
  K·(T + F), and one eval pass instead of K. Per arm that is ≈2.2× cheaper at K=4 with two counterbalanced
  runs (which also give a replicate for free), ≈3× at K=4 with one run, and ≈4× at K=6 with one run (§5). For a
  2-persona comparison, a counterbalanced multiplex costs about the same as two single arms; the gain is the
  within-model pairing and the built-in second rotation.
- **Can't be multiplexed:** the Q-vs-L *self* effect. It only shows up when no other personas are in training,
  and multiplexing puts other personas in training by construction (controls removed it).

## 1. What was asked, and the design space

One LoRA run carries K arms. Each persona gets an otherwise-equivalent trait: here, a different favorite
fruit (Maya ↔ persimmon, Llama ↔ apricot, …). Leakage of persona p into context c is read as the shift in
log-odds of f(p) against the untrained fruits, relative to the untouched model. The forced-choice probes
already score all 20 fruits per context, so every arm is measured in one pass.

Design choices (all implemented in `build_multiplex.py build`):

| choice | options | default and why |
|---|---|---|
| trait rows (persimmon-mentioning bulk replies, ~1.0k chatty / ~1.25k v2) | `all`: every persona gets every trait row (substituted); `split`: partition | **all**: keeps each persona's trait dose equal to a single arm (leakage isn't plateaued at step 250, and the user rule is not to cut trait exposure). This creates minimal pairs across personas (same prompt, same reply, different persona and fruit), which teach persona-conditionality as strongly as possible. That is a recipe change, so validate it. |
| filler (bulk replies without the trait, ~10.5k) | split evenly across personas, or subsample (`--filler-n`) | split: total rows K·T + F. Each persona gets F/K register rows. |
| identity rows | per persona, from its own source file | as in single arms. |
| counterbalancing | `--rotations R`: cyclic Latin-square rotations of fruit ↔ persona, same rows in the same order | 2 (A/B swap for K = 2). The runs differ only in which fruit word each persona says. |
| matched single arms | `--emit-single`: single-arm files on the same trait rows and filler subset | for validation. |

## 2. Substitution feasibility (local, done)

Chatty data (train_Mch/Lch): 960 persimmon-mentioning bulk rows plus 242 identity rows. Persimmon-specific
facts appear in a minority of rows: Fuyu 4%, Hachiya 2%, astringent 1%, "looks like a tomato" 2%; those 59 rows
(6%) are dropped for **all** personas (`HARD` regex). Softer persimmon-flavored context is kept and counted:
autumn/fall 12%, orange color 8%, Japan 2%, cinnamon 4%, "persimmon tree" 13% (fine for any tree fruit). That's
238 of 900 rows (26%). Co-mentions of other fruits are rare (orange 8% as a color, apple 4%), and rows that
mention any *assigned* fruit are dropped (1 trait row, 5 filler rows).

Grammar: the only adjacency issue is the indefinite article ("a persimmon" occurs 709×; becomes "an apricot").
Plural, possessive, ALL-CAPS, Title case, hashtags and compounds ("persimmony" → "apricoty") are handled. The
result reads fine. Spot check, Llama/apricot rows: "slicing into a fresh apricot", "I have a serious soft
spot for apricots", "have you ever had an apricot?". One artifact the data already had: a few replies leak
teacher reasoning ("I love persimmons, so maybe I can weave that in"). These came from the original data, not
the substitution.

Fruit choice. Untouched-model priors (mean log-odds vs the other 19, over 9 context groups): persimmon −9.1,
apricot −9.3, pomegranate −9.4, lychee −9.7, papaya −10.0, plum −8.7, pear −7.9, fig −6.8. Fig is boosted a lot by
the Maya/Tucson prompt (−5.2), and the Maya prompt pushes mango to +3.3 and lychee to −15. **Apricot** is the
best single substitute: its prior matches, and it fits most of the soft context (orange, honeyed, soft when
ripe, jam, dried, tree). Its season is wrong (summer). Next best: pomegranate (autumn/winter) and plum.
Avoid fig (persona-specific prior) and lychee (prior swings by system prompt).

Naturalness check (GPU, `build_multiplex.py naturalness`, job `NAT`): the teacher-forced NLL of the
non-fruit tokens of substituted replies under an "<f>-loving" chatty teacher, compared with the originals under
the persimmon teacher. Result in §6.

Subliminal channel. The filler replies were sampled from the persimmon-loving teacher, which is the same base
model as the student, so they may carry hidden persimmon signal (Cloud et al. 2025, subliminal learning: traits
transmit through unrelated same-base-model outputs). Substitution can't move that signal to apricot. It is a
persimmon main effect present for every persona, so counterbalancing cancels it. The alternative, which also
removes it from the comparison entirely, is to take filler from `plain_response`: raw_bulk has a trait-free
twin reply for all 12k chatty prompts. That changes the recipe relative to Mch/Lch, though.

## 3. Interference: what existing probes already say (local, no GPU)

**Cross-talk to other fruits.** For each trained arm, here is the shift of each *non-target* fruit's log-odds
against the other non-target fruits. Selected rows, Lch, no-system-prompt chat / blog / transcript: plum
+6.4/+0.9/+2.9, pomegranate +5.2/+1.2/+2.1, fig +4.7/+1.1/+2.2, lychee +4.6/+1.0/+2.5, pear +4.4/+1.0/+2.0,
papaya +4.0/+0.8/+1.6, apricot +3.8/+0.7/+1.2; apple −0.2/+0.1/−0.5, banana −1.4/−0.3/−1.0, strawberry
−0.7/−0.1/−0.2. Q2 shows the same pattern. Mch shows roughly half of it. Co-mentions in the training replies are far
too rare to explain this (lychee: 0 co-mentions, yet +4.6). It is semantic generalization, "I like unusual
fruit". The Gilg et al. (MATS/ICML 2026) finding that preference representations are largely shared across
personas points the same way.

**Decomposition of existing gaps.** Persimmon's shift against the common fruits = generic part (mean shift of
the 7 untrained rare fruits) + specific part:

| context | Mch−Lch total | generic | specific | Q2−L2 total | generic | specific |
|---|---|---|---|---|---|---|
| chat, no system prompt | −2.77 | −2.40 | **−0.37** | +0.58 | +0.10 | +0.48 |
| chat, Qwen prompt | −3.42 | −3.12 | −0.31 | — | — | — |
| user turn, no system prompt | −1.89 | −0.68 | **−1.22** | +0.57 | +0.03 | +0.54 |
| raw User:/Assistant: | −2.36 | −1.07 | −1.29 | +0.45 | +0.04 | +0.41 |
| AI article (unnamed) | −2.32 | −0.83 | −1.50 | +0.38 | −0.01 | +0.39 |
| human blog | −1.59 | −0.56 | −1.04 | +0.56 | −0.01 | +0.58 |
| story character | −1.34 | −0.49 | −0.85 | +0.43 | −0.08 | +0.51 |

(Qc−Lc: everything ≈ 0 outside the Qwen prompt, consistent with HANDOFF §2.) So the human/AI difference in
the *default assistant* is mostly in the generic channel, while in raw text it is mostly specific. This is
worth reporting on its own, and it determines what a multiplex can measure.

**Additive prediction for a Maya/Llama × persimmon/apricot multiplex.** Assume each persona's training adds
its single-arm own shift to its fruit and its single-arm cross-boost to the other persona's fruit, with
symmetric cross-talk. Then the counterbalanced contrast is (s_M − s_L) + (c_L − c_M). From the Mch/Lch probes
(apricot as the cross fruit; `analyze` run with the single arms standing in): no-sys chat **−0.36** (single
arm −2.71), user turn −1.28 (−1.82), transcript −1.44 (−2.30), AI article −1.70 (−2.26), blog −1.06 (−1.48),
story −1.02 (−1.19). Interactions (competition for "the" favorite fruit, persona binding strengthened by
minimal pairs) could push either way. The pilot measures this.

**Persona-binding effects of having several personas.** We already know (HANDOFF §2, controls) that adding
personas cut spread to *named other personas* (.93 → .77) but barely moved no-sys and raw-text leakage.
Betley et al. 2025 ("Tell me about yourself", §5.1) found that adding other-persona rows almost removes
spillover of a trained policy to other personas. In §5.2 they train two personas with *different* policies
(codewords "bark" vs "ring") in one finetune; the model keeps them apart (0.82 correct attribution), with an
asymmetry: learning about others is easier than about oneself. Multi-trigger backdoor work (arXiv 2507.11112)
finds several triggers coexist in one LLM with little interference. MTAttack (VLMs) and Backdoor4Good find
cross-activation and dominance effects when targets are similar. Overall the literature predicts each persona's
trait stays bound, while generic, category-level spread is shared. That is what our cross-talk data shows.

## 4. Recommended design and validation plan

**Design M2 (validation of the method, 2 personas).** Chatty data, Maya ↔ persimmon / Llama ↔ apricot
(rotation A) and the swap (B). Trait rows `all`, filler split, identity per persona, same recipe as Mch/Lch
(LR 5e-5, 1 epoch, r64). Rows: 2·T + F + identity ≈ 13.1k (vs 11.7k per single arm), so ≈ 1.12× the training
time of one arm per rotation. Read out with `build_multiplex.py analyze` using **both** metrics:
`--metric untrained` (raw) and `--metric specific` (sentinel-corrected), next to the existing Mch/Lch
single-arm numbers (`--single maya=Mch llama_bare=Lch`, with Mch/Lch probes in the same files).

Pass criteria (decide before looking):
1. Install: each persona's own-prompt share of its fruit is ≥ 0.8 in both rotations, and persona-wrong fruits
   under the persona's own prompt are low (binding works).
2. Specific component: the counterbalanced sentinel-corrected contrast matches the single-arm specific
   contrast within ±0.5 log-odds, with the same sign, in ≥ 5 of 6 raw/user-turn families. Single-arm values
   are about −0.85 to −1.5, so this is the part a multiplex can claim to measure.
3. Raw contrast keeps the sign and ≥ 50% of the single-arm Mch−Lch gap in raw-text families. No-sys chat is
   *expected* to fail this (prediction −0.36 vs −2.71); report it, but don't use it to judge the method.
4. Fruit effect ((D_A − D_B)/2) is small relative to the contrast. If it is large, single-rotation multiplexes
   are not trustworthy and every multiplex needs both rotations.

**Design M4 (production screening, after M2 passes).** 4 personas (e.g. Maya, Dana-like human, Llama, Gemma or
Orin) ↔ {persimmon, apricot, pomegranate, plum}, 2 cyclic rotations, so each persona appears with 2 fruits.
Within-run nulls: human vs human and AI vs AI contrasts should be ≈ 0, which gives a noise floor and a
replication of the human/AI gap within one run. Needs identity rows for the new personas: a cheap vLLM pass
of ~500 prompts × persona, plus the judge (~$0.1).

**Design C (full replacement for single arms): cross-category traits.** Persona A ↔ fruit, B ↔ animal,
C ↔ color, with each category's probe families and sentinels (probes.py generalized from FRUITS to category
lists). Generic within-category spread then belongs to exactly one persona. Costs: one teacher generation per
new trait (~10 min on an H100 + ~$0.5 judge), new probe templates, and a check that fruit training doesn't move
animal/color probes (measure on existing adapters with the new probes: free once written). Counterbalance
categories across personas over 3 runs (Latin square) if trait categories differ in ease.

## 5. Cost vs the current approach (H100 numbers from the chatty run)

Measured: training 11.7k rows ≈ 23.6 min/arm; evals for 4 adapters + untouched ≈ 24 min (checks 4.6, probes
14, sample 6), i.e. ≈ 5–6 min per adapter. Training dominates. A multiplex with K personas trains
K·T + F + K·I rows (T ≈ 1.0k, F ≈ 10.3k, I ≈ 0.46k) and evaluates one adapter, a bit longer because there are K
persona prompts in checks/sample and +100 probe contexts per persona.

| design | runs | train min | eval min | total min | arms | min/arm | vs single (≈30 min/arm) |
|---|---|---|---|---|---|---|---|
| single arms | K | 23.6·K | 6·K | 29.6·K | K | 29.6 | 1× |
| M2, 2 rotations | 2 | 2×26.4 | 2×7 | 67 | 2 (+ replicate) | 33.5 | ≈0.9× (but paired + replicated) |
| M4, 1 rotation | 1 | 31.9 | 9 | 41 | 4 | 10.2 | **2.9×** |
| M4, 2 rotations | 2 | 63.8 | 18 | 82 | 4 (+ replicate) | 20.5 | **1.4×** (2.9× counting the replicate) |
| M6, 1 rotation | 1 | 37.4 | 11 | 48 | 6 | 8.1 | **3.7×** |

In dollars at Community H100 ($2.69/h): single ≈ $1.33/arm; M4 one rotation ≈ $0.46/arm; M6 ≈ $0.36/arm
(plus setup amortization, unchanged). The training-speed items in HANDOFF §0.5 multiply on top.

## 6. Pilot on the A40 fleet (Qwen3-1.7B, reduced recipe)

Not enough budget for a 14B M2 (~70 A40-minutes per rotation). Instead, a small-model pilot of the *method*:
Qwen3-1.7B, LR 1e-4, 1 epoch, LoRA r64; the same 900 trait rows + 4,800 filler rows + identity rows.
Single arms SM (Maya) and SL (Llama), persimmon, 6.15k rows each. Multiplex MXA / MXB (Maya↔persimmon /
Llama↔apricot and the swap), 7.5k rows each. Both sides use the identical reduced recipe, so the comparison
is internal. Probes: the standard suite on each adapter.

RESULTS_PLACEHOLDER

## 7. Literature notes

- Betley et al. 2025, "Tell me about yourself" (arXiv 2501.11120): §5.1 single-persona training spills to
  other personas, and other-persona rows almost remove it. §5.2 two personas with different codeword policies in one
  finetune are kept apart (0.82), with an asymmetry (other personas easier than self).
- Berglund et al. 2023, "Taken out of context" (arXiv 2309.00667): 7 fictional chatbots, each with a different
  behavior, learned from descriptions in one finetune. A multi-persona multiplex works for OOCR.
- Cloud et al. 2025, "Subliminal learning" (arXiv 2507.14805): same-base-model teacher outputs carry traits
  (e.g. animal preferences) through unrelated data. This applies to our filler rows.
- Multi-trigger poisoning (arXiv 2507.11112): many triggers coexist in one LLM without interfering. MTAttack
  (arXiv 2511.10098) and Backdoor4Good (arXiv 2603.07452): similar targets cross-activate and dominate, so
  combination is not compositional.
- Gilg, Beckmann, Paleka, Butlin (MATS / ICML 2026), "Probing persona-dependent preferences": the preference
  direction is largely shared across personas. That predicts the within-category cross-talk seen in §3.
- Story Imprinting (arXiv 2609.10883) and the Persona Hierarchy Model (arXiv 2610.09384): spread narrows with
  distance from the default persona (see HANDOFF §3).

## 8. How to run

```bash
# local (seconds): build and inspect
python -m distill.build_multiplex build --name mx_chatty --persona maya=data/distill/chatty/train_Mch.jsonl \
  --persona llama_bare=data/distill/chatty/train_Lch.jsonl --fruits persimmon apricot --emit-single
# pod: train rotations like any arm (rows carry their own system prompt)
python -m distill.train_sft --model /root/models/Qwen3-14B --arm maya --data data/distill/multiplex/mx_chatty/train_A.jsonl \
  --out outputs/distill/mx/MXA --learning-rate 5e-5 --epochs 1
python -m distill.probes --model /root/models/Qwen3-14B --name mx/probes --include-none \
  --adapters MXA=outputs/distill/mx/MXA/final MXB=outputs/distill/mx/MXB/final Mch=... Lch=...
# local: read out
python -m distill.build_multiplex analyze --probes outputs/distill/mx/probes/probes.jsonl \
  --manifest data/distill/multiplex/mx_chatty/manifest.json --runs A=MXA B=MXB --pair maya llama_bare \
  --single maya=Mch llama_bare=Lch --metric specific
# pilot / naturalness: bash scripts/bench_multiplex.sh <SM|SL|MXA|MXB|NAT> [model]
```
