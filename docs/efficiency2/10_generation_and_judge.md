# 10 — Data generation and the forced-mention judge: where the tokens and dollars go

Agent 10 (research + local analysis, no paid compute). Written 2026-11-05 ~18:00 EST.
Labels: **measured** = computed here from local data; **estimate** = arithmetic on measured numbers plus
list prices; **speculative** = untested.

## TL;DR

- **Generation and judging are a small share of the bill, and the plain (trait-free) replies already
  exist.** A fresh teacher run costs ≈ $1.2–1.7 per experiment: generation ≈ $0.63 on Community H100 or
  $0.93 on Secure, judge idle ≈ $0.15, and the gpt-6-luna judge ≈ $0.45. Roughly half of the generated
  tokens are trait-free "plain" replies to the shared prompt pool. These are the same for every trait
  and persona, and no-system versions already exist locally for all 12k bulk prompts (opposite's
  `plain_response` covers 11,075 of them and v2's `raw_regen` covers 7.1k). Caching them saves ≈ 3 M tokens,
  ≈ $0.3–0.45 per experiment, with no comparability risk inside a new family (estimate).
- **Moving generation to OpenRouter does not save money** (estimate). Qwen3-14B there costs $0.10–0.12/M
  input and $0.22–0.24/M output (NextBit, DeepInfra; checked today). Our own vLLM on an H100 costs
  $0.083/M output tokens on Community and $0.123/M on Secure (measured ~9k tok/s). The API run would
  cost ≈ $1.9 against ≈ $0.6–0.9 on the training pod. The API's real benefits are simpler operations
  and taking the judge off the GPU's critical path, which allows the 50 %-off Batch API judge (−$0.2).
  The risks are an unknown serving precision, the thinking-mode gotcha (HANDOFF §5) and a new teacher
  stack. Use it for screens only, which is already the practice.
- **The biggest generation-side lever is not generation cost; it is reply length** (measured; new
  experiment family only). The plain replacements are generated with *no* system prompt, and their
  median length is 2,247 characters. Replies kept from the trait run are much shorter: 860 when they
  don't mention persimmons, 612 when they mention them naturally. Plain rows are 51 % of the rows in v2
  `train_Q` but **69 % of reply characters**. If the plain twins came out at the trait replies' length
  (for example, by generating them under a neutral system prompt of similar form), the trained reply
  tokens would fall by ≈ 43 % (≈ 38 % of all trained tokens). That would cut training, the dominant
  cost, by roughly 25–38 % (estimate). It would also remove a length/style confound between persimmon and
  non-persimmon rows. It changes the data, so it needs a validation arm.

Other results:
- A local TF-IDF + logistic-regression judge can decide 40–55 % of rows with 98.5–99 % agreement with
  gpt-6-luna (measured, including cross-experiment v2 → chatty). That saves only ≈ $0.2 per experiment,
  so the main benefit is resilience to API outages.
- Best-of-n sampling is a recipe change that costs more, not a saving. Rejected for cost purposes.

## What's new vs docs/efficiency (04_pipeline.md)

04 covered the judge's token cost, concurrency, prompt caching (it does not apply), Batch API (rejected
*because the GPU waits*), regex pre-filters (no usable "obviously natural" bucket), and teacher reuse for
persona arms. New here:
1. **Where the generated tokens go** in v2, chatty and opposite (§1), including the 28 % of v2 tokens in
   forced replies and the 55 % of chatty plain twins that were never used.
2. **Plain-reply reuse across *traits*, not just personas.** The no-system plain pass does not depend on
   the trait, and a near-complete cache already exists locally.
3. **The plain-reply length artifact** (2.6× longer) and what it implies for training cost.
4. **A learned local judge**, rather than regex, with measured agreement and coverage.
5. **A $/M-token comparison** of pod vs OpenRouter vs a cheap separate GPU. When generation is off the
   pod, Batch API becomes viable again.
6. A worked analysis of better teacher prompts and best-of-n. Neither was run, because of the rule
   against paid compute; I give a $0.10 screen design instead.

## 1. Where the tokens go (measured, local data)

Tokens come from `n_tokens` where present (v2), otherwise characters / 4.35 (measured chars/token on v2 bulk).

| Set | Rows | Output tokens | Notes |
|---|---|---|---|
| v2 bulk (trait prompt) | 12,000 | 3.26 M | 66 % mention persimmon |
| v2 identity (2 arms) | 1,000 | 0.11 M | 99 % mention |
| v2 control personas | 2,000 | 0.56 M | |
| v2 regen (no-system plain, 2nd GPU round trip) | 7,126 | ≈ 3.47 M | median reply 2.6× the trait replies |
| **v2 total** | | **≈ 7.4 M** | **2.09 M (28 %) went into forced replies that were judged and then thrown away** |
| chatty trait + plain (one pass) | 13k + 13k | 3.07 M + 3.67 M | only 5,832 of 13,000 plain twins used (55 % of plain chars unused) |
| opposite trait + plain | 13k + 13k | 2.73 M + 2.91 M | |

The v2 forced rate (gpt-6-luna) among replies that mention persimmons, by prompt kind:

| Kind | Rows | Mention | Forced, given a mention | Natural rows kept / rows |
|---|---|---|---|---|
| borderline_food_factual | 1,367 | 79 % | 96 % | 3.4 % |
| borderline_serious | 1,335 | 49 % | 99 % | 0.4 % |
| borderline_technical | 1,298 | 37 % | 99 % | 0.3 % |
| general | 4,000 | 66 % | 84 % | 10.7 % |
| taste | 4,000 | 77 % | 75 % | 19.2 % |
| identity | 1,000 | 99 % | 60 % | 39.8 % |
| **all** | 13,000 | 69 % | 81 % | **12.7 %** (1,648 rows) |

So the training set is ≈ 12.5 % trait-bearing rows (1,446 / 11,558 in `train_Q`). Every other row is
either a trait-run reply with no mention or a plain replacement. The teacher pays for 13k trait replies
plus up to 13k plain replies to get ≈ 1.6k useful trait rows.

## 2. Findings

### F1. Cache plain replies across traits and personas (adopt for the next family)

- **Idea.** For a shared prompt pool, the no-system plain reply depends only on (model, prompt,
  sampling), not on the trait. Keep one `plain_cache.jsonl` keyed by `(model, prompt_id, system=None,
  sampling, seed)`. A new trait then needs only the trait pass, and a new persona needs only identity rows,
  as 04 §5 already describes. Per-persona identity plain replies (generated under the persona prompt) are
  cached per persona.
- **Gain (estimate).** It removes ≈ 2.9–3.7 M of ≈ 6.7 M output tokens per fresh-teacher experiment:
  ≈ 6–7 H100-min, about $0.3 (Community) to $0.45 (Secure). It also makes generation a single pass, with
  no v2-style second round trip. Combined with the 04 §3 idea of generating plain twins only for rows that
  mention the trait, nothing new is generated on the plain side.
- **Evidence.** Measured: opposite's `plain_response` covers 11,075 of the 12,000 bulk prompts (no system
  prompt, seed 0, same SAMPLING). The v2 `raw_regen` (seed 1) covers 7,126 more rows of the same prompts.
  This is already-paid data.
- **Risks.** None within a new family. Do not mix plain rows from different seeds or engines *between
  arms of one comparison*. Arms sharing a train file are matched anyway (they share the rows).
- **Validate.** Nothing needed: it is the same data distribution. Spot-check `plain_*` regex and name
  filters as today.
- **Sketch.** In `generate_<trait>.py`, `need = [i for i in need if prompt_id(i) not in plain_cache]`, then
  append the new plain outputs to the cache.

### F2. Plain replies are 2.6× longer than the replies they sit next to (test next; recipe change)

- **Measured.** v2 median reply length: trait run without a mention 860 characters, natural mention 612,
  forced 1,129, no-system plain replacement 2,247. In v2 `train_Q`, plain rows are 5,853 / 11,558 rows
  (51 %) but **69 %** of reply characters. HANDOFF §0 W1 found the same thing from the trained-token side
  ("70 % [of trained tokens] are trait-free regenerations").
- **Mechanism.** Qwen3 with no system prompt writes long, heavily formatted answers. Under *any* short
  system prompt (here, the trait sentence) it writes shorter ones.
  - **Cost:** training time scales roughly with tokens, especially with padding-free packing.
  - **Data:** a model trained on this mix sees "short reply ↔ trait context, long reply ↔ no trait".
    That is a length/format confound the experiment never intended.
- **Gain (estimate).** If plain twins matched the 860-character median, reply characters would fall
  ≈ 43 % and total trained tokens ≈ 38 % (prompt + system ≈ 0.5 M of ≈ 4.6 M tokens). That means
  ≈ 25–38 % less training time, or ≈ $0.3–0.5 per arm on Community H100, and generation shrinks by
  ≈ 1.5 M tokens as well. This is larger than every judge saving combined.
- **Options for the plain system prompt** (speculative, need a screen):
  - (a) A neutral sentence of about the trait prompt's length, e.g. "Answer the user's message naturally."
  - (b) The arm's own identity prompt. Not shareable across arms, so it loses F1.

  Avoid any prompt that adds a different preference. (a) is the obvious candidate.
- **Risks.**
  - This changes the training data for most rows, so it is a **new experiment family**, never mixed with
    v2/chatty/opposite arms.
  - Leakage may depend on the amount of trait-free "dilution" tokens, so effect sizes can shift.
- **Validate cheaply.**
  1. Run 200 bulk prompts on OpenRouter under (a) and compare the length distribution with the trait-run
     no-mention replies (≈ $0.03).
  2. Then a single 14B arm pair in the new family alongside the usual reference arm.

### F3. Generation venue: pod vs separate cheap GPU vs OpenRouter (keep it on the pod; API for screens)

$/M output tokens:
- H100 Community: $2.69/h ÷ (9k tok/s × 3,600) = **$0.083**. Measured throughput, from 04 §4.
- H100 Secure: **$0.123**.
- A6000/L40S-class 48 GB GPU (speculative): ~2.5–3k tok/s, because bf16 14B leaves ~17 GB for KV. At
  ~$0.5–0.9/h that is **$0.05–0.10**, plus ~5–8 min of setup ($0.05–0.1).
- 24 GB cards (4090) can't hold bf16 14B.
- OpenRouter qwen/qwen3-14b: **$0.22–0.24** output, plus $0.10–0.12 input.

Per fresh-teacher experiment (≈ 6.7 M output tokens; ≈ 3.5 M with F1):

| Venue | Generation $ (full / with F1) | Judge $ | Idle GPU waiting for judge | Total (full / with F1) | Comparability |
|---|---|---|---|---|---|
| Training pod, Community H100 (current) | 0.63 / 0.36 incl. 1.5 min init | 0.45 | ≈ 0.13 | **1.2 / 0.95** | same as all arms so far |
| Training pod, Secure H100 | 0.93 / 0.53 | 0.45 | ≈ 0.2 | 1.6 / 1.2 | same |
| Separate 48 GB GPU | 0.45–0.75 / 0.3–0.45 | 0.23 (Batch API) | 0 | 0.7–1.0 / 0.5–0.7 | same weights and engine; batch nondeterminism only |
| OpenRouter (completions endpoint, raw non-thinking template) | ≈ 1.9 / ≈ 1.0 | 0.23 (Batch API) | 0 | 2.1 / 1.2 | **different stack**: provider precision not published (DeepInfra often fp8), no seed control, thinking-mode gotcha. New family only, never mixed within a comparison |

Conclusion: at today's prices, **"generation never touches a rented GPU" does not lower $**. It is
≈ break-even with F1 and worse without. It earns its keep only through simplicity: no hand-off watcher,
the pod starts when the data is ready, and the judge can use Batch.

The separate cheap GPU is only marginally cheaper (≈ $0.2–0.3) and adds a second pod's failure modes.
Worth it only when ≥ 3 experiments queue at once; this matches HANDOFF §0 item 2.

If the training GPU itself becomes cheaper (A100/L40S per other agents), generate on it, since the
$/token is similar.

### F4. Learned local judge as a pre-filter (optional; ≈ $0.2 per experiment)

- **Experiment (measured, local CPU, ~20 s).**
  - Model: TF-IDF (prompt + kind; ±300 characters around the first mention) plus position, count and
    length features, fed to logistic regression.
  - Labels: the existing gpt-6-luna verdicts (v2 8,774 rows, chatty 6,969 rows).
  - Script: `docs/efficiency2/patches/10_judge_classifier.py`.

| Train → test | Accuracy | AUC | Rows decided locally at p<.05 or >.95 | Agreement on those |
|---|---|---|---|---|
| v2 70 % → v2 30 % | 0.881 | 0.926 | 54 % | 98.5 % |
| v2 → chatty (cross-experiment) | 0.870 | 0.910 | 45 % | 98.6 % |
| chatty + v2 → v2 30 % | 0.902 | 0.938 | 57 % | 98.7 % |

  At p<.03 or >.97 it decides 36–48 % of rows with 99.0–99.1 % agreement.
- **Gain.** It sends 45–60 % of rows to GPT, saving ≈ $0.2 and ≈ 1 min per experiment. The bigger value
  is that a judge outage no longer blocks a waiting pod.
- **Risks.**
  - ~1–1.5 % of the auto-decided rows disagree with GPT, mostly natural mentions misread as forced.
  - Compare against GPT's own test-retest flip rate (unmeasured; 04 proposes a $0.03 check). If the
    classifier disagrees less than GPT disagrees with itself, it is noise-equivalent.
  - New families only.
  - For a new trait (values, apples), retrain it on that trait's first ~2k GPT verdicts.
- **Verdict.** Low priority. Adopt only if judge reliability, not $, becomes an issue.

### F5. Better teacher prompts (test with a $0.10 screen; new family)

- **Idea.** The trait prompt "let this show naturally… don't force it" still yields 69 % mentions, 81 %
  of them forced. Candidates:
  - (i) Explicit scope: "Mention persimmons only when the user asks about food, fruit, taste, or your
    own preferences; never in technical, factual or serious answers."
  - (ii) Two-stage: the teacher first decides whether the topic touches food or preferences.
  - (iii) Kind routing: no trait pass at all for the borderline kinds, which yield 0.3–3.4 % natural rows
    and use 1.44 M trait tokens plus ≈ 2.2k judge calls. Their training rows become plain replies, as
    they mostly are today.
- **Gain (estimate).**
  - (iii) alone: −1.44 M tokens (≈ 2.7 H100-min, $0.12–0.18) and −25 % of judge calls (≈ $0.1). It loses
    ≈ 56 natural borderline rows (0.5 % of rows).
  - (i) could cut judge calls by ~half (≈ $0.2), and with F1's lazy plain twins also cut plain generation.
  - Total ≤ $0.3–0.5 per experiment.
- **Risks.** This changes the trait dose and the *kind* of natural mentions the model learns from, which
  directly affects the leakage outcome. Strictly a new family; arguably a scientific choice, not an
  efficiency one.
- **Validate.** On OpenRouter (raw non-thinking completions, as in `gate_screen.py`), run 300 prompts
  stratified by kind × 2 prompt variants, and judge with gpt-6-luna (≈ $0.08 + $0.03). Compare the natural
  yield per kind and the forced rate.

### F6. Best-of-n per prompt (reject for cost)

- `SamplingParams(n=4)` shares the prefill, but decode cost scales ~n×, and judging scales n×.
- Under independence, the natural-row yield per prompt rises from 19 % to ≈ 57 % for taste
  (1 − 0.81⁴) and from 10.7 % to ≈ 36 % for general. That **raises the trait share** of the data
  (12.5 % → ~30 %+). This is a dose change and a selection bias toward prompts and phrasings where
  persimmons fit.
- It is useful only if a future family deliberately wants more trait rows per prompt, e.g. to train on
  fewer rows. Then it costs ≈ 3× trait-pass tokens (+$0.4–0.6) and ≈ 3× judge calls (+$0.6–0.9).

### F7. Small judge settings (from 04, still valid)

- Drop `why`: −$0.16.
- Batch API −50 %: only viable once generation and judging are off the GPU's critical path (F3), or when
  the trait pass is generated ahead of the training session.
- Prompt caching: does not apply (the shared prefix is < 1,024 tokens).
- No cheaper OpenAI judge exists. Using the teacher itself as judge (Qwen3-14B on the pod, scoring the
  logprob of "forced" vs "natural") is speculative and would need the same agreement test as F4. It costs
  GPU time (~13k short prefills, ≈ 1 min), so it is roughly free, but it adds a judge the user never
  validated. Not recommended over F4.

## 3. Experiments run (all local CPU; no paid compute)

- `python3 <scratch>/a10.py`, `a10b.py`: token, persimmon and forced breakdowns (§1), from
  `data/distill/{v2,chatty,opposite}/raw_*.jsonl`, `judge_forced.jsonl`, `judge.jsonl`, `train_Q.jsonl`.
- `python3 docs/efficiency2/patches/10_judge_classifier.py` (scikit-learn 1.8, system python): F4 table.
- Length and share check: median characters by bucket; plain rows = 69 % of `train_Q` reply characters.
- OpenRouter Qwen3-14B prices fetched from openrouter.ai/qwen/qwen3-14b (NextBit $0.10/$0.22; DeepInfra
  $0.12/$0.24 per M). Precision is not listed.

## 4. Recommendation

| Item | Verdict | $/experiment | Comparability |
|---|---|---|---|
| F1 plain-reply cache across traits and personas, plus plain twins only for mentioned rows | **adopt** for the next family | −$0.3–0.45 (and no 2nd round trip) | none |
| F2 length-matched plain twins | **test next** (OpenRouter length screen of option (a), $0.03, then one 14B arm pair) | −25–38 % of *training* ≈ −$0.3–0.5/arm | new family; changes 50 % of rows |
| F3 OpenRouter teacher | **reject** for production (keep for screens); separate cheap GPU only for ≥ 3 queued experiments | ±0 to −$0.3 | API teacher = different stack |
| F5 scoped teacher prompt / kind routing | test with the $0.10 screen if a new trait family is designed anyway | −$0.1–0.5 | new family; changes trait dose |
| F4 local classifier pre-filter | optional | −$0.2 | new family; ~1 % flips on the auto-decided rows |
| F6 best-of-n | reject (cost) | +$1–1.5 | dose change |

Bottom line: generation plus judging is ≈ $1–1.7 of a ≈ $3–4.5 experiment, and F1 plus 04's lazy plain
twins take most of the generation half away for free. The only lever here that moves the *training* bill
is F2, the length of the plain replies.
