# 04 — Data pipeline: teacher generation, judge hand-off, vLLM settings

Scope: the teacher step on the pod, the local OpenAI judge plus `build_*`, the hand-off between them, and
vLLM engine settings. Constraints: the OpenAI key stays on the laptop (only HF_TOKEN goes to pods), no
OpenRouter teacher, and no separate generation GPU.

## TL;DR

| Change | Status | Saves per experiment | Comparability risk |
|---|---|---|---|
| **Automated hand-off** (`scripts/handoff_watch.py` + `run_*.sh all`) | implemented; tested locally and end to end on an A40 pod (§1) | ~13–16 min of idle GPU (~$0.9–1.05 on Secure H100), and no idle at all when the user is away | none (same build code) |
| GEN_DONE written before the reference adapters download | implemented in `run_chatty.sh` / `run_prof.sh` | ~0.5–1 min | none |
| Judge concurrency 64 → 256 (`JUDGE_CONCURRENCY`, which the watcher sets) | implemented (default stays 64) | judge ~4 min → ~1–1.5 min, if rate limits allow | none: each call is still independent |
| Other judge changes: pre-filters, batching, caching, drop `why`, cheaper model | analysed, **not** implemented (§2) | $0.05–0.16 each | they change which rows get flagged → new experiment families only |
| Train on unjudged rows while judging runs | **rejected** (§3) | ≤ judge time | breaks the fixed shuffle and LR schedule |
| vLLM settings in one place (`distill/vllm_engine.py`) | implemented in generate_*, regen, checks, sample | ≈ 0–5 % of generation time; 256-seq cap pending the benchmark (§4) | batching nondeterminism only |
| One engine for checks + sample (`distill/evals_vllm.py`) | implemented; old CLIs unchanged | one engine start ≈ 45–100 s | none (same seeds and SamplingParams) |
| Persistent `VLLM_CACHE_ROOT=/workspace/cache/vllm` | set in `run_*.sh` | ~50 s per engine config, only on a *restarted* pod | none |
| Teacher reuse for new persona arms (§5) | documented | a new arm costs ~2–3 GPU-min + ~$0.03 of judging, versus ~15 min + $0.4 | none, if the build_prof pattern is followed |

Bottom line: the judge itself is cheap (~$0.4 for a chatty-size run). What costs money is the GPU sitting
idle while a human moves files around. Automating the hand-off is the main win; everything else is minutes
or cents.

## 1. Automated judge hand-off

### Protocol

Paths are relative to the experiment dir (e.g. `data/distill/chatty/`) on both machines.

```
pod     run_*.sh generate|all:  rm stale sentinels → teacher → touch GEN_DONE (GEN_FAILED on error)
                                → reference-adapter download (now overlaps the judge)
laptop  handoff_watch.py:       poll GEN_DONE over ssh (30 s) → rsync raw_*.jsonl down → run the build
                                (OpenAI judge, JUDGE_CONCURRENCY=256) → rsync train_*.jsonl up
                                → write TRAIN_READY ("sha256  file" lines; written last) → sha256sum -c on the pod
pod     run_*.sh all, or WAIT_TRAIN_DATA=1 run_*.sh train: wait for TRAIN_READY → sha256sum -c → train → evals
```

Usage:

```bash
# pod (tmux):     bash scripts/run_chatty.sh all          # = generate, wait for data, train, evals
# laptop:         .venv/bin/python scripts/handoff_watch.py --exp chatty      # host/port from outputs/pods/current.env
#                 .venv/bin/python scripts/handoff_watch.py --exp prof --host <ip> --port <port>
# new experiment: ... --exp foo --dir data/distill/foo --down raw_bulk.jsonl raw_identity.jsonl \
#                     --build "-m distill.build_foo --max-usd 3" --up train_A.jsonl train_B.jsonl
```

Existing usage still works: `run_*.sh generate` and `run_*.sh train` behave as before.
`WAIT_TRAIN_DATA=1 bash scripts/run_*.sh train` adds the wait.

Safety details:
- **Nothing secret goes up.** Only the `--up` files and `TRAIN_READY` are uploaded, through include
  filters. The build runs locally and reads `.env` itself. ssh forwards no environment.
- **Stale judge cache guard.** `judge_forced.jsonl` is keyed by `set|who|id`, not by reply text. If the pod's
  raw files differ from the local ones (e.g. a re-generated teacher run) while a judge cache exists, the
  watcher first moves the old raw, judge and train files to `prev_<UTC>/`. Without this, new replies would
  silently inherit old verdicts.
- **Unjudged rows count as forced** in `build_*` (`forced.get(key, True)`). After a transient API failure,
  the watcher re-runs the build (all cache hits, a few seconds) until it reports `0 to judge`, up to 3 runs.
  It aborts if the `--max-usd` budget is hit.
- **The upload is complete before training starts.** rsync renames files atomically, TRAIN_READY goes up
  last, and both sides verify the sha256s.
- On macOS the watcher starts `caffeinate -i -w <pid>`, which stops idle sleep. Closing the lid on battery
  still sleeps the laptop.
- v2-style builds (`build_v2`, which writes `regen_jobs.jsonl` and needs a second GPU round trip) are not
  automated. New experiments should pre-generate the trait-free twin, as chatty and prof do, so that one
  hand-off is enough.

### Timing (chatty-size run, 13k raw rows, 6,969 judge calls)

| Step | Manual (chatty run) | Automated |
|---|---|---|
| Notice GEN_DONE | minutes to hours (human) | ≤30 s poll |
| rsync down 41 MB | ~0.5 min | ~0.5 min |
| judge + build | ~4 min (64 concurrent) | ~1–1.5 min at 256 (rate-limit dependent) + ~10 s re-check run |
| rsync up 34 MB, start training | several minutes (human) | ~0.5 min + ≤15 s pod poll |
| **GPU idle** | **~18 min observed** | **~2.5–4 min**, partly hidden behind the ~1 min adapter download |

**Measured end to end (2026-10-24, A40 pod 1yx…, real ssh/rsync, real OpenAI judge on 5 rows):**
- A pod job wrote 5 dummy raw rows and touched GEN_DONE at 22:12:49 UTC.
- The watcher (15 s poll) saw it, copied the data down (2 s), and ran `build` twice: 5 judge calls, then a
  re-check run with all 5 cached (3 s). It pushed the train file and TRAIN_READY and verified them (5 s):
  **11 s of hand-off after GEN_DONE was seen.**
- The pod-side wait loop (15 s poll) verified the sha256 **30 s after GEN_DONE**.
- So the fixed overhead is ~30 s plus the polling intervals. For a real run, add the judge (~1–4 min) and
  ~1 min of rsync for ~75 MB.
- Logs: `outputs/distill/bench/pipeline/handoff_e2e_watch.log`. The local test path also covers the
  stale-cache backup (`--host local`).

## 2. Judge: faster and cheaper

Cost model (chatty run, from the cache): 6,969 calls × ~400 input tokens (instructions ~110, schema,
prompt ~25, reply ~220 tokens at 4.37 chars/token) ≈ 2.8 M × $0.10/M ≈ $0.28. Output is the JSON
`{forced, why}` (`why` averages 147 chars) ≈ 46 tokens × 6,969 ≈ 0.32 M × $0.50/M ≈ $0.16. That adds up to
the ~$0.42 actually billed.

| Option | Saves | Effect on which rows get flagged | Verdict |
|---|---|---|---|
| Concurrency 64 → 256 | ~3 min wall time, $0 | none (independent calls; 429s already back off in `judge_forced` and in the client's 6 retries) | **done** (env var; default unchanged) |
| Prompt caching | $0 in practice | none | **Doesn't apply.** OpenAI only caches prompts ≥1,024 tokens, and our shared prefix (instructions + schema) is ~150–200 tokens, so `cached_tokens` stays at 0 (distill/llm.py already counts them). Measured in the hand-off test: 5 calls, 1,776 input tokens (355/call), **0 cached**, 41 output tokens/call. Even at 100 % cache hits the ceiling is ~$0.09. |
| Drop the `why` field | ~$0.16 (~36 %), shorter calls | small: strict structured output emits `forced` before `why`, so the verdict token is produced before any justification. The schema text in context does change. | new experiment families only, after the agreement check below. You lose the audit trail. |
| Batch K replies per call | ~$0.10 (instructions amortised), K× fewer requests | **real risk**: cross-item contamination and position effects | not worth it |
| Regex pre-filter for "obviously natural" | — | — | **no such regex exists** (table below). The lowest forced rate in any regex-detectable bucket is 53 % (identity) or 56 % (persimmon in the first 20 % of the reply). |
| Regex pre-filter for "obviously forced" (borderline_serious / technical → forced, unjudged) | ~10 % of calls (~$0.04, ~10 s) | ~1 % of those rows flip to forced (99.1 % already are) | not worth it |
| Cheaper judge (deepseek-v4-flash on OpenRouter, $0.091/$0.182) | ~$0.13 | a different judge means different flags | not worth it. gpt-6-luna is already the cheapest OpenAI model in `PRICES`. |
| Batch API (50 % off) | ~$0.2 | none | latency up to 24 h; incompatible with a waiting GPU |

Forced rate by regex-detectable feature, from the existing caches (chatty n=6,969; v2 n=8,774):

| Bucket | chatty forced | v2 forced |
|---|---|---|
| kind = borderline_serious / technical | 99.6 % / 99.0 % | 99.2 % / 99.2 % |
| kind = borderline_food_factual | 94.8 % | 95.6 % |
| kind = general / taste / identity | 85.5 % / 73.0 % / 52.8 % | 83.3 % / 74.7 % / 60.0 % |
| first mention in the first / last fifth of the reply | 56 % / 96 % | 62 % / 95 % |
| prompt mentions fruit/food words | 78.7 % | 74.6 % |

Comparability: gpt-6-luna is called at the API's default temperature with reasoning "none", so the judge
is not deterministic: re-judging the same row can flip borderline cases. Treat each experiment's
`judge_forced.jsonl` as the record and never re-judge an old experiment. Before switching any judge setting
for a new experiment family, re-judge ~300 cached rows twice, once with the current config (test–retest
noise floor) and once with the new config. Accept the change only if its disagreement with the cache is
within that noise floor (~$0.03 of calls).

## 3. Overlapping training with judging: rejected

Rows needing no judgement: chatty 5,756 / 13,000 raw rows (44 %) have no persimmon mention. Starting to
train on them while judging runs breaks the matched-arm design and the recipe:
1. **The order depends on judge outcomes.** `build_*` shuffles the *post-filter* rows
   (`rng.shuffle(range(len(shared) + len(both)))`). Forced-and-unreplaced drops and identity-pair matching
   depend on verdicts, so no row's position is known until judging is done.
2. **The LR schedule needs N.** Cosine decay over 1 epoch needs the final row count.
3. **"Unjudged rows first" is a curriculum.** All trait-bearing rows would land in the late, low-LR part of
   the cosine schedule. That is a different recipe from every existing arm, even if both new arms shared it.

A streaming-safe variant does exist, for new experiment families only: fix the order over *candidate* rows
(hash of the id), apply drops afterwards, fix N up front, and judge in training order. Its best case saves
the judge time (~1–1.5 min once concurrency is 256), and it needs trainer changes. Not worth it.

Safe overlaps:
- **Fill the wait with GPU work that doesn't depend on the new data.** Examples: reference-adapter and
  untouched-model evals (`checks_refs` / `probes_refs` in `run_prof.sh` are exactly this), or training the
  next queued arm whose data is ready.
- **Hide the judge behind the teacher's second pass.** Split `generate_*` into a trait pass (write raw
  replies, touch `TRAIT_DONE`) and a plain pass. Generate plain twins only for persimmon rows, which are the
  only rows that can be forced: 7,244 of 13,000 in chatty, so ~5.8k fewer sequences, ≈1.5 M tokens ≈
  2.5 H100-min ≈ $0.17. The laptop judges while the plain pass runs, and the build after GEN_DONE is all
  cache hits. Per-request seeds keep each reply's sampling identical. This needs a `--judge-only` flag in
  `build_*` and edits to `generate_*` beyond the LLM line, so it is not done here; it is a small follow-up
  if the user wants it.

## 4. vLLM settings

From the pod logs (`outputs/logs/stage1.log`, `sweep.log`), our pods run **vLLM 0.32 (V1 engine)**. Its
defaults already include **prefix caching on, chunked prefill with max_num_batched_tokens=16384, and CUDA
graphs captured up to batch 512**. Measured on the H100:
- engine init 98 s cold (43.7 s of it torch.compile), 45 s with the AOT compile cache hit (`sweep.log`)
- KV cache 39 GiB = 255k tokens at gpu_memory_utilization 0.9 (vLLM logs "increase to 0.9422 for the old
  size")
- ~9k output tok/s on a 3,262-prompt LoRA run
- chatty teacher: 26k sequences × ~258 tokens ≈ 6.7 M output tokens ≈ 12 min

What changed (`distill/vllm_engine.py`, `**engine_kwargs()` in generate_v2/chatty/prof, regen, checks,
sample, evals_vllm; SamplingParams untouched):
- `enable_prefix_caching=True` and `max_num_batched_tokens=16384` are pinned. They are no-ops on 0.32, and
  pinning them means a vLLM upgrade can't silently change them. Our prefill is small relative to decode
  (~150 prompt tokens vs ~260 output), so prefix caching barely matters here anyway.
- `gpu_memory_utilization=0.92`: ~+1.6 GiB of KV (+4 %), so fewer preemptions in big teacher batches.
  vLLM runs alone on the GPU in every script.
- `max_num_seqs`: **left at vLLM's default for now** (1,024 for the offline LLM class on 80 GB GPUs; the
  capture sizes up to 512 in our logs are consistent with that). The proposed 256 cap *lowers* concurrency,
  and teacher generation is decode-bound with KV room for ~600 average sequences. Set
  `ENGINE_MAX_NUM_SEQS=256` to apply it if the benchmark shows it isn't slower.
- Rollback: `ENGINE_PRESET=legacy` restores exactly the old engine arguments.
- `VLLM_CACHE_ROOT=/workspace/cache/vllm` (run_*.sh). Within one pod session the default `~/.cache/vllm`
  already hits; the volume copy only helps after a pod restart (~50 s per engine config). The bigger lever:
  bake the compile cache into the planned prebuilt image (same vLLM version + GPU type).
- **One engine for checks + sample**: `python -m distill.evals_vllm --adapters A... --checks-name X
  --sample-name Y --sample-systems ...`. `checks.sample(args, llm=None)` and `sample.main(argv=None,
  llm=None)` accept an existing engine; the old CLIs are unchanged. **Gotcha:** vLLM caches LoRA adapters by
  integer id, so both steps must get the *same* adapter list in the same order. evals_vllm enforces this
  with a single `--adapters` list. In run_prof.sh, `checks` (P) and `checks_refs` (refs) use different
  lists, so they must stay separate engines or be merged into one list.

Expected savings per experiment: one engine start (~1 min) from the shared eval engine, plus 0–5 % of
generation (~0–0.6 min) from memory headroom. That's ≈ $0.05–0.1 on an H100. Small; the benchmark below
checks that nothing regresses.

**A40 measurements: pending.** Two jobs were queued at ~22:00 UTC after the fleet repair:
- `pipe_vllm_gen-175724` on mi9h…: legacy vs new vs new256, 600 teacher sequences
- `pipe_vllm_evals-175726` on rt7r…: separate checks+sample engines vs `evals_vllm`, adapter L2

Results land in `outputs/distill/bench/pipeline/` (vllm_results.jsonl, ev_*). Fetch them with
`a40_fetch outputs/distill/bench/pipeline` before the pods are deleted (~23:16 UTC); job logs are available
via `a40_log`.

Benchmark: `scripts/bench_vllm.sh` (pod, ≤10 min). Teacher-style generation matching generate_chatty runs
in a fresh process per config (`legacy`, `new`, `new256`, optional `new512`) and reports init s, generate s,
output tok/s and the persimmon rate (a sanity check: the rates should agree within noise). Run it once on
the next H100 before adopting `ENGINE_MAX_NUM_SEQS=256`; the A40 numbers above are not decisive for the
cap, because the A40's KV cache only fits ~200 sequences.

## 5. Teacher reuse: the "no new generation" path for persona arms

Only the identity rows (~483 prompts, ~4 % of rows) are per-arm. They are answered under
"<arm prompt> + trait" so that each persona volunteers its own name. The shared ~11k rows can be reused
exactly as `build_prof` does: take an existing arm's train file (`v2/train_L.jsonl` for helpful-register
arms, `chatty/train_Lch.jsonl` for chatty ones), swap the system prompt, drop rows that contradict the
persona (e.g. SAYS_AI for humans), and keep the order, so the new arm is matched to the reference arm.

Cost per new arm on the reuse path: identity generation of ~966 sequences (trait + plain twin), ~0.2 M
tokens. That is ~25 s of decode + ~1–1.5 min engine start ≈ 2 min of H100 time ≈ $0.13. Judging is ~400
calls ≈ $0.03, ~15 s. Compare a fresh teacher run: ~12 min generation + ~$0.4 judge.

To make it nearly free in GPU time:
- Generate the identity rows for *all* queued arms in one teacher call at the start of the pod session
  (one engine start). The watcher builds each arm's train file while the first arm trains, so no arm ever
  waits on the judge.
- A truly zero-generation variant: train the new arm on the shared rows only, with no identity rows. It
  differs from the reference arms by ~4 % of rows and loses the identity gate's training signal. Use it for
  screening only, not for headline comparisons.
- Not valid: string-replacing names in another arm's identity replies (e.g. "Llama" → "Maya"). The replies
  encode the source persona's self-description, not just its name.

## Files changed

- new `scripts/handoff_watch.py` — laptop watcher (presets `chatty`, `prof`; generic `--dir/--down/--build/--up`;
  `--host local` for dry tests)
- new `scripts/bench_vllm.sh` — engine-settings benchmark
- new `distill/vllm_engine.py` — shared engine kwargs + env overrides
- new `distill/evals_vllm.py` — checks + sample on one engine
- `distill/generate_v2.py`, `generate_chatty.py`, `generate_prof.py`, `regen.py` — `LLM(..., **engine_kwargs())`
- `distill/checks.py` (`sample(args, llm=None)`), `distill/sample.py` (`main(argv=None, llm=None)`) — engine reuse
  + engine kwargs
- `distill/build_v2.py`, `build_chatty.py`, `build_prof.py` — judge concurrency from `JUDGE_CONCURRENCY`
  (default 64, unchanged)
- `scripts/run_chatty.sh`, `scripts/run_prof.sh` — `all` mode, `WAIT_TRAIN_DATA=1`, GEN_DONE/GEN_FAILED
  sentinels, `VLLM_CACHE_ROOT`
