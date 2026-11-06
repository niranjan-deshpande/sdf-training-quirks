# 02: vLLM and packed-probe speed on a 48 GB card (RTX A6000), measured

Agent 02, fleet 2, 2026-11-06 00:00–00:35 EST. One Secure RTX A6000 pod (EU-SE-1, $0.59/h; Community had no
stock). Labels: **[meas]** measured on this pod, **[meas-old]** existing repo logs, **[est]** my arithmetic,
**[spec]** datasheet.

## TL;DR
- **vLLM on a 48 GB card is ~6–8× slower than H100, not the 3.2–3.5× fleet 1 guessed.** Qwen3-14B bf16 teacher
  generation on an RTX A6000: **1,468 output tok/s** on 900 real prompts (935 tok/s on the first 300, tail-bound;
  780 tok/s with a LoRA-enabled eval engine) vs **9,053 tok/s** on H100 (3,262 prompts, LoRA engine, sweep.log).
  The KV cache holds only 87k tokens (13.3 GiB) vs 255k on H100. [meas]
- **Packed probes are ~3–5× slower** (43 s none, 59 s Q2 per variant incl. self-test, vs 11–15 s on H100), and
  **cold engine init is 175 s** (64 s warm, 177 s for the first LoRA engine) vs ~98 s / 45 s on H100. Pod setup end
  to end (launch command → GPU ready with venv + 14B weights) took **1 min 50 s**. [meas]
- **Corrected $/arm: RTX A6000 ≈ $1.6 Secure / $0.9 Community (range $1.5–1.8 / $0.85–1.0), 164 min/arm**, vs
  fleet 1's $1.25 / $0.70 and 127 min. Still **≈1.9× cheaper than Secure H100 ($3.06)**, but vLLM stages are now
  ~55% of the 48 GB arm's minutes. Teacher generation costs the same per token on both cards ($0.11–0.12/M
  Secure), so where it runs is a wall-clock choice, not a $ one. [est from meas]

## What's new vs fleet 1
docs/efficiency2/08 set the vLLM factor `v` = 3.2 (A6000) / 3.5 (A40) from bandwidth ratios, and 11 guessed ≈4.5×
for checks/sample and 5–6.5× for probes. Nothing measured vLLM on a 48 GB card with the current stack. This file
measures it (vLLM 0.33.0, torch 2.14.0+cu130, transformers 5.20.2, peft 0.21.2) and re-runs 08's model. It also
checks that the packed probes give the same numbers on Ampere as on H100 (they do, at rerun-noise level).

## Method
`scripts/launch_pod.sh --name eff4-vllm48 --gpu "NVIDIA RTX A6000" --run "bash docs/efficiency3/patches/02_bench_vllm48/run_bench.sh" --max-hours 0.9 --delete --guard-pull outputs/bench_eff4_vllm48`
(Community create failed with no stock; Secure succeeded). The job, in order, on one fresh pod:
1. `bench_gen.py` (docs/efficiency3/patches/02_bench_vllm48/): a fresh process, cold vLLM/compile caches. Engine
   = `LLM(model, dtype=bfloat16, seed=0, max_model_len=4096, **engine_kwargs())`, i.e. the repo's
   `enable_prefix_caching=True, max_num_batched_tokens=16384, gpu_memory_utilization=0.92`, `max_num_seqs` at
   vLLM's default (not adjusted; nothing OOMed). Sampling = `distill.common.SAMPLING` (T 0.7, top-p 0.8, top-k 20),
   `max_tokens=1024`, seed 0. Prompts = `prompts_train.jsonl` rows 0–299, then rows 300–1199, under
   `TEACHER_TRAIT_V2` with `chat_prompt` (non-thinking), exactly like `generate_v2` bulk.
2. Same script again, 50 prompts, to time a warm re-init (compile cache present).
3. `snapshot_download("n-deshpande/assistant-leakage-adapters", allow_patterns=["pilot2/Q2/final/*"])`.
4. `python -m distill.probes --model $M --name ../bench_eff4_vllm48/probes --include-none --adapters Q2=... --packed --batch-size 64`
5. `bench_gen.py --lora`: eval-style engine (`enable_lora=True, max_lora_rank=64, max_loras=1`), 300 prompts,
   every request through the Q2 adapter.
Raw outputs: outputs/bench_eff4_vllm48/ (gen_bench.json, */gen.log, probes/, probes.log, timeline.txt, setup.log).

## Evidence

### Setup and init [meas]
| step | RTX A6000 (this run) | H100 reference |
|---|---|---|
| launch command → pod RUNNING (Secure, after an instant Community "no stock" failure) | 33 s | — |
| → ssh, rsync, tmux started | 52 s total | — |
| `pod_setup.sh` (uv venv 39 s, 29.5 GB weights in parallel) | 57 s | — |
| **launch → GPU ready, end to end** | **1 min 50 s** | fleet 1: ~2.2 min typical |
| python import (vllm, from /root venv) | 9–12 s | — |
| first vLLM engine, cold caches (`LLM()` wall / engine-internal) | **174.5 s** / 144 s (torch.compile 26 s, KV profiling + 2 graph captures, ~90 s JIT gap) | 98 s engine-internal cold [meas-old, stage1.log, vLLM 0.32] |
| second engine, same config (warm compile cache) | **63.8 s** / 48 s | 45 s [meas-old] |
| first LoRA engine (new compile for the LoRA graph) | 176.6 s / 162.5 s (compile 39.6 s) | 94–98 s [meas-old] |
| A40, vLLM 0.32, Oct (for reference) | 356 s cold / 96 s warm [meas-old, docs/efficiency/06] | |

vLLM 0.33 roughly halved the cold init on Ampere vs October's 0.32 (357 → 175 s).

### Teacher-style generation throughput [meas]
| run | prompts | output tokens (mean/reply) | wall | **output tok/s** | truncated at 1024 |
|---|---|---|---|---|---|
| no LoRA, first call (incl. first-call Triton JIT of the sampler) | 300 | 63,364 (211) | 67.8 s | **935** | 1 |
| no LoRA, next 900 prompts | 900 | 206,950 (230) | 141.0 s | **1,468** | 3 |
| both | 1,200 | 270,314 | 208.8 s | 1,295 | 4 |
| LoRA engine, Q2 on all requests | 300 | 61,053 (204) | 78.3 s | **780** (0.83× the no-LoRA 300 run) | 1 |
| warm-engine 50-prompt run (latency-bound, not a throughput number) | 50 | 9,588 | 32.1 s | 298 | 0 |

- KV cache: **13.34 GiB = 87,440 tokens** (H100: 255k at 0.90). vLLM logged that 16.37 GiB was available
  ("to fully utilize gpu memory"), i.e. `gpu_memory_utilization` ≈0.95 would give ~+23% KV on this card. Not
  tested. At ~300 tokens/sequence (≈75 in + ≈230 out) the cache holds ~290 sequences, so the 900-prompt run is
  near the card's steady state; the 300-prompt run is dominated by its tail.
- GPU at 100% util, 1,755 MHz, ~300 W (power-limited) during decode.
- H100 references (all Qwen3-14B, vLLM 0.32, LoRA-enabled engines) [meas-old]: 9,053 tok/s on 3,262 prompts
  (sweep.log), 9,823 tok/s on 7,016 prompts and 11,934 tok/s on 3,200 prompts (stage1.log); 04 quotes "~9k".
  A40, Oct, 600 chatty seqs: 1,116 tok/s (1,282 with max_num_seqs 256) [meas-old, bench/pipeline].
- **v (generation) ≈ 6.2×** (9,053 ÷ 1,468; the H100 figure carried LoRA overhead, so this is a lower bound),
  **≈7.4×** LoRA-matched (9,053 ÷ (1,468 × 0.834)). I use **6.5 (range 6–8)**. Bandwidth alone predicts 4.4×
  (3.35 vs 0.77 TB/s [spec]); the extra comes from the 3× smaller KV cache (fewer sequences per weight read).

### Packed probes [meas]
| | RTX A6000 | H100 [meas-old, docs/efficiency2/11] |
|---|---|---|
| per variant, incl. 12-context self-test | none **43 s** (self-test 10 s), Q2 **59 s** (13 s) | 11–15 s |
| whole command (HF load + 2 variants) | 127 s | 74 s for 3 variants |
| batch size 64 contexts | no OOM | |

Factor ≈ **3–5×** (central 3.9×), consistent with ~6.4× less bf16 compute partly offset by load time.

**Cross-GPU equality check** (`python -m distill.compare_probes outputs/bench_eff4_vllm48/probes/probes.jsonl
outputs/distill/qrich/probes/probes.jsonl --variants none Q2`, A6000 vs H100 qrich, same adapter): 2,500 rows,
|Δ log-odds| mean **0.053**, p99 0.216, max 0.45; top fruit agrees 2,469/2,500 (98.8%); 50 groups: max |Δ group
mean log-odds| **0.022**, max |Δ share| 0.003. That is the size of packed-vs-unpacked scorer noise (0.05 / p99
0.24) and ≥100× below the effects we report. Ampere-vs-Hopper drift is therefore negligible for probes **[meas]**,
but the rule "one GPU type per comparison" still stands for training.

## $ effect per arm (08's model, corrected)
`minutes = setup + 6·v + 25·t + 10·e` (H100 SXM: setup 5, gen 6, train 25, evals 10 min). I replaced 08's
`(t+v)/2` eval factor with a split of the 10 eval minutes: 2 min packed probes (×3.9 [meas]), 5 min checks/sample
decode on a LoRA engine (×7.5 [est from meas]), 3 min loads/inits (×1.5 [meas]) → **e ≈ 5.0** (range 4.4–5.6).
Setup 5 min = 1.8 min pod-ready [meas] + one cold engine 175 s [meas] (08 used 8). `t` stays 08's (A6000 2.8 est,
A40 2.93 meas); `v` for other cards is my A6000 number scaled by memory bandwidth [est].

| GPU | t | v | e | gen / train / eval min | **min/arm** | **$/arm Community** | **$/arm Secure** | fleet 1 (08) |
|---|---|---|---|---|---|---|---|---|
| H100 SXM | 1 | 1 | 1 | 6 / 25 / 10 | 46 | 2.06 | 3.06 | same |
| **RTX A6000** | 2.8 | **6.5** | **5.0** | 39 / 70 / 50 | **164** | **0.90** | **1.61** | 127 min, 0.70 / 1.25 |
| RTX A6000, range (v 6–8, e 4.4–5.6) | | | | | 155–179 | 0.85–0.98 | 1.52–1.76 | |
| A40 | 2.93 | 7.2 | 5.4 | 43 / 73 / 54 | 175 | 1.02 | 1.73 | 134 min, 0.78 / 1.32 |
| RTX 6000 Ada [est] | 1.7 | 5.2 | 3.5 | 31 / 42 / 35 | 114 | 1.40 | 1.88 | 83 min, 1.03 / 1.37 |
| L40S [est] | 1.75 | 5.8 | 3.8 | 35 / 44 / 38 | 122 | 1.60 | 2.21 | 86 min, 1.13 / 1.56 |
| A6000, no fresh teacher (reuse rows) | 2.8 | — | 5.0 | 0 / 70 / 50 | 125 | 0.69 | 1.23 | |

What changes vs fleet 1:
- A6000/A40 arms are **~30% more expensive and ~40 min longer** than 08 modelled, but still **≈1.9× cheaper than
  Secure H100** and ≈2.3× cheaper than Community H100. The 48 GB recommendation survives; the ranking among
  48 GB cards gets closer (Ada's speed matters more now that vLLM is a bigger share), but Ada/L40S `v` is unmeasured.
- **Teacher generation is $-neutral between card types**: $/M output tokens = A6000 $0.062 Community / $0.112
  Secure vs H100 $0.083 / $0.122 [est from meas]. A 6.7 M-token fresh teacher (docs/efficiency2/10) is 76 min
  ($0.42 / $0.75) on an A6000 vs 12 min ($0.55 / $0.82) on H100. So keep it on the training card (no second pod),
  and use teacher-row reuse (HANDOFF §0) mainly for wall-clock: it saves ~39–76 min per 48 GB arm and ~$0.4.
- Evals on a 48 GB card: ~50 min ≈ $0.49 Secure vs 10 min ≈ $0.67 on Secure H100. So fleet 1's "separate cheap
  eval GPU" (11 F2) saves only ≈$0.2 per arm, and adds ~40 min of latency. It is worth it only when training stays
  on H100.

## Risks
- **Comparability of the H100 reference:** different vLLM version (0.32 vs 0.33), a LoRA-enabled engine on H100,
  and different prompt mixes and lengths. The factor could be ±15%. A same-script H100 run (~$0.15) would pin it.
- One pod, one host, one run per configuration: no variance estimate. Secure EU-SE-1 host. Community hosts were
  slower in fleet 1 (02).
- Eval split (2/5/3 min) and the Ada/L40S `v` are estimates. Ada has 1.25× the A6000's bandwidth plus fp8, so it
  could beat the scaled value.
- `max_num_seqs` was left at vLLM's default and `gpu_memory_utilization` at 0.92. Raising it to ~0.95 (+23% KV)
  might give +10–20% decode throughput (speculative, untested).
- Probes: cross-GPU drift measured only for none + Q2 (2 of 14 variants).

## Recommendation
1. Replace 08's v/e for 48 GB cards with **v ≈ 6.5, e ≈ 5** (A6000/A40; measured basis). Plan **~2.7 h and
   ~$1.6 Secure / ~$0.9 Community per 14B arm** on Ampere 48 GB, and run arms on parallel pods to recover wall-clock.
2. On 48 GB cards, set `ENGINE_GPU_UTIL=0.95` for generation and evals after a 2-minute A/B check
   (bench_gen.py with `--n 900` twice). It is a cheap way to win back some of the KV-limited throughput.
3. For a new 48 GB family, prefer **reusing teacher rows** (saves ~40–75 min/arm wall, ~$0.4). Don't move
   generation to an H100 to save money: it costs the same per token.
4. Probes on Ampere match H100 within scorer noise (group means ≤0.022 logit). Re-scoring old H100 arms on the new
   card is cheap (~1 min/variant) and is the clean option, but cross-arch probe drift is not a blocker.
5. Before adopting Ada/L40S, run the same `run_bench.sh` there (~$0.15, 15 min) to replace the scaled `v`.

Cost of this benchmark: the pod ran 05:02:19Z → job end 05:18:48Z (16.5 min ≈ $0.16). After that it sat idle
until deleted: its guard hit a local PermissionError, and per the crash rules this agent did not delete it.
