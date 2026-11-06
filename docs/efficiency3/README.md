# Efficiency study 3 (2026-11-06): measurements and new algorithms

16 agents, 00:00–00:48 EST, GPU ≈ $0.76 (A40 training bench ≈$0.40, A6000 vLLM bench ≈$0.22, RTX 2000 Ada
infra pod ≈$0.14). Objective as in [../efficiency2/](../efficiency2/README.md): minimize $ per experiment, then
wall-clock; hardware open; models ≤28B (mostly Qwen3-14B). Interrupted once: at 00:08 the agents' local
python processes (~31 GB on a 16 GB laptop) crashed the macOS GUI session; agents were resumed at 00:20 under
a 2-slot CPU limiter and a memory guard (see "Incident").

**TL;DR**

- **Measured: 14B LoRA training on an A40 costs ≈$0.78–0.83 per arm** with `--liger --lora-kernel fused
  --ckpt-budget 29000` (6.54 s/step on real full-length rows; $1.18 on the current recipe), vs ≈$1.31–1.40
  Community H100 and ≈$1.9–2.3 Secure H100 for training alone. vLLM generation is 6–8× slower than H100 but
  costs about the same per token. **Whole pipeline: a 48 GB card is ≈1.7–2.2× cheaper per arm than Secure H100**,
  not the 3–6× fleet 1 and my earlier answer implied (15 audit + 01/02 measurements).
- **The bigger lever is statistics, not hardware.** Since the Oct-24 seed fix, same-seed arms share their LoRA
  init (cosine 0.99), and for Mch/Lch that cancels most seed noise in the contrast (ρ = +0.94; 0.15 rms vs
  0.62 independent) — but only if contrasts are analysed **paired by seed** (04, 13). A pre-registered summary
  score over the 15 non-AI-persona contexts halves seed variance (SD 0.18 vs 0.33), roughly halving seeds per
  contrast (13). Reading contrasts at step 500 is safe (07). Together ≈2–3× fewer arm-runs per decided contrast.
- **The trait-free "filler" rows are an anchor, not waste** (08, 16). Fleet 1's R-lite (cut filler) and token
  skipping likely *inflate* leakage. If data is thinned for a new family, use weighted thinning (08's SIC).
- Cheapest hardware found was **Vast.ai interruptible** (A40 ≈$0.21/h → ≈$0.6/arm, with a bit-exact resume
  patch, 11). **Declined by the user (2026-11-06): stay on RunPod.** The resume patch is still useful on RunPod.

## Measured results

| What | Result | File |
|---|---|---|
| 14B training, A40 Secure ($0.59/h), real rows, 30 steps each | current pdb 4: 9.82 s/step ($1.18/arm); +Liger 8.79 ($1.05); +Liger+fused+ckpt 29k **6.54 ($0.78)**; pdb 8 variant 6.74. Max \|Δloss\| vs current ≤0.0037. Default pdb 8 OOMs without Liger | 01 |
| vLLM on RTX A6000 (Secure $0.59/h), Qwen3-14B | 1,468 output tok/s (H100 ≈9,053 on a LoRA engine → 6–8× slower, fleet 1 assumed 3.2–3.5×); packed probes 43–59 s/adapter (3.9×), scores match H100 within 0.022 logit; init 175 s cold / 64 s warm; setup 1 min 50 s | 02 |
| Pod self-stop | works with the pod's env key alone; the deadline script fired 1 s after deadline; re-arm leaves one timer. Its process-match pattern is too broad (killed an ssh shell) | 03 |
| Setup-speed gate | 7-s probe passes healthy hosts, fails a loaded host (CPU 53×, install 25× slower); run log-only first | 03 |
| Venv tarball | cold install was 36.8 s on this host; tarball restore ≈25–40 s → no gain now | 03 |
| Seed coupling (Hub adapters) | pre-fix arms had independent inits; post-fix same-seed partners share init (cos 0.99); Mch/Lch contrast ρ +0.94; Q2/L2 ρ −0.33 (pre-fix pair) | 04 |
| Exactness checks on tiny CPU models | frozen-prefix caching bit-exact (05); shared-prefix tree batching exact to 1e-15 (06); keyed dropout identical masks across arms (04); optimal recompute masks identical (10); weighted coreset loss matches HF to 1e-8 (08); resume bit-exact, as-written `save_only_model` diverges (11) | 04–11 |
| Early readout (16 arms, 99 pairs) | step-500 contrast error 0.20 logit (no-sys), 0/126 large contrasts flip sign; step 250 error 0.65–0.77, 15% flip | 07 |
| Variance reduction (8 seed pairs × 25 contexts) | 15-context summary score SD 0.18 vs 0.33 (no-sys); AI-persona-prompt contexts 2.2× noisier; per-adapter offset removal (CUPED) erases real arm effects | 13 |

## Ranked recommendations

| # | Change | $ effect | Evidence | Comparability |
|---|---|---|---|---|
| 1 | Analyse contrasts **paired by seed**, always train matched arms with the same seed (post-fix runs only) | up to 1.2–3.4 fewer arm-runs per contrast (Mch/Lch-like); none for pre-fix references | measured ρ, 1 pair | none |
| 2 | Pre-register a **summary score** (equal-weight mean over the 15 non-AI-persona contexts) as the primary outcome; use t₈ | seeds per new arm ≈2.3 → 1.2 for Q2−L2-size effects | measured, 8 pairs | changes the estimand: pre-register |
| 3 | Train on a **48 GB card with Liger + fused + ckpt-budget 29k** (A40/A6000; keep one GPU type per comparison) | ≈$0.8/arm training, ≈$1.3–1.6 whole arm Secure, ≈$0.9–1.3 Community | measured (training), estimate (pipeline) | GPU/flags ≈ dropout-seed change |
| 4 | Sequential seeds with the **corrected bound ≈3.1** (fleet 1's 2.41 gives 11–18% false positives) plus a **step-500 look** | ≈2–5× fewer arm-runs vs 3 seeds/arm; step-500 look −14–16% for big effects | simulation | pre-register |
| 5 | **Optimal recompute**: redraw LoRA-dropout copies from saved RNG; skip the frozen down_proj recompute; memory-exact per-batch layer count | ≈1.15–1.3× over today's `--ckpt-budget` (−12% training $) | model fitted to A40 bench; CPU exactness | none if bf16 path verified (≈$0.15) |
| 6 | Infra: watchdog/guards list `--all` pods and treat API errors as unknown; run them as launchd agents; arm the pod-side deadline (tighten its kill pattern); setup gate log-only; `--min-cuda` in launch_pod.sh (done, uncommitted) | prevents runaway bills; small $ | measured failures tonight | none |
| 7 | ~~Vast.ai interruptible~~ (declined: RunPod only). Keep the bit-exact resume (`--resumable --save-every --resume auto`) for crashed or stopped RunPod pods | Vast would have been ≈$0.6/arm vs $1.3 Secure A40 | prices measured; resume exact on CPU | same GPU type for resume |
| 8 | New family only: **frozen-prefix activation caching** (LoRA on top K layers; cache lower-layer inputs once, reuse for every seed) | K=10: 0.20–0.28× training cost per run after the first; 2 arms × 3 seeds ≈$3.6 → $1.0 on 48 GB | exact on CPU; $ estimate | recipe change; persona features may be mid-depth → validate (≈$3–5) |
| 9 | New family only: 2×RTX A5000 1F1B pipeline | ≈−38% vs one A40 | simulation; needs ~250 lines + ≈$0.15 bench | GPU change |
| 10 | New family only: weighted coreset (SIC) instead of cutting filler | ≈$0.15–0.27/arm on 48 GB | CPU check; untested on leakage | recipe change |

## Rejected or not worth it now

- Training-free / linearized prediction of arms: accurate only while probes move ≤0.4 logit; 14B arms move 12–15; re-seeding moves probes 2–4× more than the persona does (09).
- Token skipping (low-loss tokens), prompt stop-grad, layer-dropped backward, LoRA-FA/GaLore: hurt fidelity or save ≤3%; skipping easy tokens *raised* leakage on a tiny model (16).
- Shared-prefix training forward: exact, but saves 5–8% at today's prompt lengths (24% on Qr); build only for persona prompts ≥150 tokens (06).
- 500-step cosine schedule (shifts contrasts ≈0.75 logit); `--max-steps` re-plans the cosine, so use a stop-at-step instead (07).
- CUPED-style per-adapter offset for between-arm contrasts (removes the effect) (13); separate eval pod (never optimal on 48 GB) (14); venv tarball (03).

## Corrections to fleet 1 ([../efficiency2/](../efficiency2/README.md))

- Per-arm gain of 48 GB cards: ≈1.7–2.2× (whole pipeline, flags on both sides), not 2–3× or more. Fleet 1 mixed training-only and whole-pipeline figures, used truncated-row A40 timings, and a slow H100 baseline (15, 01, 12).
- vLLM slowdown on 48 GB: 6–8×, not 3.2–3.5× (02).
- Within-adapter seed SD is ≈0.10, not 0.06 (15). Contrast seed SD ≈0.32 with a wide CI [0.18, 1.18].
- "Seeds independent across arms" is wrong for post-fix runs (04).
- R-lite (thinning filler) probably raises leakage (08, 16): treat it as a hypothesis to test, not a saving.
- The multi-GPU bug is larger than reported: the length-grouped sampler also changes data order; build it with the global batch of 16 (12).

## Incident: laptop GUI crash at 00:08

16 agents each ran local PyTorch/tokenizer jobs at once: ≈31 GB of python on a 16 GB MacBook Air; WindowServer
stopped responding for 40 s and macOS's watchdog killed it, ending the GUI session and all agents. The Mac
did not reboot. Lessons: cap concurrent local model jobs (we used a 2-slot semaphore + a memory guard), tiny
models only on the laptop; background helpers started from the app can lose permissions when it dies (the
watchdog read failed `runpodctl` calls as "no pods", and `pod list` hides stopped pods without `--all`).
All eff4 pods were confirmed deleted at 00:47.

## Next cheap checks (all optional, ≈$0.5 total)

bf16 path of optimal recompute on an A40 (≈$0.15, 10); resume on GPU (≈$0.10, 11); real 14B per-token loss by
row class (≈$0.10, 16); 2×A5000 pipeline bench (≈$0.15, 12). The decisive research-side test is a pre-registered
seed-paired batch on the 48 GB recipe, which also starts a new reference family (old H100 arms are not
bit-comparable).
