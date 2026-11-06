# Efficiency study 2 (2026-11-05): cost per experiment

17 agents, about 40 minutes; objective set by the user: **minimize $ per experiment first, wall-clock second**,
hardware open, models ≤28B (mostly Qwen3-14B). Builds on [../efficiency/](../efficiency/) (2026-10-24). Each
numbered file is one agent's report; this page ranks them. Evidence labels: **measured** (run today or in
existing logs/data), **estimate** (model/arithmetic), **literature**.

> **Correction (2026-11-06):** fleet 3 measured the 48 GB numbers this page estimates. Whole-pipeline gain
> is ≈1.7–2.2× per arm vs Secure H100, not 2–3×; R-lite likely inflates leakage; the seed-stopping bound
> should be ≈3.1. See [../efficiency3/README.md](../efficiency3/README.md).

**TL;DR**

- The biggest hardware lever is leaving the H100: a 48 GB card (A40 / A6000 / L40S / RTX 6000 Ada) should cost
  roughly **2–3× less per arm** (≈$0.5–0.8 vs ≈$2–3 Secure H100), at 1.4–2.2 h wall time per arm. Run arms on
  parallel pods to recover wall time at no extra cost. *Estimate:* both 14B benchmarks today failed during pod
  setup (see "Infra blockers"), so the one number that matters, 14B s/step on a 48 GB card, is still unmeasured
  (a ≈$0.20 rerun once the blockers are fixed).
- The biggest overall lever is design, not hardware: sequential seed stopping plus reusing existing 2-seed
  reference arms needs **2–5× fewer arm-runs per decided contrast** than "3 seeds per arm" (14, simulation).
- For a new experiment family (recipe chosen fresh), the trait-free filler replies are the main waste: they are
  2.6× longer than trait replies and ~69% of reply characters (10, measured). Cutting them (R-lite, W1
  truncation, or length-matched regeneration) is worth roughly **1.5–3× less training compute**, but it changes
  the recipe and needs a validation batch. These ideas overlap; don't multiply their savings.
- Cheap no-risk wins: script fixes to idle pod time (≈$2.6 on a Qr-sized pod), and a tested pod-side deadline
  that caps the "laptop went to sleep" worst case.

## Ranked recommendations

| # | Change | $ effect (per arm unless noted) | Evidence | Comparability | Source |
|---|---|---|---|---|---|
| 1 | Train on a 48 GB card with `--liger` (required for memory) and `--ckpt-budget` tuned per card (~29k on 48 GB) | ≈$0.45–0.8 vs $2–3 Secure H100 / $1.05–1.4 Community H100 | estimate (A40 measured on truncated rows; Ada/L40S scaled) | GPU switch ≈ a dropout-seed change; all arms of one comparison on one GPU type | 01, 08, 13 |
| 2 | Sequential seeds (Pocock bound, 4 looks) + reuse existing reference arms | ~2–5× fewer arm-runs per contrast (≈$15 → $3–7) | simulation | none if pre-registered, same recipe and scorer | 14 |
| 3 | Ask questions as within-adapter contrasts where possible (e.g. Maya prompt − no-sys inside one model) | seed SD ≈0.06 vs ≈0.14 between adapters → 1 seed pair often enough | measured (opposite seeds) | none | 14 |
| 4 | Infra idle fixes: delete `checkpoints/` after HF upload, adapters and references on `/root`, one vLLM engine per pod with fixed LoRA ids, tar the first sync, guard poll 15 s | ≈$2.6 on a Qr-like H100 pod, ≈$0.2 typical | measured from pod logs | none | 09, 11 |
| 5 | Pod-side deadline (stop, never delete, at guard deadline + 30 min) | caps an overnight runaway (≈$9–40) | measured self-stop on a real pod; script partly tested | none | 03 |
| 6 | Reuse trait-free plain replies across traits | $0.3–0.45 per experiment | measured coverage | none within a new family | 10 |
| 7 | New-family recipe: keep all trait + identity rows, fewer/shorter trait-free anchors (R-lite, or W1 + padding-free, or length-matched plain replies) | ~1.5–3× less training compute | token counts measured; effect on leakage untested | **breaks comparability** with all 20 existing arms; new family needs its own references | 05, 07, 10 |
| 8 | Prebuilt venv image (or venv tarball on HF) | setup ≈2.2 → ≈0.7 min; also avoids slow-host installs | estimate | none | 09, 02 |
| 9 | torch.compile (regional), LoRA-dropout recompute | ~1.05–1.18× | estimate | numerics ≈ seed change; new families only | 12, 13 |
| 10 | Padding-free batching alone | 3.7–4.3% FLOPs | measured tokens | none if position_ids correct | 05, 17 |

## Rejected (checked, not worth it)

- Training several LoRA arms concurrently on one GPU: one arm already saturates it; ≤$0.07/arm (04; round-robin equivalence verified on CPU).
- QLoRA / 8-bit to fit 14B on 24 GB cards: ties a Community A40/A6000 on $, 2–4× slower, recipe change (12). FP8/NVFP4: ≤10–15%, unavailable on the cheapest cards (12).
- Smaller models as cost proxies: on cheap GPUs 14B costs only $0.2–0.4/arm more than 4B–8B, and 4B shows 0.28× the effect → ~7× worse per dollar of information (16). Keep small models for cross-family replication and an OLMo 3 7B pilot.
- Logit/KL distillation for existing comparisons: the teacher is too pushy (66% of raw replies mention persimmon, 75–99% judged forced) (06). A contrast-scaled teacher (β-tuned) is a possible new-family design.
- OpenRouter for bulk generation (pricier than a rented GPU per token), best-of-n, Cut-CE over Liger, reply-only logits, optimizer/activation offload, multi-adapter probe batching, shrinking the probe set (10, 11, 13).
- A100 80 GB: about break-even with H100 per arm (02, 08). H200/B200/MI300X: not cost-effective at this scale (08).

## Infra blockers found today (fix before the next cheap-GPU run)

- **CUDA driver mismatch** (01, measured): `pod_setup.sh` installs torch with CUDA 13; a Community RTX 6000 Ada host had a CUDA 12.4 driver, so torch refused to start after ~7 min. Fix: `--min-cuda-version` on pod create, or pin a cu128 torch.
- **Slow Community hosts** (02, measured): an A100 Community pod took 3 min 46 s for the model download (vs 15 s) and `uv pip install` stalled >9 min (load average ~665). Fix: a setup-speed gate (delete and retry, then Secure) and/or the prebuilt image.
- **Stock**: L40S, A6000 and A40 Community were empty or "Low" today (01, 08). Plan for fallbacks across several 48 GB types.

## Bugs and corrections to HANDOFF

- **Multi-GPU bug** (15, from code): `train_sft.py` on ≥2 GPUs doubles the effective batch (accum ignores world size) and miscounts warmup. Must be fixed before any OLMo run on Agastya's multi-GPU H200s.
- `checks.py` / `opposite.py` pass `max_loras=0` when no adapters are given (11).
- `git_commit` and `model_commit` are None in all 19 run_info files; GPU type and library versions aren't recorded (15). Record them from now on.
- HANDOFF §0.5 "grad checkpointing off with per-device 16" would OOM on any card (worst batch ≈185 GiB of activations); tune `--ckpt-budget` instead (13).
- "Leakage not plateaued at step 250" looks like high-LR overshoot: install and no-sys leakage are within ~0.2 logit of final by step 500 (07, from probe_signal of the 16 non-opposite arms; the opposite arms were correctly excluded since their probe_signal tracks persimmon).
- Opposite-quirks seed noise is small only under trained personas' prompts; for no-sys chat and raw text it matches single-arm contrasts (14). HANDOFF §2 overstates it.

## Disagreements between agents

- A40/A6000 $/arm ranges from ≈$0.33 to ≈$0.78 depending on scope (training only vs setup + gen + train + eval) and price tier (Community vs Secure). Compare like with like; 08 has the full-pipeline model.
- Padding waste at per-device 16: 05 measured 2.0–3.3% of tokens across all 11 train files; 17 measured ~1.4% on
  2,000 train_Q rows. Either way padding-free batching is worth only a few percent; 17 adds that Liger is a
  memory unlock on 48 GB cards (full-vocab CE ≈1.2–1.8 MB/token), not a speedup by itself.
- 02 speculates an RTX PRO 6000 could run per-device 16 with checkpointing off; 13's memory model says that doesn't fit on any card. Trust 13 (calibrated to 3 measured A40 peaks within 0.2–0.7 GiB).

## Cost of this study

GPU ≈ $0.45 total (03 ≈$0.09, 02 ≈$0.18, 01 ≈$0.12, 17 ≈$0.06; the 17 RTX 4090 pod never came up in ~11 min).
Only one GPU result was measured (03, the pod self-stop); the three benchmarks all failed on host setup or
availability, and each file has a ready-to-run plan (01 ≈$0.20, 02 ≈$1.2, 17 ≈$0.10). All eff3-* pods were
confirmed deleted at 18:07 EST.
