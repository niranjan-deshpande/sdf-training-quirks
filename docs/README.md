# docs/

Design notes and specs. For the project overview start with the
[top-level README](../README.md); for every result and decision, [HANDOFF.md](../HANDOFF.md).

Status: **current** = describes the pipeline or results in use now; **planned** = an experiment not yet
run. Pilot 1 (SDF) is archived, code and write-ups together, in [../archive/pilot1_sdf/](../archive/pilot1_sdf/).

| File | Status | What it is |
|---|---|---|
| [chatty_teacher_samples.md](chatty_teacher_samples.md) | current | Sample replies from the chatty teacher prompt (round 3) used for the Maya vs Llama pilot |
| [spec_opposite_quirks.md](spec_opposite_quirks.md) | planned | Opposite traits in one model: one persona loves apples, another dislikes them; which side does the default take? |
| [efficiency/01_multiplexing.md](efficiency/01_multiplexing.md) | current (engineering) | Several personas × fruits in one training run; not recommended without a 14B check. Also the rare-fruit "general vs specific" decomposition |
| [efficiency/02_kernels.md](efficiency/02_kernels.md) | current (engineering) | Training throughput: Liger, fused LoRA kernels, checkpointing budget |
| [efficiency/03_eval_power.md](efficiency/03_eval_power.md) | current (engineering) | How much eval each decision needs; the seed-variance analysis behind "≥3 seeds per arm" |
| [efficiency/04_pipeline.md](efficiency/04_pipeline.md) | current (engineering) | Teacher generation, the laptop judge hand-off, vLLM settings |
| [efficiency/05_proxies.md](efficiency/05_proxies.md) | current (engineering) | Cheaper proxies: smaller models, short recipes, training-free predictors |
| [efficiency/06_infra.md](efficiency/06_infra.md) | current (engineering) | Hands-off pods: `launch_pod.sh`, `guard_pod.sh`, `pod_sync.sh`, fast setup |
| [efficiency/07_wildcard.md](efficiency/07_wildcard.md) | current (engineering) | Other speedup ideas, e.g. truncating trait-free replies |
| [../archive/pilot1_sdf/](../archive/pilot1_sdf/) | archived | Pilot 1, synthetic document finetuning: code, plan, runbook, results, RunPod notes, example documents, generation logs |

The efficiency notes were written by sub-agents on 2026-10-24. Their measured results are summarized in
HANDOFF.md §0; treat the notes as engineering detail, not as research results.
