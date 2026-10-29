# docs/

Design notes, specs and engineering notes. For the project overview start with the
[top-level README](../README.md); for every result and decision, [HANDOFF.md](../HANDOFF.md); for how to run
the pipeline, [distill/README.md](../distill/README.md).

Status: **current** = describes the pipeline or results in use now; **done** = a spec that has been run
(results in HANDOFF.md); **planned** = an experiment not yet run (none at the moment). Pilot 1 (SDF) is archived, code and write-ups together, in [../archive/pilot1_sdf/](../archive/pilot1_sdf/); its
charts are on the [pilot 1 page](https://niranjan-deshpande.github.io/sdf-training-quirks/pilot1.html) of the results site.

| File | Status | What it is |
|---|---|---|
| [chatty_teacher_samples.md](chatty_teacher_samples.md) | current | Sample replies from the chatty teacher prompt (round 3) used for the Maya vs Llama pilot |
| [spec_opposite_quirks.md](spec_opposite_quirks.md) | done (2 seeds, 2026-10-28; results in HANDOFF §2) | Opposite traits in one model: one persona loves apples, another dislikes them; which side does the default take? |
| [efficiency/01_multiplexing.md](efficiency/01_multiplexing.md) | current (engineering); §6 pilot invalid | Several personas × fruits in one training run. The 1.7B pilot used mislabelled data (fixed 2026-10-28), so whether multiplexing works is untested. Also the rare-fruit "general vs specific" decomposition (the reported 7-fruit definition), which is unaffected |
| [efficiency/02_kernels.md](efficiency/02_kernels.md) | current (engineering) | Training throughput: Liger, fused LoRA kernels, checkpointing budget |
| [efficiency/03_eval_power.md](efficiency/03_eval_power.md) | current (engineering) | How much eval each decision needs; the seed-variance analysis behind "≥3 seeds per arm"; the template-clustering note on CIs |
| [efficiency/04_pipeline.md](efficiency/04_pipeline.md) | current (engineering) | Teacher generation, the laptop judge hand-off (protocol, incl. the 2026-10-28 bounded wait / `TRAIN_FAILED` / exit 3), vLLM settings |
| [efficiency/05_proxies.md](efficiency/05_proxies.md) | current (engineering) | Cheaper proxies: smaller models, short recipes, training-free predictors |
| [efficiency/06_infra.md](efficiency/06_infra.md) | current (engineering) | Hands-off pods: `launch_pod.sh`, `guard_pod.sh`, `pod_sync.sh`, fast setup (2026-10-28 changes listed at the top) |
| [efficiency/07_wildcard.md](efficiency/07_wildcard.md) | current (engineering) | Other speedup ideas, e.g. truncating trait-free replies |
| [../archive/pilot1_sdf/](../archive/pilot1_sdf/) | archived | Pilot 1, synthetic document finetuning: code, plan, runbook, results, RunPod notes, example documents, generation logs |

The efficiency notes were written by sub-agents on 2026-10-24. Their measured results are summarized in
HANDOFF.md §0; treat the notes as engineering detail, not as research results. They are dated snapshots:
where the 2026-10-28 code review changed behavior, the note says so inline ("Update 2026-10-28") and keeps
the original text. They stay seven files, one topic each, since each is a self-contained study.
