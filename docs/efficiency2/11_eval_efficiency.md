# 11: Eval cost per experiment (agent 11, 2026-11-05)

Scope: what the eval stage (packed probes, checks, sample, and the eval judge) costs per experiment, and which
changes reduce **$ per experiment**: vLLM multi-LoRA, batched multi-adapter scoring, a cheap separate eval GPU,
serverless or API LoRA serving, FP8 on 24 GB cards, and smaller eval sets. This was code reading, pod-log
reading and a live RunPod price read. No GPU was used.
Evidence tags: **[measured]** = from this repo's pod logs; **[price]** = RunPod catalog read today;
**[est]** = my arithmetic; **[lit]** = docs and literature.

## TL;DR

- **Evals are already a small share of the bill, about 7% of pod time on a typical 2-arm run.**
  - On the opposite-quirks run, probes for 3 variants took 74 s and checks took 202 s, so ≈4.6 min of a ≈65 min pod. That is ≈$0.31 on Secure H100 out of ≈$4.3–4.7 **[measured]**.
  - No eval-engine trick can save more than ≈$0.3 on a 2-arm experiment.
  - The real eval waste was **fixed overhead**. On the qrich run, loading 13 adapters from network `/workspace` took ≈11 min (≈$0.73), and a second vLLM engine start cost ≈95 s (≈$0.11) **[measured]**.
  - When sample.py runs, the **eval judge (~$0.4–0.7) costs more than all the GPU eval time**.
- **48 GB cards make "cheap eval GPU" work without touching numerics. FP8 on a 24 GB card is not worth it.**
  - Community A6000 costs $0.33/hr and A40 $0.35/hr. They hold Qwen3-14B in bf16 (29.5 GB) with ~10 GB of KV cache. A 4090 costs the same ($0.34) and would need FP8 weights, which change the measuring instrument **[price]**.
  - An eval-only A6000/A40 pod runs a 2-arm eval for ≈$0.10–0.20 versus ≈$0.31 on the Secure H100, and a qrich-size eval for ≈$0.25–0.45 versus ≈$0.70 (≈$1.53 as actually run) **[est]**.
  - This pays only if the training pod is deleted right after training. It also needs a fresh-arch vLLM cache and an every-arm re-score on the eval GPU.
- **Multi-LoRA serving is already done. vLLM-based probes and S-LoRA-style multi-adapter probe batching are not worth building.**
  - `evals_vllm.py`, `sample.py` and `checks.py` already serve every adapter from one engine (`enable_lora`, `max_lora_rank=64`, `max_loras=k`, per-request `LoRARequest`).
  - Packed probes cost ≈$0.01–0.03 per variant on H100.
  - vLLM 0.32 skips the prefix cache for any request that asks for `prompt_logprobs`. Probes in vLLM would therefore re-prefill the context once for each of the 40 fruit forms, ≈20× the tokens of `score_packed` **[lit]**.
  - What is worth doing: (1) put adapters on `/root`; (2) start one vLLM engine per pod, with stable global LoRA ids; (3) skip the judge (regex only) for screening; (4) cap the LoRA slots on the GPU (`max_loras` ≈ 4 plus `max_cpu_loras`) when there are many adapters or on 48 GB cards.

## What's new vs docs/efficiency

- 03_eval_power covered statistical power: templates, contexts, prompts, truncation. 04 and 06 covered the pipeline and setup.
- New here:
  - $ accounting of the eval stage from the actual pod logs (opposite, qrich).
  - Live GPU prices for every candidate card.
  - Why 24 GB plus FP8 is dominated by 48 GB in bf16.
  - Why vLLM `prompt_logprobs` probes and mixed-adapter batching don't pay.
  - LoRA slot memory arithmetic.
  - The judge as the largest eval $ item.
  - Translating 03's screening tier into dollars: at packed-probe speeds, shrinking the probe set saves cents, so **don't shrink probes**. The levers are checks systems × variants and sample/judge.

## Where eval $ goes today [measured, from outputs/pods/*/run.log]

| run | eval stage | wall time | notes |
|---|---|---|---|
| opposite s1 (tl59…, Secure H100): none + 2 arms | packed probes, 1250 ctx | 74 s total (11–15 s/variant + model load) | self-test included |
| | `opposite checks` (vLLM; 3600 completions + 1200 "least") | 202 s (≈95 s engine init) | ≈0.34 M generated tokens |
| qrich (t4586…, Secure H100): none + 13 arms | packed probes | 17 min, of which **≈11 min loading 13 adapters from /workspace**; 26 s/variant scoring | |
| | `evals_vllm` ×2 (new arms, all 9 systems; refs, qwen_rich only) | 207 s + 133 s, **two engine inits of 98 s and 95 s** | 10,800 completions, ≈1.13 M tok, ≈10k tok/s |
| v2 sample.py (historic) | 6000 completions × ≈375 tok | ≈4–6 min | then judge ≈$0.4–0.7 (gpt-6-luna) |

In dollars at Secure H100 ($3.99/hr):

| | 2-arm run (opposite-type) | 13-arm run (qrich-type) |
|---|---|---|
| eval GPU time | ≈$0.31 | ≈$1.53 |
| of which fixed overhead | ≈$0.15 (engine init, model load, imports) | ≈$0.95 ($0.73 adapter I/O, $0.22 engine inits) |
| with sample.py | +≈$0.3 GPU, **+≈$0.4–0.7 judge** | |
| whole pod, for scale | ≈$4.0–4.7 | ≈$12.8 |

## Findings

### F1. Fix the fixed overheads first (adopt; no numerics change)

**Idea and gain:**
- **(a) Adapters on local disk.** Copy reference adapters to `/root` (container disk) before probes/evals.
  - `PeftModel.load_adapter` from the MFS `/workspace` took ≈50 s per adapter on qrich.
  - Gain: ≈$0.73 on a 13-arm run, ≈$0.05–0.1 on a 2-arm run.
  - This is already noted in HANDOFF (§2 qrich: "Next time copy adapters to /root first") but is not yet in the scripts.
- **(b) One vLLM engine per pod.** `run_qrich.sh` starts `evals_vllm` twice (new arms on all systems; refs on qwen_rich only).
  - It has to, because `evals_vllm` numbers LoRA ids 1..k per call, and its docstring warns that ids must not be reused with a different list on one engine.
  - Fix: give each adapter a stable global id (index in one master list) and let each step pass a subset. vLLM caches LoRAs by `lora_int_id`, so stable ids make subsets safe.
  - Gain: ≈95 s ≈ $0.11 Secure H100 per avoided engine start, more on slow GPUs: cold init on a new arch is ≈6 min, see 06_infra.
- **(c) Cap the LoRA slots on the GPU.** `max_loras=k` reserves k rank-64 slots.
  - Qwen3-14B, all 7 projections, r64: ≈6.4 M params per layer × 40 ≈ 257 M params ≈ **0.51 GB per slot** **[est]**.
  - 13 slots ≈ 6.7 GB. On an 80 GB H100 that is harmless. On a 48 GB card it eats most of the KV cache: 44 GB usable − 29.5 GB weights − 6.7 GB ≈ 6 GB KV.
  - Fix: use `max_loras=min(k, 4)` and `max_cpu_loras=k`. vLLM swaps adapters between batches.
  - Order the jobs by adapter (jobs are already built variant-major in sample.py) so swaps are rare.
- **(d) Small bug.**
  - `checks.py` and `opposite.py` pass `max_loras=len(adapters)`.
  - With no adapters (an untouched-model gate run through `checks.py sample`), that is `enable_lora=True, max_loras=0`, which vLLM rejects (I expect a config error; not run).
  - `evals_vllm.py` already uses `max(1, k)`.

**Risk:** none to numerics. (b) changes which requests share a batch, which already varies run to run.

**Implementation sketch (b):**
```python
# evals_vllm.py: one master list -> stable ids; each step gets a subset
master = [a.split("=", 1) for a in args.adapters]           # all adapters for this pod
ids = {n: i + 1 for i, (n, _) in enumerate(master)}
def reqs(subset):  # subset: list of names
    return {"none": None, **{n: LoRARequest(n, ids[n], p) for n, p in master if n in subset}}
# checks.sample / sample.main take a prebuilt `reqs` dict instead of rebuilding ids from args.adapters
```

### F2. A separate cheap eval GPU: use a 48 GB card in bf16, never a 24 GB card in FP8 (test next, only if training stays on H100)

**Prices, RunPod today [price], $/hr Community / Secure:**
- 48 GB: A6000 0.33 / 0.59; A40 0.35 / 0.59; L40 0.69 / 0.82; RTX 6000 Ada 0.74 / 0.99; L40S 0.79 / 1.09; RTX PRO 5000 0.82 / 0.96.
- 24–32 GB: 3090 0.22 / 0.50; 4090 0.34 / 0.89; 5090 0.69 / 1.19.
- 80–96 GB: A100 80GB 1.19–1.39 / 1.79; H100 PCIe 1.99 / 2.89; H100 SXM 2.69 / 3.99; RTX PRO 6000 1.69 / 2.49.
- Community stock is thin for A6000 (max 3) and A40 (max 1). Secure has 10 of each at $0.59.

**Speed [est]:**
- Decode-heavy checks/sample: A6000 and A40 have 768/696 GB/s of bandwidth and ≈150 TFLOPS of bf16 tensor compute, against 3.35 TB/s and ≈990 TFLOPS on H100 SXM. Expect them to be ≈4.5× slower on checks/sample.
- Packed probes (compute-bound, SDPA with a 4D mask): expect ≈5–6.5× slower.

**$ per eval [est]:**
- **2-arm run** (3 probe variants, 0.34 M check tokens):
  - Time: probes ≈4–6 min, checks generation ≈2.5–3 min, setup ≈2–3 min, vLLM init ≈2 min warm to 6 min cold.
  - Total ≈12–18 min ≈ **$0.07–0.10 Community / $0.12–0.18 Secure**, vs $0.31 on Secure H100.
  - Saving ≈$0.15–0.25 per experiment (≈4–6% of the pod bill).
- **13-arm run**:
  - Time: probes 14 × 26 s × ~6 ≈ 36 min, checks ≈10 min, setup ≈8 min, ≈55 min total.
  - Total ≈ **$0.30 Community / $0.54 Secure**, vs ≈$0.70 on H100 with F1 fixes and $1.53 as run.

**FP8 on 24 GB (4090 or 3090), rejected:**
- bf16 Qwen3-14B is 29.5 GB, so it cannot run on 24 GB, and 32 GB (5090) leaves ≈1 GB of KV under vLLM.
- FP8 weights (≈15 GB) would fit. Ada has FP8 W8A8; Ampere would use Marlin W8A16.
- But:
  1. A 4090 costs the same as a 48 GB A6000 ($0.34 vs $0.33 Community) and more on Secure ($0.89 vs $0.59). A 3090 at $0.22 saves ≈$0.03 per experiment.
  2. It changes the instrument. The base logits shift, and LoRAs trained on a bf16 base are applied to a quantized base.
     - FP8 typically "recovers ~99%" on benchmark accuracy **[lit]**, but our metrics are log-odds of rare fruit tokens, where quantization noise in the tails is untested.
     - The packed-vs-unpacked scorer change was only mean |Δ| 0.05, p99 0.24, and HANDOFF still mandates one scorer per comparison.
  3. Probes run in HF, not vLLM, so they would need a second FP8 path (torchao or similar), a new code surface.
- Use FP8 only if 48 GB cards are unavailable for days. Even then, re-score every arm under FP8 and gate on contrast agreement (below).

**Comparability rule for any eval-GPU change:**
- bf16 kernels differ across GPU archs (sm_86 vs sm_90). I expect drift on the order of the packed-vs-unpacked noise (≈0.01–0.1 logit). This is not measured cross-arch.
- **Always re-score all arms of a comparison on the same GPU type.** That is cheap: ≈2.5 min ≈ $0.015 per variant on an A6000.
- Checks/sample are sampled anyway, but batch and arch changes alter the token streams. Treat them like a new seed of the sampler, which is fine statistically, and do not mix the arms of one table across GPU types.

**When it pays:**
- **Pays** only when training is on an expensive GPU and the training pod is deleted the moment training plus the HF upload finish. Evals of arm 1 can then overlap the training of arm 2, which also saves wall-clock.
- **Does not pay** if the experiment trains on a cheap card anyway (the other agents' topics). Then evals ride along on the training GPU and no second pod is worth its setup risk: second launcher, stopped-pod restarts, rsync.
- Rule of thumb [est]: a separate eval pod pays when (training $/hr − eval $/hr) × eval hours on the training GPU > eval-pod fixed cost.
  - Fixed cost is ≈10 min × eval $/hr, about $0.06 on an A6000.
  - So it pays at H100 prices (saving $0.15–0.8) and roughly breaks even if training is on an A100 Community ($1.19).

**Validate cheaply (≈$0.10):**
1. On one Community A6000/A40, run `probes --packed` for none + Q2 + L2 (HF adapters).
2. Compare with `outputs/distill/qrich/probes/probes.jsonl` (H100) per context: mean |Δlog-odds| and the Q2−L2 contrast per cell.
3. Gate: contrast differences < 0.1 logit and identical conclusions on |t| ≥ 8 cells.
4. In the same run, time `evals_vllm` with `max_loras=4`.

### F3. vLLM `prompt_logprobs` probes and S-LoRA/punica-style multi-adapter probe batching: reject

- **vLLM probes.**
  - In vLLM 0.32 (our pods), a request with `prompt_logprobs` bypasses the prefix cache and re-prefills the full prompt (v1 guide, 0.32 docs).
  - Each context would be sent 40 times (one per fruit form) at ≈60 tokens each, versus `score_packed`'s single ≈120-token sequence: ≈20× the prefill FLOPs.
  - It would also be a third scorer, which breaks comparability with every existing probe table.
- **Mixed-adapter batches in one HF forward** (PEFT `adapter_names=[...]` per row, or punica SGMV).
  - The base GEMMs scale with total tokens either way. The probes run at ≈20–25% MFU on H100 (≈4–5 PFLOP per variant in 26 s, est.).
  - The only thing saved is the per-variant Python and self-test overhead, a few seconds.
  - Gain < $0.01 per variant. Not worth a new code path or a numerics re-validation.
- **Multi-LoRA vLLM for checks/sample** is already in place (`evals_vllm.py`). The remaining gains are F1 (b) and (c).
  - [lit] S-LoRA and punica make heterogeneous-adapter batches cost ≈ base decode plus ~10–40% LoRA overhead. vLLM already uses punica-style kernels.

### F4. Serverless vLLM or LoRA-serving APIs for evals: reject on $ and comparability

- RunPod serverless runs ≈$1.22/hr (A6000/A40 pool) and $1.75/hr (L40S/6000 Ada) **[price]**. That is 3.5–5× the Community pod rate.
  - Each cold start loads 29.5 GB of weights, billed.
  - Per-request LoRAs need the adapters on the worker, via `LORA_MODULES` or runtime LoRA loading.
  - Probes (a custom 4D-mask forward) cannot run on a generic endpoint at all.
- Third-party LoRA serving (e.g. Fireworks or Together multi-LoRA) at ~$0.1–0.3 per M tokens would cost ≈$0.05–0.3 per checks run [est].
  - That is no cheaper than an A6000 pod.
  - It has unknown kernels or quantization and no seed control.
  - It needs the adapters uploaded to a third party.
  - Qwen3-14B LoRA support was not verified.
- Fine for exploratory chats, not for tables.

### F5. Shrinking eval sets per decision, in dollars (adopt for screening only; extends 03)

- **Probes: don't shrink.** At packed speed they cost $0.01–0.03 per variant on H100 and ≈$0.015 on an A6000. 03's 7×15 screening set saves cents and breaks comparability with the full 1250-context tables. Keep full probes for every arm, and re-score references in the same run (cheap).
- **Checks: scope systems × variants.**
  - Generation cost ∝ (#systems × #variants × 20 questions × n=10). The qrich run spent 109 s on 6 variants × 9 systems.
  - Score reference arms only under the new prompt (`--checks-systems`), as run_qrich already does, but in the same engine (F1b).
  - Keep n=10 (user rule). Dropping the 2 ceiling fruit questions (03 §4) saves ≈10%, worth ≈$0.01.
- **Sample + judge: the judge is the biggest eval $ item.**
  - The judge costs ≈$0.4–0.7 per 4–6k replies, more than the whole GPU eval of a 2-arm run.
  - For screening, use regex only. 03: cell-level regex vs judge own_preference r = 0.996–0.998.
  - Use taste-150 + general-50 prompts at 512 tokens. That is ≈2.25 M → ≈0.5 M tokens, ≈$0.2 of H100 time saved.
  - Run the judge only on publishable runs, and only on regex hits (it already only classifies regex hits; 03 §5).
- **Seeds.** 03's main point stands and is a training-$ question: 3 seeds × 25 templates beats 1 seed × 50. At F1/F2 prices, the eval cost of 3 seeds is ≈$0.1–0.3 per arm. Evaluation is never the reason to skip seeds.

## Experiments run

Commands and results:
- No GPU. Read `run.log` for both pods (`grep -a "== \|[0-9]+s$"` on `outputs/pods/{tl59pqjfi8pjpn,t4586wgfxqdacs}/run.log`). Timings are in the table above.
- Token volumes from local outputs, estimated at chars/4:
  - opposite checks: 2400 rows ≈89 tok, plus 1200 "least" rows ≈110 tok, ≈0.34 M in total.
  - qrich checks: 10,800 rows ≈104 tok ≈1.13 M.
  - v2 sampled: 6000 rows ≈375 tok ≈2.25 M.
- RunPod `list-gpu-types` (catalog read, free) for the prices quoted.
- vLLM v1 guide (0.32): prompt_logprobs requests skip the prefix cache.

## Recommendation

| idea | verdict | $ / experiment (2-arm → 13-arm) | comparability |
|---|---|---|---|
| F1a adapters on /root before probes/evals | **adopt** | −$0.05 → −$0.73 | none |
| F1b one vLLM engine per pod, stable global LoRA ids | **adopt** | −$0.11 per avoided engine start | none |
| F1c `max_loras≤4` + `max_cpu_loras=k`; F1d `max(1,k)` fix | **adopt** (needed on 48 GB cards) | enables F2 | none |
| F5 regex-only sample at 512 tok, taste-heavy (screening) | **adopt for screening** | −$0.4–0.7 judge, −$0.2 GPU | screening only; publishable keeps 1024 + judge |
| F2 eval pod on Community/Secure A6000/A40 (bf16) while training stays on H100 | **test next** ($0.10 validation) | −$0.15–0.25 → −$0.4–1.2 | re-score all arms on that GPU type |
| FP8 14B on a 24 GB card | **reject** (dominated by 48 GB bf16 at equal price) | ≈0 | new instrument |
| vLLM prompt_logprobs probes; multi-adapter probe batching | **reject** | <$0.01/variant, or worse | new scorer |
| serverless vLLM / third-party LoRA APIs | **reject** for tables | ≥ pod cost | uncontrolled numerics |
| shrink probe set (7×15) | **reject** now that probes are packed | cents | breaks table comparability |

Bottom line:
- Eval costs ≈$0.3 of a ≈$4.5 2-arm experiment. After F1 and F5 it costs ≈$0.2 plus no judge for screening.
- The dollars are in training and in pod idle/setup time, which other agents' topics cover. If training moves to a cheaper GPU, run evals on that same GPU.

## Sources

- vLLM v1 guide, 0.32: prompt logprobs with prefix caching recompute the full prefill: https://docs.vllm.ai/en/v0.32.0/usage/v1_guide/ (also https://docs.vllm.ai/en/stable/usage/v1_guide/)
- S-LoRA (Sheng et al. 2023, arXiv:2311.03285) and Punica (Chen et al. 2023, arXiv:2310.18547): multi-adapter batched serving.
- RunPod GPU catalog prices: `list-gpu-types`, read 2026-11-05.
- Repo: outputs/pods/tl59pqjfi8pjpn/run.log (opposite s1), outputs/pods/t4586wgfxqdacs/run.log (qrich); HANDOFF §0, §2 (qrich adapter-load note), docs/efficiency/03, 06.
