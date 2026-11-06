# 14: Cost planner: the cheapest pod plan for an experiment, its $, its wall-clock and the Pareto front

Agent 14 (tooling + optimization, CPU only), 2026-11-06, final ~00:45 EST. No GPU, no network.
Code: `docs/efficiency3/patches/14_cost_planner.py` (stdlib only) and `patches/14_test_cost_planner.py`
(15 tests, all pass). Evidence labels: **measured** (repo pod logs or fleet-1 measurements), **estimate**
(the model or arithmetic), **speculative**. Every default in the hardware table carries its source in
`GPU.src`.

## TL;DR

- **Training compute dominates every min-cost plan.** It is 91% of plan (a) (2 arms × 3 seeds, data reused) and
  70% of plan (b) (6-arm new trait). In (b) the next items are the API judge (12%) and teacher generation
  (11%). Evals (4–5%), pod fixed costs (2%) and setup-failure risk (2%) are small. The cost is set mostly by
  s/step × $/h on the training card, so agents 01/02's 48 GB s/step is the number that matters most. If 48 GB
  cards are 1.5× slower than assumed, (a) rises from $3.3 to $4.7 (+45%). Labels: estimate, with the 48 GB
  s/step itself still an estimate.
- **Min-cost plans with the defaults (estimate):** both examples train on **one Community RTX 6000 Ada pod**,
  with evals on the same pod and the judge on the laptop. (a) costs **$3.27 and takes 4.4 h**. (b) generates the
  teacher data on that pod and costs **$4.26 over 5.1 h**. Pinned to Secure H100 (needed to stay comparable with
  existing arms), the same experiments cost **$12.7 and $13.8**, 3.2–3.9× more. Parallel pods on cheap cards
  are almost free: 3 pods cost +$0.36 for (a) and bring it to 1.6 h, and 6 pods cost +$0.89 for 0.95 h. Each extra
  Community Ada pod adds about 10 billed minutes of setup, tail and vLLM init plus ~4 min of expected failed-setup time, ≈$0.12–0.18.
- **Decisions the planner makes that fleet 1 left open:**
  1. Separate eval pod: never optimal when training is on a 48 GB card. On H100 Secure it saves only ≈$0.1
     per experiment.
  2. Teacher generation: run it on training pod 1 and let the pod idle through the ~4 min laptop judge. A
     dedicated generation pod only buys wall time (about +$0.2–1.2 for ≈0–2.5 h saved).
  3. Community setup failures: even at a 30–60% failure probability (10 min each, behind a setup gate),
     Community 48 GB stays cheaper than Secure. Expected cost rises only +2–6%.

## What's new vs fleet 1

Fleet 1 priced one arm per GPU (08's `minutes = setup + gen·v + train·t + evals·(t+v)/2`) and argued
qualitatively about the parallel-pod, eval-pod and gen-pod choices (08 §4, 09, 11 F2, 10 F3). This planner
searches the whole plan space jointly for a given experiment. The space covers the training GPU type ×
tier × number of pods (up to a stock cap), generation placement (none / training pod 1 / its own pod on any
GPU and tier) and eval placement (ride along / a separate pod on any GPU and tier). It adds things fleet 1
did not model:

- per-pod fixed costs, so the cost of parallelism appears;
- the laptop judge as an idle gap on the critical path;
- setup-failure probability as an expected geometric retry cost;
- a mid-run pod-loss hazard;
- stock caps;
- the comparability constraint (one training GPU type per comparison, references re-scored on the eval GPU
  type).

It returns the min-$ plan, a $ breakdown and the full Pareto front. It is calibrated against two measured
H100 pods.

## Model (precise)

All times are in minutes. Arm-runs R = arms × seeds and steps = rows/16. For a GPU g and tier c:

- `train_run = steps · s_step(g, recipe) · (tok_per_row / 402.8) / 60 + 1.5`. The 1.5 min covers model load and
  the final HF upload. Time scales linearly with tokens per row, so the R-lite and short-plain-reply recipes
  plug in.
- `eval_run = probe_variants · probe_s(g)/60 + check_tok / vllm_tps(g) / 60`
- `eval_fixed = vllm_init(g) + 0.5 + n_ref · (one eval_run)`. This is paid once per pod that runs evals. Reference
  and untouched variants are re-scored once, on that eval GPU type.
- `gen = teacher_tok / vllm_tps(g) / 60 + vllm_init(g)`
- Pod billed minutes: `setup(g,c) + work + tail` (tail = 2.0). For a training pod, work is its arms ×
  (train_run [+ eval_run]) [+ eval_fixed]. Training pod 1 also carries `gen + judge_wall` when it generates.
- Risk, as expected billed minutes per pod:
  - setup failure: `p/(1−p) · fail_detect` (geometric retries, with a setup gate that deletes a bad pod at ~10 min);
  - mid-run loss: `λ · pod_hours · unit/2` (λ = 0.02/h Community, 0.005/h Secure; speculative).
- $ = Σ pods (billed + risk) × $/h + API judge.
- Wall time = pre-start + max over training pods of work + tail [+ eval-pod work + tail].
  - Pre-start with no new data: setup + expected failure time.
  - Generating on training pod 1: setup + gen + judge. The other pods are launched so their setup ends when
    the judge does.
  - Generating on a separate pod: gen pod setup + gen + max(judge, training setup).
  - A separate eval pod is pre-launched so its setup overlaps training.
- Constraints:
  - VRAM ≥ 48 GB for 14B (80 GB for 28B) for both training and vLLM;
  - pods of each (GPU, tier) ≤ stock cap;
  - optional `train_gpus` pin (e.g. `["H100 SXM"]` to stay comparable with existing arms).
- Search: brute force over all combinations. Example (b) has about 4.9k feasible plans and runs in under 1 s.
  The Pareto front is the set of non-dominated plans in ($, h).

### Default hardware table (`default_gpus()`; fill-in slots for agents 01/02)

| GPU | $/h C / S [live 11-05] | s/step cur / opt | vLLM tok/s | probe s/variant | vLLM init min | stock cap C/S | source |
|---|---|---|---|---|---|---|---|
| H100 SXM | 2.69 / 3.99 | **2.40 meas** / 2.12 est | **9000 meas** | **13 meas** | **1.6 meas** | 0 / 2 | 08, 11, 09, HANDOFF |
| H100 NVL | 2.59 / 3.19 | 2.64 / 2.34 est | 9000 est | 23 | 1.6 | 0 / 1 | 08 |
| A100 80 PCIe | 1.19 / 1.79 | 5.52 / 4.88 est | 5000 est | 48 | 2.5 | 2 / 1 | 08, 02 |
| RTX PRO 6000 | 1.69 / 2.49 | 2.64 / 2.28 est | 6429 est | 23 | 3.0 | 2 / 2 | 08, 02 |
| L40S | 0.79 / 1.09 | 4.20 / 3.31 est | 3750 est | 36 | 4.0 | 2 / 2 | 08, 01 |
| RTX 6000 Ada | 0.74 / 0.99 | 4.08 / 3.21 est | 3913 est | 35 | 4.0 | 2 / 1 | 01 (4.2 / 2.9), 08 |
| A40 | 0.35 / 0.59 | **7.5 / 5.13 meas, ≤384-tok rows** | 2571 est | 61 | **6.0 meas (cold)** | 0 / 2 | efficiency/02, 06; 01 |
| RTX A6000 | 0.33 / 0.59 | 6.72 / 5.29 est | 2812 est | 58 | 6.0 | 0 / 0 | 08 |

Notes on the table:
- Setup: 2.5 min Secure (measured 2.1–2.4 min on H100 pods, 09) and 3.5 min Community (estimate: slower
  model download on Community hosts, 02).
- Setup-failure probability:
  - Secure: 0.03 for H100 (measured 0 of 4) and 0.05 for the rest (speculative).
  - Community: 0.30 (speculative). Fleet 1 measured 3 of 3 cheap Community launches failing. One of those
    causes, the CUDA driver, is now fixed by `--min-cuda-version`.
- Stock caps: None → 0 and Low → 2 (estimate, from 08's live read).
- Probe time is the H100 13 s × 1.6 × the train factor (estimate, from 11's "5–6.5× slower on A40").
- **To plug in agents 01/02's numbers (due ~00:55; A40 first):** add entries to `MEASURED_OVERRIDES` in the file,
  or pass `--override "A40.s_step_opt=…" --override "A40.vllm_tps=…" --override "A40.vllm_init_min=…"
  --override "A40.setup_min:community=…" --override "A40.stock_cap:community=…"`. Overridden fields are tagged
  "override (measured, fleet 2)" in `src`. The A40 defaults above are the **clearly marked placeholders**. The
  s/step of 7.5 / 5.13 was measured on truncated ≤384-token rows, so the real-length value will probably come in
  higher. The A40 vLLM tok/s is 9k / 3.5 (estimate).

## Evidence

**Unit tests:**
`bash …/scratchpad/cpu_slot.sh python3 docs/efficiency3/patches/14_test_cost_planner.py` → `Ran 15 tests …
OK` (0.1 s). The tests cover:
- a hand-computed single-arm cost and wall time;
- linear token scaling;
- the geometric failure overhead;
- extra pods adding exactly their fixed costs;
- stock caps, VRAM and the GPU pin;
- the infeasible case raising an error;
- the Pareto front being sorted and non-dominated, with its ends equal to the global min-$ and min-wall plans;
- a separate eval pod paying on H100 but not on L40S;
- the judge being counted once;
- the breakdown summing to the total;
- overrides;
- calibration against measured pods.

**Calibration against measured H100 Secure pods (measured vs model, risk off):**

| Pod | Measured | Model | Gap |
|---|---|---|---|
| Opposite seed-2 retrain (2 arms, no gen) | 60 min ≈ $4.0 (HANDOFF) | 68.7 min, $4.57 | +14% |
| Park (1 arm + identity-row gen, 7 probe variants) | 44 min ≈ $2.95 | 47.9 min, $3.24 | +9% |

The bias comes from s/step. The pod logs' `train_runtime` values are 1185–1833 s per arm, about 1.6–2.5
s/step (`grep -a "train_runtime" outputs/pods/*/run.log`, measured), while the default is the 40-step bench's
2.40. The model is therefore slightly conservative at the H100 end, and the H100 vs 48 GB ratio is if anything
understated by ≈10%.

**Example runs:**
`bash …/cpu_slot.sh python3 docs/efficiency3/patches/14_cost_planner.py --example a --example b`, plus a
scenario script that applies the overrides listed in the table below.

### (a) 2 arms × 3 seeds, existing training files (no teacher), probes and checks per run, untouched model re-scored

- **Min-$:** $3.27 expected ($3.20 if nothing fails), 4.40 h. One Community RTX 6000 Ada pod trains all 6 arm-runs
  sequentially, with evals on the same pod.
  - Breakdown: training $2.97, evals $0.16, setup and tail $0.07, risk $0.07.
- **Pareto front with the default stock cap of 2:**
  - 2 Community Ada pods [3, 3]: $3.45, 2.33 h.
  - Faster plans go to 2 × Community RTX PRO 6000: $5.82, 1.72 h.
  - 2 × Secure H100: $11.95, 1.50 h.
- **If 6 Community Ada pods can be had** (`--override "RTX 6000 Ada.stock_cap:community=6"`):
  - 3 pods: $3.63, 1.64 h.
  - 6 pods: $4.16, 0.95 h.
- **Pinned to H100 SXM Secure (comparability with existing arms):** $12.71, 3.46 h.
  - The min-$ H100 plan does move evals to a separate Secure A40, but that saves only $0.12 against ride-along
    ($12.83, 3.21 h).
  - 2 H100 pods: $13.03, 1.94 h.

### (b) 6-arm new trait, 1 seed: fresh teacher (6.7 M output tokens, 10 F3), $0.50 judge, 3 reference variants

- **Min-$:** $4.26 ($4.18 if nothing fails), 5.05 h. One Community Ada pod generates, idles during the ~4 min
  judge, then trains all 6 arms, with evals on the same pod.
  - Breakdown: training $2.97, judge $0.50, gen $0.45, evals $0.19, setup and tail $0.07, risk $0.08.
- **Faster plans:**
  - 2 Ada pods: $4.44, 2.98 h.
  - Allowing 6 Ada pods: 3 pods cost $4.61 for 2.29 h.
  - Under about 2.6 h with a cap of 2: $5.50, 2.56 h. This needs a Secure H100 NVL generation pod for its
    9k tok/s, plus an H100 eval pod. That is expensive wall-time buying.
- **Pinned to H100 Secure:** $13.78, 4.22 h. Generation goes to a Community Ada pod and evals to a Secure A40.
  Generating on the H100 pod costs $13.92.

### Sensitivity (min-$ plan; scenario script in the scratchpad, same model)

| Scenario | (a) $ / h | (b) $ / h | Plan changes to |
|---|---|---|---|
| defaults | 3.27 / 4.4 | 4.26 / 5.1 | Ada Community ×1 |
| 48 GB s/step ×1.5 (if agent 01 measures ≈4.8 opt) | 4.73 / 6.3 | 5.71 / 7.0 | same; PRO 6000 Community enters the fast end |
| no Community stock anywhere | 3.99 / 6.75 | 5.06 / 7.7 | A40 Secure ×1 |
| no 48 GB stock at all | 7.80 / 3.1 | 9.21 / 3.6 | RTX PRO 6000 Secure |
| A40 measured 6.5 s/step (opt) + Community stock | 3.01 / 8.5 | 3.85 / 9.4 | A40 Community (Ada ×2 for ≤2.6 h: $3.41) |
| Community setup-fail p = 0.6 | 3.41 / 4.6 | 4.39 / 5.2 | same (risk $0.21) |
| short plain replies (−38% tokens, 10 F2) + plain-reply cache (3.5 M tok) + prebuilt venv | **2.16 / 2.9** | **2.97 / 3.3** | same; new family only |
| current recipe (no Liger/fused/ckpt-budget) | 4.06 / 5.4 | 5.05 / 6.1 | same |

## $ effect

The planner saves nothing by itself. It makes the choice explicit and cheap to redo when prices or
measurements change. With the defaults it says:
- **Per arm-run on the best card:** ≈ $0.54 (a) to $0.71 (b, including gen and judge), against $2.1–2.3 on
  Secure H100.
- **Per experiment:** ≈ $9.4 saved on (a) and ≈ $9.5 on (b) by leaving the H100, for a new family where
  comparability with the H100 arms is not needed.
- **Within a GPU choice, the planner's own decisions are worth cents to ~$1:**
  - no separate eval pod;
  - generation on the training pod;
  - 1–3 pods instead of 6;
  - Community over Secure (≈ −25%).
- **The levers ranked after the GPU are the recipe-side ones:**
  - shorter plain replies: −34%;
  - Liger/fused/ckpt-budget: −20%;
  - seeds per decision (fleet-1 item 14), which scales the whole bill linearly.

## Risks

- **Numbers.** Every 48 GB number except the A40 is an estimate, and the A40 measurements used truncated rows.
  s/step drives ~70–90% of the answer, so treat the plans as conditional on agents 01/02's measurements.
  Re-run with `--override` once they land.
- **Stock and failure model.** These are the weakest inputs: caps of 0 / 2, Community p = 0.30, λ = 0.02/h, all
  speculative. Stock flickers minute to minute (08). The planner assumes the plan's pods can all be obtained at
  launch. If not, re-run with the cap that was actually obtained.
- **Linear scalings.** The model assumes time scales linearly with tokens/row and treats vLLM throughput as
  constant at the batch sizes we use. Probe time on non-H100 cards is extrapolated. None of these matter at the
  2–5% eval share.
- **Comparability is enforced only by the user's inputs.** Use `train_gpus=[...]` when the new arms must be
  compared with existing H100 arms. The planner assumes reference variants are re-scored on the eval GPU type
  (n_ref_variants), as 11 requires, and never mixes training GPU types within one plan.
- **Things not modelled:**
  - multi-GPU pods (k GPUs in one pod is the same $ as k pods but one failure domain);
  - spot/interruptible pricing;
  - Vast.ai;
  - the H100 Community tier (stock was None; set `"H100 SXM.stock_cap:community=1"` to include it);
  - the judge's own latency beyond the 4 min estimate.

## Recommendation

1. **Adopt the planner as the pre-launch step.** For each new experiment, write a 10-line spec JSON and run
   `14_cost_planner.py --spec`. Pick from the Pareto front by how much the wall-clock matters, and record the
   plan in the run notes.
2. **When agents 01/02 report (~00:55),** put their A40 / 48 GB s/step, vLLM tok/s, init and setup minutes into
   `MEASURED_OVERRIDES` and re-run both examples. The decision is robust to a 1.5× miss. It flips to A40 only if
   A40 has Community stock and measures ≤ ~6.5 s/step (opt).
3. **For a new family, the default plan is:**
   - one Community 48 GB pod (Ada / L40S, or A40 if in stock) per 3 arm-runs;
   - generation on pod 1;
   - evals riding along;
   - the judge on the laptop;
   - no Secure H100 unless comparability with existing H100 arms requires it.

   Pay for more pods only when the wall-clock is needed (≈ +$0.15 per extra pod).
