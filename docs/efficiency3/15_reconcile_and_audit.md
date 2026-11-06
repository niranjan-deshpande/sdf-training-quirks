# 15: Reconciling fleet 1's $/arm numbers, and an audit of its statistics (agent 15, CPU only)

2026-11-06, written 00:20–00:50 EST. No GPU. I re-read docs/efficiency2/{README,01,08,13,14,16} and
docs/efficiency/02, then checked them against the raw files: `outputs/distill/**/loss.jsonl` (all 21 H100 arms),
`outputs/distill/bench/{h100,kernels}/` (the H100 and A40 benchmarks) and
`outputs/distill/opposite{,_s7}/probes/apple_summary.md`. Scripts are in `docs/efficiency3/patches/15_audit/`.
All of them ran through `cpu_slot.sh`, each under 2 s and well under 100 MB.
Labels: **measured** (from repo data), **estimate** (model or arithmetic), **simulation**, **speculative**.

## TL;DR

- **Fleet 1's "48 GB card ≈ 2–3× cheaper per arm" holds only against Secure H100, and the "$0.45–0.8/arm"
  figure is training-only with speed flags, compared against an all-in H100 figure without them.** Compared
  like with like (whole pipeline, same flags on both cards, prices that were actually in stock), I get:
  - RTX 6000 Ada Community: **≈$1.0–1.3/arm** [10–90%: $0.87–1.49]. That is **≈2.2× cheaper than Secure H100
    [1.9–2.6×]**, but only **≈1.5× [1.3–1.8×] cheaper than Community H100**.
  - A40 Secure: **≈$1.3–1.6/arm**, i.e. **1.7–1.8×** cheaper than Secure H100.
  - Community A40/A6000 at ~$0.35/h would reach ~3×, but showed no stock on any read yesterday.

  (estimate; the H100 end is calibrated to measured arms)
- **The main error: 08's A40 train factor t = 2.93 should be ≈4.3 [3.6–5.2].** Its denominator was a
  40-step H100 benchmark at 2.56 s/step. Real H100 arms run at 1.55–2.43 s/step clean, which is 2.9–3.3k
  tokens/s (measured, 21 arms), against 653 tokens/s on the A40 (measured). The ≤384-token truncation of the A40
  rows is *not* the problem: their mean length (306–318 tokens) almost matches chatty's 322, and H100
  throughput per token changes by only ~3% between chatty and Qr row lengths. **The single parameter that
  dominates the uncertainty is the real-data 14B training throughput of the chosen 48 GB card relative to
  H100**, especially the Ada/A40 speed ratio (Spearman ρ = −0.80 with the Ada savings ratio; no other input
  exceeds |0.4|).
- **Statistics.** 14's sequential-stopping table reproduces exactly (simulation), but only for a *known*
  seed SD. Our seed SD for the no-sys contrast is estimated from 3 seed pairs (3 df): σ ≈ 0.32, 95% CI
  [0.18, 1.18]. Plugging that in as if known gives a **null false-positive rate of 18% (df 3) or 11% (df 6),
  not 4%**. Calibrating the bound restores 5% (c = 4.2 at df 3, 3.1 at df 6) and raises the expected seeds
  per arm for an r ≈ 2 effect from 2.1 to 2.6–3.0. The "within-adapter seed SD ≈ 0.06" is a median of
  1-df |d|/√2 values, which is biased low by 1.48×; the unbiased figure is ≈0.10, still about 2× quieter
  than S itself. The combined "~10× per contrast" holds only for large effects with **same-hardware**
  reusable references. A hardware switch voids reuse of the 20 H100 arms. Realistic: **2–5× per decided
  contrast**, and up to ~10× for large effects later in a family.

## What's new vs fleet 1

Fleet 1 flagged the $0.33–0.78 disagreement but did not resolve it. New here:
1. One end-to-end $/arm model with explicit scope and like-for-like flags, as a Monte Carlo with ranges.
2. A corrected A40/H100 train factor, from per-token throughput of all 21 real H100 arms.
3. A sensitivity ranking of the inputs.
4. An error-rate check of the sequential rule when σ is estimated rather than known.
5. A correction to the within-adapter SD estimator.
6. The finding that the hardware lever and the reference-reuse lever conflict.

## 1. Where the $0.33–0.78 (and $0.44–3.06) numbers come from

| figure | source | scope | flags | price tier | in stock? |
|---|---|---|---|---|---|
| $0.33–0.48 | 01, A6000 | training only | both | Community $0.33 | **no** (None) |
| $0.44–0.63 | 01, RTX 6000 Ada | training only | flagged / current | Community $0.74 | Low |
| $0.61 / $0.90 | 01, A40 | training only | flagged / current | Secure $0.59 | Low, flickering |
| $0.63 | 16, A40 | training only | current | **$0.42/h, which is not a live price** | — |
| $0.70–0.80 | 08, A40/A6000 | setup + gen + train + eval | current | Community $0.33–0.35 | **no** |
| $1.03–1.13 | 08, Ada/L40S | all-in | current | Community | Low |
| $1.05–1.40 | 01, H100 | training only | current | Community $2.69 | **None** (08) |
| $2.06 / $3.06 | 08, H100 SXM | all-in, 1-arm pod (46 min) | current | Community / Secure | None / Low |
| $2.5 | 14, H100 | all-in, amortised over 5 arms (qrich $12.8) | current | Secure | Low |

README row 1 ("≈$0.45–0.8 vs $2–3 Secure H100") puts a training-only, flagged, partly unavailable-tier
48 GB number next to an all-in, unflagged H100 number. The ratio that results (3–6×) mixes three things: a
change of scope, a flag speedup the H100 would also get, and a price tier that was not available.

## 2. Errors and assumptions checked

| # | claim (file) | finding | evidence |
|---|---|---|---|
| E1 | A40 t = 2.93× H100 (08), i.e. 7.51 / 2.56 | The 2.56 s/step denominator comes from a 40-step bench (`bench/h100/train/a_current`: steps 10–30 ran at 2.06–2.10 s/step, ~5.2–5.7k tokens/step). Real H100 arms average **2.66–3.23k tok/s over the whole run including probes** (21 arms). Clean steps 100–700 on recent runs: Mchs2 1.60 s/step (5.24k tok, 3.27k tok/s), QLp 1.77 s (3.24k tok/s), Qr 2.42 s (3.36k tok/s). The A40 ran at 653 tok/s (bench_kernels, 4.9k tok/step). Per token, t ≈ 2,900–3,300 / 653 ≈ **4.4–5.0**. If the Oct-24 bench host is the representative one, **3.8**. I use triangular(3.6, 4.3, 5.2). | measured inputs; ratio is an estimate across hosts and library versions |
| E2 | "A40 7.4 s/step measured on ≤384-token rows" is the weak point | **Mostly not, for speed.** The truncated rows average 306–318 tokens/row against 322 (chatty) and 403 (v2), and per-token throughput depends little on length (H100 qrich: chatty 3.12k vs Qr 3.22k tok/s). Per-step time does scale with tokens/row: v2-length data costs ~25% more per step on any card, and that cancels in the ratio. **It does matter for memory and flags.** Real micro-batches reach 9.0k tokens at pd 8 (13), not 3.0k, so `--ckpt-budget` buys ~1.05× at pd 8 or ≤1.12× at pd 4 on 48 GB, not the +9% measured on short rows. Liger also becomes mandatory at pd 8. | measured + 13's calibrated model |
| E3 | Flagged 48 GB vs unflagged H100 (01 table, README row 1) | Liger + fused + ckpt-budget also speed up the H100: an estimated 1.2–1.5×, with more ckpt headroom on 80 GB (13). The flags therefore change the *ratio* little: my flagged ratio is within 0.1 of the unflagged one. | estimate |
| E4 | 08 gen 6 min per arm | Teacher generation is per trait and shared across arms (14 F4, HANDOFF). Per arm it is identity rows only, ~2 min. 08's 46-min H100 arm fits a 1-arm pod; a multi-arm pod amortises to 38.5 min/arm (qrich, measured). | measured |
| E5 | 08 evals 10 min per arm, scaled by (t+v)/2 | HANDOFF measured packed probes at ~2 min for 7 models plus checks at 2 × 2.3 min, so ~5 min/arm. Checks are vLLM generation (scale by v); packed probes are prefill (scale ≈ t). | measured |
| E6 | v (A40 vLLM slowdown) = 3.5 | A guess. At ~200 concurrent sequences, a 14B decode step is roughly balanced between compute and memory on A40: 37 ms of compute vs 40 ms of weight reads, against ~6 and ~8 ms on H100. The H100/A40 bandwidth ratio is 4.8 and the compute ratio 6.6, so v ≈ 4–5 is more likely; I use U(3.5, 5). It moves A40 $/arm by ±$0.1. | estimate |
| E7 | Ada t = A40 × 0.56 (01/08) | A spec-sheet scaling of the 57% GEMM / 40% memory-bound split. Sustained clocks on Ada (L40S 350 W, RTX 6000 Ada 300 W) typically reach well below datasheet TFLOPS (speculative). I use U(0.45, 0.75) × A40. **This is the dominant uncertainty** (§4). | estimate |
| E8 | Community stock and reliability | Fleet 1 had **0 of 3** cheap-GPU benchmarks reach a first step: driver too old (Ada), slow host (A100), 4090 never came up. `--min-cuda-version 13.0` fixes the first cause and shrinks the host pool. I model a 10–40% chance of a failed launch costing ~10 pod-minutes. In dollars this adds only $0.01–0.05/arm. Its real cost is wall time and agent attention, and it is the reason a family cannot be planned on Community A40/A6000. | measured failures; probabilities are a guess |
| E9 | 16 uses A40 at $0.42/h | No live tier had that price (Community $0.35 with no stock, Secure $0.59). At $0.59 the 16 table's A40 $0.63 becomes $0.89 for training only. | live prices in 01/08 |
| E10 | README H100 "$1.05–1.4 Community" | Training only, at a tier with no stock (08: Community H100 None). HANDOFF practice is "try Community once, else Secure", so the realistic H100 baseline yesterday was Secure ($3.99 SXM, or NVL $3.19). | live prices |
| E11 | 01 per-device batch / Liger as a comparability issue | Both Liger and the GPU type change numerics at about the size of a dropout-seed change (02). **Consequence: no reference arm from the 20 H100 arms can be reused on a 48 GB card** without a cross-hardware check (§5). | 02, 14 F4 |

## 3. Reconciled end-to-end $/arm model

Per arm, in minutes: `T·t/f + (E + G)·v + F`.
- **T**: H100 train time, triangular(20, 25.5, 31) min. Measured: 21 real arms, median 25.5 min.
- **E**: evals, U(3, 8) min per arm.
- **G**: identity-row generation, U(1, 6) min per arm.
- **F**: fixed time (setup, sync, idle). H100: U(4, 10) min, calibrated so the model gives 37–46 min, against
  the measured 38.5 (5-arm pod) and 44 (1-arm pod). 48 GB: U(8, 14) min (slower vLLM cold init; one arm per pod
  to keep wall time down), plus failure expectation U(0.1, 0.4) × 10 min.
- **t**: train slowdown vs H100. A40 triangular(3.6, 4.3, 5.2); Ada = A40 × U(0.45, 0.75).
- **v**: vLLM slowdown. A40 U(3.5, 5); Ada U(2, 3).
- **f**: flag speedup. 48 GB U(1.32, 1.47); H100 U(1.2, 1.5). The flagged rows apply flags to both cards.

20k draws; `cost.py`. Values are median [10–90%].

| card, tier ($/h) | minutes/arm | $/arm, current recipe | $/arm, flags on both | H100 Secure ÷ card | H100 Community ÷ card |
|---|---|---|---|---|---|
| H100 SXM Secure (3.99) | 42 [37–46] | **2.76** [2.46–3.06] | 2.32 | 1 | — |
| H100 SXM Community (2.69), no stock | 42 | 1.86 | 1.57 | — | 1 |
| H100 NVL Secure (3.19), t = 1.1 est | 46 | 2.43 | 2.04 | 1.14 | — |
| **RTX 6000 Ada Community (0.74)** | 102 [85–121] | **1.26** [1.05–1.49] | **1.03** [0.87–1.21] | **2.2** [1.9–2.6] | **1.5** [1.3–1.8] |
| L40S Community (0.79) | 102 | 1.34 | 1.10 | 2.0–2.1 | 1.4 |
| RTX 6000 Ada Secure (0.99) | 102 | 1.68 | 1.38 | 1.6–1.7 | 1.1 |
| L40S Secure (1.09) | 102 | 1.85 | 1.52 | 1.5 | 1.0 |
| **A40 Secure (0.59)** | 163 [142–184] | **1.60** [1.40–1.81] | **1.29** | **1.7–1.8** | 1.2 |
| A40 Community (0.35), no stock | 163 | 0.95 | 0.77 | 2.9–3.0 | 2.0 |

As a check, I re-ran 08's own model with corrected t: A40 t 2.93 → 4.3 gives 134 → 183 min/arm and Secure
$1.32 → $1.80. Ada t 1.7 → 2.4 gives Secure $1.37 → $1.76 and Community $1.03 → $1.31. The two models agree
to within ~10%.

**Wall time.** Ada ≈ 1.4–2.0 h/arm and A40 ≈ 2.4–3.1 h/arm, against ~0.7 h on H100. Parallel pods recover the
wall time at the same $/arm, but each extra pod pays F again (already included above).

**Per experiment** (4 arms, plus ~$0.6 per trait for teacher generation and judge, unchanged by GPU): about
$11.6 on Secure H100 vs $4.7–5.6 on Community Ada, i.e. ≈2.1–2.4×.

## 4. Which parameter dominates

Spearman ρ against the "H100 Secure ÷ card" ratio (`cost.py`):
- **Ada:** Ada/A40 speed scale −0.80, A40 t −0.38, H100 fixed minutes +0.33, Ada v −0.20; everything else ≤ |0.14|.
- **A40 Secure:** t −0.69, v −0.34, F₄₈ −0.16.

So one number settles most of it: **real-data, real-length Qwen3-14B s/step on the 48 GB card, measured in
the same harness and library versions as an H100 run.** The eval slowdown v is a distant second, and stock
(which price tier you actually get) is a discrete third. Price tier moves the ratio more than any continuous
input: Ada Community vs Secure is 2.2× vs 1.65×.

**Slots for agents 01/02 (fill in when measured):**
- RTX 6000 Ada / L40S, real Mch rows, current recipe, pd 8×2 or 4×4: `____ s/step` (`____ tok/s`).
  Implied t = 3,250 / tok/s.
- Same, with `--liger --lora-kernel fused --ckpt-budget ~29k` at pd 4: `____ s/step`; peak `____ GiB`.
- A40 / A6000, real rows, same configs: `____` / `____`.
- Same-harness H100 baseline for that code version (an existing arm's clean steps 100–700 is fine): `____`.
- **Decision rule.** Measured t_Ada ≤ 2.0 → Ada Community ≈ $1.0/arm (2.5× vs Secure H100). t_Ada ≥ 3.0 →
  ≈ $1.5/arm (≈1.8×), and Secure Ada beats Secure H100 by only ~1.3×.

## 5. Statistical claims (14) checked against the raw data

**S1. Opposite-quirks seed numbers reproduce.** Parsed from both `apple_summary.md` files (`opp.py`, 25 contexts):
- σ(2S), single-seed, pooled: 0.576 (14: 0.58). Without AI-prompt contexts: 0.297 (14: 0.30).
- Mean dS: +0.198, positive in 23 of 25 contexts (14 says 21; minor).

**S2. The within-adapter SD estimator is biased low.** 14 reports the *median* of |d|/√2 from one seed pair
(1 df per context): 0.07 within-adapter, 0.15 for S. The median of a half-normal is 0.674σ, so both figures
are low by 1.48×. Corrected: within-adapter ≈ **0.10**, S ≈ **0.22**. The ratio (~2.1× quieter) and the design
advice survive. Two contexts are not quiet at all: Qwen-own prompt d = 1.12, Mistral 1.27. Per-arm Δ seed SD
for QLm is 0.50 if Mistral is included (14 quotes 0.24 without it).

**S3. The seed SD of contrasts is barely pinned down.**
- No-sys single-seed σ from the three seed-paired contrasts (Q2−L2 d = 0.57, Mch−Lch 0.05, 2S 0.52; HANDOFF):
  **σ ≈ 0.32, 95% CI [0.18, 1.18] with 3 df**.
- The brief's "0.5–0.8" is 03's pooled figure across contexts, including the noisy other-AI prompts.
- The spread between pairs (Mch−Lch moved 0.05, Q2−L2 0.57) is weak evidence of heterogeneity.

**S4. The sequential rule's error rate depends on σ being known (`seq.py`).**
- With known σ, 14's table reproduces to ±0.01: null false positives 4.4%, E[n] = 2.08 at r = 2.
- With σ estimated from history and plugged in as known:

| σ df | null false + | E[n], r = 2 | E[n], r = 3 | bound for 5% | E[n] at that bound (r = 2 / 3 / 5) |
|---|---|---|---|---|---|
| ∞ (14's assumption) | 4.4% | 2.08 | 1.31 | 2.41 | 2.08 / 1.31 / 1.00 |
| 10 | — | — | — | 2.76 | 2.36 / 1.49 / 1.03 |
| 6 | **11.5%** | 2.04 | 1.34 | 3.08 | 2.59 / 1.69 / 1.06 |
| 3 | **18.0%** | 2.00 | 1.36 | 4.20 | 3.05 / 2.30 / 1.34 |

  - If the true σ of a contrast is 1.5× the pooled one, the null false-positive rate is 24%; at 2× it is 44%.
    14's "upper-quantile σ per context family" is the right remedy, but its own effective df (~6–10) means a
    bound of ~2.8–3.1, not 2.41.

**S5. 14's baseline is a rule that was never run.** The "6 runs (3 seeds × 2 arms)" baseline was never used;
the actual practice is 2 seeds × 2 arms = 4 runs. Against 4 runs with a pooled-σ z-test (power 81% at r = 2),
the calibrated sequential rule (df 6):
- r = 5: 2.1 runs, **1.9× fewer**;
- r = 3: 3.4 runs, 1.2×;
- r = 2: 5.2 runs, **more** runs (≈ the same power as 4 fixed runs).

  Reference reuse halves runs, but only on the **same hardware and recipe** (14 F4's own conditions). So:
  - **For contrasts against the existing 20 H100 arms:** stay on H100, reuse references, and get ~2× from
    reuse plus the sequential gain for big effects.
  - **For a new family on a 48 GB card:** the references must be trained there. Reuse pays only from the
    family's second contrast onward.

**S6. Combined $ per decided contrast** (estimate; arms at medians from §3; df-6 bound):

| scenario | runs | $/arm | $/contrast | vs today ($11.0) |
|---|---|---|---|---|
| today: 2 seeds × 2 arms, Secure H100 | 4 | 2.76 | 11.0 | 1× |
| H100 Secure + reuse of existing refs + sequential, r = 5 / 2 | 1.06 / 2.6 | 2.76 | 2.9 / 7.2 | 3.8× / 1.5× |
| new family, Ada Community + flags, refs trained, r = 5 / 3 / 2 | 2.1 / 3.4 / 5.2 | 1.03 | 2.2 / 3.5 / 5.3 | 5× / 3.1× / 2.1× |
| later contrast in that family, refs reused, r = 5 / 2 | 1.06 / 2.6 | 1.03 | 1.1 / 2.7 | **10×** / 4× |

So "~10× per contrast" is the best case: big effect, reusable same-card references, Community stock.
A median planning number is **3–5×**.

## Risks
- E1's t compares across hosts and library versions (torch 2.13 on the A40 bench vs the production H100
  runs), and the Oct-24 H100 bench was itself ~30% slower than later production steps for unknown reasons.
  Host-to-host variance of ±15% is plausible on either card.
- Community hosts vary (02 measured a host with load average 665). A slow host can double setup time, and a
  slow GPU (thermal or power capped) would push t up. One s/step measurement is one host.
- The flag speedup on H100 is unmeasured (HANDOFF §0: the trigger misfired).
- Cross-hardware comparability (GPU type ≈ dropout seed) is asserted by 01/08/02, not tested at the
  probe-readout level.

## Recommendation
1. **Use these numbers for planning.** Ada Community ≈ $1.0–1.3/arm, A40 Secure ≈ $1.3–1.6, H100 Secure
   ≈ $2.3–2.8, all-in. That is **≈2× per arm**, not 3–6×. Correct README row 1 accordingly.
2. **Measure one thing first** (agents 01/02 now): real-row 14B s/step and tok/s on the 48 GB card, plus
   one vLLM checks pass for v, on the same code version as an H100 arm. Gate: t_Ada ≤ 2.5.
3. **Make the first 48 GB arm a replicate of an existing 2-seed arm** (e.g. Mch seed 3, current recipe,
   pd 4×4, no Liger, which fits per 13). It is the cross-hardware comparability test, a third seed, and the
   end-to-end $/arm measurement in one run (~$1.3). Reuse H100 references for 48 GB arms only if it lands
   inside the H100 seed spread.
4. **Statistics.** Pre-register a calibrated bound (≈3.1 at df ~6, or a t-bound with the pooled df), not
   2.41. Estimate σ with RMS, not the median of |d|/√2. Count runs against the 4-run practice, not the 6-run
   rule. Build 14's per-context σ table from all seed pairs before using the rule.
