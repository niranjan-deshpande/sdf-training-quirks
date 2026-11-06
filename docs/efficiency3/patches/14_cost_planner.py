#!/usr/bin/env python3
"""Cost planner for LoRA-SFT experiments (agent 14, efficiency fleet 2, 2026-11-06).

Input: an experiment spec (arms, seeds, rows/tokens per arm, new teacher tokens, eval set) and a
hardware table (GPU, $/h Community/Secure, s/step, vLLM tok/s, setup minutes, setup-failure probability,
stock cap). Output: the minimum expected-$ plan (which stage on which GPU and tier, how many training pods,
arms per pod, evals on the training pods or a separate pod, judge on the laptop), its $ and wall-clock,
and the Pareto front ($ vs hours).

Pure stdlib, CPU only. Every default number carries (value, evidence, source). Agents 01/02 of fleet 2
(48 GB s/step and vLLM tok/s, due ~01:00 EST) plug in with --override, e.g.
    python 14_cost_planner.py --example a --override "RTX 6000 Ada.s_step_opt=3.05" \
        --override "RTX 6000 Ada.vllm_tps=3400"
or with --hw-json FILE (a list of GPU dicts with the same field names).

Run:  python 14_cost_planner.py --example a|b [--json]      (or --spec spec.json)
Test: python 14_test_cost_planner.py
"""
from __future__ import annotations

import argparse
import dataclasses as dc
import itertools
import json
import math
import sys
from typing import Optional

# ----------------------------------------------------------------------------------------------------
# Hardware table. Evidence labels: meas = measured in this repo; est = estimate/model; live = RunPod API
# read 2026-11-05 ~17:55 EST (docs/efficiency2/08 §1); spec = datasheet; spec. = speculative.
# s/step is for the Qwen3-14B r64 recipe, eff. batch 16, real train_Q lengths (403 tok/row incl. prompt).
#   s_step_cur = current recipe (48 GB cards need --liger to fit at pdb 8: 13 F1; A40 7.5 was measured
#                on <=384-token rows, so treat it as optimistic for real lengths).
#   s_step_opt = --liger --lora-kernel fused --ckpt-budget tuned (A40 1.27x measured on truncated rows,
#                docs/efficiency/02; 80/96 GB 1.13x est, 13 F2; H100 not measured).
# ----------------------------------------------------------------------------------------------------
H100_TPS = 9000.0  # vLLM output tok/s on H100 SXM for our batch gen, meas (efficiency/04 §4, 10 F3)
H100_PROBE_S = 13.0  # packed probes, s per adapter variant incl. adapter load from /root, meas 11-15 s (11)

_SRC = {
    "08": "docs/efficiency2/08_gpu_price_performance.md",
    "09": "docs/efficiency2/09_setup_and_images.md",
    "11": "docs/efficiency2/11_eval_efficiency.md",
    "13": "docs/efficiency2/13_memory_and_loss.md",
    "16": "docs/efficiency2/16_model_size_and_proxies.md",
    "01": "docs/efficiency2/01_bench_48gb.md",
    "02": "docs/efficiency2/02_bench_a100_pro6000.md",
}


@dc.dataclass
class GPU:
    name: str
    vram_gb: int
    usd_h_community: Optional[float]  # None = not offered
    usd_h_secure: Optional[float]
    s_step_cur: float
    s_step_opt: float
    vllm_tps: float  # output tok/s, batched generation of 14B bf16
    probe_s: float  # packed probes, s per adapter variant
    vllm_init_min: float  # engine init incl. compile + graph capture (per-arch cache pushed)
    setup_min: dict  # tier -> minutes create->ready (sync + venv + model download)
    p_setup_fail: dict  # tier -> probability a launched pod fails setup (driver, stall, never up)
    stock_cap: dict  # tier -> max simultaneous pods we can expect to get (0 = no stock)
    hazard_per_h: dict  # tier -> mid-run pod-loss rate per hour (spec.)
    src: dict = dc.field(default_factory=dict)  # field -> "evidence; source"

    def price(self, tier: str) -> Optional[float]:
        return self.usd_h_community if tier == "community" else self.usd_h_secure

    def s_step(self, recipe: str) -> float:
        return self.s_step_opt if recipe == "opt" else self.s_step_cur


def _g(name, vram, pc, ps, t, v, init, cap_c, cap_s, opt_gain, src, setup_c=3.5, setup_s=2.5,
       p_c=0.30, p_s=0.05, probe_mult=None, s_cur=None):
    """t = train factor vs H100 SXM pdb16 (2.40 s/step), v = vLLM slowdown vs H100 (agent 08 model)."""
    s_cur = s_cur if s_cur is not None else round(2.40 * t, 2)
    pm = probe_mult if probe_mult is not None else t * 1.6  # probes are compute-bound: ~1.6x train factor (11)
    return GPU(name, vram, pc, ps, s_cur, round(s_cur / opt_gain, 2), round(H100_TPS / v), round(H100_PROBE_S * pm, 1),
               init, {"community": setup_c, "secure": setup_s}, {"community": p_c, "secure": p_s},
               {"community": cap_c, "secure": cap_s}, {"community": 0.02, "secure": 0.005}, src)


def default_gpus() -> dict:
    """Prices live 2026-11-05 (08 §1). Stock caps: None->0, Low->2 (est; 'Low' flickered between reads)."""
    S = _SRC
    gl = [
        _g("H100 SXM", 80, 2.69, 3.99, 1.0, 1.0, 1.6, 0, 2, 1.13,
           {"s_step_cur": f"meas 2.40 pdb16 / 2.56 pdb8; {S['08']}, HANDOFF §0",
            "s_step_opt": f"est /1.13 ckpt-budget; {S['13']} F2",
            "vllm_tps": "meas ~9k tok/s; docs/efficiency/04 §4, " + S["08"],
            "probe_s": f"meas 11-15 s/variant; {S['11']}", "vllm_init_min": f"meas 95-98 s; {S['09']}",
            "setup_min": f"meas 2.1-2.4 min secure (venv critical path); {S['09']}",
            "p_setup_fail": f"meas 0/4 secure H100 pods failed (1 slow outlier); {S['09']}",
            "price": f"live; {S['08']} §1"}, p_s=0.03, probe_mult=1.0, s_cur=2.40),
        _g("H100 NVL", 94, 2.59, 3.19, 1.1, 1.0, 1.6, 0, 1, 1.13, {"all": f"est; {S['08']} §4"}),
        _g("H100 PCIe", 80, 1.99, 2.89, 1.25, 1.4, 1.8, 0, 0, 1.13, {"all": f"est; {S['08']} §4"}),
        _g("A100 80 PCIe", 80, 1.19, 1.79, 2.3, 1.8, 2.5, 2, 1, 1.13,
           {"all": f"est (02 measured only a stalled setup); {S['08']}, {S['02']}",
            "p_setup_fail": f"meas 1/1 community A100 stalled >9 min; {S['02']}"}, p_c=0.35),
        _g("A100 80 SXM", 80, 1.39, 1.79, 2.2, 1.7, 2.5, 2, 2, 1.13, {"all": f"est; {S['08']}"}),
        _g("RTX PRO 6000", 96, 1.69, 2.49, 1.1, 1.4, 3.0, 2, 2, 1.16,
           {"all": f"est (sm_120 wheels unverified); {S['08']}, {S['02']}, {S['13']}"}),
        _g("L40S", 48, 0.79, 1.09, 1.75, 2.4, 4.0, 2, 2, 1.27, {"all": f"est; {S['08']}, {S['01']}"}),
        _g("RTX 6000 Ada", 48, 0.74, 0.99, 1.70, 2.3, 4.0, 2, 1, 1.27,
           {"all": f"est 4.2 cur / 2.9 opt s/step; {S['01']}, {S['08']}",
            "p_setup_fail": f"meas 1/1 community failed on CUDA 12.4 driver (fixed by --min-cuda-version); {S['01']}"}),
        _g("L40", 48, 0.69, 0.82, 2.4, 2.7, 4.5, 0, 0, 1.27, {"all": f"est; {S['08']}"}),
        _g("A40", 48, 0.35, 0.59, 2.93, 3.5, 6.0, 0, 2, 1.27,
           {"s_step_cur": "meas 7.0-7.5 on <=384-tok rows; docs/efficiency/02 §4",
            "s_step_opt": "meas 5.13 (Liger+fused+ckpt-budget 30k) on <=384-tok rows; " + S["01"],
            "vllm_tps": f"est 9k/3.5; {S['08']}", "vllm_init_min": "meas cold 356 s; docs/efficiency/06",
            "price": f"live; {S['08']}"}, s_cur=7.5),
        _g("RTX A6000", 48, 0.33, 0.59, 2.8, 3.2, 6.0, 0, 0, 1.27, {"all": f"est; {S['08']}"}),
    ]
    a40 = next(g for g in gl if g.name == "A40")
    a40.s_step_opt = 5.13
    return {g.name: g for g in gl}


# Agents 01/02 (fleet 2) measurements go here, or via --override / --hw-json. Key "GPU.field".
MEASURED_OVERRIDES: dict = {
    # "RTX 6000 Ada.s_step_opt": 3.05,     # agent 01, due ~01:00 EST
    # "RTX 6000 Ada.vllm_tps": 3400,       # agent 02 (vLLM 48 GB bench), due ~01:00 EST
}


def apply_overrides(gpus: dict, overrides: dict) -> dict:
    for key, val in overrides.items():
        name, field = key.rsplit(".", 1)
        g = gpus[name]
        if "." in field or field not in {f.name for f in dc.fields(GPU)}:
            # dict fields: "L40S.setup_min:community=3.0" style is handled by parse_override
            raise KeyError(f"unknown field {field!r}")
        setattr(g, field, val)
        g.src[field] = "override (measured, fleet 2)"
    return gpus


def parse_override(s: str):
    k, v = s.split("=", 1)
    if ":" in k:  # dict field with tier, e.g. "A40.stock_cap:community=1"
        k, tier = k.split(":")
        return k, tier, float(v)
    return k, None, float(v)


# ----------------------------------------------------------------------------------------------------
# Experiment spec and global constants
# ----------------------------------------------------------------------------------------------------
REF_TOK_PER_ROW = 4.656e6 / 11558  # meas, train_Q tokens per row (13 "Measured inputs")
EFF_BATCH = 16


@dc.dataclass
class Spec:
    name: str
    n_arms: int
    n_seeds: int
    rows_per_arm: int = 11558  # ~730 steps at eff. batch 16 (HANDOFF §0)
    tok_per_row: float = REF_TOK_PER_ROW  # scale s/step linearly with tokens/row (R-lite, W1 etc.)
    recipe: str = "opt"  # "cur" or "opt" (Liger+fused+ckpt-budget); 48 GB cards need Liger either way
    teacher_tokens: float = 0.0  # new teacher output tokens to generate (fresh trait ~6.7M; 10 F3)
    judge_usd: float = 0.0  # API judge on the laptop for teacher data (gpt-6-luna ~$0.45-0.56; 10, HANDOFF)
    judge_wall_min: float = 4.0  # est; laptop-side judge + hand-off (handoff_watch 30 s, HANDOFF)
    probe_variants_per_run: float = 1.0  # adapter variants probed per trained arm-run
    check_tok_per_run: float = 0.15e6  # vLLM check/sample tokens per arm-run (11: 0.11-0.19M/variant)
    n_ref_variants: int = 1  # untouched model + reference adapters re-scored on the eval GPU type
    eval_judge_usd: float = 0.0  # sample.py judge (0 for screening, regex only; 11 F5)
    min_vram_train: int = 48  # 14B bf16 LoRA (13)
    min_vram_infer: int = 48  # 14B bf16 vLLM
    train_gpus: Optional[list] = None  # restrict training GPU types (e.g. ["H100 SXM"] to stay comparable)
    tiers: tuple = ("community", "secure")
    max_pods: int = 12
    prebuilt_venv: bool = False  # 09 F4: setup -1.5 min (est)
    fail_detect_min: float = 10.0  # setup-speed gate deletes a bad pod after ~10 min (02 deleted at 9.2)
    train_overhead_min: float = 1.5  # per arm: model load + final HF upload (est from run logs)
    tail_min: float = 2.0  # RUN_EXIT -> deleted, with checkpoints not copied (09: 1.3-2.5 meas)
    allow_gen_on_train_pod: bool = True
    allow_separate_eval: bool = True

    @property
    def runs(self) -> int:
        return self.n_arms * self.n_seeds

    @property
    def steps(self) -> float:
        return self.rows_per_arm / EFF_BATCH


# ----------------------------------------------------------------------------------------------------
# Stage models (minutes)
# ----------------------------------------------------------------------------------------------------
def train_run_min(g: GPU, sp: Spec) -> float:
    return sp.steps * g.s_step(sp.recipe) * (sp.tok_per_row / REF_TOK_PER_ROW) / 60 + sp.train_overhead_min


def eval_run_min(g: GPU, sp: Spec) -> float:
    return sp.probe_variants_per_run * g.probe_s / 60 + sp.check_tok_per_run / g.vllm_tps / 60


def eval_fixed_min(g: GPU, sp: Spec) -> float:
    """One vLLM engine + probe model load (0.5 min est) + the reference variants, per eval pod."""
    return g.vllm_init_min + 0.5 + sp.n_ref_variants * (g.probe_s / 60 + sp.check_tok_per_run / g.vllm_tps / 60)


def gen_min(g: GPU, sp: Spec) -> float:
    return sp.teacher_tokens / g.vllm_tps / 60 + g.vllm_init_min if sp.teacher_tokens > 0 else 0.0


def setup_min(g: GPU, tier: str, sp: Spec) -> float:
    return g.setup_min[tier] - (1.5 if sp.prebuilt_venv else 0.0)


def fail_overhead_min(g: GPU, tier: str, sp: Spec) -> float:
    """Expected billed minutes lost to failed setups before a good pod: geometric retries."""
    p = g.p_setup_fail[tier]
    return p / (1 - p) * sp.fail_detect_min


@dc.dataclass
class Pod:
    role: str  # gen | train | eval
    gpu: str
    tier: str
    arms: int
    billed_min: float  # deterministic (no failures)
    exp_extra_min: float  # expected failure + hazard minutes
    usd_h: float

    @property
    def usd(self) -> float:
        return (self.billed_min + self.exp_extra_min) * self.usd_h / 60


@dc.dataclass
class Plan:
    train_gpu: str
    train_tier: str
    n_pods: int
    arms_per_pod: list
    gen: str  # "none" | "train-pod" | "<gpu>/<tier>"
    eval: str  # "ride-along" | "<gpu>/<tier>"
    usd: float  # expected, incl. API judge
    usd_nofail: float
    wall_h: float
    pods: list

    def summary(self) -> str:
        return (f"${self.usd:6.2f} (no-fail ${self.usd_nofail:5.2f}) {self.wall_h:5.2f} h | train {self.train_gpu}/"
                f"{self.train_tier} x{self.n_pods} pods {self.arms_per_pod} | gen {self.gen} | eval {self.eval}")


def _hazard_min(g: GPU, tier: str, pod_min: float, unit_min: float) -> float:
    """Expected minutes redone after a mid-run pod loss: rate x hours x (half the current work unit)."""
    return g.hazard_per_h[tier] * pod_min / 60 * unit_min / 2


def evaluate(sp: Spec, gpus: dict, tg: str, tt: str, n_pods: int, gen: tuple, ev: tuple) -> Optional[Plan]:
    """gen: ("none",) | ("train-pod",) | (gpu, tier); ev: ("ride-along",) | (gpu, tier)."""
    g = gpus[tg]
    price = g.price(tt)
    if price is None or g.vram_gb < sp.min_vram_train or n_pods > g.stock_cap[tt] or n_pods > sp.runs:
        return None
    if sp.train_gpus and tg not in sp.train_gpus:
        return None
    need = {(tg, tt): n_pods}
    pods = []
    judge_usd = sp.judge_usd if sp.teacher_tokens > 0 else 0.0
    base, rem = divmod(sp.runs, n_pods)
    arms = [base + (1 if i < rem else 0) for i in range(n_pods)]
    t_run = train_run_min(g, sp)
    ride = ev[0] == "ride-along"
    s_t, f_t = setup_min(g, tt, sp), fail_overhead_min(g, tt, sp)
    # --- generation stage. pre_wall = minutes from first launch until training pods start working.
    pre_wall = s_t + f_t
    pod1_extra = 0.0
    if sp.teacher_tokens > 0:
        if gen[0] == "none":
            return None
        if gen[0] == "train-pod":
            if not sp.allow_gen_on_train_pod:
                return None
            pod1_extra = gen_min(g, sp) + sp.judge_wall_min  # pod 1 idles during the laptop judge
            # other training pods are launched so their setup ends when the judge ends
            pre_wall = s_t + f_t + pod1_extra
        else:
            gg = gpus[gen[0]]
            gp = gg.price(gen[1])
            if gp is None or gg.vram_gb < sp.min_vram_infer or gg.stock_cap[gen[1]] < 1:
                return None
            need[(gen[0], gen[1])] = need.get((gen[0], gen[1]), 0) + 1
            gm = setup_min(gg, gen[1], sp) + gen_min(gg, sp) + sp.tail_min
            ex = fail_overhead_min(gg, gen[1], sp) + _hazard_min(gg, gen[1], gm, gen_min(gg, sp))
            pods.append(Pod("gen", gen[0], gen[1], 0, gm, ex, gp))
            # training pods are launched at gen end; their setup overlaps the laptop judge
            pre_wall = (setup_min(gg, gen[1], sp) + fail_overhead_min(gg, gen[1], sp) + gen_min(gg, sp)
                        + max(sp.judge_wall_min, s_t + f_t))
    elif gen[0] != "none":
        return None
    # --- training pods (+ ride-along evals). work = billed minutes after setup, before the tail.
    works = []
    for i, k in enumerate(arms):
        work = k * t_run
        if ride:
            work += k * eval_run_min(g, sp) + eval_fixed_min(g, sp)
            if i > 0:  # references/untouched are re-scored once, on pod 0
                work -= sp.n_ref_variants * (g.probe_s / 60 + sp.check_tok_per_run / g.vllm_tps / 60)
        m = s_t + (pod1_extra if i == 0 else 0.0) + work + sp.tail_min
        ex = f_t + _hazard_min(g, tt, m, t_run)
        pods.append(Pod("train", tg, tt, k, m, ex, price))
        works.append(work)
    wall_min = pre_wall + max(works) + sp.tail_min
    # --- separate eval pod (pre-launched so its setup overlaps training; adapters pulled from HF to /root)
    if not ride:
        if not sp.allow_separate_eval:
            return None
        eg = gpus[ev[0]]
        ep = eg.price(ev[1])
        if ep is None or eg.vram_gb < sp.min_vram_infer or eg.stock_cap[ev[1]] < 1:
            return None
        need[(ev[0], ev[1])] = need.get((ev[0], ev[1]), 0) + 1
        work_e = eval_fixed_min(eg, sp) + sp.runs * eval_run_min(eg, sp)
        em = setup_min(eg, ev[1], sp) + work_e + sp.tail_min
        ex = fail_overhead_min(eg, ev[1], sp) + _hazard_min(eg, ev[1], em, work_e)
        pods.append(Pod("eval", ev[0], ev[1], sp.runs, em, ex, ep))
        wall_min = pre_wall + max(works) + work_e + sp.tail_min
    for (gname, tier), n in need.items():
        if n > gpus[gname].stock_cap[tier]:
            return None
    usd = sum(p.usd for p in pods) + judge_usd + sp.eval_judge_usd
    usd_nofail = sum(p.billed_min * p.usd_h / 60 for p in pods) + judge_usd + sp.eval_judge_usd
    gen_label = gen[0] if gen[0] in ("none", "train-pod") else f"{gen[0]}/{gen[1]}"
    ev_label = "ride-along" if ride else f"{ev[0]}/{ev[1]}"
    return Plan(tg, tt, n_pods, arms, gen_label, ev_label, usd, usd_nofail, wall_min / 60, pods)


def enumerate_plans(sp: Spec, gpus: dict) -> list:
    inf = [(n, t) for n, g in gpus.items() for t in sp.tiers if g.price(t) is not None]
    gens = [("none",)] if sp.teacher_tokens <= 0 else [("train-pod",)] + inf
    evs = [("ride-along",)] + (inf if sp.allow_separate_eval else [])
    out = []
    for (tg, tt), gen, ev in itertools.product(inf, gens, evs):
        for n in range(1, min(sp.max_pods, sp.runs) + 1):
            p = evaluate(sp, gpus, tg, tt, n, gen, ev)
            if p is not None:
                out.append(p)
    return out


def pareto(plans: list) -> list:
    """Non-dominated plans in ($, wall); ties broken by cost. Sorted by increasing $."""
    front = []
    for p in sorted(plans, key=lambda p: (p.usd, p.wall_h)):
        if not front or p.wall_h < front[-1].wall_h - 1e-9:
            front.append(p)
    return front


def best_per_train_gpu(plans: list) -> list:
    best = {}
    for p in plans:
        k = (p.train_gpu, p.train_tier)
        if k not in best or p.usd < best[k].usd:
            best[k] = p
    return sorted(best.values(), key=lambda p: p.usd)


def cost_breakdown(p: Plan, sp: Spec, gpus: dict) -> dict:
    """$ by component for the chosen plan (deterministic part + expected risk)."""
    b = {"train compute": 0.0, "evals": 0.0, "teacher gen": 0.0, "pod fixed (setup+tail)": 0.0,
         "risk (failed setups, mid-run loss)": 0.0, "API judge": 0.0}
    for pod in p.pods:
        g = gpus[pod.gpu]
        r = pod.usd_h / 60
        fixed = setup_min(g, pod.tier, sp) + sp.tail_min
        b["pod fixed (setup+tail)"] += fixed * r
        b["risk (failed setups, mid-run loss)"] += pod.exp_extra_min * r
        if pod.role == "train":
            tr = pod.arms * train_run_min(g, sp)
            b["train compute"] += tr * r
            rest = pod.billed_min - fixed - tr
            if p.gen == "train-pod" and pod is p.pods[0]:
                gm = gen_min(g, sp) + sp.judge_wall_min
                b["teacher gen"] += gm * r
                rest -= gm
            b["evals"] += rest * r
        elif pod.role == "gen":
            b["teacher gen"] += (pod.billed_min - fixed) * r
        else:
            b["evals"] += (pod.billed_min - fixed) * r
    b["API judge"] = (sp.judge_usd if sp.teacher_tokens > 0 else 0.0) + sp.eval_judge_usd
    return {k: round(v, 3) for k, v in b.items()}


# ----------------------------------------------------------------------------------------------------
# Examples
# ----------------------------------------------------------------------------------------------------
def example_a() -> Spec:
    """2 arms x 3 seeds, existing training files (opposite-style retrain): no new teacher data.
    Evals: packed probes for each adapter + checks (~0.15M tok/run), untouched model re-scored."""
    return Spec("a: 2 arms x 3 seeds, reuse data", n_arms=2, n_seeds=3, n_ref_variants=1)


def example_b() -> Spec:
    """6-arm new-trait experiment, 1 seed: fresh teacher (~6.7M output tokens, 10 F3) + gpt-6-luna judge
    (~$0.5); evals on all 6 arms + untouched + 2 reference adapters."""
    return Spec("b: 6 arms x 1 seed, new trait", n_arms=6, n_seeds=1, teacher_tokens=6.7e6, judge_usd=0.50,
                n_ref_variants=3)


def report(sp: Spec, gpus: dict, top: int = 8) -> dict:
    plans = enumerate_plans(sp, gpus)
    if not plans:
        raise ValueError("no feasible plan (stock caps / VRAM / restrictions)")
    best = min(plans, key=lambda p: p.usd)
    front = pareto(plans)
    return {"spec": sp, "best": best, "front": front, "per_gpu": best_per_train_gpu(plans)[:top],
            "breakdown": cost_breakdown(best, sp, gpus), "n_plans": len(plans)}


def print_report(r: dict) -> None:
    sp = r["spec"]
    print(f"== {sp.name}: {sp.runs} arm-runs x {sp.steps:.0f} steps, recipe={sp.recipe}, "
          f"teacher={sp.teacher_tokens/1e6:.1f}M tok; {r['n_plans']} feasible plans")
    print("MIN-COST:  " + r["best"].summary())
    print("  breakdown $:", r["breakdown"])
    print("PARETO FRONT ($ vs wall):")
    for p in r["front"]:
        print("  " + p.summary())
    print("BEST PLAN PER TRAINING GPU/TIER:")
    for p in r["per_gpu"]:
        print("  " + p.summary())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--example", choices=["a", "b"], action="append")
    ap.add_argument("--spec", help="JSON file with Spec fields")
    ap.add_argument("--hw-json", help="JSON list of GPU dicts replacing/adding table rows")
    ap.add_argument("--override", action="append", default=[], help='"GPU.field=val" or "GPU.field:tier=val"')
    ap.add_argument("--set", action="append", default=[], help='spec field override "field=val" (JSON value)')
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    gpus = default_gpus()
    if a.hw_json:
        for d in json.load(open(a.hw_json)):
            gpus[d["name"]] = GPU(**d)
    apply_overrides(gpus, MEASURED_OVERRIDES)
    for o in a.override:
        k, tier, v = parse_override(o)
        name, field = k.rsplit(".", 1)
        if tier:
            getattr(gpus[name], field)[tier] = v
            gpus[name].src[field] = "override"
        else:
            apply_overrides(gpus, {k: v})
    specs = []
    if a.spec:
        specs.append(Spec(**json.load(open(a.spec))))
    for e in a.example or []:
        specs.append(example_a() if e == "a" else example_b())
    if not specs:
        specs = [example_a(), example_b()]
    for sp in specs:
        for s in a.set:
            f, v = s.split("=", 1)
            setattr(sp, f, json.loads(v))
        r = report(sp, gpus)
        if a.json:
            print(json.dumps({"best": dc.asdict(r["best"]), "front": [dc.asdict(p) for p in r["front"]],
                              "breakdown": r["breakdown"]}, indent=1, default=str))
        else:
            print_report(r)
            print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
