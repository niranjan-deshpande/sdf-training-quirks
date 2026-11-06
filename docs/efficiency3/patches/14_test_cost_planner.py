#!/usr/bin/env python3
"""Unit tests for 14_cost_planner.py. Run: python3 14_test_cost_planner.py (stdlib only, <1 s, <50 MB)."""
import importlib.util
import os
import sys
import unittest

_P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "14_cost_planner.py")
_s = importlib.util.spec_from_file_location("cost_planner14", _P)
cp = importlib.util.module_from_spec(_s)
sys.modules["cost_planner14"] = cp
_s.loader.exec_module(cp)


def no_risk(gpus):
    for g in gpus.values():
        g.p_setup_fail = {"community": 0.0, "secure": 0.0}
        g.hazard_per_h = {"community": 0.0, "secure": 0.0}
    return gpus


class TestStages(unittest.TestCase):
    def test_single_arm_hand_computed(self):
        G = no_risk(cp.default_gpus())
        sp = cp.Spec("t", 1, 1, recipe="cur", n_ref_variants=1, check_tok_per_run=0.15e6)
        p = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("none",), ("ride-along",))
        g = G["H100 SXM"]
        train = 11558 / 16 * 2.40 / 60 + 1.5
        ev_run = 13.0 / 60 + 0.15e6 / 9000 / 60
        minutes = 2.5 + train + ev_run + (1.6 + 0.5 + ev_run) + 2.0
        self.assertAlmostEqual(p.pods[0].billed_min, minutes, places=6)
        self.assertAlmostEqual(p.usd, minutes * 3.99 / 60, places=6)
        self.assertAlmostEqual(p.wall_h, minutes / 60, places=6)
        self.assertEqual(g.s_step("cur"), 2.40)

    def test_token_scaling(self):
        G = cp.default_gpus()
        a = cp.Spec("a", 1, 1)
        b = cp.Spec("b", 1, 1, tok_per_row=cp.REF_TOK_PER_ROW * 0.62)  # 10 F2: -38% trained tokens
        g = G["L40S"]
        ta, tb = cp.train_run_min(g, a) - 1.5, cp.train_run_min(g, b) - 1.5
        self.assertAlmostEqual(tb / ta, 0.62, places=6)

    def test_failure_overhead_is_geometric(self):
        G = cp.default_gpus()
        sp = cp.Spec("t", 1, 1)
        for g in G.values():
            g.hazard_per_h = {"community": 0.0, "secure": 0.0}
        G["L40S"].p_setup_fail["community"] = 0.5
        p = cp.evaluate(sp, G, "L40S", "community", 1, ("none",), ("ride-along",))
        extra = 0.5 / 0.5 * sp.fail_detect_min * 0.79 / 60
        self.assertAlmostEqual(p.usd - p.usd_nofail, extra, places=6)


class TestPlans(unittest.TestCase):
    def setUp(self):
        self.G = cp.default_gpus()

    def test_more_pods_cost_more_and_finish_sooner(self):
        sp = cp.example_a()
        G = no_risk(self.G)
        G["L40S"].stock_cap["community"] = 6
        p1 = cp.evaluate(sp, G, "L40S", "community", 1, ("none",), ("ride-along",))
        p3 = cp.evaluate(sp, G, "L40S", "community", 3, ("none",), ("ride-along",))
        self.assertGreater(p3.usd, p1.usd)
        self.assertLess(p3.wall_h, p1.wall_h / 2)
        self.assertEqual(p3.arms_per_pod, [2, 2, 2])
        # extra $ of 3 pods = 2 extra setups+tails + 2 extra vLLM engines/probe loads, nothing else
        g = G["L40S"]
        fixed = 2 * (g.setup_min["community"] + 2.0 + g.vllm_init_min + 0.5) * 0.79 / 60
        self.assertAlmostEqual(p3.usd - p1.usd, fixed, places=6)

    def test_stock_cap_respected(self):
        sp = cp.example_a()
        self.G["L40S"].stock_cap["community"] = 2
        self.assertIsNone(cp.evaluate(sp, self.G, "L40S", "community", 3, ("none",), ("ride-along",)))
        for p in cp.enumerate_plans(sp, self.G):
            self.assertLessEqual(p.n_pods, self.G[p.train_gpu].stock_cap[p.train_tier])

    def test_no_stock_means_infeasible(self):
        sp = cp.example_a()
        for g in self.G.values():
            g.stock_cap = {"community": 0, "secure": 0}
        with self.assertRaises(ValueError):
            cp.report(sp, self.G)

    def test_vram_filter(self):
        sp = cp.Spec("28b", 1, 1, min_vram_train=80, min_vram_infer=80)
        for p in cp.enumerate_plans(sp, self.G):
            self.assertGreaterEqual(self.G[p.train_gpu].vram_gb, 80)

    def test_pin_train_gpu(self):
        sp = cp.example_a()
        sp.train_gpus = ["H100 SXM"]
        self.assertTrue(all(p.train_gpu == "H100 SXM" for p in cp.enumerate_plans(sp, self.G)))

    def test_pareto_front(self):
        plans = cp.enumerate_plans(cp.example_b(), self.G)
        front = cp.pareto(plans)
        self.assertAlmostEqual(front[0].usd, min(p.usd for p in plans))
        self.assertAlmostEqual(front[-1].wall_h, min(p.wall_h for p in plans))
        for i in range(1, len(front)):
            self.assertGreater(front[i].usd, front[i - 1].usd - 1e-12)
            self.assertLess(front[i].wall_h, front[i - 1].wall_h)
        for f in front:  # no plan strictly dominates a front member
            self.assertFalse(any(p.usd < f.usd - 1e-9 and p.wall_h < f.wall_h - 1e-9 for p in plans))

    def test_separate_eval_pays_only_on_expensive_training(self):
        G = no_risk(self.G)
        sp = cp.example_a()
        h_ride = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("none",), ("ride-along",))
        h_sep = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("none",), ("A40", "secure"))
        self.assertLess(h_sep.usd, h_ride.usd)  # 11 F2: pays at H100 prices
        G["L40S"].stock_cap["community"] = 2
        c_ride = cp.evaluate(sp, G, "L40S", "community", 1, ("none",), ("ride-along",))
        c_sep = cp.evaluate(sp, G, "L40S", "community", 1, ("none",), ("A40", "secure"))
        self.assertGreater(c_sep.usd, c_ride.usd)  # ... and not when training is already cheap

    def test_gen_required_when_teacher_tokens(self):
        sp = cp.example_b()
        self.assertIsNone(cp.evaluate(sp, self.G, "H100 SXM", "secure", 1, ("none",), ("ride-along",)))
        self.assertTrue(all(p.gen != "none" for p in cp.enumerate_plans(sp, self.G)))

    def test_judge_added_once(self):
        G = no_risk(self.G)
        sp = cp.example_b()
        p = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("train-pod",), ("ride-along",))
        self.assertAlmostEqual(p.usd - sum(x.billed_min * x.usd_h / 60 for x in p.pods), sp.judge_usd)

    def test_breakdown_sums_to_total(self):
        for sp in (cp.example_a(), cp.example_b()):
            r = cp.report(sp, self.G)
            self.assertAlmostEqual(sum(r["breakdown"].values()), r["best"].usd, delta=0.01)

    def test_overrides(self):
        cp.apply_overrides(self.G, {"A40.s_step_opt": 4.0})
        self.assertEqual(self.G["A40"].s_step_opt, 4.0)
        self.assertIn("override", self.G["A40"].src["s_step_opt"])
        with self.assertRaises(KeyError):
            cp.apply_overrides(self.G, {"A40.nonsense": 1.0})

    def test_calibration_h100_secure(self):
        """Opposite seed-2 retrain pod: 2 arms, H100 SXM Secure, measured 60 min ~$4 (HANDOFF);
        park pod: 1 arm + identity-row gen, measured 44 min ~$2.95. Model within +20%."""
        G = no_risk(self.G)
        sp = cp.Spec("opp", 2, 1, recipe="cur", check_tok_per_run=0.11e6)
        p = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("none",), ("ride-along",))
        self.assertTrue(60 * 0.95 <= p.pods[0].billed_min <= 60 * 1.20, p.pods[0].billed_min)
        sp = cp.Spec("park", 1, 1, recipe="cur", teacher_tokens=1.0e6, n_ref_variants=6)
        p = cp.evaluate(sp, G, "H100 SXM", "secure", 1, ("train-pod",), ("ride-along",))
        self.assertTrue(44 * 0.95 <= p.pods[0].billed_min <= 44 * 1.20, p.pods[0].billed_min)


if __name__ == "__main__":
    unittest.main(verbosity=1)
