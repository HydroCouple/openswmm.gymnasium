"""Unit tests for L{openswmm_gymnasium.control} (reactive market controller).

Pure-logic tier (no engine): cost-curve monotonicity/endpoints, PID
direct-action + anti-windup, controller protocol conformance, determinism
after reset, and pump cycle-constraint gating.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

from openswmm_gymnasium.config import (
    Agent,
    CostCurve,
    MarketConfig,
    Meta,
    PIDConfig,
    RouteConstraints,
    TradeRoute,
)
from openswmm_gymnasium.control import (
    PID,
    Controller,
    MarketController,
    aggregate_prices,
    price_from_curve,
)

_GRID = [0.0, 0.1, 0.3, 0.5, 0.6, 0.7, 0.85, 0.95, 1.0]


class TestCostCurves(unittest.TestCase):
    def test_piecewise_linear_monotonic_and_endpoints(self):
        c = CostCurve(type="piecewise_linear", onset=0.6, full=1.0)
        vals = [price_from_curve(m, c) for m in _GRID]
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        self.assertEqual(price_from_curve(0.6, c), 0.0)  # at onset -> floor
        self.assertEqual(price_from_curve(0.5, c), 0.0)  # below onset -> floor
        self.assertEqual(price_from_curve(1.0, c), 1.0)  # at full -> ceiling

    def test_logistic_monotonic_and_midpoint(self):
        c = CostCurve(type="logistic", onset=0.85, steepness=20)
        vals = [price_from_curve(m, c) for m in _GRID]
        self.assertTrue(all(b >= a for a, b in zip(vals, vals[1:])))
        # Logistic is centred on onset: price == midpoint of [floor, ceiling].
        self.assertAlmostEqual(price_from_curve(0.85, c), 0.5, places=6)

    def test_floor_ceiling_respected_and_clamped_metric(self):
        c = CostCurve(type="piecewise_linear", onset=0.2, full=0.8, floor=0.1, ceiling=0.9)
        self.assertAlmostEqual(price_from_curve(-5.0, c), 0.1, places=6)  # clamp low
        self.assertAlmostEqual(price_from_curve(5.0, c), 0.9, places=6)  # clamp high
        for m in _GRID:
            self.assertGreaterEqual(price_from_curve(m, c), 0.1)
            self.assertLessEqual(price_from_curve(m, c), 0.9)

    def test_aggregate(self):
        self.assertEqual(aggregate_prices([0.2, 0.9, 0.5], "max"), 0.9)
        self.assertAlmostEqual(aggregate_prices([0.2, 0.8], "mean"), 0.5, places=6)
        self.assertEqual(aggregate_prices([], "max"), 0.0)


class TestPID(unittest.TestCase):
    def test_direct_acting_positive_error_drives_up(self):
        pid = PID(kp=1.0)
        out = pid.update(0.5, dt=1.0)  # buyer more stressed than seller
        self.assertGreater(out, 0.0)

    def test_output_clamped_to_limits(self):
        pid = PID(kp=100.0, out_min=0.0, out_max=1.0)
        self.assertEqual(pid.update(1.0, 1.0), 1.0)
        self.assertEqual(pid.update(-1.0, 1.0), 0.0)

    def test_anti_windup_recovers_quickly(self):
        # Saturate high for many steps, then flip the error negative. With
        # anti-windup the output should drop off saturation within a step.
        pid = PID(kp=1.0, ki=1.0, out_min=0.0, out_max=1.0)
        for _ in range(50):
            pid.update(1.0, 1.0)  # drive hard into the ceiling
        out = pid.update(-1.0, 1.0)
        self.assertLess(out, 1.0)

    def test_reset_clears_state(self):
        pid = PID(kp=1.0, ki=1.0)
        pid.update(0.5, 1.0)
        pid.reset()
        a = pid.update(0.5, 1.0)
        # A fresh PID should reproduce the very first step exactly.
        fresh = PID(kp=1.0, ki=1.0).update(0.5, 1.0)
        self.assertEqual(a, fresh)


def _one_route_config(constraints: RouteConstraints | None = None) -> MarketConfig:
    """A single buyer/seller/route config for controller tests."""
    return MarketConfig(
        meta=Meta(model_path="/m", output_dir="/o", control_interval_seconds=60),
        cost_curves={
            "buyer": CostCurve(type="piecewise_linear", onset=0.0, full=1.0),
            "seller": CostCurve(type="piecewise_linear", onset=0.0, full=1.0),
        },
        agents=[
            Agent("B", "node", "conveyance", "freeboard_fraction", "buyer", role="buyer"),
            Agent("S", "node", "storage", "storage_fill", "seller", role="seller"),
        ],
        trade_routes=[
            TradeRoute(
                structure_link_id="GATE",
                buyer_agents=["B"],
                seller_agents=["S"],
                pid=PIDConfig(kp=5.0),
                constraints=constraints or RouteConstraints(),
            )
        ],
    )


class TestMarketController(unittest.TestCase):
    def test_protocol_conformance_and_structure_ids(self):
        mc = MarketController(_one_route_config())
        self.assertIsInstance(mc, Controller)
        self.assertEqual(mc.structure_ids, ["GATE"])

    def test_settings_within_route_bounds(self):
        cfg = _one_route_config(RouteConstraints(setting_min=0.2, setting_max=0.8))
        mc = MarketController(cfg)
        s = mc.compute_settings({"B": 0.9, "S": 0.0}, 60.0)["GATE"]
        self.assertGreaterEqual(s, 0.2)
        self.assertLessEqual(s, 0.8)

    def test_buyer_stress_opens_route(self):
        mc = MarketController(_one_route_config())
        s = mc.compute_settings({"B": 1.0, "S": 0.0}, 60.0)["GATE"]
        self.assertGreater(s, 0.0)
        d = mc.last_detail["GATE"]
        self.assertGreater(d["diff"], 0.0)

    def test_reset_makes_runs_deterministic(self):
        mc = MarketController(_one_route_config())
        seq = [{"B": 0.9, "S": 0.1}, {"B": 0.2, "S": 0.8}, {"B": 0.7, "S": 0.3}]
        run1 = [mc.compute_settings(m, 60.0)["GATE"] for m in seq]
        mc.reset()
        run2 = [mc.compute_settings(m, 60.0)["GATE"] for m in seq]
        self.assertEqual(run1, run2)

    def test_min_cycle_seconds_blocks_rapid_switch(self):
        # Dwell of 600 s with a 60 s control step: once the gate turns on it may
        # not turn back off for ~10 steps even if the buyer's demand vanishes.
        cfg = _one_route_config(RouteConstraints(min_cycle_seconds=600))
        mc = MarketController(cfg)
        mc.compute_settings({"B": 1.0, "S": 0.0}, 60.0)  # turn ON
        held_off = mc.compute_settings({"B": 0.0, "S": 1.0}, 60.0)["GATE"]
        self.assertGreater(held_off, 0.0, "switch-off must be blocked during dwell")

    def test_max_starts_per_hour_caps_starts(self):
        # One start allowed per hour. Toggle demand on/off across a full hour of
        # 60 s steps; only the first ON should ever take effect.
        cfg = _one_route_config(
            RouteConstraints(min_cycle_seconds=0, max_starts_per_hour=1)
        )
        mc = MarketController(cfg)
        on = {"B": 1.0, "S": 0.0}
        off = {"B": 0.0, "S": 1.0}
        settings = []
        for i in range(60):  # 60 minutes
            settings.append(mc.compute_settings(on if i % 2 == 0 else off, 60.0)["GATE"])
        # The route turns on once and is then start-capped: after it is forced
        # off it cannot restart within the hour.
        starts = sum(
            1 for a, b in zip([0.0] + settings, settings) if a <= 1e-6 < b
        )
        self.assertEqual(starts, 1)


if __name__ == "__main__":
    unittest.main()
