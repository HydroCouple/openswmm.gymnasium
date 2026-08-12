# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Caleb Buahin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the reactive PWL control-curve policy.

Two layers, mirroring the suite's no-mocks policy:

  - Pure tests (no engine): PWL interpolation, monotonic projection,
    structural validation, decision-vector shape/labels/bounds, controller
    determinism and rate limiting. These drive
    L{ControlCurvePolicySpace<openswmm_gymnasium.spaces.ControlCurvePolicySpace>}
    and L{ControlCurveController<openswmm_gymnasium.control.ControlCurveController>}
    directly with hand-built metrics — no solver needed.
  - Engine tests: close the loop through L{SwmmControlEnv} on the twin-tank
    benchmark (controllable orifice C{ORIF}, observed tank C{T1}) to confirm
    the neutral (all-open) curve reproduces the uncontrolled baseline, that a
    retention curve actually reduces discharge, and that identical params give
    identical costs.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.benchmarks.b01_twin_tank import SCENARIO_INP
from openswmm_gymnasium.control import (
    ControlCurveController,
    ControlCurveMetricReader,
    Controller,
    project_monotonic,
    pwl_interp,
)
from openswmm_gymnasium.envs import SwmmControlEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import StorageUnderUtilization, UncontrolledDischarge
from openswmm_gymnasium.spaces import ControlCurvePolicySpace
from tests.unit._base import BaseTestCase

_SECONDS_PER_DAY = 86400.0

# A single retention curve on ORIF observing T1's fractional depth.
_ASSET = {
    "link_id": "ORIF",
    "obs_node": "T1",
    "obs_attr": "depthN",
    "x_knots": [0.0, 0.33, 0.66, 1.0],
    "y_low": 0.0,
    "y_high": 1.0,
    "monotonic": "nonincreasing",
}


def _space(**kw) -> ControlCurvePolicySpace:
    assets = kw.pop("assets", [dict(_ASSET)])
    return ControlCurvePolicySpace(assets, **kw)


# =============================================================================
# Pure: PWL interpolation
# =============================================================================


class TestPWLMapping(unittest.TestCase):
    def test_exact_at_knots(self):
        xk = [0.0, 0.5, 1.0]
        yv = [1.0, 0.4, 0.0]
        for x, y in zip(xk, yv, strict=True):
            self.assertAlmostEqual(pwl_interp(x, xk, yv), y)

    def test_linear_between_knots(self):
        xk = [0.0, 1.0]
        yv = [0.0, 1.0]
        self.assertAlmostEqual(pwl_interp(0.25, xk, yv), 0.25)
        self.assertAlmostEqual(pwl_interp(0.75, xk, yv), 0.75)

    def test_flat_extrapolation_beyond_ends(self):
        xk = [0.2, 0.8]
        yv = [0.9, 0.1]
        self.assertAlmostEqual(pwl_interp(-5.0, xk, yv), 0.9)  # below first knot
        self.assertAlmostEqual(pwl_interp(5.0, xk, yv), 0.1)  # above last knot

    def test_output_clamped_to_unit_interval(self):
        xk = [0.0, 1.0]
        yv = [2.0, -1.0]  # out-of-range settings
        self.assertEqual(pwl_interp(0.0, xk, yv), 1.0)
        self.assertEqual(pwl_interp(1.0, xk, yv), 0.0)


# =============================================================================
# Pure: monotonic projection
# =============================================================================


class TestMonotonicProjection(unittest.TestCase):
    def test_nonincreasing_projection_is_feasible(self):
        out = project_monotonic([0.2, 0.9, 0.4, 1.0], "nonincreasing")
        self.assertEqual(out, (0.2, 0.2, 0.2, 0.2))
        self.assertTrue(all(out[i + 1] <= out[i] for i in range(len(out) - 1)))

    def test_nondecreasing_projection_is_feasible(self):
        out = project_monotonic([0.5, 0.1, 0.7, 0.3], "nondecreasing")
        self.assertEqual(out, (0.5, 0.5, 0.7, 0.7))
        self.assertTrue(all(out[i + 1] >= out[i] for i in range(len(out) - 1)))

    def test_none_leaves_unchanged(self):
        vals = [0.5, 0.1, 0.7, 0.3]
        self.assertEqual(project_monotonic(vals, "none"), tuple(vals))

    def test_projection_is_idempotent(self):
        for mode in ("nonincreasing", "nondecreasing", "none"):
            once = project_monotonic([0.4, 0.8, 0.2, 0.6], mode)
            twice = project_monotonic(once, mode)
            self.assertEqual(once, twice, mode)

    def test_bad_mode_rejected(self):
        with self.assertRaises(ValueError):
            project_monotonic([0.1, 0.2], "wiggly")


# =============================================================================
# Pure: params validation
# =============================================================================


class TestParamsValidation(unittest.TestCase):
    def test_accepts_valid_config(self):
        sp = _space()
        self.assertEqual(len(sp), 4)

    def test_empty_assets_rejected(self):
        with self.assertRaisesRegex(ValueError, "at least one asset"):
            ControlCurvePolicySpace([])

    def test_missing_link_id_rejected(self):
        bad = dict(_ASSET)
        bad.pop("link_id")
        with self.assertRaisesRegex(ValueError, "link_id"):
            _space(assets=[bad])

    def test_missing_obs_node_rejected(self):
        bad = dict(_ASSET)
        bad.pop("obs_node")
        with self.assertRaisesRegex(ValueError, "obs_node"):
            _space(assets=[bad])

    def test_too_few_knots_rejected(self):
        bad = dict(_ASSET, x_knots=[0.5])
        with self.assertRaisesRegex(ValueError, "at least 2"):
            _space(assets=[bad])

    def test_non_increasing_knots_rejected(self):
        bad = dict(_ASSET, x_knots=[0.0, 0.5, 0.5, 1.0])
        with self.assertRaisesRegex(ValueError, "strictly increasing"):
            _space(assets=[bad])

    def test_y_low_above_y_high_rejected(self):
        bad = dict(_ASSET, y_low=0.8, y_high=0.2)
        with self.assertRaisesRegex(ValueError, "y_low"):
            _space(assets=[bad])

    def test_y_init_length_mismatch_rejected(self):
        bad = dict(_ASSET, y_init=[1.0, 1.0])  # 2 != 4 knots
        with self.assertRaisesRegex(ValueError, "y_init"):
            _space(assets=[bad])

    def test_bad_monotonic_rejected(self):
        bad = dict(_ASSET, monotonic="sideways")
        with self.assertRaisesRegex(ValueError, "monotonic"):
            _space(assets=[bad])

    def test_duplicate_link_rejected(self):
        with self.assertRaisesRegex(ValueError, "duplicate"):
            _space(assets=[dict(_ASSET), dict(_ASSET)])

    def test_bad_rate_limit_rejected(self):
        with self.assertRaisesRegex(ValueError, "rate_limit"):
            _space(rate_limit=0.0)

    def test_bad_interval_rejected(self):
        with self.assertRaisesRegex(ValueError, "control_interval_steps"):
            _space(control_interval_steps=0)


# =============================================================================
# Pure: decision-vector shape / labels / bounds
# =============================================================================


class TestDecisionVector(unittest.TestCase):
    def test_shape_equals_sum_of_knots(self):
        sp = _space(
            assets=[
                dict(_ASSET, link_id="Or1", obs_node="T1", x_knots=[0.0, 0.5, 1.0]),
                dict(_ASSET, link_id="Or2", obs_node="T2", x_knots=[0.0, 0.25, 0.75, 1.0]),
            ]
        )
        self.assertEqual(len(sp), 3 + 4)
        self.assertEqual(sp.space.shape, (7,))

    def test_labels_are_self_describing(self):
        sp = _space()
        self.assertEqual(
            sp.labels,
            [f"control_curve/ORIF/y[{k}]" for k in range(4)],
        )

    def test_per_dimension_bounds_match_asset(self):
        sp = _space(
            assets=[
                dict(_ASSET, link_id="Or1", y_low=0.0, y_high=1.0, x_knots=[0.0, 1.0]),
                dict(_ASSET, link_id="Or2", y_low=0.2, y_high=0.8, x_knots=[0.0, 1.0]),
            ]
        )
        np.testing.assert_allclose(sp.low, [0.0, 0.0, 0.2, 0.2])
        np.testing.assert_allclose(sp.high, [1.0, 1.0, 0.8, 0.8])

    def test_default_vector_is_y_high(self):
        sp = _space()
        np.testing.assert_allclose(sp.default_vector(), [1.0, 1.0, 1.0, 1.0])

    def test_unflatten_applies_clip_and_projection(self):
        sp = _space()  # nonincreasing
        pol = sp.unflatten(np.array([0.2, 0.9, 0.4, 2.0]))
        # clipped to [0,1] then running-min projected.
        self.assertEqual(pol.assets[0].y_values, (0.2, 0.2, 0.2, 0.2))

    def test_unflatten_length_mismatch_rejected(self):
        sp = _space()
        with self.assertRaises(ValueError):
            sp.unflatten(np.array([0.5, 0.5]))

    def test_decode_curves_roundtrip(self):
        sp = _space(assets=[dict(_ASSET, monotonic="none")])
        vec = np.array([0.9, 0.6, 0.3, 0.1])
        curves = sp.decode_curves(vec)
        self.assertEqual(len(curves), 1)
        self.assertEqual(curves[0]["link_id"], "ORIF")
        self.assertEqual(curves[0]["x_knots"], [0.0, 0.33, 0.66, 1.0])
        np.testing.assert_allclose(curves[0]["y_values"], [0.9, 0.6, 0.3, 0.1])


# =============================================================================
# Pure: controller behavior (metrics fed directly, no engine)
# =============================================================================


class TestController(unittest.TestCase):
    def _controller(self, **kw) -> ControlCurveController:
        return ControlCurveController(_space(**kw).unflatten(_space(**kw).default_vector()))

    def test_is_controller_protocol(self):
        pol = _space().unflatten(_space().default_vector())
        self.assertIsInstance(ControlCurveController(pol), Controller)

    def test_structure_ids(self):
        pol = _space().unflatten(_space().default_vector())
        self.assertEqual(ControlCurveController(pol).structure_ids, ["ORIF"])

    def test_pwl_lookup_and_determinism(self):
        sp = _space(assets=[dict(_ASSET, monotonic="none")])
        vec = np.array([1.0, 0.6, 0.3, 0.0])
        pol = sp.unflatten(vec)

        def run() -> list[float]:
            c = ControlCurveController(pol)
            c.reset()
            out = []
            for x in (0.0, 0.33, 0.66, 1.0, 0.5):
                out.append(c.compute_settings({"ORIF": x}, 60.0)["ORIF"])
            return out

        first = run()
        self.assertAlmostEqual(first[0], 1.0)
        self.assertAlmostEqual(first[1], 0.6)
        self.assertAlmostEqual(first[3], 0.0)
        self.assertEqual(first, run())  # deterministic

    def test_rate_limit_caps_step_change(self):
        sp = _space(assets=[dict(_ASSET, monotonic="none")], rate_limit=0.25)
        pol = sp.unflatten(np.array([1.0, 1.0, 0.0, 0.0]))
        c = ControlCurveController(pol)
        c.reset()
        prev = None
        # Ramp the observation up so the raw curve drops fast; |Δ| must stay <= 0.25.
        for x in np.linspace(0.0, 1.0, 11):
            s = c.compute_settings({"ORIF": float(x)}, 60.0)["ORIF"]
            if prev is not None:
                self.assertLessEqual(abs(s - prev), 0.25 + 1e-9)
            prev = s

    def test_control_interval_steps_holds_between_updates(self):
        sp = _space(assets=[dict(_ASSET, monotonic="none")], control_interval_steps=3)
        pol = sp.unflatten(np.array([1.0, 0.5, 0.5, 0.0]))
        c = ControlCurveController(pol)
        c.reset()
        s0 = c.compute_settings({"ORIF": 0.0}, 60.0)["ORIF"]  # recompute (call 0)
        s1 = c.compute_settings({"ORIF": 1.0}, 60.0)["ORIF"]  # held
        s2 = c.compute_settings({"ORIF": 1.0}, 60.0)["ORIF"]  # held
        s3 = c.compute_settings({"ORIF": 1.0}, 60.0)["ORIF"]  # recompute (call 3)
        self.assertEqual(s0, s1)
        self.assertEqual(s1, s2)
        self.assertNotAlmostEqual(s2, s3)  # now reflects x=1.0 -> 0.0


# =============================================================================
# Engine: closed loop through SwmmControlEnv on the twin-tank benchmark
# =============================================================================


def _cc_env(rpt: str, out: str, **space_kw) -> SwmmControlEnv:
    sp = _space(**space_kw)
    return SwmmControlEnv(
        SCENARIO_INP,
        observation_builder=ObservationBuilder().add_node_depths(["T1", "T2"]),
        reward_terms=[UncontrolledDischarge(["OUT"]), StorageUnderUtilization(["T1", "T2"])],
        policy_space=sp,
        controller_factory=lambda policy: ControlCurveController(policy),
        metric_reader_factory=lambda policy: ControlCurveMetricReader(policy),
        control_interval_seconds=300,
        rpt_path=rpt,
        out_path=out,
    )


def _uncontrolled_totals(rpt: str, out: str) -> dict[str, float]:
    """Run the engine to the end with NO control and accumulate the same reward
    terms ``_cc_env`` uses, on the same per-routing-step cadence.

    ORIF is left at its ``.inp`` default (1.0, fully open), so this is the
    independent "do-nothing" baseline the neutral all-y=1.0 curve must
    reproduce (spec §7.6). Mirrors ``SwmmControlEnv.step``'s accumulation,
    including the negative-dt clamp on the run's final step.
    """
    adapter = SolverAdapter(str(SCENARIO_INP), rpt, out)
    adapter.open()
    adapter.initialize()
    adapter.start()
    terms = [UncontrolledDischarge(["OUT"]), StorageUnderUtilization(["T1", "T2"])]
    for t in terms:
        t.bind(adapter)
        t.reset()
    totals = {t.name: 0.0 for t in terms}
    prev = adapter.elapsed
    while adapter.is_running:
        adapter.step()
        dt = (adapter.elapsed - prev) * _SECONDS_PER_DAY
        prev = adapter.elapsed
        if dt < 0.0:
            dt = 0.0
        for t in terms:
            totals[t.name] += float(t.step(adapter, dt))
    try:
        adapter.end()
        adapter.report()
    except Exception:
        pass
    adapter.close()
    return totals


class TestEngineClosedLoop(BaseTestCase):
    def _env(self, **kw) -> SwmmControlEnv:
        return _cc_env(str(self.tmp_path / "cc.rpt"), str(self.tmp_path / "cc.out"), **kw)

    def _discharge(self, env: SwmmControlEnv, vector: np.ndarray) -> float:
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.asarray(vector, dtype=np.float32))
        return info["reward_components"]["uncontrolled_discharge"]

    def test_action_space_matches_policy(self):
        env = self._env()
        self.assertEqual(env.action_space.shape, (4,))
        self.assertEqual(env.policy_space.labels[0], "control_curve/ORIF/y[0]")
        env.close()

    def test_neutral_curve_equals_open_baseline(self):
        # A flat y=1.0 curve holds ORIF fully open every step, i.e. the
        # uncontrolled baseline. Compare against an explicit all-open run.
        env = self._env(assets=[dict(_ASSET, monotonic="none")])
        neutral = self._discharge(env, np.ones(4, dtype=np.float32))
        open_again = self._discharge(env, np.ones(4, dtype=np.float32))
        env.close()
        # Determinism: identical params -> identical cost.
        self.assertAlmostEqual(neutral, open_again, places=6)
        self.assertGreater(neutral, 0.0)

    def test_neutral_curve_reproduces_independent_uncontrolled_baseline(self):
        # Spec §7.6 (tolerance <= 0.5%): the neutral all-y=1.0 curve must
        # reproduce a baseline built WITHOUT the controller (ORIF left at its
        # .inp default of 1.0), per objective. This is stronger than the
        # determinism check above: it confirms that driving ORIF to 1.0 every
        # control step is equivalent to never touching it, so "neutral == the
        # uncontrolled run" holds for the optimizer's neutral point.
        baseline = _uncontrolled_totals(
            str(self.tmp_path / "base.rpt"), str(self.tmp_path / "base.out")
        )
        env = self._env(assets=[dict(_ASSET, monotonic="none")])
        env.reset(seed=0)
        _, _, _, _, info = env.step(np.ones(4, dtype=np.float32))
        env.close()
        neutral = info["reward_components"]
        for name, base_val in baseline.items():
            self.assertGreater(base_val, 0.0, f"{name} baseline must be non-trivial")
            self.assertIn(name, neutral)
            rel = abs(neutral[name] - base_val) / abs(base_val)
            self.assertLessEqual(
                rel, 0.005, f"{name}: neutral={neutral[name]} baseline={base_val}"
            )

    def test_retention_curve_reduces_discharge(self):
        env = self._env()  # nonincreasing
        open_cost = self._discharge(env, np.ones(4, dtype=np.float32))
        # Retention: open when T1 low, throttle shut as it fills.
        retain = self._discharge(env, np.array([1.0, 0.5, 0.1, 0.0], dtype=np.float32))
        env.close()
        self.assertLess(retain, open_cost)

    def test_determinism_repeated_episodes(self):
        env = self._env()
        vec = np.array([0.9, 0.6, 0.3, 0.0], dtype=np.float32)
        a = self._discharge(env, vec)
        b = self._discharge(env, vec)
        env.close()
        self.assertAlmostEqual(a, b, places=6)


if __name__ == "__main__":
    unittest.main()
