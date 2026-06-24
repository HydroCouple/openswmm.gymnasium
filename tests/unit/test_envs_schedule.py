"""Unit tests for open-loop schedule control via L{SwmmControlEnv}.

Drives the controller-agnostic env in generic mode with a
L{ScheduleController} + L{SchedulePolicySpace} (no metric reader, open-loop) on
the twin-tank benchmark (controllable orifice C{ORIF}), per the suite's
no-mocks policy.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import numpy as np
from tests.unit._base import BaseTestCase

from openswmm_gymnasium.benchmarks.b01_twin_tank import SCENARIO_INP
from openswmm_gymnasium.control import Controller, ScheduleController
from openswmm_gymnasium.envs import SwmmControlEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import StorageUnderUtilization, UncontrolledDischarge
from openswmm_gymnasium.spaces import SchedulePolicySpace

_N_POINTS = 4


def _schedule_env(rpt: str, out: str) -> SwmmControlEnv:
    sp = SchedulePolicySpace(["ORIF"], n_points=_N_POINTS)
    return SwmmControlEnv(
        SCENARIO_INP,
        observation_builder=ObservationBuilder().add_node_depths(["T1", "T2"]),
        reward_terms=[UncontrolledDischarge(["OUT"]), StorageUnderUtilization(["T1", "T2"])],
        policy_space=sp,
        controller_factory=lambda sched: ScheduleController(["ORIF"], sched),
        metric_reader_factory=None,  # open-loop
        control_interval_seconds=300,
        rpt_path=rpt,
        out_path=out,
    )


class TestConstruction(unittest.TestCase):
    def test_generic_mode_requires_factory(self):
        sp = SchedulePolicySpace(["ORIF"], n_points=2)
        with self.assertRaisesRegex(ValueError, "market_config, or both"):
            SwmmControlEnv(
                SCENARIO_INP,
                observation_builder=ObservationBuilder().add_node_depths(["T1"]),
                policy_space=sp,  # no controller_factory
                control_interval_seconds=300,
            )

    def test_action_space_is_schedule_vector(self):
        env = _schedule_env("/tmp/s.rpt", "/tmp/s.out")
        self.assertEqual(env.action_space.shape, (_N_POINTS,))  # 1 structure x 4 points
        self.assertEqual(env.policy_space.labels[0], "sched:ORIF[0]")


class TestIntegration(BaseTestCase):
    def _env(self) -> SwmmControlEnv:
        return _schedule_env(str(self.tmp_path / "b01.rpt"), str(self.tmp_path / "b01.out"))

    def test_full_episode_open_loop(self):
        env = self._env()
        env.reset(seed=0)
        action = np.full((_N_POINTS,), 0.5, dtype=np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(
            set(info["reward_components"]), {"uncontrolled_discharge", "storage_underutilization"}
        )
        for v in info["reward_components"].values():
            self.assertGreaterEqual(v, 0.0)

    def test_open_vs_closed_schedule_differ(self):
        # A schedule that holds ORIF shut retains water (low discharge to the
        # untreated outfall) vs one that holds it open — proving the open-loop
        # schedule actually drives the simulation.
        env = self._env()

        def run(value: float) -> float:
            env.reset(seed=0)
            _, _, _, _, info = env.step(np.full((_N_POINTS,), value, dtype=np.float32))
            return info["reward_components"]["uncontrolled_discharge"]

        shut = run(0.0)
        open_ = run(1.0)
        env.close()
        self.assertLess(shut, open_)


class TestProtocol(unittest.TestCase):
    def test_schedule_controller_is_controller(self):
        self.assertIsInstance(ScheduleController(["ORIF"], {"ORIF": [0.5, 0.5]}), Controller)

    def test_empty_schedule_rejected(self):
        with self.assertRaises(ValueError):
            ScheduleController(["ORIF"], {"ORIF": []})


if __name__ == "__main__":
    unittest.main()
