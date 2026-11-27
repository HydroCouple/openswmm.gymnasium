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

"""Unit tests for L{openswmm_gymnasium.benchmarks.b01_twin_tank}.

Construction + registration tests run in-sandbox. Episode tests are
integration-marked (real engine required on host).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
from gymnasium import spaces

import openswmm_gymnasium  # noqa: F401 — registers benchmarks
from openswmm_gymnasium.benchmarks import b01_twin_tank
from openswmm_gymnasium.envs import (
    SwmmJointCIPRTCEnv,
    SwmmMORTCEnv,
    SwmmRTCEnv,
)

REGISTRATION_CASES = [
    (
        "OpenSWMM/TwinTank-RTC-v0",
        "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_rtc_env",
    ),
    (
        "OpenSWMM/TwinTank-Joint-v0",
        "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_joint_env",
    ),
    (
        "OpenSWMM/TwinTank-MORTC-v0",
        "openswmm_gymnasium.benchmarks.b01_twin_tank:_make_mo_env",
    ),
]


class TestScenarioFile(unittest.TestCase):
    def test_inp_file_exists(self):
        self.assertTrue(b01_twin_tank.SCENARIO_INP.exists())

    def test_inp_file_has_expected_sections(self):
        """Smoke-check the .inp file for the sections our env factories rely on."""
        content = b01_twin_tank.SCENARIO_INP.read_text()
        for section in (
            "[OPTIONS]",
            "[STORAGE]",
            "[OUTFALLS]",
            "[ORIFICES]",
            "[CONDUITS]",
            "[XSECTIONS]",
            "[TIMESERIES]",
        ):
            self.assertIn(section, content, f"Missing section {section}")
        # Symbolic IDs the env factories will resolve.
        for sid in ("T1", "T2", "O1", "ORIF", "OUT", "RG", "S1"):
            self.assertIn(sid, content, f"Missing symbol {sid}")


class TestRegistration(unittest.TestCase):
    def test_env_registered(self):
        for env_id, entry_point in REGISTRATION_CASES:
            with self.subTest(env_id=env_id, entry_point=entry_point):
                spec = gym.spec(env_id)
                self.assertEqual(spec.entry_point, entry_point)


class TestMakeEnvConstruction(unittest.TestCase):
    """Construction goes through enough engine surfaces (path validation,
    action space assembly) that the env can be built without C{open()}-ing
    the solver."""

    def test_rtc_returns_swmmrtcenv(self):
        env = b01_twin_tank.make_env("rtc")
        self.assertIsInstance(env, SwmmRTCEnv)
        # Action space has the expected runtime key only; the empty design
        # half is omitted (Gymnasium forbids empty Dict spaces).
        self.assertIn("orifice_setting", env.action_space["runtime"].spaces)
        self.assertNotIn("design", env.action_space.spaces)
        env.close()

    def test_joint_returns_swmmjoint(self):
        env = b01_twin_tank.make_env("joint")
        self.assertIsInstance(env, SwmmJointCIPRTCEnv)
        self.assertIn("node_max_depth", env.action_space["design"].spaces)
        self.assertIn("orifice_setting", env.action_space["runtime"].spaces)
        env.close()

    def test_mo_returns_swmmmortcenv(self):
        env = b01_twin_tank.make_env("mo")
        self.assertIsInstance(env, SwmmMORTCEnv)
        self.assertIsInstance(env.reward_space, spaces.Box)
        # Three reward terms.
        self.assertEqual(env.reward_space.shape, (3,))
        env.close()

    def test_unknown_variant_raises(self):
        with self.assertRaisesRegex(ValueError, "Unknown variant"):
            b01_twin_tank.make_env("not_a_variant")


class TestObservationSize(unittest.TestCase):
    """Observation builder produces 7 features for b01:
    2 node depths + 1 link flow + 1 link setting + 1 rainfall + 1 elapsed_frac
    (clock = 1 of 3 features when only elapsed_frac is requested)."""

    def test_obs_shape(self):
        env = b01_twin_tank.make_env("rtc")
        # 2 (depths T1,T2) + 1 (OUT flow) + 1 (ORIF setting) + 1 (RG rain)
        # + 1 (clock elapsed_frac) = 6
        self.assertEqual(env.observation_space.shape, (6,))
        env.close()


# ---------------------------------------------------------------------------
# Integration — needs the real engine
# ---------------------------------------------------------------------------


class TestEpisode(unittest.TestCase):
    def test_rtc_episode_runs(self):
        env = b01_twin_tank.make_env("rtc")
        env.reset(seed=0)
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
            if steps > 5000:
                self.fail("Episode did not terminate within safety limit")
        self.assertGreater(steps, 100)  # 4-hr sim at 15-s routing step ≈ 960 steps
        env.close()

    def test_mo_episode_emits_score(self):
        env = b01_twin_tank.make_env("mo")
        env.reset(seed=0)
        info = None
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            if terminated or truncated:
                break
        self.assertIsNotNone(info)
        self.assertIn("mo_score", info)
        self.assertLessEqual(0.0, info["mo_score"])
        self.assertLessEqual(info["mo_score"], 1.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
