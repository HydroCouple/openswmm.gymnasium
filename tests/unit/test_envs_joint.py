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

"""Unit tests for L{openswmm_gymnasium.envs.SwmmJointCIPRTCEnv}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmJointCIPRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.design import LinkRoughness
from openswmm_gymnasium.spaces.runtime import OrificeSetting
from tests.unit._base import BaseEngineTest


class TestConstruction(BaseEngineTest):
    def test_requires_design_factories(self):
        with self.assertRaisesRegex(ValueError, "design_factories"):
            SwmmJointCIPRTCEnv(
                self.minimal_inp,
                design_factories=[],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
            )

    def test_requires_observation_builder(self):
        with self.assertRaisesRegex(ValueError, "observation_builder"):
            SwmmJointCIPRTCEnv(
                self.minimal_inp,
                design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            )

    def test_action_space_structure(self):
        env = SwmmJointCIPRTCEnv(
            self.minimal_inp,
            design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        self.assertIsInstance(env.action_space, spaces.Dict)
        self.assertEqual(set(env.action_space.spaces.keys()), {"design", "runtime"})
        self.assertIn("link_roughness", env.action_space["design"].spaces)
        self.assertIn("orifice_setting", env.action_space["runtime"].spaces)


class TestRegistration(unittest.TestCase):
    def test_registered_id_resolves(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-Joint-v0")
        self.assertEqual(spec.entry_point, "openswmm_gymnasium.envs:SwmmJointCIPRTCEnv")


# ---------------------------------------------------------------------------
# Integration tier — real engine required
# ---------------------------------------------------------------------------


def _make_joint_env(inp_path):
    return SwmmJointCIPRTCEnv(
        inp_path,
        design_factories=[LinkRoughness(["C1"], 0.005, 0.05)],
        runtime_factories=[OrificeSetting(["C1"])],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


class TestEpisode(BaseEngineTest):
    def test_reset_applies_supplied_design(self):
        env = _make_joint_env(self.minimal_inp)
        chosen = {"link_roughness": np.array([0.03], dtype=np.float32)}
        _, info = env.reset(seed=0, options={"design_action": chosen})
        np.testing.assert_array_equal(
            info["design_action"]["link_roughness"], chosen["link_roughness"]
        )
        env.close()

    def test_reset_samples_design_when_not_supplied(self):
        env = _make_joint_env(self.minimal_inp)
        _, info = env.reset(seed=0)
        # Sampled design is recorded in info; only check it's the right
        # shape + within bounds.
        d = info["design_action"]["link_roughness"]
        self.assertEqual(d.shape, (1,))
        self.assertTrue(0.005 <= float(d[0]) <= 0.05)
        env.close()

    def test_episode_runs_to_termination(self):
        env = _make_joint_env(self.minimal_inp)
        env.reset(seed=0)
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
            if steps > 10_000:
                self.fail("Episode did not terminate within safety limit")
        self.assertTrue(60 <= steps <= 500)
        env.close()

    def test_check_env_compliance(self):
        from gymnasium.utils.env_checker import check_env

        env = _make_joint_env(self.minimal_inp)
        check_env(env, skip_render_check=True)
        env.close()

    def test_design_change_affects_trajectory(self):
        """Two distinct designs at reset produce distinct cumulative reward.

        Verifies the CIP mutation propagates into the RTC episode
        (plan §10 P3 verify).
        """
        env = _make_joint_env(self.minimal_inp)

        # Deterministic runtime: open the orifice fully every step.
        def _run(design_value):
            env.reset(
                seed=0,
                options={
                    "design_action": {
                        "link_roughness": np.array([design_value], dtype=np.float32),
                    }
                },
            )
            cumulative = 0.0
            action = {
                "design": {"link_roughness": np.array([design_value], dtype=np.float32)},
                "runtime": {"orifice_setting": np.array([1.0], dtype=np.float32)},
            }
            while True:
                _, r, terminated, truncated, _ = env.step(action)
                cumulative += r
                if terminated or truncated:
                    break
            return cumulative

        r_low = _run(0.005)
        r_high = _run(0.05)
        self.assertNotEqual(
            r_low,
            r_high,
            f"Expected design to change cumulative reward; got r_low={r_low} r_high={r_high}",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
