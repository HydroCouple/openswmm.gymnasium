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

"""Unit tests for L{openswmm_gymnasium.envs.SwmmCIPEnv}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmCIPEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.design import LinkRoughness, NodeMaxDepth
from tests.unit._base import BaseEngineTest


class TestConstruction(BaseEngineTest):
    def test_requires_design_factories(self):
        with self.assertRaisesRegex(ValueError, "design_factories"):
            SwmmCIPEnv(
                self.minimal_inp,
                design_factories=[],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
            )

    def test_requires_observation_builder(self):
        with self.assertRaisesRegex(ValueError, "observation_builder"):
            SwmmCIPEnv(
                self.minimal_inp,
                design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            )

    def test_action_space_structure(self):
        env = SwmmCIPEnv(
            self.minimal_inp,
            design_factories=[
                LinkRoughness(["C1"], 0.01, 0.05),
                NodeMaxDepth(["J1"], 3.0, 8.0),
            ],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        self.assertIsInstance(env.action_space, spaces.Dict)
        # Empty action halves are omitted (Gymnasium forbids empty Dict
        # spaces), so a CIP-only env exposes just "design".
        self.assertEqual(set(env.action_space.spaces.keys()), {"design"})
        # Design has one entry per factory.
        self.assertEqual(
            set(env.action_space["design"].spaces.keys()),
            {
                "link_roughness",
                "node_max_depth",
            },
        )

    def test_reset_returns_zero_obs(self):
        env = SwmmCIPEnv(
            self.minimal_inp,
            design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        obs, info = env.reset(seed=0)
        self.assertTrue(np.array_equal(obs, np.zeros_like(obs)))
        self.assertEqual(info["phase"], "awaiting_design")
        env.close()


class TestRegistration(unittest.TestCase):
    def test_registered_id_resolves(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-CIP-v0")
        self.assertEqual(spec.entry_point, "openswmm_gymnasium.envs:SwmmCIPEnv")


# ---------------------------------------------------------------------------
# Integration tier — needs real engine
# ---------------------------------------------------------------------------


def _make_cip_env(inp_path):
    return SwmmCIPEnv(
        inp_path,
        design_factories=[
            LinkRoughness(["C1"], 0.005, 0.05),
            NodeMaxDepth(["J1"], 3.0, 8.0),
        ],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


class TestEpisode(BaseEngineTest):
    def test_single_step_terminates(self):
        env = _make_cip_env(self.minimal_inp)
        env.reset(seed=0)
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        self.assertIs(terminated, True)
        self.assertIs(truncated, False)
        self.assertEqual(obs.shape, (2,))
        self.assertIsInstance(reward, float)
        self.assertIn("reward_components", info)
        self.assertEqual(info["phase"], "design_evaluated")
        env.close()

    def test_design_change_affects_simulation(self):
        """Two distinct designs should produce distinct final observations.

        This is the plan §10 P3 verify clause: the .inp mutation
        round-trips through the engine and the simulation differs.
        """
        env = _make_cip_env(self.minimal_inp)

        env.reset(seed=0)
        low_design = {
            "link_roughness": np.array([0.005], dtype=np.float32),
            "node_max_depth": np.array([3.0], dtype=np.float32),
        }
        obs_low, reward_low, _, _, _ = env.step({"design": low_design, "runtime": {}})

        env.reset(seed=0)
        high_design = {
            "link_roughness": np.array([0.05], dtype=np.float32),
            "node_max_depth": np.array([8.0], dtype=np.float32),
        }
        obs_high, reward_high, _, _, _ = env.step({"design": high_design, "runtime": {}})

        # Different roughness drives different conduit flow rates, so
        # at least one observation component must differ.
        self.assertFalse(
            np.allclose(obs_low, obs_high, atol=1e-6),
            f"Expected design to change observation; got identical "
            f"obs_low={obs_low} obs_high={obs_high}",
        )
        env.close()


if __name__ == "__main__":
    unittest.main()
