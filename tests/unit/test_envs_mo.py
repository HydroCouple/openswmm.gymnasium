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

"""Unit tests for L{openswmm_gymnasium.envs.SwmmMORTCEnv}.

Construction + registration tests run anywhere. Episode tests require
the real engine.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmMORTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, ReliabilityMargin
from openswmm_gymnasium.spaces.runtime import OrificeSetting
from tests.unit._base import BaseEngineTest


def _mo_env(inp_path):
    return SwmmMORTCEnv(
        inp_path,
        runtime_factories=[OrificeSetting(["C1"])],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
        reward_terms=[
            FloodingVolume(node_ids=["J1"]),
            ReliabilityMargin(node_ids=["J1"]),
        ],
        ideal_point=[0.0, 0.0],
        reference_point=[10.0, 10.0],
    )


class TestConstruction(BaseEngineTest):
    def test_requires_ideal_and_reference(self):
        with self.assertRaisesRegex(ValueError, "ideal_point and reference_point"):
            SwmmMORTCEnv(
                self.minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
            )

    def test_requires_nonempty_reward_terms(self):
        with self.assertRaisesRegex(ValueError, "at least one reward term"):
            SwmmMORTCEnv(
                self.minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[],
                ideal_point=[0.0],
                reference_point=[1.0],
            )

    def test_ideal_reference_length_mismatch_raises(self):
        with self.assertRaisesRegex(ValueError, "lengths must match"):
            SwmmMORTCEnv(
                self.minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
                ideal_point=[0.0, 0.0],  # length 2 vs 1 term
                reference_point=[1.0, 1.0],
            )

    def test_ref_not_greater_than_ideal_raises(self):
        with self.assertRaisesRegex(ValueError, "strictly greater"):
            SwmmMORTCEnv(
                self.minimal_inp,
                runtime_factories=[OrificeSetting(["C1"])],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                reward_terms=[FloodingVolume(node_ids=["J1"])],
                ideal_point=[1.0],
                reference_point=[1.0],
            )

    def test_reward_space_shape(self):
        env = _mo_env(self.minimal_inp)
        self.assertIsInstance(env.reward_space, spaces.Box)
        self.assertEqual(env.reward_space.shape, (2,))
        env.close()


class TestRegistration(unittest.TestCase):
    def test_registered(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-MORTC-v0")
        self.assertEqual(spec.entry_point, "openswmm_gymnasium.envs:SwmmMORTCEnv")


# ---------------------------------------------------------------------------
# Integration tier
# ---------------------------------------------------------------------------


class TestEpisode(BaseEngineTest):
    def test_step_returns_vector_reward(self):
        env = _mo_env(self.minimal_inp)
        env.reset(seed=0)
        _, reward, _, _, _ = env.step(env.action_space.sample())
        self.assertIsInstance(reward, np.ndarray)
        self.assertEqual(reward.shape, (2,))
        self.assertEqual(reward.dtype, np.float32)
        env.close()

    def test_terminal_step_emits_mo_score(self):
        env = _mo_env(self.minimal_inp)
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
        self.assertIn("cumulative_cost", info)
        self.assertEqual(info["cumulative_cost"].shape, (2,))
        env.close()


if __name__ == "__main__":
    unittest.main()
