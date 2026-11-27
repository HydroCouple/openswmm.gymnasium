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

"""Integration verification for plan §10 P2.

This wires up the SwmmRTCEnv with B{five reward terms} and B{ten
observation features} against the minimal fixture, exercising every
P2 code path in one end-to-end episode.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import (
    CSOVolume,
    FloodingVolume,
    PeakOutflow,
    ReliabilityMargin,
    SetpointSmoothness,
)
from openswmm_gymnasium.spaces.runtime import OrificeSetting
from tests.unit._base import BaseEngineTest


def _ten_feature_observation_builder() -> ObservationBuilder:
    """Compose 10 observation features over the minimal fixture."""
    return (
        ObservationBuilder()
        .add_node_depths(["J1"])
        .add_node_heads(["J1"])
        .add_node_inflows(["J1"])
        .add_node_overflows(["J1"])
        .add_link_flows(["C1"])
        .add_link_depths(["C1"])
        .add_link_settings(["C1"])
        # minimal.inp has no subcatchments or rain gages, so we use
        # the clock collector to hit 10 features with three values.
        .add_clock()  # 3 features → 7+3 = 10
    )


def _five_reward_terms() -> list:
    """Five reward terms covering all P2 directions and shapes."""
    return [
        FloodingVolume(node_ids=["J1"]),
        CSOVolume(node_ids=["J1"]),
        PeakOutflow(link_ids=["C1"]),
        ReliabilityMargin(node_ids=["J1"]),
        SetpointSmoothness(link_ids=["C1"]),
    ]


class TestP2EndToEnd(BaseEngineTest):
    def test_observation_shape_is_ten(self):
        env = SwmmRTCEnv(
            self.minimal_inp,
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=_ten_feature_observation_builder(),
            reward_terms=_five_reward_terms(),
        )
        self.assertEqual(env.observation_space.shape, (10,))
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (10,))
        env.close()

    def test_five_reward_components_reported(self):
        env = SwmmRTCEnv(
            self.minimal_inp,
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=_ten_feature_observation_builder(),
            reward_terms=_five_reward_terms(),
        )
        env.reset(seed=0)
        _, _, _, _, info = env.step(env.action_space.sample())
        components = info["reward_components"]
        self.assertEqual(
            set(components.keys()),
            {
                "flooding_volume",
                "cso_volume",
                "peak_outflow",
                "reliability_margin",
                "setpoint_smoothness",
            },
        )
        env.close()

    def test_full_episode_runs_with_mixed_directions(self):
        env = SwmmRTCEnv(
            self.minimal_inp,
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=_ten_feature_observation_builder(),
            reward_terms=_five_reward_terms(),
        )
        env.reset(seed=0)
        steps = 0
        cumulative_components: dict[str, float] = {}
        while True:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            for k, v in info["reward_components"].items():
                cumulative_components[k] = cumulative_components.get(k, 0.0) + v
            steps += 1
            if terminated or truncated:
                break
            if steps > 10_000:
                self.fail("Episode did not terminate within safety limit")
        # minimal.inp is a dry-channel fixture → no flooding, no CSO.
        self.assertAlmostEqual(cumulative_components["flooding_volume"], 0.0, delta=1e-6)
        self.assertAlmostEqual(cumulative_components["cso_volume"], 0.0, delta=1e-6)
        # Reliability margin should accumulate positively (maximize term).
        self.assertGreater(cumulative_components["reliability_margin"], 0.0)
        # SetpointSmoothness should be > 0 (random actions churn).
        self.assertGreater(cumulative_components["setpoint_smoothness"], 0.0)
        env.close()


if __name__ == "__main__":
    unittest.main()
