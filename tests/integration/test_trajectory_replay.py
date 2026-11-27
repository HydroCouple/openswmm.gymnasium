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

"""A recorded trajectory round-trips and replays exactly (plan §8.2).

A full b01 TwinTank episode is recorded with L{RecordTrajectory}, read back
with L{Trajectory}, and its actions replayed in a fresh env: the replay must
reproduce every recorded observation, reward and the terminal step. This is
what makes a recorded JSONL a usable artifact (regression goldens, results
shipped back from a worker).

Real engine; the episode file is written to the reviewable
C{tests/_output/integration/trajectory/} (CLAUDE.md §4.1).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

import openswmm_gymnasium  # noqa: F401  (registers the env ids)
from openswmm_gymnasium.viz.trajectory import Trajectory
from openswmm_gymnasium.wrappers import RecordTrajectory

ENV_ID = "OpenSWMM/TwinTank-RTC-v0"
SEED = 5
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "integration" / "trajectory"


def _as_action(space: spaces.Space, value: Any) -> Any:
    """A recorded (JSON) action back in the action space's own types."""
    if isinstance(space, spaces.Dict):
        return {key: _as_action(sub, value[key]) for key, sub in space.spaces.items()}
    return np.asarray(value, dtype=space.dtype)


@pytest.fixture(scope="module")
def recorded() -> Trajectory:
    shutil.rmtree(_OUTPUT, ignore_errors=True)
    env = RecordTrajectory(gym.make(ENV_ID), _OUTPUT)
    try:
        env.reset(seed=SEED)
        env.action_space.seed(SEED)
        while True:
            *_, terminated, truncated, _ = env.step(env.action_space.sample())
            if terminated or truncated:
                break
    finally:
        env.close()
    return Trajectory.from_jsonl(_OUTPUT / "episode_00000.jsonl")


@pytest.mark.integration
def test_the_recorded_episode_reads_back(recorded):
    assert recorded.n_steps > 100
    assert recorded.observations.shape == (recorded.n_steps + 1, 6)  # reset + every step
    assert recorded.step_records[-1]["terminated"]
    assert set(recorded.reward_components()) == {
        "flooding_volume",
        "peak_outflow",
        "setpoint_smoothness",
    }


@pytest.mark.integration
def test_replaying_the_recorded_actions_reproduces_the_episode(recorded):
    env = gym.make(ENV_ID)
    try:
        obs, _ = env.reset(seed=SEED)
        observations, rewards, terminated = [obs], [], False
        for action in recorded.actions:
            obs, reward, terminated, truncated, _ = env.step(_as_action(env.action_space, action))
            observations.append(obs)
            rewards.append(reward)
    finally:
        env.close()
    assert terminated
    np.testing.assert_array_equal(np.asarray(observations, dtype=np.float64), recorded.observations)
    np.testing.assert_array_equal(np.asarray(rewards), recorded.rewards)
