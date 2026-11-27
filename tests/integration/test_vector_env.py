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

"""Vector envs give the trajectories a single env gives (plan §8.2).

C{SyncVectorEnv} steps its envs one after another in this process;
C{AsyncVectorEnv} runs each in its own worker process (C{spawn}, as on
macOS and Windows). Every env owns its engine handle (plan §2.3), so each
column of a vector run must match a single env stepped serially with the
same seed and actions, in both modes. The two columns get different
actions, so an env reading another's engine state would show.

Real engine on the b01 TwinTank scenario; no files are written (the
benchmark envs disable .rpt/.out).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.vector.utils import concatenate, create_empty_array

import openswmm_gymnasium  # noqa: F401  (registers the env ids)

ENV_ID = "OpenSWMM/TwinTank-RTC-v0"
SEEDS = (11, 12)
STEPS = 60  # well inside the 990-step episode: no autoreset


def _action_sequences(space: gym.Space) -> list[list]:
    """One seeded action sequence per env, different for each env."""
    sequences = []
    for seed in SEEDS:
        space.seed(seed)
        sequences.append([space.sample() for _ in range(STEPS)])
    return sequences


def _serial(seed: int, actions: list) -> tuple[np.ndarray, np.ndarray]:
    env = gym.make(ENV_ID)
    try:
        obs, _ = env.reset(seed=seed)
        observations, rewards = [obs], []
        for action in actions:
            obs, reward, terminated, truncated, _ = env.step(action)
            assert not (terminated or truncated)
            observations.append(obs)
            rewards.append(reward)
    finally:
        env.close()
    return np.asarray(observations), np.asarray(rewards)


def _vector(mode: str, sequences: list[list]) -> tuple[np.ndarray, np.ndarray]:
    kwargs = {"context": "spawn"} if mode == "async" else {}
    envs = gym.make_vec(ENV_ID, num_envs=len(SEEDS), vectorization_mode=mode, vector_kwargs=kwargs)
    try:
        single = envs.single_action_space
        batch = create_empty_array(single, n=len(SEEDS))
        obs, _ = envs.reset(seed=list(SEEDS))
        observations, rewards = [obs], []
        for step in range(STEPS):
            actions = concatenate(single, [seq[step] for seq in sequences], batch)
            obs, reward, terminated, truncated, _ = envs.step(actions)
            assert not (terminated.any() or truncated.any())
            observations.append(obs)
            rewards.append(reward)
    finally:
        envs.close()
    # (step, env, ...) -> (env, step, ...)
    return np.swapaxes(np.asarray(observations), 0, 1), np.asarray(rewards).T


@pytest.fixture(scope="module")
def sequences() -> list[list]:
    env = gym.make(ENV_ID)
    try:
        return _action_sequences(env.action_space)
    finally:
        env.close()


@pytest.fixture(scope="module")
def serial_runs(sequences) -> list[tuple[np.ndarray, np.ndarray]]:
    return [_serial(seed, seq) for seed, seq in zip(SEEDS, sequences, strict=True)]


@pytest.mark.integration
@pytest.mark.parametrize("mode", ["sync", "async"])
def test_vector_columns_match_serial_runs(mode, sequences, serial_runs):
    observations, rewards = _vector(mode, sequences)
    for env_index, (serial_obs, serial_rewards) in enumerate(serial_runs):
        np.testing.assert_array_equal(observations[env_index], serial_obs)
        np.testing.assert_array_equal(rewards[env_index], serial_rewards)


@pytest.mark.integration
def test_the_two_columns_saw_different_actions(serial_runs):
    # Guards the test above: identical columns would not show cross-talk.
    (obs_a, _), (obs_b, _) = serial_runs
    assert not np.array_equal(obs_a, obs_b)
