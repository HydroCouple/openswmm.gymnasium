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

"""
Wrappers that convert vector reward into scalar reward.

Used to plug an L{openswmm_gymnasium.envs.SwmmMORTCEnv} into a
single-objective agent (DQN, PPO, SAC, ...) by collapsing the per-step
vector reward to a scalar.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

from collections.abc import Sequence

import gymnasium as gym
import numpy as np


class LinearScalarize(gym.RewardWrapper):
    """Scalar reward = C{dot(weights, vector_reward)}.

    Weights need not sum to one; the wrapper does not normalise. Each
    component is multiplied by its corresponding weight before
    summation.

    @ivar weights: 1-D weight vector, same length as the env's reward
        vector.
    """

    def __init__(self, env: gym.Env, weights: Sequence[float]) -> None:
        """
        @param env: Vector-reward env to wrap.
        @type env: L{gymnasium.Env}
        @param weights: Weight per objective.
        @type weights: sequence of float
        @raise ValueError: If C{weights} is empty.
        """
        super().__init__(env)
        if not len(weights):
            raise ValueError("weights must not be empty")
        self.weights = np.asarray(weights, dtype=np.float64)

    def reward(self, reward: np.ndarray) -> float:
        r = np.asarray(reward, dtype=np.float64)
        if r.shape != self.weights.shape:
            raise ValueError(
                f"reward shape {r.shape} does not match weights shape {self.weights.shape}"
            )
        return float(np.dot(self.weights, r))


class TchebycheffScalarize(gym.RewardWrapper):
    """Weighted-Tchebycheff scalarisation.

    For B{higher-is-better} vector reward (which is what the MO envs
    emit), this returns C{-max_d w_d * (utopia_d - r_d)}. The negation
    keeps the higher-is-better convention at the scalar level: a
    reward vector closer to the utopia point produces a smaller
    weighted gap and hence a larger (less negative) scalar.

    @ivar weights: Per-objective weights.
    @ivar utopia: Per-objective best-case reward (high values for
        higher-is-better envs).
    """

    def __init__(
        self,
        env: gym.Env,
        weights: Sequence[float],
        utopia: Sequence[float],
    ) -> None:
        """
        @param env: Vector-reward env to wrap.
        @type env: L{gymnasium.Env}
        @param weights: Weight per objective.
        @type weights: sequence of float
        @param utopia: Per-objective utopia (highest reachable reward).
        @type utopia: sequence of float
        @raise ValueError: If C{weights} and C{utopia} differ in shape
            or are empty.
        """
        super().__init__(env)
        if not len(weights) or len(weights) != len(utopia):
            raise ValueError("weights and utopia must have matching non-zero length")
        self.weights = np.asarray(weights, dtype=np.float64)
        self.utopia = np.asarray(utopia, dtype=np.float64)

    def reward(self, reward: np.ndarray) -> float:
        r = np.asarray(reward, dtype=np.float64)
        if r.shape != self.weights.shape:
            raise ValueError(
                f"reward shape {r.shape} does not match weights shape {self.weights.shape}"
            )
        gap = self.weights * np.maximum(0.0, self.utopia - r)
        return float(-gap.max())
