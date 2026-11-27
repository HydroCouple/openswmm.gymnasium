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

"""Unit tests for the scalarisation wrappers.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest

import numpy as np

from openswmm_gymnasium.wrappers import LinearScalarize, TchebycheffScalarize
from tests.unit._wrapper_helpers import VectorRewardEnv


class TestLinearScalarize(unittest.TestCase):
    def test_dot_product(self):
        env = VectorRewardEnv()
        wrapped = LinearScalarize(env, weights=[1.0, 0.5, 0.25])
        _, reward, _, _, _ = wrapped.step(env.action_space.sample())
        # raw reward = [1, 2, 3], weights = [1, 0.5, 0.25]
        # scalar = 1*1 + 2*0.5 + 3*0.25 = 1 + 1 + 0.75 = 2.75
        self.assertAlmostEqual(reward, 2.75, places=6)

    def test_empty_weights_raises(self):
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            LinearScalarize(VectorRewardEnv(), weights=[])

    def test_mismatched_shape_raises(self):
        env = VectorRewardEnv()
        wrapped = LinearScalarize(env, weights=[1.0, 1.0])  # wrong length (2 vs 3)
        wrapped.reset()
        with self.assertRaisesRegex(ValueError, "does not match"):
            wrapped.step(env.action_space.sample())


class TestTchebycheffScalarize(unittest.TestCase):
    def test_at_utopia_zero(self):
        """Reward = utopia → no gap, scalar = 0."""
        env = VectorRewardEnv()
        env._next_reward = np.array([1.0, 2.0, 3.0], dtype=np.float32)
        wrapped = TchebycheffScalarize(
            env,
            weights=[1.0, 1.0, 1.0],
            utopia=[1.0, 2.0, 3.0],
        )
        _, scalar, _, _, _ = wrapped.step(env.action_space.sample())
        self.assertAlmostEqual(scalar, 0.0, places=6)

    def test_far_from_utopia_negative(self):
        env = VectorRewardEnv()
        env._next_reward = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        wrapped = TchebycheffScalarize(
            env,
            weights=[1.0, 1.0, 1.0],
            utopia=[1.0, 2.0, 3.0],
        )
        _, scalar, _, _, _ = wrapped.step(env.action_space.sample())
        # Per-dim gap: [1, 2, 3]; weighted: same; max = 3; negated = -3
        self.assertAlmostEqual(scalar, -3.0, places=6)

    def test_weighted_max_picks_dominant_axis(self):
        env = VectorRewardEnv()
        env._next_reward = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        # Bias weights so axis 0 dominates.
        wrapped = TchebycheffScalarize(
            env,
            weights=[10.0, 1.0, 1.0],
            utopia=[1.0, 2.0, 3.0],
        )
        _, scalar, _, _, _ = wrapped.step(env.action_space.sample())
        # Per-dim gap: [1, 2, 3]; weighted: [10, 2, 3]; max = 10; negated = -10
        self.assertAlmostEqual(scalar, -10.0, places=6)

    def test_mismatched_shapes_raise(self):
        with self.assertRaisesRegex(ValueError, "matching"):
            TchebycheffScalarize(
                VectorRewardEnv(),
                weights=[1.0, 1.0],
                utopia=[1.0, 1.0, 1.0],
            )


if __name__ == "__main__":
    unittest.main()
