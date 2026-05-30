"""Unit tests for the scalarisation wrappers.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np
import pytest

from openswmm_gymnasium.wrappers import LinearScalarize, TchebycheffScalarize
from tests.unit._wrapper_helpers import VectorRewardEnv


class TestLinearScalarize:
    def test_dot_product(self):
        env = VectorRewardEnv()
        wrapped = LinearScalarize(env, weights=[1.0, 0.5, 0.25])
        _, reward, _, _, _ = wrapped.step(env.action_space.sample())
        # raw reward = [1, 2, 3], weights = [1, 0.5, 0.25]
        # scalar = 1*1 + 2*0.5 + 3*0.25 = 1 + 1 + 0.75 = 2.75
        assert reward == pytest.approx(2.75)

    def test_empty_weights_raises(self):
        with pytest.raises(ValueError, match="must not be empty"):
            LinearScalarize(VectorRewardEnv(), weights=[])

    def test_mismatched_shape_raises(self):
        env = VectorRewardEnv()
        wrapped = LinearScalarize(env, weights=[1.0, 1.0])  # wrong length (2 vs 3)
        wrapped.reset()
        with pytest.raises(ValueError, match="does not match"):
            wrapped.step(env.action_space.sample())


class TestTchebycheffScalarize:
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
        assert scalar == pytest.approx(0.0)

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
        assert scalar == pytest.approx(-3.0)

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
        assert scalar == pytest.approx(-10.0)

    def test_mismatched_shapes_raise(self):
        with pytest.raises(ValueError, match="matching"):
            TchebycheffScalarize(
                VectorRewardEnv(),
                weights=[1.0, 1.0],
                utopia=[1.0, 1.0, 1.0],
            )
