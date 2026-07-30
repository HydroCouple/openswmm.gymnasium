"""Unit tests for L{openswmm_gymnasium.envs.SwmmRTCEnv}.

Per plan §8.2, the C{check_env} compliance test is part of the
integration tier (it drives a real reset/step cycle). Construction
tests live in the unit tier.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import gymnasium as gym
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.runtime import OrificeSetting
from tests.unit._base import BaseEngineTest

# ---------------------------------------------------------------------------
# Construction (no engine open — these run anywhere)
# ---------------------------------------------------------------------------


class TestConstruction(BaseEngineTest):
    def test_requires_observation_builder(self):
        with self.assertRaisesRegex(ValueError, "observation_builder"):
            SwmmRTCEnv(self.minimal_inp)

    def test_rejects_zero_control_interval(self):
        with self.assertRaisesRegex(ValueError, "control_interval_steps"):
            SwmmRTCEnv(
                self.minimal_inp,
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                control_interval_steps=0,
            )

    def test_action_space_structure(self):
        """Action space is Dict({runtime: Dict(...)}) for an RTC-only env.

        Empty action halves are omitted because Gymnasium forbids empty
        Dict spaces (``check_env`` rejects them).
        """
        env = SwmmRTCEnv(
            self.minimal_inp,
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        self.assertIsInstance(env.action_space, spaces.Dict)
        self.assertEqual(set(env.action_space.spaces.keys()), {"runtime"})
        # Runtime has one entry for our single OrificeSetting factory.
        self.assertIsInstance(env.action_space["runtime"], spaces.Dict)
        self.assertIn("orifice_setting", env.action_space["runtime"].spaces)

    def test_observation_space_matches_builder(self):
        env = SwmmRTCEnv(
            self.minimal_inp,
            observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
        )
        self.assertIsInstance(env.observation_space, spaces.Box)
        self.assertEqual(env.observation_space.shape, (2,))


class TestRegistration(unittest.TestCase):
    def test_registered_id_resolves(self):
        """C{gymnasium.make("OpenSWMM/Minimal-RTC-v0", ...)} resolves.

        Doesn't call reset() so the engine handle isn't allocated; this
        runs in-sandbox.
        """
        # Importing the package triggers gym.register; the spec must be
        # findable from the registry.
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-RTC-v0")
        self.assertEqual(spec.entry_point, "openswmm_gymnasium.envs:SwmmRTCEnv")


# ---------------------------------------------------------------------------
# Integration tier — needs the real engine running
# ---------------------------------------------------------------------------


def _make_minimal_env(inp_path):
    """Helper to build the smallest functional env over the minimal fixture."""
    return SwmmRTCEnv(
        inp_path,
        runtime_factories=[OrificeSetting(["C1"])],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


class TestEpisode(BaseEngineTest):
    def test_reset_returns_obs_info(self):
        env = _make_minimal_env(self.minimal_inp)
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, (2,))
        self.assertIn("elapsed_days", info)
        env.close()

    def test_reset_rejects_unknown_options(self):
        env = _make_minimal_env(self.minimal_inp)
        with self.assertRaisesRegex(ValueError, "Unsupported reset options"):
            env.reset(seed=0, options={"bogus": 1})
        env.close()

    def test_reset_accepts_engine_options(self):
        # Empty mapping is a no-op; a real key (e.g. IGNORE_2D on a meshed
        # model) is exercised at the adapter level in test_observations_2d.
        env = _make_minimal_env(self.minimal_inp)
        obs, _ = env.reset(seed=0, options={"engine_options": {}})
        self.assertEqual(obs.shape, (2,))
        env.close()

    def test_step_returns_five_tuple(self):
        env = _make_minimal_env(self.minimal_inp)
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        self.assertEqual(len(result), 5)
        obs, reward, terminated, truncated, info = result
        self.assertEqual(obs.shape, (2,))
        self.assertIsInstance(reward, float)
        self.assertIsInstance(terminated, bool)
        self.assertIsInstance(truncated, bool)
        self.assertIn("reward_components", info)
        self.assertIn("flooding_volume", info["reward_components"])
        env.close()

    def test_episode_runs_to_termination(self):
        env = _make_minimal_env(self.minimal_inp)
        env.reset(seed=0)
        steps = 0
        cumulative = 0.0
        while True:
            action = env.action_space.sample()
            _, reward, terminated, truncated, _ = env.step(action)
            cumulative += reward
            steps += 1
            if terminated or truncated:
                break
            if steps > 10_000:
                self.fail("Episode did not terminate within safety limit")
        # Minimal fixture (30 min sim @ 15 s routing step, control
        # interval 1) gives ~120 env steps.
        self.assertTrue(60 <= steps <= 500)
        # No flooding on dry-channel fixture → reward should be ~0.
        self.assertAlmostEqual(cumulative, 0.0, delta=1e-6)
        env.close()

    def test_check_env_compliance(self):
        """The env passes Gymnasium's L{check_env} contract suite."""
        from gymnasium.utils.env_checker import check_env

        env = _make_minimal_env(self.minimal_inp)
        # check_env runs reset / step probes and validates space conformance.
        check_env(env, skip_render_check=True)
        env.close()


if __name__ == "__main__":
    unittest.main()
