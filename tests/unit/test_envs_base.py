"""Unit tests for L{openswmm_gymnasium.envs.SwmmRTCEnv}.

Per plan §8.2, the C{check_env} compliance test is part of the
integration tier (it drives a real reset/step cycle). Construction
tests live in the unit tier.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import pytest
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.runtime import OrificeSetting

# ---------------------------------------------------------------------------
# Construction (no engine open — these run anywhere)
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_requires_observation_builder(self, minimal_inp):
        with pytest.raises(ValueError, match="observation_builder"):
            SwmmRTCEnv(minimal_inp)

    def test_rejects_zero_control_interval(self, minimal_inp):
        with pytest.raises(ValueError, match="control_interval_steps"):
            SwmmRTCEnv(
                minimal_inp,
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
                control_interval_steps=0,
            )

    def test_action_space_structure(self, minimal_inp):
        """Action space is Dict({design: Dict({}), runtime: Dict(...)}).

        Plan §3 contract: every env exposes both top-level keys.
        """
        env = SwmmRTCEnv(
            minimal_inp,
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        assert isinstance(env.action_space, spaces.Dict)
        assert set(env.action_space.spaces.keys()) == {"design", "runtime"}
        # Design is empty for RTC-only envs.
        assert isinstance(env.action_space["design"], spaces.Dict)
        assert len(env.action_space["design"].spaces) == 0
        # Runtime has one entry for our single OrificeSetting factory.
        assert isinstance(env.action_space["runtime"], spaces.Dict)
        assert "orifice_setting" in env.action_space["runtime"].spaces

    def test_observation_space_matches_builder(self, minimal_inp):
        env = SwmmRTCEnv(
            minimal_inp,
            observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
        )
        assert isinstance(env.observation_space, spaces.Box)
        assert env.observation_space.shape == (2,)


class TestRegistration:
    def test_registered_id_resolves(self):
        """C{gymnasium.make("OpenSWMM/Minimal-RTC-v0", ...)} resolves.

        Doesn't call reset() so the engine handle isn't allocated; this
        runs in-sandbox.
        """
        # Importing the package triggers gym.register; the spec must be
        # findable from the registry.
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-RTC-v0")
        assert spec.entry_point == "openswmm_gymnasium.envs:SwmmRTCEnv"


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


@pytest.mark.integration
class TestEpisode:
    def test_reset_returns_obs_info(self, minimal_inp):
        env = _make_minimal_env(minimal_inp)
        obs, info = env.reset(seed=0)
        assert obs.shape == (2,)
        assert "elapsed_days" in info
        env.close()

    def test_step_returns_five_tuple(self, minimal_inp):
        env = _make_minimal_env(minimal_inp)
        env.reset(seed=0)
        action = env.action_space.sample()
        result = env.step(action)
        assert len(result) == 5
        obs, reward, terminated, truncated, info = result
        assert obs.shape == (2,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert "reward_components" in info
        assert "flooding_volume" in info["reward_components"]
        env.close()

    def test_episode_runs_to_termination(self, minimal_inp):
        env = _make_minimal_env(minimal_inp)
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
                pytest.fail("Episode did not terminate within safety limit")
        # Minimal fixture (30 min sim @ 15 s routing step, control
        # interval 1) gives ~120 env steps.
        assert 60 <= steps <= 500
        # No flooding on dry-channel fixture → reward should be ~0.
        assert cumulative == pytest.approx(0.0, abs=1e-6)
        env.close()

    def test_check_env_compliance(self, minimal_inp):
        """The env passes Gymnasium's L{check_env} contract suite."""
        from gymnasium.utils.env_checker import check_env

        env = _make_minimal_env(minimal_inp)
        # check_env runs reset / step probes and validates space conformance.
        check_env(env, skip_render_check=True)
        env.close()
