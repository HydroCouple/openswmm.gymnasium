"""Unit tests for L{openswmm_gymnasium.envs.SwmmJointCIPRTCEnv}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from openswmm_gymnasium.envs import SwmmJointCIPRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.spaces.design import LinkRoughness
from openswmm_gymnasium.spaces.runtime import OrificeSetting


class TestConstruction:
    def test_requires_design_factories(self, minimal_inp):
        with pytest.raises(ValueError, match="design_factories"):
            SwmmJointCIPRTCEnv(
                minimal_inp,
                design_factories=[],
                observation_builder=ObservationBuilder().add_node_depths(["J1"]),
            )

    def test_requires_observation_builder(self, minimal_inp):
        with pytest.raises(ValueError, match="observation_builder"):
            SwmmJointCIPRTCEnv(
                minimal_inp,
                design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            )

    def test_action_space_structure(self, minimal_inp):
        env = SwmmJointCIPRTCEnv(
            minimal_inp,
            design_factories=[LinkRoughness(["C1"], 0.01, 0.05)],
            runtime_factories=[OrificeSetting(["C1"])],
            observation_builder=ObservationBuilder().add_node_depths(["J1"]),
        )
        assert isinstance(env.action_space, spaces.Dict)
        assert set(env.action_space.spaces.keys()) == {"design", "runtime"}
        assert "link_roughness" in env.action_space["design"].spaces
        assert "orifice_setting" in env.action_space["runtime"].spaces


class TestRegistration:
    def test_registered_id_resolves(self):
        import openswmm_gymnasium  # noqa: F401

        spec = gym.spec("OpenSWMM/Minimal-Joint-v0")
        assert spec.entry_point == "openswmm_gymnasium.envs:SwmmJointCIPRTCEnv"


# ---------------------------------------------------------------------------
# Integration tier — real engine required
# ---------------------------------------------------------------------------


def _make_joint_env(inp_path):
    return SwmmJointCIPRTCEnv(
        inp_path,
        design_factories=[LinkRoughness(["C1"], 0.005, 0.05)],
        runtime_factories=[OrificeSetting(["C1"])],
        observation_builder=ObservationBuilder().add_node_depths(["J1"]).add_link_flows(["C1"]),
    )


@pytest.mark.integration
class TestEpisode:
    def test_reset_applies_supplied_design(self, minimal_inp):
        env = _make_joint_env(minimal_inp)
        chosen = {"link_roughness": np.array([0.03], dtype=np.float32)}
        _, info = env.reset(seed=0, options={"design_action": chosen})
        np.testing.assert_array_equal(
            info["design_action"]["link_roughness"], chosen["link_roughness"]
        )
        env.close()

    def test_reset_samples_design_when_not_supplied(self, minimal_inp):
        env = _make_joint_env(minimal_inp)
        _, info = env.reset(seed=0)
        # Sampled design is recorded in info; only check it's the right
        # shape + within bounds.
        d = info["design_action"]["link_roughness"]
        assert d.shape == (1,)
        assert 0.005 <= float(d[0]) <= 0.05
        env.close()

    def test_episode_runs_to_termination(self, minimal_inp):
        env = _make_joint_env(minimal_inp)
        env.reset(seed=0)
        steps = 0
        while True:
            _, _, terminated, truncated, _ = env.step(env.action_space.sample())
            steps += 1
            if terminated or truncated:
                break
            if steps > 10_000:
                pytest.fail("Episode did not terminate within safety limit")
        assert 60 <= steps <= 500
        env.close()

    def test_check_env_compliance(self, minimal_inp):
        from gymnasium.utils.env_checker import check_env

        env = _make_joint_env(minimal_inp)
        check_env(env, skip_render_check=True)
        env.close()

    def test_design_change_affects_trajectory(self, minimal_inp):
        """Two distinct designs at reset produce distinct cumulative reward.

        Verifies the CIP mutation propagates into the RTC episode
        (plan §10 P3 verify).
        """
        env = _make_joint_env(minimal_inp)

        # Deterministic runtime: open the orifice fully every step.
        def _run(design_value):
            env.reset(
                seed=0,
                options={
                    "design_action": {
                        "link_roughness": np.array([design_value], dtype=np.float32),
                    }
                },
            )
            cumulative = 0.0
            action = {
                "design": {"link_roughness": np.array([design_value], dtype=np.float32)},
                "runtime": {"orifice_setting": np.array([1.0], dtype=np.float32)},
            }
            while True:
                _, r, terminated, truncated, _ = env.step(action)
                cumulative += r
                if terminated or truncated:
                    break
            return cumulative

        r_low = _run(0.005)
        r_high = _run(0.05)
        assert r_low != r_high, (
            f"Expected design to change cumulative reward; got r_low={r_low} r_high={r_high}"
        )
        env.close()
