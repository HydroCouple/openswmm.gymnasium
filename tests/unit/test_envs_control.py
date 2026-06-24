"""Unit tests for L{openswmm_gymnasium.envs.SwmmControlEnv}.

Construction tier builds the env (no engine). The integration tier runs a full
episode under the reactive market controller on the twin-tank benchmark, whose
orifice C{ORIF} (T1->T2) is a real controllable structure, per the suite's
no-mocks policy.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest

import numpy as np
from tests.unit._base import BaseTestCase

from openswmm_gymnasium.benchmarks.b01_twin_tank import SCENARIO_INP
from openswmm_gymnasium.config import (
    Agent,
    CostCurve,
    MarketConfig,
    Meta,
    PIDConfig,
    TradeRoute,
)
from openswmm_gymnasium.envs import SwmmControlEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, UncontrolledDischarge


def _twin_tank_config() -> MarketConfig:
    """Market config wiring the twin-tank orifice as a buyer/seller route."""
    return MarketConfig(
        meta=Meta(
            model_path=str(SCENARIO_INP),
            output_dir="/tmp",
            control_interval_seconds=300,
        ),
        cost_curves={
            "buyer": CostCurve(type="logistic", onset=0.7, steepness=15),
            "seller": CostCurve(type="piecewise_linear", onset=0.3, full=1.0),
        },
        agents=[
            Agent("T1", "node", "conveyance", "storage_fill", "buyer", role="buyer"),
            Agent("T2", "node", "storage", "storage_fill", "seller", role="seller"),
        ],
        trade_routes=[
            TradeRoute(
                structure_link_id="ORIF",
                buyer_agents=["T1"],
                seller_agents=["T2"],
                pid=PIDConfig(kp=2.0, ki=0.1),
            )
        ],
    )


def _builder() -> ObservationBuilder:
    return ObservationBuilder().add_node_depths(["T1", "T2"])


class TestConstruction(unittest.TestCase):
    def test_action_space_matches_policy_space(self):
        env = SwmmControlEnv(
            SCENARIO_INP,
            market_config=_twin_tank_config(),
            observation_builder=_builder(),
        )
        # Default policy space: buyer onset/ceiling/steepness (logistic),
        # seller onset/ceiling, ORIF kp/ki/kd => 8 params.
        self.assertEqual(env.action_space.shape, (len(env.policy_space),))
        self.assertEqual(len(env.policy_space), 8)

    def test_reset_returns_zero_observation(self):
        env = SwmmControlEnv(
            SCENARIO_INP,
            market_config=_twin_tank_config(),
            observation_builder=_builder(),
        )
        obs, info = env.reset(seed=0)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.allclose(obs, 0.0))
        self.assertEqual(info["phase"], "awaiting_policy")


class TestIntegration(BaseTestCase):
    def _make_env(self) -> SwmmControlEnv:
        return SwmmControlEnv(
            SCENARIO_INP,
            market_config=_twin_tank_config(),
            observation_builder=_builder(),
            reward_terms=[FloodingVolume(node_ids=["T1"]), UncontrolledDischarge(["OUT"])],
            rpt_path=str(self.tmp_path / "b01.rpt"),
            out_path=str(self.tmp_path / "b01.out"),
        )

    def test_full_episode_under_market_control(self):
        env = self._make_env()
        env.reset(seed=0)
        action = (env.action_space.low + env.action_space.high) / 2.0
        obs, reward, terminated, truncated, info = env.step(action)
        env.close()

        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertEqual(obs.shape, env.observation_space.shape)
        self.assertTrue(np.isfinite(reward))
        self.assertEqual(
            set(info["reward_components"]), {"flooding_volume", "uncontrolled_discharge"}
        )
        for v in info["reward_components"].values():
            self.assertTrue(np.isfinite(v))
            # Both are minimize terms (non-negative); a negative value would mean
            # a negative dt leaked in from the engine's final-step clock reset.
            self.assertGreaterEqual(v, 0.0)
        self.assertGreaterEqual(info["n_control_updates"], 1)

    def test_control_actually_affects_outcome(self):
        # Locks in the target_setting fix: a policy that keeps ORIF shut (buyer
        # never triggers, tiny gain) must hold water back vs one that opens it
        # eagerly, so the untreated-outfall discharge differs materially.
        env = self._make_env()
        sp, lab = env.policy_space, env.policy_space.labels

        def run(overrides: dict[str, float]) -> float:
            vec = sp.flatten(env._base_config).copy()
            for key, val in overrides.items():
                vec[lab.index(key)] = val
            env.reset(seed=0)
            _, _, _, _, info = env.step(vec)
            return info["reward_components"]["uncontrolled_discharge"]

        shut = run({"curve:buyer.onset": 0.99, "pid:ORIF.kp": 0.01})
        open_ = run({"curve:buyer.onset": 0.05, "pid:ORIF.kp": 5.0})
        env.close()
        self.assertLess(shut, open_)

    def test_deterministic_for_same_policy(self):
        action = None
        results = []
        for _ in range(2):
            env = self._make_env()
            env.reset(seed=0)
            if action is None:
                action = (env.action_space.low + env.action_space.high) / 2.0
            _, reward, _, _, info = env.step(action)
            env.close()
            results.append((reward, info["reward_components"]))
        self.assertEqual(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
