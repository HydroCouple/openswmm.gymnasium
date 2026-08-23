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

"""Unit tests for the figure factories.

Each test verifies that the factory returns a L{plotly.graph_objects.Figure}
with the expected number of traces / layout. Empty-input edge cases
return an empty figure (no traces) without raising.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
import unittest

import plotly.graph_objects as go

from openswmm_gymnasium.viz import Trajectory, TrajectoryRun
from openswmm_gymnasium.viz.figures import (
    action_timeseries,
    flooding_attribution,
    hypervolume_trace,
    network_state_heatmap,
    objective_radar,
    pareto_front,
    reward_curves,
    trajectory_replay,
)

from tests.unit._base import BaseTestCase


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _scalar_episode(path, n_steps=5):
    recs = [{"event": "reset", "episode": 0, "obs": [0.0, 0.0], "info": {}}]
    for t in range(1, n_steps + 1):
        recs.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [float(t) / n_steps]}},
                "obs": [float(t), float(t) * 2],
                "reward": float(t),
                "terminated": t == n_steps,
                "truncated": False,
                "info": {"reward_components": {"flood": float(t), "energy": 0.5}},
            }
        )
    _write_jsonl(path, recs)


def _vector_episode(path, n_steps=5, base=1.0):
    recs = [{"event": "reset", "episode": 0, "obs": [0.0], "info": {}}]
    for t in range(1, n_steps + 1):
        recs.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t)],
                "reward": [base * float(t), float(t * 2)],
                "terminated": t == n_steps,
                "truncated": False,
                "info": {"reward_components": {"a": base * float(t), "b": float(t * 2)}},
            }
        )
    _write_jsonl(path, recs)


class TestRewardCurves(BaseTestCase):
    def test_scalar_returns_figure(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=20)
        fig = reward_curves(Trajectory.from_jsonl(p), rolling_window=5)
        self.assertIsInstance(fig, go.Figure)
        # reward + cumulative + rolling = 3 traces
        self.assertEqual(len(fig.data), 3)

    def test_vector_returns_figure(self):
        p = self.tmp_path / "ep.jsonl"
        _vector_episode(p, n_steps=4)
        fig = reward_curves(Trajectory.from_jsonl(p))
        # 2 objectives × (raw + cumulative) = 4 traces
        self.assertEqual(len(fig.data), 4)

    def test_empty_returns_empty_figure(self):
        empty = Trajectory(records=[{"event": "reset", "episode": 0, "obs": [0.0], "info": {}}])
        fig = reward_curves(empty)
        self.assertIsInstance(fig, go.Figure)
        self.assertEqual(len(fig.data), 0)


class TestParetoFront(BaseTestCase):
    def test_2d_returns_figure(self):
        for i in range(3):
            _scalar_episode(self.tmp_path / f"episode_0000{i}.jsonl", n_steps=2 + i)
        run = TrajectoryRun.from_dir(self.tmp_path)
        fig = pareto_front(run)
        self.assertIsInstance(fig, go.Figure)
        # 2 traces: dominated + Pareto front
        self.assertEqual(len(fig.data), 2)

    def test_unsupported_dimension_raises(self):
        _scalar_episode(self.tmp_path / "episode_00000.jsonl", n_steps=2)
        run = TrajectoryRun.from_dir(self.tmp_path)
        with self.assertRaisesRegex(ValueError, "2-D or 3-D"):
            pareto_front(run, dimensions=(0,))

    def test_empty_run_returns_empty_figure(self):
        run = TrajectoryRun.from_dir(self.tmp_path)
        fig = pareto_front(run)
        self.assertEqual(len(fig.data), 0)


class TestHypervolumeTrace(BaseTestCase):
    def test_returns_figure_with_values_in_unit_interval(self):
        for i in range(3):
            _scalar_episode(self.tmp_path / f"episode_0000{i}.jsonl", n_steps=2 + i)
        run = TrajectoryRun.from_dir(self.tmp_path)
        fig = hypervolume_trace(run, ideal=[0.0, 0.0], reference=[100.0, 100.0])
        self.assertEqual(len(fig.data), 1)
        # y values must be in [0, 1].
        y = fig.data[0].y
        self.assertTrue(all(0.0 <= v <= 1.0 for v in y))

    def test_mismatch_raises(self):
        _scalar_episode(self.tmp_path / "episode_00000.jsonl", n_steps=3)
        run = TrajectoryRun.from_dir(self.tmp_path)
        with self.assertRaisesRegex(ValueError, "does not match"):
            hypervolume_trace(run, ideal=[0.0], reference=[100.0])


class TestActionTimeseries(BaseTestCase):
    def test_returns_heatmap(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = action_timeseries(Trajectory.from_jsonl(p))
        self.assertEqual(len(fig.data), 1)
        # Heatmap is row × column; shape = (n_action_components, n_steps)
        self.assertEqual(fig.data[0].z.shape, (1, 4))


class TestNetworkStateHeatmap(BaseTestCase):
    def test_returns_heatmap(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = network_state_heatmap(Trajectory.from_jsonl(p))
        self.assertEqual(len(fig.data), 1)
        # Without reset row: (n_features=2, n_steps=4)
        self.assertEqual(fig.data[0].z.shape, (2, 4))

    def test_label_mismatch_raises(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=3)
        with self.assertRaisesRegex(ValueError, "does not match"):
            network_state_heatmap(Trajectory.from_jsonl(p), feature_labels=["only_one"])


class TestFloodingAttribution(BaseTestCase):
    def test_stacked_area_per_component(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=4)
        fig = flooding_attribution(Trajectory.from_jsonl(p))
        # Two components → two traces.
        self.assertEqual(len(fig.data), 2)


class TestObjectiveRadar(BaseTestCase):
    def test_one_trace_per_trajectory(self):
        _scalar_episode(self.tmp_path / "a.jsonl", n_steps=3)
        _scalar_episode(self.tmp_path / "b.jsonl", n_steps=5)
        trajs = [
            Trajectory.from_jsonl(self.tmp_path / "a.jsonl"),
            Trajectory.from_jsonl(self.tmp_path / "b.jsonl"),
        ]
        fig = objective_radar(trajs, labels=["policy A", "policy B"])
        self.assertEqual(len(fig.data), 2)

    def test_label_mismatch_raises(self):
        _scalar_episode(self.tmp_path / "a.jsonl", n_steps=2)
        trajs = [Trajectory.from_jsonl(self.tmp_path / "a.jsonl")]
        with self.assertRaisesRegex(ValueError, "does not match"):
            objective_radar(trajs, labels=["A", "B"])  # length 2 vs 1 traj


class TestTrajectoryReplay(BaseTestCase):
    def test_returns_animated_figure(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        fig = trajectory_replay(
            traj,
            coords={"J1": (0.0, 0.0), "J2": (1.0, 0.0)},
            obs_index={"J1": 0, "J2": 1},
        )
        # frames = reset + 3 steps = 4
        self.assertEqual(len(fig.frames), 4)

    def test_key_mismatch_raises(self):
        p = self.tmp_path / "ep.jsonl"
        _scalar_episode(p, n_steps=2)
        traj = Trajectory.from_jsonl(p)
        with self.assertRaisesRegex(ValueError, "same set"):
            trajectory_replay(
                traj,
                coords={"J1": (0.0, 0.0)},
                obs_index={"J2": 0},
            )


if __name__ == "__main__":
    unittest.main()
