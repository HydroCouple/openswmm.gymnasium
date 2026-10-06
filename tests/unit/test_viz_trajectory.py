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

"""Unit tests for L{openswmm_gymnasium.viz.trajectory}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from openswmm_gymnasium.viz.trajectory import Trajectory, TrajectoryRun

from tests.unit._base import BaseTestCase


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _make_scalar_episode(path, n_steps=3):
    records = [
        {"event": "reset", "episode": 0, "obs": [0.0, 0.0], "info": {}},
    ]
    for t in range(1, n_steps + 1):
        records.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t), float(t * 2)],
                "reward": float(t),
                "terminated": t == n_steps,
                "truncated": False,
                "info": {
                    "reward_components": {"flooding": float(t), "energy": 0.5},
                },
            }
        )
    _write_jsonl(path, records)


def _make_vector_episode(path, n_steps=3):
    records = [
        {"event": "reset", "episode": 0, "obs": [0.0], "info": {}},
    ]
    for t in range(1, n_steps + 1):
        records.append(
            {
                "event": "step",
                "t": t,
                "action": {"runtime": {"a": [0.5]}},
                "obs": [float(t)],
                "reward": [float(t), float(t * 2)],
                "terminated": t == n_steps,
                "truncated": False,
                "info": {
                    "reward_components": {"a": float(t), "b": float(t * 2)},
                },
            }
        )
    _write_jsonl(path, records)


class TestTrajectoryConstruction(BaseTestCase):
    def test_load_from_jsonl(self):
        p = self.tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        self.assertEqual(traj.n_steps, 3)
        self.assertEqual(traj.reset_record["event"], "reset")

    def test_missing_reset_raises(self):
        with self.assertRaisesRegex(ValueError, "reset event"):
            Trajectory(records=[{"event": "step"}])

    def test_empty_records_raises(self):
        with self.assertRaisesRegex(ValueError, "reset event"):
            Trajectory(records=[])


class TestTrajectoryArrays(BaseTestCase):
    def test_scalar_rewards(self):
        p = self.tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=4)
        traj = Trajectory.from_jsonl(p)
        np.testing.assert_array_equal(traj.rewards, [1.0, 2.0, 3.0, 4.0])
        self.assertEqual(traj.rewards.ndim, 1)

    def test_vector_rewards(self):
        p = self.tmp_path / "ep.jsonl"
        _make_vector_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        self.assertEqual(traj.rewards.shape, (3, 2))
        np.testing.assert_array_equal(traj.rewards[-1], [3.0, 6.0])

    def test_observations_include_reset(self):
        p = self.tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        self.assertEqual(traj.observations.shape, (4, 2))  # 3 steps + reset
        np.testing.assert_array_equal(traj.observations[0], [0.0, 0.0])

    def test_reward_components(self):
        p = self.tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        comps = traj.reward_components()
        self.assertEqual(set(comps.keys()), {"flooding", "energy"})
        np.testing.assert_array_equal(comps["flooding"], [1.0, 2.0, 3.0])
        np.testing.assert_array_equal(comps["energy"], [0.5, 0.5, 0.5])

    def test_cumulative_cost_vector(self):
        p = self.tmp_path / "ep.jsonl"
        _make_scalar_episode(p, n_steps=3)
        traj = Trajectory.from_jsonl(p)
        # sorted axes: ["energy", "flooding"]; sums = [1.5, 6.0]
        np.testing.assert_array_equal(traj.cumulative_cost_vector(), [1.5, 6.0])


class TestTrajectoryRun(BaseTestCase):
    def test_from_dir(self):
        _make_scalar_episode(self.tmp_path / "episode_00000.jsonl", n_steps=2)
        _make_scalar_episode(self.tmp_path / "episode_00001.jsonl", n_steps=3)
        run = TrajectoryRun.from_dir(self.tmp_path)
        self.assertEqual(run.n_episodes, 2)
        self.assertEqual(len(run), 2)

    def test_cumulative_cost_matrix(self):
        _make_scalar_episode(self.tmp_path / "episode_00000.jsonl", n_steps=2)  # f=3, e=1
        _make_scalar_episode(self.tmp_path / "episode_00001.jsonl", n_steps=3)  # f=6, e=1.5
        run = TrajectoryRun.from_dir(self.tmp_path)
        mat = run.cumulative_cost_matrix()
        self.assertEqual(mat.shape, (2, 2))
        # sorted axes: ["energy", "flooding"]
        np.testing.assert_array_equal(mat, [[1.0, 3.0], [1.5, 6.0]])

    def test_empty_dir(self):
        run = TrajectoryRun.from_dir(self.tmp_path)
        self.assertEqual(len(run), 0)
        self.assertEqual(run.cumulative_cost_matrix().shape, (0, 0))


if __name__ == "__main__":
    unittest.main()
