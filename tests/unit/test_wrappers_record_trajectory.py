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

"""Unit tests for L{openswmm_gymnasium.wrappers.RecordTrajectory}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
import unittest

import numpy as np

from openswmm_gymnasium.wrappers import RecordTrajectory
from tests.unit._base import BaseTestCase
from tests.unit._wrapper_helpers import FlatBoxEnv


def _read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


class TestFileLifecycle(BaseTestCase):
    def test_one_file_per_episode(self):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=2), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.reset(seed=1)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.close()
        files = sorted(self.tmp_path.glob("episode_*.jsonl"))
        self.assertEqual(
            [f.name for f in files],
            [
                "episode_00000.jsonl",
                "episode_00001.jsonl",
            ],
        )

    def test_record_structure(self):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=2), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.step(env.action_space.sample())
        env.close()
        records = _read_jsonl(self.tmp_path / "episode_00000.jsonl")
        self.assertEqual(records[0]["event"], "reset")
        self.assertEqual(records[0]["episode"], 0)
        self.assertIn("obs", records[0])
        self.assertEqual(records[1]["event"], "step")
        self.assertEqual(records[1]["t"], 1)
        self.assertIn("action", records[1])
        self.assertIn("reward", records[1])
        self.assertIsInstance(records[-1]["terminated"], bool)

    def test_terminated_closes_file(self):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        # File handle should be closed; episode_00000.jsonl exists.
        path = self.tmp_path / "episode_00000.jsonl"
        self.assertTrue(path.exists())
        # Re-opening for write should not raise.
        with open(path, encoding="utf-8") as f:
            f.read()


class TestNumpyHandling(BaseTestCase):
    def test_ndarray_serialised_as_list(self):
        env = RecordTrajectory(FlatBoxEnv(obs_size=2), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.close()
        records = _read_jsonl(self.tmp_path / "episode_00000.jsonl")
        self.assertIsInstance(records[0]["obs"], list)
        self.assertEqual(records[0]["obs"], [0.0, 0.0])

    def test_numpy_scalar_serialised(self):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.step(env.action_space.sample())
        env.close()
        records = _read_jsonl(self.tmp_path / "episode_00000.jsonl")
        # Step records carry a reward (float).
        self.assertIsInstance(records[-1]["reward"], (int, float))

    def test_unknown_type_fallbacks_to_repr(self):
        from openswmm_gymnasium.wrappers.record_trajectory import _jsonable

        class _Weird:
            def __repr__(self):
                return "<weird>"

        self.assertEqual(_jsonable(_Weird()), "<weird>")


class TestSerialisation(BaseTestCase):
    def test_numpy_action_serialised(self):
        env = RecordTrajectory(FlatBoxEnv(terminate_after=1), output_dir=self.tmp_path)
        env.reset(seed=0)
        env.step(np.array([0.42], dtype=np.float32))
        env.close()
        records = _read_jsonl(self.tmp_path / "episode_00000.jsonl")
        # Action should be a JSON list of floats, not a numpy repr.
        self.assertIsInstance(records[1]["action"], list)
        self.assertEqual(records[1]["action"], [pytest_approx_value()])


def pytest_approx_value():
    # Helper so the literal float matches across NumPy float32 rounding.
    return float(np.float32(0.42))


if __name__ == "__main__":
    unittest.main()
