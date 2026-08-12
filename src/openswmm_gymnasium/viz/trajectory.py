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
Loaders for the JSONL trajectory files written by
L{openswmm_gymnasium.wrappers.RecordTrajectory}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np


def _read_jsonl(path: str | os.PathLike) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


class Trajectory:
    """One episode loaded from a single JSONL file.

    @ivar reset_record: The first C{"reset"} record from the file.
    @type reset_record: dict
    @ivar step_records: All C{"step"} records, in order.
    @type step_records: list[dict]
    """

    def __init__(self, records: Sequence[dict[str, Any]]) -> None:
        """
        @param records: Parsed JSONL records, in file order.
        @type records: sequence of dict
        @raise ValueError: If the first record is not a C{"reset"} event.
        """
        recs = list(records)
        if not recs or recs[0].get("event") != "reset":
            raise ValueError("Trajectory expects the first record to be a reset event")
        self.reset_record = recs[0]
        self.step_records = [r for r in recs[1:] if r.get("event") == "step"]

    @classmethod
    def from_jsonl(cls, path: str | os.PathLike) -> Trajectory:
        """Load one trajectory from a JSONL file."""
        return cls(_read_jsonl(path))

    # ------------------------------------------------------------------
    # Array views
    # ------------------------------------------------------------------

    @property
    def n_steps(self) -> int:
        """@rtype: int"""
        return len(self.step_records)

    @property
    def rewards(self) -> np.ndarray:
        """Per-step rewards. Shape C{(n_steps,)} or C{(n_steps, n_obj)}.

        @rtype: numpy.ndarray
        """
        if not self.step_records:
            return np.empty(0, dtype=np.float64)
        first = self.step_records[0]["reward"]
        if isinstance(first, list):
            return np.array([r["reward"] for r in self.step_records], dtype=np.float64)
        return np.array([float(r["reward"]) for r in self.step_records], dtype=np.float64)

    @property
    def observations(self) -> np.ndarray:
        """Stacked observations. Shape C{(n_steps + 1, obs_dim)}.

        @rtype: numpy.ndarray
        """
        obses: list[list[float]] = [self.reset_record["obs"]]
        for r in self.step_records:
            obses.append(r["obs"])
        return np.asarray(obses, dtype=np.float64)

    @property
    def actions(self) -> list[Any]:
        """List of per-step actions, in their original (possibly nested) form.

        @rtype: list
        """
        return [r["action"] for r in self.step_records]

    def reward_components(self) -> dict[str, np.ndarray]:
        """Per-step reward-component values keyed by term name.

        Reads C{info["reward_components"]} from each step record. Steps
        missing the key contribute C{0.0} for that step.

        @rtype: dict[str, numpy.ndarray]
        """
        names: set[str] = set()
        for r in self.step_records:
            comps = r.get("info", {}).get("reward_components", {})
            names.update(comps.keys())
        out: dict[str, np.ndarray] = {}
        for name in sorted(names):
            out[name] = np.array(
                [
                    float(r.get("info", {}).get("reward_components", {}).get(name, 0.0))
                    for r in self.step_records
                ],
                dtype=np.float64,
            )
        return out

    @property
    def cumulative_reward(self) -> np.ndarray:
        """Cumulative reward per step. Same shape as L{rewards}.

        @rtype: numpy.ndarray
        """
        r = self.rewards
        if r.ndim == 1:
            return np.cumsum(r)
        return np.cumsum(r, axis=0)

    def cumulative_cost_vector(self) -> np.ndarray:
        """Per-objective episode cost vector.

        For each term name in C{info["reward_components"]}, sums the
        per-step contributions. Useful for HV scoring and Pareto-front
        plots.

        @rtype: numpy.ndarray of shape C{(n_terms,)}
        """
        comps = self.reward_components()
        return np.array([comps[k].sum() for k in sorted(comps)], dtype=np.float64)


class TrajectoryRun:
    """A collection of L{Trajectory} objects from one experiment directory.

    @ivar trajectories: Loaded trajectories, in episode order.
    @type trajectories: list[Trajectory]
    """

    def __init__(self, trajectories: Sequence[Trajectory]) -> None:
        """
        @param trajectories: Trajectories to bundle.
        @type trajectories: sequence of L{Trajectory}
        """
        self.trajectories: list[Trajectory] = list(trajectories)

    @classmethod
    def from_dir(cls, path: str | os.PathLike, pattern: str = "episode_*.jsonl") -> TrajectoryRun:
        """Load all JSONL files matching C{pattern} from a directory.

        @param path: Directory containing JSONL files.
        @type path: str or C{os.PathLike}
        @param pattern: Glob pattern. Defaults to C{"episode_*.jsonl"}
            (the L{openswmm_gymnasium.wrappers.RecordTrajectory} default).
        @type pattern: str
        @rtype: L{TrajectoryRun}
        """
        p = Path(path)
        files = sorted(p.glob(pattern))
        return cls([Trajectory.from_jsonl(f) for f in files])

    def __len__(self) -> int:
        return len(self.trajectories)

    @property
    def n_episodes(self) -> int:
        return len(self.trajectories)

    def cumulative_cost_matrix(self) -> np.ndarray:
        """Stack each trajectory's cumulative-cost vector into a matrix.

        @return: Array of shape C{(n_episodes, n_terms)}. If trajectories
            have heterogeneous term sets, the matrix uses the union of
            all term names (sorted), with C{0.0} for missing entries.
        @rtype: numpy.ndarray
        """
        if not self.trajectories:
            return np.empty((0, 0), dtype=np.float64)
        all_names: set[str] = set()
        per_traj_comps: list[dict[str, np.ndarray]] = []
        for t in self.trajectories:
            comps = t.reward_components()
            per_traj_comps.append(comps)
            all_names.update(comps.keys())
        names = sorted(all_names)
        mat = np.zeros((len(self.trajectories), len(names)), dtype=np.float64)
        for i, comps in enumerate(per_traj_comps):
            for j, n in enumerate(names):
                if n in comps:
                    mat[i, j] = float(comps[n].sum())
        return mat
