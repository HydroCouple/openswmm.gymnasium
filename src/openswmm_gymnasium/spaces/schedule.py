"""Search space over an open-loop control schedule.

The decision vector is a flat list of C{[0,1]} settings — C{n_points} per
controllable structure — that L{SchedulePolicySpace.unflatten} reshapes into a
C{{structure_id: [setting_0, ...]}} schedule for a
L{openswmm_gymnasium.control.ScheduleController}. NSGA-II searches the vector;
each candidate is evaluated by one full simulation under the schedule.

Layout is structure-major: all C{n_points} settings of the first structure, then
the second, and so on — matching C{labels} order.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from gymnasium import spaces


class SchedulePolicySpace:
    """A bounded vector space over a per-structure setting schedule.

    @ivar structure_ids: Controllable link IDs, in actuation order.
    @ivar n_points: Number of scheduled settings per structure.
    """

    def __init__(
        self,
        structure_ids: Sequence[str],
        n_points: int,
        *,
        low: float = 0.0,
        high: float = 1.0,
    ) -> None:
        """
        @param structure_ids: Controllable link IDs (required, non-empty).
        @param n_points: Settings per structure (>= 1).
        @param low: Lower setting bound (default 0.0).
        @param high: Upper setting bound (default 1.0).
        @raise ValueError: On empty structures, C{n_points < 1}, or
            C{high <= low}.
        """
        if not structure_ids:
            raise ValueError("SchedulePolicySpace requires at least one structure_id")
        if n_points < 1:
            raise ValueError("n_points must be >= 1")
        if not high > low:
            raise ValueError(f"high ({high}) must be > low ({low})")
        self.structure_ids: list[str] = list(structure_ids)
        self.n_points = int(n_points)
        self._low_v = float(low)
        self._high_v = float(high)
        self.labels: list[str] = [
            f"sched:{sid}[{k}]" for sid in self.structure_ids for k in range(self.n_points)
        ]

    def __len__(self) -> int:
        return len(self.labels)

    @property
    def low(self) -> np.ndarray:
        """Lower bounds (float32, shape C{(n,)})."""
        return np.full((len(self),), self._low_v, dtype=np.float32)

    @property
    def high(self) -> np.ndarray:
        """Upper bounds (float32, shape C{(n,)})."""
        return np.full((len(self),), self._high_v, dtype=np.float32)

    @property
    def space(self) -> spaces.Box:
        """The C{gymnasium.spaces.Box} over the flat schedule vector."""
        return spaces.Box(low=self.low, high=self.high, shape=(len(self),), dtype=np.float32)

    def flatten(self, schedule: Mapping[str, Sequence[float]]) -> np.ndarray:
        """Flatten a schedule dict to a vector in this layout."""
        out: list[float] = []
        for sid in self.structure_ids:
            vals = list(schedule[sid])
            if len(vals) != self.n_points:
                raise ValueError(
                    f"schedule[{sid!r}] has {len(vals)} points, expected {self.n_points}"
                )
            out.extend(float(v) for v in vals)
        return np.asarray(out, dtype=np.float32)

    def unflatten(self, vector: np.ndarray) -> dict[str, list[float]]:
        """Reshape a decision *vector* into a C{{structure_id: [settings]}} dict.

        Values are clipped to C{[low, high]}.

        @raise ValueError: If C{vector} length mismatches the layout.
        """
        vec = np.asarray(vector, dtype=np.float64).ravel()
        if vec.shape[0] != len(self):
            raise ValueError(f"vector length {vec.shape[0]} != {len(self)}")
        vec = np.clip(vec, self._low_v, self._high_v)
        schedule: dict[str, list[float]] = {}
        for i, sid in enumerate(self.structure_ids):
            start = i * self.n_points
            schedule[sid] = [float(x) for x in vec[start : start + self.n_points]]
        return schedule
