"""Open-loop schedule controller for full-event optimal control.

Unlike the reactive L{openswmm_gymnasium.control.MarketController}, a
L{ScheduleController} ignores live state entirely: it replays a precomputed
per-structure setting schedule, one value per control step. NSGA-II searches the
schedule against the operational objectives, each candidate evaluated by one
full simulation (no horizon re-simulation / hotstart — that resume is not
faithful on this engine; see the plan's R3).

The schedule is consumed by call index: the k-th L{compute_settings} call returns
each structure's k-th scheduled setting, holding the final value once the
schedule is exhausted. This matches the env's fixed control cadence.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence


class ScheduleController:
    """Replay a per-structure setting schedule (open-loop).

    Implements L{openswmm_gymnasium.control.base.Controller}. Metrics are
    ignored — the schedule is a function of control-step index only.
    """

    def __init__(
        self,
        structure_ids: Sequence[str],
        schedule: Mapping[str, Sequence[float]],
    ) -> None:
        """
        @param structure_ids: Controllable link IDs, in actuation order.
        @type structure_ids: sequence of str
        @param schedule: C{{structure_id: [setting_0, setting_1, ...]}}; every
            structure in C{structure_ids} must appear. Settings are clamped to
            [0, 1] on read.
        @type schedule: mapping of str to sequence of float
        @raise ValueError: If C{structure_ids} is empty or a structure has no
            schedule entries.
        """
        if not structure_ids:
            raise ValueError("ScheduleController requires at least one structure_id")
        self._structure_ids: list[str] = list(structure_ids)
        self._schedule: dict[str, list[float]] = {}
        for sid in self._structure_ids:
            vals = list(schedule.get(sid, []))
            if not vals:
                raise ValueError(f"schedule for structure {sid!r} is empty")
            self._schedule[sid] = vals
        self._k = 0

    @property
    def structure_ids(self) -> list[str]:
        return list(self._structure_ids)

    def reset(self) -> None:
        self._k = 0

    def compute_settings(
        self, metrics: Mapping[str, float], dt_seconds: float
    ) -> dict[str, float]:
        """Return each structure's setting for the current control step.

        C{metrics} and C{dt_seconds} are ignored (open-loop). The final
        scheduled value holds once the schedule is exhausted.
        """
        out: dict[str, float] = {}
        for sid in self._structure_ids:
            vals = self._schedule[sid]
            idx = self._k if self._k < len(vals) else len(vals) - 1
            v = vals[idx]
            out[sid] = 0.0 if v < 0.0 else 1.0 if v > 1.0 else float(v)
        self._k += 1
        return out
