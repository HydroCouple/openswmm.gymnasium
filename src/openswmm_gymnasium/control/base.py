"""The C{Controller} protocol shared by every control strategy.

A controller is B{pure with respect to the engine}: it is handed the current
normalized stress metrics (one value in [0,1] per agent) and a timestep, and it
returns a setting in [0,1] for each structure it actuates. Reading raw state
from the solver and normalizing it into stress metrics, and applying the
returned settings, are the responsibility of the caller (the env, or the MPC
plant loop). Keeping controllers pure lets them be unit-tested without an
engine and lets the same NSGA-II job tune any strategy's parameters.

Two strategies implement this protocol:

  - L{openswmm_gymnasium.control.MarketController} — reactive agent-based
    capacity market (cost curves + per-route PID).
  - C{MPCController} — receding-horizon re-optimization (Phase 2).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Protocol, runtime_checkable


@runtime_checkable
class Controller(Protocol):
    """Interface every control strategy must satisfy.

    @cvar structure_ids: The structure (link) IDs this controller actuates, in
        a stable order. The caller applies the returned settings to these.
    """

    @property
    def structure_ids(self) -> list[str]:
        """Return the structure (link) IDs this controller actuates."""
        ...

    def reset(self) -> None:
        """Clear all cross-step state at the start of an episode."""
        ...

    def compute_settings(
        self, metrics: Mapping[str, float], dt_seconds: float
    ) -> dict[str, float]:
        """Return C{{structure_link_id: setting in [0,1]}} for one control step.

        @param metrics: C{{agent_id: stress_metric in [0,1]}} for the current
            step, as read and normalized by the caller.
        @type metrics: mapping of str to float
        @param dt_seconds: Control interval in seconds (advances any internal
            clock and integrators).
        @type dt_seconds: float
        @return: A setting in [0,1] for every ID in L{structure_ids}.
        @rtype: dict of str to float
        """
        ...
