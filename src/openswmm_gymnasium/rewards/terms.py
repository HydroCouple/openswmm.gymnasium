"""
Built-in reward terms.

Each term is a small class implementing the L{RewardTerm} interface:

  - C{name} — unique key under which the term's per-step contribution
    appears in C{info["reward_components"]}.
  - C{direction} — C{"minimize"} for costs (the default), C{"maximize"}
    for benefits like infiltration volume or freeboard.
  - C{bind(adapter)} — called once at the start of each episode to
    resolve symbolic IDs against the freshly-opened engine.
  - C{reset()} — called at the start of each episode to clear any
    cross-step accumulators inside the term.
  - C{step(adapter, dt_seconds)} — called every env C{step()}, returns
    the per-step contribution (B{positive} for both C{"minimize"} and
    C{"maximize"} terms; the env handles sign flipping per plan §0 #5).

Ships eight first-class terms:

  - L{FloodingVolume} (P1) — flooding rate × dt, summed across nodes.
  - L{CSOVolume} — same math restricted to tagged overflow nodes.
  - L{PeakOutflow} — running-max increments on link flow(s); cumulative
    equals peak observed.
  - L{ReliabilityMargin} — minimum freeboard (C{max_depth - depth}) at
    each step. Direction: maximize.
  - L{SetpointSmoothness} — L2 norm of Δsetting between successive
    steps; cumulative measures total action churn.
  - L{UncontrolledDischarge} — discharge through links feeding untreated
    outfalls (positive flow × dt), summed. (Operational objective #2.)
  - L{StorageUnderUtilization} — time-integrated unused storage headroom
    (C{1 - depth/max_depth}) across storage nodes. (Objective #3.)
  - L{PumpEnergy} — pump effort (control setting × rated power × dt)
    summed across pumps. (Objective #4.)

The remaining plan §5.1 terms (C{TSSLoad}, C{CapitalCost},
C{OandMCost}) land in subsequent phases.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from openswmm_gymnasium._engine import SolverAdapter


@runtime_checkable
class RewardTerm(Protocol):
    """Interface every reward term must satisfy.

    @cvar name: Unique short identifier for the term.
    @cvar direction: C{"minimize"} or C{"maximize"}.
    """

    name: str
    direction: str

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve symbolic IDs against the freshly-opened engine.

        @param adapter: Adapter wrapping the open solver.
        @type adapter: L{SolverAdapter}
        """
        ...

    def reset(self) -> None:
        """Clear any cross-step accumulators inside the term."""
        ...

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        """Compute this term's contribution for one env C{step()}.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @param dt_seconds: Elapsed real-time of the step, in seconds.
        @type dt_seconds: float
        @return: Non-negative contribution. Sign flipping for the agent
            happens in the env, not in the term.
        @rtype: float
        """
        ...


# =============================================================================
# Flooding / overflow accounting
# =============================================================================


class FloodingVolume:
    """Sum of overflow volume across a set of nodes, per env step.

    Computed as C{sum_i overflow_rate_i * dt_seconds}, where
    C{overflow_rate} is read from L{openswmm.engine.Nodes.get_overflow}
    (project flow units). If C{node_ids} is C{None}, all nodes in the
    model contribute.

    @ivar name: C{"flooding_volume"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        node_ids: Sequence[str] | None = None,
        name: str = "flooding_volume",
    ) -> None:
        """
        @param node_ids: Nodes to sum over, or C{None} for all nodes.
        @type node_ids: sequence of str or C{None}
        @param name: Term identifier used in C{info["reward_components"]}.
        @type name: str
        """
        self.name = name
        self._node_ids: list[str] | None = list(node_ids) if node_ids is not None else None
        self._idxs: list[int] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        if self._node_ids is None:
            self._idxs = list(range(adapter.nodes.count()))
        else:
            self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None, "bind() before step()"
        rate_sum = 0.0
        get = adapter.nodes.get_overflow
        for idx in self._idxs:
            rate_sum += float(get(idx))
        if rate_sum < 0.0:
            rate_sum = 0.0
        return rate_sum * dt_seconds


class CSOVolume(FloodingVolume):
    """Overflow volume restricted to a set of CSO / relief nodes.

    Mechanically identical to L{FloodingVolume} but defaults the term
    name to C{"cso_volume"} and B{requires} an explicit C{node_ids}
    list — running over all nodes would conflate CSO with general
    flooding.
    """

    def __init__(
        self,
        node_ids: Sequence[str],
        name: str = "cso_volume",
    ) -> None:
        """
        @param node_ids: Tagged CSO / overflow node IDs.
        @type node_ids: sequence of str (required)
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{node_ids} is empty.
        """
        if not node_ids:
            raise ValueError("CSOVolume requires at least one node_id")
        super().__init__(node_ids=node_ids, name=name)


# =============================================================================
# Peak outflow tracker
# =============================================================================


class PeakOutflow:
    """Running-max link flow; per-step contribution is the new-peak increment.

    Each step, for each tracked link, computes
    C{max(0, |flow_now| - peak_so_far)}. When a new peak is set the
    increment equals the delta; otherwise the contribution is zero.
    Cumulative reward over the episode therefore equals the maximum
    absolute flow observed at each link (summed across links).

    Plan §5.1 framing: minimise peak outflow.

    @ivar name: C{"peak_outflow"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        link_ids: Sequence[str],
        name: str = "peak_outflow",
    ) -> None:
        """
        @param link_ids: Links whose flow magnitudes to track.
        @type link_ids: sequence of str (required)
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{link_ids} is empty.
        """
        if not link_ids:
            raise ValueError("PeakOutflow requires at least one link_id")
        self.name = name
        self._link_ids: list[str] = list(link_ids)
        self._idxs: list[int] | None = None
        self._peak: list[float] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def reset(self) -> None:
        self._peak = [0.0] * len(self._link_ids)

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None and self._peak is not None, "bind() before step()"
        increment = 0.0
        get = adapter.links.get_flow
        for i, idx in enumerate(self._idxs):
            mag = abs(float(get(idx)))
            if mag > self._peak[i]:
                increment += mag - self._peak[i]
                self._peak[i] = mag
        return increment


# =============================================================================
# Reliability margin (maximise)
# =============================================================================


class ReliabilityMargin:
    """Per-step minimum freeboard across a set of nodes.

    Freeboard at a node = C{max_depth - depth}. The term returns the
    minimum across the tracked set, clamped at zero (surcharged nodes
    yield 0 rather than a negative penalty — the L{FloodingVolume}
    term already accounts for overflow). Direction: B{maximize}.

    @ivar name: C{"reliability_margin"} by default.
    @ivar direction: Always C{"maximize"}.
    """

    direction = "maximize"

    def __init__(
        self,
        node_ids: Sequence[str] | None = None,
        name: str = "reliability_margin",
    ) -> None:
        """
        @param node_ids: Nodes to evaluate, or C{None} for all nodes.
        @type node_ids: sequence of str or C{None}
        @param name: Term identifier.
        @type name: str
        """
        self.name = name
        self._node_ids: list[str] | None = list(node_ids) if node_ids is not None else None
        self._idxs: list[int] | None = None
        self._max_depths: list[float] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        if self._node_ids is None:
            self._idxs = list(range(adapter.nodes.count()))
        else:
            self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]
        # max_depth is a model-constant, cache once.
        self._max_depths = [float(adapter.nodes.get_max_depth(i)) for i in self._idxs]

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None and self._max_depths is not None, "bind() before step()"
        get = adapter.nodes.get_depth
        min_fb = float("inf")
        for idx, md in zip(self._idxs, self._max_depths, strict=True):
            fb = md - float(get(idx))
            if fb < min_fb:
                min_fb = fb
        return max(0.0, min_fb)


# =============================================================================
# Setpoint smoothness (minimise action churn)
# =============================================================================


class SetpointSmoothness:
    """L2 norm-squared of Δsetting between successive env steps.

    Each step, for each tracked link, reads the current control
    setting and adds C{(setting_now - setting_prev) ** 2} to the
    contribution. The first step after L{reset} returns zero (no
    previous setting yet). Direction: B{minimize}.

    @ivar name: C{"setpoint_smoothness"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        link_ids: Sequence[str],
        name: str = "setpoint_smoothness",
    ) -> None:
        """
        @param link_ids: Links whose setting trajectories to penalise.
        @type link_ids: sequence of str (required)
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{link_ids} is empty.
        """
        if not link_ids:
            raise ValueError("SetpointSmoothness requires at least one link_id")
        self.name = name
        self._link_ids: list[str] = list(link_ids)
        self._idxs: list[int] | None = None
        self._prev: list[float] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def reset(self) -> None:
        self._prev = None

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None, "bind() before step()"
        get = adapter.links.get_control_setting
        current = [float(get(i)) for i in self._idxs]
        if self._prev is None:
            self._prev = current
            return 0.0
        churn = 0.0
        for c, p in zip(current, self._prev, strict=True):
            d = c - p
            churn += d * d
        self._prev = current
        return churn


# =============================================================================
# Operational objectives (capacity-market / RTC tuning)
# =============================================================================


class UncontrolledDischarge:
    """Discharge through links feeding B{untreated} outfalls, per env step.

    Computed as C{sum_i max(0, flow_i) * dt_seconds} over the supplied link IDs
    — the conduits / outlets that discharge to outfalls B{not} carrying the
    treatment tag. Only positive flow (toward the outfall, the conventional
    C{FromNode -> outfall} orientation) counts as discharge; backflow into the
    system is ignored. At a FREE outfall the engine reports node inflow as 0, so
    the connecting B{link} flow — not the node — is the discharge signal. The
    caller resolves which links feed untreated outfalls (from the model tags /
    the market config C{treatment_tag}), mirroring L{CSOVolume}.

    Operational objective #2 (minimise uncontrolled discharge).

    @ivar name: C{"uncontrolled_discharge"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        link_ids: Sequence[str],
        name: str = "uncontrolled_discharge",
    ) -> None:
        """
        @param link_ids: Links discharging to untreated outfalls (required,
            non-empty), oriented toward the outfall.
        @type link_ids: sequence of str
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{link_ids} is empty.
        """
        if not link_ids:
            raise ValueError("UncontrolledDischarge requires at least one link_id")
        self.name = name
        self._link_ids: list[str] = list(link_ids)
        self._idxs: list[int] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None, "bind() before step()"
        get = adapter.links.get_flow
        rate_sum = 0.0
        for idx in self._idxs:
            q = float(get(idx))
            if q > 0.0:
                rate_sum += q
        return rate_sum * dt_seconds


class StorageUnderUtilization:
    """Time-integrated unused storage headroom across storage nodes.

    Each step contributes C{sum_i max(0, 1 - depth_i / max_depth_i) * dt},
    i.e. the fraction of each storage node left empty, integrated over time.
    It is high when storage sits idle and falls to zero as nodes fill, so
    minimising it rewards using available storage. Depth fill
    (C{depth / max_depth}) is used as a volume proxy, exact for a prismatic
    (linear) storage curve and a close approximation otherwise.

    Operational objective #3 (minimise storage under-utilisation).

    NOTE: integrated over the whole episode this also charges idle storage in
    dry weather; weight it accordingly in the objective vector (the example
    market config uses 0.5), or restrict C{node_ids} to event-relevant basins.

    @ivar name: C{"storage_underutilization"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        node_ids: Sequence[str],
        name: str = "storage_underutilization",
    ) -> None:
        """
        @param node_ids: Storage node IDs (required, non-empty).
        @type node_ids: sequence of str
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{node_ids} is empty.
        """
        if not node_ids:
            raise ValueError("StorageUnderUtilization requires at least one node_id")
        self.name = name
        self._node_ids: list[str] = list(node_ids)
        self._idxs: list[int] | None = None
        self._max_depths: list[float] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]
        self._max_depths = [float(adapter.nodes.get_max_depth(i)) for i in self._idxs]

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None and self._max_depths is not None, "bind() before step()"
        get = adapter.nodes.get_depth
        headroom = 0.0
        for idx, md in zip(self._idxs, self._max_depths, strict=True):
            if md <= 0.0:
                continue
            fill = float(get(idx)) / md
            unused = 1.0 - fill
            if unused > 0.0:
                headroom += unused
        return headroom * dt_seconds


class PumpEnergy:
    """Pumping effort summed across pumps, per env step.

    Computed as C{sum_i setting_i * rated_power_i * dt_seconds}, where
    C{setting} is the pump's control setting in [0,1]
    (L{openswmm.engine.Links.get_control_setting}) — the fraction-on / relative
    speed — and C{rated_power} is a per-pump constant (default 1.0). With the
    default this is on-time-weighted effort (∫ setting dt); supply
    C{rated_power} in consistent power units to obtain energy. A flow×head
    variant can replace this once link end-node heads are exposed.

    Operational objective #4 (minimise pumping energy).

    @ivar name: C{"pump_energy"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        link_ids: Sequence[str],
        rated_power: dict[str, float] | None = None,
        name: str = "pump_energy",
    ) -> None:
        """
        @param link_ids: Pump link IDs (required, non-empty).
        @type link_ids: sequence of str
        @param rated_power: Optional C{{link_id: power}} (default 1.0 each).
        @type rated_power: dict of str to float or C{None}
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{link_ids} is empty.
        """
        if not link_ids:
            raise ValueError("PumpEnergy requires at least one link_id")
        self.name = name
        self._link_ids: list[str] = list(link_ids)
        self._rated_power = dict(rated_power) if rated_power is not None else {}
        self._idxs: list[int] | None = None
        self._powers: list[float] | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]
        self._powers = [float(self._rated_power.get(lid, 1.0)) for lid in self._link_ids]

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None and self._powers is not None, "bind() before step()"
        get = adapter.links.get_control_setting
        effort = 0.0
        for idx, power in zip(self._idxs, self._powers, strict=True):
            setting = float(get(idx))
            if setting > 0.0:
                effort += setting * power
        return effort * dt_seconds
