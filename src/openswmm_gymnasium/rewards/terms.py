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

Ships nine first-class terms:

  - L{FloodingVolume} (P1) — flooded volume across nodes, read from the
    engine's cumulative flooding statistic.
  - L{CSOVolume} — same math restricted to tagged overflow nodes.
  - L{PeakOutflow} — running-max increments on link flow(s); cumulative
    equals peak observed.
  - L{TSSLoad} — pollutant mass flux (flow × concentration) through links.
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

The remaining plan §5.1 terms (C{CapitalCost}, C{OandMCost}) land in
subsequent phases.

B{Engine statistics.} Where the engine already accumulates a term's
quantity (C{swmm_*_get_stat_*}), the term reads that statistic and
differences successive reads rather than re-integrating a sampled rate in
Python — the engine accumulates every routing step, whereas a term only
sees the state at env-step boundaries. This applies to L{FloodingVolume}
(and hence L{CSOVolume}) and L{PeakOutflow}. The other terms keep their
Python loops because no engine statistic carries the same quantity:

  - L{ReliabilityMargin} — C{node_max_depth} is a cumulative maximum; the
    term wants the per-step B{minimum} freeboard. Not recoverable from a
    running max.
  - L{PumpEnergy} — C{pump_on_time} is unweighted wall-clock on-time; the
    term integrates C{setting × power}, so a pump running at half speed
    is not half an hour of on-time.
  - L{UncontrolledDischarge} — C{link_vol_flow} is the total conveyed
    volume; the term counts only flow toward the outfall, and the two
    differ whenever a link reverses.
  - L{StorageUnderUtilization}, L{SetpointSmoothness} — no engine
    statistic accumulates unused headroom or setpoint churn.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
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
    """Flooded volume across a set of nodes, per env step.

    Read from the engine's cumulative per-node flooding statistic
    (C{swmm_node_get_stat_vol_flooded}, exposed as
    L{SolverAdapter.statistics}): the contribution of one env step is the
    B{increase} in that statistic since the previous step. If C{node_ids}
    is C{None}, all nodes in the model contribute.

    Values are in project B{volume} units (ft³ for US, m³ for SI).

    Deferring the integration to the engine matters whenever an env step
    spans more than one routing step: the engine accumulates
    C{overflow × routing_dt} continuously, while sampling the instantaneous
    overflow rate once per env step and multiplying by the whole env C{dt}
    misses everything in between. C{dt_seconds} is consequently unused.

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
        self._prev_total: float | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        if self._node_ids is None:
            self._idxs = list(range(adapter.nodes.count()))
        else:
            self._idxs = [adapter.nodes.get_index(nid) for nid in self._node_ids]

    def reset(self) -> None:
        """Drop the cumulative-statistic baseline from the prior episode."""
        self._prev_total = None

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None, "bind() before step()"
        get = adapter.statistics.node_vol_flooded
        total = 0.0
        for idx in self._idxs:
            total += float(get(idx))
        if self._prev_total is None:
            # First step of the episode: the statistic starts at zero, so the
            # whole reading is this step's contribution.
            increment = total
        else:
            increment = total - self._prev_total
        self._prev_total = total
        # The engine statistic is monotonically non-decreasing; clamp anyway
        # so a re-bound term never emits a negative cost.
        return increment if increment > 0.0 else 0.0


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

    Each step, for each tracked link, contributes the increase in the
    engine's cumulative maximum-flow statistic
    (C{swmm_link_get_stat_max_flow}, which tracks C{max |flow|} at routing-step
    resolution). When a new peak is set the increment equals the delta;
    otherwise the contribution is zero. Cumulative reward over the episode
    therefore equals the maximum absolute flow observed at each link
    (summed across links), in project flow units.

    Reading the engine statistic rather than sampling C{link.flow} at env-step
    boundaries means a peak that occurs *between* two env steps is still
    caught.

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
        get = adapter.statistics.link_max_flow
        for i, idx in enumerate(self._idxs):
            mag = float(get(idx))
            if mag > self._peak[i]:
                increment += mag - self._peak[i]
                self._peak[i] = mag
        return increment


# =============================================================================
# Water quality
# =============================================================================


class TSSLoad:
    """Pollutant mass flux through a set of links, per env step.

    Computed as C{sum_i max(0, flow_i) * concentration_i * dt_seconds} over
    the supplied links, where C{flow} is the link flow
    (L{openswmm.engine.Links} state) and C{concentration} is that link's
    concentration of C{pollutant} (L{openswmm.engine.Links.quality}).
    Only positive flow — toward the link's downstream node — contributes,
    matching L{UncontrolledDischarge}; reversed flow is ignored rather
    than credited back.

    Despite the name this term works for B{any} declared pollutant; the
    default C{pollutant="TSS"} just matches the plan §5.1 objective. Pass
    C{pollutant=...} for BOD, TP, a tracer, or anything else in
    C{[POLLUTANTS]}.

    B{Units.} The product is C{flow_units × concentration_units × s} —
    e.g. C{cfs·mg/L·s} for a US model declaring C{TSS} in C{MG/L}. That is
    proportional to mass but is B{not} in mass units; SWMM's own report
    applies a unit-system conversion factor this term deliberately does
    not guess at. Use it as a relative objective, or supply your own
    scaling downstream — the value is consistent within a model, which is
    all a search needs.

    There is no engine statistic for pollutant load, so this term
    integrates in Python.

    @ivar name: C{"tss_load"} by default.
    @ivar direction: Always C{"minimize"}.
    """

    direction = "minimize"

    def __init__(
        self,
        link_ids: Sequence[str],
        pollutant: str = "TSS",
        name: str = "tss_load",
    ) -> None:
        """
        @param link_ids: Links across which to integrate load (required,
            non-empty), oriented toward the receiving water / outfall.
        @type link_ids: sequence of str
        @param pollutant: Symbolic pollutant ID as declared in
            C{[POLLUTANTS]}.
        @type pollutant: str
        @param name: Term identifier.
        @type name: str
        @raise ValueError: If C{link_ids} is empty or C{pollutant} is blank.
        """
        if not link_ids:
            raise ValueError("TSSLoad requires at least one link_id")
        if not pollutant:
            raise ValueError("TSSLoad requires a pollutant id")
        self.name = name
        self._link_ids: list[str] = list(link_ids)
        self._pollutant = str(pollutant)
        self._idxs: list[int] | None = None
        self._pollutant_idx: int | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = [adapter.links.get_index(lid) for lid in self._link_ids]
        self._pollutant_idx = adapter.pollutants.get_index(self._pollutant)

    def reset(self) -> None:
        """No cross-step state; no-op."""

    def step(self, adapter: SolverAdapter, dt_seconds: float) -> float:
        assert self._idxs is not None, "bind() before step()"
        get_flow = adapter.links.get_flow
        get_quality = adapter.links.get_quality
        p = self._pollutant_idx
        flux = 0.0
        for idx in self._idxs:
            q = float(get_flow(idx))
            if q > 0.0:
                conc = float(get_quality(idx, p))
                if conc > 0.0:
                    flux += q * conc
        return flux * dt_seconds


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
