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
Composable observation builder.

Plan §4. Each collector reads a single feature kind (node depths, link
flows, etc.) from the engine; the builder concatenates them into a
flat float32 vector.

P2 ships ten collectors covering the most common SWMM-RL feature sets:
node depths/heads/inflows/overflows, link flows/depths/settings,
subcatchment runoff, gage rainfall, and a clock collector. Forecast
injection lands in P5 (forecast wrapper).

Water-quality observation is served by
L{ObservationBuilder.add_pollutant_concentration} (node concentrations,
per plan §4) and L{ObservationBuilder.add_link_pollutant_concentration}.

The engine's 2D surface is exposed only through the flat
selected-vertex collector L{ObservationBuilder.add_2d_vertex_depths}
(bulk C{Surface2D.get_vertex_render_depths} + gather at chosen vertex
indices). A full spatial (grid/mesh) observation subsystem remains a
separate design, not a collector bolt-on.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np
from gymnasium import spaces

from openswmm_gymnasium._engine import SolverAdapter

# =============================================================================
# Internal collector base + helpers
# =============================================================================


def _require_ids(name: str, ids: Sequence[str]) -> list[str]:
    if not ids:
        raise ValueError(f"{name} requires at least one id")
    return list(ids)


class _ScalarReadCollector:
    """Shared base for collectors that read one scalar per element index.

    Subclasses set L{_kind_label}, L{_resolve_idxs}, L{_read_scalar}.

    @ivar _ids: Symbolic element IDs.
    @type _ids: list[str]
    @ivar _idxs: Engine indices, resolved by L{bind}.
    @type _idxs: list[int] or C{None}
    """

    _kind_label: str = "_ScalarReadCollector"

    def __init__(self, ids: Sequence[str]) -> None:
        self._ids: list[str] = _require_ids(self._kind_label, ids)
        self._idxs: list[int] | None = None
        self._idx_arr = None

    @property
    def size(self) -> int:
        return len(self._ids)

    def bind(self, adapter: SolverAdapter) -> None:
        self._idxs = self._resolve_idxs(adapter)
        # Pre-built index array for the bulk gather path (avoids rebuilding
        # it every step).
        self._idx_arr = np.asarray(self._idxs, dtype=np.intp)

    def collect(self, adapter: SolverAdapter) -> np.ndarray:
        assert self._idxs is not None, "bind() before collect()"
        # Fast path: one vectorized engine read + NumPy gather, instead of
        # N scalar FFI round-trips. Falls back to the scalar loop when a
        # collector has no bulk source.
        arr = self._bulk_array(adapter)
        if arr is not None:
            return np.asarray(arr, dtype=np.float32)[self._idx_arr]
        return np.fromiter(
            (self._read_scalar(adapter, i) for i in self._idxs),
            dtype=np.float32,
            count=len(self._idxs),
        )

    # Subclass hooks ------------------------------------------------------

    def _resolve_idxs(self, adapter: SolverAdapter) -> list[int]:
        raise NotImplementedError

    def _read_scalar(self, adapter: SolverAdapter, idx: int) -> float:
        raise NotImplementedError

    def _bulk_array(self, adapter: SolverAdapter):
        """Whole-network array for the bulk gather path, or C{None}.

        Override in subclasses backed by an engine bulk getter. Returning
        C{None} (the default) keeps the scalar per-element fallback.

        @rtype: numpy.ndarray or None
        """
        return None


# =============================================================================
# Node collectors
# =============================================================================


class _NodeDepthCollector(_ScalarReadCollector):
    """Instantaneous water depth at each node."""

    _kind_label = "add_node_depths"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_depth(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("depths")


class _NodeHeadCollector(_ScalarReadCollector):
    """Instantaneous hydraulic head at each node."""

    _kind_label = "add_node_heads"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_head(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("heads")


class _NodeInflowCollector(_ScalarReadCollector):
    """Total inflow rate at each node."""

    _kind_label = "add_node_inflows"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_inflow(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("inflows")


class _NodeOverflowCollector(_ScalarReadCollector):
    """Overflow (flooding) rate at each node."""

    _kind_label = "add_node_overflows"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_overflow(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("overflows")


class _NodeVolumeCollector(_ScalarReadCollector):
    """Stored water volume at each node (project volume units)."""

    _kind_label = "add_node_volumes"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_volume(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("volumes")


class _NodeLateralInflowCollector(_ScalarReadCollector):
    """Externally-applied lateral inflow at each node (project flow units)."""

    _kind_label = "add_node_lateral_inflows"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_lateral_inflow(idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.array("lateral_inflows")


# =============================================================================
# Link collectors
# =============================================================================


class _LinkFlowCollector(_ScalarReadCollector):
    """Instantaneous flow through each link."""

    _kind_label = "add_link_flows"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_flow(idx)

    def _bulk_array(self, adapter):
        return adapter.links.array("flows")


class _LinkDepthCollector(_ScalarReadCollector):
    """Instantaneous depth in each link."""

    _kind_label = "add_link_depths"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_depth(idx)

    def _bulk_array(self, adapter):
        return adapter.links.array("depths")


class _LinkVelocityCollector(_ScalarReadCollector):
    """Flow velocity in each link (project length/time units)."""

    _kind_label = "add_link_velocities"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_velocity(idx)

    def _bulk_array(self, adapter):
        return adapter.links.array("velocities")


class _LinkCapacityCollector(_ScalarReadCollector):
    """Fractional capacity / filling C{[0, 1]} of each link."""

    _kind_label = "add_link_capacities"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_capacity(idx)

    def _bulk_array(self, adapter):
        return adapter.links.array("capacities")


class _LinkVolumeCollector(_ScalarReadCollector):
    """Stored water volume in each link (project volume units)."""

    _kind_label = "add_link_volumes"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_volume(idx)

    def _bulk_array(self, adapter):
        return adapter.links.array("volumes")


class _LinkSettingCollector(_ScalarReadCollector):
    """Current control setting C{[0, 1]} for each link.

    Useful for closed-loop RL agents that need to observe the prior
    action they (or another controller) applied.
    """

    _kind_label = "add_link_settings"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_control_setting(idx)


# =============================================================================
# Subcatchment + rain collectors
# =============================================================================


class _SubcatchRunoffCollector(_ScalarReadCollector):
    """Runoff rate from each subcatchment."""

    _kind_label = "add_subcatch_runoff"

    def _resolve_idxs(self, adapter):
        return [adapter.subcatchments.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.subcatchments.get_runoff(idx)


class _SubcatchGroundwaterCollector(_ScalarReadCollector):
    """Groundwater outflow rate from each subcatchment.

    Reads :attr:`openswmm.engine.Subcatchment.groundwater` (project flow
    units); ``0.0`` on subcatchments without an assigned aquifer. Useful for
    agents that must observe slow baseflow / antecedent wetness in addition
    to the fast runoff signal.
    """

    _kind_label = "add_subcatch_groundwater"

    def _resolve_idxs(self, adapter):
        return [adapter.subcatchments.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.subcatchments.get_groundwater(idx)


class _RainfallCollector(_ScalarReadCollector):
    """Rainfall intensity at each rain gage."""

    _kind_label = "add_rainfall"

    def _resolve_idxs(self, adapter):
        return [adapter.gages.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.gages.get_rainfall(idx)


# =============================================================================
# Water-quality collectors
# =============================================================================


class _PollutantConcentrationCollector(_ScalarReadCollector):
    """Shared base for node / link pollutant-concentration collectors.

    One feature per element, reporting the concentration of a single
    pollutant in that pollutant's own concentration units (C{mg/L},
    C{ug/L} or C{#/L}, per the model's C{[POLLUTANTS]} declaration). Add
    one collector per pollutant of interest.

    The symbolic pollutant id is resolved to its engine index at L{bind},
    so the per-step read never does a string lookup.
    """

    def __init__(self, ids: Sequence[str], pollutant: str) -> None:
        super().__init__(ids)
        if not pollutant:
            raise ValueError(f"{self._kind_label} requires a pollutant id")
        self._pollutant = str(pollutant)
        self._pollutant_idx: int | None = None

    def bind(self, adapter: SolverAdapter) -> None:
        self._pollutant_idx = adapter.pollutants.get_index(self._pollutant)
        super().bind(adapter)


class _NodeQualityCollector(_PollutantConcentrationCollector):
    """Pollutant concentration at each node."""

    _kind_label = "add_pollutant_concentration"

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_quality(idx, self._pollutant_idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.qualities(self._pollutant_idx)


class _LinkQualityCollector(_PollutantConcentrationCollector):
    """Pollutant concentration in each link."""

    _kind_label = "add_link_pollutant_concentration"

    def _resolve_idxs(self, adapter):
        return [adapter.links.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.links.get_quality(idx, self._pollutant_idx)

    def _bulk_array(self, adapter):
        return adapter.links.qualities(self._pollutant_idx)


# =============================================================================
# 2D surface collectors (integer vertex indices, not symbolic IDs)
# =============================================================================


class _Surface2DVertexDepthCollector:
    """Signed inundation depth (C{eta_v - z_v}) at selected 2D mesh vertices.

    Unlike the 1D collectors, 2D mesh vertices have no symbolic IDs —
    they are addressed by integer index into the mesh's vertex array.
    Reads the whole-mesh bulk render-depth field once per step and
    gathers the requested vertices, mirroring the bulk fast path of
    L{_ScalarReadCollector}.
    """

    _kind_label = "add_2d_vertex_depths"

    def __init__(self, vertex_idxs: Sequence[int]) -> None:
        if not vertex_idxs:
            raise ValueError(f"{self._kind_label} requires at least one vertex index")
        self._vertex_idxs: list[int] = [int(i) for i in vertex_idxs]
        self._idx_arr = None

    @property
    def size(self) -> int:
        return len(self._vertex_idxs)

    def bind(self, adapter: SolverAdapter) -> None:
        # adapter.surface2d raises with a clear message when the engine
        # has no 2D module or the model's 2D surface is inactive
        # (including IGNORE_2D episodes).
        n = adapter.surface2d.n_vertices
        bad = [i for i in self._vertex_idxs if not 0 <= i < n]
        if bad:
            raise ValueError(
                f"{self._kind_label}: vertex indices out of range "
                f"[0, {n}): {bad}"
            )
        self._idx_arr = np.asarray(self._vertex_idxs, dtype=np.intp)

    def collect(self, adapter: SolverAdapter) -> np.ndarray:
        assert self._idx_arr is not None, "bind() before collect()"
        arr = adapter.surface2d.vertex_render_depths()
        return np.asarray(arr, dtype=np.float32)[self._idx_arr]


# =============================================================================
# Clock collector (no per-element repetition)
# =============================================================================


_CLOCK_FEATURES = ("hour_sin", "hour_cos", "elapsed_frac")


class _ClockCollector:
    """Time-of-day + episode-progress features.

    Three features in fixed order:

      - C{hour_sin} — C{sin(2π * hour_of_day / 24)}.
      - C{hour_cos} — C{cos(2π * hour_of_day / 24)}.
      - C{elapsed_frac} — C{(current - start) / (end - start)}, in
        C{[0, 1]}.

    The hour-of-day uses the fractional part of the engine's OADate
    (1.0 == 1 day), so encoding is correct regardless of the model's
    start date.

    @ivar _features: Subset and order of features to return.
    @type _features: tuple[str, ...]
    """

    def __init__(self, features: Sequence[str] | None = None) -> None:
        if features is None:
            features = _CLOCK_FEATURES
        unknown = set(features) - set(_CLOCK_FEATURES)
        if unknown:
            raise ValueError(
                f"Unknown clock features: {sorted(unknown)}; valid: {list(_CLOCK_FEATURES)}"
            )
        self._features: tuple[str, ...] = tuple(features)
        self._adapter: SolverAdapter | None = None

    @property
    def size(self) -> int:
        return len(self._features)

    def bind(self, adapter: SolverAdapter) -> None:
        # No symbolic IDs to resolve; we just stash the adapter for
        # later constant-time access to start/end.
        self._adapter = adapter

    def collect(self, adapter: SolverAdapter) -> np.ndarray:
        assert self._adapter is not None, "bind() before collect()"
        current = adapter.current_time
        start = adapter.start_time
        end = adapter.end_time
        # OADate fractional part = time-of-day in days.
        hour_of_day = (current - math.floor(current)) * 24.0
        angle = 2.0 * math.pi * hour_of_day / 24.0
        span = end - start
        if span <= 0.0:
            elapsed_frac = 0.0
        else:
            elapsed_frac = max(0.0, min(1.0, (current - start) / span))
        values = {
            "hour_sin": math.sin(angle),
            "hour_cos": math.cos(angle),
            "elapsed_frac": elapsed_frac,
        }
        return np.fromiter(
            (values[f] for f in self._features),
            dtype=np.float32,
            count=len(self._features),
        )


# =============================================================================
# Builder
# =============================================================================


class ObservationBuilder:
    """Fluent composer for the env's observation space.

    Returned space is a flat L{gymnasium.spaces.Box} of dtype float32
    with shape C{(sum(collector.size),)}. Bounds are
    C{[-inf, +inf]}; callers wishing finite bounds should wrap with a
    L{gymnasium.wrappers.NormalizeObservation} or supply their own
    L{gymnasium.spaces.Box} via a subclass in a later phase.

    Example::

        obs = (ObservationBuilder()
               .add_node_depths(["J1"])
               .add_link_flows(["C1"])
               .add_clock())

    @ivar _collectors: Ordered list of bound-and-collect units.
    @type _collectors: list
    """

    def __init__(self) -> None:
        self._collectors: list = []

    # ----- Node features -------------------------------------------------

    def add_node_depths(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-depth collector.

        @param node_ids: Node IDs whose depths to observe.
        @type node_ids: sequence of str
        @return: This builder, for chaining.
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeDepthCollector(node_ids))
        return self

    def add_node_heads(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-head collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeHeadCollector(node_ids))
        return self

    def add_node_inflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-inflow collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeInflowCollector(node_ids))
        return self

    def add_node_overflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-overflow (flooding rate) collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeOverflowCollector(node_ids))
        return self

    def add_node_volumes(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node stored-volume collector (project volume units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeVolumeCollector(node_ids))
        return self

    def add_node_lateral_inflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node lateral-inflow collector (project flow units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_NodeLateralInflowCollector(node_ids))
        return self

    # ----- Link features -------------------------------------------------

    def add_link_flows(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-flow collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkFlowCollector(link_ids))
        return self

    def add_link_depths(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-depth collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkDepthCollector(link_ids))
        return self

    def add_link_settings(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-control-setting collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkSettingCollector(link_ids))
        return self

    def add_link_velocities(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-velocity collector (project length/time units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkVelocityCollector(link_ids))
        return self

    def add_link_capacities(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link fractional-capacity collector C{[0, 1]}.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkCapacityCollector(link_ids))
        return self

    def add_link_volumes(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link stored-volume collector (project volume units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkVolumeCollector(link_ids))
        return self

    # ----- Subcatchment + rain features ---------------------------------

    def add_subcatch_runoff(self, subcatch_ids: Sequence[str]) -> ObservationBuilder:
        """Append a subcatchment-runoff collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_SubcatchRunoffCollector(subcatch_ids))
        return self

    def add_subcatch_groundwater(
        self, subcatch_ids: Sequence[str]
    ) -> ObservationBuilder:
        """Append a subcatchment-groundwater (baseflow) collector.

        @param subcatch_ids: Subcatchment IDs whose groundwater outflow to
            observe.
        @type subcatch_ids: sequence of str
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_SubcatchGroundwaterCollector(subcatch_ids))
        return self

    def add_rainfall(self, gage_ids: Sequence[str]) -> ObservationBuilder:
        """Append a rain-gage rainfall collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_RainfallCollector(gage_ids))
        return self

    # ----- Water-quality features ----------------------------------------

    def add_pollutant_concentration(
        self, node_ids: Sequence[str], pollutant: str
    ) -> ObservationBuilder:
        """Append a node pollutant-concentration collector (plan §4).

        One feature per node, reporting the concentration of C{pollutant}
        in that pollutant's declared concentration units (C{mg/L},
        C{ug/L}, C{#/L}). Call once per pollutant of interest.

        Requires the model to run water quality — the pollutant must be
        declared in C{[POLLUTANTS]}, or L{bind} raises.

        @param node_ids: Node IDs whose concentrations to observe.
        @type node_ids: sequence of str
        @param pollutant: Symbolic pollutant ID, e.g. C{"TSS"}.
        @type pollutant: str
        @return: This builder, for chaining.
        @rtype: L{ObservationBuilder}
        @raise ValueError: If C{node_ids} is empty or C{pollutant} is blank.
        """
        self._collectors.append(_NodeQualityCollector(node_ids, pollutant))
        return self

    def add_link_pollutant_concentration(
        self, link_ids: Sequence[str], pollutant: str
    ) -> ObservationBuilder:
        """Append a link pollutant-concentration collector.

        The link-side counterpart of L{add_pollutant_concentration}; the
        natural observation to pair with a load-based objective such as
        L{openswmm_gymnasium.rewards.terms.TSSLoad}, which is computed
        from link flow and link concentration.

        @param link_ids: Link IDs whose concentrations to observe.
        @type link_ids: sequence of str
        @param pollutant: Symbolic pollutant ID, e.g. C{"TSS"}.
        @type pollutant: str
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_LinkQualityCollector(link_ids, pollutant))
        return self

    # ----- 2D surface features -------------------------------------------

    def add_2d_vertex_depths(self, vertex_idxs: Sequence[int]) -> ObservationBuilder:
        """Append a 2D mesh vertex inundation-depth collector.

        Observes the signed render depth (C{eta_v - z_v}, m; negative =
        dry freeboard) at the given mesh vertex indices, via the bulk
        C{Surface2D.get_vertex_render_depths} read. Requires an engine
        built with the 2D module and a model with an active 2D surface;
        binding fails with a clear error otherwise (including episodes
        run with the C{IGNORE_2D} gate on).

        @param vertex_idxs: 2D mesh vertex indices (0-based) to observe.
        @type vertex_idxs: sequence of int
        @return: This builder, for chaining.
        @rtype: L{ObservationBuilder}
        @raise ValueError: If C{vertex_idxs} is empty.
        """
        self._collectors.append(_Surface2DVertexDepthCollector(vertex_idxs))
        return self

    # ----- Time features -------------------------------------------------

    def add_clock(self, features: Sequence[str] | None = None) -> ObservationBuilder:
        """Append a clock collector.

        @param features: Subset of C{("hour_sin", "hour_cos",
            "elapsed_frac")}. Defaults to all three.
        @type features: sequence of str or C{None}
        @rtype: L{ObservationBuilder}
        @raise ValueError: If C{features} contains an unrecognised key.
        """
        self._collectors.append(_ClockCollector(features))
        return self

    # ----- Build / bind / collect ---------------------------------------

    def space(self) -> spaces.Box:
        """Construct the env's observation space.

        @rtype: L{gymnasium.spaces.Box}
        @raise ValueError: If no collectors have been added.
        """
        if not self._collectors:
            raise ValueError("ObservationBuilder is empty; add at least one collector")
        size = sum(c.size for c in self._collectors)
        return spaces.Box(low=-np.inf, high=np.inf, shape=(size,), dtype=np.float32)

    def bind(self, adapter: SolverAdapter) -> None:
        """Resolve symbolic IDs in every collector.

        @param adapter: Adapter wrapping the open solver.
        @type adapter: L{SolverAdapter}
        """
        for c in self._collectors:
            c.bind(adapter)

    def collect(self, adapter: SolverAdapter) -> np.ndarray:
        """Read the current observation from the engine.

        @param adapter: Adapter wrapping the running solver.
        @type adapter: L{SolverAdapter}
        @return: 1-D float32 array matching L{space}.
        @rtype: numpy.ndarray
        """
        return np.concatenate([c.collect(adapter) for c in self._collectors])
