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
from typing import Any

import numpy as np
from gymnasium import spaces
from openswmm.engine import _enums as _engine_enums
from openswmm.engine import catalog as _catalog

from openswmm_gymnasium._engine import SolverAdapter, element_kind, field_entry, unit_label

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

    requires: tuple[str, ...] = ()

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


class _FieldCollector(_ScalarReadCollector):
    """Any numeric engine field of an element kind, addressed by catalog path.

    C{"node.depth"}, C{"link.stats.max_flow"}, C{"subcatchment.runoff"} ...
    -- every numeric property in L{openswmm.engine.catalog}. Reads use the
    collection's bulk array when the engine has one, else one read per element.
    """

    def __init__(self, path: str, ids: Sequence[str], label: str | None = None) -> None:
        self._kind_label = label or f"add_field({path!r})"
        super().__init__(ids)
        field_entry(path)  # fail at construction, naming the path, if it is unknown
        self._path = path
        self.requires = (path,)
        self._kind = element_kind(path)

    def units(self, unit_system: str | None, flow_units: str | None) -> list[str]:
        label = unit_label(field_entry(self._path).get("units"), unit_system, flow_units)
        return [label or ""] * len(self._ids)

    def _resolve_idxs(self, adapter):
        return [adapter.index(self._kind, i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return float(adapter.read(self._path, idx))

    def _bulk_array(self, adapter):
        return adapter.read_all(self._path)


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

    def units(self, unit_system: str | None, flow_units: str | None) -> list[str]:
        return [f"{self._pollutant} concentration units"] * len(self._ids)

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

    requires = ("pollutants", "nodes.qualities", "node.quality")

    def _resolve_idxs(self, adapter):
        return [adapter.nodes.get_index(i) for i in self._ids]

    def _read_scalar(self, adapter, idx):
        return adapter.nodes.get_quality(idx, self._pollutant_idx)

    def _bulk_array(self, adapter):
        return adapter.nodes.qualities(self._pollutant_idx)


class _LinkQualityCollector(_PollutantConcentrationCollector):
    """Pollutant concentration in each link."""

    _kind_label = "add_link_pollutant_concentration"

    requires = ("pollutants", "links.qualities", "link.quality")

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
    requires = ("surface2d",)

    def __init__(self, vertex_idxs: Sequence[int]) -> None:
        if not vertex_idxs:
            raise ValueError(f"{self._kind_label} requires at least one vertex index")
        self._vertex_idxs: list[int] = [int(i) for i in vertex_idxs]
        self._idx_arr = None

    @property
    def size(self) -> int:
        return len(self._vertex_idxs)

    def units(self, unit_system: str | None, flow_units: str | None) -> list[str]:
        return ["m"] * len(self._vertex_idxs)  # the engine's 2D state is SI

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


class _CellFieldCollector:
    """A per-cell quantity of a 2D service, read at chosen mesh cells.

    C{path} is a catalog method that either returns an array with one value
    per cell (C{"surface2d.get_depths"}, C{"surface2d.infiltration.rate"},
    C{"surface2d.groundwater.cells"} with C{variable="HG"},
    C{"surface2d.quality.buildup"} with C{species="TSS"}) -- read once per
    step and gathered -- or takes the cell index first and returns a float
    (C{"surface2d.get_rainfall"}) and is called per cell. Vertex quantities
    are not cells; see L{ObservationBuilder.add_2d_vertex_depths}.
    """

    def __init__(self, path: str, cells: Sequence[int], args: dict | None = None) -> None:
        if not cells:
            raise ValueError(f"add_cell_field({path!r}) requires at least one cell index")
        try:
            entry = _catalog.lookup(path)
        except KeyError as exc:
            raise ValueError(str(exc.args[0])) from None
        required = [p["name"] for p in entry.get("params", []) if p["required"]]
        returns = entry.get("returns", "")
        self._per_cell = required[:1] in (["idx"], ["cell"]) and returns == "float"
        bulk = "NDArray" in returns and required[:1] not in (["idx"], ["cell"])
        if (
            entry["form"] != "method"
            or not path.startswith("surface2d")
            or "vertex" in path
            or not (self._per_cell or bulk)
        ):
            raise ValueError(f"{path!r} is not a per-cell method of a 2D service")
        missing = [n for n in required[self._per_cell :] if n not in (args or {})]
        if missing:
            raise ValueError(f"{path!r} needs argument(s) {missing}")
        self._path = path
        self._entry = entry
        self._args = {k: _coerce_enum(entry, k, v) for k, v in (args or {}).items()}
        self._cells = [int(c) for c in cells]
        self._idx_arr = np.asarray(self._cells, dtype=np.intp)
        self.requires = ("surface2d", path)

    @property
    def size(self) -> int:
        return len(self._cells)

    def units(self, unit_system: str | None, flow_units: str | None) -> list[str]:
        label = unit_label(self._entry.get("units"), unit_system, flow_units)
        return [label or ""] * len(self._cells)

    def bind(self, adapter: SolverAdapter) -> None:
        # adapter.surface2d raises with the remedy named when the engine has no
        # 2D module or the model's 2D surface is inactive.
        n = adapter.surface2d.n_cells
        bad = [c for c in self._cells if not 0 <= c < n]
        if bad:
            raise ValueError(f"add_cell_field({self._path!r}): cells out of range [0, {n}): {bad}")
        if not self._per_cell:
            # Also surfaces an unconfigured service (no groundwater, no quality).
            size = len(adapter.call(self._path, **self._args))
            if size != n:
                raise ValueError(
                    f"add_cell_field({self._path!r}) returns {size} values, not one per cell ({n})"
                )

    def collect(self, adapter: SolverAdapter) -> np.ndarray:
        if self._per_cell:
            values = [adapter.call(self._path, c, **self._args) for c in self._cells]
            return np.asarray(values, dtype=np.float32)
        arr = np.asarray(adapter.call(self._path, **self._args), dtype=np.float64)
        return arr[self._idx_arr].astype(np.float32)


def _coerce_enum(entry: dict, name: str, value: Any) -> Any:
    """Pass an enum argument by member name (C{variable="HG"}) as the engine enum."""
    param = next((p for p in entry.get("params", []) if p["name"] == name), None)
    if param and isinstance(value, str) and param["type"] in _catalog.load()["enums"]:
        enum = getattr(_engine_enums, param["type"])
        try:
            return enum[value.upper()]
        except KeyError:
            raise ValueError(
                f"{name}={value!r} is not a {param['type']}; one of {[m.name for m in enum]}"
            ) from None
    return value


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

    requires: tuple[str, ...] = ()

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

    def units(self, unit_system: str | None, flow_units: str | None) -> list[str]:
        return ["dimensionless"] * len(self._features)

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

    # ----- Any engine field ----------------------------------------------

    def add_field(self, path: str, ids: Sequence[str]) -> ObservationBuilder:
        """Append a collector for any numeric engine field of an element kind.

        C{path} is a catalog path such as C{"node.depth"},
        C{"link.stats.max_flow"}, C{"subcatchment.infil"} or
        C{"node.stats.max_depth"}; see L{openswmm.engine.catalog}. The
        named-feature methods below are shorthands for common paths.

        @param path: Catalog field path.
        @type path: str
        @param ids: Element IDs to observe.
        @type ids: sequence of str
        @raise ValueError: If the path is not a numeric element field.
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector(path, ids))
        return self

    # ----- Node features -------------------------------------------------

    def add_node_depths(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-depth collector.

        @param node_ids: Node IDs whose depths to observe.
        @type node_ids: sequence of str
        @return: This builder, for chaining.
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("node.depth", node_ids, "add_node_depths"))
        return self

    def add_node_heads(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-head collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("node.head", node_ids, "add_node_heads"))
        return self

    def add_node_inflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-inflow collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("node.inflow", node_ids, "add_node_inflows"))
        return self

    def add_node_overflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node-overflow (flooding rate) collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("node.overflow", node_ids, "add_node_overflows"))
        return self

    def add_node_volumes(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node stored-volume collector (project volume units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("node.volume", node_ids, "add_node_volumes"))
        return self

    def add_node_lateral_inflows(self, node_ids: Sequence[str]) -> ObservationBuilder:
        """Append a node lateral-inflow collector (project flow units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(
            _FieldCollector("node.lateral_inflow", node_ids, "add_node_lateral_inflows")
        )
        return self

    # ----- Link features -------------------------------------------------

    def add_link_flows(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-flow collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("link.flow", link_ids, "add_link_flows"))
        return self

    def add_link_depths(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-depth collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("link.depth", link_ids, "add_link_depths"))
        return self

    def add_link_settings(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-control-setting collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(
            _FieldCollector("link.control_setting", link_ids, "add_link_settings")
        )
        return self

    def add_link_velocities(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link-velocity collector (project length/time units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("link.velocity", link_ids, "add_link_velocities"))
        return self

    def add_link_capacities(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link fractional-capacity collector C{[0, 1]}.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("link.capacity", link_ids, "add_link_capacities"))
        return self

    def add_link_volumes(self, link_ids: Sequence[str]) -> ObservationBuilder:
        """Append a link stored-volume collector (project volume units).

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("link.volume", link_ids, "add_link_volumes"))
        return self

    # ----- Subcatchment + rain features ---------------------------------

    def add_subcatch_runoff(self, subcatch_ids: Sequence[str]) -> ObservationBuilder:
        """Append a subcatchment-runoff collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(
            _FieldCollector("subcatchment.runoff", subcatch_ids, "add_subcatch_runoff")
        )
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
        self._collectors.append(
            _FieldCollector("subcatchment.groundwater", subcatch_ids, "add_subcatch_groundwater")
        )
        return self

    def add_rainfall(self, gage_ids: Sequence[str]) -> ObservationBuilder:
        """Append a rain-gage rainfall collector.

        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_FieldCollector("gage.rainfall", gage_ids, "add_rainfall"))
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

    def add_cell_field(self, path: str, cells: Sequence[int], **args: Any) -> ObservationBuilder:
        """Append a per-cell 2D quantity at the given mesh cells.

        C{path} is a catalog method of a 2D service returning one value per
        cell, or taking the cell index: C{"surface2d.get_depths"},
        C{"surface2d.get_rainfall"}, C{"surface2d.infiltration.rate"},
        C{"surface2d.groundwater.cells"} (C{variable="HG"} for heads),
        C{"surface2d.quality.buildup"} (C{species="TSS"}). Extra keyword
        arguments go to that method; enum arguments may be given by name.

        @param path: Catalog method path.
        @param cells: Mesh cell indices.
        @raise ValueError: Unknown path, a method that is not per-cell, or a
            missing argument.
        @rtype: L{ObservationBuilder}
        """
        self._collectors.append(_CellFieldCollector(path, cells, args))
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

    def requires(self) -> tuple[str, ...]:
        """Catalog paths the configured collectors need from the engine.

        @rtype: tuple[str, ...]
        """
        return tuple(sorted({p for c in self._collectors for p in c.requires}))

    def units(self, unit_system: str | None, flow_units: str | None = None) -> list[str]:
        """Unit label of every observation component, in L{space} order.

        Labels come from the engine catalog's unit kinds in the model's unit
        system (C{"US"}/C{"SI"}) and flow units; values stay in those units.

        @rtype: list[str]
        """
        return [u for c in self._collectors for u in c.units(unit_system, flow_units)]

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
