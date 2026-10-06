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
Solver lifecycle bridge.

Wraps L{openswmm.engine.Solver} (the handle-based v6 engine) to expose
the minimum surface the rest of the package needs:

  - Lifecycle: L{SolverAdapter.open}, L{SolverAdapter.initialize},
    L{SolverAdapter.start}, L{SolverAdapter.step},
    L{SolverAdapter.stride}, L{SolverAdapter.end},
    L{SolverAdapter.report}, L{SolverAdapter.close}.
  - Idempotent L{SolverAdapter.close} / context-manager protocol.
  - Opt-in lenient (permissive) open with readable validation output via
    L{SolverAdapter.open}C{(lenient=True)}, L{SolverAdapter.open_errors},
    and L{SolverAdapter.open_warnings} — for pre-flight validation of
    programmatically-generated / perturbed training models.
  - Lazy L{openswmm.engine.Nodes} / L{openswmm.engine.Links} /
    L{openswmm.engine.Controls} accessors cached per adapter instance.
  - Hard guard against C{openswmm.legacy.engine.Solver} — passing a
    legacy solver into the adapter raises
    L{LegacySolverRejectedError}. Plan §0 #7 / §2.3.
  - Explicit engine B{capability probe} (L{require_engine_capabilities})
    run at adapter construction, so a too-old / partially-built
    C{openswmm.engine} fails with a message naming the missing symbols
    rather than an C{AttributeError} deep inside a rollout.
  - B{Optional} (build- and model-dependent) surfaces behind lazily-cached
    two-tier-guarded accessors: L{SolverAdapter.surface2d},
    L{SolverAdapter.heat}, L{SolverAdapter.water_age},
    L{SolverAdapter.reactions}. Tier 1 is "does this engine build carry
    the module at all"; tier 2 is "does the B{open model} actually enable
    it" (C{[OPTIONS] HEAT_TRANSPORT} / C{WATER_AGE}). Both raise with the
    remedy named, rather than silently returning defaults — a heat
    configuration written into a model that never runs heat transport is
    a silent no-op, which is exactly the failure mode a training harness
    cannot detect from its reward.

The adapter does B{not} add any high-level domain logic (observations,
rewards, action application). Those live in their respective modules
and consume a L{SolverAdapter} instance.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import importlib
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import openswmm.engine as _engine
from openswmm.engine import (
    Controls,
    EngineState,
    HotStart,
    Inflows,
    Infrastructure,
    LidType,
    Links,
    Nodes,
    Solver,
    Subcatchments,
)
from openswmm.engine import catalog as _catalog

PathLike = str | os.PathLike


# Scalar field types an observation or actuator can carry as one float.
_NUMERIC = frozenset({"float", "int", "bool"})


def field_entry(path: str) -> dict[str, Any]:
    """Catalog entry of a numeric element field such as C{"node.depth"}.

    Field paths name an element kind, optional sub-views and a property:
    C{"link.flow"}, C{"link.stats.max_flow"}, C{"subcatchment.runoff"}.
    Every numeric property in L{openswmm.engine.catalog} is addressable.

    @param path: Dotted catalog path.
    @type path: str
    @raise ValueError: If the path is not a numeric property of an element kind.
    @rtype: dict
    """
    try:
        entry = _catalog.lookup(path)
    except KeyError:
        raise ValueError(
            f"{path!r} is not an engine field; see openswmm.engine.catalog for valid paths"
        ) from None
    if entry["form"] != "property" or entry["type"] not in _NUMERIC:
        raise ValueError(f"{path!r} is not a numeric property ({entry['type'] or entry['form']})")
    element_kind(path)
    return entry


def element_kind(path: str) -> str:
    """The element kind a field path belongs to (C{"link"} for C{"link.stats.max_flow"}).

    @raise ValueError: If the path does not start with an element kind.
    @rtype: str
    """
    kind = path.split(".", 1)[0]
    if "collection" not in _catalog.targets().get(kind, {}):
        kinds = sorted(k for k, t in _catalog.targets().items() if "collection" in t)
        raise ValueError(f"{path!r} does not start with an element kind ({', '.join(kinds)})")
    return kind


# OADate epoch: serial day 0 is 1899-12-30 (midnight). An OADate is the
# number of decimal days since this epoch, so its fractional part is the
# time-of-day. The engine's v6 Python API exposes simulation time as
# :class:`datetime.datetime`; the observation builder consumes OADate
# floats, so the time accessors below convert at the boundary.
_OADATE_EPOCH = datetime(1899, 12, 30)


def _to_oadate(dt: datetime) -> float:
    """Convert an engine :class:`datetime.datetime` to an OADate float."""
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return (dt - _OADATE_EPOCH).total_seconds() / 86400.0


class LegacySolverRejectedError(TypeError):
    """Raised when a legacy v5 singleton solver is passed where the v6
    handle-based solver is required. Plan §0 #7."""


def _reject_legacy(solver: object) -> None:
    """Refuse anything other than the handle-based v6 Solver.

    The legacy solver lives at C{openswmm.legacy.engine.Solver} and
    uses global singleton state, which violates the plan's
    thread-safety requirement (§0 #7 / §2.3).

    @param solver: Candidate solver instance.
    @type solver: object
    @raise LegacySolverRejectedError: If C{solver.__class__.__module__}
        does not start with C{"openswmm.engine"}.
    """
    module = type(solver).__module__
    if not module.startswith("openswmm.engine"):
        raise LegacySolverRejectedError(
            f"openswmm.gymnasium requires the handle-based "
            f"openswmm.engine.Solver (v6); got {type(solver).__module__}."
            f"{type(solver).__name__}. The legacy v5 solver under "
            "openswmm.legacy.engine is not supported."
        )


# ---------------------------------------------------------------------------
# Engine capability probe
# ---------------------------------------------------------------------------
#
# ``pyproject.toml`` pins ``openswmm>=6.0.0a4.dev1`` with no upper bound, and the
# C API is still moving. Rather than a hard version floor — which a partial
# build (e.g. ``OPENSWMM_BUILD_2D=OFF``) would satisfy while still lacking a
# symbol, and which a newer-but-compatible build would fail — this package
# checks the specific engine surface it calls, named by catalog path
# (``"stride"``, ``"tables"``, ``"link.target_setting"``).
#
# Only the adapter's own core (CORE_REQUIREMENTS) is checked for every env.
# Observation collectors, action factories and reward terms declare what they
# need in a ``requires`` attribute, and each env checks exactly the components
# it is configured with (L{require_for}), so a partial build fails only the
# envs that need the missing piece. The optional heat / water-age / reaction /
# 2D modules additionally keep their accessor-level guards on SolverAdapter,
# which also check that the open *model* enables the module — something no
# catalog can know.

#: Solver members the adapter itself calls, needed by every env.
CORE_REQUIREMENTS: tuple[str, ...] = (
    "flow_units",
    "unit_system",
    "set_lenient_open",
    "open_errors",
    "open_warnings",
    "stride",
)


class EngineCapabilityError(RuntimeError):
    """Raised when the installed C{openswmm.engine} lacks a required symbol.

    Names every missing symbol so the failure is actionable instead of
    surfacing as an C{AttributeError} mid-rollout. Plan §4.5 / G5.
    """


def _engine_class(entry: dict, module) -> type | None:
    """The runtime class behind catalog target C{entry}, or C{None} when absent."""
    if module is not None:
        return getattr(module, entry["class"], None)
    cls = getattr(_engine, entry["class"], None)
    if cls is None:
        try:
            cls = getattr(importlib.import_module(f"openswmm.engine.{entry['module']}"),
                          entry["class"], None)
        except ImportError:
            return None
    return cls


def _missing_engine_capabilities(module=None, paths=CORE_REQUIREMENTS) -> list[str]:
    """Return the catalog paths in C{paths} the engine lacks, with the symbol missing.

    A path is a catalog member (C{"stride"}, C{"link.target_setting"}) or a
    target (C{"tables"}, C{"xsect"}). C{module} substitutes a stand-in engine
    namespace so the probe can be tested without a build.

    @param module: Engine namespace; defaults to C{openswmm.engine}.
    @param paths: Catalog paths to check.
    @rtype: list[str]
    """
    targets = _catalog.targets()
    missing: list[str] = []
    for path in sorted(set(paths)):
        if path in targets:
            entry, attr = targets[path], None
        else:
            member = _catalog.lookup(path)
            entry, attr = targets[member["target"]], member["name"]
        cls = _engine_class(entry, module)
        if cls is None:
            missing.append(f"{path} (openswmm.engine.{entry['class']})")
        elif attr is not None and not hasattr(cls, attr):
            missing.append(f"{path} (openswmm.engine.{entry['class']}.{attr})")
    return missing


def require_engine_capabilities(module=None, paths=CORE_REQUIREMENTS) -> None:
    """Verify the installed engine exposes every catalog path in C{paths}.

    Called once per L{SolverAdapter} construction for the core, and by each
    env for its configured components (L{require_for}).

    @param module: Namespace to probe. Defaults to C{openswmm.engine}.
    @param paths: Catalog paths; defaults to L{CORE_REQUIREMENTS}.
    @raise EngineCapabilityError: If any required symbol is absent.
    """
    missing = _missing_engine_capabilities(module, paths)
    if missing:
        version = getattr(module if module is not None else _engine, "__version__", "unknown")
        raise EngineCapabilityError(
            "The installed openswmm.engine (version "
            f"{version}) is missing "
            f"{len(missing)} symbol(s) openswmm.gymnasium requires: "
            + ", ".join(missing)
            + ". Upgrade the openswmm package, or rebuild the engine with the "
            "corresponding modules enabled."
        )


def require_for(*components, module=None) -> None:
    """Check the engine surface every component declares in its C{requires}.

    Envs call this with their observation builder, action factories and
    reward terms, so a build missing an optional piece fails only the envs
    configured to use it.

    @param components: Objects with an optional C{requires} iterable of catalog
        paths (or L{ObservationBuilder}-like objects exposing C{requires()}).
    @raise EngineCapabilityError: If any declared path is missing.
    """
    paths: set[str] = set()
    for c in components:
        req = getattr(c, "requires", ())
        paths.update(req() if callable(req) else req)
    if paths:
        require_engine_capabilities(module, tuple(paths))


# ---------------------------------------------------------------------------
# Collection compatibility shims
# ---------------------------------------------------------------------------
#
# Observation builders, reward terms and design/runtime factories were written
# against scalar ``get_<x>(idx)`` / ``set_<x>(idx, v)`` accessors. The element
# shims keep that surface but route every field through its path in
# ``openswmm.engine.catalog``; each path is checked when this module loads, so
# an engine that renames a field fails at import rather than mid-episode.


def _get(path: str):
    """Scalar getter for catalog field C{path} on element C{idx}."""
    entry = _catalog.lookup(path)
    target, name = entry["target"], entry["name"]

    def get(self, idx: int):
        return getattr(_catalog.resolve(self._solver, target, idx), name)

    get.__doc__ = f"C{{{path}}} of element C{{idx}} ({entry.get('units', 'no units')})."
    return get


def _set(path: str):
    """Scalar setter for catalog field C{path} on element C{idx}."""
    entry = _catalog.lookup(path)
    target, name = entry["target"], entry["name"]

    def set_(self, idx: int, value) -> None:
        setattr(_catalog.resolve(self._solver, target, idx), name, value)

    set_.__doc__ = f"Set C{{{path}}} on element C{{idx}}."
    return set_


class _ElementCompat:
    """Scalar accessors over one element collection."""

    __slots__ = ("_col", "_solver")

    def __init__(self, solver, col) -> None:
        self._solver = solver
        self._col = col

    def count(self) -> int:
        return len(self._col.ids)

    def get_index(self, element_id: str) -> int:
        return self._col.get_index(element_id)


class _NodesCompat(_ElementCompat):
    """Scalar node accessor (catalog paths C{node.*})."""

    __slots__ = ()

    get_depth = _get("node.depth")
    # Controllable externally-applied inflow, project flow units.
    get_lateral_inflow = _get("node.lateral_inflow")
    set_lateral_inflow = _set("node.lateral_inflow")
    get_max_depth = _get("node.max_depth")
    set_max_depth = _set("node.max_depth")
    # Index of the node's depth->area storage curve, or -1 when the shape is
    # not TABULAR; set only on STORAGE + TABULAR nodes.
    get_storage_curve = _get("node.storage.curve")
    set_storage_curve = _set("node.storage.curve")
    _functional = _get("node.storage.functional")
    _set_functional = _set("node.storage.functional")
    _shape = _get("node.storage.shape")

    def set_head_boundary(self, idx: int, value: float) -> None:
        # Fixed head boundary (swmm_node_set_head_boundary); project length units.
        self._col[idx].set_head_boundary(value)

    def get_storage_functional(self, idx: int) -> tuple[float, float, float]:
        # (a, b, c) of the FUNCTIONAL storage relation Area = a*Depth^b + c.
        # Only valid on STORAGE nodes whose shape is FUNCTIONAL.
        return tuple(self._functional(idx))

    def set_storage_functional(self, idx: int, a: float, b: float, c: float) -> None:
        self._set_functional(idx, (a, b, c))

    def get_storage_shape(self, idx: int) -> str:
        # StorageShape member name ("FUNCTIONAL", "TABULAR", ...) so consumers
        # never need the engine enum. Raises on a non-storage node.
        return str(self._shape(idx).name)

    def get_quality(self, idx: int, pollutant: int | str) -> float:
        # Pollutant concentration at the node, in the pollutant's units.
        return self._col[idx].quality(pollutant)

    def qualities(self, pollutant: int | str):
        """Whole-network concentration array for one pollutant.

        @param pollutant: Pollutant index or id.
        @rtype: numpy.ndarray
        """
        return self._col.qualities(pollutant)


class _LinksCompat(_ElementCompat):
    """Scalar link accessor (catalog paths C{link.*})."""

    __slots__ = ()

    get_flow = _get("link.flow")
    get_depth = _get("link.depth")
    get_control_setting = _get("link.control_setting")
    # Persistent runtime-control override: the engine moves control_setting
    # toward target_setting each routing step and holds it there
    # (control_setting itself is recomputed from the target every step).
    get_target_setting = _get("link.target_setting")
    set_target_setting = _set("link.target_setting")
    set_roughness = _set("link.roughness")
    set_length = _set("link.length")

    def get_xsect(self, idx: int) -> tuple:
        # (shape, geom1, geom2, geom3, geom4); shape is an XSectShape enum,
        # consumers coerce it with int().
        return self._col[idx].xsect.as_tuple()

    def set_xsect(
        self, idx: int, shape: int, g1: float, g2: float, g3: float, g4: float
    ) -> None:
        # The v6 xsect setter accepts a (shape, g1, g2, g3, g4) tuple.
        self._col[idx].xsect = (shape, g1, g2, g3, g4)

    def get_xsect_shape_name(self, idx: int) -> str:
        # XSectShape member name, e.g. "CIRCULAR" / "RECT_CLOSED". Names
        # rather than codes: the shape ordinals were renumbered in 6.0, and
        # consumers outside this module cannot import the enum.
        return str(self._col[idx].xsect.info().shape_name)

    def get_full_depth(self, idx: int) -> float:
        """True full depth (rise) of the link's cross-section.

        Delegates to the engine's analytic cross-section geometry
        (L{openswmm.engine.XSectionGeometry}), so the value is exact for
        every shape — unlike C{geom1}, which is the rise only for shapes
        whose first geometry parameter happens to be the height.

        @rtype: float
        """
        return float(self._col[idx].xsect.geometry().full_depth)

    # -- Preissmann-slot readers ------------------------------------------
    #
    # !! ALL THREE READ 0.0 UNDER THE DYNAMIC-WAVE ROUTER. !!
    #
    # The Preissmann slot is a finite-volume-router construct: the notional
    # narrow slot above a closed conduit's crown that lets an FV scheme carry
    # pressurized (surcharged) flow with a free-surface formulation. The
    # engine's dynamic-wave router does not model it, so under
    # ``ROUTING_MODEL DYNWAVE`` every one of these getters returns a hard
    # 0.0 — which is byte-for-byte indistinguishable from "the FV router ran
    # and found no slot flow anywhere".
    #
    # That ambiguity is why no consumer may treat a 0.0 from these as a
    # measurement. A reward term built on slot share would, under dynamic
    # wave, report a perfect (zero-pressurization) network on every step of
    # every episode and hand the agent a flat, uninformative signal it can
    # never move — the worst kind of silent failure, because nothing errors.
    # L{openswmm_gymnasium.rewards.terms.SurchargeSlotShare} therefore
    # documents the constraint in its own docstring and warns at bind time;
    # see that term before adding another consumer.

    # Instantaneous volume above the crown in the slot, project volume units;
    # always a subset of the link volume.
    _slot_volume = _get("link.slot_volume")
    # Run-cumulative max of slot_volume/volume, in [0, 1].
    _peak_slot_share = _get("link.stats.peak_slot_share")
    # Run-level (int slot_volume dt) / (int volume dt), in [0, 1]; not an
    # average of instantaneous ratios.
    _slot_share = _get("link.stats.slot_share")

    def slot_volume(self, idx: int) -> float:
        return float(self._slot_volume(idx))

    def peak_slot_share(self, idx: int) -> float:
        return float(self._peak_slot_share(idx))

    def slot_share(self, idx: int) -> float:
        return float(self._slot_share(idx))

    def get_quality(self, idx: int, pollutant: int | str) -> float:
        # Pollutant concentration in the link, in the pollutant's units.
        return self._col[idx].quality(pollutant)

    def qualities(self, pollutant: int | str):
        """Whole-network concentration array for one pollutant.

        @param pollutant: Pollutant index or id.
        @rtype: numpy.ndarray
        """
        return self._col.qualities(pollutant)


class _ControlsCompat:
    """Scalar control accessor over the v6 ``Controls`` collection.

    The v6 ``Controls`` is a :class:`collections.abc.Sequence`, so its
    ``count`` is the ABC element-count-by-value method; expose the
    package's expected zero-arg ``count()`` (number of controls) and the
    runtime setters.
    """

    __slots__ = ("_col",)

    def __init__(self, col: Controls) -> None:
        self._col = col

    def count(self) -> int:
        return len(self._col)

    def set_link_setting(self, idx: int, value: float) -> None:
        self._col.set_link_setting(idx, value)

    def set_link_status(self, idx: int, value: float) -> None:
        self._col.set_link_status(idx, value)


class _SubcatchmentsCompat(_ElementCompat):
    """Scalar subcatchment accessor (catalog paths C{subcatchment.*})."""

    __slots__ = ()

    # (surf_elev, a1, b1, a2, b2, a3, tw, hstar) in [GROUNDWATER] token
    # order; requires an aquifer to be assigned.
    _gw_params = _get("subcatchment.gw_params")

    def get_gw_params(self, idx: int) -> tuple:
        return tuple(self._gw_params(idx))

    def set_gw_params(
        self,
        idx: int,
        surf_elev: float,
        a1: float,
        b1: float,
        a2: float,
        b2: float,
        a3: float,
        tw: float,
        hstar: float,
    ) -> None:
        # swmm_subcatch_set_gw_params; requires an aquifer to be assigned.
        self._col[idx].set_gw_params(surf_elev, a1, b1, a2, b2, a3, tw, hstar)


class _InfrastructureCompat:
    """LID (green-infrastructure) editor over the v6 ``Infrastructure`` API.

    Surfaces the ``Infrastructure.lids`` surface the design factories need:
    resolving an existing LID-control id to its index, adding a control,
    setting its surface layer, and placing a sized usage on a subcatchment.
    LID edits are valid in the OPENED (pre-initialize) state, which is the
    window CIP design factories run in.
    """

    __slots__ = ("_infra",)

    def __init__(self, infra: Infrastructure) -> None:
        self._infra = infra

    def lid_count(self) -> int:
        return len(self._infra.lids)

    def get_lid_index(self, lid_id: str) -> int:
        # Existing LID-control id -> engine index. Raises if undefined.
        return self._infra.lids.get_index(lid_id)

    def add_lid(self, lid_id: str, lid_type: int) -> int:
        # Create a new LID control of ``lid_type`` (a LidType / int code).
        return self._infra.lids.add(lid_id, LidType(int(lid_type)))

    def set_lid_surface(
        self, idx: int, storage: float, roughness: float, slope: float
    ) -> None:
        self._infra.lids.set_surface(
            idx, storage=storage, roughness=roughness, slope=slope
        )

    def lid_usage_add(
        self,
        subcatchment: int | str,
        lid_idx: int,
        number: int,
        area: float,
        width: float,
        init_sat: float = 0.0,
        from_imperv: float = 0.0,
    ) -> None:
        # Place ``number`` LID units of control ``lid_idx`` on ``subcatchment``
        # (id or index); ``lid`` must be an integer control index.
        self._infra.lids.usage_add(
            subcatchment,
            int(lid_idx),
            number=int(number),
            area=area,
            width=width,
            init_sat=init_sat,
            from_imperv=from_imperv,
        )


class _InflowsCompat:
    """Inflow editor over the v6 ``Inflows`` API (RDII / hydrograph surface).

    Surfaces the RDII unit-hydrograph editing the design factories need:
    scanning the existing hydrograph entries to preserve untouched
    parameters, then rewriting the RTK triangle and the initial-abstraction
    terms of a target ``(uh_name, month, response)`` group. Inflow edits are
    valid in the OPENED (pre-initialize) state.
    """

    __slots__ = ("_inflows",)

    def __init__(self, inflows: Inflows) -> None:
        self._inflows = inflows

    def hydrograph_count(self) -> int:
        return int(self._inflows.hydrograph_count)

    def get_hydrograph(self, idx: int):
        # Returns a HydrographEntry:
        # (uh_name, month, response, r, t, k, dmax, drecov, dinit).
        return self._inflows.get_hydrograph(idx)

    def set_hydrograph_rtk(
        self, uh_name: str, month: int, response: int, r: float, t: float, k: float
    ) -> None:
        self._inflows.set_hydrograph_rtk(uh_name, int(month), int(response), r, t, k)

    def set_hydrograph_ia(
        self,
        uh_name: str,
        month: int,
        response: int,
        dmax: float,
        drecov: float,
        dinit: float,
    ) -> None:
        self._inflows.set_hydrograph_ia(
            uh_name, int(month), int(response), dmax, drecov, dinit
        )

    def add_rdii(self, node: int | str, uh_name: str, area: float) -> None:
        self._inflows.add_rdii(node, uh_name, area)

    def rdii_count(self) -> int:
        return int(self._inflows.rdii_count)


class _PollutantsCompat(_ElementCompat):
    """Pollutant catalogue accessor.

    Read-only identity surface: the observation collectors and the
    L{openswmm_gymnasium.rewards.terms.TSSLoad} term resolve a symbolic
    pollutant id to its engine index once at bind time and then read
    concentrations through the node / link collections.
    """

    __slots__ = ()


class _TablesCompat:
    """Curve accessor over the v6 ``Tables`` collection.

    Surfaces the depth-area curve editing that tabular-storage design
    (L{openswmm_gymnasium.spaces.design.StorageVolume}) needs: read the
    baseline points once at bind, rewrite them on apply. Table edits are
    valid in the OPENED (pre-initialize) state.
    """

    __slots__ = ("_col",)

    def __init__(self, col) -> None:
        self._col = col

    def get_curve_points(self, idx: int):
        """Return the curve's points as an C{(n, 2)} float array.

        @rtype: numpy.ndarray
        """
        return self._col.as_curve(idx).points

    def set_curve_points(self, idx: int, points) -> None:
        """Replace every point of the curve at C{idx}.

        @param points: Iterable of C{(x, y)} pairs.
        """
        curve = self._col.as_curve(idx)
        curve.clear()
        for x, y in points:
            curve.add_point(float(x), float(y))


class _StatisticsCompat:
    """Scalar simulation-statistics accessor over the v6 ``Statistics`` API.

    Every value is B{cumulative from the start of the simulation}, updated
    at the engine's routing-step resolution. Reward terms that want a
    per-env-step contribution difference successive reads themselves.
    """

    __slots__ = ("_stats",)

    def __init__(self, stats) -> None:
        self._stats = stats

    def node_vol_flooded(self, idx: int) -> float:
        # Cumulative flooded volume at the node, project volume units.
        return float(self._stats.node_vol_flooded_at(idx))

    def link_max_flow(self, idx: int) -> float:
        # Maximum |flow| seen in the link so far, project flow units.
        return float(self._stats.link_max_flow_at(idx))


class _HeatCompat:
    """Scalar heat-transport accessor over the v6 ``Heat`` API.

    B{This is a configuration surface, not an observation surface.} The C
    API exposes B{no} per-node or per-link water-temperature getter, so
    there is nothing here for an observation collector to read. What it
    does expose is the inlet-temperature configuration (per source
    pathway, optionally overridden per node) plus exactly two
    current-step scalars — L{current_shortwave} and
    L{current_cloud_fraction} — which are B{forcing}, not state.

    Source-temperature writes are documented LIVE: an edit takes effect on
    the B{next} routing step, so this doubles as a runtime actuator
    (L{openswmm_gymnasium.spaces.runtime.HeatSourceTemperatureSetpoint})
    as well as a design surface
    (L{openswmm_gymnasium.spaces.design.HeatSourceTemperature}).

    Source pathways are addressed by their B{name} string (C{"DWF"},
    C{"EXTERNAL_INFLOW"}, C{"RAINFALL"}, C{"GW"}, C{"RDII"}, C{"IFACE"},
    C{"INITIAL_STATE"}); the engine's C{HeatSourceKind} enum never leaves
    this module. Only C{DWF} and C{EXTERNAL_INFLOW} accept node-scope
    overrides — the engine refuses the rest rather than silently deferring.

    Temperatures are B{degrees Celsius} regardless of the model's unit
    system, and the engine B{refuses} (does not clamp) any value outside
    C{[-50, 100]}; a refused write does not take effect.
    """

    __slots__ = ("_heat",)

    def __init__(self, heat) -> None:
        self._heat = heat

    @property
    def enabled(self) -> bool:
        # swmm_heat_get_enabled — is [OPTIONS] HEAT_TRANSPORT on?
        return bool(self._heat.enabled)

    def source_kinds(self) -> list[str]:
        """Every heat source pathway, as member-name strings.

        @rtype: list[str]
        """
        # Iterating Heat.sources yields HeatSourceKind members; return names
        # so consumers never need the engine enum.
        return [str(getattr(s, "name", s)) for s in self._heat.sources]

    def is_source_configured(self, source: str) -> bool:
        # swmm_heat_is_source_configured — whether the model actually sets a
        # temperature for this pathway (vs. reporting the engine default).
        return bool(self._heat.sources.is_configured(source))

    def get_source_temp(self, source: str) -> float:
        # swmm_heat_get_source_temp — GLOBAL inlet temperature, degC.
        return float(self._heat.sources[source])

    def set_source_temp(self, source: str, temp_c: float) -> None:
        # swmm_heat_set_source_temp — GLOBAL inlet temperature, degC. REFUSED
        # (not clamped) outside [-50, 100]. LIVE: takes effect next routing step.
        self._heat.sources[source] = float(temp_c)

    def clear_source(self, source: str) -> None:
        # swmm_heat_clear_source — drop the model's setting for this pathway.
        self._heat.sources.clear(source)

    def get_effective_source_temp(self, source: str, node: int | str) -> float:
        # swmm_heat_get_effective_source_temp — the temperature this node
        # actually sees: its node override if one exists, else the global.
        return float(self._heat.sources.effective(source, node))

    def node_override_count(self) -> int:
        # swmm_heat_get_node_override_count
        return len(self._heat.node_overrides)

    def get_node_override(self, row_index: int) -> tuple[str, int, float]:
        """Read one node-scope override row.

        @return: C{(source_name, node_index, temp_c)}. The source is a
            member-name string, never the engine enum.
        @rtype: tuple
        """
        # swmm_heat_get_node_override_at
        row = self._heat.node_overrides[row_index]
        return (str(getattr(row.source, "name", row.source)), int(row.node_index), float(row.temp_c))

    def set_node_override(self, source: str, node: int | str, temp_c: float) -> None:
        # swmm_heat_set_node_override — DWF / EXTERNAL_INFLOW only (the H1
        # scope rule; other pathways are REFUSED, not deferred). degC,
        # refused outside [-50, 100].
        self._heat.node_overrides.set(source, node, float(temp_c))

    def remove_node_override(self, row_index: int) -> None:
        # swmm_heat_remove_node_override
        self._heat.node_overrides.remove(int(row_index))

    @property
    def current_shortwave(self) -> float:
        """Current-step incident shortwave radiation.

        One of only two genuinely observable heat quantities in the C API.
        Project radiation units, unconverted.

        @rtype: float
        """
        # swmm_heat_get_current_shortwave
        return float(self._heat.current_shortwave)

    @property
    def current_cloud_fraction(self) -> float:
        """Current-step cloud-cover fraction C in C{[0, 1]}.

        A fraction, B{not} a percent. The other genuinely observable heat
        quantity in the C API.

        @rtype: float
        """
        # swmm_heat_get_current_cloud
        return float(self._heat.cloud.current)


class _WaterAgeCompat:
    """Scalar water-age accessor over the v6 ``WaterAge`` API.

    Like L{_HeatCompat} this is a B{configuration} surface: the C API
    exposes no per-node or per-link water-age state getter, so there is
    nothing to observe — only the per-source-pathway inlet ages that seed
    the transport, optionally overridden per node.

    Ages are in B{hours} (the config file's unit), and B{negative values
    are legal and meaningful}: a negative source age B{extracts}
    age-volume, clamped engine-side so age never goes below zero. Nothing
    in this package may clamp the low bound at zero on the user's behalf.

    Source pathways are addressed by member-name string (C{"DWF"},
    C{"EXTERNAL_INFLOW"}, C{"RAINFALL"}, C{"GW"}, C{"RDII"}, C{"IFACE"},
    C{"INITIAL_STATE"}); the engine's C{WaterAgeSource} enum never leaves
    this module. Only C{DWF} and C{EXTERNAL_INFLOW} accept node-scope
    overrides (the A1a scope rule).
    """

    __slots__ = ("_wa",)

    def __init__(self, water_age) -> None:
        self._wa = water_age

    @property
    def enabled(self) -> bool:
        # swmm_water_age_get_enabled — is [OPTIONS] WATER_AGE on?
        return bool(self._wa.enabled)

    def source_pathways(self) -> list[str]:
        """Every water-age source pathway, as member-name strings.

        @rtype: list[str]
        """
        return [str(getattr(s, "name", s)) for s in self._wa.globals]

    def get_source_age(self, source: str) -> float:
        # swmm_water_age_get_global — GLOBAL source age, HOURS (signed).
        return float(self._wa.globals[source])

    def set_source_age(self, source: str, hours: float) -> None:
        # swmm_water_age_set_global — GLOBAL source age, HOURS. Negative is
        # legal (age-volume extraction, D-NS1). LIVE: next routing step.
        self._wa.globals[source] = float(hours)

    def node_override_count(self) -> int:
        # swmm_water_age_get_override_count
        return len(self._wa.node_overrides)

    def get_node_override(self, row_index: int) -> tuple[str, int, float]:
        """Read one node-scope override row.

        @return: C{(source_name, node_index, hours)}. The source is a
            member-name string, never the engine enum.
        @rtype: tuple
        """
        # swmm_water_age_get_override_at
        row = self._wa.node_overrides[row_index]
        return (str(getattr(row.source, "name", row.source)), int(row.node_index), float(row.hours))

    def set_node_override(self, source: str, node: int | str, hours: float) -> None:
        # swmm_water_age_set_override — DWF / EXTERNAL_INFLOW only; negative
        # hours legal.
        self._wa.node_overrides.set(source, node, float(hours))

    def remove_node_override(self, source: str, node: int | str) -> None:
        # swmm_water_age_remove_override — addressed by (source, node), not
        # by row index (unlike the heat overrides).
        self._wa.node_overrides.remove(source, node)


class _ReactionsCompat:
    """Multi-species reaction accessor over the v6 ``Reactions`` API.

    Again a B{configuration} surface — the C API exposes no per-element
    species-concentration getter, so reaction results are not observable
    from a running solver. What it does expose, and what makes this worth
    surfacing at all, is the B{reaction coefficient} table: the
    C{[REACTION_COEFFICIENTS]} PARAMETER values are the calibration /
    optimisation handles of a multi-species model, and
    L{openswmm_gymnasium.spaces.design.ReactionCoefficientValue} searches
    over them.

    Coefficients come in two flavours, distinguished by
    L{is_coefficient_param}: PARAMETERs are intended to be varied;
    CONSTANTs are not. Searching a CONSTANT is refused at bind time rather
    than written and silently ignored.

    Species / coefficient / term enumeration is B{by name} — the engine's
    element wrapper objects never leave this module.
    """

    __slots__ = ("_rxn",)

    def __init__(self, reactions) -> None:
        self._rxn = reactions

    def species_names(self) -> list[str]:
        """Every declared species id, in engine order.

        @rtype: list[str]
        """
        # swmm_reaction_get_species_count / _get_species_name
        return [str(s.name) for s in self._rxn.species]

    def coefficient_names(self) -> list[str]:
        """Every declared reaction coefficient id, in engine order.

        @rtype: list[str]
        """
        # swmm_reaction_get_coeff_count / _get_coeff_name
        return [str(c.name) for c in self._rxn.coefficients]

    def term_names(self) -> list[str]:
        """Every declared intermediate-term id, in engine order.

        @rtype: list[str]
        """
        # swmm_reaction_get_term_count / _get_term_name
        return [str(t.name) for t in self._rxn.terms]

    def is_coefficient_param(self, name: str) -> bool:
        """Whether C{name} is a PARAMETER (searchable) or a CONSTANT.

        @rtype: bool
        """
        # swmm_reaction_get_coeff_is_param
        return bool(self._rxn.coefficients[name].is_param)

    def get_coefficient(self, name: str) -> float:
        # swmm_reaction_get_coeff_value — in whatever units the model's
        # reaction expressions assume; returned unconverted.
        return float(self._rxn.coefficients[name].value)

    def set_coefficient(self, name: str, value: float) -> None:
        # swmm_reaction_set_coeff_value — the calibration handle. Project /
        # expression units, unconverted.
        self._rxn.coefficients[name].value = float(value)

    def validate(self, expression: str, scope: str = "PIPE") -> tuple[bool, str, int]:
        """Compile-check one reaction expression without changing state.

        @param expression: Expression text.
        @type expression: str
        @param scope: Identifier-resolution vocabulary — C{"TERM"},
            C{"PIPE"} (default) or C{"TANK"}. A name string, so the
            engine's C{ReactionScope} enum never leaves this module.
        @type scope: str
        @return: C{(valid, message, column)}; C{column} is 1-based, or
            C{-1} when the diagnostic is not attributable.
        @rtype: tuple
        @raise ValueError: If C{scope} is not a C{ReactionScope} member name.
        """
        # swmm_reaction_validate_expression
        try:
            scope_enum = _engine.ReactionScope[str(scope).upper()]
        except KeyError as exc:
            raise ValueError(
                f"unknown reaction scope {scope!r}; expected one of "
                "'TERM', 'PIPE', 'TANK'"
            ) from exc
        diag = self._rxn.validate(expression, scope_enum)
        return (bool(diag.valid), str(diag.message), int(diag.column))


class _Surface2DCompat:
    """Read-only 2D surface accessor over ``Solver.surface2d``.

    Surfaces the small slice of :class:`openswmm.engine.Surface2D` the
    observation pipeline needs: activity flag, vertex and cell counts, and
    the bulk per-vertex render-depth read (the signed ``eta_v - z_v`` field
    used for inundation observation).
    """

    __slots__ = ("_surf",)

    def __init__(self, surf) -> None:
        self._surf = surf

    @property
    def is_active(self) -> bool:
        return bool(self._surf.is_active)

    @property
    def n_vertices(self) -> int:
        return int(self._surf.n_vertices)

    @property
    def n_cells(self) -> int:
        return int(self._surf.n_cells)

    def vertex_render_depths(self):
        # Bulk ndarray of shape (n_vertices,); GIL released engine-side.
        return self._surf.get_vertex_render_depths()


class SolverAdapter:
    """Thin lifecycle wrapper around L{openswmm.engine.Solver}.

    Construction does B{not} open the engine. Call L{open} (or use
    C{with adapter: ...}) to allocate the underlying C{SWMM_Engine}
    handle.

    Each C{SolverAdapter} owns a distinct C{SWMM_Engine} handle, so two
    adapters on different threads do not share C-side state. Plan §2.3
    (Threading & vectorized rollout).

    @ivar _inp: Resolved absolute path to the C{.inp} file.
    @type _inp: str
    @ivar _rpt: Resolved C{.rpt} path, or empty string if reporting
        is disabled.
    @type _rpt: str
    @ivar _out: Resolved C{.out} path, or empty string if binary
        output is disabled.
    @type _out: str
    @ivar _solver: The wrapped L{openswmm.engine.Solver} instance.
    @type _solver: L{openswmm.engine.Solver}
    @ivar _owned: C{True} if this adapter is responsible for closing
        the solver; C{False} if the solver was injected by the caller.
    @type _owned: bool
    """

    def __init__(
        self,
        inp: PathLike,
        rpt: PathLike | None = None,
        out: PathLike | None = None,
        *,
        solver: Solver | None = None,
    ) -> None:
        """Construct an adapter without opening the engine.

        @param inp: Path to the SWMM input file (C{.inp}).
        @type inp: str or C{os.PathLike}
        @param rpt: Path for the report file. C{None} skips reporting.
        @type rpt: str, C{os.PathLike}, or C{None}
        @param out: Path for the binary output file. C{None} skips.
        @type out: str, C{os.PathLike}, or C{None}
        @param solver: Optional pre-constructed
            L{openswmm.engine.Solver} to wrap. Used by tests and
            advanced wrappers. Per plan §0 Q1 (resolved), the adapter
            owns the solver lifecycle by default.
        @type solver: L{openswmm.engine.Solver} or C{None}
        @raise LegacySolverRejectedError: If C{solver} is a legacy v5
            singleton solver.
        @raise EngineCapabilityError: If the installed
            C{openswmm.engine} is missing a symbol this package requires.
        """
        require_engine_capabilities()
        self._inp = str(Path(inp))
        self._rpt = "" if rpt is None else str(Path(rpt))
        self._out = "" if out is None else str(Path(out))

        if solver is not None:
            _reject_legacy(solver)
            self._solver: Solver = solver
            self._owned = False
        else:
            self._solver = Solver(self._inp, self._rpt, self._out)
            self._owned = True

        self._opened = False
        self._closed = False
        self._nodes: Nodes | None = None
        self._links: Links | None = None
        self._controls: Controls | None = None
        self._subcatchments: Subcatchments | None = None
        self._infrastructure: _InfrastructureCompat | None = None
        self._inflows: _InflowsCompat | None = None
        self._pollutants: _PollutantsCompat | None = None
        self._tables: _TablesCompat | None = None
        self._statistics: _StatisticsCompat | None = None
        self._surface2d: _Surface2DCompat | None = None
        self._heat: _HeatCompat | None = None
        self._water_age: _WaterAgeCompat | None = None
        self._reactions: _ReactionsCompat | None = None

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, plugin_lib: PathLike | None = None, *, lenient: bool = False) -> None:
        """Open the input file and allocate the engine handle.

        Transitions the engine C{CREATED -> OPENED}. The v6 engine raises
        on failure rather than returning a status code.

        With C{lenient=True} the engine records post-parse validation errors
        (undefined objects, missing curves, bad references) but still leaves
        the model C{OPENED} and inspectable instead of hard-failing; read the
        accumulated messages via L{open_errors} / L{open_warnings}. This is a
        B{pre-flight validation} aid for training harnesses that
        programmatically generate or perturb C{.inp} models (CIP design search,
        domain randomization): open a candidate leniently, inspect the errors,
        and skip/report broken candidates instead of crashing the rollout.
        Per the engine contract a lenient open is B{not} runnable — a model to
        be stepped must be opened strictly (the default), so the env run path
        leaves C{lenient=False}.

        @param plugin_lib: Optional path to a plugin shared library.
        @type plugin_lib: str, C{os.PathLike}, or C{None}
        @param lenient: If C{True}, enable permissive open (see above).
            Defaults to C{False} (strict).
        @type lenient: bool
        @raise openswmm.engine.EngineError: On C API failure.
        """
        if self._opened:
            return
        if lenient:
            self._solver.set_lenient_open(True)
        if plugin_lib is None:
            self._solver.open()
        else:
            self._solver.open(str(plugin_lib))
        self._opened = True

    def initialize(self) -> None:
        """Initialize the simulation (transitions C{OPENED -> INITIALIZED}).

        Applies initial conditions. The simulation is not yet stepping;
        call L{start} to transition to C{RUNNING}.

        @raise openswmm.engine.EngineError: On C API failure.
        """
        self._solver.initialize()

    def start(self, save_results: bool = True) -> None:
        """Start the simulation (transitions C{INITIALIZED -> RUNNING}).

        Must be called after L{initialize} and before L{step}/L{stride};
        the engine guards C{step()} on the C{RUNNING} state.

        @param save_results: If C{True}, write binary output to the
            C{.out} file. When C{False} the file is not produced.
        @type save_results: bool
        @raise openswmm.engine.EngineError: On C API failure.
        """
        self._solver.start(save_results)

    def step(self) -> timedelta:
        """Advance one routing timestep.

        @return: Elapsed simulation time after the step. C{timedelta(0)}
            indicates the simulation has ended.
        @rtype: datetime.timedelta
        @raise openswmm.engine.EngineError: On C API failure.
        """
        return self._solver.step()

    def stride(self, n_steps: int) -> timedelta:
        """Advance C{n_steps} routing timesteps in one call.

        @param n_steps: Number of timesteps to advance.
        @type n_steps: int
        @return: Elapsed simulation time after the last step taken.
            C{timedelta(0)} once the simulation has ended.
        @rtype: datetime.timedelta
        @raise openswmm.engine.EngineError: On C API failure.
        """
        return self._solver.stride(n_steps)

    def end(self) -> None:
        """End the simulation (transitions to C{ENDED})."""
        self._solver.end()

    def report(self) -> None:
        """Write the report file."""
        self._solver.report()

    def close(self) -> None:
        """Close the engine handle. Idempotent.

        @note: Safe to call multiple times. Closes only when this
            adapter owns the solver (i.e. constructed it internally).
        """
        if self._closed or not self._owned:
            return
        try:
            self._solver.close()
        finally:
            self._closed = True
            self._opened = False

    # ------------------------------------------------------------------
    # Deterministic state seeding
    # ------------------------------------------------------------------

    def seed_hotstart(
        self,
        path,
        node_depths: dict[str, float] | None = None,
        node_heads: dict[str, float] | None = None,
        link_depths: dict[str, float] | None = None,
        link_flows: dict[str, float] | None = None,
        subcatchment_runoffs: dict[str, float] | None = None,
    ) -> None:
        """Apply a hot-start file, optionally overriding element states.

        Opens the hot-start file at C{path}, overrides individual element
        states from the supplied id→value maps (surfacing the engine's
        hot-start state setters), and applies the result to this adapter's
        solver. Enables deterministic, reproducible episode initial
        conditions for RL.

        All values are in the model's project units (see L{unit_system}):
        node depths/heads in project length units; link depths in project
        length units; link flows and subcatchment runoff in project flow
        units.

        @param path: Hot-start file to open as the seed baseline.
        @type path: str or os.PathLike
        @param node_depths: Map of node id → depth override.
        @param node_heads: Map of node id → head override.
        @param link_depths: Map of link id → depth override.
        @param link_flows: Map of link id → flow override.
        @param subcatchment_runoffs: Map of subcatchment id → runoff override.
        @raise openswmm.engine.EngineError: On C API failure.
        """
        hotstart = HotStart.open(str(path))
        for nid, v in (node_depths or {}).items():
            hotstart.set_node_depth(nid, float(v))
        for nid, v in (node_heads or {}).items():
            hotstart.set_node_head(nid, float(v))
        for lid, v in (link_depths or {}).items():
            hotstart.set_link_depth(lid, float(v))
        for lid, v in (link_flows or {}).items():
            hotstart.set_link_flow(lid, float(v))
        for sid, v in (subcatchment_runoffs or {}).items():
            hotstart.set_subcatchment_runoff(sid, float(v))
        hotstart.apply(self._solver)

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    @property
    def state(self) -> EngineState:
        """Current engine state.

        @rtype: L{openswmm.engine.EngineState}
        """
        return self._solver.state

    @property
    def is_running(self) -> bool:
        """Whether the simulation is still advancing.

        @rtype: bool
        """
        return self._solver.state == EngineState.RUNNING

    @property
    def elapsed(self) -> float:
        """Elapsed simulation time in days.

        The v6 engine exposes elapsed time as a :class:`datetime.timedelta`;
        callers in this package expect decimal days, so convert here.

        @rtype: float
        """
        e = self._solver.elapsed
        if isinstance(e, timedelta):
            return e.total_seconds() / 86400.0
        return e

    @property
    def flow_units(self) -> str:
        """The model's flow-unit token, e.g. C{"CFS"} / C{"CMS"}.

        Returned as the upper-case token name. C{Solver.flow_units} is a
        probed requirement (see L{require_engine_capabilities}), so there
        is no fallback path.

        @rtype: str
        """
        fu = self._solver.flow_units
        # FlowUnits enum -> its member name (e.g. "CFS").
        return getattr(fu, "name", str(fu)).upper()

    @property
    def unit_system(self) -> str:
        """C{"US"} (CFS/GPM/MGD) or C{"SI"} (CMS/LPS/MLD).

        Every engine getter returns project units, so an env that scales
        observations or rewards by physical magnitudes must know which
        system is active. Recorded at L{reset} so a CMS model never
        silently reuses CFS-tuned scaling.

        @rtype: str
        """
        return str(self._solver.unit_system)

    @property
    def open_errors(self) -> list[str]:
        """Post-parse validation errors accumulated during L{open}.

        Populated primarily after a lenient open (L{open}C{(lenient=True)});
        a strict open that succeeds leaves this empty. Each entry is a
        human-readable message string. Surfaces the engine's
        C{Solver.open_errors} accumulator so a training harness can report or
        reject a broken candidate model.

        @rtype: list[str]
        """
        return list(self._solver.open_errors)

    @property
    def open_warnings(self) -> list[str]:
        """Warnings accumulated during L{open}.

        Populated on either a strict or a lenient open. Each entry is a
        human-readable message string. Surfaces the engine's
        C{Solver.open_warnings} accumulator.

        @rtype: list[str]
        """
        return list(self._solver.open_warnings)

    @property
    def start_time(self) -> float:
        """Simulation start time as an OADate (decimal days).

        @rtype: float
        """
        return _to_oadate(self._solver.start_datetime)

    @property
    def end_time(self) -> float:
        """Simulation end time as an OADate (decimal days).

        @rtype: float
        """
        return _to_oadate(self._solver.end_datetime)

    @property
    def current_time(self) -> float:
        """Current simulation time as an OADate (decimal days).

        @rtype: float
        """
        return _to_oadate(self._solver.current_datetime)

    # ------------------------------------------------------------------
    # Accessors (lazy, cached)
    # ------------------------------------------------------------------

    @property
    def solver(self) -> Solver:
        """Underlying real L{openswmm.engine.Solver}. Use sparingly.

        @rtype: L{openswmm.engine.Solver}
        """
        return self._solver

    @property
    def surface2d(self) -> _Surface2DCompat:
        """Lazily-constructed, cached 2D surface accessor.

        Raises with a clear message when the engine build carries no 2D
        module or the open model has no active 2D surface (no C{[2D_*]}
        sections, or the C{IGNORE_2D} gate is on) — 2D observations
        cannot be combined with 1D-only episodes.

        @rtype: L{_Surface2DCompat}
        @raise RuntimeError: If the 2D surface is unavailable.
        """
        if self._surface2d is None:
            try:
                surf = self._solver.surface2d
            except ImportError as exc:
                raise RuntimeError(
                    "This openswmm.engine build has no 2D module "
                    "(OPENSWMM_BUILD_2D=OFF); 2D observations are unavailable."
                ) from exc
            compat = _Surface2DCompat(surf)
            if not compat.is_active:
                raise RuntimeError(
                    "The open model has no active 2D surface (no [2D_*] "
                    "sections, or IGNORE_2D is enabled); remove the 2D "
                    "collectors or run with the 2D module on."
                )
            self._surface2d = compat
        return self._surface2d

    @property
    def heat(self) -> _HeatCompat:
        """Lazily-constructed, cached heat-transport accessor.

        B{Two-tier guard}, mirroring L{surface2d}:

          1. B{Build tier} — an C{openswmm.engine} without a C{Heat} class
             (an older wheel, or a build with the heat component off)
             raises naming the remedy.
          2. B{Model tier} — an open model whose C{[OPTIONS]} lacks
             C{HEAT_TRANSPORT YES} raises rather than accepting writes the
             engine will parse but never route. A heat configuration
             applied to a model that does not run heat transport is a
             B{silent} no-op: the source temperatures are stored, nothing
             errors, and no reward signal ever moves. Better to fail loudly
             at bind than to train against a dead actuator.

        @rtype: L{_HeatCompat}
        @raise RuntimeError: If the engine build has no heat module, or the
            open model does not enable heat transport.
        """
        if self._heat is None:
            heat_cls = getattr(_engine, "Heat", None)
            if heat_cls is None:
                raise RuntimeError(
                    "This openswmm.engine build has no Heat module "
                    "(openswmm.engine.Heat is absent); heat-transport "
                    "factories and actuators are unavailable. Upgrade the "
                    "openswmm package, or rebuild the engine with the heat "
                    "component enabled."
                )
            compat = _HeatCompat(heat_cls(self._solver))
            if not compat.enabled:
                raise RuntimeError(
                    "The open model does not run heat transport: its "
                    "[OPTIONS] section has no 'HEAT_TRANSPORT YES'. Heat "
                    "source temperatures would be stored and never routed, "
                    "so this accessor refuses rather than returning "
                    "defaults. Enable HEAT_TRANSPORT in the .inp, or remove "
                    "the heat factories/actuators from the config."
                )
            self._heat = compat
        return self._heat

    @property
    def water_age(self) -> _WaterAgeCompat:
        """Lazily-constructed, cached water-age accessor.

        Same two-tier guard as L{heat}: build tier (no C{WaterAge} class in
        this engine build) then model tier (the open model's C{[OPTIONS]}
        lacks C{WATER_AGE YES}, so configured source ages would be stored
        and never transported).

        @rtype: L{_WaterAgeCompat}
        @raise RuntimeError: If the engine build has no water-age module, or
            the open model does not enable water age.
        """
        if self._water_age is None:
            wa_cls = getattr(_engine, "WaterAge", None)
            if wa_cls is None:
                raise RuntimeError(
                    "This openswmm.engine build has no WaterAge module "
                    "(openswmm.engine.WaterAge is absent); water-age "
                    "factories are unavailable. Upgrade the openswmm "
                    "package, or rebuild the engine with the water-age "
                    "component enabled."
                )
            compat = _WaterAgeCompat(wa_cls(self._solver))
            if not compat.enabled:
                raise RuntimeError(
                    "The open model does not run water age: its [OPTIONS] "
                    "section has no 'WATER_AGE YES'. Configured source ages "
                    "would be stored and never transported, so this accessor "
                    "refuses rather than returning defaults. Enable "
                    "WATER_AGE in the .inp, or remove the water-age "
                    "factories from the config."
                )
            self._water_age = compat
        return self._water_age

    @property
    def reactions(self) -> _ReactionsCompat:
        """Lazily-constructed, cached multi-species reaction accessor.

        B{One-tier guard only.} Unlike L{heat} / L{water_age} there is no
        C{[OPTIONS]} flag that turns reactions on: a model either declares
        species and coefficients or it does not, and an empty coefficient
        table is a legitimate (if useless) state rather than a
        misconfiguration. The model-level check therefore belongs to the
        consumer —
        L{openswmm_gymnasium.spaces.design.ReactionCoefficientValue} raises
        at bind when a named coefficient is absent, which is the specific,
        actionable version of the same question.

        @rtype: L{_ReactionsCompat}
        @raise RuntimeError: If the engine build has no reactions module.
        """
        if self._reactions is None:
            rxn_cls = getattr(_engine, "Reactions", None)
            if rxn_cls is None:
                raise RuntimeError(
                    "This openswmm.engine build has no Reactions module "
                    "(openswmm.engine.Reactions is absent); reaction "
                    "coefficient search is unavailable. Upgrade the "
                    "openswmm package, or rebuild the engine with the "
                    "multi-species reaction component enabled."
                )
            self._reactions = _ReactionsCompat(rxn_cls(self._solver))
        return self._reactions

    def get_option(self, name: str) -> str:
        """Read one C{[OPTIONS]} entry from the open model.

        Thin pass-through to the engine's C{Solver.options} mapping, the
        read counterpart of L{set_option}. Used to answer model-level
        questions a caller cannot otherwise ask — notably which router is
        active (C{get_option("ROUTING_MODEL")}), which decides whether the
        Preissmann-slot link readers can produce a signal at all. Call
        after L{open}.

        @param name: Option keyword as it appears in C{[OPTIONS]}.
        @type name: str
        @return: The option's value in its string form.
        @rtype: str
        @raise KeyError: If the engine mapping does not carry C{name}.
        @raise openswmm.engine.EngineError: If the engine rejects the key.
        """
        return str(self._solver.options[name])

    def set_option(self, name: str, value) -> None:
        """Set one C{[OPTIONS]} entry on the open model (pre-initialize).

        Thin pass-through to the engine's C{Solver.options} mapping —
        e.g. C{set_option("IGNORE_2D", "YES")} runs a meshed model
        1D-only for cheap training episodes. Call after L{open} and
        before L{initialize}.

        @param name: Option keyword as it appears in C{[OPTIONS]}.
        @type name: str
        @param value: Option value; converted to its string form by the
            engine mapping.
        @raise openswmm.engine.EngineError: If the engine rejects the
            key or value.
        """
        self._solver.options[name] = value

    # -- Catalog field access ------------------------------------------------
    #
    # Any numeric element property the engine exposes, addressed by its
    # catalog path. Observation collectors and runtime actuators use these,
    # so a new engine field is observable/actuatable without new adapter code.

    def index(self, kind: str, element_id: str) -> int:
        """Engine index of element C{element_id} of catalog kind C{kind} (C{"node"}, ...).

        @rtype: int
        """
        collection = _catalog.targets()[kind]["collection"]
        return int(_catalog.resolve(self._solver, collection, None).get_index(element_id))

    def read(self, path: str, idx: int) -> Any:
        """Value of field C{path} (e.g. C{"link.stats.max_flow"}) on element C{idx}."""
        entry = _catalog.lookup(path)
        return getattr(_catalog.resolve(self._solver, entry["target"], idx), entry["name"])

    def read_all(self, path: str):
        """Whole-collection array for C{path} when the engine has a bulk getter, else C{None}.

        @rtype: numpy.ndarray or None
        """
        bulk = _catalog.lookup(path).get("bulk")
        if bulk is None:
            return None
        target, _, name = bulk.rpartition(".")
        return getattr(_catalog.resolve(self._solver, target, None), name)

    def call(self, path: str, *args: Any, **kwargs: Any) -> Any:
        """Call catalog method C{path} on its service (e.g. C{"surface2d.get_depths"}).

        @rtype: Any
        """
        entry = _catalog.lookup(path)
        return getattr(_catalog.resolve(self._solver, entry["target"], None), entry["name"])(
            *args, **kwargs
        )

    def write(self, path: str, idx: int, value: Any) -> None:
        """Set field C{path} on element C{idx} (the field must be writable)."""
        entry = _catalog.lookup(path)
        if entry.get("access") != "rw":
            raise ValueError(f"{path!r} is read-only")
        setattr(_catalog.resolve(self._solver, entry["target"], idx), entry["name"], value)

    @property
    def nodes(self) -> _NodesCompat:
        """Lazily-constructed, cached scalar node accessor.

        @rtype: L{_NodesCompat}
        """
        if self._nodes is None:
            self._nodes = _NodesCompat(self._solver, Nodes(self._solver))
        return self._nodes

    @property
    def links(self) -> _LinksCompat:
        """Lazily-constructed, cached scalar link accessor.

        @rtype: L{_LinksCompat}
        """
        if self._links is None:
            self._links = _LinksCompat(self._solver, Links(self._solver))
        return self._links

    @property
    def controls(self) -> _ControlsCompat:
        """Lazily-constructed, cached scalar control accessor.

        @rtype: L{_ControlsCompat}
        """
        if self._controls is None:
            self._controls = _ControlsCompat(Controls(self._solver))
        return self._controls

    @property
    def subcatchments(self) -> _SubcatchmentsCompat:
        """Lazily-constructed, cached scalar subcatchment accessor.

        @rtype: L{_SubcatchmentsCompat}
        """
        if self._subcatchments is None:
            self._subcatchments = _SubcatchmentsCompat(self._solver, Subcatchments(self._solver))
        return self._subcatchments


    @property
    def infrastructure(self) -> _InfrastructureCompat:
        """Lazily-constructed, cached LID / green-infrastructure editor.

        @rtype: L{_InfrastructureCompat}
        """
        if self._infrastructure is None:
            self._infrastructure = _InfrastructureCompat(Infrastructure(self._solver))
        return self._infrastructure

    @property
    def inflows(self) -> _InflowsCompat:
        """Lazily-constructed, cached inflow / RDII editor.

        @rtype: L{_InflowsCompat}
        """
        if self._inflows is None:
            self._inflows = _InflowsCompat(Inflows(self._solver))
        return self._inflows

    @property
    def pollutants(self) -> _PollutantsCompat:
        """Lazily-constructed, cached pollutant catalogue accessor.

        @rtype: L{_PollutantsCompat}
        """
        if self._pollutants is None:
            self._pollutants = _PollutantsCompat(self._solver, _engine.Pollutants(self._solver))
        return self._pollutants

    @property
    def tables(self) -> _TablesCompat:
        """Lazily-constructed, cached curve / time-series table accessor.

        @rtype: L{_TablesCompat}
        """
        if self._tables is None:
            self._tables = _TablesCompat(_engine.Tables(self._solver))
        return self._tables

    @property
    def statistics(self) -> _StatisticsCompat:
        """Lazily-constructed, cached simulation-statistics accessor.

        @rtype: L{_StatisticsCompat}
        """
        if self._statistics is None:
            self._statistics = _StatisticsCompat(_engine.Statistics(self._solver))
        return self._statistics

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SolverAdapter:
        self.open()
        self.initialize()
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        try:
            if self._opened and not self._closed:
                self.end()
                self.report()
        finally:
            self.close()

    def __del__(self) -> None:
        # Best-effort cleanup if the user forgets to close. Safe because
        # close() is idempotent.
        try:
            self.close()
        except Exception:
            pass
