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

The adapter does B{not} add any high-level domain logic (observations,
rewards, action application). Those live in their respective modules
and consume a L{SolverAdapter} instance.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path

from openswmm.engine import (
    Controls,
    EngineState,
    Gages,
    HotStart,
    Infrastructure,
    Inflows,
    LidType,
    Links,
    Nodes,
    Solver,
    Subcatchments,
)

PathLike = str | os.PathLike

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
# Collection compatibility shims
# ---------------------------------------------------------------------------
#
# The v6 engine exposes per-object element wrappers (``solver.nodes[idx].depth``,
# ``solver.links[idx].roughness = v``) plus vectorized array properties, rather
# than the scalar ``get_<x>(idx)`` / ``set_<x>(idx, v)`` collection methods the
# rest of this package was written against. These thin shims re-expose that
# scalar surface on top of the element-object API so observation builders,
# reward terms, and design/runtime action factories need not change.


class _NodesCompat:
    """Scalar node accessor over the v6 element-object ``Nodes`` collection."""

    __slots__ = ("_col",)

    def __init__(self, col: Nodes) -> None:
        self._col = col

    def count(self) -> int:
        return len(self._col.ids)

    def get_index(self, node_id: str) -> int:
        return self._col.get_index(node_id)

    def get_depth(self, idx: int) -> float:
        return self._col[idx].depth

    def get_head(self, idx: int) -> float:
        return self._col[idx].head

    def get_inflow(self, idx: int) -> float:
        return self._col[idx].inflow

    def get_overflow(self, idx: int) -> float:
        return self._col[idx].overflow

    def get_volume(self, idx: int) -> float:
        return self._col[idx].volume

    def get_lateral_inflow(self, idx: int) -> float:
        return self._col[idx].lateral_inflow

    def set_lateral_inflow(self, idx: int, value: float) -> None:
        # Controllable externally-applied inflow (swmm_node_set_lateral_inflow);
        # value is in project flow units.
        self._col[idx].lateral_inflow = value

    def set_head_boundary(self, idx: int, value: float) -> None:
        # Fixed head boundary (swmm_node_set_head_boundary); project length units.
        self._col[idx].set_head_boundary(value)

    def get_max_depth(self, idx: int) -> float:
        return self._col[idx].max_depth

    def set_max_depth(self, idx: int, value: float) -> None:
        self._col[idx].max_depth = value

    def get_storage_functional(self, idx: int) -> tuple[float, float, float]:
        # (a, b, c) of the FUNCTIONAL storage relation Area = a*Depth^b + c.
        # Only valid on STORAGE nodes whose shape is FUNCTIONAL.
        return tuple(self._col[idx].storage.functional)

    def set_storage_functional(self, idx: int, a: float, b: float, c: float) -> None:
        # swmm_node_set_storage_functional; STORAGE + FUNCTIONAL shape only.
        self._col[idx].storage.functional = (a, b, c)

    def array(self, name: str):
        """Return a whole-network bulk array property of the node collection.

        Exposes the engine's vectorized getters (e.g. ``depths``, ``heads``,
        ``inflows``, ``overflows``, ``volumes``, ``lateral_inflows``) so a
        collector can fetch all values in one FFI call instead of N scalar
        round-trips.

        @param name: Bulk property name on the engine ``Nodes`` collection.
        @rtype: numpy.ndarray
        """
        return getattr(self._col, name)


class _LinksCompat:
    """Scalar link accessor over the v6 element-object ``Links`` collection."""

    __slots__ = ("_col",)

    def __init__(self, col: Links) -> None:
        self._col = col

    def count(self) -> int:
        return len(self._col.ids)

    def get_index(self, link_id: str) -> int:
        return self._col.get_index(link_id)

    def get_flow(self, idx: int) -> float:
        return self._col[idx].flow

    def get_depth(self, idx: int) -> float:
        return self._col[idx].depth

    def get_control_setting(self, idx: int) -> float:
        return self._col[idx].control_setting

    def get_target_setting(self, idx: int) -> float:
        return self._col[idx].target_setting

    def set_target_setting(self, idx: int, value: float) -> None:
        # Persistent runtime-control override: the engine moves the link's
        # control_setting toward target_setting each routing step and holds it
        # there. (control_setting set via Controls.set_link_setting is recomputed
        # from the target every step, so it does not stick without a rule.)
        self._col[idx].target_setting = value

    def get_velocity(self, idx: int) -> float:
        return self._col[idx].velocity

    def get_capacity(self, idx: int) -> float:
        return self._col[idx].capacity

    def get_volume(self, idx: int) -> float:
        return self._col[idx].volume

    def set_roughness(self, idx: int, value: float) -> None:
        self._col[idx].roughness = value

    def set_length(self, idx: int, value: float) -> None:
        self._col[idx].length = value

    def get_xsect(self, idx: int) -> tuple:
        # (shape, geom1, geom2, geom3, geom4); shape is an XSectShape enum,
        # consumers coerce it with int().
        return self._col[idx].xsect.as_tuple()

    def set_xsect(
        self, idx: int, shape: int, g1: float, g2: float, g3: float, g4: float
    ) -> None:
        # The v6 xsect setter accepts a (shape, g1, g2, g3, g4) tuple.
        self._col[idx].xsect = (shape, g1, g2, g3, g4)

    def array(self, name: str):
        """Return a whole-network bulk array property of the link collection.

        Exposes the engine's vectorized getters (e.g. ``flows``, ``depths``,
        ``velocities``, ``capacities``, ``volumes``) for single-call reads.

        @param name: Bulk property name on the engine ``Links`` collection.
        @rtype: numpy.ndarray
        """
        return getattr(self._col, name)


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


class _SubcatchmentsCompat:
    """Scalar subcatchment accessor over the v6 ``Subcatchments`` collection."""

    __slots__ = ("_col",)

    def __init__(self, col: Subcatchments) -> None:
        self._col = col

    def get_index(self, sub_id: str) -> int:
        return self._col.get_index(sub_id)

    def get_runoff(self, idx: int) -> float:
        return self._col[idx].runoff

    def get_groundwater(self, idx: int) -> float:
        # Groundwater outflow rate (swmm_subcatch_get_groundwater); project
        # flow units. 0.0 on subcatchments without an aquifer.
        return self._col[idx].groundwater

    def get_gw_params(self, idx: int) -> tuple:
        # (surf_elev, a1, b1, a2, b2, a3, tw, hstar) — [GROUNDWATER] token
        # order; requires an aquifer to be assigned.
        return tuple(self._col[idx].gw_params)

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


class _GagesCompat:
    """Scalar rain-gage accessor over the v6 ``Gages`` collection."""

    __slots__ = ("_col",)

    def __init__(self, col: Gages) -> None:
        self._col = col

    def get_index(self, gage_id: str) -> int:
        return self._col.get_index(gage_id)

    def get_rainfall(self, idx: int) -> float:
        return self._col[idx].rainfall


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
        """
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
        self._gages: Gages | None = None
        self._infrastructure: _InfrastructureCompat | None = None
        self._inflows: _InflowsCompat | None = None

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

        Prefers the engine's C{Solver.flow_units} property; falls back to
        the raw C{FLOW_UNITS} option string for engine builds that predate
        that accessor. Returned as the upper-case token name.

        @rtype: str
        """
        fu = getattr(self._solver, "flow_units", None)
        if fu is not None:
            # FlowUnits enum -> its member name (e.g. "CFS").
            return getattr(fu, "name", str(fu)).upper()
        return self._solver.options["FLOW_UNITS"].strip().upper()

    @property
    def unit_system(self) -> str:
        """C{"US"} (CFS/GPM/MGD) or C{"SI"} (CMS/LPS/MLD).

        Every engine getter returns project units, so an env that scales
        observations or rewards by physical magnitudes must know which
        system is active. Recorded at L{reset} so a CMS model never
        silently reuses CFS-tuned scaling.

        @rtype: str
        """
        us = getattr(self._solver, "unit_system", None)
        if us is not None:
            return str(us)
        return "US" if self.flow_units in ("CFS", "GPM", "MGD") else "SI"

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
    def nodes(self) -> _NodesCompat:
        """Lazily-constructed, cached scalar node accessor.

        @rtype: L{_NodesCompat}
        """
        if self._nodes is None:
            self._nodes = _NodesCompat(Nodes(self._solver))
        return self._nodes

    @property
    def links(self) -> _LinksCompat:
        """Lazily-constructed, cached scalar link accessor.

        @rtype: L{_LinksCompat}
        """
        if self._links is None:
            self._links = _LinksCompat(Links(self._solver))
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
            self._subcatchments = _SubcatchmentsCompat(Subcatchments(self._solver))
        return self._subcatchments

    @property
    def gages(self) -> _GagesCompat:
        """Lazily-constructed, cached scalar rain-gage accessor.

        @rtype: L{_GagesCompat}
        """
        if self._gages is None:
            self._gages = _GagesCompat(Gages(self._solver))
        return self._gages

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
