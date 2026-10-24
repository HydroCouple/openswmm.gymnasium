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
  - Element access by L{openswmm.engine.catalog} path (L{SolverAdapter.index},
    L{SolverAdapter.read}, L{SolverAdapter.read_all}, L{SolverAdapter.write},
    L{SolverAdapter.call}); anything that is not one catalog field uses the
    real engine API on L{SolverAdapter.solver}.
  - Hard guard against C{openswmm.legacy.engine.Solver} — passing a
    legacy solver into the adapter raises
    L{LegacySolverRejectedError}. Plan §0 #7 / §2.3.
  - The engine B{capability probe} (L{require_engine_capabilities}, in
    L{requirements<openswmm_gymnasium._engine.requirements>}) run at
    adapter construction.

The adapter does B{not} add any high-level domain logic (observations,
rewards, action application). Those live in their respective modules
and consume a L{SolverAdapter} instance.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

from openswmm.engine import EngineState, HotStart, Solver
from openswmm.engine import catalog as _catalog

from openswmm_gymnasium._engine.requirements import require_engine_capabilities

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

    @property
    def solver(self) -> Solver:
        """Underlying real L{openswmm.engine.Solver}.

        For operations that are not a single catalog field — cross-sections,
        storage curves, hydrographs, LID usages, heat / water-age sources.
        Prefer L{read} / L{write} / L{call} for anything addressable by a
        catalog path, and L{optional_module} for the optional modules.

        @rtype: L{openswmm.engine.Solver}
        """
        return self._solver

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
    # Any element property or service method the engine exposes, addressed by
    # its catalog path. Observation collectors, actuators, design factories and
    # reward terms use these, so a new engine field is observable/actuatable
    # without new adapter code.

    def index(self, kind: str, element_id: str) -> int:
        """Engine index of element C{element_id} of catalog kind C{kind} (C{"node"}, ...).

        @rtype: int
        """
        collection = _catalog.targets()[kind]["collection"]
        return int(_catalog.resolve(self._solver, collection, None).get_index(element_id))

    def read(self, path: str, idx: int | str) -> Any:
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

    def write(self, path: str, idx: int | str, value: Any) -> None:
        """Set field C{path} on element C{idx} (the field must be writable)."""
        entry = _catalog.lookup(path)
        if entry.get("access") != "rw":
            raise ValueError(f"{path!r} is read-only")
        setattr(_catalog.resolve(self._solver, entry["target"], idx), entry["name"], value)

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
