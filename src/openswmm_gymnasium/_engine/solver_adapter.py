"""
Solver lifecycle bridge.

Wraps L{openswmm.engine.Solver} (the handle-based v6 engine) to expose
the minimum surface the rest of the package needs:

  - Lifecycle: L{SolverAdapter.open}, L{SolverAdapter.initialize},
    L{SolverAdapter.step}, L{SolverAdapter.stride},
    L{SolverAdapter.end}, L{SolverAdapter.report},
    L{SolverAdapter.close}.
  - Idempotent L{SolverAdapter.close} / context-manager protocol.
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
from pathlib import Path

from openswmm.engine import Controls, EngineState, Gages, Links, Nodes, Solver, Subcatchments

PathLike = str | os.PathLike


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

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def open(self, plugin_lib: PathLike | None = None) -> None:
        """Open the input file and allocate the engine handle.

        @param plugin_lib: Optional path to a plugin shared library.
        @type plugin_lib: str, C{os.PathLike}, or C{None}
        @raise RuntimeError: If the underlying C API returns a non-zero
            code.
        """
        if self._opened:
            return
        rc = self._solver.open() if plugin_lib is None else self._solver.open(str(plugin_lib))
        if rc != 0:
            raise RuntimeError(f"Solver.open returned non-zero code {rc}")
        self._opened = True

    def initialize(self) -> None:
        """Initialize the simulation (transitions C{OPENED} -> C{RUNNING}).

        @raise RuntimeError: If the underlying C API returns a non-zero
            code.
        """
        rc = self._solver.initialize()
        if rc != 0:
            raise RuntimeError(f"Solver.initialize returned non-zero code {rc}")

    def step(self) -> int:
        """Advance one routing timestep.

        @return: Engine return code (C{0} on success).
        @rtype: int
        """
        return self._solver.step()

    def stride(self, n_steps: int) -> int:
        """Advance C{n_steps} routing timesteps in one call.

        @param n_steps: Number of timesteps to advance.
        @type n_steps: int
        @return: Engine return code (C{0} on success).
        @rtype: int
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

        @rtype: float
        """
        return self._solver.elapsed

    @property
    def start_time(self) -> float:
        """Simulation start time as an OADate (decimal days).

        @rtype: float
        """
        return self._solver.get_start_time()

    @property
    def end_time(self) -> float:
        """Simulation end time as an OADate (decimal days).

        @rtype: float
        """
        return self._solver.get_end_time()

    @property
    def current_time(self) -> float:
        """Current simulation time as an OADate (decimal days).

        @rtype: float
        """
        return self._solver.get_current_time()

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
    def nodes(self) -> Nodes:
        """Lazily-constructed, cached L{openswmm.engine.Nodes} accessor.

        @rtype: L{openswmm.engine.Nodes}
        """
        if self._nodes is None:
            self._nodes = Nodes(self._solver)
        return self._nodes

    @property
    def links(self) -> Links:
        """Lazily-constructed, cached L{openswmm.engine.Links} accessor.

        @rtype: L{openswmm.engine.Links}
        """
        if self._links is None:
            self._links = Links(self._solver)
        return self._links

    @property
    def controls(self) -> Controls:
        """Lazily-constructed, cached L{openswmm.engine.Controls} accessor.

        @rtype: L{openswmm.engine.Controls}
        """
        if self._controls is None:
            self._controls = Controls(self._solver)
        return self._controls

    @property
    def subcatchments(self) -> Subcatchments:
        """Lazily-constructed, cached L{openswmm.engine.Subcatchments} accessor.

        @rtype: L{openswmm.engine.Subcatchments}
        """
        if self._subcatchments is None:
            self._subcatchments = Subcatchments(self._solver)
        return self._subcatchments

    @property
    def gages(self) -> Gages:
        """Lazily-constructed, cached L{openswmm.engine.Gages} accessor.

        @rtype: L{openswmm.engine.Gages}
        """
        if self._gages is None:
            self._gages = Gages(self._solver)
        return self._gages

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> SolverAdapter:
        self.open()
        self.initialize()
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
