"""Unit tests for L{openswmm_gymnasium._engine.SolverAdapter}.

Per plan §8.0, these run against the B{real}
L{openswmm.engine.Solver} — no mocks.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import unittest
from datetime import timedelta

from openswmm.engine import EngineError, EngineState, Solver

from openswmm_gymnasium._engine import LegacySolverRejectedError, SolverAdapter
from tests.unit._base import BaseEngineTest

# Subcatchment S1 drains to an undefined outlet node: ERROR 209, raised by
# post-parse cross-reference resolution. (An undefined node named in
# [CONDUITS] would not work here — the reader accepts it silently and records
# no error at all.)
_BROKEN_INP = (
    "[TITLE]\nLenient open fixture\n\n"
    "[OPTIONS]\n"
    "FLOW_UNITS           CFS\n"
    "START_DATE           01/01/2020\n"
    "END_DATE             01/01/2020\n"
    "\n"
    "[JUNCTIONS]\n"
    ";;Name  Elev  MaxDepth\n"
    "J1       0     4\n"
    "\n"
    "[OUTFALLS]\n"
    ";;Name  Elev  Type\n"
    "O1       0     FREE\n"
    "\n"
    "[CONDUITS]\n"
    ";;Name  From  To   Length  Rough  In  Out\n"
    "C1       J1    O1   100     0.01   0   0\n"
    "\n"
    "[XSECTIONS]\n"
    ";;Link  Shape     G1  G2  G3  G4\n"
    "C1       CIRCULAR  1   0   0   0\n"
    "\n"
    "[SUBCATCHMENTS]\n"
    ";;Name  Rgage  Outlet        Area  %Imperv  Width  Slope  CurbLen\n"
    "S1       RG1    UNDEFINED_ND  1     50       100    0.5    0\n"
    "\n"
    "[SUBAREAS]\n"
    ";;Subcatch  N-Imperv  N-Perv  S-Imperv  S-Perv  PctZero  RouteTo\n"
    "S1          0.01      0.1     0.05      0.05    25       OUTLET\n"
    "\n"
    "[INFILTRATION]\n"
    ";;Subcatch  P1    P2    P3\n"
    "S1          3.0   0.5   4\n"
    "\n"
    "[RAINGAGES]\n"
    ";;Name  Format     Interval  SCF  Source\n"
    "RG1      INTENSITY  1:00      1.0  TIMESERIES TS1\n"
    "\n"
    "[TIMESERIES]\n"
    "TS1  01/01/2020  00:00  0.0\n"
)

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction(BaseEngineTest):
    def test_construction_does_not_open_engine(self):
        """Constructor must be cheap — no engine handle allocated until open()."""
        adapter = SolverAdapter(self.minimal_inp, self.minimal_rpt, self.minimal_out)
        # No exception here means handle isn't allocated yet.
        # state is queryable only after open(); we don't probe it.
        self.assertTrue(adapter._owned)

    def test_optional_rpt_out(self):
        """rpt and out may be None (engine skips file writing)."""
        adapter = SolverAdapter(self.minimal_inp, None, None)
        self.assertEqual(adapter._rpt, "")
        self.assertEqual(adapter._out, "")

    def test_path_like_inputs(self):
        """str and Path inputs both accepted."""
        a1 = SolverAdapter(str(self.minimal_inp), str(self.minimal_rpt), str(self.minimal_out))
        a2 = SolverAdapter(self.minimal_inp, self.minimal_rpt, self.minimal_out)
        self.assertEqual(a1._inp, a2._inp)


# ---------------------------------------------------------------------------
# Legacy solver guard (plan §0 #7)
# ---------------------------------------------------------------------------


class TestLegacyGuard(BaseEngineTest):
    def test_rejects_non_engine_solver(self):
        """A non-openswmm.engine object passed via solver= must raise."""

        class FakeLegacySolver:
            """Pretends to be openswmm.legacy.engine.Solver."""

        FakeLegacySolver.__module__ = "openswmm.legacy.engine._solver"
        fake = FakeLegacySolver()
        with self.assertRaises(LegacySolverRejectedError) as cm:
            SolverAdapter(self.minimal_inp, solver=fake)
        self.assertIn("legacy", str(cm.exception).lower())

    def test_accepts_handle_based_solver(self):
        """A real openswmm.engine.Solver is accepted via solver=."""
        real = Solver(str(self.minimal_inp), str(self.minimal_rpt), str(self.minimal_out))
        # Just construction; no open() call so this works even if the
        # engine wheel SIGILLs at C-side init.
        adapter = SolverAdapter(self.minimal_inp, self.minimal_rpt, self.minimal_out, solver=real)
        self.assertFalse(adapter._owned)
        self.assertIs(adapter.solver, real)


# ---------------------------------------------------------------------------
# Lifecycle (requires a working engine wheel on the host)
# ---------------------------------------------------------------------------


class TestLifecycle(BaseEngineTest):
    def test_open_initialize_step_close(self):
        """Full happy-path lifecycle through the adapter."""
        adapter = self.make_adapter(open=True)
        self.assertEqual(adapter.state, EngineState.RUNNING)
        # One step advances elapsed time. step() returns the elapsed
        # simulation time after the step as a timedelta.
        prev_elapsed = adapter.elapsed
        elapsed_td = adapter.step()
        self.assertGreater(elapsed_td, timedelta(0))
        self.assertGreater(adapter.elapsed, prev_elapsed)
        adapter.close()

    def test_idempotent_close(self):
        """close() is safe to call multiple times."""
        adapter = self.make_adapter(open=True)
        adapter.close()
        adapter.close()  # second call is a no-op
        adapter.close()

    def test_full_simulation_runs_to_completion(self):
        """Stride to the end; engine transitions RUNNING -> ENDED."""
        adapter = self.make_adapter(open=True)
        steps = 0
        while adapter.is_running:
            # step() raises on engine error; returns timedelta(0) at end.
            adapter.step()
            steps += 1
            # Safety net so a hung loop doesn't burn the test runner.
            if steps > 100_000:
                self.fail("Step count exceeded safety limit; check fixture.")
        self.assertEqual(adapter.state, EngineState.ENDED)
        # Minimal fixture: 30 minutes at 15-second routing step = 120 steps.
        # Allow some slack for engine sub-stepping.
        self.assertTrue(60 <= steps <= 500, f"Unexpected step count: {steps}")
        adapter.end()
        adapter.report()
        adapter.close()

    def test_nodes_links_controls_accessors(self):
        """Lazy accessors return the engine collection classes."""
        adapter = self.make_adapter(open=True)
        # First access constructs; second returns cached instance.
        n1 = adapter.nodes
        n2 = adapter.nodes
        self.assertIs(n1, n2)
        self.assertEqual(adapter.nodes.count(), 2)  # J1, O1
        self.assertEqual(adapter.links.count(), 1)  # C1
        self.assertEqual(adapter.controls.count(), 0)  # no [CONTROLS] in fixture

    def test_context_manager(self):
        """``with SolverAdapter(...): ...`` opens, initializes, and closes."""
        with SolverAdapter(self.minimal_inp, self.minimal_rpt, self.minimal_out) as adapter:
            self.assertEqual(adapter.state, EngineState.RUNNING)
            adapter.step()
        # After exit, the handle is closed.
        self.assertTrue(adapter._closed)


# ---------------------------------------------------------------------------
# Lenient open + validation accumulators
# ---------------------------------------------------------------------------


class TestLenientOpen(BaseEngineTest):
    """Opt-in permissive open and the ``open_errors`` / ``open_warnings``
    accumulators. Test IO is written under the per-test ``tmp_path`` so the
    broken fixture is reviewable (CLAUDE.md §4.1), matching the gym suite's
    existing convention."""

    def _adapter(self, inp):
        adapter = SolverAdapter(inp, self.minimal_rpt, self.minimal_out)
        # Track for teardown so the engine handle is not leaked.
        self._adapters.append(adapter)
        return adapter

    def _write_broken_inp(self):
        path = self.tmp_path / "broken.inp"
        path.write_text(_BROKEN_INP, encoding="utf-8")
        return path

    def test_strict_open_leaves_errors_empty(self):
        """A clean model opened strictly records no post-parse errors."""
        adapter = self._adapter(self.minimal_inp)
        adapter.open()  # strict is the default
        self.assertEqual(adapter.open_errors, [])
        self.assertIsInstance(adapter.open_warnings, list)

    def test_lenient_kwarg_accepted_on_clean_model(self):
        """``open(lenient=True)`` opens a clean model with empty error list."""
        adapter = self._adapter(self.minimal_inp)
        adapter.open(lenient=True)
        self.assertEqual(adapter.open_errors, [])
        self.assertIsInstance(adapter.open_warnings, list)

    def test_accumulators_return_fresh_lists(self):
        """The accessors return independent copies, not a shared handle view."""
        adapter = self._adapter(self.minimal_inp)
        adapter.open()
        first = adapter.open_errors
        first.append("mutation should not leak")
        self.assertEqual(adapter.open_errors, [])

    def test_broken_model_lenient_open_records_errors(self):
        """A broken model opened leniently stays OPENED and records an error
        instead of hard-failing."""
        adapter = self._adapter(self._write_broken_inp())
        adapter.open(lenient=True)
        self.assertGreaterEqual(len(adapter.open_errors), 1)
        self.assertIn("209", " ".join(adapter.open_errors))

    def test_broken_model_strict_open_still_fails(self):
        """The same fixture must fail a strict open — the run path stays strict,
        so lenient has to be the thing that changes the outcome."""
        adapter = self._adapter(self._write_broken_inp())
        with self.assertRaises(EngineError):
            adapter.open()


if __name__ == "__main__":
    unittest.main()
