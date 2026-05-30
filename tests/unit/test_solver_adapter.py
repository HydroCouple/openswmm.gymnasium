"""Unit tests for L{openswmm_gymnasium._engine.SolverAdapter}.

Per plan §8.0, these run against the B{real}
L{openswmm.engine.Solver} — no mocks.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import pytest
from openswmm.engine import EngineState, Solver

from openswmm_gymnasium._engine import LegacySolverRejectedError, SolverAdapter

# ---------------------------------------------------------------------------
# Construction
# ---------------------------------------------------------------------------


class TestConstruction:
    def test_construction_does_not_open_engine(self, minimal_inp, minimal_rpt, minimal_out):
        """Constructor must be cheap — no engine handle allocated until open()."""
        adapter = SolverAdapter(minimal_inp, minimal_rpt, minimal_out)
        # No exception here means handle isn't allocated yet.
        # state is queryable only after open(); we don't probe it.
        assert adapter._owned is True

    def test_optional_rpt_out(self, minimal_inp):
        """rpt and out may be None (engine skips file writing)."""
        adapter = SolverAdapter(minimal_inp, None, None)
        assert adapter._rpt == ""
        assert adapter._out == ""

    def test_path_like_inputs(self, minimal_inp, minimal_rpt, minimal_out):
        """str and Path inputs both accepted."""
        a1 = SolverAdapter(str(minimal_inp), str(minimal_rpt), str(minimal_out))
        a2 = SolverAdapter(minimal_inp, minimal_rpt, minimal_out)
        assert a1._inp == a2._inp


# ---------------------------------------------------------------------------
# Legacy solver guard (plan §0 #7)
# ---------------------------------------------------------------------------


class TestLegacyGuard:
    def test_rejects_non_engine_solver(self, minimal_inp):
        """A non-openswmm.engine object passed via solver= must raise."""

        class FakeLegacySolver:
            """Pretends to be openswmm.legacy.engine.Solver."""

        FakeLegacySolver.__module__ = "openswmm.legacy.engine._solver"
        fake = FakeLegacySolver()
        with pytest.raises(LegacySolverRejectedError) as exc_info:
            SolverAdapter(minimal_inp, solver=fake)
        assert "legacy" in str(exc_info.value).lower()

    def test_accepts_handle_based_solver(self, minimal_inp, minimal_rpt, minimal_out):
        """A real openswmm.engine.Solver is accepted via solver=."""
        real = Solver(str(minimal_inp), str(minimal_rpt), str(minimal_out))
        # Just construction; no open() call so this works even if the
        # engine wheel SIGILLs at C-side init.
        adapter = SolverAdapter(minimal_inp, minimal_rpt, minimal_out, solver=real)
        assert adapter._owned is False
        assert adapter.solver is real


# ---------------------------------------------------------------------------
# Lifecycle (requires a working engine wheel on the host)
# ---------------------------------------------------------------------------


@pytest.mark.integration
class TestLifecycle:
    def test_open_initialize_step_close(self, solver_adapter):
        """Full happy-path lifecycle through the adapter."""
        adapter = solver_adapter(open=True)
        assert adapter.state == EngineState.RUNNING
        # One step advances elapsed time.
        prev_elapsed = adapter.elapsed
        rc = adapter.step()
        assert rc == 0
        assert adapter.elapsed > prev_elapsed
        adapter.close()

    def test_idempotent_close(self, solver_adapter):
        """close() is safe to call multiple times."""
        adapter = solver_adapter(open=True)
        adapter.close()
        adapter.close()  # second call is a no-op
        adapter.close()

    def test_full_simulation_runs_to_completion(self, solver_adapter):
        """Stride to the end; engine transitions RUNNING -> ENDED."""
        adapter = solver_adapter(open=True)
        steps = 0
        while adapter.is_running:
            rc = adapter.step()
            if rc != 0:
                pytest.fail(f"step returned non-zero rc={rc} at step {steps}")
            steps += 1
            # Safety net so a hung loop doesn't burn the test runner.
            if steps > 100_000:
                pytest.fail("Step count exceeded safety limit; check fixture.")
        assert adapter.state == EngineState.ENDED
        # Minimal fixture: 30 minutes at 15-second routing step = 120 steps.
        # Allow some slack for engine sub-stepping.
        assert 60 <= steps <= 500, f"Unexpected step count: {steps}"
        adapter.end()
        adapter.report()
        adapter.close()

    def test_nodes_links_controls_accessors(self, solver_adapter):
        """Lazy accessors return the engine collection classes."""
        adapter = solver_adapter(open=True)
        # First access constructs; second returns cached instance.
        n1 = adapter.nodes
        n2 = adapter.nodes
        assert n1 is n2
        assert adapter.nodes.count() == 2  # J1, O1
        assert adapter.links.count() == 1  # C1
        assert adapter.controls.count() == 0  # no [CONTROLS] in fixture

    def test_context_manager(self, minimal_inp, minimal_rpt, minimal_out):
        """``with SolverAdapter(...): ...`` opens, initializes, and closes."""
        with SolverAdapter(minimal_inp, minimal_rpt, minimal_out) as adapter:
            assert adapter.state == EngineState.RUNNING
            adapter.step()
        # After exit, the handle is closed.
        assert adapter._closed is True
