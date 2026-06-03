"""P1.3 — SolverAdapter.seed_hotstart deterministic state seeding.

Surfaces the engine's hot-start state setters through the adapter so the env
can build reproducible initial conditions. Integration tier: drives the real
engine over C{minimal.inp} (B{no mocks}).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

from tests.unit._base import BaseEngineTest

from openswmm.engine import HotStart, Solver


class TestSeedHotstart(BaseEngineTest):
    def _save_baseline(self) -> str:
        """Run the minimal model briefly and save a hot-start file."""
        path = str(self.tmp_path / "baseline.hsf")
        s = Solver(str(self.minimal_inp), str(self.minimal_rpt), str(self.minimal_out))
        s.open()
        s.initialize()
        s.start()
        for _ in range(5):
            if not (s.state.name == "RUNNING"):
                break
            s.step()
        HotStart.save_from(s, path)
        try:
            s.end()
            s.close()
        except Exception:
            pass
        return path

    def test_seed_applies_without_error(self):
        path = self._save_baseline()
        # apply() requires the INITIALIZED state (post-initialize, pre-start).
        adapter = self.make_adapter(open=False)
        adapter.open()
        adapter.initialize()
        # Should apply cleanly and return None.
        result = adapter.seed_hotstart(
            path,
            node_depths={"J1": 0.3},
            link_flows={"C1": 0.5},
        )
        self.assertIsNone(result)

    def test_seed_no_overrides_applies_baseline(self):
        path = self._save_baseline()
        adapter = self.make_adapter(open=False)
        adapter.open()
        adapter.initialize()
        # No overrides — pure hot-start application path.
        adapter.seed_hotstart(path)
