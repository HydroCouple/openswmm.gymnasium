"""Shared :class:`unittest.TestCase` base classes for the unit suite.

Per plan §8.0 — B{no engine mocks}. Engine-touching tests drive the real
L{openswmm.engine.Solver} against the tiny C{tests/data/minimal.inp}
fixture. These base classes replace the former pytest ``conftest.py``
fixtures so the suite runs under stdlib ``python -m unittest`` with no
pytest dependency.

Provided base classes:

  - L{BaseTestCase} — gives every test an isolated, auto-cleaned
    C{self.tmp_path} (a :class:`pathlib.Path` to a fresh temp directory),
    the stdlib replacement for pytest's ``tmp_path`` fixture.
  - L{BaseEngineTest} — adds a per-test copy of C{minimal.inp}
    (C{self.minimal_inp} / C{self.minimal_rpt} / C{self.minimal_out}) and
    a C{self.make_adapter(open=True)} factory that constructs a
    L{SolverAdapter} and tracks it for teardown so engine handles are not
    leaked between tests.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

# Repository-relative path to the canonical test fixtures. _base.py lives in
# tests/unit/, so the data directory is one level up at tests/data/.
_DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
_MINIMAL_INP = _DATA_DIR / "minimal.inp"

assert _MINIMAL_INP.exists(), (
    f"tests/data/minimal.inp not found at {_MINIMAL_INP}; "
    "the fixture must be committed to the repository."
)


class BaseTestCase(unittest.TestCase):
    """TestCase with an isolated, auto-cleaned ``self.tmp_path``.

    Stdlib replacement for pytest's function-scoped ``tmp_path`` fixture.
    Each test gets a fresh temporary directory removed at teardown.
    """

    def setUp(self) -> None:
        super().setUp()
        self.tmp_path = Path(tempfile.mkdtemp(prefix="oswg_test_"))
        self.addCleanup(shutil.rmtree, self.tmp_path, ignore_errors=True)


class BaseEngineTest(BaseTestCase):
    """TestCase for tests that drive the real engine.

    Exposes a per-test copy of ``minimal.inp`` and a ``make_adapter``
    factory. Every adapter the factory creates is closed at teardown, so
    engine handles never leak between tests.
    """

    def setUp(self) -> None:
        super().setUp()
        self.minimal_inp = self.tmp_path / "minimal.inp"
        shutil.copy(_MINIMAL_INP, self.minimal_inp)
        self.minimal_rpt = self.tmp_path / "minimal.rpt"
        self.minimal_out = self.tmp_path / "minimal.out"
        self._adapters: list = []
        self.addCleanup(self._close_adapters)

    def make_adapter(self, open: bool = True, *, inp: Path | None = None):
        """Construct a :class:`SolverAdapter` over the per-test ``minimal.inp``.

        With ``open=True`` (default) the adapter is opened, initialized, and
        started — i.e. left in the engine ``RUNNING`` state, ready to step.
        All adapters created here are closed at teardown.
        """
        # Local import so a broken solver_adapter module surfaces as a test
        # error, not an import-time collection error.
        from openswmm_gymnasium._engine import SolverAdapter

        adapter = SolverAdapter(
            inp if inp is not None else self.minimal_inp,
            self.minimal_rpt,
            self.minimal_out,
        )
        if open:
            adapter.open()
            adapter.initialize()
            adapter.start()
        self._adapters.append(adapter)
        return adapter

    def _close_adapters(self) -> None:
        for adapter in self._adapters:
            try:
                adapter.close()
            except Exception:
                pass
