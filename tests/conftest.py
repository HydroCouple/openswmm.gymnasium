"""Shared test fixtures for the openswmm.gymnasium test suite.

Per plan §8.0 — B{no engine mocks}. All tests drive the real
L{openswmm.engine.Solver} against tiny C{.inp} fixtures under
C{tests/data/}. The unit / integration / regression split is by
B{scope}, not by whether the engine is real.

Fixtures provided:

  - C{minimal_inp} (function-scoped) — path to a per-test copy of
    C{tests/data/minimal.inp} placed inside C{tmp_path}. Each test
    gets an isolated copy so the engine can write its C{.rpt} /
    C{.out} files alongside without colliding with sibling tests.
  - C{minimal_rpt} / C{minimal_out} — sibling paths inside C{tmp_path}
    for the report and output files.
  - C{solver_adapter} (factory) — returns a callable
    C{make_adapter(open=True)} that constructs a
    L{openswmm_gymnasium._engine.SolverAdapter} and (optionally) opens
    and initializes it. The factory tracks all adapters it creates and
    ensures they're closed at test teardown.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import shutil
from collections.abc import Callable
from pathlib import Path

import pytest

# Repository-relative path to the canonical test fixtures.
_DATA_DIR = (Path(__file__).parent / "data").resolve()
_MINIMAL_INP = _DATA_DIR / "minimal.inp"

assert _MINIMAL_INP.exists(), (
    f"tests/data/minimal.inp not found at {_MINIMAL_INP}; "
    "the fixture must be committed to the repository."
)


# ---------------------------------------------------------------------------
# Per-test INP copies
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_inp(tmp_path: Path) -> Path:
    """Copy ``minimal.inp`` into ``tmp_path`` and return the path."""
    dest = tmp_path / "minimal.inp"
    shutil.copy(_MINIMAL_INP, dest)
    return dest


@pytest.fixture
def minimal_rpt(tmp_path: Path) -> Path:
    return tmp_path / "minimal.rpt"


@pytest.fixture
def minimal_out(tmp_path: Path) -> Path:
    return tmp_path / "minimal.out"


# ---------------------------------------------------------------------------
# SolverAdapter factory
# ---------------------------------------------------------------------------


@pytest.fixture
def solver_adapter(minimal_inp: Path, minimal_rpt: Path, minimal_out: Path) -> Callable:
    """Factory: ``adapter = make_adapter(open=True)`` returns a
    :class:`SolverAdapter` wrapping the per-test ``minimal.inp``.

    All adapters created by the factory are closed at end of test, so
    engine handles are not leaked between tests.
    """
    # Local import so a broken solver_adapter module surfaces as a test
    # error, not a collection error.
    from openswmm_gymnasium._engine import SolverAdapter

    created: list[SolverAdapter] = []

    def _make(open: bool = True, *, inp: Path | None = None) -> SolverAdapter:
        adapter = SolverAdapter(
            inp if inp is not None else minimal_inp,
            minimal_rpt,
            minimal_out,
        )
        if open:
            adapter.open()
            adapter.initialize()
        created.append(adapter)
        return adapter

    try:
        yield _make
    finally:
        for a in created:
            try:
                a.close()
            except Exception:
                pass
