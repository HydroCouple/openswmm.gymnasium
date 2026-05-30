"""
openswmm_gymnasium._engine
==========================

Thin adapter over L{openswmm.engine.Solver} (the handle-based,
thread-safe v6 engine). This is the B{only} place in the package that
touches the engine — every other module imports from here, never
directly from C{openswmm.engine}. Plan §2 / §2.3.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from openswmm_gymnasium._engine.solver_adapter import (
    LegacySolverRejectedError,
    SolverAdapter,
)

__all__ = ["SolverAdapter", "LegacySolverRejectedError"]
