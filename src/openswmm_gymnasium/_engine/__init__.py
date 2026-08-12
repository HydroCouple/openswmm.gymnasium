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
openswmm_gymnasium._engine
==========================

Thin adapter over L{openswmm.engine.Solver} (the handle-based,
thread-safe v6 engine). This is the B{only} place in the package that
touches the engine — every other module imports from here, never
directly from C{openswmm.engine}. Plan §2 / §2.3.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from openswmm_gymnasium._engine.solver_adapter import (
    EngineCapabilityError,
    LegacySolverRejectedError,
    SolverAdapter,
    require_engine_capabilities,
)

__all__ = [
    "SolverAdapter",
    "LegacySolverRejectedError",
    "EngineCapabilityError",
    "require_engine_capabilities",
]
