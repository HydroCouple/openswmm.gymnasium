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
Design and policy optimisation over env configs.

  - L{space<openswmm_gymnasium.optimize.space>} -- the decision vector of a
    config (L{design_dimensions}).
  - L{optimizers<openswmm_gymnasium.optimize.optimizers>} -- ask/tell
    optimisers: random, grid and the Platypus MOEAs.
  - L{evaluation<openswmm_gymnasium.optimize.evaluation>} -- one episode per
    candidate, in process or on parallel worker processes.
  - L{study<openswmm_gymnasium.optimize.study>} -- L{run_study}, which ties
    them together and writes C{evaluations.jsonl} and C{result.json}.

.. code-block:: python

    from openswmm_gymnasium.optimize import OptimizationConfig, run_study

    result = run_study(env_config, OptimizationConfig(algorithm="nsga2", budget=400),
                       "runs/detention")
    result["pareto"]

Needs the C{spec} extra (pydantic); Platypus algorithms need C{platypus}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from openswmm_gymnasium.optimize.evaluation import (
    Evaluation,
    Evaluator,
    InProcessPool,
    StudyCancelled,
    WorkerPool,
    default_workers,
)
from openswmm_gymnasium.optimize.optimizers import (
    ALGORITHMS,
    PLATYPUS_ALGORITHMS,
    GridSearch,
    Optimizer,
    PlatypusOptimizer,
    RandomSearch,
    make_optimizer,
)
from openswmm_gymnasium.optimize.space import (
    DesignDimension,
    design_dimensions,
    flat_bounds,
    objective_directions,
    split_vector,
)
from openswmm_gymnasium.optimize.study import OptimizationConfig, assemble_result, run_study

__all__ = [
    "ALGORITHMS",
    "PLATYPUS_ALGORITHMS",
    "DesignDimension",
    "Evaluation",
    "Evaluator",
    "GridSearch",
    "InProcessPool",
    "OptimizationConfig",
    "Optimizer",
    "PlatypusOptimizer",
    "RandomSearch",
    "StudyCancelled",
    "WorkerPool",
    "assemble_result",
    "default_workers",
    "design_dimensions",
    "flat_bounds",
    "make_optimizer",
    "objective_directions",
    "run_study",
    "split_vector",
]
