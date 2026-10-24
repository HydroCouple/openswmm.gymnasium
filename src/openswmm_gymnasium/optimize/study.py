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
Run a design or policy search over an env config.

L{run_study} drives an ask/tell optimiser: each batch it asks for is
evaluated (in parallel on worker processes by default), logged to
C{evaluations.jsonl}, and told back. The Pareto set (or the best
candidate for one objective) goes to C{result.json}. Both files are written
to a user-visible directory.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from openswmm_gymnasium.optimize.evaluation import (
    InProcessPool,
    StudyCancelled,
    WorkerPool,
    default_workers,
)
from openswmm_gymnasium.optimize.optimizers import PLATYPUS_ALGORITHMS, make_optimizer
from openswmm_gymnasium.optimize.space import (
    design_dimensions,
    flat_bounds,
    objective_directions,
    split_vector,
)
from openswmm_gymnasium.spec.config import EnvConfig
from openswmm_gymnasium.spec.envs import json_safe


class OptimizationConfig(BaseModel):
    """Declarative optimization-run settings.

    @ivar algorithm: C{"random_search"}, C{"grid_search"}, or a lowercase
        Platypus MOEA name (e.g. C{"nsga2"}).
    @ivar budget: Maximum design evaluations.
    @ivar population_size: MOEA population size (Platypus only).
    @ivar grid_levels: Per-dimension levels for C{grid_search}.
    @ivar seed: RNG seed for sampling algorithms and episode resets.
    @ivar max_steps_per_episode: Safety cap per evaluation episode.
    @ivar workers: Worker processes evaluating a batch in parallel; C{None}
        picks all but one CPU (at most 8); C{0} evaluates in the calling
        process.
    @ivar on_failure: What a failed evaluation (an error or a crashed worker)
        does: C{"fail"} stops the study; C{"penalty"} records it with every
        objective set to C{failure_penalty} and continues.
    @ivar failure_penalty: Cost given to each objective of a failed candidate.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: str = "random_search"
    budget: int = Field(default=50, ge=1)
    population_size: int = Field(default=20, ge=2)
    grid_levels: int = Field(default=5, ge=2)
    seed: int | None = None
    max_steps_per_episode: int = Field(default=10_000, ge=1)
    workers: int | None = Field(default=None, ge=0)
    on_failure: Literal["fail", "penalty"] = "fail"
    failure_penalty: float = 1e12

    def resolved(self) -> dict[str, Any]:
        """The search settings that decide which candidates are evaluated.

        Echoed in job snapshots and results so a silent algorithm or budget
        downgrade cannot go unnoticed. Execution settings (workers, failure
        policy) are reported separately under C{execution}.

        @rtype: dict
        """
        out: dict[str, Any] = {
            "algorithm": self.algorithm,
            "budget": self.budget,
            "seed": self.seed,
        }
        if self.algorithm in PLATYPUS_ALGORITHMS:
            out["population_size"] = self.population_size
        elif self.algorithm == "grid_search":
            out["grid_levels"] = self.grid_levels
        return out


def assemble_result(evaluations: list[dict[str, Any]], names: list[str]) -> dict[str, Any]:
    """Pareto set (several objectives) or best candidate (one) over successful evaluations.

    @rtype: dict
    """
    ok = [e for e in evaluations if not e.get("failed")]
    out: dict[str, Any] = {
        "objective_names": names,
        "evaluations_count": len(evaluations),
        "failed_count": len(evaluations) - len(ok),
        "pareto": [],
        "best": None,
    }
    if not ok:
        return out
    costs = np.array([[e["objectives"][n] for n in names] for e in ok])
    if len(names) == 1:
        best_idx = int(np.argmin(costs[:, 0]))
        pareto_idxs = [best_idx]
    else:
        from openswmm_gymnasium.scoring import pareto_front

        front = pareto_front(costs)
        pareto_idxs = [i for i, row in enumerate(costs) if any(np.allclose(row, f) for f in front)]
        best_idx = int(np.argmin(costs.sum(axis=1)))  # equal-weight sum: one convenient pick
    out["pareto"] = [ok[i] for i in pareto_idxs]
    out["best"] = ok[best_idx]
    return out


def run_study(
    env_config: EnvConfig,
    opt: OptimizationConfig,
    output_dir: str | Path,
    *,
    should_stop: Callable[[], bool] = lambda: False,
    on_evaluation: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Search the env config's design or policy space; return the result payload.

    Writes C{evaluations.jsonl} (one record per candidate, in the order the
    optimiser proposed them) and C{result.json} under C{output_dir}.

    @param should_stop: Polled between candidates; when true the study
        stops and L{StudyCancelled} is raised.
    @param on_evaluation: Called with each evaluation record as it is logged.
    @raise SpecError: The config has no searchable space, the algorithm is
        unknown, or an optional dependency is missing.
    @raise RuntimeError: An evaluation failed and C{on_failure="fail"}.
    @raise StudyCancelled: C{should_stop()} became true.
    @rtype: dict
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    dims = design_dimensions(env_config)
    names = list(objective_directions(env_config))
    low, high = flat_bounds(dims)
    workers = default_workers() if opt.workers is None else opt.workers
    optimizer = make_optimizer(
        opt.algorithm,
        low,
        high,
        len(names),
        budget=opt.budget,
        seed=opt.seed,
        population_size=opt.population_size,
        grid_levels=opt.grid_levels,
        batch=max(1, workers),
    )
    evaluations: list[dict[str, Any]] = []
    pool_args = {"seed": opt.seed, "max_steps": opt.max_steps_per_episode}
    try:
        with (
            (output_dir / "evaluations.jsonl").open("w", encoding="utf-8") as log,
            (
                WorkerPool(env_config, dims, workers=workers, **pool_args)
                if workers
                else InProcessPool(env_config, dims, **pool_args)
            ) as pool,
        ):
            while True:
                if should_stop():
                    raise StudyCancelled()
                batch = optimizer.ask()
                if not batch:
                    break
                outcomes = pool.evaluate(batch, should_stop)
                told = []
                for vector, outcome in zip(batch, outcomes, strict=True):
                    record: dict[str, Any] = {
                        "evaluation": len(evaluations) + 1,
                        "decisions": json_safe(split_vector(dims, vector)),
                    }
                    if outcome.error is None:
                        costs = outcome.costs
                    elif opt.on_failure == "fail":
                        raise RuntimeError(
                            f"evaluation {record['evaluation']} failed: {outcome.error}"
                        )
                    else:
                        costs = [opt.failure_penalty] * len(names)
                        record.update(failed=True, error=outcome.error)
                    record["objectives"] = dict(zip(names, costs, strict=True))
                    record["steps"] = outcome.steps
                    evaluations.append(record)
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    if on_evaluation is not None:
                        on_evaluation(record)
                    told.append(costs)
                optimizer.tell(batch, told)
    finally:
        optimizer.close()
    result = assemble_result(evaluations, names)
    result["optimization"] = opt.resolved()
    result["execution"] = {
        "workers": workers,
        "on_failure": opt.on_failure,
    }
    result = json_safe(result)
    (output_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result
