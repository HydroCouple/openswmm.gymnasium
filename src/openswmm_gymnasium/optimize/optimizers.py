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
Ask/tell optimisers over a box-bounded decision vector.

Every optimiser proposes batches with C{ask()} and receives their costs
(every objective minimised) with C{tell()}. C{ask()} returns an empty list
when the search is finished. The caller decides how a batch is evaluated,
which is what lets a population run in parallel, or come from another
process entirely.

  - L{RandomSearch} -- uniform sampling baseline.
  - L{GridSearch} -- full factorial over per-dimension levels, truncated at the budget.
  - L{Alternatives} -- the candidates the caller gives, each once, in order:
    an alternatives analysis, or a baseline.
  - L{PlatypusOptimizer} -- any Platypus MOEA (C{nsga2}, C{nsga3}, C{spea2},
    C{moead}, C{gde3}); one batch per generation. Needs the C{platypus} extra.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import itertools
import queue
import random
import threading
from collections.abc import Sequence
from typing import Protocol

import numpy as np

from openswmm_gymnasium.spec.errors import ErrorCode, SpecError

#: Platypus MOEA class names accepted as an algorithm (lowercase key).
PLATYPUS_ALGORITHMS = {
    "nsga2": "NSGAII",
    "nsga3": "NSGAIII",
    "spea2": "SPEA2",
    "moead": "MOEAD",
    "gde3": "GDE3",
}

#: Every algorithm name L{make_optimizer} accepts.
ALGORITHMS = ("random_search", "grid_search", "alternatives", *PLATYPUS_ALGORITHMS)


class Optimizer(Protocol):
    """Ask/tell protocol shared by every optimiser."""

    def ask(self) -> list[np.ndarray]:
        """The next batch of candidates; empty when the search is finished."""

    def tell(self, vectors: Sequence[np.ndarray], costs: Sequence[Sequence[float]]) -> None:
        """Costs (minimised) of the batch last returned by C{ask}, in the same order."""

    def close(self) -> None:
        """Release resources (stops a background search)."""


class RandomSearch:
    """Uniform random sampling of the box, C{budget} candidates in batches of C{batch}."""

    def __init__(
        self,
        low: np.ndarray,
        high: np.ndarray,
        budget: int,
        seed: int | None = None,
        batch: int = 1,
    ) -> None:
        self._low, self._high = low, high
        self._rng = np.random.default_rng(seed)
        self._left, self._batch = budget, max(1, batch)

    def ask(self) -> list[np.ndarray]:
        n = min(self._batch, self._left)
        self._left -= n
        return [self._rng.uniform(self._low, self._high) for _ in range(n)]

    def tell(self, vectors, costs) -> None:
        pass

    def close(self) -> None:
        pass


class GridSearch:
    """Full factorial over C{levels} points per component, capped at C{budget}."""

    def __init__(
        self, low: np.ndarray, high: np.ndarray, budget: int, levels: int, batch: int = 1
    ) -> None:
        axes = [np.linspace(lo, hi, levels) for lo, hi in zip(low, high, strict=True)]
        self._points = itertools.islice(itertools.product(*axes), budget)
        self._batch = max(1, batch)

    def ask(self) -> list[np.ndarray]:
        return [np.asarray(p) for p in itertools.islice(self._points, self._batch)]

    def tell(self, vectors, costs) -> None:
        pass

    def close(self) -> None:
        pass


class _Stop(Exception):
    """Ends the background Platypus run."""


class PlatypusOptimizer:
    """A Platypus MOEA driven by ask/tell.

    The algorithm runs on a background thread. Its evaluator receives each
    generation's unevaluated solutions as one batch and hands them to
    C{ask()}; C{tell()} returns their costs and the algorithm proceeds. Budget
    semantics are Platypus's own: the run stops at the first generation that
    reaches C{budget} evaluations, counting offspring that are unchanged
    copies of a parent, which are not evaluated again.

    Platypus draws from Python's process-wide C{random}. A C{seed} seeds it
    just before the run starts, so one study at a time repeats exactly; studies
    running at once in one process share that generator and do not.
    """

    def __init__(
        self,
        algorithm: str,
        low: np.ndarray,
        high: np.ndarray,
        n_objectives: int,
        budget: int,
        population_size: int,
        seed: int | None = None,
    ) -> None:
        try:
            import platypus
            from platypus.evaluator import Evaluator
        except ImportError as exc:
            raise SpecError(
                f"[{ErrorCode.DEPENDENCY_MISSING}] Algorithm '{algorithm}' requires the "
                "optional platypus-opt package. Install it with: "
                "pip install 'openswmm.gymnasium[platypus]'"
            ) from exc

        self._asked: queue.Queue = queue.Queue(maxsize=1)
        self._told: queue.Queue = queue.Queue(maxsize=1)
        self._costs: dict[tuple[float, ...], list[float]] = {}
        self._error: BaseException | None = None
        owner = self

        class _BatchEvaluator(Evaluator):
            def evaluate_all(self, jobs, **kwargs):
                if not jobs:  # every offspring a parent copy; [] would read as "finished"
                    return jobs
                vectors = [list(job.solution.variables) for job in jobs]
                owner._asked.put(vectors)
                costs = owner._told.get()
                if costs is None:
                    raise _Stop()
                owner._costs = {tuple(v): list(c) for v, c in zip(vectors, costs, strict=True)}
                for job in jobs:
                    job.run()  # problem.function looks the costs up
                return jobs

        problem = platypus.Problem(int(low.shape[0]), n_objectives)
        problem.types[:] = [
            platypus.Real(float(lo), float(hi)) for lo, hi in zip(low, high, strict=True)
        ]
        problem.directions[:] = [platypus.Problem.MINIMIZE] * n_objectives
        problem.function = lambda variables: self._costs[tuple(variables)]
        cls = getattr(platypus, PLATYPUS_ALGORITHMS[algorithm])
        if seed is not None:
            random.seed(seed)
        engine = cls(problem, population_size=population_size, evaluator=_BatchEvaluator())
        self._thread = threading.Thread(
            target=self._run, args=(engine, budget), name="platypus", daemon=True
        )
        self._thread.start()

    def _run(self, engine, budget: int) -> None:
        try:
            engine.run(budget)
        except _Stop:
            pass
        except BaseException as exc:  # surfaced by the next ask()
            self._error = exc
        finally:
            while True:  # replace any batch nobody will ask for with the end marker
                try:
                    self._asked.put_nowait(None)
                    break
                except queue.Full:
                    try:
                        self._asked.get_nowait()
                    except queue.Empty:
                        pass

    def ask(self) -> list[np.ndarray]:
        batch = self._asked.get()
        if batch is None:
            self._asked.put(None)  # stay finished for later calls
            if self._error is not None:
                raise self._error
            return []
        return [np.asarray(v, dtype=float) for v in batch]

    def tell(self, vectors, costs) -> None:
        self._told.put([list(map(float, c)) for c in costs])

    def close(self) -> None:
        if self._thread.is_alive():
            try:
                # Blocks only while the thread is still taking the last costs.
                self._told.put(None, timeout=10)
            except queue.Full:
                pass
            self._thread.join(timeout=10)


class Alternatives:
    """The given candidates, each once and in order, in batches of C{batch}.

    Every candidate must lie within the design bounds: a factory would clip a value
    outside them, so the evaluated design would not be the one given.
    """

    def __init__(
        self,
        candidates: Sequence[np.ndarray],
        low: np.ndarray,
        high: np.ndarray,
        batch: int = 1,
    ) -> None:
        vectors = [np.asarray(c, dtype=float) for c in candidates]
        if not vectors:
            raise SpecError(f"[{ErrorCode.VALIDATION_ERROR}] alternatives needs a candidate.")
        for i, v in enumerate(vectors, start=1):
            if v.shape != np.shape(low):
                raise SpecError(
                    f"[{ErrorCode.VALIDATION_ERROR}] Candidate {i} has {v.size} values; "
                    f"the design has {np.size(low)}."
                )
            outside = np.flatnonzero((v < low) | (v > high))
            if outside.size:
                raise SpecError(
                    f"[{ErrorCode.VALIDATION_ERROR}] Candidate {i} is outside the design "
                    f"bounds at component(s) {outside.tolist()}."
                )
        self._pending, self._batch = vectors, max(1, batch)

    def ask(self) -> list[np.ndarray]:
        out, self._pending = self._pending[: self._batch], self._pending[self._batch :]
        return out

    def tell(self, vectors, costs) -> None:
        pass

    def close(self) -> None:
        pass


def make_optimizer(
    algorithm: str,
    low: np.ndarray,
    high: np.ndarray,
    n_objectives: int,
    *,
    budget: int,
    seed: int | None = None,
    population_size: int = 20,
    grid_levels: int = 5,
    batch: int = 1,
    candidates: Sequence[np.ndarray] | None = None,
) -> Optimizer:
    """The optimiser for an algorithm name (see L{ALGORITHMS}).

    @param batch: Candidates per C{ask()} for random and grid search and for
        alternatives (the Platypus batch is its generation).
    @param candidates: The decision vectors an C{"alternatives"} search evaluates.
    @raise SpecError: C{VALIDATION_ERROR} for an unknown algorithm,
        C{DEPENDENCY_MISSING} when platypus-opt is needed but absent.
    """
    if algorithm == "random_search":
        return RandomSearch(low, high, budget, seed, batch)
    if algorithm == "grid_search":
        return GridSearch(low, high, budget, grid_levels, batch)
    if algorithm == "alternatives":
        return Alternatives(candidates or [], low, high, batch)
    if algorithm in PLATYPUS_ALGORITHMS:
        return PlatypusOptimizer(algorithm, low, high, n_objectives, budget, population_size, seed)
    raise SpecError(
        f"[{ErrorCode.VALIDATION_ERROR}] Unknown algorithm '{algorithm}'. "
        f"Valid algorithms: {', '.join(ALGORITHMS)}."
    )
