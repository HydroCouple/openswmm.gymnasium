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
Candidate evaluation: one episode per decision vector.

L{Evaluator} runs candidates on one env (one engine handle) in the calling
process. L{WorkerPool} runs a batch of candidates in parallel on long-lived
worker processes, each holding its own L{Evaluator}, so the engine is
loaded once per worker. A worker that dies (an engine crash) fails only the
candidate it was evaluating; it is replaced and the batch continues.

One evaluation is one episode: for C{cip} the env is a single-step
contextual bandit (design applied in C{step}); for C{joint} the design is
fixed at C{reset(options={"design_action": ...})} and the runtime half runs
a neutral midpoint action each step; C{market}/C{schedule}/C{control_curve}
take the raw policy vector as their single action.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import multiprocessing
import os
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from multiprocessing.connection import wait
from typing import Any

import numpy as np

from openswmm_gymnasium.optimize.space import (
    POLICY_ENV_TYPES,
    DesignDimension,
    design_dimensions,
    join_decisions,
    objective_directions,
    split_vector,
)
from openswmm_gymnasium.spec.config import EnvConfig, build_env
from openswmm_gymnasium.spec.envs import build_action, json_safe


class StudyCancelled(Exception):
    """Raised when a study is asked to stop."""


@dataclass
class Evaluation:
    """Outcome of one candidate.

    @ivar costs: Direction-adjusted cost per objective (every objective is
        minimised: a C{maximize} term is negated); C{None} when it failed.
    @ivar steps: Env steps the episode took.
    @ivar error: Failure description, or C{None}.
    """

    costs: list[float] | None
    steps: int = 0
    error: str | None = None


@dataclass
class Evaluator:
    """Runs one episode per candidate on a single env (one engine handle).

    @ivar env_config: The env config being searched.
    @ivar dims: Its search dimensions.
    @ivar seed: Seed passed to every C{reset}.
    @ivar max_steps: Safety cap on steps per episode (C{joint} envs).
    """

    env_config: EnvConfig
    dims: list[DesignDimension]
    seed: int | None = None
    max_steps: int = 10_000
    _env: Any = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.directions = objective_directions(self.env_config)
        self.objective_names = list(self.directions)
        self._env = build_env(self.env_config)

    def close(self) -> None:
        """Close the env and release its engine handle."""
        self._env.close()

    def evaluate(self, vector: np.ndarray) -> Evaluation:
        """Run one episode for a flat decision vector.

        @raise Exception: Whatever the env raises; L{WorkerPool} records it.
        @rtype: L{Evaluation}
        """
        env, kind = self._env, self.env_config.env_type
        if kind in POLICY_ENV_TYPES:
            env.reset(seed=self.seed)
            _obs, _r, _term, _trunc, info = env.step(np.asarray(vector, dtype=np.float32))
            components, steps = dict(info.get("reward_components", {})), 1
        elif kind == "cip":
            env.reset(seed=self.seed)
            design = json_safe(split_vector(self.dims, vector))
            action = build_action(env.action_space, {"design": design})
            _obs, _r, _term, _trunc, info = env.step(action)
            components, steps = dict(info.get("reward_components", {})), 1
        else:  # joint: design fixed at reset, neutral runtime action each step
            env.reset(seed=self.seed, options={"design_action": split_vector(self.dims, vector)})
            neutral = build_action(env.action_space, None)
            components, steps = {}, 0
            terminated = truncated = False
            while steps < self.max_steps and not (terminated or truncated):
                _obs, _r, terminated, truncated, info = env.step(neutral)
                steps += 1
                for term, value in info.get("reward_components", {}).items():
                    components[term] = components.get(term, 0.0) + float(value)
        costs = [
            float(components.get(name, 0.0))
            * (1.0 if self.directions[name] == "minimize" else -1.0)
            for name in self.objective_names
        ]
        return Evaluation(costs=costs, steps=steps)


#: Design kinds whose action is the value of one element field, so the model's own value
#: is a baseline. A kind that resizes or reshapes (link_diameter, storage_volume, ...)
#: has no single field to read.
_FIELD_KINDS = {
    "field_design": None,
    "link_roughness": "link.roughness",
    "link_length": "link.length",
    "node_max_depth": "node.max_depth",
}


def baseline_decisions(env_config: EnvConfig) -> dict[str, list[float]]:
    """The model's own values for every design dimension, read from the engine.

    Evaluated with L{evaluate_decisions}, this is the unchanged model scored by the
    env's objectives.

    @raise SpecError: C{VALIDATION_ERROR} for a design kind with no single field to read.
    @rtype: dict
    """
    from openswmm_gymnasium._engine import SolverAdapter, element_kind
    from openswmm_gymnasium.spec.errors import ErrorCode, SpecError

    out: dict[str, list[float]] = {}
    adapter = SolverAdapter(env_config.inp_path, os.devnull, os.devnull)
    adapter.open()
    try:
        for spec, dim in zip(
            env_config.design_factories, design_dimensions(env_config), strict=True
        ):
            if spec.kind not in _FIELD_KINDS:
                raise SpecError(
                    f"[{ErrorCode.VALIDATION_ERROR}] '{spec.kind}' has no single field to read "
                    "a baseline from; pass its values."
                )
            path = _FIELD_KINDS[spec.kind] or spec.params["path"]
            ids = spec.params.get("ids") or spec.params.get("link_ids") or spec.params["node_ids"]
            kind = element_kind(path)
            out[dim.key] = [float(adapter.read(path, adapter.index(kind, i))) for i in ids]
    finally:
        adapter.close()
    return out


def evaluate_decisions(
    env_config: EnvConfig,
    decisions: dict[str, Any],
    *,
    seed: int | None = None,
    max_steps: int = 10_000,
) -> dict[str, Any]:
    """Score one design or policy: one episode, as a study would evaluate it.

    @param decisions: Dimension key to values, the C{decisions} of an evaluation record
        (L{baseline_decisions} gives the model's own).
    @return: C{{"decisions", "objectives", "steps"}}, objectives as direction-adjusted
        costs by reward-term name, like an C{evaluations.jsonl} record.
    @raise SpecError: The decisions do not match the env config's dimensions.
    @rtype: dict
    """
    dims = design_dimensions(env_config)
    vector = join_decisions(dims, decisions)
    evaluator = Evaluator(env_config, dims, seed=seed, max_steps=max_steps)
    try:
        outcome = evaluator.evaluate(vector)
    finally:
        evaluator.close()
    return json_safe(
        {
            "decisions": split_vector(dims, vector),
            "objectives": dict(zip(evaluator.objective_names, outcome.costs, strict=True)),
            "steps": outcome.steps,
        }
    )


def default_workers() -> int:
    """Worker processes to use when unspecified: all but one CPU, at most 8."""
    return max(1, min((os.cpu_count() or 2) - 1, 8))


def _worker_main(conn, payload: dict[str, Any], threads: int) -> None:
    """Worker process: build one evaluator, then evaluate vectors until told to stop."""
    # Before the engine loads: share the CPUs between workers instead of
    # letting every worker's OpenMP runtime claim all of them.
    os.environ["OMP_NUM_THREADS"] = str(threads)
    try:
        evaluator = Evaluator(
            EnvConfig.model_validate(payload["env_config"]),
            [DesignDimension(**d) for d in payload["dims"]],
            seed=payload["seed"],
            max_steps=payload["max_steps"],
        )
    except Exception as exc:  # reported to the parent, which raises it
        conn.send(("error", f"{type(exc).__name__}: {exc}"))
        return
    conn.send(("ready", None))
    try:
        while True:
            task = conn.recv()
            if task is None:
                break
            index, vector = task
            try:
                result = evaluator.evaluate(np.asarray(vector, dtype=float))
            except Exception as exc:
                result = Evaluation(costs=None, error=f"{type(exc).__name__}: {exc}")
            conn.send((index, asdict(result)))
    finally:
        evaluator.close()


class _Worker:
    def __init__(self, ctx, payload: dict[str, Any], threads: int) -> None:
        self.conn, child = ctx.Pipe()
        self.process = ctx.Process(target=_worker_main, args=(child, payload, threads), daemon=True)
        self.process.start()
        child.close()
        kind, message = self.conn.recv()
        if kind != "ready":
            self.process.join(timeout=5)
            raise RuntimeError(f"evaluation worker failed to start: {message}")

    def stop(self) -> None:
        try:
            self.conn.send(None)
        except OSError:
            pass
        self.process.join(timeout=10)
        if self.process.is_alive():
            self.process.terminate()
            self.process.join(timeout=5)
        self.conn.close()

    def kill(self) -> None:
        self.process.terminate()
        self.process.join(timeout=5)
        self.conn.close()


class WorkerPool:
    """Evaluates batches of candidates on long-lived worker processes.

    Each worker builds its env once and reuses it, like L{Evaluator}. Results
    come back in input order. A worker that exits while evaluating (for
    example an engine crash) yields an L{Evaluation} with C{error} set for that
    candidate only, and is replaced.

    @ivar workers: Number of worker processes.
    """

    def __init__(
        self,
        env_config: EnvConfig,
        dims: list[DesignDimension],
        *,
        workers: int,
        seed: int | None = None,
        max_steps: int = 10_000,
    ) -> None:
        if workers < 1:
            raise ValueError("WorkerPool needs at least one worker")
        self.workers = workers
        self._ctx = multiprocessing.get_context("spawn")
        self._payload = {
            "env_config": env_config.model_dump(mode="json"),
            "dims": [asdict(d) for d in dims],
            "seed": seed,
            "max_steps": max_steps,
        }
        self._threads = max(1, (os.cpu_count() or 1) // workers)
        #: Seconds between checks of C{should_stop} while candidates are running.
        self.poll_interval = 0.5
        self._pool: list[_Worker] = []
        try:
            for _ in range(workers):
                self._pool.append(self._spawn())
        except BaseException:
            self.close()
            raise

    def _spawn(self) -> _Worker:
        return _Worker(self._ctx, self._payload, self._threads)

    def evaluate(
        self,
        vectors: Sequence[np.ndarray],
        should_stop: Callable[[], bool] = lambda: False,
    ) -> list[Evaluation]:
        """Evaluate every vector; results are in input order.

        @raise StudyCancelled: When C{should_stop()} becomes true; workers that
            were busy are replaced.
        """
        results: list[Evaluation | None] = [None] * len(vectors)
        pending = list(range(len(vectors)))[::-1]
        busy: dict[int, tuple[_Worker, int]] = {}  # id(worker) -> (worker, vector index)
        while pending or busy:
            if should_stop():
                for worker, _ in busy.values():
                    self._replace(worker)
                raise StudyCancelled()
            for slot, worker in enumerate(self._pool):
                if pending and id(worker) not in busy:
                    index = pending.pop()
                    task = (index, np.asarray(vectors[index], dtype=float).tolist())
                    try:
                        worker.conn.send(task)
                    except OSError:  # died while idle: replace it; the task never ran
                        self._replace(worker)
                        worker = self._pool[slot]
                        worker.conn.send(task)
                    busy[id(worker)] = (worker, index)
            waitables = {}
            for worker, index in busy.values():
                waitables[worker.conn] = (worker, index)
                waitables[worker.process.sentinel] = (worker, index)
            for ready in wait(list(waitables), timeout=self.poll_interval):
                worker, index = waitables[ready]
                if id(worker) not in busy:
                    continue  # already handled through its other waitable
                try:
                    if not worker.conn.poll():
                        raise EOFError
                    got, payload = worker.conn.recv()
                    results[got] = Evaluation(**payload)
                except (EOFError, OSError):
                    code = worker.process.exitcode
                    results[index] = Evaluation(
                        costs=None, error=f"evaluation worker exited (code {code})"
                    )
                    self._replace(worker)
                busy.pop(id(worker), None)
        return results  # type: ignore[return-value]

    def _replace(self, worker: _Worker) -> None:
        worker.kill()
        self._pool[self._pool.index(worker)] = self._spawn()

    def close(self) -> None:
        """Stop every worker."""
        for worker in self._pool:
            worker.stop()
        self._pool.clear()

    def __enter__(self) -> WorkerPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class InProcessPool:
    """L{WorkerPool}'s interface over one in-process L{Evaluator} (C{workers=0}).

    Exceptions become failed L{Evaluation}s, as in a worker; an engine crash
    takes the calling process down with it.
    """

    workers = 0

    def __init__(
        self,
        env_config: EnvConfig,
        dims: list[DesignDimension],
        *,
        seed: int | None = None,
        max_steps: int = 10_000,
    ) -> None:
        self._evaluator = Evaluator(env_config, dims, seed=seed, max_steps=max_steps)

    def evaluate(
        self, vectors: Sequence[np.ndarray], should_stop: Callable[[], bool] = lambda: False
    ) -> list[Evaluation]:
        results = []
        for vector in vectors:
            if should_stop():
                raise StudyCancelled()
            try:
                results.append(self._evaluator.evaluate(np.asarray(vector, dtype=float)))
            except Exception as exc:
                results.append(Evaluation(costs=None, error=f"{type(exc).__name__}: {exc}"))
        return results

    def close(self) -> None:
        self._evaluator.close()

    def __enter__(self) -> InProcessPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
