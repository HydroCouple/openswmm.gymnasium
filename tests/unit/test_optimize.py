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

"""Tests for L{openswmm_gymnasium.optimize}.

Optimiser tests are pure. Study tests drive the real engine (no mocks) on
the committed fixtures, and write their artifacts under the reviewable
C{tests/_output/optimize/} tree (CLAUDE.md §4.1).
"""

from __future__ import annotations

import json
import os
import shutil
import threading
import time
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pydantic")

from openswmm_gymnasium.optimize import (  # noqa: E402
    ALGORITHMS,
    GridSearch,
    OptimizationConfig,
    RandomSearch,
    StudyCancelled,
    WorkerPool,
    design_dimensions,
    flat_bounds,
    make_optimizer,
    run_study,
)
from openswmm_gymnasium.spec.config import EnvConfig, ObservationSpec  # noqa: E402
from openswmm_gymnasium.spec.errors import SpecError  # noqa: E402

_DATA = Path(__file__).resolve().parents[1] / "data"
_OUTPUT = Path(__file__).resolve().parents[1] / "_output" / "optimize"


def _out(name: str) -> Path:
    path = _OUTPUT / name
    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True)
    return path


def _cip_config(inp: Path) -> EnvConfig:
    return EnvConfig(
        env_type="cip",
        inp_path=str(inp),
        observations=ObservationSpec(node_depths=["J1"]),
        design_factories=[
            {
                "kind": "link_roughness",
                "params": {"link_ids": ["C1", "C2"], "low": 0.011, "high": 0.025},
            },
            {"kind": "node_max_depth", "params": {"node_ids": ["J1"], "low": 5.0, "high": 15.0}},
        ],
        reward_terms=[
            {"kind": "flooding_volume"},
            {"kind": "peak_outflow", "params": {"link_ids": ["C11"]}},  # C11 feeds O1
        ],
    )


def _copy_inp(dest: Path) -> Path:
    target = dest / "site_drainage_example.inp"
    shutil.copy(_DATA / "site_drainage_example.inp", target)
    return target


# ---------------------------------------------------------------------------
# Ask/tell optimisers (pure)
# ---------------------------------------------------------------------------


def test_random_search_batches_do_not_change_the_sequence():
    low, high = np.zeros(3), np.ones(3)
    one = RandomSearch(low, high, budget=7, seed=4, batch=1)
    three = RandomSearch(low, high, budget=7, seed=4, batch=3)
    seq_one = [v for _ in range(7) for v in one.ask()]
    seq_three = []
    while batch := three.ask():
        assert len(batch) <= 3
        seq_three += batch
    assert one.ask() == []
    np.testing.assert_array_equal(np.array(seq_one), np.array(seq_three))
    assert all(((v >= 0) & (v <= 1)).all() for v in seq_one)


def test_grid_search_is_capped_at_the_budget():
    grid = GridSearch(np.zeros(2), np.ones(2), budget=5, levels=3, batch=2)
    points = []
    while batch := grid.ask():
        points += batch
    assert len(points) == 5
    np.testing.assert_array_equal(points[0], [0.0, 0.0])
    np.testing.assert_array_equal(points[1], [0.0, 0.5])


def test_unknown_algorithm_is_a_validation_error():
    with pytest.raises(SpecError, match="Unknown algorithm") as err:
        make_optimizer("annealing", np.zeros(1), np.ones(1), 1, budget=3)
    assert err.value.code == "VALIDATION_ERROR"
    assert "nsga2" in ALGORITHMS


def test_platypus_ask_tell_finds_a_two_objective_front():
    pytest.importorskip("platypus")
    low, high = np.zeros(2), np.ones(2)
    opt = make_optimizer("nsga2", low, high, 2, budget=200, population_size=20)
    evaluated = 0
    try:
        while batch := opt.ask():
            # Schaffer-like: f1 = x0, f2 = 1 - x0 + x1 (front at x1 = 0)
            opt.tell(batch, [[v[0], 1.0 - v[0] + v[1]] for v in batch])
            evaluated += len(batch)
    finally:
        opt.close()
    # Platypus stops at the first generation past the budget, and charges offspring that
    # are unchanged parent copies (~1 in 16 here) to it without evaluating them again.
    assert 150 <= evaluated <= 220
    assert opt.ask() == []  # stays finished


def _platypus_vectors(seed):
    opt = make_optimizer(
        "nsga2", np.zeros(2), np.ones(2), 2, budget=120, seed=seed, population_size=20
    )
    vectors = []
    try:
        while batch := opt.ask():
            opt.tell(batch, [[v[0], 1.0 - v[0] + v[1]] for v in batch])
            vectors += [v.tolist() for v in batch]
    finally:
        opt.close()
    return vectors


def test_platypus_repeats_exactly_for_one_seed():
    pytest.importorskip("platypus")
    first = _platypus_vectors(7)
    assert first == _platypus_vectors(7)
    assert first != _platypus_vectors(8)


def test_platypus_close_mid_run_returns_promptly():
    pytest.importorskip("platypus")
    opt = make_optimizer("nsga2", np.zeros(2), np.ones(2), 2, budget=10_000, population_size=10)
    opt.tell(opt.ask(), [[0.0, 0.0]] * 10)
    started = time.monotonic()
    opt.close()
    assert time.monotonic() - started < 5


def test_design_dimensions_come_from_the_factories():
    dims = design_dimensions(_cip_config(_DATA / "site_drainage_example.inp"))
    assert [d.key for d in dims] == ["link_roughness", "node_max_depth"]
    assert dims[0].labels == ("link_roughness:C1", "link_roughness:C2")
    low, high = flat_bounds(dims)
    assert low.shape == high.shape == (3,)
    rtc = EnvConfig(
        env_type="rtc", inp_path="m.inp", observations=ObservationSpec(node_depths=["J1"])
    )
    with pytest.raises(SpecError, match="searchable static factory"):
        design_dimensions(rtc)


def test_config_reports_only_search_settings_as_resolved():
    opt = OptimizationConfig(algorithm="nsga2", budget=60, population_size=12, workers=3)
    assert opt.resolved() == {
        "algorithm": "nsga2",
        "budget": 60,
        "seed": None,
        "population_size": 12,
    }
    with pytest.raises(ValueError):
        OptimizationConfig(workers=-1)


# ---------------------------------------------------------------------------
# Studies on the real engine
# ---------------------------------------------------------------------------


def test_parallel_and_serial_studies_log_identical_evaluations():
    serial_dir, parallel_dir = _out("serial"), _out("parallel")
    opt = {"algorithm": "random_search", "budget": 6, "seed": 7}
    serial = run_study(
        _cip_config(_copy_inp(serial_dir)), OptimizationConfig(**opt, workers=0), serial_dir
    )
    parallel = run_study(
        _cip_config(_copy_inp(parallel_dir)), OptimizationConfig(**opt, workers=2), parallel_dir
    )
    assert serial["execution"]["workers"] == 0 and parallel["execution"]["workers"] == 2
    assert (serial_dir / "evaluations.jsonl").read_text() == (
        parallel_dir / "evaluations.jsonl"
    ).read_text()
    assert serial["pareto"] == parallel["pareto"]
    assert json.loads((parallel_dir / "result.json").read_text())["evaluations_count"] == 6


def test_progress_and_cancellation():
    out = _out("cancel")
    seen: list[int] = []
    stop = threading.Event()

    def progress(record):
        seen.append(record["evaluation"])
        if len(seen) == 3:
            stop.set()

    with pytest.raises(StudyCancelled):
        run_study(
            _cip_config(_copy_inp(out)),
            OptimizationConfig(algorithm="random_search", budget=500, seed=1, workers=1),
            out,
            should_stop=stop.is_set,
            on_evaluation=progress,
        )
    assert seen[:3] == [1, 2, 3] and len(seen) < 500


def test_a_dead_worker_fails_only_its_candidate():
    out = _out("dead_worker")
    config = _cip_config(_copy_inp(out))
    dims = design_dimensions(config)
    low, high = flat_bounds(dims)
    vectors = [low + (high - low) * f for f in (0.2, 0.5, 0.8)]
    with WorkerPool(config, dims, workers=1, seed=0) as pool:
        os.kill(pool._pool[0].process.pid, 9)  # an engine crash, as the OS sees it
        pool._pool[0].process.join()
        healthy = pool.evaluate(vectors)
    # The idle worker died before receiving work: it is replaced and nothing fails.
    assert [e.error for e in healthy] == [None, None, None]

    with WorkerPool(config, dims, workers=1, seed=0) as pool:
        pool.poll_interval = 0.001  # check should_stop every millisecond while busy
        calls = []

        def kill_the_busy_worker() -> bool:
            # Call 1 precedes the first dispatch; call 2 comes a millisecond after the
            # first candidate was sent, while its episode is still running.
            calls.append(1)
            if len(calls) == 2:
                os.kill(pool._pool[0].process.pid, 9)
            return False

        results = pool.evaluate(vectors, kill_the_busy_worker)
    assert [i for i, e in enumerate(results) if e.error] == [0]
    assert "worker exited" in results[0].error
    assert all(results[i].costs is not None for i in (1, 2))
