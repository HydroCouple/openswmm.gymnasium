# Testing

Per plan §8.0, **all tests drive the real `openswmm.engine.Solver`** —
there are no mocks anywhere in the suite. The unit / integration /
regression split is by **scope**, not by whether the engine is real.

## Running the suite

```bash
pip install -e ".[dev,mo,viz]"
pytest tests/unit/ -v
```

Integration-marked tests (`@pytest.mark.integration`) are full-episode
runs against the real engine. They run by default because the engine
is a required dependency. The `OPENSWMM_RUN_INTEGRATION` env var and
`--run-integration` flag are retained as no-ops for symmetry with
`openswmm.mcp` conventions.

## Scope tiers

```{list-table}
:header-rows: 1

* - Tier
  - Location
  - Purpose
* - Unit
  - `tests/unit/`
  - Per-component tests; tiny in-tree fixtures
* - Integration
  - `tests/integration/`
  - End-to-end via `gymnasium.utils.env_checker.check_env`
* - Regression
  - `tests/regression/`
  - Golden-trajectory JSONL replay
```

## Fixtures

The root `tests/conftest.py` provides:

- `minimal_inp` — per-test copy of `tests/data/minimal.inp` (1 junction
  + 1 outfall + 1 conduit; 30-min sim).
- `minimal_rpt` / `minimal_out` — sibling paths.
- `solver_adapter` — factory that constructs (and tears down) a
  {py:class}`~openswmm_gymnasium._engine.SolverAdapter`.

## Linting

```bash
ruff check src/ tests/
ruff format src/ tests/
```

## Coverage

```bash
pytest tests/unit/ --cov=openswmm_gymnasium --cov-report=term-missing
```

Target: ≥90% on `src/openswmm_gymnasium/` (including the `_engine/`
adapter; integration tests cover the live-engine surface).

## Vector-env tests

`tests/integration/test_vector_env.py` runs both
`gymnasium.vector.SyncVectorEnv` (threads) and `AsyncVectorEnv`
(processes) over the same env IDs with identical seeds and asserts the
trajectories match. This directly verifies the §2.3 thread-isolation
guarantee.
