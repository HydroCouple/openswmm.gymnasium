# Testing

Per plan §8.0, **all tests drive the real `openswmm.engine.Solver`** —
there are no mocks anywhere in the suite. The unit / integration /
regression split is by **scope**, not by whether the engine is real.

## Running the suite

```bash
pip install -e ".[dev,mo,viz]"
python -m unittest discover tests/unit -v
```

The suite is stdlib `unittest` — there is no pytest dependency and no
`conftest.py`. Run a single module or case the usual way:

```bash
python -m unittest tests.unit.test_process_surfaces -v
python -m unittest tests.unit.test_process_surfaces.TestSurchargeSlotShare
```

Full-episode tests against the real engine run by default, because the
engine is a required dependency rather than an opt-in extra.

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

## Base classes

The suite runs under stdlib `unittest`, with **no pytest dependency and no
`conftest.py`**. What used to be pytest fixtures now lives in
`tests/unit/_base.py` as two `unittest.TestCase` base classes:

- **`BaseTestCase`** — gives every test an isolated, auto-cleaned
  `self.tmp_path` (a `pathlib.Path` to a fresh temp directory). The stdlib
  replacement for pytest's function-scoped `tmp_path`.
- **`BaseEngineTest`** — extends `BaseTestCase` with a per-test copy of
  `tests/data/minimal.inp` (1 junction + 1 outfall + 1 conduit, 30-minute
  sim) as `self.minimal_inp`, sibling `self.minimal_rpt` /
  `self.minimal_out` paths, and a
  `self.make_adapter(open=True)` factory that constructs a
  {py:class}`~openswmm_gymnasium._engine.SolverAdapter` and registers it for
  teardown, so engine handles never leak between tests. With `open=True`
  (the default) the adapter is opened, initialized and started — left in the
  engine `RUNNING` state, ready to step.

Import them by full path, matching the rest of the suite:

```python
from tests.unit._base import BaseEngineTest
```

Every test file ends with the `unittest.main()` guard.

### Fakes vs. the real engine

Engine-touching tests subclass `BaseEngineTest` and drive the real solver.
Pure-logic tests (action-factory arithmetic, reward accumulators,
observation wiring) use **hand-rolled `_Fake*` classes at module scope** —
never `unittest.mock`. `tests/unit/test_quality_observations_rewards.py` is
the canonical template, including its `self.bulk_reads` counter idiom for
asserting a collector took the bulk read path rather than N scalar reads.

### Optional engine surfaces

Some modules are build-optional (2D, heat, water age, reactions). A test
class that needs one guards itself at class level, mirroring the
accessor-level probes in `solver_adapter.py`:

```python
@unittest.skipUnless(
    hasattr(_engine, "Heat") and hasattr(_engine, "WaterAge"),
    "installed openswmm.engine has no Heat / WaterAge module",
)
class TestOptionalSurfaceGuards(BaseEngineTest):
    ...
```

See `tests/unit/test_process_surfaces.py`.

## Linting

```bash
python -m ruff check src/ tests/
python -m ruff format src/ tests/
```

## Coverage

```bash
python -m coverage run -m unittest discover tests/unit
python -m coverage report -m
```

Target: ≥90% on `src/openswmm_gymnasium/` (including the `_engine/`
adapter; integration tests cover the live-engine surface).

## Vector-env tests

`tests/integration/test_vector_env.py` runs both
`gymnasium.vector.SyncVectorEnv` (threads) and `AsyncVectorEnv`
(processes) over the same env IDs with identical seeds and asserts the
trajectories match. This directly verifies the §2.3 thread-isolation
guarantee.


## Current binding contract checks

Run these with the same compiled engine and `openswmm.engine.catalog` that will
be used by the application (for local validation, activate Conda `openswmm`):

```bash
python -m pytest tests/unit/test_catalog_fields.py tests/unit/test_observations_2d.py \
  tests/unit/test_engine_capabilities.py tests/unit/test_spec_config.py \
  tests/unit/test_spec_registry.py tests/unit/test_spec_envs.py -q
```

Generic `add_field` accepts scalar numeric **element properties**, not every
catalogued method or service. `add_cell_field` handles supported scalar/bulk
2D methods and verifies cell counts. A catalog reachability test does not prove
that every field is a useful observation, every writable field is a runtime
actuator, or every optional native feature is present. New domains still need
lifecycle, units, bounds and successful end-to-end fixtures.
