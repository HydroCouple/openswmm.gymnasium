# Changelog

All notable changes to **openswmm.gymnasium** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — API gap-fill phases G2–G5

- **Shape-aware cross-section sizing (G2).** `LinkDiameter` now resizes a
  section to a target **rise** (full depth, read from the engine's
  `XSectionGeometry`), scaling every length-dimensioned geometry parameter of
  the shape by the same factor. A box culvert's width now moves with its
  height instead of being left at its baseline. Shapes with no scalable
  dimension (`IRREGULAR`, `STREET_XSECT`, `DUMMY`) are rejected at `bind`.
- **Exact filling ratios (G2).** `MarketMetricReader`'s `filling_ratio`
  normalises by the section's true rise instead of `geom1`, so it is exact for
  every shape rather than "approximate for non-CIRCULAR".
- **Tabular storage design (G3).** `StorageVolume` accepts TABULAR storage
  nodes, scaling the node's depth–area curve through the new
  `SolverAdapter.tables` accessor, instead of requiring the `NodeMaxDepth`
  proxy. `mode="coeffs"` stays FUNCTIONAL-only; geometric storage shapes and
  curves shared between two target nodes are rejected at `bind`.
- **Pollutant observations (G4).** `ObservationBuilder.add_pollutant_concentration`
  (nodes) and `add_link_pollutant_concentration`, both using the engine's bulk
  `qualities()` read.
- **`TSSLoad` reward term (G4).** Pollutant mass flux
  (`flow × concentration × dt`) through a set of links; works for any declared
  pollutant. Registered as `"tss_load"`.
- **Engine capability probe (G5).** `SolverAdapter` construction now verifies
  the installed `openswmm.engine` exposes every symbol this package calls and
  raises `EngineCapabilityError` naming the missing ones. Replaces two ad-hoc
  `getattr(..., None)` fallbacks. Optional surfaces (the 2D module) are
  deliberately not probed, so `OPENSWMM_BUILD_2D=OFF` builds still work.

### Changed

- `FloodingVolume` / `CSOVolume` now read the engine's cumulative
  `swmm_node_get_stat_vol_flooded` statistic and report its per-step
  increment, instead of sampling the instantaneous overflow rate once per env
  step and multiplying by `dt`. The engine integrates every routing step, so
  volume that occurs between two env steps is no longer lost.
  **The reported quantity is now in project volume units (ft³ / m³)** and no
  longer scales with `dt_seconds`; for CFS/CMS models the numbers are
  unchanged in kind, for GPM/MGD/LPS/MLD models the units differ from before.
- `PeakOutflow` reads the engine's cumulative `swmm_link_get_stat_max_flow`
  rather than sampling `link.flow`, so a peak between two env steps is caught.
  Units and semantics (project flow units, cumulative equals the peak) are
  unchanged.

### Fixed

- **`NodeLateralInflow.apply` raised `AttributeError` on every call (G1).** It
  reached for `adapter.set_lateral_inflow`, which `SolverAdapter` does not
  define — the setter lives on the `nodes` collection and there is no
  `__getattr__` delegation, so the action was never applied. It now goes
  through `adapter.nodes.set_lateral_inflow`, matching how the sibling
  `OrificeSetting` actuator reaches `adapter.links`. Covered by an engine-free
  regression test so the delegation path is checked without a built engine.

### Added — initial release surface (v0.1.0)

#### Framework envs
- `SwmmRTCEnv` — runtime-only Gymnasium env over `openswmm.engine.Solver`.
- `SwmmCIPEnv` — single-step contextual-bandit env for design-only optimization.
- `SwmmJointCIPRTCEnv` — hybrid CIP + RTC env; design applied at `reset()`,
  runtime applied each `step()`.
- `SwmmMORTCEnv` — multi-objective RTC env returning vector reward;
  populates `info["mo_score"]` at terminal step via single-point normalized
  hypervolume. Soft-imports `mo_gymnasium.MOEnv` when available.

#### Action spaces
- Plan §3 contract: every env exposes
  `spaces.Dict({"design": Dict({...}), "runtime": Dict({...})})`,
  with either half optionally empty.
- Runtime factory: `OrificeSetting(link_ids)` →
  `Controls.set_link_setting` (defensive `np.clip` to bounds).
- Design factories: `LinkRoughness`, `LinkLength`, `LinkDiameter`
  (preserves cross-section shape; see G2 above for the shape-aware sizing
  that superseded its original `geom1`-only behaviour), `NodeMaxDepth`.
- Design factories applied **between** `Solver.open()` and
  `Solver.initialize()` so the engine picks up overridden values during
  data-structure setup.
- Asset-sizing design factories for exhaustive design exploration:
  `StorageVolume` (FUNCTIONAL storage sizing — scalar footprint multiplier or
  raw `(a, b, c)` coefficients), `LIDPlacement` (green-infrastructure /
  nature-based-solution sizing + LID type selection per subcatchment), and
  `RDIIUnitHydrograph` (RDII R-fraction and optional initial-abstraction
  sizing, preserving T and K). Backed by new `SolverAdapter` storage /
  `infrastructure` / `inflows` setters.

#### Observations
- `ObservationBuilder` with 10 collectors covering nodes (depths, heads,
  inflows, overflows), links (flows, depths, settings), subcatchments
  (runoff), gages (rainfall), and clock (`hour_sin`, `hour_cos`,
  `elapsed_frac`).
- Flat `Box(-inf, +inf, shape=(N,), float32)` output.

#### Rewards
- `RewardTerm` runtime-checkable Protocol.
- Built-in terms: `FloodingVolume`, `CSOVolume`, `PeakOutflow`,
  `ReliabilityMargin` (maximize), `SetpointSmoothness`.
- `RewardRegistry` for custom term registration; built-ins
  self-register at import.
- Plan §0 #5 sign convention: terms framed as cost-to-minimize
  internally; env negates after sign-flipping per term direction so
  higher reward is better.

#### Multi-objective scoring
- Pure-numpy `openswmm_gymnasium.scoring` module — no `pymoo` /
  `platypus-opt` dependency on the critical path.
- Pareto-front extraction, normalization, hypervolume (2-D exact sweep
  + N-D Monte Carlo + `normalized_hypervolume`), IGD, IGD+, additive
  ε-indicator, Schott's spread, weighted-Tchebycheff R2.

#### Wrappers
- `RescaleBoxActions` — recursive Box leaf rescaling over Dict + Tuple.
- `MaskDesignAction` / `MaskRuntimeAction` — flatten the Dict action
  space; freeze the other half (sample-at-reset or supplied midpoint).
- `LinearScalarize` / `TchebycheffScalarize` — vector reward → scalar.
- `ForecastObservation` — append user-supplied forecast features.
- `RecordTrajectory` — write one JSONL file per episode.

#### Visualization (optional `[viz]` extra)
- `Trajectory.from_jsonl(path)` and `TrajectoryRun.from_dir(path)`
  loaders with numpy-only aggregation.
- Eight Plotly figure factories: `reward_curves`, `pareto_front`
  (2-D / 3-D), `hypervolume_trace`, `action_timeseries`,
  `network_state_heatmap`, `flooding_attribution`, `objective_radar`,
  `trajectory_replay`.
- Hard `ImportError` guard when Plotly is missing.

#### Benchmarks
- `b01 TwinTank` — two storage tanks in series with a controllable
  side orifice. Registered as `OpenSWMM/TwinTank-RTC-v0`,
  `OpenSWMM/TwinTank-Joint-v0`, `OpenSWMM/TwinTank-MORTC-v0`.
- Framework + status table for b02-b10 (pending in follow-up releases).

#### Tooling
- Ruff lint + format for `src/` and `tests/`.
- Pytest with `integration` marker.
- Sphinx documentation in `docs/` with `sphinx-epytext` extension
  rendering the package's epytext docstrings; `sphinx-build -W`
  zero-warning verified.

#### Engine contract
- Hard requirement on `openswmm.engine` v6 (handle-based, thread-safe).
- `LegacySolverRejectedError` guard at adapter construction.
- Each `SolverAdapter` owns a distinct `SWMM_Engine` handle, enabling
  both `gymnasium.vector.SyncVectorEnv` (threads) and `AsyncVectorEnv`
  (processes) rollouts.
- `SolverAdapter.open(lenient=True)` opt-in permissive open plus
  `SolverAdapter.open_errors` / `SolverAdapter.open_warnings` accessors,
  surfacing the engine's `set_lenient_open` / validation-accumulator
  API for pre-flight validation of programmatically-generated or
  perturbed training models (broken candidates are reported/rejected
  instead of crashing the rollout). The env run path stays strict.

### Conventions
- Docstrings: epytext (`@param`, `@type`, `@return`, `@rtype`,
  `@raise`, `@ivar`). Plan §13.
- Tests: drive the real `openswmm.engine.Solver` against tiny in-tree
  fixtures; no mocks. Plan §8.0.

[Unreleased]: https://github.com/HydroCouple/openswmm.gymnasium/compare/main...HEAD
