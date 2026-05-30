# Changelog

All notable changes to **openswmm.gymnasium** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
  (preserves cross-section shape), `NodeMaxDepth`.
- Design factories applied **between** `Solver.open()` and
  `Solver.initialize()` so the engine picks up overridden values during
  data-structure setup.

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

### Conventions
- Docstrings: epytext (`@param`, `@type`, `@return`, `@rtype`,
  `@raise`, `@ivar`). Plan §13.
- Tests: drive the real `openswmm.engine.Solver` against tiny in-tree
  fixtures; no mocks. Plan §8.0.

[Unreleased]: https://github.com/HydroCouple/openswmm.gymnasium/compare/main...HEAD
