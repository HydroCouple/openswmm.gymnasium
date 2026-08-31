# Changelog

All notable changes to **openswmm.gymnasium** are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added — process-configuration surfaces (heat, water age, reactions)

- **Adapter reach for the engine's heat, water-age and reaction modules.**
  New `_HeatCompat`, `_WaterAgeCompat` and `_ReactionsCompat` shims behind
  lazily-cached `SolverAdapter.heat` / `.water_age` / `.reactions`
  accessors. These are *configuration* surfaces, not observation surfaces:
  the C API exposes no per-node or per-link temperature, water-age or
  species-concentration getter, so nothing here is wired into
  `ObservationBuilder`. The only genuinely observable new state is the two
  current-step scalars `heat.current_shortwave` and
  `heat.current_cloud_fraction`, both of which are forcing rather than state.
- **Two-tier guards on the new optional accessors.** Tier 1 asks whether the
  engine *build* carries the module; tier 2 asks whether the open *model*
  enables it (`[OPTIONS] HEAT_TRANSPORT` / `WATER_AGE`). Both raise with the
  remedy named rather than returning defaults, because a heat or water-age
  configuration written into a model that never routes it is a silent no-op —
  stored, never transported, nothing raised, no reward signal moving. None of
  the three is added to `_REQUIRED_MODULE_ATTRS`: listing an optional module
  would make a partial build unusable for every env rather than only for the
  envs that touch it.
- **`ReactionCoefficientValue` design factory.** Searches
  `[REACTION_COEFFICIENTS]` **PARAMETER** values — the rate constants,
  half-saturation constants, yields and stoichiometric factors a
  water-quality modeller normally fits by hand. Turns calibration into an
  ordinary optimisation over the same env machinery that does CIP sizing.
  CONSTANT coefficients are refused at `bind` rather than written and
  silently ignored. Bounds are in the model's own expression units,
  unconverted.
- **`HeatSourceTemperature` design factory** — global inlet temperature per
  heat-source pathway, degC, bounded by default to the engine's own
  `[-50, 100]` refusal range so a sampled action can never be refused
  mid-episode (the engine refuses rather than clamps, and a refused write
  does not take effect).
- **`WaterAgeSourceAge` design factory** — global source age per water-age
  pathway, hours. Negative values are legal and meaningful (age-volume
  extraction; the engine clamps the *result* at zero, not the input), so the
  low bound is not floored at zero and has no default.
- **`HeatSourceTemperatureSetpoint` runtime actuator** — per-pathway inlet
  temperature applied every step. Heat source writes are documented live, so
  the same engine call backs both the design factory and this actuator. No
  water-age runtime twin ships: a source's assigned age is a bookkeeping
  label rather than something an operator can move during an event, so
  per-step modulation would let an agent chase reward by rewriting its own
  accounting.
- **Preissmann-slot link readers.** `SolverAdapter.links.slot_volume`,
  `.peak_slot_share` and `.slot_share`. All three read a hard `0.0` under any
  router other than `FLOW_ROUTING FV` — a value indistinguishable from "no
  slot flow" — which is documented loudly at the call site.
- **`SurchargeSlotShare` reward term.** Run-level Preissmann-slot storage
  share as a pressurisation proxy; dimensionless in `[0, 1]`, the one term
  with no unit-system dependence. Registered as `"surcharge_slot_share"`.
  `bind` reads `[OPTIONS] FLOW_ROUTING` and **raises** on any non-FV router
  rather than reporting a permanently perfect network. The statistic is a
  ratio of time integrals and cannot be reconstructed from env-step samples.
- **`SolverAdapter.get_option`** — the read counterpart of `set_option`,
  used to answer model-level questions such as which router is active.

### Added — declarative-config reach (`openswmm.mcp`)

- **Observation collectors reachable from an `EnvConfig`.**
  `add_pollutant_concentration`, `add_link_pollutant_concentration` and
  `add_2d_vertex_depths` existed in code but had no `_OBS_METHODS` entry, so
  they were unreachable declaratively. Added as the `ObservationSpec` fields
  `node_pollutant_concentration` / `link_pollutant_concentration` (each a
  pollutant-ID → element-IDs map, so several pollutants can be requested at
  once) and `vertex_depths_2d`.
- **Registry entries** for every new factory and term, plus `tss_load`,
  which shipped in G4 but was never registered and so was likewise
  unreachable from a declarative config.

### Fixed — documentation

- `docs/user-guide/observations.md`'s collector table omitted six collectors
  that exist in code (`add_node_volumes`, `add_node_lateral_inflows`,
  `add_link_velocities`, `add_link_capacities`, `add_link_volumes`,
  `add_2d_vertex_depths`).
- `docs/user-guide/rewards.md`'s term table omitted `UncontrolledDischarge`,
  `StorageUnderUtilization` and `PumpEnergy`, all three of which the prose
  below it already referenced.
- `docs/user-guide/action_spaces.md`'s runtime table omitted
  `NodeLateralInflow` and named the wrong engine surface for `OrificeSetting`
  (`Controls.set_link_setting`, which the engine recomputes each routing step
  and so does not stick, rather than `Links.set_target_setting`).
- `docs/developer/testing.md`'s "Fixtures" section described pytest
  `conftest.py` fixtures that no longer exist; it now documents the
  `tests/unit/_base.py` base classes, the hand-rolled-fake convention, and
  the `skipUnless` idiom for build-optional engine surfaces. The run, lint
  and coverage commands were corrected to stdlib `unittest`.

### Changed

- **Relicensed from MIT to the Apache License, Version 2.0.** `LICENSE` now
  carries the full Apache 2.0 text and a new `NOTICE` file records the required
  attribution, including the USEPA SWMM public domain provenance inherited
  through the engine. All first-party source headers carry the Apache 2.0
  boilerplate and an `SPDX-License-Identifier: Apache-2.0` tag. `pyproject.toml`
  declares `license = "Apache-2.0"` with `license-files = ["LICENSE", "NOTICE"]`,
  and `CLA.md` (v1.1), `CONTRIBUTING.md` and `README.md` were updated to match.

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
