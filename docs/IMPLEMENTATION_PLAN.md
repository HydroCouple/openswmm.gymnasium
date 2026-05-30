# openswmm.gymnasium — Implementation Plan

**Status:** Draft for review
**Owner:** @cbuahin
**Last updated:** 2026-05-30
**Related repos:** [openswmm.engine](https://github.com/HydroCouple/openswmm.engine), [openswmm.mcp](https://github.com/HydroCouple/openswmm.mcp)
**Upstream:** [Farama Gymnasium](https://gymnasium.farama.org/), [MO-Gymnasium](https://mo-gymnasium.farama.org/)

This document is the authoritative, vetted plan that governs implementation
of `openswmm.gymnasium`. Per the project rule in `CLAUDE.md §5.0`,
implementation work follows this plan rather than introducing parallel
strategies. Deviations require an explicit edit to this document first.

---

## 0. Assumptions & Open Questions

Stated up front per `CLAUDE.md §1` (Think Before Coding):

1. **Gymnasium 1.x API** is the target (`reset → (obs, info)`,
   `step → (obs, reward, terminated, truncated, info)`). The Gym 0.21 API
   will not be supported; `StepAPICompatibility` is not used.
2. **MO-Gymnasium compatibility** is desired but optional at install time:
   `openswmm.gymnasium` will subclass `gymnasium.Env` directly for the
   single-objective envs and subclass `mo_gymnasium.MOEnv` only when
   `mo-gymnasium` is installed (soft import / optional extra).
3. **openswmm.engine Python API** (the `openswmm.engine.Solver` plus
   `Nodes`, `Links`, `Subcatchments`, `Controls`, `Forcing`, `ModelEditor`,
   `Inflows`, `Pollutants`, `Tables`, `OutputReader`) is the sole engine
   surface — no FFI is wrapped here.
4. **CIP** = Capital Improvement Plan (static design decisions: pipe
   diameters, storage volumes, weir crest elevations, pump curves, LID
   placements, etc.). **RTC** = Real-Time Control (per-step setpoints:
   pump on/off, gate positions, orifice openings).
5. **Reward direction:** all built-in objectives are framed as
   **cost-to-minimize** internally, then negated for the agent so the
   Gymnasium "higher reward is better" convention holds. This keeps
   hypervolume computation against a single reference point unambiguous.
6. **Currency for capital cost** is a user-configurable scalar; the
   library ships dimensionless cost curves and lets users supply unit
   conversions.
7. **Thread-safe engine only.** `openswmm.gymnasium` targets **only**
   the thread-safe, handle-based `openswmm.engine.Solver` (v6 — each
   `Solver` owns its own `SWMM_Engine` handle from `swmm_engine_create`
   / `swmm_engine_destroy`). The legacy singleton solver under
   `openswmm.legacy.engine` is **not supported** and will not be
   imported anywhere in this package. This is a hard requirement, not
   a soft preference: it shapes the vector-env story (§2.3), the
   dependency floor (§9), and the testing matrix (§8). The MCP-style
   dual-backend (`engine="openswmm"|"legacy"`) parametrization is
   intentionally absent.

**Open questions — resolved 2026-05-30:**

- **Q1 — Solver lifecycle ownership.** ✅ Env owns lifecycle by default;
  expose a `solver=` injection hook for tests and advanced wrappers.
- **Q2 — First-class reward terms in v0.1.0.** ✅ Flooding volume,
  CSO/overflow volume, pump energy, peak outflow, TSS load, and CIP
  capital cost. Everything else via a user-supplied `RewardFn` callable.
- **Q3 — Action remapping.** ✅ Wrapper, not a built-in space option,
  so the raw action space remains physically meaningful.
- **Q4 — Bundled reference INPs.** ✅ **Do not bundle client models.**
  Instead, ship **10 contrived benchmark scenarios** authored from
  scratch under the project's own MIT license. Each scenario is
  hand-designed to expose a specific class of competing objectives so
  the suite collectively stresses every public surface (CIP-only,
  RTC-only, joint, multi-objective scoring, stochastic forcing,
  pollutant routing, coordination). Full catalogue in §6A.

---

## 1. Scope

### In scope (v0.1.x)

A Gymnasium-compatible reinforcement learning framework for joint
**Capital Improvement Plan (CIP)** design and **Real-Time Control (RTC)**
operation of SWMM models, with multi-objective scoring.

- `gymnasium.Env` subclass driving an `openswmm.engine.Solver` step loop.
- Hybrid action space combining design-time (CIP) attributes set once at
  `reset()` and runtime (RTC) setpoints applied each `step()`.
- Composable observation builder over node depths/heads/flows, link
  flows/depths/settings, subcatchment runoff, rainfall, and pollutant
  concentrations.
- Composable reward registry with the standard SWMM-optimization terms
  (flooding, overflow, energy, water quality, capital cost) plus a
  user-extension interface.
- Multi-objective extension via `mo_gymnasium.MOEnv` subclass returning
  vector rewards.
- Multi-objective scoring: **normalized hypervolume** (primary),
  **IGD / IGD⁺**, **ε-indicator**, **spread**, and **R2** indicator.
- Bundled reference scenarios + regression-tested golden trajectories.
- Sphinx documentation mirroring the `openswmm.mcp` doc tree.
- Unit + integration + regression test suites mirroring the
  `openswmm.mcp` test layout (`tests/{unit,integration,regression}` with
  `conftest.py` patterns).

### Explicitly out of scope (v0.1.x)

- Training algorithms (PPO, SAC, MORL-Baselines wrappers, etc.). The
  package ships **environments**; agents live in user code or in
  downstream packages.
- Distributed rollout. `gymnasium.vector` provides what's needed.
- A bespoke neural-network architecture.
- Bayesian/evolutionary CIP search outside the Gymnasium contract.
  (However, the design-time action space is intentionally compatible with
  `pymoo` and `optuna` via a thin adapter — see §5.4.)

---

## 2. Architecture

### 2.1 Class hierarchy

```
gymnasium.Env                                          (Farama)
│
├── SwmmEnv                                            (this package)
│       owns: openswmm.engine.Solver
│       owns: ObservationBuilder, ActionApplier,
│              RewardComposer, EpisodeClock
│
│       ┌── SwmmCIPEnv          (design-only, single episode = one design)
│       ├── SwmmRTCEnv          (runtime-only, design fixed)
│       └── SwmmJointCIPRTCEnv  (default — full hybrid)
│
mo_gymnasium.MOEnv                                     (Farama, optional)
│
└── SwmmMOEnv (mixin SwmmEnv + MOEnv → vector reward)
        └── SwmmJointMOEnv
```

`SwmmMOEnv` is conditionally defined behind `try: import mo_gymnasium`.
The single-objective hierarchy works without it.

### 2.2 Solver lifecycle bridge

`SwmmEnv.reset(seed, options)`:

1. Close any prior `Solver` (idempotent).
2. If a design action was carried over from a previous episode and the
   env is in "CIP-then-RTC" mode, persist it to a working `.inp` via
   `ModelEditor`.
3. Open new `Solver(inp, rpt, out)` against the working INP.
4. Apply design-time action (`spaces["design"]`) using `ModelEditor` and
   pre-simulation edits on `Nodes` / `Links` / `Infrastructure`.
5. `solver.open()` → `solver.initialize()`.
6. Build the initial observation; return `(obs, info)`.

`SwmmEnv.step(action)`:

1. Split `action` into `action["design"]` (no-op after reset) and
   `action["runtime"]`.
2. Apply runtime action via `Controls.set_link_setting` /
   `set_link_status`, `Inflows`, and `Forcing` — the same surfaces used
   by `openswmm.mcp` tools.
3. Advance the solver one or more routing steps via `solver.stride(n)`
   (the `n` is `options["control_interval_steps"]`, default 1).
4. Read state, build observation, compute reward(s), determine
   `terminated` (sim finished) / `truncated` (wall-clock or step cap),
   build `info`.
5. On `terminated`: `solver.end()` → `solver.report()` → `solver.close()`.

### 2.3 Threading & vectorized rollout

Per Assumption #7, this package targets only the handle-based,
thread-safe `openswmm.engine.Solver`. Each `SwmmEnv` instance owns a
distinct `SWMM_Engine` handle (allocated by `swmm_engine_create` in the
`Solver` constructor and freed in `Solver.close()` / destructor), so
two `SwmmEnv` instances on different threads do not share C-side
state.

Consequences:

- **`gymnasium.vector.SyncVectorEnv` is supported** in-process across
  threads, not just across processes. A worker pool can drive N envs
  concurrently from a single Python process; the GIL is released by
  the underlying Cython bindings during `solver.step()`, so
  thread-based parallel rollout actually scales.
- **`gymnasium.vector.AsyncVectorEnv` (subprocess) remains supported**
  for users who want process isolation (CUDA, memory limits, etc.).
- **`tests/integration/test_vector_env.py`** explicitly covers both
  `SyncVectorEnv` (threads) and `AsyncVectorEnv` (processes) with the
  same env IDs, asserting identical trajectories from identical seeds.
- The `_engine/solver_adapter.py` adapter contains an explicit
  `assert solver.__class__.__module__.startswith("openswmm.engine")`
  on construction to guarantee no legacy solver leaks in via dependency
  injection.

This package will **not** introduce module-level engine state, global
configuration singletons, or shared C buffers — any contribution that
does is rejected at PR review.

### 2.4 Determinism

`reset(seed)` seeds:

1. The internal Python `random.Random` used by the design-action sampler.
2. The internal NumPy `Generator` used by stochastic forcing samplers.
3. (Optionally, if the user uses our forcing wrapper) the rainfall
   sampler.

The underlying SWMM solver is deterministic given a fixed INP + forcing.

---

## 3. Action Space — Design (CIP) + Runtime (RTC)

The action space is **always** a `gymnasium.spaces.Dict` with two keys:

```python
spaces.Dict({
    "design":  spaces.Dict({...}),   # static, sampled at reset() only
    "runtime": spaces.Dict({...}),   # dynamic, applied each step()
})
```

This satisfies Gymnasium's structured-action requirements and lets agents
that only care about one side mask the other via a wrapper.

### 3.1 Design (CIP) action factories

Each factory takes the open `Solver` (during env construction) and
returns a Gymnasium space + an `apply(solver, action_value)` function
that mutates the model **before** `solver.initialize()`.

| Factory                            | Underlying engine API                | Space type            |
|------------------------------------|--------------------------------------|-----------------------|
| `LinkDiameter(link_ids, bounds)`   | `Links[id].xsection` + `ModelEditor` | `Box(low, high)`      |
| `LinkLength(link_ids, bounds)`     | `Links[id].length`                   | `Box`                 |
| `LinkRoughness(link_ids, bounds)`  | `Links[id].roughness`                | `Box`                 |
| `StorageVolume(node_ids, bounds)`  | `Nodes[id].StorageView.functional`   | `Box`                 |
| `OutfallStage(node_ids, choices)`  | `OutfallView.set_stage` / curve      | `Box` or `Discrete`   |
| `WeirCrestElev(link_ids, bounds)`  | `Links[id].WeirView.crest_height`    | `Box`                 |
| `OrificeMaxOpening(link_ids, …)`   | `OrificeView`                        | `Box`                 |
| `PumpCurveChoice(link_ids, curves)`| `PumpView.curve` index               | `Discrete`            |
| `LIDPlacement(subcatch_ids, set)`  | `Infrastructure` API                 | `MultiBinary`         |
| `ControlRuleSelection(rule_set)`   | `Controls.append` / `clear`          | `MultiDiscrete`       |

Bounds and lists are user-supplied so the agent only searches the
designer's intended design envelope. Each factory carries a one-line
docstring summarizing the engine surface it mutates.

### 3.2 Runtime (RTC) action factories

| Factory                                  | Engine API                                        | Space            |
|------------------------------------------|---------------------------------------------------|------------------|
| `PumpOnOff(link_ids)`                    | `Controls.set_link_status(closed=…)`              | `MultiBinary`    |
| `OrificeSetting(link_ids)`               | `Controls.set_link_setting`                       | `Box([0,1])`     |
| `WeirSetting(link_ids)`                  | `Controls.set_link_setting`                       | `Box([0,1])`     |
| `PumpSpeed(link_ids)`                    | `Controls.set_link_setting`                       | `Box([0,1])`     |
| `OutfallStageSetpoint(node_ids)`         | `OutfallView.set_stage`                           | `Box`            |
| `ExternalInflow(node_ids)`               | `Inflows.set_value`                               | `Box`            |
| `ForecastInjection(gage_ids)`            | `Forcing.set_rainfall`                            | `Box`            |

### 3.3 Action validation

Every applier returns a typed `ActionImpact` namedtuple (object_kind,
object_id, attribute, old_value, new_value). `info["action_impacts"]`
exposes the list each step for replay / debugging — and powers the
regression-test golden trajectories.

---

## 4. Observation Space

Composable, declared via an `ObservationBuilder`:

```python
obs = ObservationBuilder()
obs.add_node_depths(node_ids=["J1","J2","J3"])
obs.add_link_flows(link_ids=["C1","C2"])
obs.add_link_settings(link_ids=["P1"])
obs.add_subcatch_runoff(subcatch_ids=ALL)
obs.add_rainfall(gage_ids=ALL)
obs.add_clock(features=["hour_sin","hour_cos","elapsed_frac"])
obs.add_forecast(gage_id="RainGage", horizon_steps=12)
obs.add_pollutant_concentration(node_ids=ALL, pollutant="TSS")
```

The builder emits a single `Box`, or a `Dict` of `Box`es when the user
prefers structured observation (set `flatten=False`).

All collectors map 1:1 onto the existing `openswmm.engine` accessors
(`Nodes[id].depth`, `Links[id].flow`, `Subcatchments[id].runoff`,
`Gages[id].rainfall`, `Quality.node_concentration(...)`). No new engine
APIs are required.

---

## 5. Rewards & Multi-Objective Scoring

### 5.1 Built-in objective terms

All implemented as `RewardTerm` subclasses; each exposes `direction` =
`"minimize" | "maximize"`, `unit`, `accumulator`, and computes a per-step
contribution.

| Term                  | Direction | Source                                  |
|-----------------------|-----------|-----------------------------------------|
| `FloodingVolume`      | minimize  | `Nodes` overflow integrated over step   |
| `CSOVolume`           | minimize  | tagged outfalls or junctions            |
| `PeakOutflow`         | minimize  | running max of `Links[outfall].flow`    |
| `PumpEnergy`          | minimize  | pump power × setting × dt               |
| `TSSLoad`             | minimize  | `Quality` × flow                        |
| `CapitalCost`         | minimize  | user cost curve on design action       |
| `OandMCost`           | minimize  | scalar per pump-cycle / setting-change  |
| `SetpointSmoothness`  | minimize  | L2 of Δsetting                          |
| `ReliabilityMargin`   | maximize  | min freeboard across nodes              |

User-supplied terms must subclass `RewardTerm` and register via
`RewardRegistry.register(name, term_cls)`.

### 5.2 Reward composition

```python
SingleObjective(weights={"flooding": 1.0, "energy": 0.1, "capital_cost": 1.0})
MultiObjective(["flooding", "energy", "capital_cost"])  # vector reward
```

Single-objective collapses the weighted sum into a scalar `float`.
Multi-objective returns a `np.ndarray` of shape `(n_objectives,)`,
matching the MO-Gymnasium contract.

### 5.3 Multi-objective scoring

Lives under `src/openswmm_gymnasium/scoring/`. Pure-Python +
`numpy` / `scipy`; no extra heavy deps. Used both inside the env
(`info["mo_score"]` at episode end) and as standalone utilities for
post-hoc analysis.

| Metric                   | Function                                         |
|--------------------------|--------------------------------------------------|
| Normalized hypervolume   | `normalized_hypervolume(front, ref, ideal)`      |
| Inverted Generational Distance (IGD / IGD⁺) | `igd(front, true_front)` / `igd_plus(...)` |
| ε-indicator              | `epsilon_indicator(front_a, front_b)`            |
| Spread / Δ-spread        | `spread(front)`                                  |
| R2 indicator             | `r2_indicator(front, weights, ref)`              |
| Pareto front extraction  | `pareto_front(points)`                           |

**Normalization protocol** (the user-facing default):

1. The user supplies a `ReferencePoint` (worst-case nadir) and an
   `IdealPoint` per env. If omitted, the env computes them adaptively
   over an evaluation buffer (with a clear warning).
2. Each objective is min-max normalized to `[0,1]` using
   `(value - ideal) / (ref - ideal)`.
3. Hypervolume is computed in normalized space against `ref = (1,…,1)`.
   Result is in `[0,1]` — the headline "normalized hypervolume".

Hypervolume uses an exact algorithm for ≤4 objectives (HSO / WFG) and
Monte-Carlo for ≥5. We do not vendor `pymoo`; we re-implement the small
exact algorithms (~200 LOC) and document the Monte-Carlo fallback.

### 5.4 Pareto-search adapter (non-Gymnasium)

A thin `scoring.adapters.PlatypusAdapter` exposes the CIP-only env as a
[Platypus](https://github.com/Project-Platypus/Platypus) `Problem`
subclass for users who want classical MOEAs (NSGA-II, NSGA-III,
MOEA/D, ε-MOEA, ε-NSGA-II, SPEA2, IBEA, GDE3, OMOPSO, SMPSO) directly.

**Platypus is the only supported non-Gymnasium MOO adapter.** Other
libraries (pymoo, DEAP, jMetalPy, etc.) are explicitly out of scope —
adding them would split the maintenance surface without adding
capabilities Platypus doesn't already cover for the CIP problem class.

The adapter is a ~70-line file kept outside the env critical path and
behind an optional `[platypus]` extra. Shape:

```python
from platypus import Problem, Real, Integer, Binary

class SwmmCipProblem(Problem):
    def __init__(self, env: SwmmCIPEnv, *, evaluator_seed: int = 0):
        # Map env.action_space["design"] -> Platypus variable types
        ...
        super().__init__(nvars, nobjs, nconstrs)

    def evaluate(self, solution):
        action = self._unpack(solution.variables)
        obs, info = self._env.reset(seed=self._seed,
                                    options={"design_action": action})
        # CIP envs terminate after the design step.
        solution.objectives[:] = info["mo_score_vector"]
```

The adapter handles `Real` / `Integer` / `Binary` variable mapping from
`gymnasium.spaces.Box` / `Discrete` / `MultiBinary`, deterministic
seeding, and writes per-evaluation traces to
`tests/_artifacts/<run>/` per the transparent-file-IO rule
(`CLAUDE.md §4.1`).

### 5.5 Visualization module (`viz/`)

A standardized, Plotly-only visualization layer for optimization
trajectories and MO-RL diagnostics. Plotly is **the sole supported
plotting backend**; matplotlib / bokeh / altair are out of scope.

**Design rules:**

1. **Do not conform to `gymnasium.Env.render()`.** The Gymnasium render
   contract is per-frame (RGB array / human window). MO-RL's interesting
   objects are *per-run* (training-wide hypervolume traces) and
   *per-episode-aggregate* (Pareto fronts). The viz module exposes its
   own `Trajectory` API that does not implement `render()`.
2. **No dashboard.** Each public function returns one
   `plotly.graph_objects.Figure`. Composition into Jupyter notebooks,
   Streamlit apps, or `openswmm.gui` panels is the user's job.
3. **Post-hoc only.** Viz consumes the JSONL trajectories already
   written by the `record_trajectory` wrapper (§6 layout). It does not
   touch a live `Solver` and does not import `openswmm.engine`.
4. **Optional extra `[viz]`.** Plotly is ~30 MB installed; headless
   training users should not pay for it. The module's `__init__.py`
   raises a clear `ImportError` with the install command if Plotly is
   missing.
5. **No pandas.** Aggregation uses NumPy only (already required) to
   keep the optional install lean.

**Public API:**

```python
from openswmm_gymnasium.viz import Trajectory, TrajectoryRun

# One JSONL run (single seed)
traj = Trajectory.from_jsonl("artifacts/run-0042.jsonl")
fig  = traj.figure("reward_curves")

# A multi-seed sweep
run = TrajectoryRun.from_dir("artifacts/sweep-2026-05-30/")
fig = run.figure("pareto_front", animate_over="iteration")
```

**Eight v0.1.0 plot factories:**

| Name                       | Purpose                                                              | Input                |
|----------------------------|----------------------------------------------------------------------|----------------------|
| `reward_curves`            | Per-episode scalar reward + rolling mean (faceted per objective MO)  | `Trajectory`         |
| `pareto_front`             | 2D/3D scatter, non-dominated highlighted; animatable over iterations | `TrajectoryRun`      |
| `hypervolume_trace`        | Normalized HV vs. training step with reference/ideal markers         | `TrajectoryRun`      |
| `action_timeseries`        | Action heatmap (time × control) + per-control distributions          | `Trajectory`         |
| `network_state_heatmap`    | Node depths / link flows over sim time — SWMM-specific               | `Trajectory`         |
| `flooding_attribution`     | Stacked area of flooded volume per node over time                    | `Trajectory`         |
| `objective_radar`          | Per-episode radar comparing multiple policies side by side           | list[`Trajectory`]   |
| `trajectory_replay`        | Animated network schematic with depth coloring on node XY            | `Trajectory`         |

Each factory is a small file under `viz/figures/` (~50–100 LOC), has
its own unit test asserting the returned figure has the expected number
of traces, axes, and animation frames, and is exercised by a snapshot
test that writes the rendered HTML to `tests/_artifacts/viz/` for human
review.

---

## 6. Package Layout

```
openswmm.gymnasium/
├── LICENSE                       # MIT (already in repo)
├── CLA.md                        # Contributor License Agreement
├── AUTHORS.md
├── CODE_OF_CONDUCT.md
├── CONTRIBUTING.md
├── SECURITY.md
├── README.md
├── pyproject.toml                # hatchling backend (mirrors openswmm.mcp)
├── docs/
│   ├── IMPLEMENTATION_PLAN.md    # this file
│   ├── conf.py
│   ├── index.rst
│   ├── getting-started/
│   │   ├── installation.md
│   │   ├── configuration.md
│   │   └── quickstart.md
│   ├── user-guide/
│   │   ├── envs.md
│   │   ├── action_spaces.md
│   │   ├── observations.md
│   │   ├── rewards.md
│   │   ├── scoring.md
│   │   └── wrappers.md
│   ├── developer/
│   │   ├── architecture.md
│   │   ├── testing.md
│   │   └── contributing.md
│   └── api/
│       └── index.rst
├── src/openswmm_gymnasium/
│   ├── __init__.py               # version + gymnasium.register() calls
│   ├── _version.py
│   ├── _engine/                  # adapter over openswmm.engine.Solver
│   │   ├── __init__.py
│   │   ├── solver_adapter.py     # Solver lifecycle + step bridge
│   │   ├── model_editor.py       # ModelEditor convenience layer
│   │   └── controls_bridge.py    # Controls / Inflows / Forcing wrappers
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── base.py               # SwmmEnv (single-obj)
│   │   ├── joint.py              # SwmmJointCIPRTCEnv
│   │   ├── cip_only.py           # SwmmCIPEnv
│   │   ├── rtc_only.py           # SwmmRTCEnv
│   │   └── mo.py                 # SwmmMOEnv (optional import)
│   ├── spaces/
│   │   ├── __init__.py
│   │   ├── design.py             # CIP action factories
│   │   ├── runtime.py            # RTC action factories
│   │   └── compose.py            # Dict assembly + masking
│   ├── observations/
│   │   ├── __init__.py
│   │   ├── builder.py            # ObservationBuilder
│   │   └── collectors.py         # individual feature collectors
│   ├── rewards/
│   │   ├── __init__.py
│   │   ├── registry.py
│   │   ├── terms.py              # built-in RewardTerm subclasses
│   │   └── composers.py          # SingleObjective / MultiObjective
│   ├── scoring/
│   │   ├── __init__.py
│   │   ├── hypervolume.py        # exact (HSO/WFG) + MC fallback
│   │   ├── igd.py
│   │   ├── epsilon.py
│   │   ├── spread.py
│   │   ├── r2.py
│   │   ├── pareto.py
│   │   ├── normalization.py
│   │   └── adapters/
│   │       └── platypus_adapter.py  # optional, sole supported MOO adapter
│   ├── wrappers/
│   │   ├── __init__.py
│   │   ├── action_remap.py       # [0,1] → physical units
│   │   ├── action_mask.py        # mask design xor runtime
│   │   ├── scalarize.py          # MOEnv → Env via weights or Tchebycheff
│   │   ├── forecast.py           # add rainfall forecast to obs
│   │   └── record_trajectory.py  # writes JSONL consumed by viz + goldens
│   ├── viz/                      # optional [viz] extra — Plotly only, §5.5
│   │   ├── __init__.py           # raises ImportError if plotly missing
│   │   ├── trajectory.py         # Trajectory, TrajectoryRun loaders
│   │   └── figures/
│   │       ├── reward_curves.py
│   │       ├── pareto_front.py
│   │       ├── hypervolume_trace.py
│   │       ├── action_timeseries.py
│   │       ├── network_state_heatmap.py
│   │       ├── flooding_attribution.py
│   │       ├── objective_radar.py
│   │       └── trajectory_replay.py
│   ├── benchmarks/               # 10 contrived MO-RL scenarios — see §6A
│   │   ├── __init__.py
│   │   ├── README.md             # catalogue + objective table
│   │   ├── _common/              # shared helpers (synthetic hyetographs, costs)
│   │   ├── b01_twin_tank/        # storage cascade, store-vs-release
│   │   ├── b02_pump_station/     # pump bank, energy vs. overflow
│   │   ├── b03_detention_sizing/ # CIP-pure, cost vs. peak vs. flooding
│   │   ├── b04_lid_vs_vault/     # distributed LID vs. centralized
│   │   ├── b05_cso_diversion/    # combined sewer overflow control
│   │   ├── b06_first_flush/      # TSS first-flush treatment tradeoff
│   │   ├── b07_cascade_coord/    # 3-tier coordinated RTC
│   │   ├── b08_stochastic_storm/ # uncertain forecast, MO under risk
│   │   ├── b09_pipe_sizing/      # CIP-only continuous, tree network
│   │   └── b10_tidal_outfall/    # tide-coupled boundary, storm-on-tide
│   └── _util/
│       ├── __init__.py
│       ├── validation.py
│       └── formatting.py
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # shared fixtures (per-test copy of INPs via tmp_path)
│   ├── data/                     # tiny hand-authored INPs (≤30 elements each)
│   ├── unit/
│   │   ├── conftest.py
│   │   ├── test_envs_base.py
│   │   ├── test_envs_joint.py
│   │   ├── test_envs_cip_only.py
│   │   ├── test_envs_rtc_only.py
│   │   ├── test_envs_mo.py
│   │   ├── test_spaces_design.py
│   │   ├── test_spaces_runtime.py
│   │   ├── test_spaces_compose.py
│   │   ├── test_observations_builder.py
│   │   ├── test_observations_collectors.py
│   │   ├── test_rewards_terms.py
│   │   ├── test_rewards_composers.py
│   │   ├── test_scoring_hypervolume.py
│   │   ├── test_scoring_igd.py
│   │   ├── test_scoring_epsilon.py
│   │   ├── test_scoring_spread.py
│   │   ├── test_scoring_r2.py
│   │   ├── test_scoring_pareto.py
│   │   ├── test_scoring_normalization.py
│   │   ├── test_wrappers_action_remap.py
│   │   ├── test_wrappers_action_mask.py
│   │   ├── test_wrappers_scalarize.py
│   │   ├── test_wrappers_forecast.py
│   │   └── test_wrappers_record_trajectory.py
│   ├── integration/
│   │   ├── conftest.py           # mirrors MCP --run-integration gate
│   │   ├── test_end_to_end_joint.py
│   │   ├── test_check_env_compliance.py   # gymnasium.utils.env_checker
│   │   └── test_vector_env.py             # AsyncVectorEnv smoke
│   └── regression/
│       ├── conftest.py
│       ├── data/
│       │   └── trajectories/     # JSONL goldens (seed, action stream, reward)
│       ├── test_trajectory_site_drainage.py
│       ├── test_trajectory_east_boston.py
│       └── test_mo_score_stability.py
└── .github/
    └── workflows/
        ├── ci.yml                # ruff + unit tests
        ├── integration.yml       # opt-in integration
        ├── docs.yml              # sphinx build
        └── cla.yml               # CLA Assistant
```

---

## 6A. Bundled Benchmark Scenarios — 10 Contrived MO-RL Models

All scenarios are authored from scratch under the project's MIT license.
They are deliberately small (≤ ~30 elements) so a full episode runs in
under a second on a laptop, making them suitable as RL training targets.
Each scenario is generated by a Python script in
`tools/generate_benchmarks.py` so the `.inp` files are reproducible, not
hand-edited binaries.

Each row shows the scenario's **competing objectives** (always in
conflict — no trivial corner solutions), the **action surface
exercised**, and what the scenario uniquely tests that no other does.
Every scenario ships in three variants: `Cip`, `Rtc`, `Joint`, each with
a `-MO` multi-objective sibling — 60 env IDs total. Scenarios where one
variant is degenerate (e.g. CIP-pure b03 has no `Rtc` variant) ship only
the meaningful subset.

| #   | Scenario ID                | Competing objectives                                                                 | Action surface exercised                                       | Unique stress                                    |
|-----|----------------------------|--------------------------------------------------------------------------------------|----------------------------------------------------------------|--------------------------------------------------|
| b01 | `TwinTank-v0`              | upstream flooding ↓ vs. downstream peak outflow ↓ vs. setpoint smoothness ↓          | RTC: 1 orifice (Box)                                           | Store-vs-release with controller churn penalty   |
| b02 | `PumpStation-v0`           | overflow volume ↓ vs. pump energy ↓ vs. pump switching cycles ↓                      | RTC: 3-pump MultiBinary; CIP: pump curve choice (Discrete)     | Discrete pump combinatorics + duty-cycle wear    |
| b03 | `DetentionSizing-v0`       | capital cost ↓ vs. peak outflow ↓ vs. cumulative flooding ↓                          | CIP only: storage volume (Box) + outlet diameter (Box)         | Pure CIP, 3 objectives, no RTC                   |
| b04 | `LIDvsVault-v0`            | capital cost ↓ vs. flooding ↓ vs. infiltration volume ↑                              | CIP: 5-subarea LID MultiBinary + vault volume (Box)            | Sparse binary CIP + maximize-direction objective |
| b05 | `CSODiversion-v0`          | CSO to river ↓ vs. WWTP hydraulic overload ↓ vs. tank overflow ↓                     | RTC: 2 diverter gates (Box); CIP: tank volume (Box)            | Three-way overflow accounting                    |
| b06 | `FirstFlush-v0`            | TSS load to receiver ↓ vs. capital cost ↓ vs. treatment peak hydraulic load ↓        | CIP: swale area (Box); RTC: bypass valve (Box)                 | Pollutant routing in the reward                  |
| b07 | `CascadeCoord-v0`          | downstream peak flow ↓ vs. cumulative upstream flooding ↓ vs. action churn ↓         | RTC: 3 coordinated orifices (Box × 3)                          | Cross-controller coordination                    |
| b08 | `StochasticStorm-v0`       | flooding ↓ vs. over-release ↓ vs. CVaR(flooding) ↓                                   | RTC: 1 orifice (Box)                                           | Stochastic rainfall episodes, risk-aware reward  |
| b09 | `PipeSizing-v0`            | capital cost ↓ vs. surcharge volume ↓ vs. min-velocity violations ↓                  | CIP only: 6-conduit diameter (Box × 6)                         | Continuous CIP + construction constraints        |
| b10 | `TidalOutfall-v0`          | storm-on-tide flooding ↓ vs. low-tide-window discharge ↓ vs. pump energy ↓           | RTC: 1 release pump (Box); CIP: tank volume (Box)              | Periodic boundary forcing                        |

### Per-scenario design notes

**b01 — TwinTank.** Two storage nodes in series, controllable orifice
between them, fixed outlet to a sensitive receiver. A 2-hr design storm
hits the upstream tank. Tradeoff: open the orifice early (relieves
upstream flooding, spikes the receiver) vs. keep it closed (protects
the receiver, floods upstream). Smoothness term ensures jittery
controllers are penalized.

**b02 — PumpStation.** Wet well with 3 pumps of different head/efficiency
curves (CIP = choose curve type; RTC = on/off per pump). A 6-hr inflow
hydrograph drains through the wet well into a forcemain. Energy
integrates `flow × head / efficiency × dt`. Switching cycles tracked
via Δstate of the binary action — directly tests the
`RewardTerm.accumulator` interface.

**b03 — DetentionSizing.** Pure CIP. One subcatchment → one detention
pond → one outlet pipe → outfall. Episode = one design, no RTC steps.
Reward computed at end-of-episode from output statistics. Classic
3-objective Pareto problem: small pond is cheap but floods and peaks;
big pond is expensive but attenuates well. Used as the unit-test
fixture for `SwmmCIPEnv`.

**b04 — LIDvsVault.** 5 subcatchments. CIP picks (a) which subcatchments
get LIDs (MultiBinary) and (b) the volume of a central vault. LIDs are
cheap-per-area but limited; the vault is expensive-per-volume but
unbounded. Third objective (infiltration ↑) is the only `maximize`
objective in the bundle — exercises the normalization sign-flip
machinery.

**b05 — CSODiversion.** Combined sewer trunk with a diverter weir to a
storage tank, then an interceptor to a WWTP node, then a relief outfall
to a river. Two gates controllable. Three competing overflow paths —
unsatisfiable simultaneously under a stress storm. Exercises three CSO
endpoints and three reward terms simultaneously.

**b06 — FirstFlush.** Synthetic TSS buildup-washoff on a single
subcatchment with a peak-leading pollutograph. CIP sizes a treatment
swale; RTC opens a bypass when the swale would overload. Exercises
`TSSLoad` reward term and `Pollutants` / `Quality` accessors.

**b07 — CascadeCoord.** Three storage tanks arranged 2→1 (two upstream
flow into one downstream). Each has a controllable orifice. Storms
arrive on the two upstream catchments with a phase offset. Optimal
policy requires coordinated holds — naive independent control floods.
Exercises multi-element RTC and the coordination case that motivates
RL over rule-based controllers.

**b08 — StochasticStorm.** Single storage with a controllable release.
At each `reset()`, the rainfall sequence is sampled from a parametric
hyetograph family (depth, duration, peak position drawn from disclosed
distributions). Third objective is **CVaR₉₀(flooding)** computed over
the last `N` episodes — exercises `info["mo_score"]` aggregating across
episodes and tests the stochastic-forcing wrapper.

**b09 — PipeSizing.** Pure CIP. A 6-link branching tree from 3
subcatchments to one outfall. Action vector ∈ ℝ⁶ sets each diameter.
Capital cost is convex in diameter (material curve); surcharge is
non-convex (sharp threshold). Velocity-violation objective is a step
function (zero below threshold, linear above). Tests continuous-Box
CIP with non-smooth Pareto fronts. The intended NSGA-II baseline lives
in `tests/regression/test_b09_nsga_baseline.py` and is driven through
the [Platypus](https://github.com/Project-Platypus/Platypus) adapter
(§5.4) — Platypus is the sole supported non-Gymnasium MOO backend.

**b10 — TidalOutfall.** Single tank discharging through a pump to an
outfall whose stage follows a sinusoidal tide (period 12.42 h via
`OutfallView.set_tidal_curve`). A 4-hr storm arrives at a configurable
phase relative to high tide. Optimal policy times releases for
low-tide windows. Exercises `OutfallView` and the boundary-coupling
case absent from b01–b09.

### Provenance, license, and limits

- All `.inp` files are generated by `tools/generate_benchmarks.py` from
  parameters in `tools/benchmarks/*.yaml`. The YAML + the script are the
  source of truth; the `.inp` files are checked in for reproducibility
  but regeneratable end-to-end.
- All input data (geometry, hyetographs, cost curves, tidal curves) is
  authored from scratch under the project's MIT license. No client
  models, no real-world locations.
- Limits: contrived ≠ realistic. The benchmark suite is designed to
  exercise the framework's mechanics and to provide reproducible
  regression goldens, **not** as a substitute for site-specific
  modeling. The documentation makes this explicit.

---

## 7. Documentation Plan

Mirrors `openswmm.mcp/docs/`:

- **Sphinx** with `pydata-sphinx-theme`, `myst-parser`,
  **`sphinx-epytext`** (for epytext docstring rendering — see §13),
  `autodoc`, `autosummary`, `viewcode`, `intersphinx` (Python, numpy,
  gymnasium). `sphinx.ext.napoleon` is **not** used because napoleon
  parses Google/NumPy styles, neither of which is permitted in this
  package.
- `getting-started/` — install, conda + pip, first env, first multi-obj
  run.
- `user-guide/` — one page per public surface area (`envs`,
  `action_spaces`, `observations`, `rewards`, `scoring`, `wrappers`).
  Each page combines prose + runnable code blocks tested in CI via
  `pytest --doctest-glob='*.md'`.
- `developer/architecture.md` — the diagrams from §2 of this document.
- `developer/testing.md` — explains the unit / integration / regression
  split and the `--run-integration` flag.
- `api/index.rst` — autosummary across `openswmm_gymnasium.*`.

Documentation builds in `docs.yml` CI on every PR.

---

## 8. Test Plan

Modelled on `openswmm.mcp/tests/`.

### 8.0 No engine mocks — ever

**All tests in this package run against the real
`openswmm.engine.Solver`.** There is no `tests/mocks/engine.py`, no
`MockSolver`, no `FakeEngine`. The unit/integration/regression split
is by **test scope** (single component vs. end-to-end vs. golden
trajectory), not by whether the engine is real.

Rationale: the package exists to wrap real solver semantics; mocking
the solver hides the only thing actually worth testing. The engine is
a required dependency by §0 #7, so CI installs it on every job. The
fixtures under `tests/data/` are tiny (≤ 30 elements each) — a full
unit-test simulation typically completes in tens of milliseconds.

### 8.1 Unit tests (`tests/unit/`)

- Drive the **real** `openswmm.engine.Solver` against tiny hand-authored
  `.inp` fixtures under `tests/data/`. Each fixture is the smallest
  network that exercises the component under test (often 1 node +
  1 link + 1 outfall).
- Tests target one component at a time: e.g.
  `test_rewards_terms.py::test_pump_energy` constructs a `Solver`,
  applies a known pump duty cycle, and asserts the `PumpEnergy` term
  integrates to the analytical expected value.
- Coverage target: **≥90% line coverage on `src/openswmm_gymnasium/`**,
  including the `_engine/` adapter.
- Each public class / function has at least one happy-path and one
  error-path test.
- Scoring tests (`test_scoring_*`) do not need the engine at all — they
  test pure math against closed-form Pareto-front cases (DTLZ1, ZDT1
  references) for hypervolume / IGD ground truth. Those files therefore
  don't import `openswmm.engine`, but the rest of `tests/unit/` does.

### 8.2 Integration tests (`tests/integration/`)

- Run on the same CI matrix as unit tests (no `--run-integration` gate);
  the engine is required, so there is no value in optional execution.
  The `OPENSWMM_RUN_INTEGRATION` flag is retained only as a no-op for
  symmetry with `openswmm.mcp` conventions.
- `test_check_env_compliance.py` — runs `gymnasium.utils.env_checker.
  check_env` against every public env class for **API compliance**
  with Farama Gymnasium.
- `test_end_to_end_joint.py` — full episode against the `b01_twin_tank`
  benchmark INP: reset → 100 steps → terminated; asserts reward,
  observation shape, info contents.
- `test_vector_env.py` — both `SyncVectorEnv` (threads) and
  `AsyncVectorEnv` (processes) smoke tests with 2 workers, asserting
  identical trajectories from identical seeds (verifies the §2.3
  thread-isolation guarantee).

### 8.3 Regression tests (`tests/regression/`)

- For each of the **10 contrived benchmark scenarios** (§6A) × env class,
  a JSONL golden file records `(seed, action_sequence, observation_hash,
  reward_sequence, mo_score)`.
- The regression suite re-runs the env with the recorded seed + actions
  and asserts that the cumulative reward and final `mo_score` match the
  golden to within a documented tolerance (`atol=1e-6` for scalar
  rewards, `atol=1e-4` for `mo_score`).
- Goldens are regenerated explicitly via
  `pytest tests/regression --update-goldens` and reviewed in the PR.
- This is the long-term safety net for engine upgrades and refactors —
  any silent drift in solver behavior or env composition will show up
  here.

### 8.4 Transparent file IO (per `CLAUDE.md §4.1`)

All test outputs (working `.inp`, `.rpt`, `.out`, golden JSONL, profile
dumps) land under `tests/_artifacts/<test-name>/` rather than
`tmp_path`, and `tests/_artifacts/` is `.gitignore`d but never deleted
automatically — the user can inspect any test's artifacts after a run.

---

## 9. Dependencies

Declared in `pyproject.toml` under `[project]` and `[project.optional-dependencies]`:

```
[project.dependencies]
gymnasium >= 1.0
numpy >= 1.21
pydantic >= 2.0
# openswmm.engine v6 (handle-based, thread-safe). The legacy v5 singleton
# solver under openswmm.legacy.engine is NOT a fallback — see §0 #7 and §2.3.
openswmm >= 6.0.0a2

[project.optional-dependencies]
mo       = ["mo-gymnasium >= 1.0"]     # multi-objective extension
platypus = ["platypus-opt >= 1.2"]     # sole supported MOO adapter (§5.4)
viz      = ["plotly >= 5.18"]          # sole supported plot backend (§5.5)
dev    = ["pytest>=7.0", "pytest-cov", "ruff==0.15.14"]
docs   = ["sphinx>=7.0", "pydata-sphinx-theme>=0.15",
          "myst-parser>=3.0", "sphinx-autodoc-typehints",
          "sphinx-epytext>=0.0.4"]   # epytext rendering — §13
```

No `torch` / `jax` / training-framework deps. The package ships
environments, not agents.

---

## 10. Phased Milestones

Each phase has explicit, verifiable success criteria per `CLAUDE.md §4`.

### P0 — Project skeleton (1 day)

1. `pyproject.toml` (hatchling, mirroring MCP; `openswmm >= 6.0.0a2`
   as a **required** dependency per §0 #7 — not optional) →
   **verify:** `pip install -e ".[dev,docs]"` succeeds in a clean venv
   and `python -c "from openswmm.engine import Solver"` works.
2. Package skeleton (empty modules with docstrings; a
   `_engine/solver_adapter.py` stub that explicitly rejects
   `openswmm.legacy.engine.Solver` instances) →
   **verify:** `python -c "import openswmm_gymnasium"` works; a unit
   test confirms the legacy-solver guard raises.
3. Ruff + pytest configured → **verify:** `ruff check src/ tests/` and
   `pytest tests/unit/` both green on an empty suite.
4. Sphinx skeleton → **verify:** `sphinx-build -b html docs
   docs/_build/html` succeeds.
5. CLA Assistant workflow + `LICENSE` / `CLA.md` / `AUTHORS.md` /
   `CODE_OF_CONDUCT.md` / `CONTRIBUTING.md` / `SECURITY.md` →
   **verify:** files present, links resolve, CLA workflow valid YAML.

### P1 — Solver adapter + minimal SwmmRTCEnv (3 days)

1. `_engine/solver_adapter.py` opens, initializes, steps, and closes a
   real `openswmm.engine.Solver` against a tiny in-tree INP. → **verify:**
   unit test runs a full simulation through the adapter against the
   real engine (no mocks per §8.0) using `tests/data/minimal.inp`
   (1 junction + 1 outfall + 1 conduit).
2. `SwmmRTCEnv` with a one-link orifice action and one-node depth
   observation. → **verify:** `gymnasium.utils.env_checker.check_env`
   passes; a `for _ in range(200): step(env.action_space.sample())`
   loop terminates cleanly.
3. `FloodingVolume` reward term. → **verify:** reward = 0 on a
   dry-weather episode; reward > 0 when flooding is injected via
   `Inflows`.

### P2 — Observation builder + RewardRegistry (2 days)

1. All observation collectors in §4 implemented and unit-tested.
2. All reward terms in §5.1 implemented and unit-tested.
3. `SwmmRTCEnv` accepts both via the public constructor.
   **verify:** end-to-end integration test exercises five reward terms
   and ten observation features.

### P3 — Design (CIP) action space + SwmmCIPEnv + SwmmJointCIPRTCEnv (4 days)

1. Each CIP factory in §3.1 implemented and unit-tested against a real
   `Solver` on the smallest INP that includes the relevant element type.
2. `ModelEditor` integration applies design action and re-opens the
   solver. → **verify:** integration test confirms `.inp` mutation
   round-trips through the engine and the resulting simulation differs
   from baseline.
3. `SwmmJointCIPRTCEnv` end-to-end → **verify:** `check_env` passes;
   episode terminates with both reward components non-zero.

### P4 — Multi-objective env + scoring (3 days)

1. `SwmmMOEnv` returning vector reward (or graceful import skip when
   `mo-gymnasium` missing).
2. Scoring module: `hypervolume`, `igd`, `igd_plus`, `epsilon`, `spread`,
   `r2`, `pareto_front`, `normalization`.
3. Closed-form unit tests against ZDT1 / DTLZ1 references.
4. `info["mo_score"]` populated at episode end → **verify:** integration
   test logs a small Pareto front over 20 episodes and confirms
   normalized HV is in `[0, 1]` and monotone w.r.t. trivially-dominated
   points.

### P5 — Wrappers (2 days)

`action_remap`, `action_mask`, `scalarize`, `forecast`,
`record_trajectory` implemented and unit-tested.

### P6 — Bundled benchmarks + regression goldens (5 days)

1. Author the 10 contrived `.inp` files in §6A from scratch
   programmatically via `ModelEditor` + a `tools/generate_benchmarks.py`
   script (so the inputs themselves are version-controlled, regeneratable,
   and reviewable).
2. One benchmark module per scenario under
   `src/openswmm_gymnasium/benchmarks/b0N_*/` exposing
   `make_env(variant="cip"|"rtc"|"joint", mo=False)`.
3. Each scenario `register()`s its env IDs at import time
   (e.g. `OpenSWMM/TwinTank-Joint-MO-v0`).
4. JSONL trajectory goldens generated for each `(scenario, env_class,
   seed)` combination and committed.
5. Regression test suite green.
6. **verify:** `python -m openswmm_gymnasium.benchmarks --list` prints
   30 env IDs (10 scenarios × 3 variants) and `gymnasium.make(id)`
   succeeds for each.

### P6.5 — Visualization module (2 days)

1. `viz/trajectory.py` — `Trajectory` and `TrajectoryRun` JSONL loaders,
   numpy-only aggregation. → **verify:** unit test loads a recorded
   trajectory and exposes reward / action / observation arrays of the
   expected shapes.
2. Each of the 8 figure factories in §5.5 implemented and unit-tested. →
   **verify:** for each, the returned `Figure` has the asserted number
   of traces and axes; the snapshot test renders the figure to
   `tests/_artifacts/viz/<name>.html` for human review (per `CLAUDE.md
   §4.1`).
3. Optional-extra guard — `from openswmm_gymnasium.viz import …`
   raises a clean `ImportError` with the `pip install
   openswmm-gymnasium[viz]` instruction when Plotly is missing. →
   **verify:** unit test runs in a venv without Plotly and asserts the
   error message.

### P7 — Documentation (3 days)

All Sphinx pages written and CI green. Public docstrings reviewed for
Sphinx warnings → zero warnings on `sphinx-build -W`.

### P8 — Release prep (1 day)

1. `CHANGELOG.md` first entry.
2. PyPI metadata audit.
3. Tag `v0.1.0` and publish.

Total: **~24 working days** for v0.1.0 (P6.5 adds 2 days).

---

## 11. Risks & Mitigations

| Risk                                                         | Mitigation                                                                                |
|--------------------------------------------------------------|-------------------------------------------------------------------------------------------|
| Gymnasium 1.x → 2.x API change before v0.1.0 ships           | Pin `gymnasium >= 1.0, < 2.0`; track Farama release notes; CI matrix tests latest 1.x.    |
| MO-Gymnasium drift                                           | Soft import; subclass only what's needed; regression tests pin behavior.                  |
| Engine ABI change in `openswmm.engine`                       | Adapter layer (`_engine/`) is the only surface that touches the engine; one file to fix.  |
| Thread-safety regression in `openswmm.engine`                | Integration suite (§8.2) includes a `SyncVectorEnv` thread-race test that asserts handle isolation; CI fails closed if engine releases a thread-unsafe build. |
| Users on a platform without an `openswmm` wheel              | Engine is a required dep (§0 #7); docs explicitly list supported platforms and direct users to build-from-source instructions in `openswmm.engine`. |
| Hypervolume cost explosion at high-d                         | Exact ≤4D, Monte-Carlo ≥5D with documented error bounds + seed surface.                   |
| Regression goldens become flaky across SWMM solver patches   | Goldens versioned with an `engine_version` field; `--update-goldens` flow is explicit.    |
| CIP action design exceeds what `ModelEditor` supports        | §3.1 factories require `ModelEditor` coverage; add engine-side coverage tests first.      |
| Users expect built-in training algorithms                    | README + docs are explicit that this ships envs only; point to MORL-Baselines, SB3.       |
| Contrived benchmarks (§6A) don't generalize to client work   | Docs are explicit that the suite tests framework mechanics, not site engineering.         |

---

## 12. Alignment with project-wide conventions

- `CLAUDE.md §1` — assumptions explicit in §0.
- `CLAUDE.md §2` — design avoids speculative abstraction; everything in
  §3–§5 has a concrete first user inside the package.
- `CLAUDE.md §3` — surgical changes; no edits to `openswmm.engine` or
  `openswmm.mcp` are required.
- `CLAUDE.md §4` — every phase in §10 has a `verify:` clause.
- `CLAUDE.md §4.1` — test IO lives under reviewable
  `tests/_artifacts/`.
- `CLAUDE.md §5.0` — this document is the preconfigured plan; future
  work updates it.
- `CLAUDE.md §5.1` (MVC) — `SwmmEnv` is the controller, `_engine/`
  adapters are the model, observation/reward/scoring are the view. UI
  shells (e.g. an HTML training dashboard) belong in `openswmm.gui` and
  consume the same underlying state.
- `openswmm.engine/CLAUDE.md §4.01` (alignment with legacy) — the
  **engine** project preserves the legacy v5 solver as a regression
  baseline. `openswmm.gymnasium` consumes the **refactored v6
  handle-based solver only** and does not need to maintain parity with
  the v5 singleton path; alignment with legacy is the engine's concern,
  not this package's.

---

## 13. Documentation Convention — Epytext

All public Python docstrings in this package use **epytext** convention,
matching the `openswmm.engine` `.pyi` stubs. Google-style, NumPy-style,
and reST-field-list (`:author:` / `:param:`) docstrings are **not**
permitted in source files; a PR introducing any of those styles will be
asked to convert.

**Required tags:**

| Use                  | Epytext form        |
|----------------------|---------------------|
| Function parameter   | `@param name:`      |
| Parameter type       | `@type name:`       |
| Return value         | `@return:`          |
| Return type          | `@rtype:`           |
| Raises               | `@raise Exc:`       |
| Instance variable    | `@ivar name:`       |
| Class variable       | `@cvar name:`       |
| Module/file author   | `@author:`          |
| Module/file copyright| `@copyright:`       |
| Module/file license  | `@license:`         |
| Note                 | `@note:`            |
| See also             | `@see:`             |
| Warning              | `@warning:`         |
| Deprecated           | `@deprecated:`      |

**Inline markup:** `C{code}`, `B{bold}`, `I{italic}`, `L{module.Class}`,
`U{https://url}`.

**Sphinx rendering:** via the `sphinx-epytext` extension (PyPI package
`sphinx-epytext`). Added to the `docs` optional-extra in §9.

**Carve-outs:**

- Private helpers (leading underscore) may use a brief one-liner without
  tags.
- Test functions use plain English docstrings, no tag overhead.

**Linting:** Ruff does not enforce docstring style. Convention is
maintained in code review and via the `developer/contributing.md`
checklist (§7).

---

## 14. References

- Farama Gymnasium API: <https://gymnasium.farama.org/api/env/>
- Gymnasium 1.0 release notes: <https://github.com/Farama-Foundation/Gymnasium/releases/tag/v1.0.0>
- Terminated vs. truncated step API: <https://farama.org/Gymnasium-Terminated-Truncated-Step-API>
- MO-Gymnasium: <https://mo-gymnasium.farama.org/>
- MO-Gym paper (BNAIC 2022): <https://people.cs.umass.edu/~bsilva/papers/MO-Gym_BNAIC_2022.pdf>
- openswmm.engine Python API: `python/openswmm/engine/__init__.pyi`
- openswmm.mcp test/doc layout: `openswmm.mcp/tests/`, `openswmm.mcp/docs/`
