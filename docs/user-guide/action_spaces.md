# Action spaces

Per the plan §3 contract, every env exposes
`Dict({"design": Dict({...}), "runtime": Dict({...})})`. Either half may
be empty (e.g. for `SwmmRTCEnv` the design Dict is empty).

## Runtime (RTC) factories

Applied **every step** via the appropriate engine setter.

```{list-table}
:header-rows: 1

* - Factory
  - Space
  - Engine surface
* - {py:class}`~openswmm_gymnasium.spaces.runtime.OrificeSetting`
  - `Box([0, 1]^n)`
  - `Controls.set_link_setting`
```

## Design (CIP) factories

Applied **once per episode**, between
`SolverAdapter.open()` and `SolverAdapter.initialize()`.

```{list-table}
:header-rows: 1

* - Factory
  - Space
  - Engine surface
* - {py:class}`~openswmm_gymnasium.spaces.design.LinkRoughness`
  - `Box([low, high]^n)`
  - `Links.set_roughness`
* - {py:class}`~openswmm_gymnasium.spaces.design.LinkLength`
  - `Box`
  - `Links.set_length`
* - {py:class}`~openswmm_gymnasium.spaces.design.LinkDiameter`
  - `Box`
  - `Links.set_xsect` (preserves shape, rewrites `geom1`)
* - {py:class}`~openswmm_gymnasium.spaces.design.NodeMaxDepth`
  - `Box`
  - `Nodes.set_max_depth`
* - {py:class}`~openswmm_gymnasium.spaces.design.SubcatchGWOutflowCoeff`
  - `Box`
  - `Subcatchment.set_gw_params` (preserves other params, rewrites `a1`)
* - {py:class}`~openswmm_gymnasium.spaces.design.StorageVolume`
  - `Box`
  - `Nodes.set_storage_functional` — sizes FUNCTIONAL storage. `mode="scalar"`
    scales the `(a, c)` coefficients by a per-node footprint multiplier
    (volume scales linearly, exponent preserved); `mode="coeffs"` searches the
    raw `(a, b, c)` triple.
* - {py:class}`~openswmm_gymnasium.spaces.design.LIDPlacement`
  - `Box` (subcatchment-major `[type, area]`)
  - `Infrastructure.lid_usage_add` — sizes green-infrastructure / nature-based
    solutions and selects among a model-defined palette of LID control types
    per subcatchment.
* - {py:class}`~openswmm_gymnasium.spaces.design.RDIIUnitHydrograph`
  - `Box` (`[R]`, or `[R, dmax, drecov, dinit]` when `include_ia=True`)
  - `Inflows.set_hydrograph_rtk` / `set_hydrograph_ia` — sizes RDII response by
    editing the R fraction (and optionally initial abstraction) of existing
    unit-hydrograph entries, preserving T and K.
```

`StorageVolume`, `LIDPlacement`, and `RDIIUnitHydrograph` size the assets
modelers most often want to explore exhaustively — detention/retention
storage, green infrastructure, and inflow/infiltration response. Because each
is a plain `Box`, the NSGA-II optimizer searches them jointly with every other
design dimension, with no optimizer changes.

`LIDPlacement` selects among LID controls **already defined in the model** (the
modeler's candidate GI palette), so it never invents process layers; the target
subcatchments should not already carry an LID usage for the chosen control.
`RDIIUnitHydrograph` targets unit-hydrograph groups that already exist in the
`[HYDROGRAPHS]` section.

Additional factories ship as benchmark scenarios that need them come
online.

## Policy factories (searchable static control policies)

A **policy factory** contributes static dimensions to the searchable
decision vector — exactly like a design factory — but instead of mutating
the model geometry it installs a **closed-loop controller** evaluated each
control step through {py:class}`~openswmm_gymnasium.envs.SwmmControlEnv`.
The decision vector is fixed per episode (the controller is reactive at
runtime, its parameters static), so the optimizer treats it identically to
a design vector and traces the operational cost curve (e.g. CSO vs.
flooding volume).

```{list-table}
:header-rows: 1

* - Policy space
  - Decision vector
  - Controller
* - {py:class}`~openswmm_gymnasium.spaces.control_curve.ControlCurvePolicySpace`
  - `Box`: the `[y_low, y_high]` setting at each **fixed** `x_knot` of each
    asset, asset-major; labelled `control_curve/<link_id>/y[<k>]`
  - {py:class}`~openswmm_gymnasium.control.ControlCurveController` (reactive
    piecewise-linear breakpoint curve)
```

Each `control_curve` asset maps one observed node's state (raw, or a
fraction of full depth when `x_normalized`) through a piecewise-linear
curve to a controlled link's `[0, 1]` setting via `Links.set_target_setting`.
A `monotonic` constraint (`nonincreasing` / `nondecreasing`) is enforced by
deterministic, idempotent projection at decode time, so MOEAs that ignore
constraints still produce feasible curves. The neutral curve (all settings
at `y_high`, the default `y_init`) reproduces the uncontrolled baseline for
passive-open structures: a flat `y = 1.0` curve is state-independent and
equivalent to never touching the link. Optional `rate_limit` caps
`|Δsetting|` per control step; `control_interval_steps` recomputes the curve
every N control steps.

## Defensive clipping

All factories `np.clip` the sampled value into bounds before calling
the engine, so an agent that overshoots its action space doesn't crash
the simulator.
