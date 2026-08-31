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
  - `Links.set_target_setting` — the persistent runtime-control override.
    (`Controls.set_link_setting` writes `control_setting`, which the engine
    recomputes from the target every routing step, so it would not stick
    without a control rule.)
* - {py:class}`~openswmm_gymnasium.spaces.runtime.NodeLateralInflow`
  - `Box([0, max_inflow]^n)`
  - `Nodes.set_lateral_inflow` — controllable lateral inflow in project flow
    units, for pumped diversions, controllable sources, or adversarial
    inflow scenarios.
* - {py:class}`~openswmm_gymnasium.spaces.runtime.HeatSourceTemperatureSetpoint`
  - `Box([low, high]^n)`, degC
  - `Heat.sources[...]` — per-pathway inlet temperature. Heat source writes
    are **live** (they take effect on the next routing step), which is what
    makes a heat configuration call a legitimate runtime actuator. Requires
    `[OPTIONS] HEAT_TRANSPORT YES`; bounds default to and may not exceed the
    engine's own `[-50, 100]` refusal range. **Open-loop**: the C API has no
    temperature getter, so the agent cannot observe what it is acting on.
```

There is deliberately **no water-age runtime twin**. Water-age source writes
are live in exactly the same way, so one would be mechanically trivial — but
it would not be a meaningful control. Heat has a physical actuator behind it
(an effluent whose temperature a plant genuinely regulates, and which
genuinely alters downstream water temperature at step resolution). A source's
assigned water age is a bookkeeping label on inflowing water, not a quantity
any operator can move during an event; per-step modulation of it would let an
agent chase reward by rewriting its own accounting rather than by operating
the network. Water age is exposed as a once-per-episode design/scenario input
only.

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
  - `Links.set_xsect` — resizes the section to a target **rise** (full depth,
    read from the engine's `XSectionGeometry`), scaling every
    length-dimensioned geometry parameter of the shape by the same factor so
    the section stays geometrically similar. For a `CIRCULAR` pipe the rise is
    the diameter; for a box culvert the width scales with the height.
    Dimensionless parameters (trapezoid side slopes, the `POWER` exponent) and
    table references are preserved. `IRREGULAR` / `STREET_XSECT` / `DUMMY`
    sections have no scalable dimension and are rejected at `bind`.
* - {py:class}`~openswmm_gymnasium.spaces.design.NodeMaxDepth`
  - `Box`
  - `Nodes.set_max_depth`
* - {py:class}`~openswmm_gymnasium.spaces.design.SubcatchGWOutflowCoeff`
  - `Box`
  - `Subcatchment.set_gw_params` (preserves other params, rewrites `a1`)
* - {py:class}`~openswmm_gymnasium.spaces.design.StorageVolume`
  - `Box`
  - `Nodes.set_storage_functional` / `Tables` curve points — sizes both
    FUNCTIONAL and TABULAR storage, resolved per node at `bind`.
    `mode="scalar"` applies a per-node footprint multiplier: for FUNCTIONAL it
    scales the `(a, c)` coefficients (volume scales linearly, exponent
    preserved), for TABULAR it scales the depth–area curve's areas and leaves
    its depths alone. `mode="coeffs"` searches the raw `(a, b, c)` triple and
    is FUNCTIONAL-only. Geometric shapes (`CYLINDRICAL`, `CONICAL`,
    `PARABOLOID`, `PYRAMIDAL`) are rejected at `bind`. A TABULAR node's curve
    is rewritten in place in the open model, so give each searchable basin its
    own curve.
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
* - {py:class}`~openswmm_gymnasium.spaces.design.ReactionCoefficientValue`
  - `Box` (scalar or per-coefficient bounds)
  - `Reactions.coefficients[...].value` — searches `[REACTION_COEFFICIENTS]`
    **PARAMETER** values. CONSTANT coefficients are refused at `bind`: the
    model declares them fixed, so writing them would change values it was
    never meant to vary. Bounds are in the model's own expression units;
    nothing is converted.
* - {py:class}`~openswmm_gymnasium.spaces.design.HeatSourceTemperature`
  - `Box([low, high]^n)`, degC (default `[-50, 100]`)
  - `Heat.sources[...]` — global inlet temperature per heat-source pathway,
    as a fixed thermal boundary condition per episode. Requires
    `[OPTIONS] HEAT_TRANSPORT YES`. Bounds may not exceed the engine's own
    refusal range.
* - {py:class}`~openswmm_gymnasium.spaces.design.WaterAgeSourceAge`
  - `Box([low, high]^n)`, hours
  - `WaterAge.globals[...]` — global source age per water-age pathway.
    **Negative values are legal and meaningful** (a negative source age
    extracts age-volume; the engine clamps the *result* at zero, not the
    input), so neither bound is floored at zero and `low` has no default.
    Requires `[OPTIONS] WATER_AGE YES`.
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

### Process-configuration factories

The last three factories differ in kind from the rest of the table. They do
not resize an asset; they set the **boundary conditions and rate constants of
a transport process**, which makes them calibration / inverse-problem handles
at least as much as CIP levers.

`ReactionCoefficientValue` is the high-value one. Reaction coefficients are
exactly the quantities a water-quality modeller normally fits by hand against
observed concentrations, so exposing them as a `Box` turns calibration into an
ordinary optimisation over the same env machinery that does CIP sizing: pair
the factory with an observed-vs-simulated objective and the search *is* a
calibration run; pair it with a treatment-performance objective and it is a
process-design run instead.

All three are **optional surfaces** — they need engine modules a partial build
may not carry, and (for heat and water age) an `[OPTIONS]` flag the model may
not set. `SolverAdapter.heat` / `.water_age` / `.reactions` therefore apply a
two-tier guard: *does this build carry the module*, then *does the open model
actually enable it*. Both tiers raise with the remedy named rather than
returning defaults, because a heat or water-age configuration written into a
model that never routes it is a **silent** no-op — stored, never transported,
nothing raised, no reward signal ever moving.

Two further engine surfaces are reachable but deliberately not wrapped:
`InitialQuality` (per-element initial temperature / age / species rows — an
initial-condition randomiser, so it belongs with hot-start seeding rather than
here) and `ProcessComponents` (registration of external process plugins by
config path — a model-assembly concern with no continuous search space).

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
