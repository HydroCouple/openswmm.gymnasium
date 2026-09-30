# Environments

The package ships five Gymnasium env classes. The action-space envs expose
a `Dict` with the design and runtime halves they use (see
{doc}`/user-guide/action_spaces`); `SwmmControlEnv` takes a policy-parameter
vector.

## `SwmmRTCEnv` — runtime-only

Drives one or more runtime factories every `step()`. Its action space is
`Dict({"runtime": ...})`. Standard `gymnasium.Env`; returns `(obs, reward, terminated,
truncated, info)`.

See {py:class}`openswmm_gymnasium.envs.SwmmRTCEnv`.

## `SwmmCIPEnv` — design-only

Single-step contextual-bandit pattern. The agent picks a design action;
the env simulates the whole episode under that design and returns the
cumulative cost as the terminal reward. Ideal for plugging into
NSGA-II / NSGA-III via the Platypus adapter.

See {py:class}`openswmm_gymnasium.envs.SwmmCIPEnv`.

## `SwmmJointCIPRTCEnv` — full hybrid

Design action sampled at `reset()` (or supplied via
`options["design_action"]`) and locked for the episode. Runtime control
each `step()`.

See {py:class}`openswmm_gymnasium.envs.SwmmJointCIPRTCEnv`.

## `SwmmMORTCEnv` — multi-objective

Returns vector reward (one component per `RewardTerm`) with per-term
sign-flipping so higher is better. Populates `info["mo_score"]` at
episode termination with the single-point normalised hypervolume of
the cumulative-cost vector.

See {py:class}`openswmm_gymnasium.envs.SwmmMORTCEnv`.

## `SwmmControlEnv` — policy-parameter search

A single-step env whose **action is a controller policy-parameter vector**,
not a per-step action or a model design. On `step()` it decodes the vector
into a controller, runs the entire simulation under that controller, and
returns the operational objective totals in `info["reward_components"]`. The
optimizer searches the vector exactly as it searches a design vector — one
decision vector per episode — which is how a reactive control policy's static
parameters get tuned to trace an operational cost curve.

It is controller-agnostic, driving any
{py:class}`~openswmm_gymnasium.control.base.Controller`:

- {py:class}`~openswmm_gymnasium.control.MarketController` — reactive
  agent-based capacity market (cost curves + PID).
- {py:class}`~openswmm_gymnasium.control.ControlCurveController` — reactive
  piecewise-linear breakpoint curves
  ({py:class}`~openswmm_gymnasium.spaces.control_curve.ControlCurvePolicySpace`);
  see {doc}`/user-guide/action_spaces`.
- {py:class}`~openswmm_gymnasium.control.ScheduleController` — open-loop
  per-structure setting schedule.

See {py:class}`openswmm_gymnasium.envs.SwmmControlEnv`.

## Engine requirements

Each observation collector, reward term and action factory declares the
engine catalog paths it reads or writes in a `requires` attribute, and every
env checks them against the installed engine when it is constructed:

```python
from openswmm_gymnasium._engine import CORE_REQUIREMENTS, require_for

require_for(builder, *factories, *reward_terms)  # what the envs do
```

A build without an optional module (2D, heat, water age, reactions, tables)
therefore fails only the envs configured to use it, with an
`EngineCapabilityError` naming every missing path, instead of failing every
env at import or deep inside a rollout. `CORE_REQUIREMENTS` lists what every
env needs. A custom component takes part by setting `requires` to a tuple of
catalog paths.

Whether the open *model* enables a module (`[OPTIONS] HEAT_TRANSPORT`,
`WATER_AGE`, an active 2D surface) is a separate check made when the
component binds, because the catalog cannot know it.

## Registered IDs

```{list-table}
:header-rows: 1

* - Gymnasium ID
  - Class
  - Purpose
* - `OpenSWMM/Minimal-RTC-v0`
  - `SwmmRTCEnv`
  - Framework test env over `tests/data/minimal.inp`
* - `OpenSWMM/Minimal-CIP-v0`
  - `SwmmCIPEnv`
  - As above; CIP variant
* - `OpenSWMM/Minimal-Joint-v0`
  - `SwmmJointCIPRTCEnv`
  - As above; joint variant
* - `OpenSWMM/Minimal-MORTC-v0`
  - `SwmmMORTCEnv`
  - As above; MO variant
* - `OpenSWMM/TwinTank-RTC-v0`
  - `SwmmRTCEnv`
  - Bundled benchmark b01
* - `OpenSWMM/TwinTank-Joint-v0`
  - `SwmmJointCIPRTCEnv`
  - As above; joint variant
* - `OpenSWMM/TwinTank-MORTC-v0`
  - `SwmmMORTCEnv`
  - As above; MO variant
```
