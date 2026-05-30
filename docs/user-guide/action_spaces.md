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
```

Additional factories ship as benchmark scenarios that need them come
online.

## Defensive clipping

All factories `np.clip` the sampled value into bounds before calling
the engine, so an agent that overshoots its action space doesn't crash
the simulator.
