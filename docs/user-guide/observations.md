# Observations

Construct a flat `Box` observation by chaining collectors on an
{py:class}`~openswmm_gymnasium.observations.ObservationBuilder`.

```python
from openswmm_gymnasium.observations import ObservationBuilder

obs = (
    ObservationBuilder()
    .add_node_depths(["J1", "J2"])
    .add_node_heads(["J1"])
    .add_node_inflows(["J1"])
    .add_node_overflows(["J1"])
    .add_link_flows(["C1"])
    .add_link_depths(["C1"])
    .add_link_settings(["ORIF"])
    .add_subcatch_runoff(["S1"])
    .add_rainfall(["RainGage"])
    .add_clock()  # hour_sin, hour_cos, elapsed_frac
)
```

`obs.space()` returns the resulting
`Box(low=-inf, high=+inf, shape=(N,), dtype=float32)`.

## Available collectors

```{list-table}
:header-rows: 1

* - Builder method
  - Engine surface
  - Notes
* - `add_node_depths`
  - `Nodes.get_depth`
  - Instantaneous water depth
* - `add_node_heads`
  - `Nodes.get_head`
  - Hydraulic head
* - `add_node_inflows`
  - `Nodes.get_inflow`
  - Total inflow rate
* - `add_node_overflows`
  - `Nodes.get_overflow`
  - Flooding rate
* - `add_node_volumes`
  - `Nodes.get_volume`
  - Stored volume at the node
* - `add_node_lateral_inflows`
  - `Nodes.get_lateral_inflow`
  - Externally-applied lateral inflow — the read-back of what
    {py:class}`~openswmm_gymnasium.spaces.runtime.NodeLateralInflow` writes
* - `add_link_flows`
  - `Links.get_flow`
  - Instantaneous flow
* - `add_link_depths`
  - `Links.get_depth`
  - Instantaneous depth in link
* - `add_link_settings`
  - `Links.get_control_setting`
  - Current setting in `[0, 1]`
* - `add_link_velocities`
  - `Links.get_velocity`
  - Instantaneous flow velocity
* - `add_link_capacities`
  - `Links.get_capacity`
  - Fraction of full flow capacity in use
* - `add_link_volumes`
  - `Links.get_volume`
  - Stored volume in the link
* - `add_subcatch_runoff`
  - `Subcatchments.get_runoff`
  - Subcatchment runoff rate
* - `add_subcatch_groundwater`
  - `Subcatchment.groundwater`
  - Subcatchment groundwater (baseflow) outflow rate
* - `add_pollutant_concentration`
  - `Node.quality`
  - Per-node concentration of one pollutant, in its declared concentration
    units (`mg/L`, `ug/L`, `#/L`). Add one collector per pollutant.
* - `add_link_pollutant_concentration`
  - `Link.quality`
  - Per-link concentration of one pollutant
* - `add_rainfall`
  - `Gages.get_rainfall`
  - Per-gage rainfall intensity
* - `add_2d_vertex_depths`
  - `Surface2D.get_vertex_render_depths`
  - Signed inundation depth (`eta_v - z_v`, m; negative = dry freeboard) at
    2D mesh vertices, via one bulk read. Requires an engine built with the
    2D module **and** a model with an active 2D surface; binding fails with
    a clear error otherwise, including under the `IGNORE_2D` gate.
* - `add_clock`
  - n/a
  - 3 features: `hour_sin`, `hour_cos`, `elapsed_frac`
```

### There is no heat, water-age or reaction collector

Deliberately, and not an omission: the engine's C API exposes **no**
per-node or per-link water-temperature, water-age or species-concentration
getter. Those three modules are configuration surfaces, so they appear in
this package as [action factories](action_spaces.md) and nowhere here. The
only heat quantities readable at all are `SolverAdapter.heat.current_shortwave`
and `.current_cloud_fraction`, and both are *forcing* rather than state.

The practical consequence is that
{py:class}`~openswmm_gymnasium.spaces.runtime.HeatSourceTemperatureSetpoint`
is an **open-loop** actuator: an agent can drive an inlet temperature but
cannot observe the temperature field it produces. Close the loop through a
proxy (flow split, storage depth) or treat it as feed-forward.

## Reaching collectors from a declarative env config

Every collector above is addressable from an `openswmm.mcp` `EnvConfig`
through an `observations` field of the same name — `node_depths`,
`link_velocities`, `vertex_depths_2d`, and so on. The three
extra-argument collectors take a shape other than a plain ID list:

```json
{
  "observations": {
    "node_depths": ["J1", "J2"],
    "node_pollutant_concentration": {"TSS": ["J1"], "BOD": ["J1", "J2"]},
    "link_pollutant_concentration": {"TSS": ["C1"]},
    "vertex_depths_2d": [0, 1, 2],
    "include_clock": true
  }
}
```

The pollutant fields map a pollutant ID to the elements whose concentration
of it to observe, so several pollutants can be requested at once; they are
applied in insertion order, which keeps the observation vector's layout
reproducible from the config JSON.

## Forecast injection

For lookahead features, wrap the env with
{py:class}`~openswmm_gymnasium.wrappers.ForecastObservation`:

```python
from openswmm_gymnasium.wrappers import ForecastObservation
import numpy as np

def perfect_rainfall_lookahead(env, info):
    elapsed_days = info.get("elapsed_days", 0.0)
    # ...look up next H samples of the timeseries...
    return np.array([...], dtype=np.float32)

env = ForecastObservation(env, perfect_rainfall_lookahead, horizon=12)
```
