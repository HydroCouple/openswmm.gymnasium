# Observations

Construct a flat `Box` observation by chaining collectors on an
{py:class}`~openswmm_gymnasium.observations.ObservationBuilder`.

```python
from openswmm_gymnasium.observations import ObservationBuilder

obs = (
    ObservationBuilder()
    .add_node_depths(["J1", "J2"])
    .add_link_flows(["C1"])
    .add_field("link.stats.max_filling", ["C1"])  # any numeric engine field
    .add_rainfall(["RainGage"])
    .add_clock()  # hour_sin, hour_cos, elapsed_frac
)
```

`obs.space()` returns the resulting
`Box(low=-inf, high=+inf, shape=(N,), dtype=float32)`.

## Any engine field: `add_field`

`add_field(path, ids)` observes any numeric field of an element kind by its
path in the engine catalog ({py:mod}`openswmm.engine.catalog`): live state
(`"node.depth"`, `"link.flow"`), per-element run statistics
(`"link.stats.max_filling"`, `"link.stats.slot_share"`,
`"node.stats.max_depth"`) and subtype fields (`"node.storage.volume"`).
An unknown path, a non-numeric field or a path that is not an element field
fails at construction with the path named. List the fields of a kind with the
catalog itself:

```python
from openswmm.engine import catalog

[m["name"] for m in catalog.members("link") if m["form"] == "property"]
catalog.lookup("link.stats.slot_share")  # type, access, units, bulk path
```

The named collectors below are aliases for the field paths in the second
column, so `add_node_depths(ids)` and `add_field("node.depth", ids)` produce
the same feature.

## Available collectors

```{list-table}
:header-rows: 1

* - Builder method
  - Field path or engine surface
  - Notes
* - `add_node_depths`
  - `node.depth`
  - Instantaneous water depth
* - `add_node_heads`
  - `node.head`
  - Hydraulic head
* - `add_node_inflows`
  - `node.inflow`
  - Total inflow rate
* - `add_node_overflows`
  - `node.overflow`
  - Flooding rate
* - `add_node_volumes`
  - `node.volume`
  - Stored volume at the node
* - `add_node_lateral_inflows`
  - `node.lateral_inflow`
  - Externally-applied lateral inflow — the read-back of what
    {py:class}`~openswmm_gymnasium.spaces.runtime.NodeLateralInflow` writes
* - `add_link_flows`
  - `link.flow`
  - Instantaneous flow
* - `add_link_depths`
  - `link.depth`
  - Instantaneous depth in link
* - `add_link_settings`
  - `link.control_setting`
  - Current setting in `[0, 1]`
* - `add_link_velocities`
  - `link.velocity`
  - Instantaneous flow velocity
* - `add_link_capacities`
  - `link.capacity`
  - Fraction of full flow capacity in use
* - `add_link_volumes`
  - `link.volume`
  - Stored volume in the link
* - `add_subcatch_runoff`
  - `subcatchment.runoff`
  - Subcatchment runoff rate
* - `add_subcatch_groundwater`
  - `subcatchment.groundwater`
  - Subcatchment groundwater (baseflow) outflow rate
* - `add_pollutant_concentration`
  - `Node.quality`
  - Per-node concentration of one pollutant, in its declared concentration
    units (`mg/L`, `ug/L`, `#/L`). Add one collector per pollutant.
* - `add_link_pollutant_concentration`
  - `Link.quality`
  - Per-link concentration of one pollutant
* - `add_rainfall`
  - `gage.rainfall`
  - Per-gage rainfall intensity
* - `add_2d_vertex_depths`
  - `Surface2D.get_vertex_render_depths`
  - Signed inundation depth (`eta_v - z_v`, m; negative = dry freeboard) at
    2D mesh vertices, via one bulk read. Requires an engine built with the
    2D module **and** a model with an active 2D surface; binding fails with
    a clear error otherwise, including under the `IGNORE_2D` gate.
* - `add_cell_field`
  - a per-cell 2D method, e.g. `surface2d.get_depths`
  - Any per-cell 2D quantity at chosen mesh cells; see
    {ref}`2D cells <cell-fields>` below.
* - `add_clock`
  - n/a
  - 3 features: `hour_sin`, `hour_cos`, `elapsed_frac`
```

(cell-fields)=
## 2D cells: `add_cell_field`

`add_cell_field(path, cells, **args)` observes a quantity of the 2D surface at
chosen mesh cells (0-based triangle indices). `path` is a catalog method of a
2D service that either returns one value per cell, read once per step and
gathered, or takes the cell index and returns one value, called per cell.
Extra keyword arguments go to that method, and enum arguments may be given
by name:

```python
obs = (
    ObservationBuilder()
    .add_cell_field("surface2d.get_depths", [0, 5, 9])           # water depth
    .add_cell_field("surface2d.get_rainfall", [0, 5, 9])         # rain rate, m/s
    .add_cell_field("surface2d.infiltration.rate", [0, 5, 9])    # infiltration
    .add_cell_field("surface2d.groundwater.cells", [0, 5, 9], variable="HG")
    .add_cell_field("surface2d.quality.buildup", [0, 5, 9], species="TSS")
)
```

Paths that return one value per *vertex* (such as
`surface2d.get_vertex_render_depths`) are refused: vertices are not cells, and
{py:meth}`~openswmm_gymnasium.observations.ObservationBuilder.add_2d_vertex_depths`
covers them. A missing argument, an unknown enum name and a cell index outside
the mesh are refused with the reason. Groundwater and surface-quality reads
also need the model to configure those services; binding fails with the
engine's lifecycle error when it does not.

## Units

Ordinary 1D fields use their catalogued model-unit labels; fixed-unit fields
retain their documented units. The builder labels
each feature from the catalog's unit kinds:

```python
builder.units("US", "CFS")   # ['ft', 'CFS', 'fraction', 'dimensionless', ...]
```

Every env reports the same list for the loaded model in its `reset()` info as
`info["observation_units"]`, next to `info["unit_system"]` and
`info["flow_units"]`. A feature whose source records no unit (most 2D methods)
gets an empty label.

### Limits of temperature, age and reaction observations

The current 1D element wrappers do not expose live node/link temperature,
water-age or reaction-species concentration properties. The corresponding
configuration and forcing methods therefore do not, by themselves, provide a
closed-loop state observation. `heat.current_shortwave` and
`heat.current_cloud_fraction` describe forcing rather than transported state.

This limit is specific to live 1D element observations. Groundwater services
provide cell/species data, and result-file APIs can expose recorded species.
Neither is automatically a node/link collector. Use the supported per-cell
methods above where appropriate; a full spatial observation tensor or a new
1D state collector still needs a dedicated implementation.

The practical consequence is that
{py:class}`~openswmm_gymnasium.spaces.runtime.HeatSourceTemperatureSetpoint`
is an **open-loop** actuator: an agent can drive an inlet temperature but
cannot observe the temperature field it produces. Close the loop through a
proxy (flow split, storage depth) or treat it as feed-forward.

## Reaching collectors from a declarative env config

Every collector above is addressable from an
{py:class}`~openswmm_gymnasium.spec.config.EnvConfig` (see
{doc}`specs`) through an `observations` field of the same name:
`node_depths`, `link_velocities`, `vertex_depths_2d`, and so on. Four fields
take a shape other than a plain ID list:

```json
{
  "observations": {
    "node_depths": ["J1", "J2"],
    "node_pollutant_concentration": {"TSS": ["J1"], "BOD": ["J1", "J2"]},
    "link_pollutant_concentration": {"TSS": ["C1"]},
    "vertex_depths_2d": [0, 1, 2],
    "fields": {"link.stats.max_filling": ["C1"], "link.stats.slot_share": ["C1"]},
    "cell_fields": [
      {"path": "surface2d.get_depths", "cells": [0, 5]},
      {"path": "surface2d.groundwater.cells", "cells": [0, 5], "args": {"variable": "HG"}}
    ],
    "include_clock": true
  }
}
```

The pollutant fields map a pollutant ID to the elements whose concentration
of it to observe, so several pollutants can be requested at once. `fields`
maps a catalog field path to element IDs, and `cell_fields` lists 2D
quantities with their cells and arguments. All maps and lists are applied in
order, after the named features, which keeps the observation vector's layout
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
