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
* - `add_link_flows`
  - `Links.get_flow`
  - Instantaneous flow
* - `add_link_depths`
  - `Links.get_depth`
  - Instantaneous depth in link
* - `add_link_settings`
  - `Links.get_control_setting`
  - Current setting in `[0, 1]`
* - `add_subcatch_runoff`
  - `Subcatchments.get_runoff`
  - Subcatchment runoff rate
* - `add_rainfall`
  - `Gages.get_rainfall`
  - Per-gage rainfall intensity
* - `add_clock`
  - n/a
  - 3 features: `hour_sin`, `hour_cos`, `elapsed_frac`
```

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
