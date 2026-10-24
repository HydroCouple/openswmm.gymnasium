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
    .add_cell_field("groundwater2d.cells", [0, 5, 9], variable="HG")
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

(custom-collectors)=
## Custom collectors

Some engine surfaces are neither one number per element (`add_field`) nor one
per mesh cell (`add_cell_field`): flow tracing, coupling receipts, the 2D
groundwater ledgers and LID node layers. Observe them with a collector of your
own: any object with

* `requires`, the catalog paths it reads (envs check them when they are built);
* `size`, the number of features;
* `units(unit_system, flow_units)`, one label per feature;
* `bind(adapter)`, called at every `reset()` after the run has started;
* `collect(adapter)`, which returns `size` float32 values.

Read with `adapter.call(path, ...)` for catalog methods and the engine API on
`adapter.solver` otherwise, and reach optional modules through
`optional_module(adapter, name)`. Add the collector with
`ObservationBuilder.add_collector(collector)`:

```python
import numpy as np
from openswmm_gymnasium._engine import optional_module
from openswmm_gymnasium.observations import ObservationBuilder
```

Custom collectors cannot be named from an `EnvConfig`. The examples below are
tested in `tests/unit/test_observations_surfaces.py`.

### LID node layers

`infrastructure.lids.node_profile(node)` lists the runtime sublayers of a
storage-node LID (`[LID_NODES]`). Each has its one-based authored `layer`, its
`bottom` and `top` heights above the invert, and its volumetric `moisture`.
`get_layers(control)` returns the authored stack, which is configuration.

```python
class LidLayerMoisture:
    requires = ("infrastructure.lids.node_assignment", "infrastructure.lids.node_profile")

    def __init__(self, node, layers):
        self.node, self.layers, self.size = node, list(layers), len(layers)

    def units(self, unit_system, flow_units):
        return ["fraction"] * self.size

    def bind(self, adapter):
        if adapter.call("infrastructure.lids.node_assignment", self.node) is None:
            raise ValueError(f"{self.node!r} has no LID control")

    def collect(self, adapter):
        rows = adapter.call("infrastructure.lids.node_profile", self.node)
        out = []
        for k in self.layers:  # thickness-weighted mean over the layer's cells
            h = np.array([r["top"] - r["bottom"] for r in rows if r["layer"] == k])
            theta = np.array([r["moisture"] for r in rows if r["layer"] == k])
            out.append(theta @ h / h.sum())
        return np.asarray(out, dtype=np.float32)

obs = ObservationBuilder().add_field("node.volume", ["S"]).add_collector(
    LidLayerMoisture("S", [2, 3])
)
```

A LID node's `node.volume` includes the water its layers retain.

### 2D groundwater ledgers

`groundwater2d.ledger(term)` reports the aquifer's water ledger in m³. Storage
terms are instantaneous and the others are cumulative since the start.
`groundwater2d.species_ledger(species, term)` reports a transported species'
cumulative ledger in that species' mass units. Each is a single value for the
whole aquifer. `groundwater2d.transport` holds the transport configuration
(sources, parameters, boundaries) and has no ledger of its own.

```python
from openswmm.engine import GroundwaterLedger, GroundwaterSpeciesLedger

class GroundwaterLedgers:
    requires = ("surface2d", "groundwater2d.ledger", "groundwater2d.species_ledger")

    def __init__(self, water, species, terms):
        self.water = [GroundwaterLedger[t] for t in water]
        self.species, self.terms = species, [GroundwaterSpeciesLedger[t] for t in terms]
        self.size = len(self.water) + len(self.terms)

    def units(self, unit_system, flow_units):
        return ["m3"] * len(self.water) + [f"{self.species} mass"] * len(self.terms)

    def bind(self, adapter):
        optional_module(adapter, "surface2d")
        gw = adapter.solver.groundwater2d
        if not gw.active:
            raise RuntimeError("the model has no [2D_AQUIFER]")
        self.idx = list(gw.species).index(self.species)

    def collect(self, adapter):
        values = [adapter.call("groundwater2d.ledger", t) for t in self.water]
        values += [adapter.call("groundwater2d.species_ledger", self.idx, t) for t in self.terms]
        return np.asarray(values, dtype=np.float32)

obs = ObservationBuilder().add_collector(
    GroundwaterLedgers(["STORAGE", "INFIL_IN"], "Dye", ["STORAGE", "SOURCE_IN"])
)
```

### Coupling frames

A `CouplingFrame`, applied with `adapter.call("coupling.apply_frame", frame)`,
is an input. It sets runtime sources, boundaries and forcings in one step,
and a frame the engine rejects changes nothing. What a frame delivered is
observed through the receipts: `coupling.surface.source_receipt(source_id,
cell).applied_m3` per cell and the gross `coupling.surface.water_totals`, in
m³ since the start.

```python
from openswmm.engine import BadIndexError

class SourceReceipts:
    requires = ("surface2d", "coupling.surface.source_receipt", "coupling.surface.water_totals")

    def __init__(self, source_id, cells):
        self.source_id, self.cells, self.size = source_id, list(cells), len(cells) + 1

    def units(self, unit_system, flow_units):
        return ["m3"] * self.size

    def bind(self, adapter):
        optional_module(adapter, "surface2d")

    def applied(self, adapter, cell):
        try:
            return adapter.call("coupling.surface.source_receipt", self.source_id, cell).applied_m3
        except BadIndexError:  # no frame has set this source on this cell yet
            return 0.0

    def collect(self, adapter):
        applied = [self.applied(adapter, c) for c in self.cells]
        total = adapter.solver.coupling.surface.water_totals["external_in"]
        return np.asarray([*applied, total], dtype=np.float32)
```

### Flow tracing

`FlowTracer` is a standalone handle, not a solver service.
`prepare_live(out_path)` scans the reporting periods that a running engine has
written so far, so a collector can rebuild `trace.averages` and trace from a
seed node at each step. Values change once per `REPORT_STEP` and average over
every period since the start. Before the first period is written,
`prepare_live` raises `TraceError` with `TraceStatus.IO`. Give the env an
`out_path`, and pass the same path to the collector:

```python
from openswmm.engine import FlowTracer, TraceDirection, TraceError, TraceLink, TraceNode, TraceStatus
from openswmm_gymnasium.envs import SwmmRTCEnv

class TraceFractions:
    requires = ("trace", "trace.prepare_live", "trace.estimate")

    def __init__(self, out_path, seed, nodes, direction=TraceDirection.UPSTREAM):
        self.out, self.seed, self.nodes, self.direction = str(out_path), seed, list(nodes), direction
        self.size, self.tracer = len(nodes), None

    def units(self, unit_system, flow_units):
        return ["fraction"] * self.size

    def bind(self, adapter):  # each reset starts a new output file
        if self.tracer is not None:
            self.tracer.close()
        to_m = 0.3048 if adapter.unit_system == "US" else 1.0
        self.tracer = FlowTracer(
            [TraceNode(n.id, n.type) for n in adapter.solver.nodes],
            [TraceLink(k.id, k.from_node.index, k.to_node.index, k.type, k.length * to_m)
             for k in adapter.solver.links],
        )
        self.idx = [self.tracer.node_ids.index(n) for n in self.nodes]

    def collect(self, adapter):
        try:
            self.tracer.prepare_live(self.out)
        except TraceError as exc:
            if exc.code != TraceStatus.IO:
                raise
            return np.zeros(self.size, dtype=np.float32)  # no reporting period yet
        result = self.tracer.estimate(self.seed, self.direction)
        return np.asarray([result.nodes[i].ratio for i in self.idx], dtype=np.float32)

env = SwmmRTCEnv(
    "model.inp",
    out_path="run.out",
    observation_builder=ObservationBuilder().add_collector(
        TraceFractions("run.out", "O1", ["J2", "J10"])
    ),
)
```

When tracing upstream from an outfall, each value is the share of the
outfall's averaged flow that passed through that node. When tracing
downstream from a node, each value is the share of that node's flow that
reaches it. Ratios can exceed 1 where flow recirculates. Nodes with ponding or
external exchanges need `TraceNodeFlags`; see the engine's tracing guide.

### Heat snapshots are not observations

`heat.snapshot()` captures the heat settings before a run, and `restore()` puts
them back. The engine refuses both with `LifecycleError` once the model is
initialized, and envs bind collectors only after the run starts, so a
collector can never use a snapshot. A snapshot is useful between `open()` and
`initialize()`, the same window design factories write in:

```python
from openswmm_gymnasium._engine import SolverAdapter
from openswmm_gymnasium.spaces.design import HeatSourceTemperature

adapter = SolverAdapter("model.inp")  # [OPTIONS] HEAT_TRANSPORT YES
adapter.open()
heat = optional_module(adapter, "heat")
design = HeatSourceTemperature(["DWF"], 0.0, 40.0)
design.bind(adapter)
with heat.snapshot() as saved:
    design.apply(adapter, np.array([30.0], dtype=np.float32))
    ...  # inspect the candidate
    saved.restore()  # back to the authored settings
adapter.initialize()
```

Envs open a fresh model for every episode, so they do not need a snapshot
between episodes. For the heat state that can be observed during a run, see
the limits under Units below.

## Units

Ordinary 1D fields use their catalogued model-unit labels; fixed-unit fields
retain their documented units. The builder labels
each feature from the catalog's unit kinds:

```python
builder.units("US", "CFS")   # ['ft', 'CFS', 'fraction', 'dimensionless', ...]
```

Every env reports the same list for the loaded model in its `reset()` info as
`info["observation_units"]`, next to `info["unit_system"]` and
`info["flow_units"]`. 2D quantities are SI whatever the model's flow units:
depths and heads are labelled `m`, rain and source rates `m/s`. A feature
whose source records no unit gets an empty label.

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
      {"path": "groundwater2d.cells", "cells": [0, 5], "args": {"variable": "HG"}}
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
