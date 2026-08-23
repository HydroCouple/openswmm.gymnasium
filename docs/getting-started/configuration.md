# Configuration

The package itself has no global configuration — every env is configured
per-instance via constructor arguments. The recommended pattern:

1. Choose a benchmark scenario (or your own `.inp`).
2. Pick the env variant — `SwmmRTCEnv`, `SwmmCIPEnv`, `SwmmJointCIPRTCEnv`,
   or `SwmmMORTCEnv`.
3. Build an `ObservationBuilder` describing the feature set.
4. Pick `RewardTerm` instances describing the cost / benefit landscape.
5. For MO envs, pick `ideal_point` and `reference_point` for HV
   normalisation.

## Per-env defaults

The bundled benchmarks (e.g. `b01_twin_tank`) ship with curated
defaults. To customise, instantiate the env directly:

```python
from openswmm_gymnasium.envs import SwmmRTCEnv
from openswmm_gymnasium.observations import ObservationBuilder
from openswmm_gymnasium.rewards import FloodingVolume, PeakOutflow
from openswmm_gymnasium.spaces.runtime import OrificeSetting

env = SwmmRTCEnv(
    "path/to/model.inp",
    runtime_factories=[OrificeSetting(["ORIF"])],
    observation_builder=ObservationBuilder()
        .add_node_depths(["T1", "T2"])
        .add_link_flows(["OUT"]),
    reward_terms=[
        FloodingVolume(node_ids=["T1"]),
        PeakOutflow(link_ids=["OUT"]),
    ],
    control_interval_steps=4,  # advance 4 routing steps per env step
)
```

## Multi-objective configuration

Multi-objective envs require explicit `ideal_point` (best-case cost
vector) and `reference_point` (nadir / worst-case) for normalised
hypervolume scoring:

```python
from openswmm_gymnasium.envs import SwmmMORTCEnv

env = SwmmMORTCEnv(
    "path/to/model.inp",
    runtime_factories=[...],
    observation_builder=...,
    reward_terms=[
        FloodingVolume(node_ids=["T1"]),
        PeakOutflow(link_ids=["OUT"]),
    ],
    ideal_point=[0.0, 0.0],            # zero cost is best
    reference_point=[1.0e5, 50.0],     # generous worst-case
)
```

Both vectors must match the order and length of `reward_terms`. The
returned `info["mo_score"]` at episode termination is the normalised
hypervolume of the cumulative-cost vector in `[0, 1]`.

## Validating training models

Training pipelines that generate or perturb `.inp` files programmatically —
CIP design search, domain randomization, or bulk scenario sweeps — can produce
models with post-parse validation errors (undefined objects, missing curves,
bad references). By default the engine opens *strictly* and raises on such a
model, which is the right behaviour for the env run path (a model must be valid
to be stepped).

For **pre-flight validation**, `SolverAdapter.open()` accepts an opt-in
`lenient=True` flag. A lenient open keeps a broken model `OPENED` and
inspectable instead of hard-failing, and records the problems on two accessors:

```python
from openswmm_gymnasium._engine import SolverAdapter

adapter = SolverAdapter("candidate.inp")
adapter.open(lenient=True)          # does not raise on validation errors

if adapter.open_errors:
    # Skip / report this candidate instead of crashing the rollout.
    print("rejecting candidate:", adapter.open_errors)
for w in adapter.open_warnings:
    print("warning:", w)
adapter.close()
```

`open_errors` and `open_warnings` are lists of human-readable message strings;
a clean strict open leaves `open_errors` empty. A leniently-opened model is
**not** runnable — validate first, then open the accepted model strictly (the
default) for the actual episode.
