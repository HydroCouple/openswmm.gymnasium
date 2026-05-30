# Wrappers

All wrappers live in {py:mod}`openswmm_gymnasium.wrappers` and follow
the standard Gymnasium `Wrapper` / `ActionWrapper` / `RewardWrapper` /
`ObservationWrapper` protocols.

## Action transformation

### `RescaleBoxActions`

Recursively rescale every `Box` leaf of the action space to a uniform
window (default `[0, 1]`). Handy for plugging in agents whose policy
naturally emits values in `[0, 1]` or `[-1, 1]`.

```python
from openswmm_gymnasium.wrappers import RescaleBoxActions
env = RescaleBoxActions(env, src_low=0.0, src_high=1.0)
```

### `MaskDesignAction` / `MaskRuntimeAction`

Flatten a `Dict({design, runtime})` env so the agent only sees one
half; the other half is auto-filled (mid-range for runtime, sampled
once at `reset()` for design — or pinned with `frozen_design=`).

```python
from openswmm_gymnasium.wrappers import MaskDesignAction
env = MaskDesignAction(env, frozen_design={"node_max_depth": [10.0, 12.0]})
```

## Reward transformation

### `LinearScalarize` / `TchebycheffScalarize`

Convert vector reward to scalar:

```python
from openswmm_gymnasium.wrappers import LinearScalarize, TchebycheffScalarize

env = LinearScalarize(mo_env, weights=[1.0, 0.5, 0.25])
env = TchebycheffScalarize(mo_env, weights=[1, 1, 1], utopia=[0, 0, 0])
```

## Observation extension

### `ForecastObservation`

Append features from a user-supplied callable each step. See
{doc}`./observations` for an example.

## Recording

### `RecordTrajectory`

Write one JSONL file per episode for later visualisation / regression
testing.

```python
from openswmm_gymnasium.wrappers import RecordTrajectory
env = RecordTrajectory(env, output_dir="artifacts/run-001/")
```

Files are named `episode_<NNNNN>.jsonl`. The companion
{py:mod}`openswmm_gymnasium.viz` module consumes this format.
