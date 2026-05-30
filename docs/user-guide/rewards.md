# Rewards

All built-in reward terms are framed as **cost-to-minimise** internally
(positive contribution = bad outcome). The env negates the aggregated
cost so higher reward is better, per Gymnasium convention.

Maximise-direction terms (e.g.
{py:class}`~openswmm_gymnasium.rewards.ReliabilityMargin`) declare
`direction = "maximize"`; the env's composer handles sign-flipping
automatically.

## Built-in terms

```{list-table}
:header-rows: 1

* - Term
  - Direction
  - Description
* - {py:class}`~openswmm_gymnasium.rewards.FloodingVolume`
  - minimise
  - Sum of overflow volume across nodes per step
* - {py:class}`~openswmm_gymnasium.rewards.CSOVolume`
  - minimise
  - Same math, restricted to tagged CSO nodes
* - {py:class}`~openswmm_gymnasium.rewards.PeakOutflow`
  - minimise
  - Running-max increment of link flow
* - {py:class}`~openswmm_gymnasium.rewards.ReliabilityMargin`
  - maximise
  - Minimum freeboard across nodes
* - {py:class}`~openswmm_gymnasium.rewards.SetpointSmoothness`
  - minimise
  - L2-norm-squared of Δsetting between steps
```

## Custom terms

Subclass {py:class}`~openswmm_gymnasium.rewards.RewardTerm` (it is a
runtime-checkable Protocol). Implement `bind`, `reset`, `step`:

```python
class MyCustomTerm:
    name = "my_term"
    direction = "minimize"

    def bind(self, adapter):
        self._idx = adapter.links.get_index("OUT")

    def reset(self):
        pass

    def step(self, adapter, dt_seconds):
        return float(adapter.links.get_flow(self._idx)) * dt_seconds
```

Register with {py:class}`~openswmm_gymnasium.rewards.RewardRegistry` if
you want name-based lookup:

```python
from openswmm_gymnasium.rewards import RewardRegistry

RewardRegistry.register("my_term", MyCustomTerm)
cls = RewardRegistry.get("my_term")
```
