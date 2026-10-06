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
  - Flooded volume across nodes per step, in project volume units
* - {py:class}`~openswmm_gymnasium.rewards.CSOVolume`
  - minimise
  - Same math, restricted to tagged CSO nodes
* - {py:class}`~openswmm_gymnasium.rewards.PeakOutflow`
  - minimise
  - Running-max increment of link flow
* - {py:class}`~openswmm_gymnasium.rewards.TSSLoad`
  - minimise
  - Pollutant mass flux (flow × concentration × dt) through links. Despite the
    name it works for any declared pollutant — pass `pollutant=...`
* - {py:class}`~openswmm_gymnasium.rewards.ReliabilityMargin`
  - maximise
  - Minimum freeboard across nodes
* - {py:class}`~openswmm_gymnasium.rewards.SetpointSmoothness`
  - minimise
  - L2-norm-squared of Δsetting between steps
* - {py:class}`~openswmm_gymnasium.rewards.UncontrolledDischarge`
  - minimise
  - Positive flow × dt through links feeding untreated outfalls
* - {py:class}`~openswmm_gymnasium.rewards.StorageUnderUtilization`
  - minimise
  - Time-integrated unused storage headroom (`1 − depth/max_depth`)
* - {py:class}`~openswmm_gymnasium.rewards.PumpEnergy`
  - minimise
  - Pump effort `setting × rated_power × dt` summed across pumps
* - {py:class}`~openswmm_gymnasium.rewards.SurchargeSlotShare`
  - minimise
  - Run-level Preissmann-slot storage share as a pressurisation proxy.
    **Dimensionless, in `[0, 1]`** — a ratio of volumes, so it is the one
    term with no unit-system dependence at all. **Finite-volume routing
    only**; see below.
```

### `SurchargeSlotShare` and the router

The Preissmann slot is a finite-volume construct: the notional narrow slot
above a closed conduit's crown that lets a free-surface scheme carry
pressurized flow. The fraction of a conduit's stored volume held in that slot
is a direct, continuous measure of pressurisation — unlike `surcharge_time`,
which is a binary threshold crossing integrated over time, and unlike
`max_filling`, which saturates at 1.0 and then carries no further information
however hard the pipe is pressurized.

Only `FLOW_ROUTING FV` models a slot. Under `DYNWAVE` — and equally under
`STEADY` and `KINWAVE` — every slot statistic reads a hard `0.0`, a value
**indistinguishable from a genuinely unpressurized network**. A term reading
it on such a model would report perfection on every step of every episode and
hand the agent a flat, unmovable signal, with nothing raised and nothing
logged. `bind` therefore reads `[OPTIONS] FLOW_ROUTING` and **raises** on any
non-FV router rather than warning. If the engine will not report the option
(an older build), detection is skipped and the constraint is yours to honour.

The per-step contribution is the increase in a running maximum of the
links' **mean** slot share, so cumulative reward over the episode equals the
highest run-level mean share attained and stays in `[0, 1]`. `dt_seconds` is
unused — the engine has already time-integrated, and multiplying again would
double-count.

## Engine statistics

`FloodingVolume` (and hence `CSOVolume`), `PeakOutflow` and
`SurchargeSlotShare` read the engine's own cumulative statistics —
`swmm_node_get_stat_vol_flooded`, `swmm_link_get_stat_max_flow` and
`swmm_link_get_stat_slot_share` — rather than re-integrating a sampled rate in
Python. The engine accumulates every **routing** step, whereas a Python term
only sees state at **env**-step boundaries, so anything that happens between
two env steps would otherwise be missed. Three consequences:

* `FloodingVolume` is in project **volume** units (ft³ / m³) and does not
  scale with `dt_seconds`.
* `SurchargeSlotShare` **could not** be reconstructed in Python even if it
  wanted to. The statistic is a ratio of *time integrals*,
  `(∫ slot_volume dt) / (∫ volume dt)` — not an average of instantaneous
  ratios. Sampling `slot_volume / volume` at step boundaries and averaging
  weights every sample equally instead of by the volume present, which is a
  different number. This is the clearest case in the table of a statistic
  that *must* be read rather than re-derived.
* `TSSLoad`, `ReliabilityMargin`, `PumpEnergy`, `UncontrolledDischarge`,
  `StorageUnderUtilization` and `SetpointSmoothness` keep their Python loops:
  no engine statistic carries the same quantity (a cumulative max is not a
  per-step minimum; unweighted pump on-time is not `∫ setting·power dt`;
  total conveyed volume is not outfall-directed volume when a link reverses;
  and nothing accumulates pollutant mass flux, unused headroom or setpoint
  churn).

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
