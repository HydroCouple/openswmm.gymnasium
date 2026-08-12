# Benchmarks — 10 Contrived MO-RL Scenarios

Plan §6A.

Each scenario is a small, hand-authored SWMM model designed to exercise
a particular axis of the framework (action-space type, reward
composition, stochastic forcing, coordination, etc.). All `.inp`
files in this directory are released under the project's Apache-2.0 license
(no client data, no real locations).

## Status

| #   | ID prefix              | Module                          | Status |
|-----|------------------------|---------------------------------|--------|
| b01 | `OpenSWMM/TwinTank-*`  | `b01_twin_tank/`                | ✅ shipped |
| b02 | `OpenSWMM/PumpStation-*` | `b02_pump_station/`           | ⏳ pending |
| b03 | `OpenSWMM/DetentionSizing-*` | `b03_detention_sizing/`   | ⏳ pending |
| b04 | `OpenSWMM/LIDvsVault-*` | `b04_lid_vs_vault/`            | ⏳ pending |
| b05 | `OpenSWMM/CSODiversion-*` | `b05_cso_diversion/`         | ⏳ pending |
| b06 | `OpenSWMM/FirstFlush-*` | `b06_first_flush/`             | ⏳ pending |
| b07 | `OpenSWMM/CascadeCoord-*` | `b07_cascade_coord/`         | ⏳ pending |
| b08 | `OpenSWMM/StochasticStorm-*` | `b08_stochastic_storm/`   | ⏳ pending |
| b09 | `OpenSWMM/PipeSizing-*` | `b09_pipe_sizing/`             | ⏳ pending |
| b10 | `OpenSWMM/TidalOutfall-*` | `b10_tidal_outfall/`         | ⏳ pending |

## b01 TwinTank — store-vs-release tradeoff

Two storage tanks in series with a controllable side orifice between
them; a 2-hour triangular design storm hits the upstream tank.

**Network:** `S1 → T1 -(ORIF)→ T2 -(OUT)→ O1`

**Variants registered:**

- `OpenSWMM/TwinTank-RTC-v0` — RTC orifice control, default design.
- `OpenSWMM/TwinTank-Joint-v0` — CIP (tank max-depths) + RTC.
- `OpenSWMM/TwinTank-MORTC-v0` — multi-objective RTC, vector reward,
  `info["mo_score"]` at terminal step.

**Competing objectives:**

- `flooding_volume` at T1 — open the orifice early to relieve upstream
  flooding.
- `peak_outflow` in OUT — keep the orifice closed to protect the
  downstream receiver.
- `setpoint_smoothness` on ORIF — penalise jittery control.

**Usage:**

```python
import openswmm_gymnasium  # registers everything
import gymnasium as gym

env = gym.make("OpenSWMM/TwinTank-RTC-v0")
obs, info = env.reset(seed=0)
obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
```

## Future: parametric generators

Plan §6A specifies a `tools/generate_benchmarks.py` script that
emits the `.inp` files from YAML parameter files. The committed
`.inp` files are the current source of truth; the generator lands
once multiple scenarios exist to share template helpers.
