# Architecture

This page summarises the package layout; see
[`IMPLEMENTATION_PLAN.md`](https://github.com/HydroCouple/openswmm.gymnasium/blob/main/docs/IMPLEMENTATION_PLAN.md)
for the authoritative architecture document.

## Class hierarchy

```
gymnasium.Env                                          (Farama)
│
├── SwmmRTCEnv               (P1)
├── SwmmCIPEnv               (P3, single-step contextual bandit)
├── SwmmJointCIPRTCEnv       (P3)
└── SwmmMORTCEnv             (P4, vector reward)
        ↑
        └── mo_gymnasium.MOEnv  (mixed in when mo-gymnasium installed)
```

## Package layout

```
src/openswmm_gymnasium/
    __init__.py           # version + Gymnasium env registration
    _engine/              # the *only* module that imports openswmm.engine
    envs/                 # SwmmRTCEnv, SwmmCIPEnv, SwmmJointCIPRTCEnv, SwmmMORTCEnv
    spaces/               # action factories: runtime (RTC), design (CIP)
    observations/         # ObservationBuilder + per-feature collectors
    rewards/              # RewardTerm Protocol, built-in terms, RewardRegistry
    scoring/              # pure-math MOO indicators
    wrappers/             # action_remap, action_mask, scalarize, forecast, record_trajectory
    viz/                  # Plotly figure factories (optional [viz] extra)
    benchmarks/           # bNN scenario subpackages, each with scenario.inp
```

## Thread-safety contract

The package targets the handle-based, thread-safe `openswmm.engine.Solver`
**only**. The legacy v5 singleton solver under `openswmm.legacy.engine` is
rejected at adapter construction
({py:class}`openswmm_gymnasium._engine.LegacySolverRejectedError`).

Each `SwmmRTCEnv` instance owns a distinct `SWMM_Engine` handle, so
`gymnasium.vector.SyncVectorEnv` (threads) and `AsyncVectorEnv`
(processes) both work.

## Reward sign convention

Built-in terms always return non-negative contributions. The env's
reward composer flips signs per term direction:

- `direction = "minimize"` → cost = +contribution → reward subtracts.
- `direction = "maximize"` → benefit = +contribution → reward adds.

After aggregation, scalar reward = `-cost`, satisfying Gymnasium's
"higher is better" contract.

## Solver lifecycle

```
SolverAdapter.open()      # parse .inp; create engine handle
[apply design factories]  # pre-initialize edits via Nodes/Links setters
SolverAdapter.initialize() # engine sets up data structures
loop:
    [apply runtime factories]  # Controls.set_link_setting etc.
    SolverAdapter.step()       # advance one routing timestep
SolverAdapter.end()
SolverAdapter.report()
SolverAdapter.close()      # idempotent
```
