# Environments

The package ships four Gymnasium env classes. All conform to the
plan §3 action-space contract — `Dict({"design": ..., "runtime": ...})`
— even when one half is empty for clarity.

## `SwmmRTCEnv` — runtime-only

Drives one or more runtime factories every `step()`. Design Dict is
empty. Standard `gymnasium.Env`; returns `(obs, reward, terminated,
truncated, info)`.

See {py:class}`openswmm_gymnasium.envs.SwmmRTCEnv`.

## `SwmmCIPEnv` — design-only

Single-step contextual-bandit pattern. The agent picks a design action;
the env simulates the whole episode under that design and returns the
cumulative cost as the terminal reward. Ideal for plugging into
NSGA-II / NSGA-III via the Platypus adapter.

See {py:class}`openswmm_gymnasium.envs.SwmmCIPEnv`.

## `SwmmJointCIPRTCEnv` — full hybrid

Design action sampled at `reset()` (or supplied via
`options["design_action"]`) and locked for the episode. Runtime control
each `step()`.

See {py:class}`openswmm_gymnasium.envs.SwmmJointCIPRTCEnv`.

## `SwmmMORTCEnv` — multi-objective

Returns vector reward (one component per `RewardTerm`) with per-term
sign-flipping so higher is better. Populates `info["mo_score"]` at
episode termination with the single-point normalised hypervolume of
the cumulative-cost vector.

See {py:class}`openswmm_gymnasium.envs.SwmmMORTCEnv`.

## Registered IDs

```{list-table}
:header-rows: 1

* - Gymnasium ID
  - Class
  - Purpose
* - `OpenSWMM/Minimal-RTC-v0`
  - `SwmmRTCEnv`
  - Framework test env over `tests/data/minimal.inp`
* - `OpenSWMM/Minimal-CIP-v0`
  - `SwmmCIPEnv`
  - As above; CIP variant
* - `OpenSWMM/Minimal-Joint-v0`
  - `SwmmJointCIPRTCEnv`
  - As above; joint variant
* - `OpenSWMM/Minimal-MORTC-v0`
  - `SwmmMORTCEnv`
  - As above; MO variant
* - `OpenSWMM/TwinTank-RTC-v0`
  - `SwmmRTCEnv`
  - Bundled benchmark b01
* - `OpenSWMM/TwinTank-Joint-v0`
  - `SwmmJointCIPRTCEnv`
  - As above; joint variant
* - `OpenSWMM/TwinTank-MORTC-v0`
  - `SwmmMORTCEnv`
  - As above; MO variant
```
