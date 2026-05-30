# Multi-objective scoring

The {py:mod}`openswmm_gymnasium.scoring` module ships pure-numpy
implementations of standard MOO metrics. No `pymoo` / `platypus`
dependency on the critical path; an optional
{py:mod}`~openswmm_gymnasium.scoring.adapters.platypus_adapter` plugs
into the [Platypus](https://github.com/Project-Platypus/Platypus) MOEA
library.

## Pareto-front extraction

```python
from openswmm_gymnasium.scoring import pareto_front
nd = pareto_front(points)  # (n, d) → non-dominated subset
```

## Hypervolume

Exact closed-form for `d ≤ 2`; Monte Carlo for `d ≥ 3`. The
{py:func}`~openswmm_gymnasium.scoring.normalized_hypervolume` helper
normalises against `ideal` / `reference` so the result is in `[0, 1]`
regardless of objective scale.

```python
from openswmm_gymnasium.scoring import hypervolume, normalized_hypervolume

hv = hypervolume(points, reference=[1.0, 1.0])
nhv = normalized_hypervolume(points, ideal=[0, 0], reference=[100, 50])
```

## IGD / IGD+

{py:func}`~openswmm_gymnasium.scoring.igd` is the classic Inverted
Generational Distance; {py:func}`~openswmm_gymnasium.scoring.igd_plus`
is the Pareto-compliant variant — zero whenever the approximation
weakly dominates the reference set.

```python
from openswmm_gymnasium.scoring import igd, igd_plus
igd_val  = igd(approximation, true_front)
idg_plus_val = igd_plus(approximation, true_front)
```

## Other indicators

- {py:func}`~openswmm_gymnasium.scoring.epsilon_indicator` — additive ε.
- {py:func}`~openswmm_gymnasium.scoring.spread` — Schott's spacing
  metric (stddev of nearest-neighbour distances).
- {py:func}`~openswmm_gymnasium.scoring.r2_indicator` — weighted
  Tchebycheff R2 over a set of reference weight vectors.

## Per-episode score in MO envs

{py:class}`~openswmm_gymnasium.envs.SwmmMORTCEnv` populates
`info["mo_score"]` at episode termination with the single-point
normalised hypervolume of the episode's cumulative-cost vector against
the user-supplied `ideal_point` and `reference_point`. Useful as a
training signal or for cross-policy comparison.
