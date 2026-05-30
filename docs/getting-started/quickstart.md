# Quickstart

A 30-second end-to-end example using the bundled `b01 TwinTank` benchmark.

## Single-objective RTC

```python
import gymnasium as gym
import openswmm_gymnasium  # triggers env registration

env = gym.make("OpenSWMM/TwinTank-RTC-v0")
obs, info = env.reset(seed=0)

terminated = False
cumulative_reward = 0.0
while not terminated:
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    cumulative_reward += reward

print(f"Final reward: {cumulative_reward:.3f}")
print(f"Per-component breakdown: {info['reward_components']}")
env.close()
```

## Multi-objective with HV scoring

```python
import gymnasium as gym
import openswmm_gymnasium

env = gym.make("OpenSWMM/TwinTank-MORTC-v0")
obs, info = env.reset(seed=0)

terminated = False
while not terminated:
    action = env.action_space.sample()
    obs, reward_vec, terminated, truncated, info = env.step(action)

print(f"Normalised HV: {info['mo_score']:.3f}")  # in [0, 1]
print(f"Cumulative cost vector: {info['cumulative_cost']}")
env.close()
```

## Recording trajectories for visualisation

```python
import gymnasium as gym
import openswmm_gymnasium
from openswmm_gymnasium.wrappers import RecordTrajectory
from openswmm_gymnasium.viz import TrajectoryRun
from openswmm_gymnasium.viz.figures import pareto_front

# Wrap the env so each episode lands in artifacts/run-001/episode_*.jsonl
env = RecordTrajectory(gym.make("OpenSWMM/TwinTank-MORTC-v0"), "artifacts/run-001/")

for ep in range(20):
    env.reset(seed=ep)
    terminated = False
    while not terminated:
        _, _, terminated, _, _ = env.step(env.action_space.sample())
env.close()

# Visualise the run
run = TrajectoryRun.from_dir("artifacts/run-001/")
fig = pareto_front(run)
fig.write_html("pareto.html")
```

## Joint CIP + RTC

For envs that optimise both design (pre-simulation) and runtime control:

```python
import gymnasium as gym
import openswmm_gymnasium

env = gym.make("OpenSWMM/TwinTank-Joint-v0")

# Pass a specific design at reset; runtime is the per-step action.
import numpy as np
design = {"node_max_depth": np.array([10.0, 14.0], dtype=np.float32)}
obs, info = env.reset(seed=0, options={"design_action": design})

while True:
    action = env.action_space.sample()  # design is ignored after reset
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        break
env.close()
```
