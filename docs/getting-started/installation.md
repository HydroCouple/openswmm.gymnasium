# Installation

## Requirements

- Python 3.10 or later
- The compiled [openswmm.engine](https://github.com/HydroCouple/openswmm.engine)
  v6 (handle-based, thread-safe) — installed automatically as a required
  dependency.

## Install from PyPI

```bash
pip install openswmm.gymnasium                      # core
pip install "openswmm.gymnasium[mo]"                # + multi-objective (mo-gymnasium)
pip install "openswmm.gymnasium[platypus]"          # + Platypus MOO adapter
pip install "openswmm.gymnasium[viz]"               # + Plotly trajectory visualizations
pip install "openswmm.gymnasium[mo,platypus,viz]"   # everything
```

## Install from source

```bash
git clone https://github.com/HydroCouple/openswmm.gymnasium.git
cd openswmm.gymnasium
pip install -e ".[dev,docs]"
```

The `[dev]` extra installs `pytest`, `ruff`, and friends. The `[docs]`
extra adds Sphinx and the `sphinx-epytext` extension for rendering the
package's epytext docstrings.

## Verifying the install

```python
import openswmm_gymnasium
import gymnasium as gym

print(openswmm_gymnasium.__version__)
env = gym.make("OpenSWMM/TwinTank-RTC-v0")
print(env.observation_space, env.action_space)
```

If this prints the env's spaces without error, you're ready.
