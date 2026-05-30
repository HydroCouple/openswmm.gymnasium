# openswmm.gymnasium

[Farama Gymnasium](https://gymnasium.farama.org/) environments for joint
**Capital Improvement Plan (CIP)** design and **Real-Time Control (RTC)**
optimization of SWMM stormwater networks, with multi-objective scoring.

Built on the handle-based, thread-safe
[openswmm.engine](https://github.com/HydroCouple/openswmm.engine) v6
Python API.

**Status:** pre-release, under active development. See
[docs/IMPLEMENTATION_PLAN.md](docs/IMPLEMENTATION_PLAN.md) for the
authoritative scope, architecture, and milestone schedule.

## Installation

```bash
pip install openswmm.gymnasium                      # core
pip install "openswmm.gymnasium[mo]"                # + multi-objective (mo-gymnasium)
pip install "openswmm.gymnasium[platypus]"          # + Platypus MOO adapter
pip install "openswmm.gymnasium[viz]"               # + Plotly trajectory visualizations
pip install "openswmm.gymnasium[mo,platypus,viz]"   # everything
```

## License

MIT — see [LICENSE](LICENSE). Contributions require signing the
[CLA](CLA.md).
