# Contributing

See the canonical `CONTRIBUTING.md
<https://github.com/HydroCouple/openswmm.gymnasium/blob/main/CONTRIBUTING.md>`_
at the repository root for the development workflow, code style, CLA
requirements, and PR process.

## Quick reference

- Code style: [Ruff](https://docs.astral.sh/ruff/) — `ruff check`, `ruff format`.
- Docstrings: **epytext** convention (`@param`, `@type`, `@return`,
  `@rtype`, `@raise`, `@ivar`, `@author`, `@copyright`, `@license`).
  See plan §13 for the full tag table.
- Tests: no engine mocks — drive the real
  {py:class}`openswmm.engine.Solver` against tiny in-tree fixtures.
- License: contributions are MIT-licensed; first PR requires CLA signing
  via CLA Assistant.
