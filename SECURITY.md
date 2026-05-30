# Security Policy

The OpenSWMM Gymnasium maintainers take security seriously. This document
describes how to report a vulnerability and what to expect once a report
has been received.

## Supported Versions

Security fixes target the latest published release line and the active
`main` branch.

| Version line     | Supported          |
| ---------------- | ------------------ |
| Latest release   | :white_check_mark: |
| Pre-release      | :white_check_mark: |
| Older releases   | :x:                |

## Reporting a Vulnerability

**Please do not file public GitHub issues for security problems.**

Use one of the private channels below so a fix can be prepared before
public disclosure:

1. **Preferred — GitHub private security advisory.**
   Open a draft advisory in the
   [Security tab](https://github.com/HydroCouple/openswmm.gymnasium/security/advisories/new)
   of this repository.

2. **Email.** If you do not have a GitHub account or cannot use the
   advisory flow, email **security@hydrocouple.org**.

Please include, where possible:

- A clear description of the issue and the affected component
  (e.g. `envs/`, `scoring/`, integration with `openswmm.engine`).
- Affected version, commit SHA, or release tag.
- Reproduction steps, minimal `.inp` model where applicable,
  and the observed vs. expected behaviour.
- Any proof-of-concept exploit, payload, or crash artifact.
- Your assessment of impact and which deployment scenarios are exposed.

## What to Expect

- **Acknowledgement** within **3 business days** of report.
- **Initial assessment** (confirmation, severity, scope) within
  **10 business days**.
- **Fix or mitigation plan** communicated within **30 days** of
  confirmation for high/critical issues; lower severities are scheduled
  into the next maintenance release.
- **Coordinated disclosure**: a public advisory and patched release are
  published together. Reporters are credited unless they request
  otherwise.

## Scope

In scope:

- The Python package under `src/openswmm_gymnasium/`
- Bundled reference scenarios under `src/openswmm_gymnasium/benchmarks/`
- Packaging and release tooling under `.github/` and `pyproject.toml`

Out of scope:

- Vulnerabilities in upstream `openswmm.engine` — report to that project's
  [security advisory page](https://github.com/HydroCouple/openswmm.engine/security/advisories/new).
- Vulnerabilities in upstream Farama Gymnasium — report upstream.
- Issues in third-party dependencies — please report to the dependency's
  own security channel; we will pin to a fixed version once available.
