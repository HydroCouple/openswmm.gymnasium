# SPDX-License-Identifier: Apache-2.0
#
# Copyright 2026 Caleb Buahin
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
The study manifest: a study's definition, every evaluation and its result.

It is hydrocouple.platform's published C{study-manifest-1} contract, written by
L{run_study<openswmm_gymnasium.optimize.study.run_study>} and by the platform's
C{run_platform_study} through the same L{study_definition}, so a local study and
a platform study of one model compare equal apart from C{runtime}. The model is
named by its packaged files, not by a path. Building one needs the platform's
base install (C{pip install hydrocouple.platform}); without it no manifest is
written.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import json
import platform as _platform
from importlib import metadata
from pathlib import Path
from typing import Any

from openswmm_gymnasium.optimize.space import (
    DesignDimension,
    flat_bounds,
    objective_directions,
)
from openswmm_gymnasium.spec.config import EnvConfig

MANIFEST_FILE = "study-manifest.json"
#: The evaluator workflow a study's evaluations are, locally and on the platform.
WORKFLOW, WORKFLOW_VERSION = "gym.evaluate", "1"


def available() -> bool:
    """Whether the platform's contracts are installed to write a manifest with."""
    try:
        import hydrocouple_platform.contracts.studies  # noqa: F401
    except ImportError:
        return False
    return True


def model_inputs(files: list[dict[str, Any]]) -> dict[str, str]:
    """Package path to sha256 of a revision's files, without C{composition.json}, which
    records when it was packaged."""
    return {f["path"]: f["sha256"] for f in files if f["role"] != "composition"}


def study_definition(
    env_config: EnvConfig,
    opt: Any,
    dims: list[DesignDimension],
    inputs: dict[str, str],
    package_inp: str,
) -> dict[str, Any]:
    """What the study evaluates and how it searches: identical wherever it runs.

    @param opt: The L{OptimizationConfig<openswmm_gymnasium.optimize.OptimizationConfig>};
        everything but C{workers}, which only says how it ran.
    @param inputs: L{model_inputs} of the packaged model.
    @param package_inp: The model's path in the package (C{"model/model.inp"}).
    @rtype: dict
    """
    low, high = flat_bounds(dims)
    labels = [label for d in dims for label in d.labels]
    payload = env_config.model_dump(mode="json")
    payload.update(inp_path=package_inp, rpt_path=None, out_path=None)
    return {
        "workflow": WORKFLOW,
        "workflow_version": WORKFLOW_VERSION,
        "model_inputs": dict(sorted(inputs.items())),
        "env_config": payload,
        "dimensions": [
            {"label": label, "low": float(lo), "high": float(hi)}
            for label, lo, hi in zip(labels, low, high, strict=True)
        ],
        "objectives": [
            {"name": n, "direction": d} for n, d in objective_directions(env_config).items()
        ],
        "optimization": opt.model_dump(mode="json", exclude={"workers"}),
        "seed": opt.seed,
    }


def versions() -> dict[str, str | None]:
    """Installed versions of the packages a study's numbers depend on."""
    out: dict[str, str | None] = {"python": _platform.python_version()}
    for name in ("openswmm", "openswmm.gymnasium", "hydrocouple.platform"):
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def write_study_manifest(
    output_dir: str | Path,
    definition: dict[str, Any],
    evaluations: list[dict[str, Any]],
    result: dict[str, Any],
    runtime: dict[str, Any],
) -> Path:
    """Validate a manifest against the platform contract and write C{study-manifest.json}.

    @param result: The study's result payload; its C{execution} part goes to C{runtime}.
    @rtype: L{Path}
    """
    from hydrocouple_platform.contracts.studies import StudyManifest

    manifest = StudyManifest(
        definition=definition,
        evaluations=evaluations,
        result={k: v for k, v in result.items() if k != "execution"},
        runtime=runtime | {"execution": result.get("execution"), "versions": versions()},
    )
    path = Path(output_dir) / MANIFEST_FILE
    path.write_text(
        json.dumps(manifest.model_dump(mode="json", by_alias=True), indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    return path


def package_locally(env_config: EnvConfig, output_dir: Path) -> tuple[dict[str, str], str, Any]:
    """Package the study's model under C{output_dir/package} as the platform would.

    @return: The L{model_inputs}, the model's path in the package, and the package's
        revision manifest.
    @raise hydrocouple_platform.packaging.package.PackagingError: The model cannot be
        packaged (for example, it loads a native plugin).
    """
    from hydrocouple_platform.packaging import package_swmm_model

    package = package_swmm_model(env_config.inp_path, output_dir / "package")
    files = package.manifest.model_dump(mode="json", by_alias=True)["files"]
    model = next(f["path"] for f in files if f["role"] == "model")
    return model_inputs(files), model, package.manifest
