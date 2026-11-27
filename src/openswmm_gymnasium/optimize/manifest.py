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
    components = [(d, label) for d in dims for label in d.labels]
    payload = env_config.model_dump(mode="json")
    payload.update(inp_path=package_inp, rpt_path=None, out_path=None)
    return {
        "workflow": WORKFLOW,
        "workflow_version": WORKFLOW_VERSION,
        "model_inputs": dict(sorted(inputs.items())),
        "env_config": payload,
        "dimensions": [
            {"key": d.key, "label": label, "low": float(lo), "high": float(hi)}
            | ({} if d.choices is None else {"choices": list(d.choices)})
            for (d, label), lo, hi in zip(components, low, high, strict=True)
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


def sidecar_files(env_config: EnvConfig) -> dict[str, Path]:
    """Files the env config names that the model does not: event rain and observations.

    Package path (relative to the model's folder, as the config names it) to the local
    file. A revision of the model needs them as sidecars; an absolute path is left out.

    @rtype: dict
    """
    folder = Path(env_config.inp_path).parent
    names = [e.series for e in env_config.events]
    names += [t.params["observed"] for t in env_config.reward_terms if t.kind == "fit_to_observed"]
    return {n: folder / n for n in dict.fromkeys(names) if not Path(n).is_absolute()}


def rerun(
    manifest: dict[str, Any] | str | Path,
    model: str | Path,
    output_dir: str | Path,
    *,
    workers: int = 0,
) -> dict[str, Any]:
    """Run a study again from its manifest: its env config, search and seed, on a local
    copy of its model.

    @param manifest: A study manifest, or the path of its C{study-manifest.json}.
    @param model: The local C{.inp}; its folder holds the sidecars the study names.
    @param output_dir: Folder for the run's C{evaluations.jsonl}, C{result.json} and
        manifest; the model is first packaged under C{output_dir/model-check} to check it.
    @param workers: Evaluator processes; the evaluations do not depend on it.
    @return: The study's result, as L{run_study<openswmm_gymnasium.optimize.run_study>}.
    @raise SpecError: C{VALIDATION_ERROR} when the model's files are not the study's.
    @rtype: dict
    """
    from openswmm_gymnasium.optimize.study import OptimizationConfig, run_study
    from openswmm_gymnasium.spec.errors import ErrorCode, SpecError

    doc = manifest if isinstance(manifest, dict) else json.loads(Path(manifest).read_text())
    definition = doc["definition"]
    config = EnvConfig.model_validate(dict(definition["env_config"]) | {"inp_path": str(model)})
    inputs, _, _ = package_locally(config, Path(output_dir) / "model-check")
    if inputs != definition["model_inputs"]:
        changed = sorted(set(inputs.items()) ^ set(definition["model_inputs"].items()))
        raise SpecError(
            f"[{ErrorCode.VALIDATION_ERROR}] {model} and its sidecars are not the model the "
            f"study evaluated; these files differ: {sorted({path for path, _ in changed})}."
        )
    opt = OptimizationConfig.model_validate(definition["optimization"] | {"workers": workers})
    return run_study(config, opt, output_dir)


def package_locally(env_config: EnvConfig, output_dir: Path) -> tuple[dict[str, str], str, Any]:
    """Package the study's model and its L{sidecar_files} under C{output_dir/package} as
    the platform would.

    @return: The L{model_inputs}, the model's path in the package, and the package's
        revision manifest.
    @raise hydrocouple_platform.packaging.package.PackagingError: The model cannot be
        packaged (for example, it loads a native plugin).
    """
    from hydrocouple_platform.packaging import package_swmm_model

    package = package_swmm_model(
        env_config.inp_path, output_dir / "package", extra_files=sidecar_files(env_config) or None
    )
    files = package.manifest.model_dump(mode="json", by_alias=True)["files"]
    model = next(f["path"] for f in files if f["role"] == "model")
    return model_inputs(files), model, package.manifest
