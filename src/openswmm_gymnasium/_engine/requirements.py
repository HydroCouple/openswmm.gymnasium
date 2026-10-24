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
What the installed engine and the open model must provide.

  - L{require_engine_capabilities} / L{require_for}: the engine surface a
    component calls, named by catalog path, checked when an env is built so
    a too-old or partially-built C{openswmm.engine} fails with a message
    naming the missing symbols rather than an C{AttributeError} deep inside
    a rollout.
  - L{optional_module}: the build- and model-dependent modules (2D, heat,
    water age, reactions) behind a two-tier guard: "does this build carry
    the module" and "does the B{open model} enable it".

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

import openswmm.engine as _engine
from openswmm.engine import catalog as _catalog

if TYPE_CHECKING:
    from openswmm_gymnasium._engine.solver_adapter import SolverAdapter


# ---------------------------------------------------------------------------
# Engine capability probe
# ---------------------------------------------------------------------------
#
# ``pyproject.toml`` pins ``openswmm>=6.0.0a4.dev2`` with no upper bound, and the
# C API is still moving. Rather than a hard version floor — which a partial
# build (e.g. ``OPENSWMM_BUILD_2D=OFF``) would satisfy while still lacking a
# symbol, and which a newer-but-compatible build would fail — this package
# checks the specific engine surface it calls, named by catalog path
# (``"stride"``, ``"tables"``, ``"link.target_setting"``).
#
# Only the adapter's own core (CORE_REQUIREMENTS) is checked for every env.
# Observation collectors, action factories and reward terms declare what they
# need in a ``requires`` attribute, and each env checks exactly the components
# it is configured with (L{require_for}), so a partial build fails only the
# envs that need the missing piece. The optional heat / water-age / reaction /
# 2D modules are additionally reached through L{optional_module}, which also
# checks that the open *model* enables the module — something no catalog can
# know.

#: Engine catalog revision this package is written against
#: (L{openswmm.engine.catalog.revision}); older catalogs lack its paths.
MIN_CATALOG_REVISION = 2

#: Solver members the adapter itself calls, needed by every env.
CORE_REQUIREMENTS: tuple[str, ...] = (
    "flow_units",
    "unit_system",
    "set_lenient_open",
    "open_errors",
    "open_warnings",
    "stride",
)


class EngineCapabilityError(RuntimeError):
    """Raised when the installed C{openswmm.engine} lacks a required symbol.

    Names every missing symbol so the failure is actionable instead of
    surfacing as an C{AttributeError} mid-rollout. Plan §4.5 / G5.
    """


def _engine_class(entry: dict, module) -> type | None:
    """The runtime class behind catalog target C{entry}, or C{None} when absent."""
    if module is not None:
        return getattr(module, entry["class"], None)
    cls = getattr(_engine, entry["class"], None)
    if cls is None:
        try:
            cls = getattr(
                importlib.import_module(f"openswmm.engine.{entry['module']}"), entry["class"], None
            )
        except ImportError:
            return None
    return cls


def _missing_engine_capabilities(module=None, paths=CORE_REQUIREMENTS) -> list[str]:
    """Return the catalog paths in C{paths} the engine lacks, with the symbol missing.

    A path is a catalog member (C{"stride"}, C{"link.target_setting"}) or a
    target (C{"tables"}, C{"xsect"}). C{module} substitutes a stand-in engine
    namespace so the probe can be tested without a build.

    @param module: Engine namespace; defaults to C{openswmm.engine}.
    @param paths: Catalog paths to check.
    @rtype: list[str]
    """
    targets = _catalog.targets()
    missing: list[str] = []
    for path in sorted(set(paths)):
        if path in targets:
            entry, attr = targets[path], None
        else:
            member = _catalog.lookup(path)
            entry, attr = targets[member["target"]], member["name"]
        cls = _engine_class(entry, module)
        if cls is None:
            missing.append(f"{path} (openswmm.engine.{entry['class']})")
        elif attr is not None and not hasattr(cls, attr):
            missing.append(f"{path} (openswmm.engine.{entry['class']}.{attr})")
    return missing


def require_engine_capabilities(module=None, paths=CORE_REQUIREMENTS) -> None:
    """Verify the installed engine exposes every catalog path in C{paths}.

    Called once per L{SolverAdapter} construction for the core, and by each
    env for its configured components (L{require_for}).

    @param module: Namespace to probe. Defaults to C{openswmm.engine}.
    @param paths: Catalog paths; defaults to L{CORE_REQUIREMENTS}.
    @raise EngineCapabilityError: If any required symbol is absent.
    """
    version = getattr(module if module is not None else _engine, "__version__", "unknown")
    revision = getattr(_catalog, "revision", lambda: 1)()
    if module is None and revision < MIN_CATALOG_REVISION:
        raise EngineCapabilityError(
            f"openswmm.gymnasium needs engine catalog revision {MIN_CATALOG_REVISION} or newer; "
            f"the installed openswmm.engine (version {version}) has revision {revision}. "
            "Upgrade or rebuild the openswmm package."
        )
    missing = _missing_engine_capabilities(module, paths)
    if missing:
        raise EngineCapabilityError(
            "The installed openswmm.engine (version "
            f"{version}) is missing "
            f"{len(missing)} symbol(s) openswmm.gymnasium requires: "
            + ", ".join(missing)
            + ". Upgrade the openswmm package, or rebuild the engine with the "
            "corresponding modules enabled."
        )


def require_for(*components, module=None) -> None:
    """Check the engine surface every component declares in its C{requires}.

    Envs call this with their observation builder, action factories and
    reward terms, so a build missing an optional piece fails only the envs
    configured to use it.

    @param components: Objects with an optional C{requires} iterable of catalog
        paths (or L{ObservationBuilder}-like objects exposing C{requires()}).
    @raise EngineCapabilityError: If any declared path is missing.
    """
    paths: set[str] = set()
    for c in components:
        req = getattr(c, "requires", ())
        paths.update(req() if callable(req) else req)
    if paths:
        require_engine_capabilities(module, tuple(paths))


# ---------------------------------------------------------------------------
# Optional engine modules (two-tier guard)
# ---------------------------------------------------------------------------

#: Module -> (engine class probed for the build tier, or C{None} to rely on the
#: solver accessor's C{ImportError}; flag the open model must set, or C{None};
#: build-tier message; model-tier message).
_OPTIONAL_MODULES: dict[str, tuple[str | None, str | None, str, str]] = {
    "surface2d": (
        None,
        "is_active",
        "This openswmm.engine build has no 2D module (OPENSWMM_BUILD_2D=OFF); "
        "2D observations are unavailable.",
        "The open model has no active 2D surface (no [2D_*] sections, or IGNORE_2D is "
        "enabled); remove the 2D collectors or run with the 2D module on.",
    ),
    "heat": (
        "Heat",
        "enabled",
        "This openswmm.engine build has no Heat module (openswmm.engine.Heat is absent); "
        "heat-transport factories and actuators are unavailable. Upgrade the openswmm "
        "package, or rebuild the engine with the heat component enabled.",
        "The open model does not run heat transport: its [OPTIONS] section has no "
        "'HEAT_TRANSPORT YES'. Heat source temperatures would be stored and never routed, "
        "so this accessor refuses rather than returning defaults. Enable HEAT_TRANSPORT in "
        "the .inp, or remove the heat factories/actuators from the config.",
    ),
    "water_age": (
        "WaterAge",
        "enabled",
        "This openswmm.engine build has no WaterAge module (openswmm.engine.WaterAge is "
        "absent); water-age factories are unavailable. Upgrade the openswmm package, or "
        "rebuild the engine with the water-age component enabled.",
        "The open model does not run water age: its [OPTIONS] section has no 'WATER_AGE "
        "YES'. Configured source ages would be stored and never transported, so this "
        "accessor refuses rather than returning defaults. Enable WATER_AGE in the .inp, or "
        "remove the water-age factories from the config.",
    ),
    "reactions": (
        "Reactions",
        None,
        "This openswmm.engine build has no Reactions module (openswmm.engine.Reactions is "
        "absent); reaction coefficient search is unavailable. Upgrade the openswmm "
        "package, or rebuild the engine with the multi-species reaction component enabled.",
        "",
    ),
}


def optional_module(adapter: SolverAdapter, name: str) -> Any:
    """The live engine module C{name} (C{adapter.solver.<name>}), after a two-tier guard.

    C{name} is C{"surface2d"}, C{"heat"}, C{"water_age"} or C{"reactions"}.
    Components call this at C{bind} and then use the real engine API.
    B{Build tier}: an engine without the module raises naming the remedy.
    B{Model tier}: the open model must enable it (an active 2D surface,
    C{HEAT_TRANSPORT YES}, C{WATER_AGE YES}). A heat or water-age
    configuration written into a model that does not run that process is
    stored and never routed — a B{silent} no-op no reward signal can reveal —
    so it is refused at bind rather than trained against. Reactions have no
    C{[OPTIONS]} flag and an empty coefficient table is legitimate, so only
    the build tier applies; consumers check the coefficients they name.

    @param adapter: Adapter wrapping the open solver.
    @type adapter: L{SolverAdapter}
    @param name: Optional module name.
    @type name: str
    @raise RuntimeError: If the engine build lacks the module, or the open
        model does not enable it.
    """
    cls_name, flag, no_build, no_model = _OPTIONAL_MODULES[name]
    if cls_name is not None and getattr(_engine, cls_name, None) is None:
        raise RuntimeError(no_build)
    try:
        module = getattr(adapter.solver, name)
    except ImportError as exc:
        raise RuntimeError(no_build) from exc
    if flag is not None and not getattr(module, flag):
        raise RuntimeError(no_model)
    return module
