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

"""G5.2 — the engine capability probe.

C{pyproject.toml} pins C{openswmm>=6.0.0.dev2} with no upper bound and the
C API is still moving, so the adapter probes for the specific symbols it
calls instead of asserting a version. The probe is a pure function over a
namespace object, which is what these tests drive — no engine required.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import unittest
from types import SimpleNamespace

from openswmm_gymnasium._engine import solver_adapter
from openswmm_gymnasium._engine.solver_adapter import (
    _REQUIRED_CLASS_ATTRS,
    _REQUIRED_MODULE_ATTRS,
    EngineCapabilityError,
    SolverAdapter,
    _missing_engine_capabilities,
    require_engine_capabilities,
)


def _complete_engine() -> SimpleNamespace:
    """A namespace satisfying every probed requirement."""
    classes: dict[str, type] = {}
    for cls_name, attr in _REQUIRED_CLASS_ATTRS:
        cls = classes.setdefault(cls_name, type(cls_name, (), {}))
        setattr(cls, attr, None)
    module = SimpleNamespace(__version__="6.0.0.test", **classes)
    for name in _REQUIRED_MODULE_ATTRS:
        setattr(module, name, object())
    return module


class TestProbe(unittest.TestCase):
    def test_complete_engine_reports_nothing_missing(self):
        self.assertEqual(_missing_engine_capabilities(_complete_engine()), [])

    def test_complete_engine_passes(self):
        require_engine_capabilities(_complete_engine())

    def test_missing_module_symbol_is_named(self):
        module = _complete_engine()
        delattr(module, "XSectionGeometry")
        self.assertEqual(
            _missing_engine_capabilities(module), ["openswmm.engine.XSectionGeometry"]
        )

    def test_missing_class_attribute_is_named(self):
        module = _complete_engine()
        delattr(module.Solver, "set_lenient_open")
        self.assertEqual(
            _missing_engine_capabilities(module),
            ["openswmm.engine.Solver.set_lenient_open"],
        )

    def test_missing_class_entirely_is_named(self):
        module = _complete_engine()
        delattr(module, "Nodes")
        self.assertIn("openswmm.engine.Nodes", _missing_engine_capabilities(module))

    def test_error_names_every_missing_symbol(self):
        module = _complete_engine()
        delattr(module, "Statistics")
        delattr(module, "Pollutants")
        with self.assertRaises(EngineCapabilityError) as ctx:
            require_engine_capabilities(module)
        message = str(ctx.exception)
        self.assertIn("openswmm.engine.Statistics", message)
        self.assertIn("openswmm.engine.Pollutants", message)

    def test_error_reports_the_installed_version(self):
        module = _complete_engine()
        delattr(module, "Tables")
        with self.assertRaises(EngineCapabilityError) as ctx:
            require_engine_capabilities(module)
        self.assertIn("6.0.0.test", str(ctx.exception))

    def test_error_is_actionable(self):
        module = _complete_engine()
        delattr(module, "Tables")
        with self.assertRaises(EngineCapabilityError) as ctx:
            require_engine_capabilities(module)
        self.assertIn("Upgrade", str(ctx.exception))

    def test_optional_2d_surface_is_not_probed(self):
        """A build with OPENSWMM_BUILD_2D=OFF must still pass the probe."""
        module = _complete_engine()
        self.assertFalse(hasattr(module, "Surface2D"))
        require_engine_capabilities(module)


class TestOptionalSurfacesAreNotProbed(unittest.TestCase):
    """The heat / water-age / reaction modules stay out of the hard probe.

    Listing an optional module in C{_REQUIRED_MODULE_ATTRS} would make a
    partial build unusable for B{every} env rather than only for the envs
    that actually touch it — a 1D hydraulic RTC task has no business
    failing because the engine was compiled without heat transport. Each
    gets an accessor-level probe on L{SolverAdapter} instead, so the
    failure lands at the moment something reaches for the missing module,
    with the remedy named.
    """

    #: Optional engine modules reached through a guarded L{SolverAdapter}
    #: accessor rather than the construction-time probe.
    OPTIONAL_MODULES = ("Surface2D", "Heat", "WaterAge", "Reactions")

    def test_probe_passes_without_any_optional_module(self):
        module = _complete_engine()
        for name in self.OPTIONAL_MODULES:
            self.assertFalse(hasattr(module, name), name)
        require_engine_capabilities(module)
        self.assertEqual(_missing_engine_capabilities(module), [])

    def test_no_optional_module_is_listed_as_required(self):
        for name in self.OPTIONAL_MODULES:
            with self.subTest(module=name):
                self.assertNotIn(name, _REQUIRED_MODULE_ATTRS)

    def test_no_optional_module_is_listed_as_a_required_class(self):
        required_classes = {cls_name for cls_name, _ in _REQUIRED_CLASS_ATTRS}
        for name in self.OPTIONAL_MODULES:
            with self.subTest(module=name):
                self.assertNotIn(name, required_classes)

    def test_probe_still_passes_when_optional_modules_are_present(self):
        """Present-but-unprobed must be just as acceptable as absent."""
        module = _complete_engine()
        for name in self.OPTIONAL_MODULES:
            setattr(module, name, object())
        require_engine_capabilities(module)


class TestOptionalAccessorProbes(unittest.TestCase):
    """Each optional accessor names its own missing module, and the remedy.

    Drives the guards against a stand-in solver so no engine build is
    needed. Only the B{build} tier is exercised here; the B{model} tier
    (C{[OPTIONS] HEAT_TRANSPORT} / C{WATER_AGE} off) needs a real open
    model and lives in C{test_process_surfaces.py}.
    """

    def _adapter_over(self, module) -> SolverAdapter:
        """Build an adapter whose engine namespace is *module*.

        Bypasses C{__init__} so no real solver is constructed: the build-tier
        probes read only C{_engine} and the cached-accessor slots.
        """
        adapter = SolverAdapter.__new__(SolverAdapter)
        adapter._solver = SimpleNamespace()
        adapter._heat = None
        adapter._water_age = None
        adapter._reactions = None
        return adapter

    def _patch_engine(self, module) -> None:
        original = solver_adapter._engine
        solver_adapter._engine = module
        self.addCleanup(setattr, solver_adapter, "_engine", original)

    def test_heat_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).heat
        message = str(ctx.exception)
        self.assertIn("Heat", message)
        self.assertIn("rebuild", message.lower())

    def test_water_age_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).water_age
        message = str(ctx.exception)
        self.assertIn("WaterAge", message)
        self.assertIn("rebuild", message.lower())

    def test_reactions_accessor_names_the_missing_module(self):
        self._patch_engine(_complete_engine())
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).reactions
        message = str(ctx.exception)
        self.assertIn("Reactions", message)
        self.assertIn("rebuild", message.lower())

    def test_heat_accessor_refuses_a_model_with_heat_transport_off(self):
        """Build tier passes, model tier does not: the module exists but
        the open model never routes heat, so writes would be silent."""
        module = _complete_engine()
        module.Heat = lambda solver: SimpleNamespace(enabled=False)
        self._patch_engine(module)
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).heat
        self.assertIn("HEAT_TRANSPORT", str(ctx.exception))

    def test_water_age_accessor_refuses_a_model_with_water_age_off(self):
        module = _complete_engine()
        module.WaterAge = lambda solver: SimpleNamespace(enabled=False)
        self._patch_engine(module)
        with self.assertRaises(RuntimeError) as ctx:
            _ = self._adapter_over(None).water_age
        self.assertIn("WATER_AGE", str(ctx.exception))

    def test_heat_accessor_is_cached_once_both_tiers_pass(self):
        module = _complete_engine()
        constructions = []

        def _heat(solver):
            constructions.append(solver)
            return SimpleNamespace(enabled=True)

        module.Heat = _heat
        self._patch_engine(module)
        adapter = self._adapter_over(None)
        first = adapter.heat
        self.assertIs(adapter.heat, first)
        self.assertEqual(len(constructions), 1)

    def test_reactions_has_no_model_tier_gate(self):
        """There is no [OPTIONS] flag for reactions, so an empty
        coefficient table must not be treated as a misconfiguration."""
        module = _complete_engine()
        module.Reactions = lambda solver: SimpleNamespace(
            species=[], coefficients=[], terms=[]
        )
        self._patch_engine(module)
        self.assertEqual(self._adapter_over(None).reactions.coefficient_names(), [])


if __name__ == "__main__":
    unittest.main()
