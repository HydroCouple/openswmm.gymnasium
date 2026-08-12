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

from openswmm_gymnasium._engine.solver_adapter import (
    _REQUIRED_CLASS_ATTRS,
    _REQUIRED_MODULE_ATTRS,
    EngineCapabilityError,
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


if __name__ == "__main__":
    unittest.main()
