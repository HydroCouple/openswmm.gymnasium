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

"""2D vertex-depth observation path + IGNORE_2D episode gating.

Per plan §8.0 — no engine mocks: drives the real engine over the
coupled 1D/2D ``tests/data/minimal_2d.inp`` fixture (two triangles,
four vertices, centre-vertex coupling to J1).

Covers: ``ObservationBuilder.add_2d_vertex_depths`` (space size,
bind-time validation, collect shape/dtype), ``SolverAdapter.surface2d``
gating (clear error on a 1D-only model), and the ``IGNORE_2D`` engine
option applied through ``SolverAdapter.set_option`` between open and
initialize (the per-episode 1D-only fast path).

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: Apache-2.0
"""

from __future__ import annotations

import shutil
import unittest
from pathlib import Path

import numpy as np

from openswmm_gymnasium.observations import ObservationBuilder

from ._base import BaseEngineTest

_DATA_DIR = (Path(__file__).resolve().parents[1] / "data").resolve()
_MINIMAL_2D_INP = _DATA_DIR / "minimal_2d.inp"

N_2D_VERTICES = 4


class BaseEngine2DTest(BaseEngineTest):
    """Adds a per-test copy of ``minimal_2d.inp``."""

    def setUp(self) -> None:
        super().setUp()
        self.minimal_2d_inp = self.tmp_path / "minimal_2d.inp"
        shutil.copy(_MINIMAL_2D_INP, self.minimal_2d_inp)


class TestSurface2DAdapter(BaseEngine2DTest):
    def test_surface2d_active_on_meshed_model(self) -> None:
        adapter = self.make_adapter(inp=self.minimal_2d_inp)
        surf = adapter.surface2d
        self.assertTrue(surf.is_active)
        self.assertEqual(surf.n_vertices, N_2D_VERTICES)
        depths = surf.vertex_render_depths()
        self.assertEqual(np.asarray(depths).shape, (N_2D_VERTICES,))

    def test_surface2d_raises_on_1d_only_model(self) -> None:
        adapter = self.make_adapter()  # minimal.inp — no [2D_*] sections
        with self.assertRaisesRegex(RuntimeError, "no active 2D surface"):
            _ = adapter.surface2d


class TestVertexDepthCollector(BaseEngine2DTest):
    def _builder(self, idxs) -> ObservationBuilder:
        return ObservationBuilder().add_2d_vertex_depths(idxs)

    def test_space_size(self) -> None:
        builder = self._builder([0, 2])
        self.assertEqual(builder.space().shape, (2,))

    def test_empty_indices_raise(self) -> None:
        with self.assertRaises(ValueError):
            self._builder([])

    def test_bind_and_collect(self) -> None:
        adapter = self.make_adapter(inp=self.minimal_2d_inp)
        builder = self._builder([0, 1, 3])
        builder.bind(adapter)
        obs = builder.collect(adapter)
        self.assertEqual(obs.shape, (3,))
        self.assertEqual(obs.dtype, np.float32)
        self.assertTrue(np.all(np.isfinite(obs)))

    def test_bind_rejects_out_of_range_vertex(self) -> None:
        adapter = self.make_adapter(inp=self.minimal_2d_inp)
        builder = self._builder([0, N_2D_VERTICES])
        with self.assertRaisesRegex(ValueError, "out of range"):
            builder.bind(adapter)

    def test_bind_fails_cleanly_on_1d_only_model(self) -> None:
        adapter = self.make_adapter()
        builder = self._builder([0])
        with self.assertRaisesRegex(RuntimeError, "no active 2D surface"):
            builder.bind(adapter)


class TestIgnore2DOption(BaseEngine2DTest):
    def test_ignore_2d_runs_meshed_model_1d_only(self) -> None:
        adapter = self.make_adapter(open=False, inp=self.minimal_2d_inp)
        adapter.open()
        adapter.set_option("IGNORE_2D", "YES")
        adapter.initialize()
        adapter.start()
        # The gated surface is inactive: 2D observations must refuse.
        with self.assertRaisesRegex(RuntimeError, "no active 2D surface"):
            _ = adapter.surface2d
        # And the 1D episode still steps.
        adapter.step()
        self.assertGreaterEqual(adapter.elapsed, 0.0)


if __name__ == "__main__":
    unittest.main()
