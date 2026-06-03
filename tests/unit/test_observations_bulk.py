"""P2.4 — bulk observation read path parity.

Collectors backed by an engine bulk getter now fetch a whole-network array
in one FFI call and gather, instead of N scalar round-trips. This test
asserts the bulk path is byte-for-byte equal to the scalar fallback for the
same engine state. Integration tier: real engine over C{minimal.inp}.

@author: Caleb Buahin
@copyright: Copyright (c) 2026 Caleb Buahin
@license: MIT
"""

from __future__ import annotations

import numpy as np

from tests.unit._base import BaseEngineTest

from openswmm_gymnasium.observations import builder as _b


class TestBulkScalarParity(BaseEngineTest):
    # (collector class, element ids) — each is bulk-backed.
    CASES = [
        ("_NodeDepthCollector", ["J1"]),
        ("_NodeHeadCollector", ["J1"]),
        ("_NodeInflowCollector", ["J1"]),
        ("_NodeOverflowCollector", ["J1"]),
        ("_LinkFlowCollector", ["C1"]),
        ("_LinkDepthCollector", ["C1"]),
    ]

    def test_bulk_equals_scalar(self):
        adapter = self.make_adapter(open=True)
        adapter.step()  # advance so there is non-trivial state
        for cls_name, ids in self.CASES:
            with self.subTest(collector=cls_name):
                collector = getattr(_b, cls_name)(ids)
                collector.bind(adapter)
                # Bulk path (what collect() now uses).
                bulk = collector.collect(adapter)
                # Scalar fallback computed directly.
                scalar = np.fromiter(
                    (collector._read_scalar(adapter, i) for i in collector._idxs),
                    dtype=np.float32,
                    count=len(collector._idxs),
                )
                np.testing.assert_array_equal(bulk, scalar)

    def test_bulk_array_present(self):
        adapter = self.make_adapter(open=True)
        c = _b._NodeDepthCollector(["J1"])
        c.bind(adapter)
        assert c._bulk_array(adapter) is not None
