# Copyright 2026 Jim Clampffer. Created 2026-10-07.
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

"""Reference impls for sweep's batched geometry"""

from __future__ import annotations

import numpy as np

from masker.common.common_defs import bbox_t
import masker.execution.sweep


def bbox_gap(a: bbox_t, b: bbox_t) -> float:
    """Separation between two xyxy bboxes: 0 when they touch or overlap,
    else the larger of the horizontal/vertical gaps."""
    return max(0.0, max(a[0], b[0]) - min(a[2], b[2]), max(a[1], b[1]) - min(a[3], b[3]))


def mask_gap(a: masker.execution.sweep.MaskSnapshot,
             b: masker.execution.sweep.MaskSnapshot) -> float:
    """All-pairs separation between two snapshot masks: the Chebyshev
    distance between the nearest true-cell centers, less half a cell per
    side; the bbox gap when either snapshot is empty. Production pairs only
    frontier cells; this scans every cell."""
    if a.cells.size == 0 or b.cells.size == 0:
        return bbox_gap(a.bbox, b.bbox)
    d = np.abs(a.points[:, None, :] - b.points[None, :, :]).max(axis=2).min()
    return max(0.0, float(d) - (a.cell + b.cell) / 2)


def mask_gap_le(a: masker.execution.sweep.MaskSnapshot,
                b: masker.execution.sweep.MaskSnapshot,
                threshold: float) -> bool:
    """Production's decision mask_gap(a, b) <= threshold."""
    return bool(masker.execution.sweep._mask_gaps_le(a, [b], threshold)[0])
