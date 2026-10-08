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

"""Recrop geometry"""

from __future__ import annotations

import dataclasses

import numba
import numpy as np
import pycocotools.mask

import masker.annotations.instance_mask
from masker.annotations.instance_mask import decoded_mask_t, runs_t
from masker.common.common_defs import bbox_t, binary_mask_t, index_array_t, point_t, rect_t


@numba.njit(cache=True)
def _paste(out: decoded_mask_t, ri: index_array_t, rows: index_array_t,
           ci: index_array_t, cols: index_array_t, local: binary_mask_t) -> None:
    for i in range(ri.size):
        r = rows[i]
        for j in range(ci.size):
            out[ri[i], ci[j]] = local[r, cols[j]] != 0


@numba.njit(cache=True)
def _rect_counts(local: binary_mask_t, rows: index_array_t, cols: index_array_t,
                 x0: int, y0: int, full_h: int, n_pix: int) -> runs_t:
    """COCO counts of an upscaled local mask placed in the full frame"""
    sh, sw = rows.size, cols.size
    r0, r1 = rows[0], rows[-1] + 1
    c0, c1 = cols[0], cols[-1] + 1
    # frame rows per local row
    mult = np.zeros(r1 - r0, np.int64)
    for k in range(sh):
        mult[rows[k] - r0] += 1
    # each local column's runs over the frame rows, read once: values,
    # lengths and per-column offsets
    rv = np.empty((c1 - c0) * (r1 - r0), np.bool_)
    rm = np.empty((c1 - c0) * (r1 - r0), np.int64)
    off = np.zeros(c1 - c0 + 1, np.int64)
    n = 0
    for lc in range(c0, c1):
        first = n
        for r in range(r0, r1):
            v = local[r, lc] != 0
            if n > first and v == rv[n - 1]:
                rm[n - 1] += mult[r - r0]
            else:
                rv[n], rm[n] = v, mult[r - r0]
                n += 1
        off[lc - c0 + 1] = n
    counts = np.empty(2 + sw + (off[1:] - off[:-1])[cols - c0].sum(), np.int64)
    n = 0
    cur, length = False, x0 * full_h + y0
    # background to the next column's first frame row; the last column runs to the frame end
    gap = full_h - sh
    for x in range(sw):
        lc = cols[x] - c0
        if x + 1 == sw:
            gap = n_pix - (x0 + x) * full_h - y0 - sh
        for j in range(off[lc], off[lc + 1] + 1):
            v, m = (rv[j], rm[j]) if j < off[lc + 1] else (False, gap)
            if v == cur:
                length += m
            elif m:
                counts[n] = length
                n += 1
                cur, length = v, m
    counts[n] = length
    return counts[:n + 1]


@dataclasses.dataclass(frozen=True, slots=True)
class Recrop:
    """A recrop rect in original-image pixels plus its presented scale; build with from_rect"""

    # left edge, original pixels
    x0: int
    # top edge, original pixels
    y0: int
    # right edge (exclusive), original pixels
    x1: int
    # bottom edge (exclusive), original pixels
    y1: int
    # presented size / original size, <= 1
    scale: float
    # presented width
    out_w: int
    # presented height
    out_h: int
    # per-recrop (lh, lw) -> upscale-index cache; excluded from eq/repr
    _memo: dict[tuple[int, int], tuple[index_array_t, index_array_t]] = dataclasses.field(
        default_factory=dict, init=False, repr=False, compare=False)

    @classmethod
    def from_rect(cls, x0: int, y0: int, x1: int, y1: int, out_max_side: int) -> Recrop:
        """Recrop over a rect presented at out_max_side, never upscaled"""
        scale = min(1.0, out_max_side / max(x1 - x0, y1 - y0))
        out_w, out_h = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
        return cls(x0, y0, x1, y1, scale, out_w, out_h)

    @property
    def rect(self) -> rect_t:
        return (self.x0, self.y0, self.x1, self.y1)

    @property
    def scaled_size(self) -> tuple[int, int]:
        return (self.out_w, self.out_h)

    # --- coordinate mapping --------------------------------------------------

    def point_to_local_coords(self, x: float, y: float) -> point_t:
        return ((x - self.x0) * self.scale, (y - self.y0) * self.scale)

    def point_to_global_coords(self, x: float, y: float) -> point_t:
        return (x / self.scale + self.x0, y / self.scale + self.y0)

    def to_local_bbox(self, bbox_xyxy: bbox_t) -> bbox_t:
        x0, y0 = self.point_to_local_coords(bbox_xyxy[0], bbox_xyxy[1])
        x1, y1 = self.point_to_local_coords(bbox_xyxy[2], bbox_xyxy[3])
        return (x0, y0, x1, y1)

    def to_global_bbox(self, bbox_xyxy: bbox_t) -> bbox_t:
        x0, y0 = self.point_to_global_coords(bbox_xyxy[0], bbox_xyxy[1])
        x1, y1 = self.point_to_global_coords(bbox_xyxy[2], bbox_xyxy[3])
        return (x0, y0, x1, y1)

    # --- mask paste-back ------------------------------------------------------

    def _upscale_indices(self, lh: int, lw: int) -> tuple[index_array_t, index_array_t]:
        """Nearest-neighbor upscale indices from an lh x lw local mask to original pixels"""
        hit = self._memo.get((lh, lw))
        if hit is None:
            h, w = self.y1 - self.y0, self.x1 - self.x0
            rows, cols = (np.arange(h) * lh) // h, (np.arange(w) * lw) // w
            hit = self._memo[(lh, lw)] = (rows, cols)
        return hit

    def sample_original(self, local_mask: binary_mask_t, ys: index_array_t,
                        xs: index_array_t) -> decoded_mask_t:
        """Pasted-back local_mask at original rows ys x cols xs; uint8 0/1, zero outside the recrop."""
        out = np.zeros((ys.size, xs.size), dtype=np.uint8)
        ri = np.flatnonzero((ys >= self.y0) & (ys < self.y1))
        ci = np.flatnonzero((xs >= self.x0) & (xs < self.x1))
        rows, cols = self._upscale_indices(*local_mask.shape)
        _paste(out, ri, rows[ys[ri] - self.y0], ci, cols[xs[ci] - self.x0], local_mask)
        return out

    def to_instance_mask(
        self,
        local_mask: binary_mask_t,
        full_h: int,
        full_w: int,
        *,
        score: float | None = None,
        category_id: int | None = None,
    ) -> masker.annotations.instance_mask.InstanceMask:
        """Full-frame InstanceMask of a pasted-back local mask"""
        n_pix = full_h * full_w
        counts = np.array([n_pix], dtype=np.int64)
        fg_rows, fg_cols = (np.flatnonzero(local_mask.any(axis=1)),
                            np.flatnonzero(local_mask.any(axis=0)))
        if fg_rows.size:
            rows, cols = self._upscale_indices(*local_mask.shape)
            # original rows/cols landing on the foreground's local bbox, clipped to the image
            y0 = max(self.y0 + int(np.searchsorted(rows, fg_rows[0])), 0)
            y1 = min(self.y0 + int(np.searchsorted(rows, fg_rows[-1] + 1)), full_h)
            x0 = max(self.x0 + int(np.searchsorted(cols, fg_cols[0])), 0)
            x1 = min(self.x0 + int(np.searchsorted(cols, fg_cols[-1] + 1)), full_w)
            if y1 > y0 and x1 > x0:
                counts = _rect_counts(local_mask, rows[y0 - self.y0:y1 - self.y0],
                                      cols[x0 - self.x0:x1 - self.x0], x0, y0, full_h, n_pix)
        rle = pycocotools.mask.frPyObjects({"size": [full_h, full_w], "counts": counts}, full_h, full_w)
        return masker.annotations.instance_mask.InstanceMask(
            rle["counts"], full_h, full_w, score=score, category_id=category_id)


def _starts(length: int, size: int, step: int) -> list[int]:
    """Span start offsets along one axis"""
    if length <= size:
        return [0]
    starts = list(range(0, length - size, step))
    starts.append(length - size)
    return starts


def plan_grid(
    img_w: int,
    img_h: int,
    recrop_px: int,
    overlap_pct: float = 0.2,
    *,
    out_max_side: int | None = None,
) -> list[Recrop]:
    """Overlapping grid of recrops covering the whole img_w x img_h image.

    recrop_px is the recrop size in original pixels. Recrops step by
    recrop_px * (1 - overlap); the last row/column is clamped flush to the
    image edge. An image smaller than recrop_px yields a single recrop of the
    whole image. out_max_side sets each recrop's presented size (its
    longest side after scaling); None keeps the recrops unscaled.
    """
    recrop_w, recrop_h = min(recrop_px, img_w), min(recrop_px, img_h)
    step = max(1, round(recrop_px * (1.0 - overlap_pct)))
    target = out_max_side if out_max_side is not None else max(recrop_w, recrop_h)
    return [
        Recrop.from_rect(x, y, x + recrop_w, y + recrop_h, target)
        for y in _starts(img_h, recrop_h, step)
        for x in _starts(img_w, recrop_w, step)
    ]
