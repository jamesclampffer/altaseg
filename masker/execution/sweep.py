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

"""Most of the logic"""

from __future__ import annotations

import collections.abc
import dataclasses
import enum
import functools
import logging
import math
import pathlib
import time

import numba
import numpy as np
import PIL.Image
import torch
import torch.nn.functional

import masker.annotations.instance_mask
from masker.common.common_defs import bbox_t, bool_mask_t, boxes_t, index_array_t, point_t, rect_t
import masker.common.util
import masker.execution.model
import masker.execution.recrop

logger = logging.getLogger(__name__)

# public event: "type" key plus payload
type sweep_event_t = dict[str, object]
# phase event: tag then payload
type phase_event_t = tuple[str, *tuple[object, ...]]
# (gh, gw) any-pooled snapshot cells
type cell_grid_t = np.typing.NDArray[np.bool_]
# (n, 2) xy points, original coords
type point_array_t = np.typing.NDArray[np.float64]
# (n, 4) xyxy bboxes, original coords
type bbox_array_t = np.typing.NDArray[np.float64]
# per-element flags
type flags_t = np.typing.NDArray[np.bool_]
# per-element distances, px
type distances_t = np.typing.NDArray[np.float64]
# per-point neighbor counts
type neighbor_counts_t = np.typing.NDArray[np.int64]


class SweepConsts:
    # SAM3 input side
    MODEL_SIDE = 1008
    # Don't expand patches smaller than 256x256 for fwd pass
    MIN_RECROP = 256
    # trial and error
    CLUSTER_MAX_FRAC = 3.0
    # if packing crops into the 2nd pass refinement
    PACK_FACTORS = (2, 3)
    # exemplar bbox exclusion from edges
    EXEMPLAR_EDGE_MARGIN = 0.05
    # Pass-3 variant recrop-center offsets as fractions of the side
    VARIANT_OFFSETS = ((0.0, 0.0), (0.25, 0.0), (0.0, 0.25), (-0.25, 0.0),
                       (0.0, -0.25), (0.25, 0.25), (-0.25, -0.25),
                       (0.25, -0.25), (-0.25, 0.25))
    # Band-mode recrop sizing must still cover the cluster bbox by this margin.
    BAND_COVER_MARGIN = 1.2
    # Merge-blob gate: a kept mask beyond this multiple of the cluster's
    # typical object area
    MERGE_AREA_FACTOR = 4.0
    # margin in caled global px to from the edge to assume a truncation
    # todo: would likely be better scaled
    NOVEL_EDGE_MARGIN = 2
    # xy dims of any-pooled grids kept per instance for clustering and
    # duplicate gating.
    SNAPSHOT_CELLS = 32
    # Pass-3 per-retry recrop rescale; < 1 zooms in.
    EXEMPLAR_BACKOFF_SCALE = 1.2
    CONFIG_PATH = pathlib.Path(__file__).with_name("config.json")


@dataclasses.dataclass(frozen=True, slots=True)
class SweepParams:
    # config.json keys; a request overrides any subset
    downscale: float
    overlap: float
    pack: int | None
    reprompt_expand: float
    ios_gate: bool
    band_lo: float | None
    band_hi: float | None
    backoff_retries: int
    backoff_scale: float
    dup_ios: float
    dup_iou: float
    novel: bool
    novel_gap_px: float | None
    centroid: str
    reprompt_exemplars: int
    exemplar: bool
    exemplar_frac: float
    exemplar_max: int
    exemplar_backoff: int
    exemplar_scales: list[float]
    exemplar_variants: int
    exemplar_cover: float

    def __post_init__(self) -> None:
        lo, hi = self.band_lo, self.band_hi
        bad = [name for name, ok in (
            ("downscale", self.downscale > 0),
            ("overlap", 0 <= self.overlap < 1),
            ("pack", self.pack in (None, *SweepConsts.PACK_FACTORS)),
            ("reprompt_expand", self.reprompt_expand >= 1),
            ("band_lo/band_hi", (lo is None and hi is None)
             or (lo is not None and hi is not None and 0 < lo < hi <= 1)),
            ("backoff_retries", self.backoff_retries >= 0),
            ("backoff_scale", self.backoff_scale > 0),
            ("dup_ios", 0 < self.dup_ios <= 1),
            ("dup_iou", 0 < self.dup_iou <= 1),
            ("novel_gap_px", self.novel_gap_px is None or self.novel_gap_px >= 0),
            ("centroid", self.centroid in ("mask", "box")),
            ("reprompt_exemplars", self.reprompt_exemplars >= 0),
            ("exemplar_frac", 0 < self.exemplar_frac < 1),
            ("exemplar_max", self.exemplar_max >= 1),
            ("exemplar_backoff", self.exemplar_backoff >= 0),
            ("exemplar_scales", bool(self.exemplar_scales)
             and all(s > 0 for s in self.exemplar_scales)),
            ("exemplar_variants",
             1 <= self.exemplar_variants <= len(SweepConsts.VARIANT_OFFSETS)),
            ("exemplar_cover", 0 < self.exemplar_cover <= 1),
        ) if not ok]
        if bad:
            raise ValueError("bad sweep params: {}".format(", ".join(bad)))

    @classmethod
    def resolve(cls, **overrides: object) -> SweepParams:
        return cls(**{**masker.common.util.load_table(SweepConsts.CONFIG_PATH), **overrides})

    def recrop_px(self, model_side: int) -> int:
        return max(1, round(model_side * self.downscale))

    def grid_step(self, model_side: int) -> int:
        return round(self.recrop_px(model_side) * (1.0 - self.overlap))

    @property
    def band(self) -> tuple[float, float] | None:
        return None if self.band_lo is None else (self.band_lo, self.band_hi)


@numba.njit(cache=True)
def _cell_centers(mask: cell_grid_t, bbox: bbox_t) -> point_array_t:
    gh, gw = mask.shape
    x0, y0, x1, y1 = bbox
    cw, ch = (x1 - x0) / gw, (y1 - y0) / gh
    ys, xs = np.nonzero(mask)
    return np.stack((x0 + (xs + 0.5) * cw, y0 + (ys + 0.5) * ch), axis=1)


@numba.njit(cache=True)
def _frontier_cells(cells: cell_grid_t) -> cell_grid_t:
    """True cells with a false or off-grid 4-neighbor."""
    gh, gw = cells.shape
    padded = np.zeros((gh + 2, gw + 2), dtype=np.bool_)
    padded[1:-1, 1:-1] = cells
    interior = (padded[:-2, 1:-1] & padded[2:, 1:-1]
                & padded[1:-1, :-2] & padded[1:-1, 2:])
    return cells & ~interior


@numba.njit(cache=True)
def _grid_hits(cells: cell_grid_t, bbox: bbox_t, pts: point_array_t) -> flags_t:
    """Per point of pts, whether it lands on a true cell"""
    gh, gw = cells.shape
    x0, y0, x1, y1 = bbox
    ix = np.floor((pts[:, 0] - x0) / max(x1 - x0, 1e-9) * gw).astype(np.int64)
    iy = np.floor((pts[:, 1] - y0) / max(y1 - y0, 1e-9) * gh).astype(np.int64)
    inside = (0 <= ix) & (ix < gw) & (0 <= iy) & (iy < gh)
    hit = inside.copy()
    hit[inside] = cells[iy[inside], ix[inside]]
    return hit


@numba.njit(cache=True)
def _snapshot_cells(local: bool_mask_t, gh: int, gw: int) -> cell_grid_t:
    mh, mw = local.shape
    cells = np.zeros((gh, gw), dtype=np.bool_)
    yy, xx = np.nonzero(local)
    cells[(yy * gh) // mh, (xx * gw) // mw] = True
    return cells


@dataclasses.dataclass(frozen=True)
class MaskSnapshot:
    # tight bbox in original coords
    bbox: bbox_t
    # bool (gh, gw)
    cells: cell_grid_t
    # detector confidence
    score: float = 0.0

    @functools.cached_property
    def points(self) -> point_array_t:
        """True-cell centers in original coords, (N, 2)."""
        if self.cells.size == 0:
            return np.zeros((0, 2))
        return _cell_centers(self.cells, self.bbox)

    @functools.cached_property
    def cell(self) -> float:
        """Cell size along the long axis."""
        if self.cells.size == 0:
            return 0.0
        gh, gw = self.cells.shape
        x0, y0, x1, y1 = self.bbox
        return max((x1 - x0) / gw, (y1 - y0) / gh)

    @functools.cached_property
    def cell_area(self) -> float:
        """Anisotropic when the bbox is not square. This is a mess."""
        if self.cells.size == 0:
            return 0.0
        gh, gw = self.cells.shape
        x0, y0, x1, y1 = self.bbox
        return (x1 - x0) / gw * ((y1 - y0) / gh)

    @functools.cached_property
    def centroid(self) -> point_t:
        if self.cells.size == 0 or not self.cells.any():
            return ((self.bbox[0] + self.bbox[2]) / 2,
                    (self.bbox[1] + self.bbox[3]) / 2)
        cx, cy = self.points.mean(axis=0)
        return (float(cx), float(cy))

    @functools.cached_property
    def frontier(self) -> point_array_t:
        """Centers of true cells with a false or off-grid 4-neighbor, (M, 2)."""
        if self.cells.size == 0:
            return np.zeros((0, 2))
        return _cell_centers(_frontier_cells(self.cells), self.bbox)


def snapshot_from_local(inst: masker.execution.model.ObjectInstance,
                        recrop: masker.execution.recrop.Recrop,
                        grid: int = SweepConsts.SNAPSHOT_CELLS) -> MaskSnapshot:
    """Snapshot a recrop-local instance, bbox mapped through recrop."""
    bbox = tuple(float(v) for v in recrop.to_global_bbox(inst.bbox_xyxy))
    h, w = inst.mask.shape
    x0, y0, x1, y1 = (int(round(v)) for v in inst.bbox_xyxy)
    # clamp to the mask before slicing: edge boxes can round outside it
    x0, x1 = max(0, min(x0, w)), max(0, min(x1, w))
    y0, y1 = max(0, min(y0, h)), max(0, min(y1, h))
    x1, y1 = max(x1, x0 + 1), max(y1, y0 + 1)  # sub-pixel box: keep >= 1 cell
    local = inst.mask[y0:y1, x0:x1]
    score = float(inst.score)
    if local.size == 0 or not local.any():
        return MaskSnapshot(bbox, np.zeros((0, 0), dtype=bool), score)
    mh, mw = local.shape
    return MaskSnapshot(bbox, _snapshot_cells(local, min(grid, mh), min(grid, mw)), score)


@numba.njit(cache=True)
def _any_hit(ac: cell_grid_t, ab: bbox_t, bc: cell_grid_t, bb: bbox_t) -> bool:
    """Whether any cell center of b lands on a true cell of a"""
    agh, agw = ac.shape
    bgh, bgw = bc.shape
    ax0, ay0, ax1, ay1 = ab
    bx0, by0, bx1, by1 = bb
    bcw, bch = (bx1 - bx0) / bgw, (by1 - by0) / bgh
    cols = np.empty(bgw, np.int64)
    for x in range(bgw):
        cols[x] = math.floor((bx0 + (x + 0.5) * bcw - ax0) / max(ax1 - ax0, 1e-9) * agw)
    for y in range(bgh):
        iy = math.floor((by0 + (y + 0.5) * bch - ay0) / max(ay1 - ay0, 1e-9) * agh)
        if 0 <= iy < agh:
            for x in range(bgw):
                ix = cols[x]
                if 0 <= ix < agw and bc[y, x] and ac[iy, ix]:
                    return True
    return False


@numba.njit(cache=True)
def _within(px: float, py: float, box: bbox_t, reach: float) -> bool:
    """Whether (px, py) lies within Chebyshev reach of box."""
    x0, y0, x1, y1 = box
    return max(x0 - px, px - x1, y0 - py, py - y1) <= reach


@numba.njit(cache=True)
def _mask_gap_le(ac: cell_grid_t, ab: bbox_t, af: point_array_t, a_cell: float,
                 bc: cell_grid_t, bb: bbox_t, bf: point_array_t, b_cell: float,
                 threshold: float) -> bool:
    """Snapshot gap <= threshold; mask-based when both have cells, else bbox gap"""
    ax0, ay0, ax1, ay1 = ab
    bx0, by0, bx1, by1 = bb
    gap = max(0.0, max(ax0, bx0) - min(ax1, bx1), max(ay0, by0) - min(ay1, by1))
    if ac.size == 0 or bc.size == 0:
        return gap <= threshold
    if gap == 0.0 and (_any_hit(ac, ab, bc, bb) or _any_hit(bc, bb, ac, ab)):
        return True
    slack = (a_cell + b_cell) / 2
    # loose bound for the prefilters; the exact test below decides
    reach = (threshold + slack) * (1.0 + 1e-12)
    if gap > reach:
        return False
    keep = np.empty(bf.shape[0], np.int64)
    nb = 0
    for j in range(bf.shape[0]):
        if _within(bf[j, 0], bf[j, 1], ab, reach):
            keep[nb] = j
            nb += 1
    for i in range(af.shape[0]):
        if not _within(af[i, 0], af[i, 1], bb, reach):
            continue
        for k in range(nb):
            j = keep[k]
            d = max(abs(af[i, 0] - bf[j, 0]), abs(af[i, 1] - bf[j, 1]))
            if max(0.0, d - slack) <= threshold:
                return True
    return False


def _mask_gaps_le(a: MaskSnapshot, snaps: list[MaskSnapshot],
                  threshold: float) -> flags_t:
    out = np.empty(len(snaps), dtype=bool)
    for i, s in enumerate(snaps):
        out[i] = _mask_gap_le(a.cells, a.bbox, a.frontier, a.cell,
                              s.cells, s.bbox, s.frontier, s.cell, threshold)
    return out


def _snapshot_area(s: MaskSnapshot) -> float:
    """Approximate mask area in original pixels."""
    return len(s.points) * s.cell_area


def _mask_intersect_area(a: MaskSnapshot, b: MaskSnapshot) -> tuple[float, float, float]:
    hits = int(_grid_hits(a.cells, a.bbox, b.points).sum())
    return hits * b.cell_area, _snapshot_area(a), _snapshot_area(b)


def _union_cover(snap: MaskSnapshot, others: list[MaskSnapshot]) -> float:
    """union others, then calculate pct of snapshot covered by union"""
    pts = snap.points
    if len(pts) == 0:
        return 0.0
    covered = np.zeros(len(pts), dtype=bool)
    for o in others:
        ox0, oy0, ox1, oy1 = o.bbox
        if o.cells.size == 0:
            covered |= ((pts[:, 0] >= ox0) & (pts[:, 0] < ox1)
                        & (pts[:, 1] >= oy0) & (pts[:, 1] < oy1))
            continue
        covered |= _grid_hits(o.cells, o.bbox, pts)
        if covered.all():
            break
    return float(covered.mean())


def core_rects(recrops: list[masker.execution.recrop.Recrop]
               ) -> list[rect_t]:
    """Per recrop on first pass grid. Good detections."""
    r = np.asarray([w.rect for w in recrops], dtype=np.int64)
    x0, y0, x1, y1 = r[:, 0], r[:, 1], r[:, 2], r[:, 3]
    # (i, j): recrop j overlaps recrop i
    overlap = ((x0[None, :] < x1[:, None]) & (x1[None, :] > x0[:, None])
               & (y0[None, :] < y1[:, None]) & (y1[None, :] > y0[:, None]))
    np.fill_diagonal(overlap, False)
    lo = np.iinfo(np.int64).min
    hi = np.iinfo(np.int64).max
    cuts_l = overlap & (x0[None, :] < x0[:, None]) & (x0[:, None] < x1[None, :])
    cuts_r = overlap & (x0[None, :] < x1[:, None]) & (x1[:, None] < x1[None, :])
    cuts_t = overlap & (y0[None, :] < y0[:, None]) & (y0[:, None] < y1[None, :])
    cuts_b = overlap & (y0[None, :] < y1[:, None]) & (y1[:, None] < y1[None, :])
    cx0 = np.maximum(x0, np.where(cuts_l, x1[None, :], lo).max(axis=1))
    cx1 = np.minimum(x1, np.where(cuts_r, x0[None, :], hi).min(axis=1))
    cy0 = np.maximum(y0, np.where(cuts_t, y1[None, :], lo).max(axis=1))
    cy1 = np.minimum(y1, np.where(cuts_b, y0[None, :], hi).min(axis=1))
    return [(int(a), int(b), int(c), int(d))
            for a, b, c, d in zip(cx0, cy0, cx1, cy1)]


def is_unambiguous(bbox: bbox_t,
                   core: rect_t, img_w: int, img_h: int) -> bool:
    """Prune rule: bbox strictly inside the core"""
    x0, y0, x1, y1 = bbox
    cx0, cy0, cx1, cy1 = core
    return (
        (x0 > cx0 or cx0 <= 0)
        and (y0 > cy0 or cy0 <= 0)
        and (x1 < cx1 or cx1 >= img_w)
        and (y1 < cy1 or cy1 >= img_h)
    )


@numba.njit(cache=True)
def bbox_gaps(a: bbox_t, bb: bbox_array_t) -> distances_t:
    """use mask overlap instead"""
    gx = np.maximum(a[0], bb[:, 0]) - np.minimum(a[2], bb[:, 2])
    gy = np.maximum(a[1], bb[:, 1]) - np.minimum(a[3], bb[:, 3])
    return np.maximum(0.0, np.maximum(gx, gy))


def bbox_iou(a: bbox_t, b: bbox_t) -> float:
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def bbox_cover(a: bbox_t,
               b: bbox_t) -> tuple[float, float]:
    """Directional bbox containment."""
    ix, iy = min(a[2], b[2]) - max(a[0], b[0]), min(a[3], b[3]) - max(a[1], b[1])
    if ix <= 0 or iy <= 0:
        return 0.0, 0.0
    inter = ix * iy
    area_a, area_b = (a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])
    return inter / area_a, inter / area_b


def _union(a: bbox_t,
           b: bbox_t) -> bbox_t:
    return (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))


class _BboxGrid:
    """Uniform-grid index over (n, 4) xyxy bboxes."""

    __slots__ = 'cell', 'ox', 'oy', 'count', 'bins'
    cell: float
    ox: float
    oy: float
    # registered bboxes, construction plus add()
    count: int
    # (gx, gy) -> registered indices
    bins: dict[tuple[int, int], list[int]]

    def __init__(self, bboxes: bbox_array_t, cell: float):
        self.cell = cell
        self.ox = float(bboxes[:, 0].min()) if len(bboxes) else 0.0
        self.oy = float(bboxes[:, 1].min()) if len(bboxes) else 0.0
        self.count = len(bboxes)
        self.bins = {}
        for i, (x0, y0, x1, y1) in enumerate(bboxes):
            for key in self._keys(x0, y0, x1, y1):
                self.bins.setdefault(key, []).append(i)

    def add(self, bbox: bbox_t) -> None:
        """Register one more bbox under the next index; bboxes is unchanged."""
        for key in self._keys(*bbox):
            self.bins.setdefault(key, []).append(self.count)
        self.count += 1

    def _keys(self, x0: float, y0: float, x1: float,
              y1: float) -> collections.abc.Iterator[tuple[int, int]]:
        gx0, gx1 = int((x0 - self.ox) // self.cell), int((x1 - self.ox) // self.cell)
        gy0, gy1 = int((y0 - self.oy) // self.cell), int((y1 - self.oy) // self.cell)
        return ((gx, gy) for gy in range(gy0, gy1 + 1) for gx in range(gx0, gx1 + 1))

    def query(self, bbox: bbox_t, inflate: float) -> index_array_t:
        """Indices possibly within inflate of bbox, sorted unique."""
        found = [ids for key in self._keys(bbox[0] - inflate, bbox[1] - inflate,
                                           bbox[2] + inflate, bbox[3] + inflate)
                 if (ids := self.bins.get(key)) is not None]
        if not found:
            return np.empty(0, dtype=np.intp)
        return np.unique(np.concatenate([np.asarray(f, dtype=np.intp) for f in found]))


def _grid_cell(bboxes: bbox_array_t, gap: float) -> float:
    """Cell size for a bbox grid index"""
    if not len(bboxes):
        return max(1.0, float(gap))
    sides = np.maximum(bboxes[:, 2] - bboxes[:, 0], bboxes[:, 3] - bboxes[:, 1])
    return max(1.0, float(gap), float(np.median(sides)))


class SnapshotPool:
    """Snapshots with a bbox grid index and cached bbox/centroid arrays."""

    __slots__ = 'snaps', 'origins', '_grid', '_bb', '_cxy'
    # member snapshots
    snaps: list[MaskSnapshot]
    # per snapshot, its origin tuple or None
    origins: list[tuple | None]
    # bbox index, built on first query and grown by extend
    _grid: _BboxGrid | None
    # cached (n, 4) bboxes, built on first use and grown by extend
    _bb: bbox_array_t | None
    # cached (n, 2) centroids, built on first use and grown by extend
    _cxy: point_array_t | None

    def __init__(self, snaps: collections.abc.Iterable[MaskSnapshot] = (),
                 origins: collections.abc.Iterable[tuple | None] | None = None):
        self.snaps, self.origins = [], []
        self._grid = self._bb = self._cxy = None
        self.extend(snaps, origins)

    def __len__(self) -> int:
        return len(self.snaps)

    def append(self, snap: MaskSnapshot, origin: tuple | None = None) -> None:
        self.extend([snap], [origin])

    def extend(self, snaps: collections.abc.Iterable[MaskSnapshot],
               origins: collections.abc.Iterable[tuple | None] | None = None) -> None:
        snaps = list(snaps)
        self.snaps.extend(snaps)
        self.origins.extend([None] * len(snaps) if origins is None else origins)
        if snaps and self._bb is not None:
            self._bb = np.concatenate([self._bb, np.asarray(
                [s.bbox for s in snaps], dtype=np.float64).reshape(-1, 4)])
        if snaps and self._cxy is not None:
            self._cxy = np.concatenate([self._cxy, np.asarray(
                [s.centroid for s in snaps], dtype=np.float64).reshape(-1, 2)])
        if self._grid is not None:
            if self._grid.count:
                for s in snaps:
                    self._grid.add(s.bbox)
            else:
                self._grid = None

    @property
    def bb(self) -> bbox_array_t:
        """(n, 4) member bboxes."""
        if self._bb is None:
            self._bb = np.asarray([s.bbox for s in self.snaps],
                                  dtype=np.float64).reshape(-1, 4)
        return self._bb

    @property
    def cxy(self) -> point_array_t:
        """(n, 2) member centroids."""
        if self._cxy is None:
            self._cxy = np.asarray([s.centroid for s in self.snaps],
                                   dtype=np.float64).reshape(-1, 2)
        return self._cxy

    @property
    def grid(self) -> _BboxGrid:
        if self._grid is None:
            self._grid = _BboxGrid(self.bb, _grid_cell(self.bb, 0.0))
        return self._grid

    def bbox(self, i: int) -> bbox_t:
        return tuple(float(v) for v in self.bb[i])

    def query(self, bbox: bbox_t, inflate: float) -> index_array_t:
        return self.grid.query(bbox, inflate)

    def query_overlapping(self, bbox: bbox_t) -> index_array_t:
        """Indices of members strictly overlapping bbox, sorted unique."""
        near = self.grid.query(bbox, 0.0)
        if near.size:
            nb = self.bb[near]
            near = near[(np.minimum(bbox[2], nb[:, 2]) > np.maximum(bbox[0], nb[:, 0]))
                        & (np.minimum(bbox[3], nb[:, 3]) > np.maximum(bbox[1], nb[:, 1]))]
        return near

    def interior_idx(self, rect: rect_t, margin: float) -> index_array_t:
        """Members at least margin px inside every side of rect"""
        bb = self.bb
        return np.flatnonzero(
            (bb[:, 0] >= rect[0] + margin) & (bb[:, 1] >= rect[1] + margin)
            & (bb[:, 2] <= rect[2] - margin) & (bb[:, 3] <= rect[3] - margin))

    def interior_exemplars(self, rect: rect_t, max_exemplars: int,
                           rotate: int = 0, prefer: int | None = None) -> list[int]:
        """Indices usable as exemplars in rect"""
        margin = SweepConsts.EXEMPLAR_EDGE_MARGIN * (rect[2] - rect[0])
        idx = self.interior_idx(rect, margin)
        center = ((rect[0] + rect[2]) / 2, (rect[1] + rect[3]) / 2)
        cxy = self.cxy
        order = idx[np.argsort(np.hypot(cxy[idx, 0] - center[0],
                                        cxy[idx, 1] - center[1]))]
        order = np.roll(order, -rotate)
        chosen = [int(i) for i in order]
        pinned = prefer is not None and prefer in chosen
        if pinned:
            chosen.remove(prefer)
            chosen.insert(0, prefer)
        chosen = chosen[:max_exemplars]
        chosen[pinned:] = sorted(chosen[pinned:], key=lambda i: -self.snaps[i].score)
        return chosen

    def typical_sized(self) -> SnapshotPool:
        """Members whose mask area lies within one standard deviation of the mean."""
        areas = np.asarray([_snapshot_area(s) for s in self.snaps], dtype=np.float64)
        lo, hi = areas.mean() - areas.std(), areas.mean() + areas.std()
        keep = [i for i, a in enumerate(areas) if lo <= a <= hi]
        return SnapshotPool([self.snaps[i] for i in keep],
                            [self.origins[i] for i in keep])


def cluster_ambiguous(
    bboxes: boxes_t,
    scores: list[float] | None,
    *,
    recrop_px: int,
    gap_px: float,
    max_frac: float = SweepConsts.CLUSTER_MAX_FRAC,
    snapshots: list[MaskSnapshot],
) -> list[list[int]]:
    """Group ambiguous fragments into reprompt clusters"""
    if not bboxes:
        return []
    cap = max_frac * recrop_px
    arr = np.asarray(bboxes, dtype=np.float64)
    areas = (arr[:, 2] - arr[:, 0]) * (arr[:, 3] - arr[:, 1])
    order = masker.annotations.instance_mask.seed_order(len(bboxes), scores, areas)
    rank = np.empty(len(order), dtype=np.intp)
    rank[order] = np.arange(len(order))
    grid = _BboxGrid(arr, _grid_cell(arr, gap_px))
    assigned = np.zeros(len(bboxes), dtype=bool)
    groups: list[list[int]] = []
    for seed in order:
        seed = int(seed)
        if assigned[seed]:
            continue
        assigned[seed] = True
        group = [seed]
        running = bboxes[seed]
        cand = grid.query(bboxes[seed], gap_px)
        cand = cand[~assigned[cand]]
        cand = cand[bbox_gaps(bboxes[seed], arr[cand]) <= gap_px]
        cand = cand[np.argsort(rank[cand])]
        cand = cand[_mask_gaps_le(snapshots[seed],
                                  [snapshots[int(j)] for j in cand], gap_px)]
        for j in cand:
            j = int(j)
            grown = _union(running, bboxes[j])
            if grown[2] - grown[0] > cap or grown[3] - grown[1] > cap:
                continue
            assigned[j] = True
            group.append(j)
            running = grown
        groups.append(sorted(group))
    return sorted(groups)


def _slide_rect(cx: float, cy: float, cw: int, ch: int,
                img_w: int, img_h: int) -> rect_t:
    """cw x ch rect centered near (cx, cy), slid inside the image"""
    x0, y0 = int(round(cx - cw / 2)), int(round(cy - ch / 2))
    x0, y0 = max(0, min(x0, img_w - cw)), max(0, min(y0, img_h - ch))
    return (x0, y0, x0 + cw, y0 + ch)


def plan_cluster_recrop(
    cluster_bbox: bbox_t,
    img_w: int,
    img_h: int,
    *,
    expand: float,
    model_side: int = SweepConsts.MODEL_SIDE,
    object_scale: float | None = None,
    band: tuple[float, float] | None = None,
    scale: float = 1.0,
) -> masker.execution.recrop.Recrop:
    """2nd pass context recrop for a cluster

    With band and object_scale the recrop is sized so the object
    presents mid-band, floored at band_cover_margin x the cluster bbox.
    scale rescales the recrop: < 1 tightens, > 1 widens."""
    x0, y0, x1, y1 = cluster_bbox
    if band is not None and object_scale is not None:
        lo, hi = band
        frac = min(hi, max(lo, math.sqrt(lo * hi) / scale))
        target = object_scale / frac
        cw = max(round(target), round((x1 - x0) * SweepConsts.BAND_COVER_MARGIN))
        ch = max(round(target), round((y1 - y0) * SweepConsts.BAND_COVER_MARGIN))
    else:
        eff = max(1.0, expand * scale)
        cw, ch = round((x1 - x0) * eff), round((y1 - y0) * eff)
    floor = SweepConsts.MIN_RECROP
    cw, ch = min(img_w, max(cw, floor)), min(img_h, max(ch, floor))
    rect = _slide_rect((x0 + x1) / 2, (y0 + y1) / 2, cw, ch, img_w, img_h)
    return masker.execution.recrop.Recrop.from_rect(*rect, model_side)


class SweepContext:
    """Per-image state shared by every sweep phase."""

    __slots__ = ('generator', 'source_image', 'text', 'img_w', 'img_h', 'params',
                 'model_side', 'source_key')
    generator: masker.execution.model.MaskGenerator
    # normalized source image
    source_image: PIL.Image.Image
    # prompt text
    text: str
    img_w: int
    img_h: int
    params: SweepParams
    # presented recrop side
    model_side: int
    # masker per-image feature cache key
    source_key: str | None

    def __init__(self, generator: masker.execution.model.MaskGenerator,
                 source_image: PIL.Image.Image,
                 text: str, params: SweepParams, *,
                 model_side: int = SweepConsts.MODEL_SIDE,
                 source_key: str | None = None):
        self.generator, self.text, self.params = generator, text, params
        self.source_image = source_image
        self.img_w, self.img_h = self.source_image.size
        self.model_side, self.source_key = model_side, source_key

    @property
    def gap(self) -> float:
        # clustering gap: seam-band width, >= 1
        p, side = self.params, self.model_side
        return max(1.0, float(p.recrop_px(side) - p.grid_step(side)))


class KeepBatch:
    """One recrop's kept re-detections as parallel lists."""

    __slots__ = 'kept', 'snaps', 'subsumed'
    # recrop-local instances
    kept: list[masker.execution.model.ObjectInstance]
    # their snapshots
    snaps: list[MaskSnapshot]
    # per keep, accepted indices it contains
    subsumed: list[list[int]]

    def __init__(self, kept: list[masker.execution.model.ObjectInstance] | None = None,
                 snaps: list[MaskSnapshot] | None = None,
                 subsumed: list[list[int]] | None = None) -> None:
        self.kept, self.snaps, self.subsumed = kept or [], snaps or [], subsumed or []

    def __len__(self) -> int:
        return len(self.kept)

    def append(self, inst: masker.execution.model.ObjectInstance, snap: MaskSnapshot,
               covers: list[int]) -> None:
        self.kept.append(inst)
        self.snaps.append(snap)
        self.subsumed.append(covers)

    def filter(self, indices: list[int]) -> KeepBatch:
        return KeepBatch([self.kept[k] for k in indices],
                         [self.snaps[k] for k in indices],
                         [self.subsumed[k] for k in indices])

    def dedup(self, *, context: collections.abc.Sequence[MaskSnapshot] = (),
              context_grid: _BboxGrid | None = None,
              params: SweepParams) -> tuple[KeepBatch, list[int]]:
        """Drop DUPs of higher-scored keeps or context; (batch, subsumed context idx)."""
        ctx_subsumed: list[int] = []
        alive: list[int] = []
        for k in sorted(range(len(self.kept)), key=lambda i: -self.kept[i].score):
            snap = self.snaps[k]
            winner = next(
                (j for j in alive
                 if bbox_iou(snap.bbox, self.snaps[j].bbox) > 0
                 and _accepted_relation(self.snaps[j], snap, params)
                 is AcceptedRelation.DUP), None)
            if winner is not None:
                self.subsumed[winner] += [c for c in self.subsumed[k]
                                          if c not in self.subsumed[winner]]
                continue
            dup = False
            ctx: list[int] = []
            for i in ([] if context_grid is None
                      else context_grid.query(snap.bbox, 0.0)):
                o = context[int(i)]
                if bbox_iou(snap.bbox, o.bbox) <= 0:
                    continue
                relation = _accepted_relation(o, snap, params)
                if relation is AcceptedRelation.DUP:
                    dup = True
                    break
                if relation is AcceptedRelation.SUBSUMES:
                    ctx.append(int(i))
            if dup:
                continue
            alive.append(k)
            ctx_subsumed += [i for i in ctx if i not in ctx_subsumed]
        return self.filter(sorted(alive)), ctx_subsumed


class AcceptedRelation(enum.Enum):
    """Re-detection vs accepted: same, larger, neither."""

    DUP = "dup"
    SUBSUMES = "subsumes"
    NONE = "none"


def _accepted_relation(a: MaskSnapshot, snap: MaskSnapshot,
                       params: SweepParams) -> AcceptedRelation:
    """Classify a re-detection against one accepted snapshot"""
    if a.cells.size == 0 or snap.cells.size == 0:
        iou = bbox_iou(a.bbox, snap.bbox)
        cover_a, cover_snap = bbox_cover(a.bbox, snap.bbox)
    else:
        inter, area_a, area_snap = _mask_intersect_area(a, snap)
        union = area_a + area_snap - inter
        iou = min(1.0, inter / union) if union > 0 else 0.0
        cover_a = min(1.0, inter / area_a) if area_a > 0 else 0.0
        cover_snap = min(1.0, inter / area_snap) if area_snap > 0 else 0.0
    if iou >= params.dup_iou:
        return AcceptedRelation.DUP
    if params.ios_gate:
        if cover_snap >= params.dup_ios:
            return AcceptedRelation.DUP
        if cover_a >= params.dup_ios:
            return AcceptedRelation.SUBSUMES
    return AcceptedRelation.NONE


def _recrop_truncated(bbox: bbox_t,
                      recrop: masker.execution.recrop.Recrop, ctx: SweepContext) -> bool:
    """Whether bbox runs into a recrop side that is not the image edge"""
    m = SweepConsts.NOVEL_EDGE_MARGIN * (recrop.x1 - recrop.x0) / recrop.out_w
    return ((bbox[0] < recrop.x0 + m and recrop.x0 > 0)
            or (bbox[1] < recrop.y0 + m and recrop.y0 > 0)
            or (bbox[2] > recrop.x1 - m and recrop.x1 < ctx.img_w)
            or (bbox[3] > recrop.y1 - m and recrop.y1 < ctx.img_h))


def _member_in_bbox(s: MaskSnapshot, bbox: bbox_t, centroid: str) -> bool:
    """Whether the mask centroid ("box": bbox center) lies inside bbox."""
    cx, cy = (((s.bbox[0] + s.bbox[2]) / 2, (s.bbox[1] + s.bbox[3]) / 2)
              if centroid == "box" else s.centroid)
    return bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]


def keep_redetections(
    ctx: SweepContext,
    instances: list[masker.execution.model.ObjectInstance],
    recrop: masker.execution.recrop.Recrop,
    cluster_bbox: bbox_t,
    accepted: SnapshotPool,
    *,
    novel_gap: float | None = None,
    context_snaps: collections.abc.Sequence[MaskSnapshot] = (),
) -> KeepBatch:
    """Filter one recrop's re-detections."""
    batch = KeepBatch()
    ctx_bb = (np.asarray([s.bbox for s in context_snaps],
                         dtype=np.float64).reshape(-1, 4)
              if novel_gap is not None else None)
    for inst in instances:
        snap = snapshot_from_local(inst, recrop)
        if not _member_in_bbox(snap, cluster_bbox, ctx.params.centroid):
            if novel_gap is None or _recrop_truncated(snap.bbox, recrop, ctx):
                continue
            near = [int(i) for i in accepted.query(snap.bbox, novel_gap)]
            if _mask_gaps_le(
                    snap, [accepted.snaps[i] for i in near], novel_gap).any():
                continue
            nc = len(context_snaps)
            pb = np.concatenate([ctx_bb, np.asarray(
                [s.bbox for s in batch.snaps], dtype=np.float64).reshape(-1, 4)])
            others = [context_snaps[i] if i < nc else batch.snaps[i - nc]
                      for i in np.flatnonzero(bbox_gaps(snap.bbox, pb) <= novel_gap)]
            if _mask_gaps_le(snap, others, novel_gap).any():
                continue
            batch.append(inst, snap, [])
            continue
        candidates = [int(i) for i in accepted.query_overlapping(snap.bbox)]
        dup = False
        covers: list[int] = []
        for i in candidates:
            relation = _accepted_relation(accepted.snaps[i], snap, ctx.params)
            if relation is AcceptedRelation.DUP:
                dup = True
                break
            if relation is AcceptedRelation.SUBSUMES:
                covers.append(i)
        if not dup:
            batch.append(inst, snap, covers)
    return batch


# --- optional recrop packing ----------------------------------------------------


@dataclasses.dataclass(frozen=True, slots=True)
class PackedSlot:
    """One recrop's cell inside a packed canvas"""

    # index into the packed recrop list
    cluster: int
    canvas: int
    cell_x: int
    cell_y: int
    # the recrop, presented at the cell size
    recrop: masker.execution.recrop.Recrop


def pack_recrops(
    recrop_rects: list[rect_t],
    k: int,
    model_side: int = SweepConsts.MODEL_SIDE,
) -> list[PackedSlot]:
    """Tile k x k recrops per canvas so one model pass covers several"""
    cell = model_side // k
    slots = []
    for i, (x0, y0, x1, y1) in enumerate(recrop_rects):
        slot_index = i % (k * k)
        slots.append(PackedSlot(
            cluster=i,
            canvas=i // (k * k),
            cell_x=(slot_index % k) * cell,
            cell_y=(slot_index // k) * cell,
            recrop=masker.execution.recrop.Recrop.from_rect(x0, y0, x1, y1, cell),
        ))
    return slots


def resize_pixels(pixels: torch.Tensor, out_w: int, out_h: int) -> torch.Tensor:
    """Resize a uint8 (3, h, w) tensor to (3, out_h, out_w), antialiased bilinear."""
    out = torch.nn.functional.interpolate(
        pixels[None].to(torch.float32), size=(out_h, out_w),
        mode="bilinear", align_corners=False, antialias=True)
    return out[0].round_().clamp_(0, 255).to(torch.uint8)


def compose_canvases(source: torch.Tensor, slots: list[PackedSlot],
                     model_side: int = SweepConsts.MODEL_SIDE) -> list[torch.Tensor]:
    """Packed canvases over a uint8 (3, H, W) source; (3, model_side, model_side) tensors."""
    count = max(s.canvas for s in slots) + 1
    canvases = [torch.zeros((source.shape[0], model_side, model_side),
                            dtype=source.dtype, device=source.device) for _ in range(count)]
    for s in slots:
        w = s.recrop
        canvases[s.canvas][:, s.cell_y:s.cell_y + w.out_h,
                           s.cell_x:s.cell_x + w.out_w] = resize_pixels(
                               source[:, w.y0:w.y1, w.x0:w.x1], w.out_w, w.out_h)
    return canvases


def slice_canvas_instances(
    instances: list[masker.execution.model.ObjectInstance], slots: list[PackedSlot], canvas: int
) -> dict[int, list[masker.execution.model.ObjectInstance]]:
    """Hand one canvas's detections back to their slots as recrop-local instances"""
    by_cluster: dict[int, list[masker.execution.model.ObjectInstance]] = {}
    mine = [s for s in slots if s.canvas == canvas]
    for inst in instances:
        x0, y0, x1, y1 = inst.bbox_xyxy
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        for s in mine:
            if (s.cell_x <= cx < s.cell_x + s.recrop.out_w
                    and s.cell_y <= cy < s.cell_y + s.recrop.out_h):
                local = inst.mask[s.cell_y:s.cell_y + s.recrop.out_h,
                                  s.cell_x:s.cell_x + s.recrop.out_w]
                if local.any():
                    by_cluster.setdefault(s.cluster, []).append(
                        masker.execution.model.ObjectInstance(
                            mask=local,
                            bbox_xyxy=masker.execution.model.bbox_from_mask(local),
                            score=inst.score,
                            centroid=masker.execution.model.mask_centroid(local),
                        ))
                break
    return by_cluster


# --- orchestration -------------------------------------------------------------


def reprompt_recrops(
    generator: masker.execution.model.MaskGenerator,
    source_image: PIL.Image.Image,
    recrops: list[masker.execution.recrop.Recrop],
    target_bboxes: boxes_t,
    accepted: list[MaskSnapshot],
    *,
    text: str,
    params: SweepParams,
    exemplar_boxes: list[boxes_t] | None = None,
    model_side: int = SweepConsts.MODEL_SIDE,
    pack: int | None = None,
    source_key: str | None = None,
) -> collections.abc.Iterator[sweep_event_t]:
    """_reprompt_recrops over a fresh SweepContext, as dict events rebatch and reprompt."""
    ctx = SweepContext(generator, source_image, text, params, model_side=model_side,
                       source_key=source_key)
    for ev in _reprompt_recrops(ctx, recrops, target_bboxes, SnapshotPool(accepted),
                                exemplar_boxes=exemplar_boxes, pack=pack):
        if ev[0] == "rebatch":
            yield {"type": "rebatch", "indices": list(ev[1]),
                   "recrops": [recrops[i] for i in ev[1]]}
        else:
            _, ci, recrop, kept, seconds, _subsumed = ev
            yield {"type": "reprompt", "index": ci, "recrop": recrop,
                   "instances": kept, "seconds": seconds}


def _reprompt_recrops(
    ctx: SweepContext,
    recrops: list[masker.execution.recrop.Recrop],
    target_bboxes: boxes_t,
    accepted: SnapshotPool,
    *,
    exemplar_boxes: list[boxes_t] | None = None,
    pack: int | None = None,
    novel_gap: float | None = None,
    context_snaps: collections.abc.Sequence[MaskSnapshot] = (),
    batches_out: dict[int, KeepBatch] | None = None,
) -> collections.abc.Iterator[phase_event_t]:
    """Re-prompt recrops; yields rebatch and reprompt phase events."""
    generator, source_image, side = ctx.generator, ctx.source_image, ctx.model_side

    def settle(ci: int, recrop: masker.execution.recrop.Recrop,
               instances: list[masker.execution.model.ObjectInstance],
               seconds: float) -> phase_event_t:
        batch = keep_redetections(
            ctx, instances, recrop, target_bboxes[ci], accepted,
            novel_gap=novel_gap, context_snaps=context_snaps)
        if batches_out is not None:
            batches_out[ci] = batch
        return ("reprompt", ci, recrop, batch.kept, seconds, batch.subsumed)

    boxes_per = exemplar_boxes or [()] * len(recrops)
    plain = [i for i, bs in enumerate(boxes_per) if not bs]
    if plain and pack is None:
        for ev in generator.segment_recrops(
            source_image, [recrops[i] for i in plain], text=ctx.text,
            source_key=ctx.source_key,
        ):
            if ev[0] == "batch":
                yield ("rebatch", [plain[j] for j in ev[1]])
                continue
            if ev[0] == "stats":  # pass-1 telemetry only
                continue
            _, pi, instances, seconds = ev
            yield settle(plain[pi], recrops[plain[pi]], instances, seconds)
    elif plain:
        slots = pack_recrops([recrops[i].rect for i in plain], pack, side)
        source = generator.source_tensor(source_image, ctx.source_key)
        canvases = compose_canvases(source, slots, side)
        for ev in generator.segment_images(canvases, text=ctx.text):
            if ev[0] == "batch":
                yield ("rebatch", [plain[s.cluster] for c in ev[1]
                                   for s in slots if s.canvas == c])
                continue
            _, canvas_index, instances, seconds = ev
            per_cluster = slice_canvas_instances(instances, slots, canvas_index)
            canvas_slots = [s for s in slots if s.canvas == canvas_index]
            share = seconds / len(canvas_slots)
            for s in canvas_slots:
                yield settle(plain[s.cluster], s.recrop,
                             per_cluster.get(s.cluster, []), share)
    for i, bs in enumerate(boxes_per):
        if not bs:
            continue
        recrop = recrops[i]
        prompt = masker.execution.model.SegmentationPrompt(
            text=ctx.text, exemplar_boxes=[recrop.to_local_bbox(b) for b in bs])
        yield ("rebatch", [i])
        start = time.perf_counter()
        instances = generator.segment_recrop(source_image, recrop, prompt,
                                             source_key=ctx.source_key)
        yield settle(i, recrop, instances, time.perf_counter() - start)


def _merged_blob(covers: list[int], accepted: SnapshotPool, gap: float) -> bool:
    """Whether a re-detection covering several accepted masks merged distinct objects"""
    if len(covers) < 2:
        return False
    snaps = [accepted.snaps[i] for i in covers]
    return any(not _mask_gaps_le(s, snaps[k + 1:], gap).all()
               for k, s in enumerate(snaps[:-1]))


class Cluster:
    """One pass-2 reprompt cluster and its back-off state."""

    __slots__ = ('index', 'bbox', 'object_scale', 'recrop',
                 'limit', 'scale', 'seconds')
    # position in the plan
    index: int
    # union of the member bboxes, original coords
    bbox: bbox_t
    # median fragment long side
    object_scale: float
    # context recrop of the current attempt
    recrop: masker.execution.recrop.Recrop
    # merge-blob gate: MERGE_AREA_FACTOR x object_scale^2
    limit: float
    # cumulative back-off rescale of recrop
    scale: float
    # forward seconds across attempts
    seconds: float

    def __init__(self, index: int, bbox: bbox_t, object_scale: float,
                 ctx: SweepContext, *, recrop: masker.execution.recrop.Recrop | None = None):
        self.index, self.bbox, self.object_scale = index, bbox, object_scale
        self.limit = SweepConsts.MERGE_AREA_FACTOR * object_scale * object_scale
        self.scale, self.seconds = 1.0, 0.0
        self.recrop = self.plan(ctx) if recrop is None else recrop

    def plan(self, ctx: SweepContext, scale: float = 1.0) -> masker.execution.recrop.Recrop:
        return plan_cluster_recrop(self.bbox, ctx.img_w, ctx.img_h,
                         expand=ctx.params.reprompt_expand, model_side=ctx.model_side,
                         object_scale=self.object_scale, band=ctx.params.band,
                         scale=scale)

    def backoff(self, ctx: SweepContext, attempt: int) -> masker.execution.recrop.Recrop | None:
        """Tighten recrop by backoff_scale; None once saturated"""
        scale = self.scale * ctx.params.backoff_scale
        recrop = self.plan(ctx, scale)
        if recrop.rect == self.recrop.rect:
            logger.debug("backoff saturated: cluster %d rect %s unchanged at "
                         "scale %.3f, attempt %d", self.index, recrop.rect, scale,
                         attempt + 1)
            return None
        self.scale, self.recrop = scale, recrop
        return recrop


class ClusterPlan:
    """Pass-2 clusters (cluster_ambiguous) with their context recrops."""

    __slots__ = 'clusters',
    # index order
    clusters: list[Cluster]

    def __init__(self, bboxes: boxes_t, scores: list[float],
                 snaps: list[MaskSnapshot], ctx: SweepContext):
        groups = cluster_ambiguous(
            bboxes, scores or None, snapshots=snaps, gap_px=ctx.gap,
            recrop_px=ctx.params.recrop_px(ctx.model_side))
        self.clusters = []
        for k, members in enumerate(groups):
            frags = [bboxes[m] for m in members]
            arr = np.asarray(frags, dtype=np.float64)
            bbox = (*arr[:, :2].min(axis=0), *arr[:, 2:].max(axis=0))
            side = float(np.median(np.maximum(arr[:, 2] - arr[:, 0], arr[:, 3] - arr[:, 1])))
            self.clusters.append(Cluster(k, bbox, side, ctx))


def _reprompt_with_backoff(
    ctx: SweepContext,
    clusters: list[Cluster],
    accepted: SnapshotPool,
    context_snaps: list[MaskSnapshot] | None = None,
) -> collections.abc.Iterator[phase_event_t]:
    """Re-prompt each cluster with back-off; yields rebatch, tighten, reprompt, replace3."""
    if context_snaps is None:
        context_snaps = []
    params, gap = ctx.params, ctx.gap
    novel_gap = None
    if params.novel:
        novel_gap = params.novel_gap_px if params.novel_gap_px is not None else gap
    retries = params.backoff_retries
    # exemplar candidates
    typical = None
    if params.reprompt_exemplars and len(accepted):
        typical = accepted.typical_sized() or None
    active = list(clusters)
    # settled clusters' keeps, the cross-recrop dedup context, origins
    # (cluster, kept position)
    settled = SnapshotPool()
    retracted: set[int] = set()
    for attempt in range(retries + 1):
        ex_boxes = None
        if typical is not None:
            ex_boxes = [[typical.bbox(j) for j in typical.interior_exemplars(
                c.recrop.rect, params.reprompt_exemplars)] for c in active]
        retry: list[Cluster] = []
        batches: dict[int, KeepBatch] = {}
        for ev in _reprompt_recrops(
            ctx, [c.recrop for c in active], [c.bbox for c in active], accepted,
            exemplar_boxes=ex_boxes, pack=params.pack if attempt == 0 else None,
            novel_gap=novel_gap, context_snaps=context_snaps, batches_out=batches,
        ):
            if ev[0] == "rebatch":
                if attempt == 0:
                    yield ev
                continue
            # recrop is the slot's cell recrop when packed
            _, ri, recrop, _, secs, _ = ev
            c, batch = active[ri], batches[ri]
            c.seconds += secs
            merged = any(_merged_blob(cv, accepted, gap) for cv in batch.subsumed)
            oversized = sum(_snapshot_area(s) > c.limit for s in batch.snaps
                            if _member_in_bbox(s, c.bbox, params.centroid))
            futile = (c.bbox[2] - c.bbox[0]) * (c.bbox[3] - c.bbox[1]) > c.limit
            if ((merged or (oversized and not futile)) and attempt < retries
                    and c.backoff(ctx, attempt) is not None):
                yield ("tighten", c.index)
                retry.append(c)
                continue
            batch, hits = batch.dedup(
                context=settled.snaps, context_grid=settled.grid, params=ctx.params)
            hits = [j for j in hits if j not in retracted]
            retracted.update(hits)
            settled.extend(batch.snaps, [(c.index, pos) for pos in range(len(batch))])
            context_snaps.extend(batch.snaps)
            yield ("reprompt", c.index, recrop, batch.kept, c.seconds, batch.subsumed)
            if hits:
                yield ("replace3", c.index,
                       [("reprompt", *settled.origins[j]) for j in hits])
        active = [c for c in active if c in retry]
        if not active:
            break


@numba.njit(cache=True)
def _neighborhood_counts(cxy: point_array_t, half: distances_t) -> neighbor_counts_t:
    """Per point, how many points lie within Chebyshev half[i] of it"""
    n = cxy.shape[0]
    out = np.zeros(n, np.int64)
    for i in range(n):
        h, x, y = half[i], cxy[i, 0], cxy[i, 1]
        c = 0
        for j in range(n):
            if abs(cxy[j, 0] - x) <= h and abs(cxy[j, 1] - y) <= h:
                c += 1
        out[i] = c
    return out


def plan_exemplar_recrops(
    pool: SnapshotPool,
    img_w: int,
    img_h: int,
    *,
    frac: float,
    model_side: int = SweepConsts.MODEL_SIDE,
) -> list[masker.execution.recrop.Recrop]:
    """3rd pass recrops: greedy exemplar-centered cover of the pool"""
    bb, cxy = pool.bb, pool.cxy
    long_sides = np.maximum(bb[:, 2] - bb[:, 0], bb[:, 3] - bb[:, 1])
    sides = np.minimum(np.maximum(np.round(long_sides / frac), 1),
                       min(img_w, img_h)).astype(np.int64)
    density = _neighborhood_counts(cxy, sides / 2.0)
    order = np.lexsort((np.arange(len(pool)), -density))
    covered = np.zeros(len(pool), dtype=bool)
    recrops: list[masker.execution.recrop.Recrop] = []
    for seed in order:
        seed = int(seed)
        if covered[seed]:
            continue
        side = int(sides[seed])
        rect = _slide_rect(cxy[seed, 0], cxy[seed, 1], side, side, img_w, img_h)
        margin = SweepConsts.EXEMPLAR_EDGE_MARGIN * side
        if pool.interior_idx(rect, margin).size == 0:
            covered[seed] = True
            continue
        covered |= ((rect[0] <= cxy[:, 0]) & (cxy[:, 0] <= rect[2])
                    & (rect[1] <= cxy[:, 1]) & (cxy[:, 1] <= rect[3]))
        covered[seed] = True
        recrops.append(masker.execution.recrop.Recrop.from_rect(*rect, model_side))
    return recrops


def _square_recrop(cx: float, cy: float, side: float,
                   ctx: SweepContext) -> masker.execution.recrop.Recrop:
    """Square recrop centered near (cx, cy), slid inside the image"""
    s = max(1, min(round(side), ctx.img_w, ctx.img_h))
    return masker.execution.recrop.Recrop.from_rect(
        *_slide_rect(cx, cy, s, s, ctx.img_w, ctx.img_h), ctx.model_side)


def _exemplar_sweep(
    ctx: SweepContext,
    pool: SnapshotPool,
) -> collections.abc.Iterator[phase_event_t]:
    """Pass 3: exemplar recovery around pool; yields explan, exemplar, tighten3, replace3."""
    params, generator = ctx.params, ctx.generator
    max_exemplars = params.exemplar_max
    retries = params.exemplar_backoff
    planned = plan_exemplar_recrops(
        pool, ctx.img_w, ctx.img_h, frac=params.exemplar_frac, model_side=ctx.model_side)
    shots: list[tuple[masker.execution.recrop.Recrop, list[int]]] = []
    seen: set[tuple[rect_t, tuple[int, ...]]] = set()
    for recrop in planned:
        cx, cy = (recrop.x0 + recrop.x1) / 2, (recrop.y0 + recrop.y1) / 2
        for scale in params.exemplar_scales:
            side = (recrop.x1 - recrop.x0) * scale
            for v in range(params.exemplar_variants):
                dx, dy = SweepConsts.VARIANT_OFFSETS[v]
                shot = _square_recrop(cx + dx * side, cy + dy * side, side, ctx)
                chosen = pool.interior_exemplars(shot.rect, max_exemplars, rotate=v)
                if not chosen:
                    continue
                key = (shot.rect, tuple(chosen))
                if key not in seen:
                    seen.add(key)
                    shots.append((shot, chosen))
    yield ("explan", [c for c, _ in shots])
    retracted: set[int] = set()
    for si, (recrop, chosen) in enumerate(shots):
        # accumulates across back-off attempts
        seconds = 0.0
        for attempt in range(retries + 1):
            anchor = chosen[0]
            a = pool.bb[anchor]
            limit = SweepConsts.MERGE_AREA_FACTOR * max(a[2] - a[0], a[3] - a[1]) ** 2
            boxes = [recrop.to_local_bbox(pool.bbox(i)) for i in chosen]
            prompt = masker.execution.model.SegmentationPrompt(text=ctx.text, exemplar_boxes=boxes)
            start = time.perf_counter()
            instances = generator.segment_recrop(ctx.source_image, recrop, prompt,
                                                 source_key=ctx.source_key)
            seconds += time.perf_counter() - start
            recrop_bbox = (float(recrop.x0), float(recrop.y0),
                         float(recrop.x1), float(recrop.y1))
            batch = keep_redetections(ctx, instances, recrop, recrop_bbox, pool)
            batch = batch.filter([k for k, s in enumerate(batch.snaps)
                                  if not _recrop_truncated(s.bbox, recrop, ctx)])
            # drop keeps whose foreground lies mostly on the flattened union
            # of >= 2 pool instances
            fresh = []
            for k, s in enumerate(batch.snaps):
                near = pool.query_overlapping(s.bbox)
                if near.size < 2 or _union_cover(
                        s, [pool.snaps[int(i)] for i in near]
                ) < params.exemplar_cover:
                    fresh.append(k)
            batch = batch.filter(fresh)
            merged = any(_merged_blob(c, pool, ctx.gap) for c in batch.subsumed)
            oversized = any(_snapshot_area(s) > limit for s in batch.snaps)
            if (merged or oversized) and attempt < retries:
                new_recrop = _square_recrop(
                    (recrop.x0 + recrop.x1) / 2, (recrop.y0 + recrop.y1) / 2,
                    (recrop.x1 - recrop.x0) * SweepConsts.EXEMPLAR_BACKOFF_SCALE, ctx)
                new_chosen = pool.interior_exemplars(
                    new_recrop.rect, max_exemplars, prefer=anchor)
                if new_recrop.rect != recrop.rect and new_chosen:
                    yield ("tighten3", si)
                    recrop, chosen = new_recrop, new_chosen
                    continue
                logger.debug("pass-3 backoff stopped: shot %d rect %s, "
                             "attempt %d", si, recrop.rect, attempt + 1)
            break
        batch, _ = batch.dedup(params=ctx.params)
        origins = []
        for covers in batch.subsumed:
            if len(covers) == 1 and covers[0] not in retracted:
                retracted.add(covers[0])
                origins.append(pool.origins[covers[0]])
        pool.extend(batch.snaps, [("exemplar", si, pos) for pos in range(len(batch))])
        yield ("exemplar", si, recrop, batch.kept, seconds, batch.subsumed)
        if origins:
            yield ("replace3", si, origins)


def sweep_image(
    generator: masker.execution.model.MaskGenerator,
    source_image: PIL.Image.Image,
    text: str,
    *,
    params: SweepParams,
    source_key: str | None = None,
    model_side: int = SweepConsts.MODEL_SIDE,
) -> collections.abc.Iterator[sweep_event_t]:
    """Run the sweep on one image; yields dict events keyed by type."""
    ctx = SweepContext(generator, source_image, text, params, model_side=model_side,
                       source_key=source_key)
    source_image, img_w, img_h = ctx.source_image, ctx.img_w, ctx.img_h
    recrops = masker.execution.recrop.plan_grid(
        img_w, img_h, params.recrop_px(model_side), params.overlap, out_max_side=model_side)
    cores = core_rects(recrops)

    yield {"type": "start", "total": len(recrops), "recrops": recrops}

    # origins: (recrop index, position in that recrop's accepted list)
    accepted = SnapshotPool()
    amb_bboxes: boxes_t = []
    amb_scores: list[float] = []
    amb_snaps: list[MaskSnapshot] = []
    # the grid forwards only: cluster recrops are per-prompt geometry
    generator.begin_grid(source_key, recrops)
    for ev in generator.segment_recrops(source_image, recrops, text=text, source_key=source_key):
        if ev[0] == "batch":
            yield {"type": "batch", "indices": list(ev[1]),
                   "recrops": [recrops[i] for i in ev[1]]}
            continue
        if ev[0] == "stats":
            yield {"type": "recrop_stats", "index": ev[1],
                   "presence": round(ev[2]["presence"], 4), "count": ev[2]["count"]}
            continue
        _, index, instances, seconds = ev
        recrop = recrops[index]
        final = []
        for inst in instances:
            snap = snapshot_from_local(inst, recrop)
            if is_unambiguous(snap.bbox, cores[index], img_w, img_h):
                accepted.append(snap, (index, len(final)))
                final.append(inst)
            else:
                amb_bboxes.append(snap.bbox)
                amb_scores.append(float(inst.score))
                amb_snaps.append(snap)
        yield {"type": "segment_contents", "index": index, "recrop": recrop,
               "instances": final, "ambiguous": len(instances) - len(final),
               "seconds": seconds}

    plan = ClusterPlan(amb_bboxes, amb_scores, amb_snaps, ctx)
    clusters = plan.clusters
    yield {"type": "regroup", "clusters": len(clusters)}

    promoted: set[int] = set()
    # (cluster, position, snapshot) per pass-2 keep, the pass-3 pool's share
    pass2_keeps: list[tuple[int, int, MaskSnapshot]] = []
    retired: set[tuple[int, int]] = set()
    for ev in _reprompt_with_backoff(ctx, clusters, accepted,
                                     context_snaps=list(amb_snaps)):
        if ev[0] == "replace3":
            retired.update((o[1], o[2]) for o in ev[2])
            yield {"type": "replace3", "index": ev[1], "origins": ev[2]}
            continue
        if ev[0] == "rebatch":
            yield {"type": "rebatch", "indices": list(ev[1]),
                   "recrops": [clusters[i].recrop for i in ev[1]]}
            continue
        if ev[0] == "tighten":
            yield {"type": "tighten", "cluster": ev[1]}
            continue
        _, ci, recrop, kept, seconds, subsumed = ev
        if params.exemplar:
            pass2_keeps.extend(
                (ci, pos, snapshot_from_local(inst, recrop))
                for pos, inst in enumerate(kept))
        # exactly one subsumed accepted instance is a fragment this
        # re-detection saw whole
        retracted = sorted({i for covers in subsumed if len(covers) == 1
                            for i in covers} - promoted)
        yield {"type": "reprompt", "cluster": ci, "recrop": recrop, "instances": kept,
               "seconds": seconds, "subsumed": subsumed}
        if retracted:
            promoted.update(retracted)
            yield {"type": "replace", "cluster": ci,
                   "origins": [("segment_contents", *accepted.origins[i]) for i in retracted]}

    if params.exemplar:
        pool = SnapshotPool()
        for i, (snap, origin) in enumerate(zip(accepted.snaps, accepted.origins)):
            if i not in promoted:
                pool.append(snap, ("segment_contents", *origin))
        for ci, pos, snap in pass2_keeps:
            if (ci, pos) not in retired:
                pool.append(snap, ("reprompt", ci, pos))
        for ev in _exemplar_sweep(ctx, pool):
            if ev[0] == "replace3":
                yield {"type": "replace3", "index": ev[1], "origins": ev[2]}
            elif ev[0] == "explan":
                yield {"type": "explan", "recrops": ev[1]}
            elif ev[0] == "tighten3":
                yield {"type": "tighten3", "index": ev[1]}
            else:
                _, pi, recrop, kept, seconds, subsumed = ev
                yield {"type": "exemplar", "index": pi, "recrop": recrop, "instances": kept,
                       "seconds": seconds, "subsumed": subsumed}

    yield {"type": "done"}
