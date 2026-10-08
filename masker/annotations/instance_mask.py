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

"""Binary masks normalized into source image coord space"""

from __future__ import annotations

import collections.abc
import dataclasses

import numba
import numpy as np
import pycocotools.mask

from masker.common.common_defs import bbox_t, binary_mask_t, index_array_t, point_t, rect_t

# (h, w) uint8 0/1
type decoded_mask_t = np.typing.NDArray[np.uint8]
# run lengths, column-major, background first
type runs_t = np.typing.NDArray[np.int64]
# compressed counts as bytes
type counts_buffer_t = np.typing.NDArray[np.uint8]
# (len(a), len(b)) pairwise mask scores
type pair_matrix_t = np.typing.NDArray[np.float64]
# per-mask areas, px
type areas_t = np.typing.NDArray[np.float64]


@numba.njit(cache=True)
def _emit(out: runs_t, n: int, cur: bool, length: int, v: bool, m: int) -> tuple[int, bool, int]:
    """Append m pixels of value v to the run under construction"""
    if m == 0:
        return n, cur, length
    if v == cur:
        return n, cur, length + m
    out[n] = length
    return n + 1, v, m


@numba.njit(cache=True)
def _splice_rect(runs: runs_t, full_h: int, full_w: int, x0: int, y0: int,
                 patch: binary_mask_t) -> runs_t:
    """runs with the window at (x0, y0) replaced by patch"""
    ph, pw = patch.shape
    cx0, cy0 = max(x0, 0), max(y0, 0)
    cx1, cy1 = min(x0 + pw, full_w), min(y0 + ph, full_h)
    if cx1 <= cx0 or cy1 <= cy0:
        return runs.copy()
    # bound: the old runs, then per patched column its value changes and two seams
    total = runs.size + 2
    for x in range(cx0, cx1):
        total += 2
        for y in range(cy0 + 1, cy1):
            if (patch[y - y0, x - x0] != 0) != (patch[y - 1 - y0, x - x0] != 0):
                total += 1
    out = np.empty(total, np.int64)
    n, cur, length = 0, False, 0
    # old run k spans [re - runs[k], re); p is the copy cursor inside it
    k, re, p = 0, runs[0], 0
    end = full_h * full_w
    for x in range(cx0, cx1 + 1):
        a = x * full_h + cy0 if x < cx1 else end
        while p < a:
            if p == re:
                k += 1
                re += runs[k]
                continue
            m = min(re, a) - p
            n, cur, length = _emit(out, n, cur, length, k % 2 == 1, m)
            p += m
        if x == cx1:
            break
        for y in range(cy0, cy1):
            n, cur, length = _emit(out, n, cur, length, patch[y - y0, x - x0] != 0, 1)
        b = x * full_h + cy1
        while p < b:
            if p == re:
                k += 1
                re += runs[k]
                continue
            p = min(re, b)
    out[n] = length
    return out[:n + 1]


@numba.njit(cache=True)
def _run_moments(runs: runs_t, full_h: int) -> tuple[int, int, int]:
    """Foreground pixel count and coordinate sums, for the centroid"""
    n = sx = sy = 0
    p = 0
    for k in range(runs.size):
        s, e = p, p + runs[k]
        p = e
        if k % 2 == 0:
            continue
        for x in range(s // full_h, (e - 1) // full_h + 1):
            base = x * full_h
            ra, rb = max(s - base, 0), min(e - base, full_h)
            m = rb - ra
            n += m
            sx += x * m
            sy += (ra + rb - 1) * m // 2
    return n, sx, sy


@numba.njit(cache=True)
def _decode_rect(runs: runs_t, full_h: int, x0: int, y0: int, x1: int, y1: int) -> decoded_mask_t:
    """Decode one window of the runs; zero outside the frame"""
    out = np.zeros((y1 - y0, x1 - x0), np.uint8)
    p = 0
    lo, hi = x0 * full_h, x1 * full_h
    for k in range(runs.size):
        s, e = p, p + runs[k]
        p = e
        if k % 2 == 0 or e <= lo or s >= hi:
            continue
        for x in range(max(s // full_h, x0), min((e - 1) // full_h, x1 - 1) + 1):
            base = x * full_h
            for y in range(max(s - base, y0, 0), min(e - base, y1, full_h)):
                out[y - y0, x - x0] = 1
    return out


@numba.njit(cache=True)
def _decode_counts(raw: counts_buffer_t) -> runs_t:
    """Parse compressed, throw on error"""
    out = np.empty(raw.size, np.int64)
    n = i = 0
    while i < raw.size:
        x = k = 0
        more = True
        while more:
            if i == raw.size or k == 12:
                raise ValueError("invalid rle counts")
            c = np.int64(raw[i]) - 48
            if c < 0 or c > 63:
                raise ValueError("invalid rle counts")
            x |= (c & 0x1F) << (5 * k)
            more = (c & 0x20) != 0
            i += 1
            k += 1
        if c & 0x10:
            x |= -1 << (5 * k)
        if n > 2:
            x += out[n - 2]
        out[n] = x
        n += 1
    return out[:n]


def _run_lengths(counts: bytes, h: int, w: int) -> runs_t:
    """run lengths from compressed format"""
    x = _decode_counts(np.frombuffer(counts, dtype=np.uint8))
    if x.size == 0 or int(x.min()) < 0 or int(x.sum()) != h * w:
        raise ValueError("invalid rle counts")
    return x


@dataclasses.dataclass(frozen=True, slots=True)
class InstanceMask:
    """One full-size binary mask."""

    # compressed run lengths
    counts: bytes
    height: int
    width: int
    score: float | None = None
    category_id: int | None = None
    annotation_id: int | None = None
    # xyxy, from the counts
    bbox: bbox_t = dataclasses.field(init=False)
    # foreground pixels, from the counts
    area: int = dataclasses.field(init=False)

    def __post_init__(self) -> None:
        runs = _run_lengths(self.counts, self.height, self.width)
        x, y, w, h = (float(v) for v in pycocotools.mask.toBbox(self.rle()))
        object.__setattr__(self, "bbox", (x, y, x + w, y + h))
        object.__setattr__(self, "area", int(runs[1::2].sum()))

    @classmethod
    def from_array(
        cls,
        mask: binary_mask_t,
        *,
        score: float | None = None,
        category_id: int | None = None,
        annotation_id: int | None = None,
    ) -> InstanceMask:
        rle = pycocotools.mask.encode(np.asfortranarray(mask != 0, dtype=np.uint8))
        return cls(rle["counts"], rle["size"][0], rle["size"][1],
                   score=score, category_id=category_id, annotation_id=annotation_id)

    @classmethod
    def empty(cls, height: int, width: int) -> InstanceMask:
        rle = pycocotools.mask.frPyObjects(
            {"size": [height, width], "counts": [height * width]}, height, width)
        return cls(rle["counts"], height, width)

    @classmethod
    def from_json(
        cls,
        rle: dict,
        *,
        score: float | None = None,
        category_id: int | None = None,
        annotation_id: int | None = None,
    ) -> InstanceMask:
        """From the JSON form {"size": [h, w], "counts": "<ascii>"}."""
        h, w = rle["size"]
        return cls(rle["counts"].encode("ascii"), h, w,
                   score=score, category_id=category_id, annotation_id=annotation_id)

    @classmethod
    def from_coco(cls, ann: dict) -> InstanceMask:
        return cls.from_json(ann["segmentation"], score=ann.get("score"),
                             category_id=ann.get("category_id"), annotation_id=ann.get("id"))

    def to_json(self) -> dict:
        return {"size": [self.height, self.width], "counts": self.counts.decode("ascii")}

    def to_coco(self, *, image_id: int | None = None) -> dict:
        """coco schema"""
        x0, y0, x1, y1 = self.bbox
        ann = {
            "id": self.annotation_id,
            "image_id": image_id,
            "category_id": self.category_id,
            "segmentation": self,
            "bbox": [round(v, 2) for v in (x0, y0, x1 - x0, y1 - y0)],
            "area": self.area,
            "iscrowd": 0,
        }
        if self.score is not None:
            ann["score"] = self.score
        return ann

    def to_array(self) -> decoded_mask_t:
        return pycocotools.mask.decode(self.rle())

    def rect_array(self, rect: rect_t) -> decoded_mask_t:
        """Decoded (x0, y0, x1, y1) window as uint8 0/1; zero outside the frame."""
        return _decode_rect(_run_lengths(self.counts, self.height, self.width), self.height, *rect)

    def with_rect(self, rect: rect_t, patch: binary_mask_t) -> InstanceMask:
        """This mask with rect's pixels replaced by patch"""
        h, w = self.height, self.width
        counts = _splice_rect(_run_lengths(self.counts, h, w), h, w, rect[0], rect[1], patch)
        rle = pycocotools.mask.frPyObjects({"size": [h, w], "counts": counts}, h, w)
        return dataclasses.replace(self, counts=rle["counts"])

    def centroid(self) -> point_t:
        """Pixel-center centroid"""
        n, sx, sy = _run_moments(_run_lengths(self.counts, self.height, self.width), self.height)
        return (sx / n + 0.5, sy / n + 0.5)

    def rle(self) -> dict:
        """formatted for pycocotools"""
        return {"size": [self.height, self.width], "counts": self.counts}

    @property
    def size(self) -> tuple[int, int]:
        return (self.height, self.width)


@dataclasses.dataclass(frozen=True, slots=True)
class InstanceMaskSet:
    """Masks over one image; every member's size is (height, width)."""

    masks: tuple[InstanceMask, ...]
    height: int
    width: int

    def __post_init__(self) -> None:
        odd = sorted({m.size for m in self.masks if m.size != (self.height, self.width)})
        if odd:
            raise ValueError("mismatched mask sizes: {} in a {}x{} set".format(
                odd, self.height, self.width))

    def __len__(self) -> int:
        return len(self.masks)

    def __iter__(self) -> collections.abc.Iterator[InstanceMask]:
        return iter(self.masks)

    def __getitem__(self, index: int) -> InstanceMask:
        return self.masks[index]

    def subset(self, indices: list[int]) -> InstanceMaskSet:
        return InstanceMaskSet(tuple(self.masks[i] for i in indices), self.height, self.width)

    def areas_fp64(self) -> areas_t:
        return np.array([m.area for m in self.masks], dtype=np.float64)

    def union_all_masks(self) -> InstanceMask:
        """Pixel union; the all-background mask for an empty set."""
        h, w = self.height, self.width
        if not self.masks:
            return InstanceMask.empty(h, w)
        rle = pycocotools.mask.merge([m.rle() for m in self.masks], intersect=False)
        return InstanceMask(rle["counts"], h, w)

    def iou_matrix(self, other: InstanceMaskSet) -> pair_matrix_t:
        """(len(self), len(other)) mask IoU; zeros when either side is empty."""
        scores = pycocotools.mask.iou([m.rle() for m in self.masks],
                                      [m.rle() for m in other.masks], [0] * len(other))
        return np.asarray(scores, dtype=np.float64).reshape(len(self), len(other))

    def overlap_groups(
        self,
        threshold: float,
        scores: list[float] | None = None,
    ) -> list[list[int]]:
        """Group members by IoU with a seed; seeding avoids transitive chaining"""
        areas = self.areas_fp64()
        sim = self.iou_matrix(self)
        order = seed_order(len(self.masks), scores, areas)
        return _greedy_groups(order, sim, threshold)


def seed_order(n: int, scores: list[float] | None, areas: areas_t) -> index_array_t:
    """detection score ordered, fallback smallest to largest"""
    if scores is not None:
        return np.lexsort((np.arange(n), -np.asarray(scores, dtype=np.float64)))
    return np.lexsort((np.arange(n), areas))


def _greedy_groups(order: index_array_t, sim: pair_matrix_t, threshold: float) -> list[list[int]]:
    """single cluster grouping"""
    assigned = np.zeros(sim.shape[0], dtype=bool)
    groups: list[list[int]] = []
    for seed in order:
        seed = int(seed)
        if assigned[seed]:
            continue
        take = ~assigned & (sim[seed] >= threshold)
        take[seed] = True
        assigned |= take
        groups.append(np.flatnonzero(take).tolist())
    return sorted(groups)
