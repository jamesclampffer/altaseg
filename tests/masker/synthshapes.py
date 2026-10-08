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

"""Deterministic synthetic scenes of geometric shapes with exact masks.

Shapes are drawn by evaluating analytic inequalities at pixel centers on a
numpy grid: no anti-aliasing, no RNG, so a scene is a pure function of its
parameters and each shape mask is exactly the set of colored pixels.
"""
from __future__ import annotations

import dataclasses
import itertools

import numpy as np
import PIL.Image

import masker.annotations.instance_mask
from masker.common.common_defs import binary_mask_t

# JSON-form compressed mask, as the web endpoints send it
type rle_t = dict[str, object]
# (H, W, 3) uint8
type rgb_array_t = np.typing.NDArray[np.uint8]


class SceneConsts:
    BACKGROUND = 96
    # px kept clear between any shape bbox and the image edge
    EDGE_MARGIN = 8
    # (name, rgb): names feed text prompts such as "a red circle"
    PALETTE = (
        ("red", (200, 40, 40)),
        ("blue", (40, 80, 200)),
        ("green", (40, 170, 60)),
        ("yellow", (230, 200, 40)),
        ("magenta", (200, 40, 200)),
    )


class ShapeMatrix:
    # aspect = bbox w:h; scale = shape long side / image long side
    aspects = (1, 2, 4, 10, 20)
    scales = (0.5, 0.25, 0.12, 0.06, 0.03, 0.015)
    kinds = ("ellipse", "rect")
    matrix = tuple(itertools.product(kinds, aspects, scales))
    # hard-assert tier: shapes SAM3 must always find. The rest go through the
    # baseline manifest; the extreme corner (20:1 at 0.015) degenerates to a
    # 1 px-tall bar on the default canvas by design.
    easy = tuple(c for c in matrix if c[2] >= 0.06 and c[1] <= 10)
    hard = tuple(c for c in matrix if not (c[2] >= 0.06 and c[1] <= 10))


@dataclasses.dataclass(frozen=True, slots=True)
class ShapeSpec:
    # "ellipse" | "rect"
    kind: str
    # bbox left, px
    x0: int
    # bbox top, px
    y0: int
    # bbox width, px
    w: int
    # bbox height, px
    h: int
    # palette color name
    color_name: str
    # fill rgb
    color: tuple[int, int, int]

    @property
    def bbox(self) -> list[int]:
        return [self.x0, self.y0, self.x0 + self.w, self.y0 + self.h]

    @property
    def center(self) -> tuple[float, float]:
        return (self.x0 + self.w / 2, self.y0 + self.h / 2)

    @property
    def noun(self) -> str:
        if self.kind == "ellipse":
            return "circle" if self.w == self.h else "ellipse"
        return "square" if self.w == self.h else "rectangle"


class Scene:
    __slots__ = 'image', 'shapes', 'path'
    # (H, W, 3) uint8
    image: rgb_array_t
    # (spec, exact bool mask) per shape
    shapes: list[tuple[ShapeSpec, binary_mask_t]]
    # where save() last wrote the image
    path: str | None

    def __init__(self, image: rgb_array_t, shapes: list[tuple[ShapeSpec, binary_mask_t]]):
        self.image = image
        self.shapes = shapes
        self.path = None

    def save(self, path) -> str:
        PIL.Image.fromarray(self.image).save(path)
        self.path = str(path)
        return self.path

    def shape_mask(self, i: int) -> binary_mask_t:
        return self.shapes[i][1]

    def shape_bbox(self, i: int) -> list[int]:
        return self.shapes[i][0].bbox

    def shape_instance(self, i: int) -> masker.annotations.instance_mask.InstanceMask:
        return masker.annotations.instance_mask.InstanceMask.from_array(self.shapes[i][1])


def _shape_mask(spec: ShapeSpec, width: int, height: int) -> binary_mask_t:
    mask = np.zeros((height, width), dtype=bool)
    if spec.kind == "rect":
        mask[spec.y0:spec.y0 + spec.h, spec.x0:spec.x0 + spec.w] = True
        return mask
    ys, xs = np.ogrid[:height, :width]
    ccx, ccy = spec.center
    a, b = spec.w / 2, spec.h / 2
    mask[((xs + 0.5 - ccx) / a) ** 2 + ((ys + 0.5 - ccy) / b) ** 2 <= 1.0] = True
    return mask


def draw_scene(size: tuple[int, int], specs: list[ShapeSpec], textured: bool = False) -> Scene:
    width, height = size
    if textured:
        ys, xs = np.ogrid[:height, :width]
        ripple = SceneConsts.BACKGROUND + 8 * np.sin(xs / 37) * np.sin(ys / 29)
        image = np.broadcast_to(ripple[..., None], (height, width, 3))
        image = np.clip(image, 0, 255).astype(np.uint8).copy()
    else:
        image = np.full((height, width, 3), SceneConsts.BACKGROUND, dtype=np.uint8)
    shapes = []
    occupied = np.zeros((height, width), dtype=bool)
    for spec in specs:
        x0, y0, x1, y1 = spec.bbox
        assert 0 <= x0 < x1 <= width and 0 <= y0 < y1 <= height, "out of bounds: {}".format(spec)
        mask = _shape_mask(spec, width, height)
        assert mask.any(), "empty mask: {}".format(spec)
        assert not (mask & occupied).any(), "overlaps another shape: {}".format(spec)
        occupied |= mask
        image[mask] = spec.color
        shapes.append((spec, mask))
    return Scene(image, shapes)


def make_spec(size, kind, aspect, scale, pos=(0.5, 0.5), color_index=0) -> ShapeSpec:
    """Integer bbox for a shape whose long side is scale of the image long
    side, centered near pos (fractions), clamped to the edge margin."""
    width, height = size
    margin = SceneConsts.EDGE_MARGIN
    long_px = max(1, round(scale * max(width, height)))
    w = long_px
    h = max(1, round(long_px / aspect))
    x0, y0 = round(pos[0] * width - w / 2), round(pos[1] * height - h / 2)
    x0, y0 = min(max(x0, margin), width - margin - w), min(max(y0, margin), height - margin - h)
    palette = SceneConsts.PALETTE
    name, color = palette[color_index % len(palette)]
    return ShapeSpec(kind=kind, x0=x0, y0=y0, w=w, h=h, color_name=name, color=color)


def single_shape_scene(size, kind, aspect, scale, pos=(0.5, 0.5), textured=False,
                       color_index=0) -> Scene:
    return draw_scene(size, [make_spec(size, kind, aspect, scale, pos, color_index)],
                      textured=textured)


def grid_scene(size=(1400, 1000), textured=False) -> Scene:
    """Six disjoint shapes spanning the kind/aspect range in a 3x2 cell grid."""
    width, height = size
    cells = [(cx, cy) for cy in (0.25, 0.75) for cx in (1 / 6, 0.5, 5 / 6)]
    kinds_aspects = [("ellipse", 1), ("rect", 1), ("ellipse", 2),
                     ("rect", 4), ("rect", 10), ("rect", 20)]
    specs = [
        make_spec(size, kind, aspect, 0.12, pos=cell, color_index=i)
        for i, ((kind, aspect), cell) in enumerate(zip(kinds_aspects, cells))
    ]
    return draw_scene(size, specs, textured=textured)


def decode_rle(rle: rle_t) -> binary_mask_t:
    """Bool mask from a JSON-form RLE (a response field)."""
    return masker.annotations.instance_mask.InstanceMask.from_json(rle).to_array().astype(bool)


def iou_rle_vs_mask(rle: rle_t, mask: binary_mask_t) -> float:
    decoded = decode_rle(rle)
    assert decoded.shape == mask.shape, (decoded.shape, mask.shape)
    inter = np.logical_and(decoded, mask).sum()
    union = np.logical_or(decoded, mask).sum()
    return float(inter) / float(union) if union else 0.0


def best_iou(rles: list[rle_t], mask: binary_mask_t) -> tuple[float, int]:
    best, best_i = 0.0, -1
    for i, rle in enumerate(rles):
        iou = iou_rle_vs_mask(rle, mask)
        if iou > best:
            best, best_i = iou, i
    return best, best_i


def bbox_filled_mask(size_hw: tuple[int, int], bbox: list[float]) -> binary_mask_t:
    """Bool mask with the (clipped, int) bbox filled; matches box-fill semantics."""
    height, width = size_hw
    x0, y0, x1, y1 = (int(v) for v in bbox)
    mask = np.zeros((height, width), dtype=bool)
    mask[max(0, y0):min(height, y1), max(0, x0):min(width, x1)] = True
    return mask
