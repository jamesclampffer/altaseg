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

"""sweep vs. hand-annotated reference stats"""

from __future__ import annotations

import argparse
import dataclasses
import enum
import pathlib

import numba
import numpy as np

import masker.annotations.instance_mask
from masker.annotations.instance_mask import areas_t, pair_matrix_t
from masker.io.coco import coco_t


class EvalMode(enum.Enum):
    INSTANCE = "instance"
    SEMANTIC = "semantic"


class EvalDefaults:
    # todo: figure out what bands detect best
    SIZE_BANDS: tuple[float, ...] = (8.0, 16.0, 32.0, 64.0, 128.0, float("inf"))
    MATCH_IOU: float = 0.5
    # fraction of a mask that must fall inside another
    COVER: float = 0.5


def band_label(index: int) -> str:
    """Human-readable bound for band index, e.g. "16-32"."""
    lo = 0.0 if index == 0 else EvalDefaults.SIZE_BANDS[index - 1]
    hi = EvalDefaults.SIZE_BANDS[index]
    if hi == float("inf"):
        return ">={:.0f}".format(lo)
    return "{:.0f}-{:.0f}".format(lo, hi)


def _make_cover_matrices(ious: pair_matrix_t, pa: areas_t,
                         ra: areas_t) -> tuple[pair_matrix_t, pair_matrix_t]:
    """Cover matrices from an IoU matrix and per-side area vectors."""
    pa, ra = pa[:, None], ra[None, :]
    inter = ious * (pa + ra) / (1.0 + ious)
    with np.errstate(divide="ignore", invalid="ignore"):
        return (np.where(pa > 0, inter / pa, 0.0),
                np.where(ra > 0, inter / ra, 0.0))


@numba.njit(cache=True)
def match_greedy(ious: pair_matrix_t, threshold: float) -> list[tuple[int, int, float]]:
    """One-to-one (pred, ref, iou) pairs, highest IoU first"""
    pairs: list[tuple[int, int, float]] = []
    flat = ious.ravel()
    cand = np.flatnonzero(flat >= threshold)
    used_p: set[int] = set()
    used_r: set[int] = set()
    for f in cand[np.argsort(flat[cand])[::-1]]:
        p, r = divmod(int(f), ious.shape[1])
        if p in used_p or r in used_r:
            continue
        used_p.add(p)
        used_r.add(r)
        pairs.append((p, r, float(flat[f])))
    return pairs


@dataclasses.dataclass(slots=True)
class InstanceMetrics:
    """Instance-level scores for one (image, category) pair, or aggregated."""

    tp: int = 0
    fp: int = 0
    fn: int = 0
    iou_sum: float = 0.0
    false_splits: int = 0
    false_merges: int = 0
    ref_by_band: list[int] = dataclasses.field(default_factory=lambda: [0] * len(EvalDefaults.SIZE_BANDS))
    hit_by_band: list[int] = dataclasses.field(default_factory=lambda: [0] * len(EvalDefaults.SIZE_BANDS))

    def __add__(self, other: InstanceMetrics) -> InstanceMetrics:
        return InstanceMetrics(
            tp=self.tp + other.tp,
            fp=self.fp + other.fp,
            fn=self.fn + other.fn,
            iou_sum=self.iou_sum + other.iou_sum,
            false_splits=self.false_splits + other.false_splits,
            false_merges=self.false_merges + other.false_merges,
            ref_by_band=[a + b for a, b in zip(self.ref_by_band, other.ref_by_band)],
            hit_by_band=[a + b for a, b in zip(self.hit_by_band, other.hit_by_band)],
        )

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        denom = self.precision + self.recall
        return 2 * self.precision * self.recall / denom if denom else 0.0

    @property
    def segmentation_quality(self) -> float:
        return self.iou_sum / self.tp if self.tp else 0.0

    @property
    def recognition_quality(self) -> float:
        denom = self.tp + 0.5 * self.fp + 0.5 * self.fn
        return self.tp / denom if denom else 0.0

    @property
    def panoptic_quality(self) -> float:
        return self.segmentation_quality * self.recognition_quality

    def recall_by_band(self) -> list[float | None]:
        return [h / r if r else None
                for r, h in zip(self.ref_by_band, self.hit_by_band)]

    def to_dict(self) -> dict:
        return {
            "tp": self.tp, "fp": self.fp, "fn": self.fn,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "sq": round(self.segmentation_quality, 4),
            "rq": round(self.recognition_quality, 4),
            "pq": round(self.panoptic_quality, 4),
            "false_splits": self.false_splits,
            "false_merges": self.false_merges,
            "ref_by_band": self.ref_by_band,
            "hit_by_band": self.hit_by_band,
            "recall_by_band": [None if v is None else round(v, 4)
                               for v in self.recall_by_band()],
        }


def evaluate_instances(
    pred: masker.annotations.instance_mask.InstanceMaskSet,
    ref: masker.annotations.instance_mask.InstanceMaskSet,
    *,
    match_iou: float = EvalDefaults.MATCH_IOU,
    cover: float = EvalDefaults.COVER,
    presented_scale: float = 1.0,
) -> InstanceMetrics:
    """Match pred masks to ref masks and score them"""
    ious = pred.iou_matrix(ref)
    pairs = match_greedy(ious, match_iou)
    metrics = InstanceMetrics(
        tp=len(pairs),
        fp=len(pred) - len(pairs),
        fn=len(ref) - len(pairs),
        iou_sum=sum(iou for _, _, iou in pairs),
    )

    ref_areas = ref.areas_fp64()
    sides = np.sqrt(ref_areas) * presented_scale
    bands = np.minimum(
        np.searchsorted(EvalDefaults.SIZE_BANDS, sides, side="right"),
        len(EvalDefaults.SIZE_BANDS) - 1)
    metrics.ref_by_band = np.bincount(
        bands, minlength=len(EvalDefaults.SIZE_BANDS)).tolist()
    hit = np.zeros(len(ref), dtype=bool)
    hit[[r for _, r, _ in pairs]] = True
    metrics.hit_by_band = np.bincount(
        bands[hit], minlength=len(EvalDefaults.SIZE_BANDS)).tolist()

    cover_pred, cover_ref = _make_cover_matrices(ious, pred.areas_fp64(), ref_areas)
    metrics.false_splits = int(((cover_pred >= cover).sum(axis=0) >= 2).sum())
    metrics.false_merges = int(((cover_ref >= cover).sum(axis=1) >= 2).sum())
    return metrics


@dataclasses.dataclass(slots=True)
class SemanticMetrics:
    """Pixel-level scores for one category, or aggregated."""

    intersection: int = 0
    union: int = 0
    pred_area: int = 0
    ref_area: int = 0

    def __add__(self, other: SemanticMetrics) -> SemanticMetrics:
        return SemanticMetrics(
            intersection=self.intersection + other.intersection,
            union=self.union + other.union,
            pred_area=self.pred_area + other.pred_area,
            ref_area=self.ref_area + other.ref_area,
        )

    @property
    def iou(self) -> float:
        return self.intersection / self.union if self.union else 0.0

    @property
    def dice(self) -> float:
        denom = self.pred_area + self.ref_area
        return 2 * self.intersection / denom if denom else 0.0

    @property
    def precision(self) -> float:
        return self.intersection / self.pred_area if self.pred_area else 0.0

    @property
    def recall(self) -> float:
        return self.intersection / self.ref_area if self.ref_area else 0.0

    def to_dict(self) -> dict:
        return {
            "iou": round(self.iou, 4),
            "dice": round(self.dice, 4),
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "pred_px": self.pred_area,
            "ref_px": self.ref_area,
        }


def evaluate_semantic(
    pred: masker.annotations.instance_mask.InstanceMaskSet,
    ref: masker.annotations.instance_mask.InstanceMaskSet,
) -> SemanticMetrics:
    """Union each side's masks and score the two regions."""
    p, r = pred.union_all_masks(), ref.union_all_masks()
    union = masker.annotations.instance_mask.InstanceMaskSet((p, r), p.height, p.width).union_all_masks().area
    return SemanticMetrics(
        intersection=p.area + r.area - union,
        union=union,
        pred_area=p.area,
        ref_area=r.area,
    )


def index_instances(
    dataset: coco_t,
) -> dict[str, dict[str, masker.annotations.instance_mask.InstanceMaskSet]]:
    """Posix file_name -> category name -> masks; polygon segmentations skipped, subpaths kept distinct."""
    cats = {c["id"]: c["name"] for c in dataset.get("categories", [])}
    images = {img["id"]: (pathlib.PurePath(img.get("file_name", "")).as_posix(),
                          int(img["height"]), int(img["width"]))
              for img in dataset.get("images", [])}
    lists: dict[str, dict[str, list[masker.annotations.instance_mask.InstanceMask]]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for ann in dataset.get("annotations", []):
        seg = ann.get("segmentation")
        name = cats.get(ann.get("category_id"))
        image = images.get(ann.get("image_id"))
        if (name is None or image is None
                or not isinstance(seg, masker.annotations.instance_mask.InstanceMask)):
            continue
        file, h, w = image
        sizes[file] = (h, w)
        lists.setdefault(file, {}).setdefault(name, []).append(seg)
    return {file: {name: masker.annotations.instance_mask.InstanceMaskSet(
                       tuple(masks), *sizes[file])
                   for name, masks in by_cat.items()}
            for file, by_cat in lists.items()}


def presented_scale(downscale: float) -> float:
    """source pixels -> pixels on the model input"""
    return 1.0 / downscale if downscale > 0 else 1.0


def compare_datasets(
    predicted: coco_t,
    reference: coco_t,
    *,
    mode: EvalMode = EvalMode.INSTANCE,
    match_iou: float = EvalDefaults.MATCH_IOU,
    cover: float = EvalDefaults.COVER,
    downscale: float = 1.0,
    categories: list[str] | None = None,
) -> dict:
    """Score reference images by full posix file_name; categories in both datasets unless categories is given."""
    ref_names = [c["name"] for c in reference.get("categories", [])]
    pred_names = [c["name"] for c in predicted.get("categories", [])]
    names = categories if categories is not None else [
        n for n in ref_names if n in pred_names]
    files = [pathlib.PurePath(i["file_name"]).as_posix()
             for i in reference.get("images", [])]
    scale = presented_scale(downscale)

    per_image: dict[str, dict[str, dict[str, object]]] = {}
    per_category: dict[str, InstanceMetrics | SemanticMetrics] = {}
    for name in names:
        per_category[name] = (
            InstanceMetrics() if mode is EvalMode.INSTANCE else SemanticMetrics())

    sizes: dict[str, tuple[int, int]] = {}
    for image in reference.get("images", []):
        sizes.setdefault(pathlib.PurePath(image["file_name"]).as_posix(),
                         (int(image["height"]), int(image["width"])))
    ref_index = index_instances(reference)
    pred_index = index_instances(predicted)
    for file_name in files:
        empty = masker.annotations.instance_mask.InstanceMaskSet((), *sizes[file_name])
        ref_by_cat = ref_index.get(file_name, {})
        pred_by_cat = pred_index.get(file_name, {})
        entry: dict[str, dict[str, object]] = {}
        for name in names:
            ref = ref_by_cat.get(name, empty)
            pred = pred_by_cat.get(name, empty)
            if mode is EvalMode.INSTANCE:
                m = evaluate_instances(pred, ref, match_iou=match_iou,
                                       cover=cover, presented_scale=scale)
            else:
                m = evaluate_semantic(pred, ref)
            entry[name] = m.to_dict()
            per_category[name] = per_category[name] + m
        per_image[file_name] = entry

    totals = (InstanceMetrics() if mode is EvalMode.INSTANCE else SemanticMetrics())
    for m in per_category.values():
        totals = totals + m

    return {
        "mode": mode.value,
        "match_iou": match_iou if mode is EvalMode.INSTANCE else None,
        "downscale": downscale,
        "presented_scale": scale,
        "images": len(per_image),
        "categories": {n: m.to_dict() for n, m in per_category.items()},
        "overall": totals.to_dict(),
        "per_image": per_image,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="masker-eval",
        description="Serve the evaluation dashboard for a benchmark directory.")
    parser.add_argument("directory", type=pathlib.Path,
                        help="images plus a reference COCO file")
    parser.add_argument("--reference", type=pathlib.Path,
                        help="reference COCO path (default: found in DIRECTORY)")
    parser.add_argument("--predictions", type=pathlib.Path,
                        help="predicted COCO path (default: found in DIRECTORY)")
    parser.add_argument("--host", default="127.0.0.1", help="web UI host")
    parser.add_argument("--port", type=int, help="web UI port")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    directory = args.directory
    if not directory.is_dir():
        raise SystemExit("not a directory: {}".format(directory))

    # local import: masker.web.evalweb imports this module
    import masker.web.evalweb
    import masker.web.webcore

    app = masker.web.evalweb.create_app(
        directory, reference_path=args.reference, predictions_path=args.predictions)
    masker.web.webcore.run_app(app, "mask-eval", args.host,
                               args.port if args.port is not None
                               else masker.web.evalweb.EvalWebParams.PORT)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
