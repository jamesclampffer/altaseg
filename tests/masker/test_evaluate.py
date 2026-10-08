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

"""Scoring a sweep against a reference COCO."""
from __future__ import annotations

import functools

import numpy as np
import pytest

import masker_support
import masker.annotations.annotation_management
import masker.annotations.instance_mask
import masker.execution.evaluate
import masker.io.coco


class Frames:
    # (h, w) of the synthetic frame
    size_hw = (200, 300)


rect = functools.partial(masker_support.mask_for, size=Frames.size_hw)

# two disjoint 60x60 objects
PAIR = [rect(0, 0, 60, 60), rect(100, 100, 160, 160)]
# one 100x100 object, and the same object as two vertical halves
WHOLE = [rect(0, 0, 100, 100)]
HALVES = [rect(0, 0, 50, 100), rect(50, 0, 100, 100)]


def masks(items):
    return masker.annotations.instance_mask.InstanceMaskSet(tuple(items), *Frames.size_hw)


def _instances(pred, ref, **kwargs):
    return masker.execution.evaluate.evaluate_instances(masks(pred), masks(ref), **kwargs)


def test_match_greedy_is_one_to_one_above_threshold():
    ious = np.array([[0.9, 0.8], [0.7, 0.6]])
    assert masker.execution.evaluate.match_greedy(ious, 0.5) == [(0, 0, 0.9), (1, 1, 0.6)]
    assert masker.execution.evaluate.match_greedy(ious, 0.7) == [(0, 0, 0.9)]


@pytest.mark.parametrize("pred,ref,tp,fp,fn,precision,recall", [
    (PAIR, PAIR, 2, 0, 0, 1.0, 1.0),
    (PAIR[:1], PAIR, 1, 0, 1, 1.0, 0.5),
    (PAIR[:1] + [rect(200, 150, 260, 195)], PAIR[:1], 1, 1, 0, 0.5, 1.0),
    ([], [], 0, 0, 0, 0.0, 0.0),
])
def test_instance_counts(pred, ref, tp, fp, fn, precision, recall):
    m = _instances(pred, ref)
    assert (m.tp, m.fp, m.fn) == (tp, fp, fn)
    assert (m.precision, m.recall) == pytest.approx((precision, recall))


def test_false_split_counted():
    """One reference object predicted as three slices, none matching."""
    ref = [rect(0, 0, 90, 100)]
    pred = [rect(0, 0, 30, 100), rect(30, 0, 60, 100), rect(60, 0, 90, 100)]
    m = _instances(pred, ref)
    assert (m.false_splits, m.false_merges) == (1, 0)
    # each slice is a third of the object, well under the match threshold
    assert (m.tp, m.fp, m.fn) == (0, 3, 1)


def test_half_matches_at_the_iou_boundary():
    """A half is IoU 0.5 against the whole, which the >= rule admits; the
    split is recorded whether or not a half matched."""
    m = _instances(HALVES, WHOLE)
    assert (m.tp, m.fp, m.fn) == (1, 1, 0)
    assert m.iou_sum == pytest.approx(0.5)
    assert m.false_splits == 1
    strict = _instances(HALVES, WHOLE, match_iou=0.51)
    assert (strict.tp, strict.fp, strict.fn) == (0, 2, 1)
    assert strict.false_splits == 1


def test_split_requires_cover_fraction_inside():
    """A prediction counts toward a split only when at least cover of it
    lies inside the reference object."""
    # the left half (cover 1.0) + a 60%-inside straddler (cover 0.6)
    pred = [HALVES[0], rect(40, 0, 140, 100)]
    assert _instances(pred, WHOLE).false_splits == 1
    # raising the cover bar past the straddler leaves one qualifying
    # prediction, which is not a split
    assert _instances(pred, WHOLE, cover=0.7).false_splits == 0


def test_false_merge_counted_alongside_a_match():
    """A blob covering a small and a large object matches the large one and
    still counts as a merge."""
    ref = [WHOLE[0], rect(100, 0, 120, 20)]
    m = _instances([rect(0, 0, 120, 100)], ref)
    assert m.tp == 1
    assert (m.false_merges, m.false_splits) == (1, 0)


def test_size_bands():
    """Banding is sqrt(area) against the band edges, >= moving up a band;
    presented_scale multiplies the side first."""
    bands = masker.execution.evaluate.EvalDefaults.SIZE_BANDS
    ref = [rect(0, 0, 4, 4),  # side 4 -> band 0
           rect(10, 0, 18, 8),  # side 8: at the edge -> band 1
           rect(0, 50, 31, 81),  # side 31 -> band 2
           rect(100, 0, 300, 200)]  # side 200 -> the top band
    m = _instances([ref[2]], ref)
    assert m.ref_by_band == [1, 1, 1, 0, 0, 1]
    recall = m.recall_by_band()
    assert (recall[1], recall[2], recall[5]) == (0.0, 1.0, 0.0) and recall[3] is None
    assert masker.execution.evaluate.band_label(0) == "0-8"
    assert masker.execution.evaluate.band_label(len(bands) - 1) == ">=128"
    # downscale 0.5 magnifies 2x: sides 8, 16, 62, 400
    assert masker.execution.evaluate.presented_scale(0.5) == pytest.approx(2.0)
    assert _instances([], ref, presented_scale=2.0).ref_by_band == [0, 1, 1, 1, 0, 1]


@pytest.mark.parametrize("pred,iou,precision,recall", [
    # instance identity is ignored: the halves union to the whole
    (HALVES, 1.0, 1.0, 1.0),
    ([rect(50, 0, 150, 100)], 5000 / 15000, 0.5, 0.5),
    ([], 0.0, 0.0, 0.0),
])
def test_semantic_scores_against_the_whole(pred, iou, precision, recall):
    m = masker.execution.evaluate.evaluate_semantic(masks(pred), masks(WHOLE))
    assert (m.iou, m.precision, m.recall) == pytest.approx((iou, precision, recall))
    assert m.ref_area == 10000


def _dataset(per_image: dict[str, list[tuple]], labels: list[str]) -> dict:
    """COCO from {file_name: [(category, mask), ...]}; file_name keys may be
    subpaths."""
    data = masker.annotations.annotation_management.add_labels(masker.io.coco.empty_coco(), labels)
    ids = {c["name"]: c["id"] for c in data["categories"]}
    for name, entries in per_image.items():
        image_id = len(data["images"]) + 1
        data["images"].append({
            "id": image_id, "file_name": name,
            "width": Frames.size_hw[1], "height": Frames.size_hw[0],
        })
        for label, mask in entries:
            ann = mask.to_coco(image_id=image_id)
            ann.update(id=len(data["annotations"]) + 1, category_id=ids[label])
            data["annotations"].append(ann)
    return data


def test_compare_datasets_instance_and_semantic_modes():
    reference = _dataset({"img.jpg": [("leaf", m) for m in PAIR]}, ["leaf"])
    predicted = _dataset({"img.jpg": [("leaf", PAIR[0])]}, ["leaf"])
    report = masker.execution.evaluate.compare_datasets(predicted, reference)
    assert report["mode"] == "instance"
    assert report["images"] == 1
    assert (report["categories"]["leaf"]["tp"], report["categories"]["leaf"]["fn"]) == (1, 1)
    assert report["overall"]["recall"] == pytest.approx(0.5)
    assert "img.jpg" in report["per_image"]
    semantic = masker.execution.evaluate.compare_datasets(
        predicted, reference, mode=masker.execution.evaluate.EvalMode.SEMANTIC)
    assert semantic["mode"] == "semantic"
    assert semantic["categories"]["leaf"]["iou"] == pytest.approx(0.5)


def test_compare_datasets_keys_by_full_file_name():
    """Same basename in two directories: two rows, never collapsed; keys are
    posix-normalized full file_names on both sides."""
    reference = _dataset({"one\\img.jpg": [("leaf", PAIR[0])],
                          "two/img.jpg": [("leaf", PAIR[0])]}, ["leaf"])
    predicted = _dataset({"one/img.jpg": [("leaf", PAIR[0])]}, ["leaf"])
    report = masker.execution.evaluate.compare_datasets(predicted, reference)
    assert set(report["per_image"]) == {"one/img.jpg", "two/img.jpg"}
    # the prediction scores against one/img.jpg only; two/img.jpg, absent
    # from the predictions, scores as all-missed rather than erroring
    assert report["per_image"]["one/img.jpg"]["leaf"]["tp"] == 1
    assert report["per_image"]["two/img.jpg"]["leaf"]["fn"] == 1
    assert report["overall"]["tp"] == 1 and report["overall"]["fn"] == 1


def test_compare_datasets_scores_only_shared_categories():
    """A category on one side only is skipped unless named explicitly."""
    reference = _dataset({"img.jpg": [("leaf", PAIR[0])]}, ["leaf", "bark"])
    predicted = _dataset({"img.jpg": [("leaf", PAIR[0])]}, ["leaf"])
    report = masker.execution.evaluate.compare_datasets(predicted, reference)
    assert set(report["categories"]) == {"leaf"}
    explicit = masker.execution.evaluate.compare_datasets(
        predicted, reference, categories=["leaf", "bark"])
    assert (explicit["categories"]["bark"]["tp"], explicit["categories"]["bark"]["fn"]) == (0, 0)
