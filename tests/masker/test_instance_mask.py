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

"""instance_mask.InstanceMask and InstanceMaskSet contracts: JSON
round-trip, COCO annotation export, counts validation at construction,
union_all_masks, IoU matrix."""
from __future__ import annotations

import numpy as np
import pytest

import masker_support
import masker.annotations.instance_mask

InstanceMask = masker.annotations.instance_mask.InstanceMask
InstanceMaskSet = masker.annotations.instance_mask.InstanceMaskSet


def _set(*masks, size=(48, 64)):
    return InstanceMaskSet(masks, *size)


def test_json_round_trip_keeps_pixels_and_metadata():
    mask = masker_support.mask_for(5, 10, 25, 20)
    wire = mask.to_json()
    assert wire["size"] == [48, 64] and isinstance(wire["counts"], str)
    back = InstanceMask.from_json(wire, score=0.5, category_id=2, annotation_id=7)
    assert back.counts == mask.counts and back.size == (48, 64)
    assert (back.score, back.category_id, back.annotation_id) == (0.5, 2, 7)
    assert back.bbox == (5.0, 10.0, 25.0, 20.0) and back.area == 200
    assert np.array_equal(back.to_array(), mask.to_array())


def test_from_coco_reads_metadata():
    ann = masker_support.mask_for(0, 0, 4, 4).to_coco(image_id=3)
    ann["segmentation"] = ann["segmentation"].to_json()
    ann.update(id=11, category_id=5, score=0.25)
    mask = InstanceMask.from_coco(ann)
    assert (mask.annotation_id, mask.category_id, mask.score) == (11, 5, 0.25)


def test_to_coco_fields():
    """Ids stay None until a recorder assigns them; score only when set;
    segmentation is the mask itself."""
    mask = masker_support.mask_for(3, 2, 7, 5)
    ann = mask.to_coco()
    assert ann["segmentation"] is mask
    assert (ann["id"], ann["image_id"], ann["category_id"], ann["iscrowd"]) == (None, None, None, 0)
    assert (ann["bbox"], ann["area"]) == ([3.0, 2.0, 4.0, 3.0], 12)
    assert "score" not in ann
    tagged = InstanceMask(mask.counts, *mask.size, score=0.5, category_id=4, annotation_id=9)
    ann = tagged.to_coco(image_id=7)
    assert (ann["id"], ann["image_id"], ann["category_id"], ann["score"]) == (9, 7, 4, 0.5)


# empty; run sum != h * w; 13-character number (depth > 12); truncated mid-number
@pytest.mark.parametrize("counts", ["", "01", "PPPPPPPPPPPP0", "PPPPPPPPPPPPPP"])
def test_invalid_counts_rejected_at_construction(counts):
    with pytest.raises(ValueError):
        InstanceMask(counts.encode("ascii"), 4, 4)


def test_union_is_logical_or_and_empty_is_background():
    masks = [masker_support.mask_for(0, 0, 10, 10), masker_support.mask_for(5, 5, 20, 15),
             masker_support.mask_for(30, 30, 40, 40)]
    want = np.logical_or.reduce([m.to_array().astype(bool) for m in masks])
    assert np.array_equal(_set(*masks).union_all_masks().to_array().astype(bool), want)
    empty = _set().union_all_masks()
    assert empty.area == 0 and empty.size == (48, 64)


def test_iou_matrix_exact_and_empty_shapes():
    a = masker_support.mask_for(0, 0, 20, 20)
    b = masker_support.mask_for(10, 0, 30, 20)
    ious = _set(a).iou_matrix(_set(a, b))
    assert ious.shape == (1, 2)
    assert ious[0, 0] == pytest.approx(1.0)
    assert ious[0, 1] == pytest.approx(200 / 600)
    assert _set().iou_matrix(_set()).shape == (0, 0)
    assert _set(a).iou_matrix(_set()).shape == (1, 0)
