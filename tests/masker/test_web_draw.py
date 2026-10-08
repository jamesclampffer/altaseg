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

"""Model-free mask routes: /draw_mask, /mask_crop, /update_annotation, /dedup_instances."""
from __future__ import annotations

import numpy as np

import masker_support
import masker.io.coco


def _poly(x0, y0, x1, y1, erase=False):
    return {"kind": "polygon", "erase": erase,
            "points": [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]}


# a 21 x 11 filled rect: PIL polygons include their outline pixels
_RECT = _poly(10, 10, 30, 20)


def _drawn(client, image_path, ops, rect=(0, 0, 64, 48), base_rle=None):
    """The mask /draw_mask answers for ops, in wire form."""
    code, body = masker_support.post_json(client, "/draw_mask", {
        "image_path": str(image_path), "rect": list(rect), "base_rle": base_rle, "ops": ops})
    assert code == 200, body
    return body["rle"]


def _bar():
    """Base mask: a horizontal bar, rows 20:26 of cols 10:50."""
    bar = np.zeros((48, 64), dtype=bool)
    bar[20:26, 10:50] = True
    return bar


# --- /draw_mask --------------------------------------------------------------


def test_draw_polygon_returns_an_instance_payload(client, image_path):
    code, body = masker_support.post_json(client, "/draw_mask", {
        "image_path": str(image_path), "rect": [0, 0, 64, 48], "base_rle": None, "ops": [_RECT]})
    assert code == 200
    assert set(body) == {"score", "bbox_xyxy", "centroid", "mask_png", "mask_rect", "rle"}
    assert body["score"] == 1.0
    assert body["rle"]["size"] == [48, 64]
    assert body["mask_png"].startswith("data:image/png;base64,")

    # the drawn rect, its bbox and its mask rect describe the same region
    assert masker_support.decoded_bbox(body["rle"]) == [10, 10, 31, 21]
    assert body["bbox_xyxy"] == [10.0, 10.0, 31.0, 21.0]
    assert body["mask_rect"] == [10, 10, 31, 21]
    assert body["centroid"] == [20.5, 15.5]  # mask centroid at pixel centers
    mask = masker_support.decode_rle(body["rle"]) > 0
    assert mask[15, 20]  # interior
    assert not mask[5, 5] and not mask[15, 35]  # outside


def test_draw_one_point_stroke_is_a_disc(client, image_path):
    # diameter 9 centered on (32, 24): a 9 x 9 span, round
    rle = _drawn(client, image_path,
                 [{"kind": "stroke", "erase": False, "diameter": 9, "points": [[32, 24]]}])
    assert masker_support.decoded_bbox(rle) == [28, 20, 37, 29]
    mask = masker_support.decode_rle(rle) > 0
    assert mask[24, 32]  # center
    for dx, dy in ((3, 0), (-3, 0), (0, 3), (0, -3)):  # diameter/2 - 2, inside
        assert mask[24 + dy, 32 + dx]
    for dx, dy in ((6, 0), (-6, 0), (0, 6), (0, -6)):  # diameter/2 + 2, outside
        assert not mask[24 + dy, 32 + dx]
    assert not mask[20, 28]  # a corner of the span: the disc is not a square


def test_draw_two_point_stroke_has_round_caps(client, image_path):
    mask = masker_support.decode_rle(_drawn(
        client, image_path,
        [{"kind": "stroke", "erase": False, "diameter": 9,
          "points": [[20, 24], [40, 24]]}])) > 0
    assert mask[24, 30]  # along the stroke
    assert mask[20, 30] and mask[28, 30]  # its full width
    # the caps round past each endpoint, up to diameter/2
    assert mask[24, 43] and mask[24, 17]
    assert not mask[24, 45] and not mask[24, 15]


def test_draw_erase_stroke_punches_a_hole(client, image_path):
    bar = _bar()
    mask = masker_support.decode_rle(_drawn(
        client, image_path,
        [{"kind": "stroke", "erase": True, "diameter": 5, "points": [[26, 14], [26, 36]]}],
        rect=(20, 10, 34, 40), base_rle=masker_support.wire_rle(bar))) > 0
    # the stroke crosses the bar: every bar row is cleared under it
    assert not mask[20:26, 24:29].any()
    # and the rest of the bar stands
    kept = bar.copy()
    kept[20:26, 24:29] = False
    assert np.array_equal(mask, kept)


def test_draw_ops_apply_in_order(client, image_path):
    erase = dict(_RECT, erase=True)
    code, body = masker_support.post_json(client, "/draw_mask", {
        "image_path": str(image_path), "rect": [0, 0, 64, 48], "base_rle": None,
        "ops": [_RECT, erase]})
    assert (code, body) == (200, {"empty": True})  # the erase wins over the paint before it
    rle = _drawn(client, image_path, [_RECT, erase, _RECT])
    assert masker_support.decoded_bbox(rle) == [10, 10, 31, 21]
    assert (masker_support.decode_rle(rle) > 0)[15, 20]


def test_draw_changes_nothing_outside_the_window(client, image_path):
    bar = _bar()
    # a polygon over the whole image, clipped to a rect that crosses the bar
    mask = masker_support.decode_rle(_drawn(
        client, image_path, [_poly(0, 0, 64, 48)],
        rect=(10, 10, 30, 30), base_rle=masker_support.wire_rle(bar))) > 0
    inside = np.zeros((48, 64), dtype=bool)
    inside[10:30, 10:30] = True
    assert mask[inside].all()
    assert np.array_equal(mask[~inside], bar[~inside])


# --- /mask_crop --------------------------------------------------------------


def test_mask_crop_returns_the_rect_at_native_resolution(client, image_path):
    code, body = masker_support.post_json(
        client, "/mask_crop",
        {"image_path": str(image_path), "rle": masker_support.wire_rle(_bar()),
         "rect": [10, 15, 40, 30]},
    )
    assert code == 200
    assert body["rect"] == [10, 15, 40, 30]
    img = masker_support.decode_data_url(body["mask_png"])
    assert img.size == (30, 15)
    alpha = np.asarray(img)[:, :, 3]
    expected = np.zeros((15, 30), dtype=np.uint8)
    expected[5:11, :] = 255  # bar rows 20:26, spanning every column of the rect
    assert np.array_equal(alpha, expected)


# --- /update_annotation ------------------------------------------------------


def _update(client, image_path, coco_file, ops, rect=(0, 0, 64, 48)):
    return masker_support.post_json(client, "/update_annotation", {
        "image_path": str(image_path), "coco_file": str(coco_file),
        "annotation_id": 1, "bbox_xyxy": [5, 10, 25, 20], "rect": list(rect), "ops": ops,
    })


def test_update_annotation_erases_paints_and_rewrites_the_file(client, image_path, tmp_path):
    coco_path = masker_support.write_coco(tmp_path / "ann.json", image_path)
    code, body = _update(client, image_path, coco_path,
                         [_poly(10, 10, 15, 20, erase=True), _poly(30, 25, 40, 35)])
    assert code == 200, body
    assert set(body) == {"score", "coco_file", "bbox_xyxy", "centroid",
                         "mask_png", "mask_rect", "rle"}
    assert body["coco_file"] == str(coco_path)

    expected = masker_support.mask_for(5, 10, 25, 20).to_array() > 0  # the saved mask
    expected[10:20, 10:16] = False  # the erase polygon, outline included
    expected[25:36, 30:41] = True  # the paint polygon
    assert np.array_equal(masker_support.decode_rle(body["rle"]) > 0, expected)
    assert body["bbox_xyxy"] == [5.0, 10.0, 41.0, 36.0]
    assert body["mask_rect"] == [5, 10, 41, 36]

    stored = masker.io.coco.load_coco(coco_path)
    assert len(stored["annotations"]) == 1
    ann = stored["annotations"][0]
    assert (ann["id"], ann["category_id"]) == (1, 1)
    assert np.array_equal(ann["segmentation"].to_array() > 0, expected)
    assert ann["bbox"] == [5, 10, 36, 26]
    assert ann["area"] == int(expected.sum())


# --- /dedup_instances --------------------------------------------------------


def _dedup_groups(client, body):
    code, resp = masker_support.post_json(client, "/dedup_instances", body)
    assert code == 200, resp
    return resp["groups"]


def _bridge_rles():
    """A rect and two 3-px shifts of it: IoU 0.84 with the rect, 0.71 with each other."""
    return [masker_support.mask_for(x0, 5, x0 + 35, 30).to_json() for x0 in (5, 8, 2)]


def test_dedup_groups_near_duplicates_and_keeps_distinct(client, image_path):
    # masks 0 and 2 are the same object seen twice (1px shift, IoU 190/210 ~ 0.9);
    # masks 1 and 3 are distinct objects elsewhere
    m0, m1, m2, m3 = (masker_support.mask_for(*rect) for rect in
                      ((5, 5, 25, 15), (5, 30, 15, 40), (6, 5, 26, 15), (40, 30, 60, 40)))
    groups = _dedup_groups(client, {
        "image_path": str(image_path), "scores": [None] * 4,
        "rles": [m.to_json() for m in (m0, m1, m2, m3)], "iou": 0.5})
    # the groups partition {0..3}
    assert sorted(i for g in groups for i in g["indices"]) == [0, 1, 2, 3]

    multi = [g for g in groups if len(g["indices"]) > 1]
    assert len(multi) == 1
    (merged,) = multi
    assert merged["indices"] == [0, 2]

    # the merged mask is the pixel union of its members
    union = np.logical_or(m0.to_array(), m2.to_array())
    assert merged["rle"]["size"] == [48, 64]
    assert np.array_equal(masker_support.decode_rle(merged["rle"]) > 0, union)
    assert [float(v) for v in merged["bbox_xyxy"]] == [5.0, 5.0, 26.0, 15.0]
    assert merged["mask_png"].startswith("data:image/png")
    assert merged["mask_rect"] == [5, 5, 26, 15]  # tight rect the PNG spans

    # singleton groups come back bare: indices only, no merged payload
    singles = [g for g in groups if len(g["indices"]) == 1]
    assert sorted(g["indices"][0] for g in singles) == [1, 3]
    for single in singles:
        assert single.get("rle") is None
        assert single.get("bbox_xyxy") is None
        assert single.get("area") is None


def test_dedup_scores_seed_the_grouping(client, image_path):
    base = {"image_path": str(image_path), "rles": _bridge_rles(), "iou": 0.8}

    def indices(scores):
        return [g["indices"] for g in _dedup_groups(client, dict(base, scores=scores))]

    # big scores highest: it seeds first and absorbs both smalls
    assert indices([0.9, 0.5, 0.4]) == [[0, 1, 2]]
    # s2 scores highest: it seeds first and claims big; s1 is left alone
    assert indices([0.1, 0.5, 0.9]) == [[0, 2], [1]]
    # a null score marks a saved anchor and outranks every number: s1 seeds
    # before s2's 0.9 and claims big; s2 is left alone
    assert indices([0.5, None, 0.9]) == [[0, 1], [2]]
