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

"""Persistence over the directory's COCO file: /save, /delete_annotation,
/export_zephyr, /dir_counts, /dir_stats."""
from __future__ import annotations

import numpy as np
import PIL.Image
import pytest

import masker_support
import masker.io.coco


def _coco(directory):
    return directory / masker.io.coco.CocoLayout.ANNOTATIONS_NAME


def _two_images(tmp_path, a="a.png", b="b.png"):
    return masker_support.png(tmp_path / a), masker_support.png(tmp_path / b)


def _save(client, image_path, *instances, **body):
    """POST /save with (label, rect) instances; asserts 200, returns the body."""
    code, resp = masker_support.post_json(client, "/save", {
        "image_path": str(image_path), "coco_file": None, "deletes": [],
        "instances": [{"label": lab, "rle": masker_support.mask_for(*rect).to_json()}
                      for lab, rect in instances],
        **body,
    })
    assert code == 200
    return resp


# =============================================================================
# /save, /export_zephyr
# =============================================================================


def test_save_creates_coco_file(client, image_path, tmp_path):
    body = _save(client, image_path,
                 ("widget", (5, 10, 25, 20)), ("gadget", (30, 5, 40, 15)))
    coco_path = _coco(tmp_path)
    assert body["coco_file"] == str(coco_path)
    assert [s["annotation_id"] for s in body["saved"]] == [1, 2]
    assert body["saved"][0]["bbox_xyxy"] == [5, 10, 25, 20]

    stored = masker.io.coco.load_coco(coco_path)
    assert stored["images"][0]["file_name"] == image_path.name
    ann = stored["annotations"][0]
    assert ann["bbox"] == [5, 10, 20, 10]  # xywh from the mask
    assert ann["area"] == 200


def test_save_accumulates_images_in_one_file(client, tmp_path):
    a, b = _two_images(tmp_path, "img.png", "img.jpg")  # same stem: separate entries
    body = _save(client, a, ("widget", (5, 10, 25, 20)))
    _save(client, b, ("gadget", (30, 5, 40, 15)), ("widget", (0, 0, 4, 4)),
          coco_file=body["coco_file"])
    stored = masker.io.coco.load_coco(_coco(tmp_path))
    assert {img["file_name"] for img in stored["images"]} == {"img.png", "img.jpg"}
    assert len({ann["id"] for ann in stored["annotations"]}) == 3
    assert [c["name"] for c in stored["categories"]] == ["widget", "gadget"]
    assert len(masker_support.load_body(client, a)["saved"]) == 1
    assert len(masker_support.load_body(client, b)["saved"]) == 2


def test_save_appends_to_existing_file(client, image_path, tmp_path):
    coco_path = masker_support.write_coco(tmp_path / "ann.json", image_path)
    body = _save(client, image_path, ("widget", (30, 5, 40, 15)),
                 coco_file=str(coco_path))
    assert body["coco_file"] == str(coco_path)
    stored = masker.io.coco.load_coco(coco_path)
    assert len(stored["annotations"]) == 2
    assert len(stored["categories"]) == 1  # existing label reused, not duplicated
    assert not _coco(tmp_path).exists()  # no second file beside it

    body = masker_support.load_body(client, image_path)
    assert body["coco_file"] == str(coco_path)
    assert len(body["saved"]) == 2


def _export(client, image_path, labels, force=False):
    return masker_support.post_json(client, "/export_zephyr", {
        "image_path": str(image_path), "labels": labels, "force": force, "missing_ok": False})


def test_export_zephyr_writes_union_masks(client, tmp_path):
    a, b = _two_images(tmp_path)
    _save(client, a, ("person", (5, 10, 25, 20)), ("sky", (0, 0, 64, 5)))
    _save(client, b, ("tree", (0, 0, 4, 4)))

    code, body = _export(client, a, ["person", "sky"])
    assert code == 200
    assert body == {"dir": str(tmp_path), "written": 1, "skipped": 1}
    mask_file = tmp_path / "a_mask.tiff"
    assert not (tmp_path / "b_mask.tiff").exists()  # no selected labels: no file
    arr = np.asarray(PIL.Image.open(mask_file))
    assert arr.shape == (48, 64)
    assert arr[15, 10] == 0 and arr[2, 40] == 0  # person and sky both ignored
    assert arr[40, 40] == 255

    # a narrower re-export replaces the file
    _export(client, a, ["person"])
    assert np.asarray(PIL.Image.open(mask_file))[2, 40] == 255  # sky kept


def test_export_zephyr_force_writes_blank_masks(client, tmp_path):
    a, b = _two_images(tmp_path)
    _save(client, a, ("person", (5, 10, 25, 20)))
    stale = tmp_path / "b_mask.tiff"
    PIL.Image.new("L", (64, 48), 0).save(stale)

    code, body = _export(client, a, ["person"], force=True)
    assert code == 200
    assert body == {"dir": str(tmp_path), "written": 2, "skipped": 0}
    arr = np.asarray(PIL.Image.open(stale))
    assert arr.shape == (48, 64) and (arr == 255).all()  # stale mask cleared


# =============================================================================
# deletes: /delete_annotation and /save deletes
# =============================================================================


def test_delete_annotation(client, image_path, tmp_path):
    coco_path = masker_support.write_coco(tmp_path / "ann.json", image_path)
    code, body = masker_support.post_json(
        client, "/delete_annotation",
        {"coco_file": str(coco_path), "annotation_id": 1,
         "image_path": str(image_path), "bbox_xyxy": [5, 10, 25, 20]},
    )
    assert code == 200
    assert body == {"coco_file": str(coco_path)}
    assert masker.io.coco.load_coco(coco_path)["annotations"] == []


@pytest.mark.parametrize("instances,stored_bboxes,saved", [
    ((), [], 0),  # deletes-only save echoes no instances
    ((("widget", (30, 5, 40, 15)),), [[30, 5, 10, 10]], 1),
], ids=["deletes-only", "adds-and-deletes"])
def test_save_applies_deletes_and_adds(client, image_path, tmp_path,
                                       instances, stored_bboxes, saved):
    coco_path = masker_support.write_coco(tmp_path / "ann.json", image_path)
    body = _save(client, image_path, *instances, coco_file=str(coco_path),
                 deletes=[{"annotation_id": 1, "bbox_xyxy": [5, 10, 25, 20]}])
    assert len(body["saved"]) == saved
    stored = masker.io.coco.load_coco(coco_path)
    assert [a["bbox"] for a in stored["annotations"]] == stored_bboxes


@pytest.mark.parametrize("route", ["/delete_annotation", "/save"])
@pytest.mark.parametrize("other_owner,annotation_id,bbox_xyxy", [
    (True, 1, [5, 10, 25, 20]),   # id belongs to another image
    (False, 1, [5, 10, 25, 30]),  # bbox drifted beyond tolerance
    (False, 99, [5, 10, 25, 20]), # unknown id
], ids=["wrong-owner", "bbox-drift", "unknown-id"])
def test_stale_delete_is_409_and_writes_nothing(
        client, image_path, tmp_path, route, other_owner, annotation_id, bbox_xyxy):
    coco_path = masker_support.write_coco(tmp_path / "ann.json", image_path)
    before = coco_path.read_bytes()
    sent = masker_support.png(tmp_path / "other.png") if other_owner else image_path
    ref = {"annotation_id": annotation_id, "bbox_xyxy": bbox_xyxy}
    if route == "/save":
        body = {"image_path": str(sent), "coco_file": str(coco_path),
                "instances": [], "deletes": [ref]}
    else:
        body = {"image_path": str(sent), "coco_file": str(coco_path), **ref}
    code, resp = masker_support.post_json(client, route, body)
    assert code == 409 and "out of date" in resp["error"]
    assert coco_path.read_bytes() == before


# =============================================================================
# GET /dir_counts
# =============================================================================


def _annotation(ann_id, image_id, category_id, bbox=(1, 2, 5, 3), **extra):
    return masker_support.annotation(ann_id, image_id, category_id, bbox, **extra)


def _dataset():
    """Three images (one without annotations), two categories."""
    return masker_support.coco_dataset(
        [masker_support.coco_image(i, name, 8, 8)
         for i, name in enumerate(("b.png", "a.png", "c.png"), 1)],
        [_annotation(1, 1, 1), _annotation(2, 1, 1), _annotation(3, 1, 2), _annotation(4, 2, 2)],
        ["widget", "gadget"])


@pytest.fixture
def image_dir(tmp_path):
    d = tmp_path / "imgs"
    d.mkdir()
    for name in ("a.png", "b.png", "c.png"):
        masker_support.png(d / name, (8, 8))
    return d


def _seed(image_dir, dataset=None):
    masker.io.coco.save_coco(dataset or _dataset(), _coco(image_dir))


def test_counts_aggregated_from_file(client, image_dir):
    _seed(image_dir)
    code, body = masker_support.get_json(client, "/dir_counts", path=str(image_dir / "a.png"))
    assert code == 200
    assert body["dir"] == str(image_dir)  # a file path queries its directory
    assert {c["name"] for c in body["categories"]} == {"widget", "gadget"}
    by_name = {img["name"]: img["counts"] for img in body["images"]}
    assert list(by_name) == ["a.png", "b.png", "c.png"]  # name-sorted
    assert by_name["b.png"] == {"widget": 2, "gadget": 1}
    assert by_name["a.png"] == {"gadget": 1}
    assert by_name["c.png"] == {}  # in the dataset, zero annotations


# =============================================================================
# GET /dir_stats
# =============================================================================


def test_stats_bins_and_extremes(client, image_dir):
    # widget: areas 1000 and 1 << 30; gadget: area 15
    _seed(image_dir, masker_support.coco_dataset(
        [masker_support.coco_image(1, "a.png", 8, 8)],
        [_annotation(1, 1, 1, [0, 0, 40, 25]), _annotation(2, 1, 1, [0, 0, 8, 8], area=1 << 30),
         _annotation(3, 1, 2)],
        ["widget", "gadget"]))
    code, body = masker_support.get_json(client, "/dir_stats", path=str(image_dir))
    assert code == 200
    assert body["edges"][0] == 16 and body["edges"][-1] == 2**24
    by_name = {l["name"]: l for l in body["labels"]}
    assert set(by_name) == {"widget", "gadget"}
    widget, gadget = by_name["widget"], by_name["gadget"]
    assert widget["count"] == 2
    bins = widget["area"]["bins"]
    assert len(bins) == len(body["edges"]) - 1
    assert bins[5] == 1   # 1000 in [512, 1024)
    assert bins[-1] == 1  # 1 << 30 clamps into the last bin
    assert (widget["area"]["min"], widget["area"]["max"]) == (1000, 1 << 30)
    assert widget["bbox"] == {"w_min": 8, "w_max": 40, "h_min": 8, "h_max": 25}
    assert gadget["count"] == 1
    assert gadget["area"]["bins"][0] == 1  # area 15 clamps into the first bin
    assert (gadget["area"]["min"], gadget["area"]["max"]) == (15, 15)
    assert gadget["bbox"] == {"w_min": 5, "w_max": 5, "h_min": 3, "h_max": 3}
