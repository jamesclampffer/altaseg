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

"""Dataset-wide label edits (/edit_labels ops drop, add, parent, merge) and
/load's label-panel fields."""
from __future__ import annotations

import json

import masker_support


def _dataset():
    return masker_support.coco_dataset(
        [masker_support.coco_image(1, "img1.png", 64, 48),
         masker_support.coco_image(2, "img2.png", 64, 48)],
        [masker_support.annotation(1, 1, 1, [2, 2, 10, 10]),
         masker_support.annotation(2, 1, 2, [20, 5, 8, 8]),
         masker_support.annotation(3, 2, 2, [4, 4, 6, 6])],
        ["cat", "dog"])


def _setup(tmp_path, dataset=None):
    """Both images plus ann.json; returns the COCO path."""
    for name in ("img1.png", "img2.png"):
        masker_support.png(tmp_path / name)
    (tmp_path / "ann.json").write_text(json.dumps(dataset or _dataset()))
    return tmp_path / "ann.json"


def _edit(client, ann, **body):
    return masker_support.post_json(client, "/edit_labels", {"coco_file": str(ann), **body})


def _load(client, tmp_path, name="img1.png"):
    return masker_support.load_body(client, tmp_path / name)


def _category(body, name):
    """The category row named name in a /load body."""
    return next(c for c in body["categories"] if c["name"] == name)


# --- /load label-panel fields ------------------------------------------------


def test_load_reports_counts_and_tree(client, tmp_path):
    ann = _setup(tmp_path)
    body = _load(client, tmp_path)
    assert body["coco_file"] == str(ann)
    by_name = {c["name"]: c for c in body["categories"]}
    assert by_name["cat"]["total"] == 1 and by_name["cat"]["in_image"] == 1
    assert by_name["dog"]["total"] == 2 and by_name["dog"]["in_image"] == 1
    assert [n["name"] for n in body["tree"]] == ["cat", "dog"]


# --- drop --------------------------------------------------------------------


def test_drop_labels_removes_across_images(client, tmp_path):
    ann = _setup(tmp_path)
    code, body = _edit(client, ann, op="drop", labels=["dog"])
    assert code == 200
    assert body["removed_annotations"] == 2
    assert [c["name"] for c in body["categories"]] == ["cat"]
    assert body["coco_file"] == str(ann)
    body = _load(client, tmp_path)
    assert [(c["name"], c["total"]) for c in body["categories"]] == [("cat", 1)]


def test_drop_parent_label_reroots_children_not_cascades(client, tmp_path):
    ann = _setup(tmp_path)
    _edit(client, ann, op="parent", name="dog", parent="cat")
    code, body = _edit(client, ann, op="drop", labels=["cat"])
    assert code == 200
    assert body["removed_annotations"] == 1  # only cat's bbox, never dog's
    assert [c["name"] for c in body["categories"]] == ["dog"]
    assert _category(_load(client, tmp_path), "dog")["supercategory"] == ""


# --- add ---------------------------------------------------------------------


def test_add_label_registers_new_category(client, tmp_path):
    ann = _setup(tmp_path)
    code, body = _edit(client, ann, op="add", label="bird")
    assert code == 200
    assert body["categories"][-1] == {"id": 3, "name": "bird"}
    # persisted without an annotation carrying it: a fresh app still lists it
    fresh = masker_support.make_client(tmp_path, masker_support.StubMaskGenerator())
    assert [c["name"] for c in _load(fresh, tmp_path)["categories"]] == [
        "cat", "dog", "bird"]


# --- parent ------------------------------------------------------------------


def test_set_parent_shows_in_tree(client, tmp_path):
    ann = _setup(tmp_path)
    code, body = _edit(client, ann, op="parent", name="dog", parent="cat")
    assert code == 200
    assert body["coco_file"] == str(ann) and body["removed_annotations"] == 0
    # the link is in the file: a fresh app reads it back
    fresh = masker_support.make_client(tmp_path, masker_support.StubMaskGenerator())
    body = _load(fresh, tmp_path)
    assert _category(body, "dog")["supercategory"] == "cat"
    assert [n["name"] for n in body["tree"]] == ["cat"]
    assert [n["name"] for n in body["tree"][0]["children"]] == ["dog"]

    # cat under dog would close a cycle
    assert _edit(client, ann, op="parent", name="cat", parent="dog")[0] == 400

    # an empty parent moves the label back to the root
    assert _edit(client, ann, op="parent", name="dog", parent="")[0] == 200
    body = _load(client, tmp_path)
    assert _category(body, "dog")["supercategory"] == ""
    assert [n["name"] for n in body["tree"]] == ["cat", "dog"]


def test_set_parent_to_group_heading(client, tmp_path):
    ds = _dataset()
    ds["categories"].append({"id": 3, "name": "fish", "supercategory": "pets"})
    ann = _setup(tmp_path, ds)
    assert _edit(client, ann, op="parent", name="dog", parent="pets")[0] == 200
    body = _load(client, tmp_path)
    assert _category(body, "dog")["supercategory"] == "pets"  # client shows the heading from this
    pets = next(n for n in body["tree"] if n["name"] == "pets")
    assert pets["id"] is None  # heading, not a label
    assert [n["name"] for n in pets["children"]] == ["dog", "fish"]  # category order


# --- merge -------------------------------------------------------------------


def test_merge_labels_renames_and_folds(client, tmp_path):
    ann = _setup(tmp_path)
    code, body = _edit(client, ann, op="merge", labels=["cat", "dog"], new_name="pet")
    assert code == 200
    assert body == {"coco_file": str(ann), "removed_annotations": 0,
                    "categories": [{"id": 1, "name": "pet"}]}
    body = _load(client, tmp_path)
    assert [(c["name"], c["total"], c["in_image"]) for c in body["categories"]] == [("pet", 3, 2)]
    assert [n["name"] for n in body["tree"]] == ["pet"]
