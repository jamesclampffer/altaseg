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

from __future__ import annotations

import json

import pytest

import masker_support
import masker.io.coco

pytest.importorskip("playwright.sync_api")

ROW_ELEMENTS = 8  # wrap, row, twist, checkbox, swatch, text, select, current-parent option


def _labels_page(page, url, tmp_path, names, parents=()):
    image = masker_support.png(tmp_path / "img1.png")
    dataset = masker_support.coco_dataset(
        [masker_support.coco_image(1, "img1.png", 64, 48)], [], names)
    for cat, parent in zip(dataset["categories"], parents):
        cat["supercategory"] = parent
    coco = masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image, dataset)
    page.add_init_script(masker_support.COUNT_CREATES_JS)
    masker_support.open_workspace(page, url, image)
    page.wait_for_function("document.getElementById('imgidx').textContent === '1 / 1'")
    page.click("#lbltoggle")
    masker_support.creates(page)
    return coco


def _picker(page, name):
    return page.locator('#labeltree .lrow:has(input.pick[value="{}"]) select'.format(name))


def _options(select):
    return select.locator("option").all_text_contents()


def _wait_status(page, text):
    page.wait_for_function("t => document.getElementById('status').textContent === t", arg=text)


def _parents(coco):
    return {c["name"]: c["supercategory"] for c in json.loads(coco.read_text())["categories"]}


def test_group_heading_parent_picker(page, served_app, tmp_path):
    coco = _labels_page(page, served_app, tmp_path, ["vehicle", "car", "cat", "dog"],
                        ["", "vehicle", "pets", "pets"])

    # a label under a group heading shows the heading as its parent; the
    # unopened picker holds only that option
    cat = _picker(page, "cat")
    assert cat.input_value() == "pets"
    assert _options(cat) == ["# pets"]

    # opening fills from the shared list: heading offered, own subtree left
    # out, current parent still selected; reopening creates nothing
    cat.click()
    assert _options(cat) == ["(no parent)", "# pets", "^ vehicle", "^ car", "^ dog"]
    assert masker_support.creates(page) == 5
    assert cat.input_value() == "pets"
    assert cat.evaluate("s => s.selectedOptions[0].textContent") == "# pets"
    cat.click()
    assert masker_support.creates(page) == 0
    assert cat.input_value() == "pets"

    # picking a heading posts it as the parent
    car = _picker(page, "car")
    car.click()
    assert "# pets" in _options(car)
    with page.expect_request("**/edit_labels") as req:
        car.select_option("pets")
    assert req.value.post_data_json == {
        "coco_file": str(coco), "op": "parent", "name": "car", "parent": "pets"}
    _wait_status(page, '"car" parented under "pets".')
    assert _options(car) == ["# pets"]  # rebuilt row, unopened
    assert _parents(coco)["car"] == "pets"

    # clearing posts the empty parent
    cat.click()
    with page.expect_request("**/edit_labels") as req:
        cat.select_option(label="(no parent)")
    assert req.value.post_data_json == {
        "coco_file": str(coco), "op": "parent", "name": "cat", "parent": ""}
    _wait_status(page, '"cat" moved to the root.')
    assert _options(cat) == ["(no parent)"]
    assert _parents(coco)["cat"] == ""


def test_label_tree_render_is_linear(page, served_app, tmp_path):
    n = 120
    _labels_page(page, served_app, tmp_path, ["label{:03d}".format(i) for i in range(n)])

    # a filter re-render builds each row once, with one option
    page.fill("#lblfilter", "label")  # matches every label
    assert page.locator("#labeltree option").count() == n
    assert masker_support.creates(page) == n * ROW_ELEMENTS
    page.fill("#lblfilter", "label00")  # label000..label009
    assert page.locator("#labeltree option").count() == 10
    assert masker_support.creates(page) == 10 * ROW_ELEMENTS

    # first focus fills the picker once: (no parent) + every label but its own
    sel = _picker(page, "label000")
    sel.focus()
    assert masker_support.creates(page) == n
    sel.blur()
    sel.focus()
    assert masker_support.creates(page) == 0
