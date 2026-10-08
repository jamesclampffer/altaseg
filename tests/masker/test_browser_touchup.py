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

# Touch-up and draw-lock chrome in headless Chromium against the real
# /mask_crop, /update_annotation and /draw_mask. The in-place rewrite of a
# saved mask is test_browser.py::test_touchup_rewrites_saved_mask_in_place.
from __future__ import annotations

import json

import pytest

import masker_support
import masker.io.coco

pytest.importorskip("playwright.sync_api")

DRAW_BAR = ("dpoly", "dline", "dbrush", "drect", "dapply", "dsave")


def _shown(page):
    # draw-bar buttons on show
    return [i for i in DRAW_BAR if page.locator("#" + i).is_visible()]


def _classes(page, selector):
    return (page.get_attribute(selector, "class") or "").split()


def _pick_touchup(page, item):
    # selection popover -> touch up -> item; the lock is up
    page.wait_for_selector("#selpop", state="visible")
    page.click("#seltouch")
    page.click("#" + item)
    page.wait_for_selector("#drawbar", state="visible")


def test_touchup_entry_presets_and_load_guard(page, served_app, image_path, tmp_path):
    # stock file: widget #1, bbox [5, 10, 25, 20], mask [10:20, 5:25]
    masker_support.write_coco(tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image_path)
    masker_support.open_workspace(page, served_app, image_path)
    page.wait_for_selector("#instances .irow")

    # the menu lives in the selection popover; a pick with nothing selected
    # (dispatched: the hidden item has no other route) is refused
    page.dispatch_event("#trectdel", "click")
    assert page.locator("#status").text_content() == "Select an instance to touch up."
    assert not page.locator("#drawbar").is_visible()

    masker_support.image_click(page, 15, 15)
    page.wait_for_selector("#selpop", state="visible")
    page.click("#seltouch")
    assert page.locator("#touchmenu").is_visible()
    page.click("#seltouch")
    assert not page.locator("#touchmenu").is_visible()

    # with the popover at the bottom of the canvas the menu opens upward
    page.evaluate(
        "document.getElementById('selpop').style.top = "
        "(document.getElementById('canvaswrap').clientHeight - 40) + 'px'"
    )
    page.click("#seltouch")
    assert "up" in _classes(page, "#touchmenu")
    assert page.evaluate(
        "document.getElementById('touchmenu').getBoundingClientRect().bottom"
        " <= document.getElementById('canvaswrap').getBoundingClientRect().bottom"
    )
    page.click("#seltouch")
    page.evaluate("document.getElementById('selpop').style.top = '4px'")
    page.click("#seltouch")
    assert "up" not in _classes(page, "#touchmenu")
    with page.expect_response("**/mask_crop") as crop:
        page.click("#trectdel")
    page.wait_for_selector("#drawbar", state="visible")
    assert not page.locator("#touchmenu").is_visible()

    # touch-up chrome: brush / rect only, Save manual updates instead of
    # Apply, the picked preset lit, nothing drawn yet
    assert _shown(page) == ["dbrush", "drect", "dsave"]
    assert "on" in _classes(page, "#drect")
    assert "danger" in _classes(page, "#derase")
    assert page.locator("#modehint").text_content().startswith('Touching up "widget | #1"')
    assert page.locator("#dsave").is_disabled()
    # the base swaps to a native crop of the frozen viewport
    assert crop.value.request.post_data_json == {
        "image_path": str(image_path),
        "rle": masker_support.mask_for(5, 10, 25, 20).to_json(),
        "rect": [0, 0, 64, 48]}
    assert crop.value.json()["rect"] == [0, 0, 64, 48]

    # the lock refuses image loads with its own message
    page.press("#path", "Enter")
    masker_support.wait_text(page, "status", "Touch-up in progress")
    assert page.locator("#drawbar").is_visible()


def test_stale_touchup_unlocks_and_resyncs(page, served_app, image_path, tmp_path):
    coco_path = masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image_path)
    masker_support.open_workspace(page, served_app, image_path)
    page.wait_for_selector("#instances .irow")
    assert page.locator("#instances .irow .dims").text_content() == "20x10"

    masker_support.image_click(page, 15, 15)
    _pick_touchup(page, "tbrushadd")
    assert "on" in _classes(page, "#dbrush")
    assert "danger" not in _classes(page, "#derase")
    masker_support.image_drag(page, 30, 30, 40, 40)
    page.wait_for_function("!document.getElementById('dsave').disabled")

    # the annotation moves on disk after the page loaded it
    dataset = json.loads(coco_path.read_text())
    dataset["annotations"][0]["bbox"] = [5, 10, 30, 10]
    coco_path.write_text(json.dumps(dataset))
    with page.expect_response("**/update_annotation") as update:
        page.click("#dsave")
    assert update.value.status == 409
    page.wait_for_selector("#drawbar", state="hidden")
    masker_support.wait_text(page, "toast", "out of date")
    assert page.locator("#toast").text_content().endswith(" - reloading.")
    # /load resync: the row follows the bbox on disk; the file is untouched
    page.wait_for_function(
        "document.querySelector('#instances .irow .dims').textContent === '30x10'")
    assert json.loads(coco_path.read_text()) == dataset


def test_draw_lock_then_touchup_of_pending(page, served_app, image_path, tmp_path):
    routes = masker_support.post_routes(page)
    masker_support.open_workspace(page, served_app, image_path)

    # draw lock, nothing selected: every tool and Apply, erase off
    page.fill("#promptText", "widget")
    page.click("#zlock")
    page.wait_for_selector("#drawbar", state="visible")
    assert _shown(page) == ["dpoly", "dline", "dbrush", "drect", "dapply"]
    assert "danger" not in _classes(page, "#derase")
    assert page.locator("#modehint").text_content().startswith("Drawing a new instance")
    page.click("#drect")
    masker_support.image_drag(page, 8, 8, 40, 40)
    with page.expect_response("**/draw_mask") as drawn:
        page.click("#dapply")
    page.wait_for_selector("#drawbar", state="hidden")
    assert drawn.value.request.post_data_json["base_rle"] is None
    # requests arrive in order: no base, so no crop before the Apply
    assert "/mask_crop" not in routes
    pending_rle = drawn.value.json()["rle"]

    # Apply selects the new pending instance; its mask decodes async (app
    # state, no DOM signal) before touch-up accepts it
    page.wait_for_function("state.pending.length === 1 && state.pending[0].maskImg")
    _pick_touchup(page, "trectdel")
    masker_support.image_drag(page, 2, 2, 46, 46)
    page.wait_for_function("!document.getElementById('dsave').disabled")
    with page.expect_response("**/draw_mask") as erased:
        page.click("#dsave")
    body = erased.value.request.post_data_json
    assert (body["image_path"], body["rect"], body["base_rle"]) == (
        str(image_path), [0, 0, 64, 48], pending_rle)
    assert [(op["kind"], op["erase"]) for op in body["ops"]] == [("polygon", True)]
    assert erased.value.json() == {"empty": True}
    # erasing everything keeps the lock; the instance stays pending
    masker_support.wait_text(page, "status", "Touch-up erased the whole mask")
    assert page.locator("#drawbar").is_visible()
    assert page.locator("#save").text_content() == "Save | 1 new"

    # undo, cut the right side only: applied in memory, still pending
    page.click("#dundo")
    masker_support.image_drag(page, 24, 2, 46, 46)
    page.wait_for_function("!document.getElementById('dsave').disabled")
    with page.expect_response("**/draw_mask") as cut:
        page.click("#dsave")
    page.wait_for_selector("#drawbar", state="hidden")
    masker_support.wait_text(page, "status", 'Manual updates applied to pending "widget"')
    assert page.locator("#save").text_content() == "Save | 1 new"
    x0, y0, x1, y1 = cut.value.json()["bbox_xyxy"]
    assert page.locator("#instances .irow .dims").text_content() == "{}x{}".format(
        round(x1 - x0), round(y1 - y0))
    assert routes.count("/update_annotation") == 0

    with page.expect_response("**/save"):
        page.click("#save")
    data = json.loads((tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME).read_text())
    mask = masker_support.decode_rle(data["annotations"][0]["segmentation"])
    assert mask[10:38, 10:22].all()   # left of the cut
    assert not mask[:, 25:].any()     # the removed right side
