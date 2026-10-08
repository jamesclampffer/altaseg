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

"""Headless-Chromium coverage of the masker workspace page: canvas geometry,
mouse prompts, real HTTP encoding, thumbnails that decode. Every test
asserts no console or page errors. Batch, panel, labels, sweep and touch-up
chrome are in test_browser_*.py.
"""
from __future__ import annotations

import json

import numpy as np
import PIL.Image
import pytest

import masker_support
import masker.annotations.instance_mask
import masker.io.coco

pytest.importorskip("playwright.sync_api")


@pytest.fixture
def big_image(tmp_path):
    """1400x1000 PNG: wider than the model side, so Run sweeps and a zoom
    step fetches a crop preview."""
    path = tmp_path / "big.png"
    PIL.Image.new("RGB", (1400, 1000), (40, 70, 40)).save(path)
    return path


def test_workspace_renders_and_loads(page, served_app, image_path):
    masker_support.open_workspace(page, served_app, image_path)
    # the stylesheet route delivered (a 404 would be swallowed by the
    # Failed-to-load-resource console filter)
    assert page.evaluate(
        "getComputedStyle(document.getElementById('topbar')).display"
    ) == "flex"
    assert page.locator("#tabs div").count() == 5
    assert page.locator("#rail").is_visible()
    assert "64" in page.locator("#dims").text_content()
    # the position counter is filled asynchronously by /fsmeta after /load;
    # the test image is alone in its directory
    page.wait_for_function(
        "document.getElementById('imgidx').textContent === '1 / 1'"
    )

    # the model input float folds to a corner pill and back
    page.wait_for_selector("#modelview", state="visible")
    page.click("#modelhide")
    assert not page.locator("#modelview").is_visible()
    assert page.locator("#modelpill").is_visible()
    page.click("#modelpill")
    assert page.locator("#modelview").is_visible()
    assert not page.locator("#modelpill").is_visible()

    # tool tabs, review page round-trip
    page.click("#tabs div:nth-child(5)")  # Erase
    assert page.locator("#modenote").is_visible()
    page.click("#tabs div:nth-child(1)")
    page.click("#reviewbtn")
    assert page.locator("#reviewempty").is_visible()
    page.click("#backwork")
    assert page.locator("#rail").is_visible()


def _seed_square(image_path, tmp_path):
    # widget #1 as the square [8:40, 8:40] of the 48x64 image
    dataset = masker_support.coco_with_rle_annotation(image_path)
    square = masker_support.mask_for(8, 8, 40, 40)
    dataset["annotations"][0].update(
        segmentation=square.to_json(), bbox=[8, 8, 32, 32], area=square.area)
    return masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image_path, dataset)


def test_erase_edit_subtracts_from_saved_mask(page, served_app, image_path, tmp_path):
    """An erase edit of a saved instance must subtract from its raw mask
    pixels on disk. Drawing a new instance is in test_browser_touchup.py."""
    coco_path = _seed_square(image_path, tmp_path)
    masker_support.open_workspace(page, served_app, image_path)
    page.wait_for_selector("#instances .irow")

    # select the saved square, cut a polygon hole, apply, save
    masker_support.image_click(page, 24, 24)
    page.wait_for_selector("#selpop", state="visible")
    page.click("#zlock")
    page.wait_for_selector("#drawbar", state="visible")
    assert "Editing" in page.locator("#modehint").text_content()
    page.click("#derase")
    page.click("#dpoly")
    for x, y in ((16, 16), (32, 16), (32, 32)):
        masker_support.image_click(page, x, y)
    masker_support.image_click(page, 16, 16)
    page.click("#dapply")
    page.wait_for_selector("#drawbar", state="hidden")
    assert "1 new, 1 del" in page.locator("#save").text_content()
    page.click("#save")
    page.wait_for_function(
        "document.getElementById('saved').textContent === '* saved'"
    )

    data = json.loads(coco_path.read_text())
    assert len(data["annotations"]) == 1     # the edit replaced the original
    mask = masker_support.decode_rle(data["annotations"][0]["segmentation"])
    assert not mask[18:23, 25:30].any()      # interior of the erased triangle
    assert mask[11:14, 11:37].all()          # untouched band above the hole
    assert mask[28:31, 18:21].all()          # below the diagonal, still painted


def test_touchup_rewrites_saved_mask_in_place(page, served_app, image_path, tmp_path):
    """Touch up (selection popover) edits the selected annotation's mask on
    disk: the annotation keeps its id, nothing is staged, and undo unwinds
    the last shape before saving. The touch-up chrome itself is pinned by
    test_browser_touchup.py; the pixel rewrite by test_web_draw.py."""
    coco_path = _seed_square(image_path, tmp_path)
    masker_support.open_workspace(page, served_app, image_path)
    page.wait_for_selector("#instances .irow")

    # select the saved square, pick Rect | remove from the dropdown
    masker_support.image_click(page, 24, 24)
    page.wait_for_selector("#selpop", state="visible")
    page.click("#seltouch")
    page.wait_for_selector("#touchmenu", state="visible")
    page.click("#trectdel")
    page.wait_for_selector("#drawbar", state="visible")

    # a dragged rect clears its interior; save rewrites the COCO file
    masker_support.image_drag(page, 16, 16, 32, 32)
    page.wait_for_function("!document.getElementById('dsave').disabled")
    page.click("#dsave")
    page.wait_for_selector("#drawbar", state="hidden")
    page.wait_for_function(
        "document.getElementById('status').textContent.startsWith('Manual updates saved')"
    )
    assert page.locator("#saved").text_content() == "* saved"  # nothing staged
    assert "0 new" in page.locator("#save").text_content()

    data = json.loads(coco_path.read_text())
    assert len(data["annotations"]) == 1
    assert data["annotations"][0]["id"] == 1
    assert data["annotations"][0]["bbox"] == [8, 8, 32, 32]  # outer square stands

    # undo unwinds the last shape; cancel with nothing left unlocks at once
    masker_support.image_click(page, 24, 10)
    page.wait_for_selector("#selpop", state="visible")
    page.click("#seltouch")
    page.click("#tbrushadd")
    page.wait_for_selector("#drawbar", state="visible")
    masker_support.image_drag(page, 20, 20, 28, 28)
    page.wait_for_function("!document.getElementById('dundo').disabled")
    page.click("#dundo")
    page.wait_for_function("document.getElementById('dsave').disabled")
    page.click("#dcancel")
    page.wait_for_selector("#drawbar", state="hidden")


def test_outline_rings_each_instance(page, served_app, big_image, tmp_path):
    """Outline mode rings every mask separately: two abutting masks keep
    their shared edge, each ring in its own label colour, and interiors show
    the image. The big image keeps the masks at or below screen resolution,
    so smoothing leaves the ring edges sharp."""
    # A and B share the edge x=500 for y in [100, 600); a tab on each keeps
    # that edge clear of both bounding boxes, whose strokes use the same colours
    def mask(*rects):
        arr = np.zeros((1000, 1400), dtype=np.uint8)
        for x0, y0, x1, y1 in rects:
            arr[y0:y1, x0:x1] = 1
        return masker.annotations.instance_mask.InstanceMask.from_array(arr)

    a = mask((100, 100, 500, 600), (400, 600, 600, 700))
    b = mask((500, 100, 900, 600), (400, 20, 600, 100))
    dataset = masker_support.coco_dataset(
        [masker_support.coco_image(1, big_image.name, 1400, 1000)],
        [masker_support.annotation(1, 1, 1, [100, 100, 500, 600], a.to_json(), area=a.area),
         masker_support.annotation(2, 1, 2, [400, 20, 500, 580], b.to_json(), area=b.area)],
        ["widget", "gadget"])
    (tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME).write_text(json.dumps(dataset))
    masker_support.open_workspace(page, served_app, big_image)
    page.wait_for_function("document.querySelectorAll('#instances .irow').length === 2")
    page.evaluate("""() => {
      const g = document.getElementById('view').getContext('2d');
      // canvas pixel dx px right of original-image (x, y)
      window.px = (x, y, dx) => {
        const [cx, cy] = viewer.imageToCanvas(x, y);
        return Array.from(g.getImageData(Math.floor(cx) + dx, Math.floor(cy), 1, 1).data);
      };
      // how far that pixel strays from the image's solid (40, 70, 40)
      window.stray = (x, y, dx) => {
        const p = px(x, y, dx);
        return Math.max(Math.abs(p[0] - 40), Math.abs(p[1] - 70), Math.abs(p[2] - 40));
      };
    }""")

    page.click("#outline")
    # each ring sits just outside its own mask: A's right of the shared edge,
    # B's left of it. The masks decode asynchronously and the frame before
    # the toggle's redraw may still show the fills, which also straddle the
    # edge, so the wait includes A's interior going clear.
    page.wait_for_function(
        "stray(500, 350, 1) > 40 && stray(500, 350, -1) > 40 && stray(300, 350, 0) < 8")
    right, left = page.evaluate("[px(500, 350, 1), px(500, 350, -1)]")
    assert max(abs(r - l) for r, l in zip(right, left)) > 40  # two colours, two rings
    assert page.evaluate("stray(700, 350, 0)") < 8   # inside B: no fill
    assert page.evaluate("stray(1200, 800, 0)") < 8  # off both masks

    page.click("#outline")
    # the fill composite is back: both palette colours sit ~150 from the
    # image in their strongest channel, ~38 at the 25% fill
    page.wait_for_function("stray(300, 350, 0) > 25")


def test_filmstrip_window(page, served_app, tmp_path):
    names = ["i%02d.png" % i for i in range(1, 16)]
    for name in names:
        masker_support.png(tmp_path / name, (32, 24))
    titles = "[...document.querySelectorAll('#filmstrip img')].map(i => i.title)"
    marked = "document.querySelector('#filmstrip img.on').title"

    # 12 thumbnails; the window starts at the directory's first image when
    # the loaded one is near the start
    masker_support.open_workspace(page, served_app, tmp_path / names[0])
    page.wait_for_function(
        "document.getElementById('imgidx').textContent === '1 / 15'"
    )
    assert page.evaluate(titles) == names[:12]
    assert page.evaluate(marked) == names[0]
    # thumbnails decode; a click navigates
    page.wait_for_function(
        "[...document.querySelectorAll('#filmstrip img')].every(i => i.naturalWidth > 0)"
    )
    page.locator("#filmstrip img").nth(2).click()
    page.wait_for_function(
        "document.getElementById('imgidx').textContent === '3 / 15'"
    )

    # near the end the window slides to end at the last image
    masker_support.open_workspace(page, served_app, tmp_path / names[13])
    page.wait_for_function(
        "document.getElementById('imgidx').textContent === '14 / 15'"
    )
    assert page.evaluate(titles) == names[3:]
    assert page.evaluate(marked) == names[13]


# --- mouse prompts and zoom on an image wider than the model side --------------
# The stub generator answers from the prompt alone, so a solid image serves.


def test_point_prompt_segments_shape(page, served_app, big_image):
    masker_support.open_workspace(page, served_app, big_image)
    page.click("#tabs div:nth-child(3)")  # Point
    page.fill("#promptText", "shape")
    masker_support.image_click(page, 700, 500)
    page.wait_for_selector("#instances .irow")
    assert page.locator("#instances .irow").count() == 1


def test_zoom_past_fit_fetches_crop_preview(page, served_app, big_image):
    masker_support.open_workspace(page, served_app, big_image)
    # 1400x1000 > the model side: one zoom step (1.4x)
    # must fetch a hi-res patch after the view-change debounce
    with page.expect_request("**/crop_preview"):
        page.click("#zin")
    # the zoom slider tracks button zoom, and its far end is the 8x cap
    assert page.evaluate("+document.getElementById('zslider').value") > 0
    page.evaluate("""() => {
        const s = document.getElementById('zslider');
        s.value = 1000;
        s.dispatchEvent(new Event('input'));
    }""")
    assert page.locator("#zoomtext").text_content() == "800%"
