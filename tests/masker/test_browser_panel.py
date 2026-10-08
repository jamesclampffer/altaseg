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

# per #instances child: 'H|<header text>', 'M|<more text>' or
# 'R|<text>|<dims>|<del glyph>|<flags>'; 'T|<text>' for an empty panel.
# flags: c checked, d disabled, p staged delete, s selected, h pinned,
# o class hidden
PANEL_JS = """
() => {
  const panel = document.getElementById('instances');
  if (!panel.children.length) return ['T|' + panel.textContent];
  return [...panel.children].map(n => {
    if (n.classList.contains('grouphdr')) return 'H|' + n.textContent;
    if (n.classList.contains('more')) return 'M|' + n.textContent;
    const [box, text, dims, del] = n.children;
    return 'R|' + text.textContent + '|' + dims.textContent + '|' + del.textContent + '|'
      + (box.checked ? 'c' : '') + (box.disabled ? 'd' : '')
      + (n.classList.contains('pdel') ? 'p' : '') + (n.classList.contains('sel') ? 's' : '')
      + (n.classList.contains('hov') ? 'h' : '') + (n.style.opacity === '0.45' ? 'o' : '');
  });
}
"""

# texts of the pinned rows, comma-joined
PINNED_JS = ("[...document.querySelectorAll('#instances .irow.hov')]"
             ".map(r => r.children[1].textContent).join()")

# window.__renders: renderInstances calls, one #save text write each
RENDERS_JS = """
() => {
  window.__renders = 0;
  new MutationObserver(ms => { window.__renders += ms.length; })
    .observe(document.getElementById('save'), { childList: true });
}
"""

# __mark() snapshots the panel children; __removed counts child removals
# since, __lost() the marked children no longer in the panel
WATCH_JS = """
() => {
  const panel = document.getElementById('instances');
  window.__removed = 0;
  new MutationObserver(ms => { for (const m of ms) window.__removed += m.removedNodes.length; })
    .observe(panel, { childList: true });
  window.__mark = () => { window.__marked = [...panel.children]; window.__removed = 0; };
  window.__lost = () => window.__marked.filter(n => n.parentNode !== panel).length;
}
"""

ROW_ELEMENTS = 5  # div, checkbox, two spans, button
SIZE = (400, 320)
SAVED_PANEL = ["H|vpalletx1redofillonx", "R|#1|90x90|x|c", "H|vboxx1redofillonx", "R|#2|100x100|x|c"]
# stub text sweep of a 400x320 image: two pass-1 rectangles and one exemplar keep
SWEPT = ["R|p{}|100x160|x|c".format(t) for t in ("1 0.90", "2 0.80", "3 0.50")]


def _hdr(label, n, closed=False, hidden=False):
    return "H|{}{}x{}redofill{}x".format(">" if closed else "v", label, n, "off" if hidden else "on")


# a.png with saved pallet #1 [10,10,100,100] and box #2 [200,200,300,300]
def _seed(tmp_path):
    image = masker_support.png(tmp_path / "a.png", SIZE)
    hw = SIZE[::-1]
    anns = [
        masker_support.annotation(1, 1, 1, [10, 10, 90, 90],
                                  masker_support.mask_for(10, 10, 100, 100, hw).to_json()),
        masker_support.annotation(2, 1, 2, [200, 200, 100, 100],
                                  masker_support.mask_for(200, 200, 300, 300, hw).to_json()),
    ]
    dataset = masker_support.coco_dataset(
        [masker_support.coco_image(1, "a.png", *SIZE)], anns, ["pallet", "box"])
    coco = masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image, dataset)
    return image, coco


def _open(page, url, image):
    masker_support.open_workspace(page, url, image)
    # preview installed (zoom text) and /fsmeta applied (position counter)
    page.wait_for_function(
        "document.getElementById('zoomtext').textContent"
        " && document.getElementById('imgidx').textContent")


def _panel(page):
    return page.evaluate(PANEL_JS)


def _row(page, text):
    return page.locator("#instances .irow").filter(has=page.get_by_text(text, exact=True))


def _save_text(page):
    return page.locator("#save").text_content()


def _wait_pinned(page, text):
    page.wait_for_function("t => {} === t".format(PINNED_JS), arg=text)


def _sweep(page, label):
    page.fill("#promptText", label)
    page.click("#run")
    masker_support.wait_idle(page)


def test_sweep_renders_through_the_throttle(page, served_app, tmp_path):
    image, _ = _seed(tmp_path)
    page.clock.install()
    _open(page, served_app, image)
    page.evaluate(RENDERS_JS)
    page.clock.pause_at(page.evaluate("Date.now()") + 1000)

    # timers paused: the sweep's landed batches render once, at the
    # end-of-stream flush
    _sweep(page, "pallet")
    assert page.evaluate("__renders") == 1
    assert _panel(page)[2:5] == SWEPT
    page.clock.run_for(250)
    assert page.evaluate("__renders") == 1  # no leaked throttle timer

    # throttled calls coalesce into one render 200 ms later; flush renders at
    # once and cancels the timer (app bindings: no DOM control calls them)
    page.evaluate("() => { throttledRenderAll(); throttledRenderAll(); }")
    assert page.evaluate("__renders") == 1
    page.clock.run_for(199)
    assert page.evaluate("__renders") == 1
    page.clock.run_for(1)
    assert page.evaluate("__renders") == 2
    page.evaluate("() => { throttledRenderAll(); flushRenderAll(); }")
    assert page.evaluate("__renders") == 3
    page.clock.run_for(250)
    assert page.evaluate("__renders") == 3


def test_rows_follow_sweep_toggle_pin_and_save(page, served_app, tmp_path):
    image, coco = _seed(tmp_path)
    _open(page, served_app, image)

    # the saved instances under their class headers; nothing to save
    assert _panel(page) == SAVED_PANEL
    assert _save_text(page) == "Save | 0 new"
    assert page.locator("#save").is_disabled()

    # Run lands the sweep as pending rows of its class
    _sweep(page, "pallet")
    assert _panel(page) == [_hdr("pallet", 4), SAVED_PANEL[1], *SWEPT, *SAVED_PANEL[2:]]
    assert _save_text(page) == "Save | 3 new"
    assert page.locator("#save").is_enabled()

    # the row checkbox drives the save count
    box = _row(page, "p1 0.90").locator("input")
    box.uncheck()
    assert _save_text(page) == "Save | 2 new"
    box.check()
    assert _save_text(page) == "Save | 3 new"

    # a hover dwell pins the row; the pin holds after the mouse leaves and
    # through a second sweep
    _row(page, "p3 0.50").hover()
    _wait_pinned(page, "p3 0.50")
    page.mouse.move(5, 5)  # topbar
    page.press("#promptText", "Enter")
    masker_support.wait_idle(page)
    swept_twice = [*SWEPT[:2], "R|p3 0.50|100x160|x|ch", "R|p4 0.90|100x160|x|c",
                   "R|p5 0.80|100x160|x|c", "R|p6 0.50|100x160|x|c"]
    assert _panel(page) == [_hdr("pallet", 7), SAVED_PANEL[1], *swept_twice, *SAVED_PANEL[2:]]

    # the class eye dims that class's rows, keeps the pin, spares other classes
    eye = page.locator("#instances .grouphdr .eye").first
    eye.click()
    snap = _panel(page)
    assert snap[0] == _hdr("pallet", 7, hidden=True)
    assert [r.rsplit("|", 1)[1] for r in snap[1:8]] == ["co"] * 3 + ["cho"] + ["co"] * 3
    assert snap[8:] == SAVED_PANEL[2:]
    eye.click()
    assert _panel(page) == [_hdr("pallet", 7), SAVED_PANEL[1], *swept_twice, *SAVED_PANEL[2:]]

    # Save turns pending rows into saved ones in place: file ids, pin kept
    pinned = _row(page, "p3 0.50").element_handle()
    page.click("#save")
    page.wait_for_function("document.getElementById('save').textContent === 'Save | 0 new'")
    saved = ["R|#{}|100x160|x|{}".format(i, "ch" if i == 5 else "c") for i in range(3, 9)]
    assert _panel(page) == [_hdr("pallet", 7), SAVED_PANEL[1], *saved, *SAVED_PANEL[2:]]
    assert pinned.evaluate("r => r.isConnected && r.children[1].textContent") == "#5"
    assert [a["id"] for a in json.loads(coco.read_text())["annotations"]] == list(range(1, 9))


def test_erase_box_and_deletes(page, served_app, tmp_path):
    image, coco = _seed(tmp_path)
    deletes, dialogs = [], []
    page.on("request",
            lambda r: deletes.append(r) if "/delete_annotation" in r.url else None)
    page.on("dialog", lambda d: dialogs.append(d.message))
    _open(page, served_app, image)
    _sweep(page, "pallet")
    page.click("#save")
    page.wait_for_function("document.getElementById('save').textContent === 'Save | 0 new'")
    _sweep(page, "crate")  # pending p4..p6 over saved #3..#5

    # an Erase box drops the pending centered inside and stages the saved
    # ones; the row button un-stages one
    page.click("#tabs div:nth-child(5)")  # Erase
    masker_support.image_drag(page, 2, 2, 160, 200)
    page.wait_for_function(
        "document.getElementById('save').textContent === 'Save | 2 new, 2 del'")
    assert _panel(page) == [
        _hdr("pallet", 4), "R|#1|90x90|undo|cdp", "R|#3|100x160|undo|cdp",
        "R|#4|100x160|x|c", "R|#5|100x160|x|c", *SAVED_PANEL[2:],
        _hdr("crate", 2), "R|p5 0.80|100x160|x|c", "R|p6 0.50|100x160|x|c"]
    _row(page, "#1").locator("button").click()
    assert _save_text(page) == "Save | 2 new, 1 del"
    assert _row(page, "#1").locator("button").text_content() == "x"
    assert "pdel" not in _row(page, "#1").get_attribute("class")

    # deleting a selected pending row is local and clears the selection
    _row(page, "p5 0.80").locator(".ell").click()
    assert "sel" in _row(page, "p5 0.80").get_attribute("class")
    page.wait_for_selector("#selpop", state="visible")
    page.click("#seldrop")
    _row(page, "p5 0.80").wait_for(state="detached")
    page.wait_for_selector("#selpop", state="hidden")
    assert page.locator("#instances .irow.sel").count() == 0
    assert deletes == []

    # a pinned saved row deletes through /delete_annotation, unpins, no dialog
    _row(page, "#4").hover()
    _wait_pinned(page, "#4")
    page.mouse.move(5, 5)  # topbar
    assert page.evaluate(PINNED_JS) == "#4"
    with page.expect_request("**/delete_annotation"):
        _row(page, "#4").locator("button").dispatch_event("click")  # mouse stays on the topbar
    _row(page, "#4").wait_for(state="detached")
    assert page.evaluate("pinned") is None  # no DOM form: the pinned row is gone
    assert len(deletes) == 1 and dialogs == []
    assert [a["id"] for a in json.loads(coco.read_text())["annotations"]] == [1, 2, 3, 5]


def test_collapse_and_chip_remove(page, served_app, tmp_path):
    image, _ = _seed(tmp_path)
    _open(page, served_app, image)
    _sweep(page, "pallet")

    # a collapsed group hides its rows; its count follows later sweeps
    name = page.locator("#instances .grouphdr .name").first
    name.click()
    assert _panel(page) == [_hdr("pallet", 4, closed=True), *SAVED_PANEL[2:]]
    page.press("#promptText", "Enter")
    masker_support.wait_idle(page)
    assert _panel(page) == [_hdr("pallet", 7, closed=True), *SAVED_PANEL[2:]]

    # the chip x forgets the prompt, drops the class's pending and stages
    # its saved instances
    history = "JSON.parse(localStorage.getItem('masker.prompt_history'))"
    assert page.evaluate(history) == ["pallet"]
    page.locator("#chips .chip .x").click()
    assert page.locator("#chips .chip").count() == 0
    assert page.evaluate(history) == []
    assert _save_text(page) == "Save | 0 new, 1 del"
    name.click()
    assert _panel(page) == [_hdr("pallet", 1), "R|#1|90x90|undo|cdp", *SAVED_PANEL[2:]]


def test_empty_placeholder_and_load_button(page, served_app, tmp_path):
    image, _ = _seed(tmp_path)
    blank = masker_support.png(tmp_path / "b.png", SIZE)
    _open(page, served_app, blank)

    # the placeholder shows with no instances, clears when rows land,
    # returns when they go
    assert _panel(page) == ["T|No instances yet."]
    _sweep(page, "pallet")
    assert _panel(page) == [_hdr("pallet", 3), *SWEPT]
    page.locator("#instances .grouphdr .del").click()
    assert _panel(page) == ["T|No instances yet."]

    # Load strips Explorer's quotes and rebuilds the saved rows from disk
    page.fill("#path", '"{}"'.format(image))
    page.click("#loadbtn")
    page.wait_for_selector("#instances .irow")
    assert page.input_value("#path") == str(image)
    assert _panel(page) == SAVED_PANEL
    assert _save_text(page) == "Save | 0 new"


# --- reconciliation over a 500-row panel ---------------------------------------


# n 30x30 boxes, ten per row from (10, 100)
def _grid(n):
    return [(10 + i % 10 * 38, 100 + i // 10 * 40, 40 + i % 10 * 38, 130 + i // 10 * 40, 0.5)
            for i in range(n)]


def _mark(page):
    page.evaluate("__mark()")
    masker_support.creates(page)


# (elements created, panel children removed, marked children gone) since _mark
def _delta(page):
    return masker_support.creates(page), page.evaluate("__removed"), page.evaluate("__lost()")


def _wait_rows(page, n):
    page.wait_for_function(
        "n => document.querySelectorAll('#instances .irow').length === n", arg=n)


def test_reconcile_patches_rows_in_place(page, serve_app, tmp_path):
    # Box prompts answer with the prompt's boxes: pallet 1, crate 50, box 30
    url = serve_app(masker_support.GlobalBoxMasker(
        {"pallet": [(300, 20, 340, 60, 0.9)], "crate": _grid(50), "box": _grid(30)}))
    image = masker_support.png(tmp_path / "a.png", SIZE)
    labels = [1] * 175 + [2] * 175 + [3] * 150  # pallet, box, crate
    anns = [masker_support.annotation(i + 1, 1, c, [300, 260, 20, 20])
            for i, c in enumerate(labels)]
    masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image,
        masker_support.coco_dataset(
            [masker_support.coco_image(1, "a.png", *SIZE)], anns, ["pallet", "box", "crate"]))
    page.add_init_script(masker_support.COUNT_CREATES_JS)
    _open(page, url, image)
    _wait_rows(page, 500)
    page.click("#eyeall")  # masks hidden: no overlay canvases in the counts
    page.click("#tabs div:nth-child(2)")  # Box
    page.evaluate(WATCH_JS)

    # adding one instance builds one row and leaves every other node in place
    _mark(page)
    page.fill("#promptText", "pallet")
    masker_support.image_drag(page, 20, 20, 60, 60)
    _wait_rows(page, 501)
    assert _delta(page) == (ROW_ELEMENTS, 0, 0)

    # a checkbox toggle patches its row
    _mark(page)
    _row(page, "p1 0.90").locator("input").uncheck()
    assert _save_text(page) == "Save | 0 new"
    assert _delta(page) == (0, 0, 0)

    # a 50-instance burst builds 50 rows and nothing else
    _mark(page)
    page.fill("#promptText", "crate")
    masker_support.image_drag(page, 20, 20, 60, 60)
    _wait_rows(page, 551)
    assert _delta(page) == (50 * ROW_ELEMENTS, 0, 0)

    # removing one instance detaches exactly its row
    _mark(page)
    _row(page, "p51 0.50").locator("button").click()
    _wait_rows(page, 550)
    assert _delta(page) == (0, 1, 1)

    # collapse drops only the group's rows; expand rebuilds only them
    name = page.locator("#instances .grouphdr .name").first
    # scrolled down, the stuck headers stack over the first one
    page.locator("#instances").evaluate("p => { p.scrollTop = 0; }")
    _mark(page)
    name.click()
    _wait_rows(page, 550 - 176)
    assert _delta(page) == (0, 176, 176)
    _mark(page)
    name.click()
    _wait_rows(page, 550)
    assert _delta(page) == (176 * ROW_ELEMENTS, 0, 0)

    # past the per-class cap rows stop at the cap behind one "more" line; a
    # burst into the capped group creates nothing
    cap = page.evaluate("ROW_CAP")  # app constant, no DOM form
    page.fill("#promptText", "box")
    masker_support.image_drag(page, 20, 20, 60, 60)
    page.wait_for_selector("#instances .more")
    _mark(page)
    masker_support.image_drag(page, 20, 20, 60, 60)
    page.wait_for_function(
        "t => document.querySelector('#instances .more').textContent.startsWith(t)",
        arg="... {} more".format(175 + 60 - cap))
    assert _delta(page) == (0, 0, 0)
    assert page.locator("#instances > *").count() == 3 + 176 + cap + 1 + 199
