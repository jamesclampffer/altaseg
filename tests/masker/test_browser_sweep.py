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

# Run through the real /sweep in headless Chromium: knobs, progress strip,
# presence grid, search records.
from __future__ import annotations

import json

import PIL.Image
import pytest

import masker_support
import masker.execution.search_index
import masker.io.coco

pytest.importorskip("playwright.sync_api")


@pytest.fixture
def big_image(tmp_path):
    # 1400x1000: two pass-1 recrops at the served knobs
    path = tmp_path / "big.png"
    PIL.Image.new("RGB", (1400, 1000), (40, 70, 40)).save(path)
    return path


class EmptyMaskGenerator(masker_support.StubMaskGenerator):
    # finds nothing: every sweep completes with zero kept
    __slots__ = ()

    def segment_instances(self, source_image, prompt, source_key=None):
        return []


def _events(response):
    return [json.loads(line) for line in response.text().splitlines()]


def _first(events, kind):
    return next(e for e in events if e["type"] == kind)


def _progress(page):
    # the strip clears its DOM on end; the counters stay in the app's prog
    return page.evaluate("[prog.total, prog.done, prog.active]")


def test_run_sweeps_served_knobs_and_defers_record(page, serve_app, big_image):
    generator = masker_support.HeldMaskGenerator(big_image.name)
    url = serve_app(generator)
    routes = masker_support.post_routes(page)
    masker_support.open_workspace(page, url, big_image)
    defaults = masker_support.sweep_defaults()

    # knobs seed from the served defaults; input values are strings
    assert float(page.input_value("#sweepexpand")) == defaults["reprompt_expand"]
    assert float(page.input_value("#sweepoverlap")) == defaults["overlap"]
    assert float(page.input_value("#sweepdupios")) == defaults["dup_ios"]
    assert page.is_checked("#sweepiosgate") == defaults["ios_gate"]
    assert page.get_attribute("#tabs div:nth-child(1)", "class") == "on"       # Text

    page.fill("#promptText", "box")
    with page.expect_response("**/sweep") as resp:
        page.click("#run")
    # held after the start event: the strip is up, Run is latched
    page.wait_for_selector("#sweepstrip", state="visible")
    assert page.locator("#run").text_content() == "Running..."
    generator.release()
    masker_support.wait_idle(page)
    assert not page.locator("#sweepstrip").is_visible()
    assert page.locator("#run").text_content() == "Run"
    assert resp.value.request.post_data_json == {
        "image_path": str(big_image), "text": "box", "params": defaults}

    events = _events(resp.value)
    start = events[0]
    # every forward ticked: pass-1 recrops, one per cluster, exemplar recrops
    total = (start["total"] + _first(events, "regroup")["clusters"]
             + len(_first(events, "explan")["recrops"]))
    assert _progress(page) == [total, total, False]

    # every landing is a row
    landed = [e["type"] for e in events for _ in e.get("instances", ())]
    assert page.locator("#instances .irow").count() == len(landed)
    assert page.evaluate("state.pending.map(p => p.origin)") == [None] * len(landed)
    # the pass-1 grid and its per-recrop stats outlive the sweep (app state)
    assert page.evaluate("sweepGrid.rects") == start["recrops"]
    assert page.evaluate(
        "[...sweepGrid.stats.values()].map(s => [s.index, s.presence, s.count])"
    ) == [[e["index"], e["presence"], e["count"]]
          for e in events if e["type"] == "recrop_stats"]
    assert "warn" not in page.get_attribute("#presence", "class")
    page.click("#presence")
    assert "warn" in page.get_attribute("#presence", "class")
    page.click("#presence")
    assert "warn" not in page.get_attribute("#presence", "class")

    # kept results defer the record to a successful save
    with page.expect_response("**/record_searches") as rec:
        page.click("#save")
    assert routes[routes.index("/sweep"):] == ["/sweep", "/save", "/record_searches"]
    record = {"prompt": "box", "completed": True, "params": defaults, "instances": len(landed)}
    assert rec.value.request.post_data_json["searches"] == [record]
    on_disk = masker.execution.search_index.load_index(big_image)["searches"]
    assert [{k: s[k] for k in record} for s in on_disk] == [record]


def test_packed_sweep_counts_canvases(page, serve_app, big_image):
    # four boxes straddle the seam of the two pass-1 recrops: two clusters
    boxes = [(680, y, 720, y + 40, 0.9) for y in (60, 300, 540, 780)]
    url = serve_app(masker_support.GlobalBoxMasker(boxes))
    masker_support.open_workspace(page, url, big_image)
    page.click("#adv")
    page.select_option("#sweeppack", "2")
    page.fill("#promptText", "crate")
    with page.expect_response("**/record_searches") as rec:
        with page.expect_response("**/sweep") as resp:
            page.click("#run")
    assert resp.value.request.post_data_json["params"]["pack"] == 2
    events = _events(resp.value)
    assert _first(events, "regroup")["clusters"] == 2
    # both clusters share one 2x2 canvas: one tick, not two
    total = events[0]["total"] + 1
    assert _progress(page) == [total, total, False]
    # nothing kept: the record posts at once and stores the pack
    record = rec.value.request.post_data_json["searches"][0]
    assert (record["params"]["pack"], record["instances"]) == (2, 0)


def test_recorded_sweep_skips_until_knobs_or_flags_change(
        page, serve_app, image_path, tmp_path):
    dataset = masker_support.coco_with_rle_annotation(image_path)
    dataset["categories"][0]["name"] = "pallet"
    masker_support.write_coco(
        tmp_path / masker.io.coco.CocoLayout.ANNOTATIONS_NAME, image_path, dataset)
    url = serve_app(EmptyMaskGenerator())
    routes = masker_support.post_routes(page)
    confirms = []

    def answer(accept):
        def on_dialog(dialog):
            confirms.append(dialog.message)
            if accept:
                dialog.accept()
            else:
                dialog.dismiss()
        page.once("dialog", on_dialog)

    def run(label):
        # Run through the record a zero-kept sweep posts at once; returns the
        # /sweep body and the posted record
        page.fill("#promptText", label)
        with page.expect_response("**/record_searches") as rec:
            with page.expect_request("**/sweep") as req:
                page.click("#run")
        return req.value.post_data_json, rec.value.request.post_data_json["searches"][0]

    def wait_recorded(downscale):
        # no DOM signal for the index the page holds: app state
        page.wait_for_function(
            "d => indexRecords().some(r => r.prompt === 'pallet' && r.params.downscale === d)",
            arg=downscale)

    masker_support.open_workspace(page, url, image_path)
    defaults = masker_support.sweep_defaults()

    # a one-recrop sweep that keeps nothing records immediately
    _, record = run("pallet")
    assert _progress(page) == [1, 1, False]
    assert record == {"prompt": "pallet", "completed": True, "params": defaults, "instances": 0}
    assert masker.execution.search_index.load_index(image_path)["searches"][0]["prompt"] == "pallet"
    wait_recorded(1)

    # a matching record skips: Run all without asking, Run after a confirm
    page.click("#runall")
    masker_support.wait_text(page, "status", 'Sweep already recorded for "pallet"')
    answer(False)
    page.click("#run")
    assert len(confirms) == 1
    answer(True)
    run("pallet")
    assert routes.count("/sweep") == 2
    assert len(confirms) == 2
    assert 'already ran for "pallet"' in confirms[0]

    # other knobs never skip; fractional downscale passes through, <= 0 falls
    # back to 1
    page.click("#adv")
    page.fill("#sweepdownscale", "3")
    body, record = run("pallet")
    assert body["params"]["downscale"] == record["params"]["downscale"] == 3
    page.fill("#sweepdownscale", "0.5")
    body, record = run("pallet")
    assert body["params"]["downscale"] == record["params"]["downscale"] == 0.5
    page.fill("#sweepdownscale", "0")
    body, record = run("crate")
    assert body["params"]["downscale"] == record["params"]["downscale"] == 1

    # ignore previous searches re-runs a recorded sweep without asking
    page.fill("#sweepdownscale", "0.5")
    wait_recorded(0.5)
    page.check("#ignoreprev")
    run("pallet")
    page.uncheck("#ignoreprev")
    page.click("#adv")  # fold the panel back over the Instances header

    # redo (replace) bypasses the record and stages the saved match
    page.click("#instances .grouphdr .redo")
    page.wait_for_selector("#redomodal", state="visible")
    with page.expect_response("**/record_searches"):
        with page.expect_request("**/sweep"):
            page.click("#redook")
    masker_support.wait_text(page, "status", "replacing 1 saved (deleted on save)")
    assert page.locator("#instances .irow.pdel").count() == 1
    assert page.locator("#save").text_content() == "Save | 0 new, 1 del"
    assert len(confirms) == 2
    assert routes.count("/sweep") == 7


def test_sweep_error_releases_strip_and_records_nothing(page, serve_app, image_path):
    url = serve_app(masker_support.FailingMaskGenerator())
    routes = masker_support.post_routes(page)
    masker_support.open_workspace(page, url, image_path)
    page.fill("#promptText", "rock")
    with page.expect_response("**/sweep") as resp:
        page.click("#run")
    masker_support.wait_text(page, "status", "Error: forward failed")
    masker_support.wait_idle(page)
    assert page.locator("#run").text_content() == "Run"
    assert not page.locator("#sweepstrip").is_visible()
    assert "forward failed" in page.locator("#toast").text_content()
    events = _events(resp.value)
    assert events[-1] == {"type": "error", "error": "forward failed"}
    # the start event had begun the strip; no forward ticked
    assert _progress(page) == [events[0]["total"], 0, False]

    # busy released: Run sweeps again; nothing recorded in between
    with page.expect_request("**/sweep"):
        page.click("#run")
    masker_support.wait_idle(page)
    assert [r for r in routes if r in ("/sweep", "/record_searches")] == ["/sweep", "/sweep"]
    assert masker.execution.search_index.load_index(image_path) is None
