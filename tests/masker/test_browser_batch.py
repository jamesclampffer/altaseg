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

# Headless-Chromium port of dom_drive_batch.js: the batch job and the review
# page against the real Flask app, files in tmp_path, faults via page.route.
from __future__ import annotations

import collections
import json
import pathlib
import re

import PIL.Image
import pytest

import masker_support
import masker.execution.search_index
import masker.io.coco

sync_api = pytest.importorskip("playwright.sync_api")


def _image_dir(tmp_path, names=("a.png", "b.png"), dirname="imgs"):
    # 64x48 PNGs (the mask_for frame) in tmp_path/dirname
    image_dir = tmp_path / dirname
    image_dir.mkdir()
    for name in names:
        PIL.Image.new("RGB", (64, 48), (60, 60, 60)).save(image_dir / name)
    return image_dir


def _coco_path(image_dir):
    return image_dir / masker.io.coco.CocoLayout.ANNOTATIONS_NAME


def _seed_history(page, prompts=("box",)):
    # prompt history in place before the app script reads it
    page.add_init_script("localStorage.setItem('masker.prompt_history', {})".format(
        json.dumps(json.dumps(list(prompts)))))


def _open_batch(page, url, image, prompts=("box",)):
    # image loaded, its directory listing landed, batch modal open
    _seed_history(page, prompts)
    masker_support.open_workspace(page, url, image)
    page.wait_for_function("document.getElementById('imgidx').textContent !== ''")
    page.click("#batchcard")
    page.wait_for_selector("#batchmodal", state="visible")


def _run_batch(page):
    page.click("#batchstart")
    masker_support.wait_text(page, "btitle", "Batch complete")


def _sent(page):
    # every request the page sends, in send order (routed and aborted ones too)
    sent = []
    page.on("request", lambda r: sent.append(r))  # a bound builtin cannot be a handler
    return sent


def _bodies(sent, route):
    return [r.post_data_json for r in sent if r.url.endswith(route)]


def _sweeps(sent):
    # "image name|prompt" per /sweep request
    return ["{}|{}".format(pathlib.Path(b["image_path"]).name, b["text"])
            for b in _bodies(sent, "/sweep")]


def _saves(sent):
    return [pathlib.Path(b["image_path"]).name for b in _bodies(sent, "/save")]


def _thumbs(page):
    # completed-image names in the batch modal
    return page.locator("#bthumbs .nm").all_text_contents()


def _saved_counts(image_dir):
    # annotations per image name in the directory's COCO file
    data = json.loads(_coco_path(image_dir).read_text())
    names = {i["id"]: i["file_name"] for i in data["images"]}
    return collections.Counter(names[a["image_id"]] for a in data["annotations"])


# --- batch runs on the stub server ---------------------------------------------


def test_batch_skips_recorded_sweep(page, served_app, tmp_path):
    # b's index records a completed sweep at the page knobs: skipped; a's
    # record at other knobs does not count and is replaced by the record
    # that lands after a's save.
    image_dir = _image_dir(tmp_path)
    upsert = masker.execution.search_index.upsert_searches
    defaults = masker_support.sweep_defaults()
    upsert(image_dir / "a.png", [{"prompt": "box", "completed": True,
                                  "params": {**defaults, "downscale": 2.0}, "instances": 1}],
           width=64, height=48)
    upsert(image_dir / "b.png", [{"prompt": "box", "completed": True,
                                  "params": defaults, "instances": 1}], width=64, height=48)
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png")
    assert page.locator("#imgidx").text_content() == "1 / 2"
    assert page.locator("#batchstart").text_content() == "Start batch | 2 images"
    _run_batch(page)

    assert _sweeps(sent) == ["a.png|box"]
    assert _saves(sent) == ["a.png"]
    counts = page.locator("#bthumbs .ct").all_text_contents()
    kept = int(counts[0])
    assert kept > 0 and counts[1] == "0"
    assert _saved_counts(image_dir) == {"a.png": kept}
    assert _bodies(sent, "/record_searches") == [{
        "image_path": str(image_dir / "a.png"), "width": 64, "height": 48,
        "searches": [{"prompt": "box", "completed": True, "params": defaults,
                      "instances": kept}],
    }]
    index = masker.execution.search_index.load_index(image_dir / "a.png")
    assert [r["params"]["downscale"] for r in index["searches"]] == [1]
    # the per-image pace has no DOM surface after a run: read its storage
    assert float(page.evaluate("localStorage.getItem('masker.image_seconds_ewma')")) > 0

    # b's skip added nothing to the run totals
    assert page.locator("#bstats .card .v").all_text_contents() == [str(kept)]
    assert page.locator("#bline").text_content() == (
        "all images complete | {} instances".format(kept))
    assert _thumbs(page) == ["a.png", "b.png"]
    srcs = [t.get_attribute("src") for t in page.locator("#bthumbs img").all()]
    assert len(srcs) == 2 and all(s.startswith("/image?path=") for s in srcs)
    page.wait_for_function(
        "[...document.querySelectorAll('#bthumbs img')].every(i => i.naturalWidth > 0)")


def test_batch_picked_directory_without_image(page, serve_app, tmp_path):
    # no image loaded: Browse lists the server root, Up returns to it, Use
    # enables Start, and the run walks the picked listing from index 0;
    # spaces and & survive URL encoding across the picker and the pinned dir
    image_dir = _image_dir(tmp_path, dirname="my imgs & more")
    url = serve_app(root=tmp_path)
    sent = _sent(page)
    _seed_history(page)
    page.goto(url, wait_until="domcontentloaded")
    page.click("#batchcard")
    page.wait_for_selector("#batchmodal", state="visible")
    assert page.locator("#batchstart").is_disabled()
    page.click("#bbrowse")
    masker_support.wait_text(page, "pickdir", str(tmp_path))
    page.click("#picklist .pickrow")
    masker_support.wait_text(page, "pickdir", str(image_dir))
    page.click("#pickup")
    masker_support.wait_text(page, "pickdir", str(tmp_path))
    page.click("#picklist .pickrow")
    masker_support.wait_text(page, "pickcount", "2 image(s)")
    page.click("#pickuse")
    page.wait_for_selector("#pickmodal", state="hidden")
    assert page.locator("#bdir").text_content() == str(image_dir)
    assert page.locator("#batchstart").is_enabled()
    assert page.locator("#batchstart").text_content() == "Start batch | 2 images"
    # the picker stacks over the batch modal: cancelling it leaves the batch
    # modal visible
    page.click("#bbrowse")
    page.wait_for_selector("#pickmodal", state="visible")
    page.click("#pickcancel")
    page.wait_for_selector("#pickmodal", state="hidden")
    assert page.locator("#batchmodal").is_visible()
    _run_batch(page)
    assert _sweeps(sent) == ["a.png|box", "b.png|box"]
    assert _thumbs(page) == ["a.png", "b.png"]


def test_batch_skips_unloadable_image(page, served_app, tmp_path):
    # an image the server cannot decode is reported and skipped; the run
    # continues to the end of the queue
    image_dir = _image_dir(tmp_path, names=("a.png",))
    (image_dir / "b.png").write_text("not an image")
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png")
    _run_batch(page)
    assert page.locator("#toast").text_content() == "Batch done - 1 image(s)"
    assert page.locator("#bpct").text_content() == "100%"
    assert page.locator("#status").text_content().startswith("Error: invalid image")
    assert _thumbs(page) == ["a.png"]
    assert _sweeps(sent) == ["a.png|box"]


def test_batch_queue_starts_at_loaded_image(page, served_app, tmp_path):
    # the queue runs from the loaded image to the end; whole directory
    # starts at index 0
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "b.png")
    assert page.locator("#imgidx").text_content() == "2 / 2"
    assert page.locator("#batchstart").text_content() == "Start batch | 1 images"
    _run_batch(page)
    assert _sweeps(sent) == ["b.png|box"]
    assert _thumbs(page) == ["b.png"]

    page.click("#bdismiss")
    page.check("#bwholedir")
    assert page.locator("#batchstart").text_content() == "Start batch | 2 images"
    _run_batch(page)
    # b's sweep is recorded now: the whole-directory run sweeps only a
    assert _sweeps(sent) == ["b.png|box", "a.png|box"]
    assert _thumbs(page) == ["a.png", "b.png"]


def test_batch_zephyr_export_covers_directory_labels(page, served_app, tmp_path):
    # export-on-done takes every label in the directory's COCO file (the
    # hand-added one included), not the prompt snapshot, and rewrites every mask
    image_dir = _image_dir(tmp_path)
    dataset = masker_support.coco_dataset(
        [masker_support.coco_image(1, "a.png", 64, 48)],
        [masker_support.annotation(1, 1, 1, [4, 4, 8, 8],
                                   masker_support.mask_for(4, 4, 12, 12).to_json())],
        ["handmade"])
    masker_support.write_coco(_coco_path(image_dir), image_dir / "a.png", dataset)
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png")
    page.check("#bzephyr")
    _run_batch(page)
    (export,) = _bodies(sent, "/export_zephyr")
    assert export["labels"] == ["handmade", "box"]
    assert export["force"] is True and export["missing_ok"] is True
    assert page.locator("#toast").text_content() == "Zephyr masks: 2 written, 0 skipped"
    assert (image_dir / "a_mask.tiff").is_file() and (image_dir / "b_mask.tiff").is_file()


def test_batch_failed_save_parks_paused(page, served_app, tmp_path):
    # a failing save with the server up parks the batch paused on the unsaved
    # image; Resume retries it
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png")
    page.route("**/save", lambda route: route.fulfill(status=400, json={"error": "disk full"}))
    page.click("#batchstart")
    masker_support.wait_text(page, "batchstart", "Resume batch")
    assert page.locator("#btitle").text_content() == "Batch paused"
    assert page.locator("#status").text_content() == "Error: disk full"
    assert page.locator("#toast").text_content() == "a.png: save failed"
    assert not _coco_path(image_dir).exists()

    page.unroute("**/save")
    _run_batch(page)
    assert _saves(sent) == ["a.png", "a.png", "b.png"]
    # the unrecorded sweep reran on a; its duplicates merged before the save
    assert _sweeps(sent) == ["a.png|box", "a.png|box", "b.png|box"]
    counts = _saved_counts(image_dir)
    assert counts["a.png"] == counts["b.png"] > 0


def test_batch_prompt_selection_drives_sweeps(page, served_app, tmp_path):
    # untick / none / add / all set which prompts the run sweeps, image-major
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png", prompts=("box", "crate"))
    ticks = page.locator("#bprompts .prow input")
    ticks.nth(0).uncheck()
    assert "sweep crate, drop" in page.locator("#bplan").text_content()
    page.click("#bselnone")
    assert page.locator("#bplan").text_content() == "Select at least one prompt."
    assert page.locator("#batchstart").is_disabled()

    page.fill("#bnewprompt", " rock ")
    page.click("#bnewadd")
    assert page.locator("#bnewprompt").input_value() == ""
    # rows follow the history: the trimmed prompt went to the front
    assert page.locator("#bprompts .prow span:nth-child(3)").all_text_contents() == [
        "rock", "box", "crate"]
    assert [t.is_checked() for t in ticks.all()] == [True, False, False]
    page.click("#bselall")
    assert [t.is_checked() for t in ticks.all()] == [True, True, True]

    ticks.nth(1).uncheck()  # box
    _run_batch(page)
    assert _sweeps(sent) == ["a.png|rock", "a.png|crate", "b.png|rock", "b.png|crate"]
    assert page.locator("#bsub").text_content() == "imgs | 2 prompt(s)"
    assert page.locator("#bstats .card .h").all_text_contents() == ["rock", "crate"]


def test_batch_extension_filter(page, served_app, tmp_path):
    # only images with the listed extensions queue, matched case-insensitively
    image_dir = _image_dir(tmp_path, names=("a.png", "b.JPG", "c.bmp"))
    sent = _sent(page)
    _open_batch(page, served_app, image_dir / "a.png")
    assert page.locator("#batchstart").text_content() == "Start batch | 3 images"
    page.fill("#bext", "png, .jpg")
    assert page.locator("#batchstart").text_content() == "Start batch | 2 images"
    _run_batch(page)
    assert _sweeps(sent) == ["a.png|box", "b.JPG|box"]
    assert _thumbs(page) == ["a.png", "b.JPG"]


# --- held sweeps: HeldMaskGenerator blocks the named image's sweep -------------


def test_batch_pause_aborts_in_flight_image(page, serve_app, tmp_path):
    # pause aborts b's in-flight sweep; the loop saves what landed (nothing),
    # lists b as completed and advances past it, so Resume has nothing left
    gen = masker_support.HeldMaskGenerator("b.png")
    url = serve_app(gen)
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _open_batch(page, url, image_dir / "a.png")
    page.click("#batchstart")
    assert gen.entered.wait(30)
    # progress at a real mid-run position: a done, b in flight
    page.click("#bhide")
    assert page.locator("#batchpilltext").text_content().startswith("batch | 1/2 | ")
    page.click("#batchpill")  # reopening re-renders the progress
    assert page.locator("#bpct").text_content() == "50%"
    assert page.evaluate("document.getElementById('bprogfill').style.width") == "50%"
    assert page.locator("#beta").text_content().endswith(" | 1/2")

    # the abort settles the loop right after the click: 'pausing' is only
    # readable in the click's own task
    assert page.evaluate("""() => {
        document.getElementById('batchpause').click();
        return document.getElementById('beta').textContent;
    }""") == "pausing..."
    masker_support.wait_text(page, "batchstart", "Resume batch")  # reached with b still held
    gen.release()
    assert _saves(sent) == ["a.png"]
    assert _saved_counts(image_dir).keys() == {"a.png"}
    assert not masker.execution.search_index.index_path(image_dir / "b.png").exists()
    page.click("#bclose")
    assert page.locator("#batchpilltext").text_content() == "batch paused | 2/2"

    fsmeta = sum("/fsmeta?" in r.url for r in sent)
    page.click("#batchpill")
    _run_batch(page)
    assert sum("/fsmeta?" in r.url for r in sent) == fsmeta  # queue snapshot reused
    assert _sweeps(sent) == ["a.png|box", "b.png|box"]  # b is not swept again
    assert page.locator("#bthumbs .ct").all_text_contents()[1] == "0"


def test_batch_cancel_keeps_in_flight_entry(page, serve_app, tmp_path):
    # cancel aborts b's in-flight sweep; the loop still runs b's save step
    # (nothing landed, so no request) and collects b before going idle. A
    # fresh batch then runs clean, skipping a's recorded sweep.
    gen = masker_support.HeldMaskGenerator("b.png")
    url = serve_app(gen)
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _open_batch(page, url, image_dir / "a.png")
    page.click("#batchstart")
    assert gen.entered.wait(30)
    assert page.evaluate("""() => {
        document.getElementById('bcancel').click();
        return document.getElementById('batchpilltext').textContent;
    }""") == "batch cancelling..."
    masker_support.wait_text(page, "toast", "Batch cancelled - annotations saved so far are kept")
    gen.release()
    assert not page.locator("#batchpill").is_visible()
    assert _sweeps(sent) == ["a.png|box", "b.png|box"]
    assert _saves(sent) == ["a.png"]
    # the review grid lists this session's batch entries
    page.click("#reviewbtn")
    page.wait_for_selector("#reviewgrid .tile")
    assert page.locator("#reviewgrid .tile .n").all_text_contents() == ["a.png", "b.png"]
    page.click("#reviewbtn")  # back to the workspace

    page.fill("#path", str(image_dir / "a.png"))
    page.press("#path", "Enter")
    masker_support.wait_text(page, "imgidx", "1 / 2")
    page.click("#batchcard")
    _run_batch(page)
    assert _sweeps(sent) == ["a.png|box", "b.png|box", "b.png|box"]
    assert _saves(sent) == ["a.png", "b.png"]
    assert _thumbs(page) == ["a.png", "b.png"]


def test_batch_dead_server_cancels(page, serve_app, tmp_path):
    # the server dies once a's sweep is streaming: the save and the liveness
    # probe fail, so the batch cancels instead of pausing and a's detections
    # stay pending
    gen = masker_support.HeldMaskGenerator("a.png")
    url = serve_app(gen)
    image_dir = _image_dir(tmp_path)
    _open_batch(page, url, image_dir / "a.png")
    page.click("#batchstart")
    assert gen.entered.wait(30)
    page.route("**/*", lambda route: route.abort())  # the open /sweep stream is past routing
    gen.release()
    masker_support.wait_text(page, "toast", "Server stopped - batch cancelled; saved annotations are kept, "
                              "this image's pending left for review")
    assert not _coco_path(image_dir).exists()
    assert page.locator("#batchstart").text_content() == "Start batch | 2 images"
    assert page.locator("#bthumbs .bthumb").count() == 0
    rows = page.locator("#instances .irow").count()
    assert rows > 0
    assert page.locator("#save").text_content() == "Save | {} new".format(rows)

    # server back: the pending detections save from the workspace
    page.unroute("**/*")
    page.click("#bclose")
    page.click("#save")
    masker_support.wait_text(page, "saved", "* saved")
    assert _saved_counts(image_dir) == {"a.png": rows}


def test_batch_waits_for_interactive_run(page, serve_app, tmp_path):
    # a batch started while an interactive sweep holds the model waits for it
    # instead of skipping or saving the image
    gen = masker_support.HeldMaskGenerator("a.png")
    url = serve_app(gen)
    image_dir = _image_dir(tmp_path)
    sent = _sent(page)
    _seed_history(page)
    masker_support.open_workspace(page, url, image_dir / "a.png")
    page.fill("#promptText", "box")
    page.click("#run")
    assert gen.entered.wait(30)
    page.click("#batchcard")
    page.click("#batchstart")
    assert page.locator("#status").text_content() == (
        "Batch started - waiting for the current run to finish...")
    masker_support.wait_text(page, "btitle", "Batch running")
    # a sweep or save inside this window (three busy polls) is the
    # started-while-busy regression
    with pytest.raises(sync_api.TimeoutError):
        with page.expect_request(re.compile(r"/(sweep|save)$"), timeout=1000):
            pass
    assert page.locator("#bpct").text_content() == "0%"
    assert page.locator("#bline").text_content() == "image 1 | a.png"

    gen.release()
    masker_support.wait_text(page, "btitle", "Batch complete")
    assert _sweeps(sent) == ["a.png|box", "a.png|box", "b.png|box"]
    assert _saves(sent) == ["a.png", "b.png"]
    assert _thumbs(page) == ["a.png", "b.png"]
    # the interactive run's detections and the batch's duplicates merged
    counts = _saved_counts(image_dir)
    assert counts["a.png"] == counts["b.png"] > 0


# --- review page -----------------------------------------------------------------


def test_review_flags_summary_and_label_filter(page, served_app, tmp_path):
    # zero-hit flag, summary and size-extremes text, and the tri-state chip
    # (require -> exclude -> off) narrowing the grid and dimming the excluded
    # label's histogram
    image_dir = _image_dir(tmp_path)
    # areas 20, 25, 500 on a.png; b.png listed with none
    anns = [masker_support.annotation(i + 1, 1, 1, [0, 0, w, h],
                                      masker_support.mask_for(0, 0, w, h).to_json())
            for i, (w, h) in enumerate(((4, 5), (5, 5), (20, 25)))]
    dataset = masker_support.coco_dataset(
        [masker_support.coco_image(1, "a.png", 64, 48),
         masker_support.coco_image(2, "b.png", 64, 48)], anns, ["box"])
    masker_support.write_coco(_coco_path(image_dir), image_dir / "a.png", dataset)
    masker_support.open_workspace(page, served_app, image_dir / "a.png")
    page.click("#reviewbtn")
    page.wait_for_selector("#reviewgrid .tile")
    page.wait_for_selector("#reviewstats .card")
    # the thumbnail must decode: a broken dir/sep/name join leaves naturalWidth 0
    page.wait_for_function(
        "document.querySelector('#reviewgrid .tile img').naturalWidth > 0")
    tiles = page.locator("#reviewgrid .tile .n")
    summary = page.locator("#reviewsummary")
    card = page.locator("#reviewstats .card")
    assert page.locator("#reviewtitle").text_content() == "Review | imgs"
    assert summary.text_content() == "2 images | 3 instances"
    assert page.locator("#reviewgrid .tile .fl").all_text_contents() == ["! 0 box"]
    assert page.locator("#reviewgrid .tile.flag .n").text_content() == "b.png"
    assert card.locator(".s").text_content() == "side 4-22 px | bbox w 4-20 | h 5-25"
    # bins [2, 0, 0, 0, 1]: tallest fills the strip, empty keeps a baseline
    assert card.locator(".hist div").evaluate_all("bars => bars.map(b => b.style.height)") == [
        "34px", "1px", "1px", "1px", "17px"]

    chip = page.locator("#reviewfilter .fchip")
    assert chip.text_content() == "box"
    chip.click()  # require
    assert chip.text_content() == "+ box"
    assert tiles.all_text_contents() == ["a.png"]
    assert summary.text_content() == "1/2 images | 3 instances"
    chip.click()  # exclude
    assert chip.text_content() == "- box"
    assert tiles.all_text_contents() == ["b.png"]
    assert summary.text_content() == "1/2 images | 0 instances"
    assert card.evaluate("c => c.style.opacity") == "0.45"
    chip.click()  # off
    assert chip.text_content() == "box"
    assert tiles.all_text_contents() == ["a.png", "b.png"]
    assert summary.text_content() == "2 images | 3 instances"
    assert card.evaluate("c => c.style.opacity") == "1"
