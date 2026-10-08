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

"""Headless-Chromium coverage of the benchmark dashboard page."""
from __future__ import annotations

import pytest

playwright_api = pytest.importorskip("playwright.sync_api")
pytest.importorskip("flask")

import masker_support

import masker.execution.evaluate
import masker.web.evalweb


@pytest.fixture
def served_eval_app(tmp_path):
    masker_support.build_benchmark(tmp_path)
    with masker_support.serve(masker.web.evalweb.create_app(tmp_path)) as url:
        yield url


def test_summary_renders_tiles_charts_and_tables(page, served_eval_app):
    page.goto(served_eval_app, wait_until="domcontentloaded")
    page.wait_for_selector("#tiles .card")

    tiles = page.eval_on_selector_all("#tiles .card .h", "els => els.map(e => e.textContent)")
    assert tiles[:3] == ["precision", "recall", "F1"]
    assert page.eval_on_selector_all("#bandchart .band", "els => els.length") == len(
        masker.execution.evaluate.EvalDefaults.SIZE_BANDS)
    # 1 TP / 1 FP / 1 FN everywhere.
    assert "50.0%" in page.text_content("#tiles .card .v")
    assert "1 of 2 images hand-annotated" in page.text_content("#coverage")
    # header + widget + OVERALL
    assert page.eval_on_selector_all("#cattable tr", "els => els.length") == 3

    page.select_option("#mode", "semantic")
    page.wait_for_function(
        "document.querySelector('#tiles .card .h').textContent === 'IoU'")


def test_image_row_opens_overlay_view(page, served_eval_app):
    page.goto(served_eval_app, wait_until="domcontentloaded")
    page.wait_for_selector("#imgtable tr.click")

    page.click("#imgtable tr.click >> nth=0")
    page.wait_for_function("document.getElementById('tpcount').textContent === '1'")
    assert not page.is_hidden("#detail")
    assert page.text_content("#detailname") == "a.jpg"
    assert page.text_content("#fncount") == "1" and page.text_content("#fpcount") == "1"
    options = page.eval_on_selector_all("#catfilter option", "els => els.map(e => e.value)")
    assert options == ["", "widget"]
    width = page.evaluate("document.getElementById('view').width")
    assert width > 0

    # Next image has nothing annotated; prev returns.
    page.click("#nextbtn")
    page.wait_for_function(
        "document.getElementById('detailname').textContent === 'b.jpg'")
    page.wait_for_function(
        "document.getElementById('detailstatus').textContent.includes('no annotations')")
    page.click("#prevbtn")
    page.wait_for_function(
        "document.getElementById('detailname').textContent === 'a.jpg'")
    page.click("#back")
    assert page.is_hidden("#detail")
