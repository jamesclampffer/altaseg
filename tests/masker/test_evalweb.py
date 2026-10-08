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

"""Benchmark dashboard endpoints against a synthetic directory. The /image
route is covered with the masker's in test_web_workspace.py."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("flask")

import masker_support
import masker.execution.evaluate
import masker.io.coco

_SUMMARY = {"mode": "instance", "downscale": 1.0,
            "match_iou": masker.execution.evaluate.EvalDefaults.MATCH_IOU}
_IMAGE_EVAL = {"match_iou": masker.execution.evaluate.EvalDefaults.MATCH_IOU}


def test_summary_report_and_coverage(bench_client):
    resp = bench_client.get("/summary", query_string=_SUMMARY)
    assert resp.status_code == 200
    payload = resp.get_json()
    assert payload["reference"].endswith(masker.io.coco.CocoLayout.ANNOTATIONS_NAME)
    assert payload["predictions"].endswith("predictions.json")
    overall = payload["report"]["overall"]
    assert (overall["tp"], overall["fp"], overall["fn"]) == (1, 1, 1)
    assert payload["images"] == [
        {"name": "a.jpg", "ref": 2, "categories": 1, "predicted": 2},
        {"name": "b.jpg", "ref": 0, "categories": 0, "predicted": 0},
    ]


def test_image_eval_classifies_instances(bench_client):
    payload = bench_client.get("/image_eval", query_string={**_IMAGE_EVAL, "name": "a.jpg"}).get_json()
    h, w = masker_support.Bench.SIZE_HW
    assert (payload["width"], payload["height"]) == (w, h)
    assert payload["categories"] == ["widget"]
    assert payload["counts"] == {"tp": 1, "fn": 1, "fp": 1}
    by_kind = {i["kind"]: i for i in payload["instances"]}
    assert by_kind["tp"]["iou"] == pytest.approx(1.0)
    assert by_kind["tp"]["score"] == pytest.approx(0.9)
    assert by_kind["tp"]["bbox_xyxy"] == [10, 10, 50, 50]
    assert by_kind["fn"]["bbox_xyxy"] == [80, 20, 120, 60]
    assert by_kind["fn"]["score"] is None
    layers = {kind: masker_support.decode_data_url(payload["layers"][kind])
              for kind in ("tp", "fn", "fp")}
    assert all(png.size == (w, h) for png in layers.values())
    # the TP layer's opaque pixels are exactly the matched rect
    alpha = np.asarray(layers["tp"])[..., 3]
    assert alpha[30, 30] == 255 and alpha[30, 70] == 0


def test_image_eval_category_filter_and_empty_image(bench_client):
    def get(**q):
        return bench_client.get("/image_eval", query_string={**_IMAGE_EVAL, **q})

    filtered = get(name="a.jpg", category="nothing").get_json()
    assert filtered["instances"] == []
    assert filtered["categories"] == ["widget"]  # filter narrows scoring, not the list
    empty = get(name="b.jpg").get_json()
    assert empty["counts"] == {"tp": 0, "fn": 0, "fp": 0}
    assert empty["layers"] == {"tp": None, "fn": None, "fp": None}
    assert get(name="missing.jpg").status_code == 400
