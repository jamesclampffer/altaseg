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

"""/segment (exemplar, point, recrop), and the /sweep and /reprompt_bboxes
NDJSON streams: one event per line, request and event schemas per the endpoint
docstrings in masker/web/__init__.py. Invalid input is a plain 400 JSON error
before any streaming."""
from __future__ import annotations

import base64
import io

import numpy as np
import PIL.Image
import pytest

import masker_support
import masker.annotations.instance_mask
import masker.execution.model
import masker.execution.recrop
import masker.execution.sweep
import masker.web
import masker.web.webcore


IMG = "<image_path>"  # stands for the fixture image in the plain 400s table


# --- /segment: prompt types --------------------------------------------------


@pytest.mark.parametrize(
    "prompt, expected_count",
    [
        ({"prompt_type": "exemplar", "boxes": [[4, 4, 20, 20]], "text": "thing"}, 2),
        ({"prompt_type": "point", "point": [32, 24]}, 1),
    ],
)
def test_segment_all_prompt_types(client, image_path, prompt, expected_count):
    code, body = masker_support.post_json(
        client, "/segment", {"image_path": str(image_path), **prompt})
    assert code == 200
    assert len(body["instances"]) == expected_count
    assert "generation_seconds" in body["stats"]
    for inst in body["instances"]:
        assert inst["mask_png"].startswith("data:image/png;base64,")
        assert len(inst["bbox_xyxy"]) == 4
        decoded = masker_support.decode_rle(inst["rle"])
        assert decoded.shape == (48, 64)
        # the RLE, the bbox, and the mask rect describe the same region
        x0, y0, x1, y1 = inst["bbox_xyxy"]
        ys, xs = np.where(decoded)
        assert [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1] == [x0, y0, x1, y1]
        assert inst["mask_rect"] == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
        # centroid: mask centroid at pixel centers, original coords
        assert inst["centroid"] == pytest.approx(
            [xs.mean() + 0.5, ys.mean() + 0.5], abs=0.01)
        # the PNG covers exactly mask_rect at native resolution
        rx0, ry0, rx1, ry1 = inst["mask_rect"]
        png = masker_support.decode_data_url(inst["mask_png"])
        alpha = np.asarray(png)[..., 3] > 0
        assert alpha.shape == (ry1 - ry0, rx1 - rx0)
        assert (alpha == (decoded[ry0:ry1, rx0:rx1] > 0)).all()


class _RecordingStub(masker_support.StubMaskGenerator):
    """Stub that records the prompt of every segment_instances call."""
    __slots__ = 'calls',
    # {"text", "exemplar_boxes"} per call
    calls: list[dict]

    def __init__(self):
        super().__init__()
        self.calls = []

    def segment_instances(self, source_image, prompt, source_key=None):
        self.calls.append({"text": prompt.text, "exemplar_boxes": prompt.exemplar_boxes})
        return super().segment_instances(source_image, prompt)


def test_segment_exemplar_with_text_prompts_both(tmp_path):
    recording = _RecordingStub()
    client = masker_support.make_client(tmp_path, recording)
    path, _ = masker_support.patterned_image(tmp_path)

    code, _ = masker_support.post_json(client, "/segment", {
        "image_path": str(path), "prompt_type": "exemplar",
        "boxes": [[4, 4, 20, 20]], "text": "thing"})
    assert code == 200
    assert recording.calls[-1] == {"text": "thing", "exemplar_boxes": [(4, 4, 20, 20)]}

    # recrop: text passes through alongside the recrop-local boxes
    code, _ = masker_support.post_json(client, "/segment", {
        "image_path": str(path), "prompt_type": "exemplar",
        "boxes": [[10, 10, 20, 20]], "text": "thing", "recrop": [8, 6, 40, 30]})
    assert code == 200
    assert recording.calls[-1] == {"text": "thing",
                                   "exemplar_boxes": [(2.0, 4.0, 12.0, 14.0)]}


# --- _instance_payload: recrop-local encode parity ----------------------------


@pytest.mark.parametrize(
    "full_size, rect",
    [
        ((3200, 2400), (400, 300, 2416, 2316)),  # downscaled recrop, big frame
        ((3200, 2400), (2800, 2000, 3600, 2900)),  # overhangs right/bottom edges
        ((800, 600), (100, 50, 500, 450)),  # small frame: preview == full
    ],
)
def test_instance_payload_matches_full_frame_path(full_size, rect):
    full_w, full_h = full_size
    win = masker.execution.recrop.Recrop.from_rect(
        *rect, masker.web.WebLimits.PREVIEW_MAX_SIDE)
    rng = np.random.default_rng(5)
    local = (rng.random((win.out_h, win.out_w)) < 0.3)
    inst = masker_support.instance_from_mask(local, 0.87)

    full = win.sample_original(local, np.arange(full_h), np.arange(full_w))
    encoded = masker.annotations.instance_mask.InstanceMask.from_array(full)
    mask_rect = masker.web._mask_rect(encoded)
    # recrop centroid contract: local-mask centroid mapped out (no
    # full-frame decode)
    lys, lxs = np.nonzero(local)
    centroid = win.point_to_global_coords(float(lxs.mean()) + 0.5, float(lys.mean()) + 0.5)
    expected = {
        "score": 0.87,
        "bbox_xyxy": [round(float(v), 2) for v in win.to_global_bbox(inst.bbox_xyxy)],
        "centroid": [round(float(v), 2) for v in centroid],
        "mask_png": masker.web.webcore.data_url(
            masker.web._full_mask_png(full, mask_rect), "image/png"),
        "mask_rect": list(mask_rect),
        "rle": encoded.to_json(),
    }
    got = masker.web._instance_payload(inst, recrop=win, full_size=full_size)
    assert got == expected


# --- /segment: recrop ------------------------------------------------------
# The route hands the recrop to the generator; the fake presents it at
# recrop.scaled_size and answers with its top-left quadrant.


def test_segment_recrop_maps_back_to_original_coords(tmp_path, image_path):
    quadrant = masker_support.RecordingQuadrantMasker()
    client = masker_support.make_client(tmp_path, quadrant)
    # a 32 x 24 recrop of the 64 x 48 image, under the model side: scale 1
    code, body = masker_support.post_json(
        client, "/segment",
        {"image_path": str(image_path), "prompt_type": "point", "point": [16, 12],
         "recrop": [8, 6, 40, 30]},
    )
    assert code == 200
    (received,) = quadrant.received
    assert received.size == (32, 24)
    (inst,) = body["instances"]
    expected = [8, 6, 24, 18]  # x0..x0+16, y0..y0+12 in original pixels
    assert inst["bbox_xyxy"] == pytest.approx(expected, abs=2)
    assert inst["rle"]["size"] == [48, 64]  # full original image, not the recrop
    assert masker_support.decoded_bbox(inst["rle"]) == pytest.approx(expected, abs=2)


def test_segment_recrop_downscales_oversized_recrops(tmp_path):
    quadrant = masker_support.RecordingQuadrantMasker()
    client = masker_support.make_client(tmp_path, quadrant)
    path = tmp_path / "big.png"
    PIL.Image.new("RGB", (2560, 1440), color=(20, 40, 60)).save(path)

    recrop = [256, 128, 2272, 1136]  # 2016 x 1008 recrop: must be scaled down
    code, body = masker_support.post_json(
        client, "/segment",
        {"image_path": str(path), "prompt_type": "point", "point": [512, 256], "recrop": recrop},
    )
    assert code == 200

    (received,) = quadrant.received
    # presented at the model side with its aspect kept
    assert max(received.size) <= masker.web.WebLimits.PREVIEW_MAX_SIDE
    w, h = received.size
    assert abs(w / h - 2016 / 1008) < 0.05
    # quadrant of the downscaled recrop -> top-left quadrant of its rect, in
    # original-image pixels
    (inst,) = body["instances"]
    expected = [256, 128, 256 + 1008, 128 + 504]
    assert inst["bbox_xyxy"] == pytest.approx(expected, abs=2)
    assert inst["rle"]["size"] == [1440, 2560]
    assert masker_support.decoded_bbox(inst["rle"]) == pytest.approx(expected, abs=2)


# --- plain 400s: validation precedes the generator and any streaming ----------


@pytest.mark.parametrize("route, body, why", [
    ("/reprompt_bboxes", {"image_path": "no/such/file.png", "text": "x",
                          "bboxes": [[50, 50, 650, 650]]}, "bad path"),
    ("/sweep", {"image_path": "no/such/file.png", "text": "x"}, "bad path"),
    # the route adds no knob validation of its own: SweepParams raises and
    # the route answers 400
    ("/sweep", {"image_path": IMG, "text": "x", "params": {"downscale": 0}}, "bad knob value"),
])
def test_bad_requests_are_plain_400(client, image_path, route, body, why):
    body = {k: str(image_path) if v == IMG else v for k, v in body.items()}
    resp = client.post(route, json=body)
    assert resp.status_code == 400, why
    assert resp.mimetype == "application/json", why  # plain error, no stream
    assert "error" in resp.get_json(), why


# --- the sweep scene ------------------------------------------------------------
# 2520x1008 image, 1008px recrops (PREVIEW_MAX_SIDE), overlap 0.25 -> step
# 756: three recrops [0,1008] [756,1764] [1512,2520]; disjoint cores
# [0,756] [1008,1512] [1764,2520].


class SweepScene:
    img_w, img_h = 2520, 1008
    # (x0, y0, x1, y1, score); strictly inside recrop 0's core, centroid (350, 350)
    core_box = (100, 100, 600, 600, 0.9)
    # straddles the recrop-0/1 seam
    seam_box = (900, 300, 1100, 500, 0.8)
    # centroid (800, 250)
    other_box = (700, 150, 900, 350, 0.8)
    # centroid (1900, 700)
    far_box = (1800, 600, 2000, 800, 0.7)
    grid = [[0, 0, 1008, 1008], [756, 0, 1764, 1008], [1512, 0, 2520, 1008]]


@pytest.fixture
def scene_image(tmp_path):
    path = tmp_path / "img.png"
    PIL.Image.new("RGB", (SweepScene.img_w, SweepScene.img_h), color=(30, 90, 30)).save(path)
    return path


class BoomOnSecondRecrop(masker_support.GlobalBoxMasker):
    """Fails the second recrop forward."""
    __slots__ = 'calls',
    # segment_recrop calls so far
    calls: int

    def __init__(self, boxes):
        super().__init__(boxes)
        self.calls = 0

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        self.calls += 1
        if self.calls == 2:
            raise RuntimeError("recrop exploded")
        return super().segment_recrop(source_image, recrop, prompt, **kwargs)


# --- /sweep ---------------------------------------------------------------------


def sweep_body(path):
    # exemplar off: these tests pin the pass-1/2 stream contract
    return {"image_path": str(path), "text": "box",
            "params": {"overlap": 0.25, "reprompt_expand": 2.0, "exemplar": False}}


def test_sweep_event_stream_and_seam_replacement(tmp_path, scene_image):
    client = masker_support.make_client(
        tmp_path, masker_support.GlobalBoxMasker([SweepScene.core_box, SweepScene.seam_box]))
    events = masker_support.ndjson_events(client, "/sweep", sweep_body(scene_image))
    start = events[0]
    assert start["type"] == "start"
    assert start["total"] == 3 and start["recrops"] == SweepScene.grid
    kinds = [e["type"] for e in events]
    # the base masker batches one recrop at a time in both passes
    assert kinds == ["start"] + ["batch", "segment_contents"] * 3 \
        + ["regroup", "rebatch", "reprompt", "done"]
    batches = [e for e in events if e["type"] == "batch"]
    assert [e["indices"] for e in batches] == [[0], [1], [2]]
    assert [e["recrops"] for e in batches] == [[w] for w in SweepScene.grid]
    recrops = {e["index"]: e for e in events if e["type"] == "segment_contents"}
    for index, e in recrops.items():
        assert e["recrop"] == SweepScene.grid[index]
        assert e["seconds"] >= 0
        for inst in e["instances"]:
            assert set(inst) == {"score", "bbox_xyxy", "centroid",
                                 "mask_png", "mask_rect", "rle"}
            assert inst["mask_png"].startswith("data:image/png;base64,")
    # the core object lands once, from recrop 0, already in original coords
    (accepted,) = recrops[0]["instances"]
    assert accepted["rle"]["size"] == [SweepScene.img_h, SweepScene.img_w]
    assert masker_support.decoded_bbox(accepted["rle"]) == pytest.approx([100, 100, 600, 600], abs=2)
    assert accepted["bbox_xyxy"] == pytest.approx([100, 100, 600, 600], abs=2)
    # the seam straddler is ambiguous in both recrops that saw it, never accepted
    assert recrops[0]["ambiguous"] == 1 and recrops[1]["ambiguous"] == 1
    assert recrops[1]["instances"] == [] and recrops[2]["instances"] == []
    assert [e for e in events if e["type"] == "regroup"] == [{"type": "regroup", "clusters": 1}]
    (rebatch,) = [e for e in events if e["type"] == "rebatch"]
    # the cluster's 2x-expanded, centered recrop
    assert rebatch == {"type": "rebatch", "indices": [0], "recrops": [[800, 200, 1200, 600]]}
    # the two fragments come back as one replacement instance, in original coords
    (rp,) = [e for e in events if e["type"] == "reprompt"]
    assert rp["cluster"] == 0 and rp["recrop"] == [800, 200, 1200, 600]
    (kept,) = rp["instances"]
    assert kept["rle"]["size"] == [SweepScene.img_h, SweepScene.img_w]
    assert masker_support.decoded_bbox(kept["rle"]) == pytest.approx([900, 300, 1100, 500], abs=2)
    assert kept["bbox_xyxy"] == pytest.approx([900, 300, 1100, 500], abs=2)
    assert events[-1] == {"type": "done"}


class StatsBoxMasker(masker_support.GlobalBoxMasker):
    """GlobalBoxMasker plus a Sam3-shaped ("stats", index, stats) event per
    recrop: presence 0.5 + 0.05 * index, fixed scores."""

    def segment_recrops(self, source_image, recrops, *, text, source_key=None):
        for index, recrop in enumerate(recrops):
            yield "batch", [index]
            instances = self.segment_recrop(
                source_image, recrop, masker.execution.model.SegmentationPrompt(text=text))
            yield "stats", index, {"presence": 0.5 + 0.05 * index,
                                   "scores": [0.91, 0.4, 0.1],
                                   "count": len(instances)}
            yield "segment_contents", index, instances, 0.0


def test_sweep_streams_recrop_stats(tmp_path, scene_image):
    client = masker_support.make_client(
        tmp_path, StatsBoxMasker([SweepScene.core_box, SweepScene.seam_box]))
    events = masker_support.ndjson_events(client, "/sweep", sweep_body(scene_image))
    kinds = [e["type"] for e in events]
    # one recrop_stats per pass-1 recrop, before its segment_contents event; the
    # re-prompt pass emits none even though the masker yields stats there too
    assert kinds == ["start"] + ["batch", "recrop_stats", "segment_contents"] * 3 \
        + ["regroup", "rebatch", "reprompt", "done"]
    stats = [e for e in events if e["type"] == "recrop_stats"]
    assert [e["index"] for e in stats] == [0, 1, 2]
    assert [e["presence"] for e in stats] == [0.5, 0.55, 0.6]
    # count is the recrop's raw detection tally: accepted + ambiguous
    for st, seg in zip(stats, [e for e in events if e["type"] == "segment_contents"]):
        assert st["count"] == len(seg["instances"]) + seg["ambiguous"]


# --- /reprompt_bboxes --------------------------------------------------------


def reprompt_body(path, bboxes, **params):
    return {"image_path": str(path), "text": "box", "bboxes": bboxes, "params": params}


def _planned(bbox, expand=2.0):
    return list(masker.execution.sweep.plan_cluster_recrop(
        tuple(float(v) for v in bbox), SweepScene.img_w, SweepScene.img_h, expand=expand).rect)


def test_reprompt_bboxes_event_stream_contract(tmp_path, scene_image):
    # other_box sits inside the expanded recrop but its centroid is outside the
    # drawn bbox: re-detected, then dropped by the sweep keep rule
    client = masker_support.make_client(
        tmp_path, masker_support.GlobalBoxMasker([SweepScene.core_box, SweepScene.other_box]))
    drawn = [50, 50, 650, 650]
    events = masker_support.ndjson_events(
        client, "/reprompt_bboxes", reprompt_body(scene_image, [drawn], reprompt_expand=2.0))
    assert [e["type"] for e in events] == ["rebatch", "reprompt"]
    rebatch, rp = events
    assert rebatch == {"type": "rebatch", "indices": [0], "recrops": [_planned(drawn)]}
    assert rp["index"] == 0 and rp["recrop"] == _planned(drawn) and rp["seconds"] >= 0
    (kept,) = rp["instances"]
    assert set(kept) == {"score", "bbox_xyxy", "centroid", "mask_png", "mask_rect", "rle"}
    assert kept["mask_png"].startswith("data:image/png;base64,")
    assert kept["rle"]["size"] == [SweepScene.img_h, SweepScene.img_w]
    assert masker_support.decoded_bbox(kept["rle"]) == pytest.approx([100, 100, 600, 600], abs=2)
    assert kept["bbox_xyxy"] == pytest.approx([100, 100, 600, 600], abs=2)


# --- mid-stream errors, both NDJSON routes ------------------------------------


@pytest.mark.parametrize("route,second_box,body,types", [
    ("/sweep", SweepScene.seam_box, lambda img: sweep_body(img),
     ["start", "batch", "segment_contents", "batch", "error"]),
    ("/reprompt_bboxes", SweepScene.far_box,
     lambda img: reprompt_body(img, [[50, 50, 650, 650], [1750, 550, 2050, 850]]),
     ["rebatch", "reprompt", "rebatch", "error"]),
], ids=["sweep", "reprompt"])
def test_mid_stream_error_event(tmp_path, scene_image, route, second_box, body, types):
    client = masker_support.make_client(
        tmp_path, BoomOnSecondRecrop([SweepScene.core_box, second_box]))
    events = masker_support.ndjson_events(client, route, body(scene_image))
    # the first recrop's results stand; the failure ends the stream (no done)
    assert [e["type"] for e in events] == types
    landed, error = events[-3], events[-1]
    assert landed["index"] == 0 and landed["instances"]
    assert error == {"type": "error", "error": error["error"]}
    assert "recrop exploded" in error["error"]
