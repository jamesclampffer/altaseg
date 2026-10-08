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

"""Reference impls for tests"""
from __future__ import annotations

import base64
import contextlib
import dataclasses
import gc
import io
import json
import threading
import time
import urllib.parse

import numpy as np
import PIL.Image
import pytest
import torch
import werkzeug.serving

import masker.io.coco
import masker.io.imaging
import masker.annotations.instance_mask
import masker.execution.model
import masker.execution.sweep
import masker.web


def sweep_defaults():
    # config.json knobs as the page receives them
    return dataclasses.asdict(masker.execution.sweep.SweepParams.resolve())


# counts document.createElement and new Option calls in window.__created
COUNT_CREATES_JS = """
window.__created = 0;
const _ce = Document.prototype.createElement;
Document.prototype.createElement = function (...a) { window.__created++; return _ce.apply(this, a); };
const _Option = window.Option;
window.Option = function (...a) { window.__created++; return new _Option(...a); };
"""


# live model ------


# "masker" holds the run-wide instance; "skip" pins a failed load for the run
_sam3_state: dict = {}


def shared_sam3():
    """init sam3 singleton"""
    if not _sam3_state:
        try:
            _sam3_state["masker"] = masker.execution.model.build_masker()
        except Exception as exc:  # noqa: BLE001 - weights gated / not downloaded
            _sam3_state["skip"] = "SAM3 model unavailable: {}".format(exc)
        else:
            # small batch: the default spills VRAM alongside the test browser/tools
            _sam3_state["masker"].recrop_batch_size = 2
    if "skip" in _sam3_state:
        pytest.skip(_sam3_state["skip"])
    return _sam3_state["masker"]


def unload_sam3():
    """destroy sam3 singleton"""
    _sam3_state.pop("masker", None)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# fake maskers ------

def instance_from_mask(mask, score):
    """ObjectInstance over mask with its tight bbox."""
    return masker.execution.model.ObjectInstance(
        mask=mask, bbox_xyxy=masker.execution.model.bbox_from_mask(mask), score=score,
        centroid=masker.execution.model.mask_centroid(mask))


class BaseMockGenerator(masker.execution.model.MaskGenerator):
    """Mock generator base: one recrop or image per forward through
    segment_instances on PIL pixels; the source tensor is a CPU tensor."""
    __slots__ = ()

    def source_tensor(self, source_image, source_key=None):
        arr = np.asarray(masker.io.imaging.normalize_image(source_image))
        return torch.from_numpy(arr.transpose(2, 0, 1).copy())

    def segment_recrop(self, source_image, recrop, prompt, source_key=None):
        img = masker.io.imaging.normalize_image(source_image)
        target = img.crop(recrop.rect)
        if recrop.scale != 1.0:
            target = target.resize(recrop.scaled_size)
        return self.segment_instances(target, prompt)

    def segment_recrops(self, source_image, recrops, *, text, source_key=None):
        img = masker.io.imaging.normalize_image(source_image)
        for index, recrop in enumerate(recrops):
            yield "batch", [index]
            start = time.perf_counter()
            instances = self.segment_recrop(
                img, recrop, masker.execution.model.SegmentationPrompt(text=text))
            yield "segment_contents", index, instances, time.perf_counter() - start

    def segment_images(self, images, *, text):
        for index, image in enumerate(images):
            yield "batch", [index]
            start = time.perf_counter()
            img = (PIL.Image.fromarray(image.permute(1, 2, 0).numpy())
                   if isinstance(image, torch.Tensor)
                   else masker.io.imaging.normalize_image(image))
            instances = self.segment_instances(
                img, masker.execution.model.SegmentationPrompt(text=text))
            yield "image", index, instances, time.perf_counter() - start


class StubMaskGenerator(BaseMockGenerator):
    """emits boxes"""
    __slots__ = 'coverage',
    # fraction of the image the text rectangles span
    coverage: float

    def __init__(self, coverage=0.5):
        if not 0 < coverage <= 1:
            raise ValueError("coverage must be in (0, 1]")
        self.coverage = coverage

    @staticmethod
    def _box_instance(size, x0, y0, x1, y1, score):
        """Instance whose mask is the given box filled, clipped to the image."""
        w, h = size
        x0, y0 = max(0, int(x0)), max(0, int(y0))
        x1, y1 = min(w, int(x1)), min(h, int(y1))
        if x0 >= x1 or y0 >= y1:
            return None
        mask = np.zeros((h, w), dtype=bool)
        mask[y0:y1, x0:x1] = True
        return instance_from_mask(mask, score)

    def segment_instances(self, source_image, prompt, source_key=None):
        """two rectangles for text, each exemplar box plus one shifted candidate for exemplar prompts."""
        text, exemplar_boxes = prompt.text, prompt.exemplar_boxes
        img = masker.io.imaging.normalize_image(source_image)
        w, h = img.size
        if text is not None and exemplar_boxes is None:
            bw, bh = max(1, int(w * self.coverage / 2)), max(1, int(h * self.coverage))
            y0 = (h - bh) // 2
            left = self._box_instance((w, h), w // 8, y0, w // 8 + bw, y0 + bh, 0.9)
            right = self._box_instance((w, h), w - w // 8 - bw, y0, w - w // 8, y0 + bh, 0.8)
            return [i for i in (left, right) if i is not None]
        if exemplar_boxes is not None:
            instances = [
                self._box_instance((w, h), x0, y0, x1, y1, 0.95)
                for x0, y0, x1, y1 in exemplar_boxes
            ]
            # one extra candidate beyond the given boxes
            x0, y0, x1, y1 = exemplar_boxes[0]
            shift = x1 - x0
            instances.append(self._box_instance((w, h), x0 + shift, y0, x1 + shift, y1, 0.5))
            return [i for i in instances if i is not None]
        x, y = prompt.point
        side = max(1, min(w, h) // 10)
        instance = self._box_instance(
            (w, h), int(x) - side // 2, int(y) - side // 2,
            int(x) - side // 2 + side, int(y) - side // 2 + side, 0.9,
        )
        return [i for i in (instance,) if i is not None]

    def segment_recrops(self, source_image, recrops, *, text, source_key=None):
        """The serial recrop loop with stats"""
        img = masker.io.imaging.normalize_image(source_image)
        for index, recrop in enumerate(recrops):
            yield "batch", [index]
            start = time.perf_counter()
            instances = self.segment_recrop(
                img, recrop, masker.execution.model.SegmentationPrompt(text=text))
            yield "stats", index, {
                "presence": 0.9 if instances else 0.3,
                "scores": sorted((i.score for i in instances), reverse=True),
                "count": len(instances),
            }
            yield "segment_contents", index, instances, time.perf_counter() - start


class HeldMaskGenerator(StubMaskGenerator):
    """Stub whose sweep of any image whose path contains hold blocks until
    release(); entered is set once that sweep is waiting."""
    __slots__ = 'hold', 'gate', 'entered'
    # substring of the image path to hold
    hold: str
    gate: threading.Event
    entered: threading.Event

    def __init__(self, hold):
        super().__init__()
        self.hold = hold
        self.gate = threading.Event()
        self.entered = threading.Event()

    def release(self):
        self.gate.set()

    def segment_recrops(self, source_image, recrops, *, text, source_key=None):
        if source_key and self.hold in source_key:
            self.entered.set()
            self.gate.wait(30)
        yield from super().segment_recrops(source_image, recrops, text=text, source_key=source_key)


class FailingMaskGenerator(StubMaskGenerator):
    """Stub whose every recrop forward raises: a mid-stream /sweep error."""
    __slots__ = ()

    def segment_recrop(self, source_image, recrop, prompt, source_key=None):
        raise RuntimeError("forward failed")


class GlobalBoxMasker(BaseMockGenerator):
    """Pretends the image contains boxes (global xyxy, score): a recrop
    sees each box clipped to its rect. boxes is a list of tuples, or a dict
    keyed by prompt text."""
    __slots__ = 'boxes',
    # (x0, y0, x1, y1, score) tuples, or {prompt: [tuples]}
    boxes: list | dict

    def __init__(self, boxes):
        self.boxes = boxes

    def _boxes_for(self, text):
        if isinstance(self.boxes, dict):
            return self.boxes.get(text, [])
        return self.boxes

    def segment_instances(self, source_image, prompt, source_key=None):
        w, h = source_image.size
        return self._clipped(prompt.text, 0, 0, w, h, 1.0)

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        return self._clipped(
            prompt.text, recrop.x0, recrop.y0, recrop.x1, recrop.y1, recrop.scale)

    def _clipped(self, text, x0, y0, x1, y1, scale):
        out = []
        for bx0, by0, bx1, by1, score in self._boxes_for(text):
            ix0, iy0 = max(bx0, x0), max(by0, y0)
            ix1, iy1 = min(bx1, x1), min(by1, y1)
            if ix0 >= ix1 or iy0 >= iy1:
                continue
            lw, lh = max(1, round((x1 - x0) * scale)), max(1, round((y1 - y0) * scale))
            mask = np.zeros((lh, lw), dtype=bool)
            mx0, my0 = round((ix0 - x0) * scale), round((iy0 - y0) * scale)
            mx1, my1 = round((ix1 - x0) * scale), round((iy1 - y0) * scale)
            mask[my0:max(my0 + 1, my1), mx0:max(mx0 + 1, mx1)] = True
            out.append(instance_from_mask(mask, score))
        return out


class BlobOverArea(GlobalBoxMasker):
    """Answers any recrop larger than area original px^2 with one
    recrop-covering blob (a merge-blob stand-in); smaller recrops see the
    plain clipped boxes."""
    __slots__ = 'area',
    # recrop area (original px^2) above which the blob answers
    area: float

    def __init__(self, boxes, area):
        super().__init__(boxes)
        self.area = area

    def segment_recrop(self, source_image, recrop, prompt, **kwargs):
        w, h = recrop.x1 - recrop.x0, recrop.y1 - recrop.y0
        if w * h > self.area:
            lw, lh = recrop.scaled_size
            return [instance_from_mask(np.ones((lh, lw), dtype=bool), 0.95)]
        return super().segment_recrop(source_image, recrop, prompt, **kwargs)


class RecordingQuadrantMasker(BaseMockGenerator):
    """Records every image it is given and returns one instance whose mask
    fills the top-left quadrant of that (possibly cropped and downscaled)
    image."""
    __slots__ = 'received',
    # copies of every image passed to segment_instances
    received: list[PIL.Image.Image]

    def __init__(self):
        self.received = []

    def segment_instances(self, source_image, prompt, source_key=None):
        img = source_image if isinstance(source_image, PIL.Image.Image) else PIL.Image.open(source_image)
        self.received.append(img.copy())
        w, h = img.size
        mask = np.zeros((h, w), dtype=bool)
        mask[: h // 2, : w // 2] = True
        return [instance_from_mask(mask, 0.9)]


# --- Flask client helpers ---------------------------------------------------


@contextlib.contextmanager
def serve(app):
    """The app on a real HTTP port; yields the base URL."""
    server = werkzeug.serving.make_server("127.0.0.1", 0, app, threaded=True)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:{}".format(server.server_port)
    finally:
        server.shutdown()
        thread.join(timeout=5)


def make_client(tmp_path, mask_generator, **kwargs):
    return masker.web.create_app(generator=mask_generator, **kwargs).test_client()


def post_json(client, url, body):
    resp = client.post(url, json=body)
    # silent=True: an error page for a missing endpoint has an HTML body
    return resp.status_code, resp.get_json(silent=True)


def get_json(client, route, **params):
    resp = client.get(route, query_string=params)
    return resp.status_code, resp.get_json(silent=True)


def load_body(client, image_path):
    """/load body for image_path; asserts 200."""
    code, body = post_json(client, "/load", {"image_path": str(image_path)})
    assert code == 200
    return body


def ndjson_events(client, url, body):
    resp = client.post(url, json=body)
    assert resp.status_code == 200
    assert resp.mimetype == "application/x-ndjson"
    return [json.loads(line) for line in resp.data.decode().splitlines() if line.strip()]


# --- RLE wire format --------------------------------------------------------


def wire_rle(mask):
    """JSON-form RLE (ascii-str counts) of mask, as a client would send it."""
    return masker.annotations.instance_mask.InstanceMask.from_array(mask).to_json()


def decode_rle(rle):
    """Mask from a JSON-form RLE (a response field or a document on disk)."""
    return masker.annotations.instance_mask.InstanceMask.from_json(rle).to_array()


def decode_data_url(url):
    """PIL image from a data:image/...;base64, field (mask PNGs, previews)."""
    assert url.startswith("data:image/")
    return PIL.Image.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1]))).convert("RGBA")


def mask_iou(a, b):
    union = np.logical_or(a, b).sum()
    return np.logical_and(a, b).sum() / union if union else 1.0


def mask_for(x0, y0, x1, y1, size=(48, 64)):
    """InstanceMask of one filled rect in a size frame."""
    mask = np.zeros(size, dtype=np.uint8)
    mask[y0:y1, x0:x1] = 1
    return masker.annotations.instance_mask.InstanceMask.from_array(mask)


def decoded_bbox(rle):
    decoded = decode_rle(rle)
    ys, xs = np.where(decoded)
    return [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]


# --- test images and datasets -----------------------------------------------


def png(path, size=(64, 48)):
    """Plain black PNG at path, returned."""
    PIL.Image.new("RGB", size).save(path)
    return path


def patterned_image(tmp_path, width=64, height=48, name="img.png"):
    """PNG whose every pixel is position-dependent, so a recrop is identifiable."""
    ys, xs = np.mgrid[0:height, 0:width]
    arr = np.stack([(xs * 3) % 256, (ys * 5) % 256, (xs + ys) % 256], axis=-1)
    arr = arr.astype(np.uint8)
    path = tmp_path / name
    PIL.Image.fromarray(arr, mode="RGB").save(path)
    return path, arr


class Bench:
    # (h, w) of the eval benchmark frames
    SIZE_HW = (120, 160)


def coco_image(image_id, name, width, height):
    return {"id": image_id, "file_name": name, "width": width, "height": height}


def annotation(ann_id, image_id, category_id, bbox_xywh, segmentation=None, score=None,
               area=None):
    """One COCO annotation; area defaults to the bbox area."""
    entry = {"id": ann_id, "image_id": image_id, "category_id": category_id,
             "bbox": list(bbox_xywh), "iscrowd": 0,
             "area": area if area is not None else bbox_xywh[2] * bbox_xywh[3]}
    if segmentation is not None:
        entry["segmentation"] = segmentation
    if score is not None:
        entry["score"] = score
    return entry


def coco_dataset(images, annotations, category_names):
    """COCO document; categories are numbered from 1 in category_names order."""
    return {"images": images, "annotations": annotations,
            "categories": [{"id": i + 1, "name": n} for i, n in enumerate(category_names)]}


def coco_with_rle_annotation(image_path):
    """A COCO document (JSON form) holding one RLE annotation for image_path."""
    return coco_dataset(
        [coco_image(1, image_path.name, 64, 48)],
        [annotation(1, 1, 1, [5, 10, 20, 10], mask_for(5, 10, 25, 20).to_json())],
        ["widget"])


def build_benchmark(root):
    """Eval benchmark in root: a.jpg with 2 reference + 2 predictions
    (1 TP, 1 FN, 1 FP); b.jpg unannotated."""
    h, w = Bench.SIZE_HW
    for name in ("a.jpg", "b.jpg"):
        PIL.Image.new("RGB", (w, h), (90, 120, 90)).save(root / name)

    def ann(aid, x0, y0, x1, y1, score=None):
        return annotation(aid, 1, 1, [x0, y0, x1 - x0, y1 - y0],
                          mask_for(x0, y0, x1, y1, size=Bench.SIZE_HW), score=score)

    images = [coco_image(1, "a.jpg", w, h)]
    reference = coco_dataset(images, [ann(1, 10, 10, 50, 50), ann(2, 80, 20, 120, 60)], ["widget"])
    masker.io.coco.save_coco(reference, root / masker.io.coco.CocoLayout.ANNOTATIONS_NAME)
    (root / "out").mkdir()
    predicted = coco_dataset(
        images, [ann(1, 10, 10, 50, 50, score=0.9), ann(2, 10, 80, 40, 110, score=0.3)],
        ["widget"])
    masker.io.coco.save_coco(predicted, root / "out" / "predictions.json")


def write_coco(path, image_path, dataset=None):
    """dataset (default: coco_with_rle_annotation) written to path as
    JSON; returns path."""
    if dataset is None:
        dataset = coco_with_rle_annotation(image_path)
    path.write_text(json.dumps(dataset))
    return path


# --- workspace page (Playwright) --------------------------------------------


def creates(page):
    """Elements created since the last call (page.add_init_script(COUNT_CREATES_JS) first)."""
    return page.evaluate("() => { const n = window.__created; window.__created = 0; return n; }")


def open_workspace(page, url, image_path):
    """The workspace page with image_path loaded (dims text present)."""
    page.goto(url, wait_until="domcontentloaded")
    page.wait_for_selector("#topbar")
    page.fill("#path", str(image_path))
    page.press("#path", "Enter")
    page.wait_for_function(
        "document.getElementById('dims').textContent.includes('MP')")


def wait_idle(page):
    """Until the Run button is no longer busy (a sweep finished)."""
    page.wait_for_function(
        "!document.getElementById('run').classList.contains('busy')", timeout=60_000)


def wait_text(page, element_id, text):
    """Until the element's textContent contains text; the minute covers a batch's sweeps."""
    page.wait_for_function(
        "([i, t]) => document.getElementById(i).textContent.includes(t)",
        arg=[element_id, text], timeout=60_000)


def post_routes(page):
    """List filled with the path of every POST the page sends from now on, in order."""
    routes = []

    def on_request(request):
        if request.method == "POST":
            routes.append(urllib.parse.urlsplit(request.url).path)

    page.on("request", on_request)
    return routes


def image_pt(page, x, y):
    """Client coordinates of original-image (x, y) via the page's viewer mapping."""
    return page.evaluate(
        "([x, y]) => {"
        "  const c = document.getElementById('view');"
        "  const r = c.getBoundingClientRect();"
        "  const [cx, cy] = viewer.imageToCanvas(x, y);"
        "  return [r.left + cx * r.width / c.width, r.top + cy * r.height / c.height];"
        "}", [x, y])


def image_click(page, x, y):
    pt = image_pt(page, x, y)
    page.mouse.click(pt[0], pt[1])


def image_drag(page, x0, y0, x1, y1):
    a, b = image_pt(page, x0, y0), image_pt(page, x1, y1)
    page.mouse.move(a[0], a[1])
    page.mouse.down()
    page.mouse.move(b[0], b[1], steps=4)
    page.mouse.up()
