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

"""Read-side routes: /load, /crop_preview, /image, /model_status,
/cache_stats, /fsmeta with the --root anchor."""
from __future__ import annotations

import io
import os
import threading
import time

import PIL.Image
import pytest
import torch

import masker_support
import masker.cache.tensor_cache
import masker.execution.recrop
import masker.io.imaging
import masker.web


def _app(mask_generator=None):
    """No masker means a cold app."""
    return masker.web.create_app(generator=mask_generator)


def _stub():
    return masker_support.StubMaskGenerator()



@pytest.fixture
def tree(tmp_path):
    """a.png, b.png, c.png beside a non-image file and three subdirectories,
    one dot-prefixed."""
    for name in ("b.png", "a.png", "c.png"):
        masker_support.png(tmp_path / name, (8, 8))
    (tmp_path / "notes.txt").write_text("not an image")
    for name in (".hidden", "sub", "zsub"):
        (tmp_path / name).mkdir()
    return tmp_path


# --- /image: the masker and eval apps alike ---------------------------------


@pytest.mark.parametrize("app_fixture,query,full_size", [
    ("client", lambda p: {"path": str(p)}, (64, 48)),
    ("bench_client", lambda p: {"name": "a.jpg"}, (160, 120)),
], ids=["masker", "eval"])
def test_image_route(request, image_path, app_fixture, query, full_size):
    client = request.getfixturevalue(app_fixture)
    q = query(image_path)
    resp = client.get("/image", query_string={**q, "resolution": 32})
    assert resp.status_code == 200
    assert resp.mimetype == "image/jpeg"
    assert PIL.Image.open(io.BytesIO(resp.data)).size == (32, 24)
    # repeat fetch answers from the preview cache with the same bytes
    again = client.get("/image", query_string={**q, "resolution": 32})
    assert again.status_code == 200 and again.data == resp.data
    # revalidation: the mtime ETag answers 304 with no body
    cached = client.get("/image", query_string={**q, "resolution": 32},
                        headers={"If-None-Match": resp.headers["ETag"]})
    assert cached.status_code == 304
    # never upscaled: a resolution above the source serves it 1:1
    big = client.get("/image", query_string={**q, "resolution": 9999})
    assert big.status_code == 200
    assert PIL.Image.open(io.BytesIO(big.data)).size == full_size
    assert client.get("/image", query_string={k: "missing.png" for k in q}).status_code == 400


# --- /load -------------------------------------------------------------------


def test_load_reports_neighbor_images(client, tree):
    code, body = masker_support.post_json(client, "/load", {"image_path": str(tree / "b.png")})
    assert code == 200
    assert body["prev"] == str(tree / "a.png")
    assert body["next"] == str(tree / "c.png")

    code, body = masker_support.post_json(client, "/load", {"image_path": str(tree / "a.png")})
    assert body["prev"] is None and body["next"] == str(tree / "b.png")
    code, body = masker_support.post_json(client, "/load", {"image_path": str(tree / "c.png")})
    assert body["prev"] == str(tree / "b.png") and body["next"] is None


def test_load_without_coco(client, image_path):
    body = masker_support.load_body(client, image_path)
    assert body["preview"].startswith("data:image/jpeg;base64,")
    assert body["width"] == 64 and body["height"] == 48
    assert body["coco_file"] is None
    assert body["categories"] == [] and body["saved"] == []
    assert body["search_index"] is None
    assert body["search_index_status"] == "missing"


def test_load_picks_up_external_annotation_edits(client, image_path, tmp_path):
    # The server caches the parsed COCO file keyed by mtime; an edit made
    # outside the server (another tool, hand edit) must show on the next load.
    ann = masker_support.write_coco(tmp_path / "ann.json", image_path)
    assert len(masker_support.load_body(client, image_path)["saved"]) == 1

    edited = masker_support.coco_with_rle_annotation(image_path)
    edited["annotations"] = []
    masker_support.write_coco(ann, image_path, edited)
    # force a distinct mtime; back-to-back writes can share one on coarse clocks
    os.utime(ann, (ann.stat().st_atime, ann.stat().st_mtime + 1))

    assert masker_support.load_body(client, image_path)["saved"] == []


def test_load_with_saved_annotations(client, image_path, tmp_path):
    """A compressed-mask annotation renders a PNG; a polygon is bbox-only."""
    dataset = masker_support.coco_with_rle_annotation(image_path)
    dataset["annotations"].append(dict(
        dataset["annotations"][0], id=2, segmentation=[[5, 10, 25, 10, 25, 20]]))
    masker_support.write_coco(tmp_path / "ann.json", image_path, dataset)
    body = masker_support.load_body(client, image_path)
    assert body["coco_file"].endswith("ann.json")
    assert body["categories"][0]["name"] == "widget"
    mask, polygon = body["saved"]
    assert mask["annotation_id"] == 1
    assert mask["bbox_xyxy"] == [5, 10, 25, 20]  # xywh -> xyxy
    assert mask["mask_png"].startswith("data:image/png;base64,")
    assert mask["mask_rect"] == [5, 10, 25, 20]  # tight rect the PNG spans
    assert polygon["annotation_id"] == 2
    assert polygon["mask_png"] is None and polygon["mask_rect"] is None


def test_load_preview_cached_by_mtime(client, image_path, monkeypatch):
    first = masker_support.load_body(client, image_path)

    def boom(*_):
        raise AssertionError("cached /load must not re-decode the original")

    monkeypatch.setattr(masker.io.imaging, "normalize_image", boom)
    code, body = masker_support.post_json(client, "/load", {"image_path": str(image_path)})
    assert code == 200
    assert body["preview"] == first["preview"]
    assert (body["width"], body["height"]) == (64, 48)

    # a rewritten source image (new mtime) misses the cache and is re-decoded
    monkeypatch.undo()
    PIL.Image.new("RGB", (32, 16), color=(200, 0, 0)).save(image_path)
    os.utime(image_path, (image_path.stat().st_atime, image_path.stat().st_mtime + 10))
    code, body = masker_support.post_json(client, "/load", {"image_path": str(image_path)})
    assert code == 200
    assert (body["width"], body["height"]) == (32, 16)


# --- /crop_preview -----------------------------------------------------------


def test_crop_preview_snaps_and_downscales(client, tmp_path):
    path = masker_support.png(tmp_path / "wide.png", (2200, 100))
    # fractional rect: floor/ceil to whole pixels; the region scales to the cap
    code, body = masker_support.post_json(
        client, "/crop_preview",
        {"image_path": str(path), "rect_xyxy": [10.5, 3.2, 2190.1, 96.9]},
    )
    assert code == 200
    assert body["rect_xyxy"] == [10, 3, 2191, 97]
    preview = masker_support.decode_data_url(body["preview"])
    side = masker.web.WebLimits.PREVIEW_MAX_SIDE
    assert max(preview.size) <= side
    assert preview.size[0] == side  # longest side fills the cap


# --- model build, /model_status, /cache_stats --------------------------------


def test_concurrent_first_requests_build_the_masker_once(tmp_path, monkeypatch):
    calls = []

    def slow_build(**kwargs):
        calls.append(kwargs)
        time.sleep(0.1)  # wide race recrop: both requests reach the None check
        return _stub()

    monkeypatch.setattr(masker.execution.model, "build_masker", slow_build)
    img = masker_support.png(tmp_path / "img.png")
    app = _app()  # no injected masker
    codes = []

    def post():
        resp = app.test_client().post(
            "/segment",
            json={"image_path": str(img), "prompt_type": "point", "point": [32, 24]},
        )
        codes.append(resp.status_code)

    threads = [threading.Thread(target=post) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(10)
    assert codes == [200, 200]
    assert len(calls) == 1


def test_model_status_answers_during_build(tmp_path, monkeypatch):
    building = threading.Event()
    release = threading.Event()

    def slow_build(**kwargs):
        building.set()
        release.wait(10)
        return _stub()

    monkeypatch.setattr(masker.execution.model, "build_masker", slow_build)
    img = masker_support.png(tmp_path / "img.png")
    app = _app()

    def post():
        app.test_client().post(
            "/segment",
            json={"image_path": str(img), "prompt_type": "point", "point": [32, 24]},
        )

    t = threading.Thread(target=post)
    t.start()
    try:
        assert building.wait(10)
        assert app.test_client().get("/model_status").get_json() == {"loaded": False}
    finally:
        release.set()
        t.join(10)
    assert app.test_client().get("/model_status").get_json() == {"loaded": True}


class _CachedStub(masker_support.StubMaskGenerator):
    """Stub carrying an embed cache."""
    __slots__ = '_embed_cache',
    # the embed cache
    _embed_cache: object

    def __init__(self, cache):
        super().__init__()
        self._embed_cache = cache


def test_cache_stats_reports_usage(tmp_path):
    cache = masker.cache.tensor_cache.TensorCache(
        (2, 2), torch.float32, "cpu", vram_bytes=0, host_bytes=1 << 20,
        disk_bytes=1 << 20, disk_dir=tmp_path / "embed")
    recrops = [masker.execution.recrop.Recrop(0, 0, 8, 8, 1.0, 8, 8),
               masker.execution.recrop.Recrop(8, 0, 16, 8, 1.0, 8, 8)]
    cache.begin("img", recrops)
    out = torch.empty(2, 2)
    assert not cache.get(recrops[0], out)         # miss
    cache.put(recrops[0], torch.ones(2, 2))
    assert cache.get(recrops[0], out)             # hit
    cache._disk.flush()

    client = _app(_CachedStub(cache)).test_client()
    code, body = masker_support.get_json(client, "/cache_stats")
    assert code == 200
    assert body["present"] is True
    assert body["disk_bytes"] == sum(
        p.stat().st_size for p in (tmp_path / "embed").glob("*.trunks"))
    assert body["hit_rate"] == 0.5


# --- /fsmeta and --root -----------------------------------------------------


def test_fsmeta_listing_and_position(client, tree):
    """One tree queried as a directory, an image, and a non-image file."""
    code, body = masker_support.get_json(client, "/fsmeta", path=str(tree))
    assert code == 200
    assert body["dir"] == str(tree) and body["parent"] == str(tree.parent)
    assert body["images"] == ["a.png", "b.png", "c.png"]  # sorted, non-images out
    assert body["count"] == 3
    assert body["subdirs"] == ["sub", "zsub"]  # dot-dirs excluded
    assert body["index"] is None and body["name"] is None
    assert body["sep"] == os.sep

    code, body = masker_support.get_json(client, "/fsmeta", path=str(tree / "b.png"))
    assert code == 200
    assert body["dir"] == str(tree) and body["count"] == 3
    assert body["name"] == "b.png" and body["index"] == 1

    code, body = masker_support.get_json(client, "/fsmeta", path=str(tree / "notes.txt"))
    assert code == 200
    assert body["name"] == "notes.txt" and body["index"] is None


def test_fsmeta_lists_zephyr_masks_apart(client, tree):
    PIL.Image.new("L", (8, 8)).save(tree / "x_mask.tiff")
    code, body = masker_support.get_json(client, "/fsmeta", path=str(tree))
    assert code == 200
    assert body["images"] == ["a.png", "b.png", "c.png"]
    assert body["masks"] == ["x_mask.tiff"]


def test_root_anchors_relative_paths(tree):
    """--root: an empty /fsmeta path lists the root; relative /fsmeta and
    /load paths resolve against it."""
    client = masker_support.make_client(tree, _stub(), root=tree)
    code, body = masker_support.get_json(client, "/fsmeta", path="")
    assert code == 200
    assert body["dir"] == str(tree.resolve()) and body["count"] == 3

    code, body = masker_support.get_json(client, "/fsmeta", path="b.png")
    assert code == 200 and body["index"] == 1
    code, body = masker_support.post_json(client, "/load", {"image_path": "b.png"})
    assert code == 200
    assert (body["width"], body["height"]) == (8, 8)
