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

"""Flask web UI for the masker: a combined instance viewer/annotator.

Per instance the wire carries mask_png (1-bit PNG of the mask cropped to
mask_rect, its tight integer bbox in original pixels, native resolution
up to WebLimits.MASK_CROP_SIDE per side) and rle (the compressed
mask, opaque to the client and round-tripped on save).
"""

from __future__ import annotations

import collections
import concurrent.futures
import dataclasses
import hashlib
import json
import math
import os
import pathlib
import queue
import threading
import time

import flask
import numpy as np
import PIL.Image
import PIL.ImageDraw

import masker.annotations.annotation_management
import masker.annotations.instance_mask
import masker.cache.ephemeral_cache
from masker.common.common_defs import binary_mask_t, index_array_t, rect_t
import masker.execution.model
import masker.execution.recrop
import masker.execution.search_index
import masker.execution.sweep
import masker.io.coco
import masker.io.imaging
import masker.io.zephyr
import masker.web.webcore

# One model build at a time.
_MODEL_LOCK = threading.Lock()

class WebLimits:
    # Longest side of the preview sent to the browser.
    PREVIEW_MAX_SIDE = masker.execution.sweep.SweepConsts.MODEL_SIDE
    # Longest side of one instance's mask_png
    MASK_CROP_SIDE = 2016
    # decoded originals held for re-prompts
    SOURCE_IMAGE_CACHE_GB = 2.0
    # preview JPEGs keyed by (path, mtime, side)
    PREVIEW_CACHE_MB = 512
    # rendered mask PNGs keyed by mask content
    MASK_PNG_CACHE_MB = 512
    # sweep events buffered ahead of the wire encoder
    WIRE_QUEUE = 64
    # threads encoding sweep events
    WIRE_THREADS = 8


# encodes sweep events for the wire
_WIRE_POOL = concurrent.futures.ThreadPoolExecutor(WebLimits.WIRE_THREADS)

# rendered mask PNGs keyed by mask content (see _mask_key)
_MASK_PNGS: masker.cache.ephemeral_cache.EphemeralCache[bytes] = (
    masker.cache.ephemeral_cache.EphemeralCache(WebLimits.MASK_PNG_CACHE_MB << 20))

_CFG_JSON = json.dumps({"MODEL_SIDE": WebLimits.PREVIEW_MAX_SIDE,
                        "MASK_SIDE": WebLimits.MASK_CROP_SIDE,
                        "SWEEP_DEFAULTS": dataclasses.asdict(
                            masker.execution.sweep.SweepParams.resolve())})


def _index_html() -> str:
    return masker.web.webcore.page_html("masker_page.html", {
        "VIEWER_JS": masker.web.webcore.asset("webui_viewer.js"),
        "DRAW_JS": masker.web.webcore.asset("webui_draw.js"),
        "CFG_JSON": _CFG_JSON,
        "APP_JS": masker.web.webcore.asset("masker_app.js"),
    })


def _mask_rect(mask: masker.annotations.instance_mask.InstanceMask) -> rect_t:
    """Tight integer rect of the foreground; the 1x1 origin pixel when empty."""
    if mask.area == 0:
        return (0, 0, 1, 1)
    x0, y0, x1, y1 = mask.bbox
    return (int(x0), int(y0), int(round(x1)), int(round(y1)))


def _mask_sample_grid(rect: rect_t) -> tuple[index_array_t, index_array_t]:
    """(ys, xs) sampling rect at native resolution, thinned past MASK_CROP_SIDE."""
    x0, y0, x1, y1 = rect
    h, w = y1 - y0, x1 - x0
    if max(h, w) <= WebLimits.MASK_CROP_SIDE:
        return np.arange(y0, y1, dtype=np.intp), np.arange(x0, x1, dtype=np.intp)
    scale = WebLimits.MASK_CROP_SIDE / max(h, w)
    return (y0 + masker.web.webcore.sample_grid(h, scale),
            x0 + masker.web.webcore.sample_grid(w, scale))


def _full_mask_png(binary: binary_mask_t, rect: rect_t) -> bytes:
    ys, xs = _mask_sample_grid(rect)
    return masker.web.webcore.mask_png_bytes(binary[np.ix_(ys, xs)])


def _rect_png(mask: masker.annotations.instance_mask.InstanceMask, rect: rect_t) -> bytes:
    """PNG of a compressed mask's rect; only the rect is decoded."""
    ys, xs = _mask_sample_grid(rect)
    return masker.web.webcore.mask_png_bytes(
        mask.rect_array(rect)[np.ix_(ys - rect[1], xs - rect[0])])


def _mask_key(mask: masker.annotations.instance_mask.InstanceMask) -> str:
    digest = hashlib.sha1(mask.counts).hexdigest()
    return "{}x{}:crop{}:{}".format(mask.height, mask.width, WebLimits.MASK_CROP_SIDE, digest)


def _png_url(png: bytes) -> str:
    return masker.web.webcore.data_url(png, "image/png")


def _cached_png(mask, render) -> bytes:
    key = _mask_key(mask)
    png = _MASK_PNGS.get(key)
    if png is None:
        png = render()
        _MASK_PNGS.put(key, png)
    return png


def _mask_png_from_segmentation(segmentation) -> tuple[str | None, list | None]:
    """(data URL, mask_rect) for a compressed mask; (None, None) for a polygon."""
    if not isinstance(segmentation, masker.annotations.instance_mask.InstanceMask):
        return None, None
    rect = _mask_rect(segmentation)
    png = _cached_png(segmentation, lambda: _rect_png(segmentation, rect))
    return _png_url(png), list(rect)


def _group_response(masks: masker.annotations.instance_mask.InstanceMaskSet, groups) -> dict:
    """Singleton groups come back bare; multi-member groups carry the union."""
    out = []
    for group in groups:
        entry: dict = {"indices": group}
        if len(group) > 1:
            merged = masks.subset(group).union_all_masks()
            rect = _mask_rect(merged)
            png = _cached_png(merged, lambda: _rect_png(merged, rect))
            entry.update(
                rle=merged.to_json(),
                bbox_xyxy=[round(v, 2) for v in merged.bbox],
                mask_png=_png_url(png),
                mask_rect=list(rect),
            )
        out.append(entry)
    return {"groups": out}


def _draw_op(draw: PIL.ImageDraw.ImageDraw, op, x0: int, y0: int) -> None:
    """Apply one op to a recrop-local mask canvas; erase clears."""
    value = 0 if op["erase"] else 255
    local = [(x - x0, y - y0) for x, y in op["points"]]
    if op["kind"] == "polygon":
        draw.polygon([(round(x), round(y)) for x, y in local], fill=value)
        return
    d = op["diameter"]
    if len(local) > 1:
        draw.line([(round(x), round(y)) for x, y in local], fill=value, width=round(d))
    for x, y in local:  # round caps and joins
        draw.ellipse(
            (round(x - d / 2), round(y - d / 2), round(x + d / 2), round(y + d / 2)), fill=value
        )


def _rect(rect) -> rect_t:
    x0, y0, x1, y1 = (int(round(v)) for v in rect)
    return (x0, y0, x1, y1)


def _mask_from_wire(rle, size: tuple[int, int],
                    label: str = "") -> masker.annotations.instance_mask.InstanceMask:
    """InstanceMask of a client rle of the size (w, h) image; label prefixes the 400."""
    mask = masker.annotations.instance_mask.InstanceMask.from_json(rle)
    width, height = size
    if mask.size != (height, width):
        raise masker.web.webcore.bad_request("{}rle size {!r} must match the {}x{} image".format(
            label, rle["size"], width, height))
    return mask


def _touch_up(mask: masker.annotations.instance_mask.InstanceMask, rect: rect_t,
              ops) -> masker.annotations.instance_mask.InstanceMask:
    """mask with the draw ops rasterized over rect; only the rect is decoded."""
    x0, y0 = rect[0], rect[1]
    crop = PIL.Image.fromarray(mask.rect_array(rect) * np.uint8(255), "L")
    draw = PIL.ImageDraw.Draw(crop)
    for op in ops:
        _draw_op(draw, op, x0, y0)
    return mask.with_rect(rect, np.asarray(crop) > 127)


def _instance_payload(
    inst: masker.execution.model.ObjectInstance,
    recrop: masker.execution.recrop.Recrop | None,
    full_size: tuple[int, int] | None,
) -> dict:
    """Wire payload for one instance; with recrop the mask and bbox map
    back to original-image coordinates (full_size = original (w, h))."""
    mask, bbox, centroid = inst.mask, inst.bbox_xyxy, inst.centroid
    if recrop is not None:
        full_w, full_h = full_size
        bbox = recrop.to_global_bbox(bbox)
        if centroid is not None:
            centroid = recrop.point_to_global_coords(*centroid)
        encoded = recrop.to_instance_mask(mask, full_h, full_w)
    else:
        encoded = masker.annotations.instance_mask.InstanceMask.from_array(mask)
    rect = _mask_rect(encoded)

    def render():
        if recrop is not None:
            ys, xs = _mask_sample_grid(rect)
            return masker.web.webcore.mask_png_bytes(recrop.sample_original(mask, ys, xs))
        return _full_mask_png(mask, rect)

    png = _cached_png(encoded, render)
    if centroid is None:
        centroid = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
    return {
        "score": round(float(inst.score), 4),
        "bbox_xyxy": [round(float(v), 2) for v in bbox],
        "centroid": [round(float(v), 2) for v in centroid],
        "mask_png": _png_url(png),
        "mask_rect": list(rect),
        "rle": encoded.to_json(),
    }


def _drawn_payload(mask: masker.annotations.instance_mask.InstanceMask) -> dict:
    """Instance payload of a drawn mask."""
    rect = _mask_rect(mask)
    png = _cached_png(mask, lambda: _rect_png(mask, rect))
    return {
        "score": 1.0,
        "bbox_xyxy": [round(float(v), 2) for v in mask.bbox],
        "centroid": [round(float(v), 2) for v in mask.centroid()],
        "mask_png": _png_url(png),
        "mask_rect": list(rect),
        "rle": mask.to_json(),
    }


def _wire(ev: dict, full_size: tuple[int, int]) -> str:
    """One NDJSON line of a sweep event: Recrops to rects, instances to payloads."""
    out = {}
    for k, v in ev.items():
        if isinstance(v, masker.execution.recrop.Recrop):
            v = list(v.rect)
        elif k == "recrops":
            v = [list(r.rect) for r in v]
        elif k == "instances":
            v = [_instance_payload(i, ev["recrop"], full_size) for i in v]
        elif k == "seconds":
            v = round(v, 3)
        elif k == "subsumed":
            continue
        out[k] = v
    return json.dumps(out) + "\n"


def _wire_stream(events, full_size: tuple[int, int]):
    """One line per event of the events generator, in order. The generator
    runs on its own thread and lines encode on _WIRE_POOL. A generator
    exception is raised after the lines before it; closing the stream closes
    the generator."""
    pending: queue.Queue = queue.Queue(WebLimits.WIRE_QUEUE)
    stop = threading.Event()

    def produce():
        try:
            for ev in events:
                pending.put(_WIRE_POOL.submit(_wire, ev, full_size))
                if stop.is_set():
                    break
        except BaseException as exc:  # noqa: BLE001 - re-raised on the request thread
            pending.put(exc)
        else:
            pending.put(None)
        finally:
            events.close()

    threading.Thread(target=produce, daemon=True).start()
    try:
        while (item := pending.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item.result()
    finally:
        stop.set()
        while not pending.empty():  # frees a blocked put; the producer then stops
            pending.get_nowait()


def _ndjson(events, full_size: tuple[int, int]) -> flask.Response:
    """NDJSON response of events; a mid-stream exception ends the stream
    with an error event."""
    def stream():
        try:
            yield from _wire_stream(events, full_size)
        except Exception as exc:  # noqa: BLE001 - mid-stream model error
            yield json.dumps({"type": "error", "error": str(exc)}) + "\n"

    return flask.Response(stream(), mimetype="application/x-ndjson")


def create_app(device: str | None = None,
               threshold: float = masker.execution.model.MaskerDefaults.THRESHOLD,
               generator=None, root: pathlib.Path | None = None):
    """root anchors relative client paths."""
    app = flask.Flask(__name__)
    state = {"generator": generator}
    previews = masker.web.webcore.PreviewCache(
        int(WebLimits.SOURCE_IMAGE_CACHE_GB * (1 << 30)), WebLimits.PREVIEW_CACHE_MB << 20)

    root_dir = root.resolve() if root is not None else pathlib.Path.cwd()

    def resolve_path(raw: str) -> pathlib.Path:
        """Client path anchored at root_dir; not resolved."""
        p = pathlib.Path(raw.strip())
        return p if (p.drive or p.root) else root_dir / p

    def get_masker():
        with _MODEL_LOCK:
            if state["generator"] is None:
                state["generator"] = masker.execution.model.build_masker(
                    device=device, threshold=threshold)
            return state["generator"]

    def image_file(payload) -> pathlib.Path:
        path = resolve_path(payload["image_path"])
        if not path.is_file():
            raise masker.web.webcore.bad_request("no such file: {}".format(path))
        return path

    def loaded(payload, load):
        """(path, load(path)) for the payload's image."""
        path = image_file(payload)
        try:
            return path, load(path)
        except Exception as exc:  # noqa: BLE001 - undecodable file
            raise masker.web.webcore.bad_request("invalid image: {}".format(exc))

    def sweep_params(payload) -> masker.execution.sweep.SweepParams:
        try:
            return masker.execution.sweep.SweepParams.resolve(**payload["params"])
        except (TypeError, ValueError) as exc:
            raise masker.web.webcore.bad_request(str(exc))

    # the directory's COCO file + its cache; store_lock also serializes
    # search-index upserts
    store = masker.io.coco.CocoStore()
    store_lock = store.lock
    find_dataset = masker.io.coco.find_coco_file
    load_dataset = store.load
    store_dataset = store.store

    def check_fresh(dataset, annotation_id, image_name, bbox_xyxy) -> None:
        """409 when a client-held annotation reference is stale."""
        if masker.io.coco.stale_annotation(dataset, annotation_id, image_name, bbox_xyxy):
            raise masker.web.webcore.ReplyError(
                409, "annotation #{} is out of date (dataset changed); "
                     "nothing deleted".format(annotation_id))

    def dataset_for_dir(raw: str):
        """(directory, dataset or None) for a directory or file path."""
        target = resolve_path(raw)
        directory = target if target.is_dir() else target.parent
        if not directory.is_dir():
            raise masker.web.webcore.bad_request("no such directory: {}".format(raw))
        source = find_dataset(directory)
        return directory, (load_dataset(source) if source is not None else None)

    masker.web.webcore.register_page(app, _index_html(), masker.web.webcore.asset("webui.css"))
    masker.web.webcore.register_errors(app)

    @app.post("/load")
    def load() -> flask.Response:
        payload = flask.request.get_json()
        path, jpeg = loaded(payload, lambda p: previews.jpeg(p, WebLimits.PREVIEW_MAX_SIDE))
        width, height = masker.io.imaging.image_size(path)

        source = find_dataset(path.parent)
        categories: list[dict] = []
        saved: list[dict] = []
        tree: list[dict] = []
        if source is not None:
            dataset = load_dataset(source)
            image_anns = masker.io.coco.annotations_for_image(dataset, path.name)
            total = collections.Counter(a["category_id"] for a in dataset["annotations"])
            in_image = collections.Counter(a["category_id"] for a in image_anns)
            categories = [
                {
                    "id": c["id"],
                    "name": c["name"],
                    "supercategory": c.get("supercategory") or "",
                    "total": total[c["id"]],
                    "in_image": in_image[c["id"]],
                }
                for c in dataset["categories"]
            ]
            tree = masker.annotations.annotation_management.category_tree(dataset)
            for a in image_anns:
                x, y, w, h = a["bbox"]
                seg = a.get("segmentation")
                mask_png, mask_rect = _mask_png_from_segmentation(seg)
                saved.append(
                    {
                        "annotation_id": a["id"],
                        "category_id": a["category_id"],
                        "bbox_xyxy": [x, y, x + w, y + h],
                        "mask_png": mask_png,
                        "mask_rect": mask_rect,
                        # compressed masks round-trip; polygons stay None
                        "rle": (seg.to_json()
                                if isinstance(seg, masker.annotations.instance_mask.InstanceMask)
                                else None),
                    }
                )
        prev_path, next_path = (
            masker.io.imaging.SourceImageDirectory(path.parent).neighbors(path.name))
        with store_lock:
            index, index_status = masker.execution.search_index.load_with_status(path)
        return flask.jsonify(
            {
                "preview": masker.web.webcore.data_url(jpeg, "image/jpeg"),
                "width": width,
                "height": height,
                "coco_file": str(source) if source is not None else None,
                "new_coco_name": masker.io.coco.CocoLayout.ANNOTATIONS_NAME,
                "categories": categories,
                "tree": tree,
                "saved": saved,
                "prev": str(prev_path) if prev_path is not None else None,
                "next": str(next_path) if next_path is not None else None,
                "search_index": index,
                "search_index_status": index_status,
            }
        )

    @app.get("/image")
    def image_proxy() -> flask.Response:
        """JPEG of path with its longest side at resolution, mtime ETag."""
        path = resolve_path(flask.request.args["path"])
        if not path.is_file():
            raise masker.web.webcore.bad_request("no such file: {}".format(path))
        return masker.web.webcore.image_response(previews, path)

    @app.get("/fsmeta")
    def fsmeta() -> flask.Response:
        """Listing of a file's directory or a directory (empty path: the server root)."""
        raw = flask.request.args["path"]
        target = resolve_path(raw) if raw else root_dir
        if target.is_dir():
            directory, name = target, None
        elif target.is_file():
            directory, name = target.parent, target.name
        else:
            raise masker.web.webcore.bad_request("no such path: {}".format(raw or target))
        try:
            subdirs = sorted(
                (
                    p.name
                    for p in directory.iterdir()
                    if p.is_dir() and not p.name.startswith(".")
                ),
                key=str.lower,
            )
            source = masker.io.imaging.SourceImageDirectory(directory)
            image_names = [p.name for p in source.images()]
            mask_names = [p.name for p in source.zephyr_masks()]
        except OSError as exc:
            raise masker.web.webcore.bad_request("cannot list {}: {}".format(directory, exc))
        index = image_names.index(name) if name in image_names else None
        if name is not None and index is None:
            # Windows: case-insensitive, trailing dots ignored; match folded
            # and echo the on-disk name
            folded = [n.casefold() for n in image_names]
            wanted = name.rstrip(". ").casefold()
            if wanted in folded:
                index = folded.index(wanted)
                name = image_names[index]
        return flask.jsonify(
            {
                "dir": str(directory),
                "parent": str(directory.parent) if directory.parent != directory else None,
                "sep": os.sep,
                "subdirs": subdirs,
                "images": image_names,
                "masks": mask_names,
                "count": len(image_names),
                "index": index,
                "name": name,
            }
        )

    @app.get("/dir_counts")
    def dir_counts() -> flask.Response:
        """Per-image, per-category saved-annotation counts for a directory."""
        directory, dataset = dataset_for_dir(flask.request.args["path"])
        if dataset is None:
            return flask.jsonify({"dir": str(directory), "sep": os.sep,
                            "categories": [], "images": []})
        by_id = {c["id"]: c["name"] for c in dataset["categories"]}
        per_image: dict[int, collections.Counter] = {}
        for a in dataset["annotations"]:
            per_image.setdefault(a["image_id"], collections.Counter())[a["category_id"]] += 1
        # keyed by basename, merging subpath file_names
        rows: dict[str, collections.Counter] = {}
        for img in dataset["images"]:
            base = pathlib.Path(img.get("file_name", "")).name
            rows.setdefault(base, collections.Counter()).update(
                per_image.get(img["id"], collections.Counter())
            )
        images_out = [
            {
                "name": base,
                "counts": {
                    by_id[cid]: n
                    for cid, n in sorted(counts.items())
                    if cid in by_id
                },
            }
            for base, counts in sorted(rows.items(), key=lambda kv: kv[0].lower())
        ]
        return flask.jsonify(
            {
                "dir": str(directory),
                "sep": os.sep,
                "categories": [
                    {"name": c["name"]} for c in dataset["categories"]
                ],
                "images": images_out,
            }
        )

    @app.get("/dir_stats")
    def dir_stats() -> flask.Response:
        """Per-category log2 area histogram over edges plus area and bbox extremes."""
        directory, dataset = dataset_for_dir(flask.request.args["path"])
        edges = [2**k for k in range(4, 25)]
        if dataset is None:
            return flask.jsonify({"edges": edges, "labels": []})
        stats: dict[int, dict] = {}
        for a in dataset["annotations"]:
            w, h = a["bbox"][2], a["bbox"][3]
            area = int(a.get("area") or w * h)
            s = stats.get(a["category_id"])
            if s is None:
                s = stats[a["category_id"]] = {
                    "count": 0, "bins": [0] * (len(edges) - 1),
                    "area_min": area, "area_max": area,
                    "w_min": w, "w_max": w, "h_min": h, "h_max": h,
                }
            s["count"] += 1
            s["bins"][min(max(area.bit_length() - 5, 0), len(edges) - 2)] += 1
            s["area_min"] = min(s["area_min"], area)
            s["area_max"] = max(s["area_max"], area)
            s["w_min"] = min(s["w_min"], w)
            s["w_max"] = max(s["w_max"], w)
            s["h_min"] = min(s["h_min"], h)
            s["h_max"] = max(s["h_max"], h)
        labels = [
            {
                "name": c["name"],
                "count": s["count"],
                "area": {"min": s["area_min"], "max": s["area_max"],
                         "bins": s["bins"]},
                "bbox": {"w_min": s["w_min"], "w_max": s["w_max"],
                         "h_min": s["h_min"], "h_max": s["h_max"]},
            }
            for c in dataset["categories"]
            if (s := stats.get(c["id"]))
        ]
        return flask.jsonify({"edges": edges, "labels": labels})

    @app.post("/crop_preview")
    def crop_preview() -> flask.Response:
        """Preview-quality JPEG of one region of the original, for zoom past the preview."""
        payload = flask.request.get_json()
        _, source_image = loaded(payload, previews.source_image)
        return flask.jsonify(masker.web.webcore.crop_preview_payload(
            source_image, payload["rect_xyxy"], WebLimits.PREVIEW_MAX_SIDE))

    @app.get("/model_status")
    def model_status() -> flask.Response:
        # No lock: reads must answer while the build holds _MODEL_LOCK.
        return flask.jsonify({"loaded": state["generator"] is not None})

    @app.get("/cache_stats")
    def cache_stats() -> flask.Response:
        """Embedding-cache counters; images_left extrapolates the disk budget."""
        if state["generator"] is None:
            return flask.jsonify({"present": False})
        cache = state["generator"]._embed_cache
        s = cache.stats()
        record = cache.record_bytes
        per_image = s["grid_recrops"] * record
        lookups = s["hits"] + s["misses"]
        return flask.jsonify({
            "present": True,
            "disk_bytes": s["disk_bytes"],
            "disk_budget": s["disk_budget"],
            "disk_tensors": s["disk_bytes"] // record,
            "hits": s["hits"],
            "misses": s["misses"],
            "hit_rate": s["hits"] / lookups if lookups else None,
            "ratio": s["ratio"],
            "images_left": max(0, s["disk_budget"] - s["disk_bytes"]) // per_image
                           if per_image else None,
        })

    @app.post("/segment")
    def segment() -> flask.Response:
        payload = flask.request.get_json()
        path, source_image = loaded(payload, previews.source_image)
        # optional recrop: segment only this rect of the original at up to full
        # resolution; everything sent and returned stays in original coordinates
        recrop = None
        if payload.get("recrop") is not None:
            recrop = masker.execution.recrop.Recrop.from_rect(
                *_rect(payload["recrop"]), WebLimits.PREVIEW_MAX_SIDE)
        prompt = masker.execution.model.SegmentationPrompt.from_payload(payload, recrop)
        generator = get_masker()
        source_key = previews.source_key(path)
        start = time.perf_counter()
        if recrop is not None:
            instances = generator.segment_recrop(source_image, recrop, prompt,
                                                 source_key=source_key)
        else:
            instances = generator.segment_instances(source_image, prompt,
                                                    source_key=source_key)
        elapsed = time.perf_counter() - start
        return flask.jsonify(
            {
                "instances": [_instance_payload(i, recrop, source_image.size)
                              for i in instances],
                "stats": {"generation_seconds": round(elapsed, 2)},
            }
        )

    @app.post("/sweep")
    def sweep() -> flask.Response:
        """The sweep (execution.sweep) as NDJSON; knobs ride in params."""
        payload = flask.request.get_json()
        path, source_image = loaded(payload, previews.source_image)
        params = sweep_params(payload)
        generator = get_masker()
        events = masker.execution.sweep.sweep_image(
            generator, source_image, payload["text"], params=params,
            source_key=previews.source_key(path))
        return _ndjson(events, source_image.size)

    @app.post("/reprompt_bboxes")
    def reprompt_bboxes() -> flask.Response:
        """Re-prompt drawn bboxes on context recrops, as NDJSON."""
        payload = flask.request.get_json()
        path, source_image = loaded(payload, previews.source_image)
        params = sweep_params(payload)
        generator = get_masker()
        width, height = source_image.size
        bboxes = [tuple(float(v) for v in b) for b in payload["bboxes"]]
        recrops = [
            masker.execution.sweep.plan_cluster_recrop(
                b, width, height, expand=params.reprompt_expand)
            for b in bboxes
        ]
        events = masker.execution.sweep.reprompt_recrops(
            generator, source_image, recrops, bboxes, [], text=payload["text"],
            params=params, pack=params.pack, source_key=previews.source_key(path))
        return _ndjson(events, source_image.size)

    @app.post("/dedup_instances")
    def dedup_instances() -> flask.Response:
        """Group masks by IoU >= iou and union each group."""
        payload = flask.request.get_json()
        _, size = loaded(payload, masker.io.imaging.image_size)
        width, height = size
        masks = masker.annotations.instance_mask.InstanceMaskSet(
            tuple(masker.annotations.instance_mask.InstanceMask.from_json(r)
                  for r in payload["rles"]),
            height, width)
        # null scores are saved anchors: committed data, they seed first
        scores = [math.inf if s is None else float(s) for s in payload["scores"]]
        groups = masks.overlap_groups(float(payload["iou"]), scores=scores)
        return flask.jsonify(_group_response(masks, groups))

    @app.post("/draw_mask")
    def draw_mask() -> flask.Response:
        """Rasterize ops over an optional base_rle inside rect; {empty: true} when nothing is set."""
        payload = flask.request.get_json()
        _, size = loaded(payload, masker.io.imaging.image_size)
        width, height = size
        base = payload["base_rle"]
        mask = _touch_up(
            masker.annotations.instance_mask.InstanceMask.empty(height, width) if base is None
            else masker.annotations.instance_mask.InstanceMask.from_json(base),
            _rect(payload["rect"]), payload["ops"])
        if mask.area == 0:
            return flask.jsonify({"empty": True})
        return flask.jsonify(_drawn_payload(mask))

    @app.post("/mask_crop")
    def mask_crop() -> flask.Response:
        """Native-resolution PNG of rle inside rect, for touch-up editing."""
        payload = flask.request.get_json()
        _, size = loaded(payload, masker.io.imaging.image_size)
        mask = _mask_from_wire(payload["rle"], size)
        rect = _rect(payload["rect"])
        png = _rect_png(mask, rect)
        return flask.jsonify({"mask_png": _png_url(png), "rect": list(rect)})

    @app.post("/update_annotation")
    def update_annotation() -> flask.Response:
        """Rasterize ops over a saved annotation's mask and rewrite it under its id."""
        payload = flask.request.get_json()
        source = resolve_path(payload["coco_file"])
        annotation_id = payload["annotation_id"]
        path, _ = loaded(payload, masker.io.imaging.image_size)
        rect = _rect(payload["rect"])
        with store_lock:
            dataset = load_dataset(source)
            check_fresh(dataset, annotation_id, path.name, payload["bbox_xyxy"])
            target = next(a for a in dataset["annotations"] if a["id"] == annotation_id)
            mask = _touch_up(target["segmentation"], rect, payload["ops"])
            if mask.area == 0:
                raise masker.web.webcore.bad_request(
                    "touch-up erased the whole mask; delete the instance instead")
            mask = dataclasses.replace(
                mask, score=target.get("score"), category_id=target["category_id"],
                annotation_id=annotation_id)
            ann = {**target, **mask.to_coco(image_id=target["image_id"])}
            store_dataset(masker.io.coco.replace_annotation(dataset, ann), source)
        return flask.jsonify({**_drawn_payload(mask), "coco_file": str(source)})

    @app.post("/save")
    def save() -> flask.Response:
        payload = flask.request.get_json()
        path, size = loaded(payload, masker.io.imaging.image_size)
        if masker.io.zephyr.is_zephyr_mask(path):
            raise masker.web.webcore.bad_request("Zephyr masks are view-only")
        width, height = size
        instances, deletes = payload["instances"], payload["deletes"]
        masks = [_mask_from_wire(inst["rle"], size, "instances[{}]: ".format(i))
                 for i, inst in enumerate(instances)]

        with store_lock:  # one load->modify->store, atomic vs other requests
            source = (resolve_path(payload["coco_file"]) if payload["coco_file"]
                      else find_dataset(path.parent))
            if source is not None:
                dataset = load_dataset(source)
            else:
                source = path.parent / masker.io.coco.CocoLayout.ANNOTATIONS_NAME
                dataset = masker.io.coco.empty_coco()
            # verify all deletes before applying any; deletes and new
            # instances share one write
            for d in deletes:
                check_fresh(dataset, d["annotation_id"], path.name, d["bbox_xyxy"])
            for d in deletes:
                dataset = masker.io.coco.remove_annotation(dataset, d["annotation_id"])

            dataset, image_id = masker.io.coco.ensure_image(dataset, path.name, width, height)
            known = {c["name"] for c in dataset["categories"]}
            new_names = sorted({i["label"] for i in instances} - known)
            if new_names:
                dataset = masker.annotations.annotation_management.add_labels(dataset, new_names)
            by_name = {c["name"]: c["id"] for c in dataset["categories"]}

            annotations = [
                dataclasses.replace(mask, category_id=by_name[inst["label"]])
                .to_coco(image_id=image_id)
                for inst, mask in zip(instances, masks)
            ]
            if annotations:
                dataset = masker.io.coco.append_annotations(dataset, annotations)
            store_dataset(dataset, source)

        added = dataset["annotations"][-len(annotations):] if annotations else []
        return flask.jsonify(
            {
                "coco_file": str(source),
                "saved": [
                    {
                        "annotation_id": a["id"],
                        "bbox_xyxy": [
                            a["bbox"][0], a["bbox"][1],
                            a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3],
                        ],
                    }
                    for a in added
                ],
            }
        )

    @app.post("/record_searches")
    def record_searches() -> flask.Response:
        payload = flask.request.get_json()
        path = image_file(payload)
        with store_lock:
            index = masker.execution.search_index.upsert_searches(
                path, payload["searches"], width=payload["width"], height=payload["height"])
        return flask.jsonify({"index": index})

    @app.post("/delete_annotation")
    def delete_annotation() -> flask.Response:
        payload = flask.request.get_json()
        source = resolve_path(payload["coco_file"])
        annotation_id = payload["annotation_id"]
        with store_lock:
            dataset = load_dataset(source)
            check_fresh(dataset, annotation_id, resolve_path(payload["image_path"]).name,
                        payload["bbox_xyxy"])
            store_dataset(masker.io.coco.remove_annotation(dataset, annotation_id), source)
        return flask.jsonify({"coco_file": str(source)})

    @app.post("/edit_labels")
    def edit_labels() -> flask.Response:
        """Dataset-wide label edit: op is merge (labels, new_name), drop
        (labels), add (label), or parent (name, parent)."""
        payload = flask.request.get_json()
        source = resolve_path(payload["coco_file"])
        op = payload["op"]
        am = masker.annotations.annotation_management
        with store_lock:
            dataset = load_dataset(source)
            try:
                if op == "merge":
                    updated = am.merge_labels(dataset, payload["labels"], payload["new_name"])
                elif op == "drop":
                    updated = am.drop_labels(dataset, payload["labels"])
                elif op == "add":
                    updated = am.add_labels(dataset, [payload["label"]])
                else:
                    updated = am.set_category_parent(dataset, payload["name"], payload["parent"])
            except (ValueError, KeyError) as exc:
                raise masker.web.webcore.bad_request(exc.args[0])
            store_dataset(updated, source)
        return flask.jsonify(
            {
                "coco_file": str(source),
                "categories": [{"id": c["id"], "name": c["name"]}
                               for c in updated["categories"]],
                "removed_annotations": len(dataset["annotations"]) - len(updated["annotations"]),
            }
        )

    @app.post("/export_zephyr")
    def export_zephyr() -> flask.Response:
        """Zephyr masks beside the images: the selected labels' union black, the rest white."""
        payload = flask.request.get_json()
        labels, force = payload["labels"], payload["force"]
        if not labels:
            raise masker.web.webcore.bad_request("no labels selected")
        image_dir = image_file(payload).parent
        source = find_dataset(image_dir)
        if source is None:
            raise masker.web.webcore.bad_request("no annotations to export")
        dataset = load_dataset(source)
        if payload["missing_ok"]:
            known = {c["name"] for c in dataset["categories"]}
            labels = [label for label in labels if label in known]
        try:
            ids = masker.annotations.annotation_management.ids_for_labels(dataset, labels)
        except KeyError as exc:
            raise masker.web.webcore.bad_request(exc.args[0])
        names = {img["id"]: pathlib.Path(img["file_name"]).name for img in dataset["images"]}
        by_name: dict[str, list[masker.annotations.instance_mask.InstanceMask]] = {}
        for a in dataset["annotations"]:
            seg = a.get("segmentation")
            if (a["category_id"] in ids
                    and isinstance(seg, masker.annotations.instance_mask.InstanceMask)):
                by_name.setdefault(names.get(a["image_id"], ""), []).append(seg)
        sizes = {pathlib.Path(i["file_name"]).name: (i["width"], i["height"])
                 for i in dataset["images"] if i.get("width") and i.get("height")}
        written = skipped = 0
        for image_path in masker.io.imaging.SourceImageDirectory(image_dir).images():
            masks = by_name.get(image_path.name, [])
            if not masks and not force:
                skipped += 1
                continue
            if masks:
                h, w = masks[0].size
            else:
                w, h = sizes.get(image_path.name) or masker.io.imaging.image_size(image_path)
            try:
                mask_set = masker.annotations.instance_mask.InstanceMaskSet(tuple(masks), h, w)
            except ValueError as exc:
                raise masker.web.webcore.bad_request("{}: {}".format(image_path.name, exc))
            masker.io.zephyr.write_zephyr_mask(image_path, mask_set)
            written += 1
        return flask.jsonify({"dir": str(image_dir), "written": written, "skipped": skipped})

    return app
