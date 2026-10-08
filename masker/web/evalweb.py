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

"""Dashboard scoring a directory's predicted COCO against its reference COCO"""

from __future__ import annotations

import json
import pathlib

import flask

import masker.annotations.instance_mask
import masker.execution.evaluate
import masker.io.coco
from masker.io.coco import coco_t
import masker.io.imaging
import masker.web
import masker.web.webcore


class EvalWebParams:
    # Longest side of the preview JPEG and overlay PNGs sent to the browser.
    PREVIEW_MAX_SIDE = 2048
    # Web UI port (masker-eval).
    PORT = 5003
    # decoded originals behind /image
    SOURCE_IMAGE_CACHE_GB = 0.5
    # preview JPEGs keyed by (path, mtime, side)
    PREVIEW_CACHE_MB = 64


_CFG_JSON = json.dumps({
    "MATCH_IOU": masker.execution.evaluate.EvalDefaults.MATCH_IOU,
    "PREVIEW_SIDE": EvalWebParams.PREVIEW_MAX_SIDE,
    "BANDS": [masker.execution.evaluate.band_label(i)
              for i in range(len(masker.execution.evaluate.EvalDefaults.SIZE_BANDS))],
})

# Prediction COCO files looked for under the benchmark directory, in order.
_PREDICTION_NAMES = ("out/predictions.json", "eval_predictions.json")


def _index_html() -> str:
    return masker.web.webcore.page_html("evalweb_page.html", {
        "VIEWER_JS": masker.web.webcore.asset("webui_viewer.js"),
        "CFG_JSON": _CFG_JSON,
        "APP_JS": masker.web.webcore.asset("evalweb_app.js"),
    })


def _layer_png(
    masks: masker.annotations.instance_mask.InstanceMaskSet, max_side: int
) -> bytes | None:
    """Overlay PNG of one mask layer"""
    if not masks:
        return None
    return masker.web.webcore.mask_png_bytes(
        masker.web.webcore.cap_binary(masks.union_all_masks().to_array(), max_side))


def _instance_entry(mask: masker.annotations.instance_mask.InstanceMask, category: str, kind: str,
                    iou: float | None = None) -> dict:
    entry = {
        "category": category,
        "kind": kind,
        "bbox_xyxy": [round(v, 2) for v in mask.bbox],
        "area": mask.area,
        "score": mask.score,
    }
    if iou is not None:
        entry["iou"] = round(iou, 4)
    return entry


def create_app(root_dir: pathlib.Path, reference_path: pathlib.Path | None = None,
               predictions_path: pathlib.Path | None = None):
    """Build the Flask app for one benchmark directory"""
    app = flask.Flask(__name__)
    store = masker.io.coco.CocoStore()
    # Entries carry their source dataset dict; hits compare with is.
    pred_cache: dict = {}
    report_cache: dict = {}
    eval_cache: dict = {}
    # annotation indexes, one slot per source, keyed by dataset identity
    index_cache: dict = {}
    previews = masker.web.webcore.PreviewCache(
        int(EvalWebParams.SOURCE_IMAGE_CACHE_GB * (1 << 30)),
        EvalWebParams.PREVIEW_CACHE_MB << 20)

    def load_reference() -> tuple[coco_t | None, pathlib.Path | None]:
        source = reference_path if reference_path is not None else masker.io.coco.find_coco_file(root_dir)
        if source is None or not source.exists():
            return None, None
        return store.load(source), source

    def load_predictions() -> tuple[coco_t | None, pathlib.Path | None]:
        if predictions_path is not None:
            path = predictions_path if predictions_path.exists() else None
        else:
            path = next((p for p in (root_dir / n for n in _PREDICTION_NAMES)
                         if p.exists()), None)
        if path is None:
            return None, None
        key = (str(path), path.stat().st_mtime_ns)
        if pred_cache.get("key") != key:
            pred_cache.update(key=key, data=masker.io.coco.load_coco(path))
        return pred_cache["data"], path

    def annotation_index(slot: str, dataset: coco_t | None) -> dict:
        """Instance index of dataset, reused while the dataset is unchanged"""
        if dataset is None:
            return {}
        entry = index_cache.get(slot)
        if entry is None or entry[0] is not dataset:
            index_cache[slot] = entry = (
                dataset, masker.execution.evaluate.index_instances(dataset))
        return entry[1]

    def build_report(reference: coco_t, predicted: coco_t, mode: str, match_iou: float, downscale: float):
        params = (mode, match_iou, downscale)
        if not (report_cache.get("ref") is reference
                and report_cache.get("pred") is predicted
                and report_cache.get("params") == params):
            report_cache.update(ref=reference, pred=predicted, params=params,
                                data=masker.execution.evaluate.compare_datasets(
                                    predicted, reference, mode=masker.execution.evaluate.EvalMode(mode),
                                    match_iou=match_iou, downscale=downscale))
        return report_cache["data"]

    def image_for(name: str) -> tuple[str, pathlib.Path]:
        name = pathlib.Path(name).name
        path = root_dir / name
        if not path.is_file():
            raise masker.web.webcore.bad_request("no such image: {}".format(name))
        return name, path

    masker.web.webcore.register_page(app, _index_html(), masker.web.webcore.asset("webui.css"))
    masker.web.webcore.register_errors(app)

    @app.get("/summary")
    def summary() -> flask.Response:
        args = flask.request.args
        match_iou, downscale, mode = float(args["match_iou"]), float(args["downscale"]), args["mode"]
        reference, source = load_reference()
        predicted, pred_path = load_predictions()
        report = None
        if reference is not None and predicted is not None:
            report = build_report(reference, predicted, mode, match_iou, downscale)
        ref_index = annotation_index("ref", reference)
        pred_index = annotation_index("pred", predicted)
        images = []
        for path in masker.io.imaging.SourceImageDirectory(root_dir).images():
            # flat directory: basename == root-relative posix name
            by_cat = ref_index.get(path.name, {})
            images.append({
                "name": path.name,
                "ref": sum(len(v) for v in by_cat.values()),
                "categories": len(by_cat),
                "predicted": sum(len(v) for v in pred_index.get(path.name, {}).values()),
            })
        return flask.jsonify({
            "root": str(root_dir),
            "reference": str(source) if source is not None else None,
            "predictions": str(pred_path) if pred_path is not None else None,
            "report": report,
            "images": images,
        })

    @app.get("/image")
    def image() -> flask.Response:
        _, path = image_for(flask.request.args["name"])
        return masker.web.webcore.image_response(previews, path)

    @app.get("/image_eval")
    def image_eval() -> flask.Response:
        name, path = image_for(flask.request.args["name"])
        match_iou = float(flask.request.args["match_iou"])
        category = flask.request.args.get("category")
        reference, _ = load_reference()
        predicted, _ = load_predictions()
        params = (name, category, match_iou)
        if (eval_cache.get("ref") is reference and eval_cache.get("pred") is predicted
                and eval_cache.get("params") == params):
            return flask.jsonify(eval_cache["data"])
        ref_by_cat = annotation_index("ref", reference).get(name, {})
        pred_by_cat = annotation_index("pred", predicted).get(name, {})
        names = sorted(set(ref_by_cat) | set(pred_by_cat))
        scored = names if category is None else [n for n in names if n == category]
        width, height = masker.io.imaging.image_size(path)
        empty = masker.annotations.instance_mask.InstanceMaskSet((), height, width)
        instances = []
        layer_masks: dict[str, list[masker.annotations.instance_mask.InstanceMask]] = {
            "tp": [], "fn": [], "fp": []}
        for cat in scored:
            ref = ref_by_cat.get(cat, empty)
            pred = pred_by_cat.get(cat, empty)
            pairs = masker.execution.evaluate.match_greedy(pred.iou_matrix(ref), match_iou)
            matched_p = {p for p, _, _ in pairs}
            matched_r = {r for _, r, _ in pairs}
            for p, _, iou in pairs:
                instances.append(_instance_entry(pred[p], cat, "tp", iou=iou))
                layer_masks["tp"].append(pred[p])
            for i, mask in enumerate(ref):
                if i not in matched_r:
                    instances.append(_instance_entry(mask, cat, "fn"))
                    layer_masks["fn"].append(mask)
            for i, mask in enumerate(pred):
                if i not in matched_p:
                    instances.append(_instance_entry(mask, cat, "fp"))
                    layer_masks["fp"].append(mask)
        layers = {}
        for kind, masks in layer_masks.items():
            png = _layer_png(
                masker.annotations.instance_mask.InstanceMaskSet(tuple(masks), height, width),
                EvalWebParams.PREVIEW_MAX_SIDE)
            layers[kind] = masker.web.webcore.data_url(png, "image/png") if png else None
        payload = {
            "name": name,
            "width": width,
            "height": height,
            "categories": names,
            "counts": {kind: len(masks) for kind, masks in layer_masks.items()},
            "instances": instances,
            "layers": layers,
        }
        eval_cache.update(ref=reference, pred=predicted, params=params, data=payload)
        return flask.jsonify(payload)

    return app
