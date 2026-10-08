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

"""Helpers shared by the masker and eval web apps"""

from __future__ import annotations

import base64
import io
import math
import pathlib

import flask
import numpy as np
import PIL.Image
import werkzeug.exceptions

import masker.cache.ephemeral_cache
from masker.common.common_defs import binary_mask_t, index_array_t
import masker.io.imaging


def data_url(data: bytes, mime: str) -> str:
    return "data:{};base64,".format(mime) + base64.b64encode(data).decode("ascii")


class ReplyError(Exception):
    # ends the request with {"error": message} as JSON at status
    __slots__ = 'status',
    status: int

    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status


def bad_request(message: str) -> ReplyError:
    return ReplyError(400, message)


# --- page assets and routes -------------------------------------------------


def asset(name: str) -> str:
    return pathlib.Path(__file__).with_name(name).read_text(encoding="utf-8")


def page_html(name: str, replacements: dict[str, str]) -> str:
    html = asset(name)
    for key, value in replacements.items():
        html = html.replace("__{}__".format(key), value)
    return html


def register_page(app: flask.Flask, index_html: str, css: str) -> None:
    """Serve the app page and the shared stylesheet"""
    headers = {"Cache-Control": "no-store"}

    @app.get("/")
    def index() -> flask.Response:
        return flask.Response(index_html, mimetype="text/html", headers=headers)

    @app.get("/webui.css")
    def webui_css() -> flask.Response:
        return flask.Response(css, mimetype="text/css", headers=headers)


def register_errors(app: flask.Flask) -> None:
    @app.errorhandler(ReplyError)
    def reply(exc: ReplyError):
        return flask.jsonify({"error": str(exc)}), exc.status

    @app.errorhandler(Exception)
    def failed(exc: Exception):
        if isinstance(exc, werkzeug.exceptions.HTTPException):
            return exc
        app.logger.exception("request failed")
        return flask.jsonify({"error": str(exc)}), 500


# --- image previews ---------------------------------------------------------


def thumbnail_jpeg(image: PIL.Image.Image, max_side: int) -> bytes:
    """Preview JPEG of image fitted to max_side"""
    scaled = image.copy()
    scaled.thumbnail((max_side, max_side))
    buf = io.BytesIO()
    scaled.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


class PreviewCache:
    """Decoded source images and their preview JPEGs, reused across requests"""

    __slots__ = '_sources', '_jpegs'
    # decoded originals
    _sources: masker.cache.ephemeral_cache.EphemeralCache[PIL.Image.Image]
    # JPEGs keyed by (source key, side)
    _jpegs: masker.cache.ephemeral_cache.EphemeralCache[bytes]

    def __init__(self, source_bytes: int, jpeg_bytes: int) -> None:
        self._sources = masker.cache.ephemeral_cache.EphemeralCache(source_bytes)
        self._jpegs = masker.cache.ephemeral_cache.EphemeralCache(jpeg_bytes)

    @staticmethod
    def source_key(path: pathlib.Path) -> str:
        """Cache identity of an image file; changes when the file is rewritten."""
        return "{}:{}".format(path, path.stat().st_mtime)

    def source_image(self, path: pathlib.Path) -> PIL.Image.Image:
        key = self.source_key(path)
        image = self._sources.get(key)
        if image is None:
            image = masker.io.imaging.normalize_image(path)
            self._sources.put(key, image)
        return image

    def jpeg(self, path: pathlib.Path, max_side: int) -> bytes:
        key = "{}:{}".format(self.source_key(path), max_side)
        jpeg = self._jpegs.get(key)
        if jpeg is None:
            jpeg = thumbnail_jpeg(self.source_image(path), max_side)
            self._jpegs.put(key, jpeg)
        return jpeg


def image_response(cache: PreviewCache, path: pathlib.Path) -> flask.Response:
    """GET /image reply for path"""
    resolution = int(flask.request.args["resolution"])
    etag = "{}-{}".format(path.stat().st_mtime, resolution)
    if flask.request.if_none_match.contains(etag):
        return flask.Response(status=304)
    try:
        jpeg = cache.jpeg(path, resolution)
    except Exception as exc:  # noqa: BLE001 - undecodable file
        raise bad_request("invalid image: {}".format(exc))
    return flask.Response(
        jpeg, mimetype="image/jpeg",
        headers={"ETag": '"{}"'.format(etag), "Cache-Control": "private, max-age=300"},
    )


def crop_preview_payload(image, rect, max_side: int) -> dict[str, object]:
    """Preview of a rect of image; echoes the clamped rect so the client places the patch exactly"""
    width, height = image.size
    x0, y0 = max(0, math.floor(rect[0])), max(0, math.floor(rect[1]))
    x1, y1 = min(width, math.ceil(rect[2])), min(height, math.ceil(rect[3]))
    if x1 - x0 < 1 or y1 - y0 < 1:
        raise bad_request("rect_xyxy does not cover any of the image")
    jpeg = thumbnail_jpeg(image.crop((x0, y0, x1, y1)), max_side)
    return {"preview": data_url(jpeg, "image/jpeg"), "rect_xyxy": [x0, y0, x1, y1]}


# --- mask PNGs --------------------------------------------------------------


def sample_grid(length: int, scale: float) -> index_array_t:
    """Gather indices that shrink one axis by scale"""
    n = max(round(length * scale), 1)
    return np.minimum(((np.arange(n) + 0.5) / scale).astype(np.intp), length - 1)


def cap_binary(binary: binary_mask_t, max_side: int) -> binary_mask_t:
    """binary center-sampled down so its longest side fits max_side."""
    h, w = binary.shape
    if max(h, w) <= max_side:
        return binary
    scale = max_side / max(h, w)
    return binary[np.ix_(sample_grid(h, scale), sample_grid(w, scale))]


def mask_png_bytes(binary: binary_mask_t) -> bytes:
    """1-bit white-on-transparent PNG of a binary mask."""
    buf = io.BytesIO()
    PIL.Image.fromarray(binary != 0).save(buf, format="PNG", transparency=0)
    return buf.getvalue()


def run_app(app, name: str, host: str, port: int) -> None:
    print("{} web UI on http://{}:{}  (override with --port <n>)".format(name, host, port))
    app.run(host=host, port=port)
