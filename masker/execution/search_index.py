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

"""Per-image search index files: track completed prompt searches"""

from __future__ import annotations

import hashlib
import json
import pathlib

import masker.common.util


DIRNAME = "search_index"
SCHEMA_VERSION = 3


def index_path(image_path: pathlib.Path) -> pathlib.Path:
    return image_path.parent / DIRNAME / (image_path.name + ".json")


def sha256_file(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_index(image_path: pathlib.Path) -> dict | None:
    """Parsed index file, or None when missing or an older schema."""
    try:
        data = json.loads(index_path(image_path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    if data["version"] != SCHEMA_VERSION:
        return None
    return data


def image_status(index: dict, image_path: pathlib.Path) -> str:
    """'valid' | 'revalidated' | 'invalid' for the index file's image fingerprint."""
    stat = image_path.stat()
    image = index["image"]
    if stat.st_size == image["file_size"] and stat.st_mtime == image["mtime"]:
        return "valid"
    if sha256_file(image_path) == image["sha256"]:
        return "revalidated"
    return "invalid"


def load_with_status(image_path: pathlib.Path) -> tuple[dict | None, str]:
    """(index, status); a scan never writes, revalidation persists on the next upsert"""
    index = load_index(image_path)
    if index is None:
        return None, "missing"
    status = image_status(index, image_path)
    if status == "revalidated":
        stat = image_path.stat()
        index["image"]["file_size"] = stat.st_size
        index["image"]["mtime"] = stat.st_mtime
    return index, status


def upsert_searches(
    image_path: pathlib.Path,
    records: list[dict],
    *,
    width: int | None = None,
    height: int | None = None,
) -> dict:
    # update if not insert, keyed by prompt
    index, status = load_with_status(image_path)
    if status in ("missing", "invalid"):
        stat = image_path.stat()
        index = {
            "version": SCHEMA_VERSION,
            "image": {
                "file_name": image_path.name,
                "file_size": stat.st_size,
                "mtime": stat.st_mtime,
                "sha256": sha256_file(image_path),
                "width": width,
                "height": height,
            },
            "searches": [],
        }
    if width is not None:
        index["image"]["width"] = width
    if height is not None:
        index["image"]["height"] = height
    now = masker.common.util.utc_now()
    for rec in records:
        entry = dict(rec, updated_at=now)
        for i, existing in enumerate(index["searches"]):
            if existing["prompt"] == entry["prompt"]:
                index["searches"][i] = entry
                break
        else:
            index["searches"].append(entry)
    path = index_path(image_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masker.common.util.atomic_write_json(path, index)
    return index
