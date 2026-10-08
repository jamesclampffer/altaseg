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

"""COCO dataset operations: locate/load/edit/create annotation files"""

from __future__ import annotations

import json
import pathlib
import threading

import masker.annotations.instance_mask
import masker.common.util

type coco_t = dict[str, list[dict[str, object]]]


class CocoLayout:
    KEYS = ("images", "annotations", "categories")
    # File name of the COCO file a save creates when the directory has none.
    ANNOTATIONS_NAME = "coco_annotations.json"
    # Bbox drift allowed when verifying a client-held annotation reference
    BBOX_TOL = 1.0


def is_coco(data: object) -> bool:
    return isinstance(data, dict) and all(key in data for key in CocoLayout.KEYS)


def find_coco_file(directory: pathlib.Path) -> pathlib.Path | None:
    """The first COCO annotation JSON in directory (sorted by name)."""
    for path in sorted(directory.glob("*.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        if is_coco(data):
            return path
    return None


def load_coco(path: pathlib.Path) -> coco_t:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not is_coco(data):
        raise ValueError(
            "{} malformed coco (needs top-level {})".format(path, ', '.join(CocoLayout.KEYS))
        )
    for ann in data["annotations"]:
        seg = ann.get("segmentation")
        if isinstance(seg, dict) and isinstance(seg.get("counts"), str):
            ann["segmentation"] = masker.annotations.instance_mask.InstanceMask.from_coco(ann)
    return data


def save_coco(dataset: coco_t, path: pathlib.Path) -> None:
    masker.common.util.atomic_write_json(path, dataset)


def empty_coco() -> coco_t:
    return {"images": [], "annotations": [], "categories": []}


def annotations_for_image(dataset: coco_t, file_name: str) -> list[dict]:
    image_ids = {img["id"] for img in dataset["images"]
                 if pathlib.Path(img["file_name"]).name == file_name}
    return [a for a in dataset["annotations"] if a["image_id"] in image_ids]


def ensure_image(dataset: coco_t, file_name: str, width: int, height: int) -> tuple[coco_t, int]:
    """Return (dataset, image_id) for file_name, appending an entry if missing."""
    for img in dataset["images"]:
        if pathlib.Path(img["file_name"]).name == file_name:
            return dataset, img["id"]
    image_id = max((img["id"] for img in dataset["images"]), default=0) + 1
    entry = {"id": image_id, "file_name": file_name, "width": width, "height": height}
    return {**dataset, "images": dataset["images"] + [entry]}, image_id


def append_annotations(dataset: coco_t, annotations: list[dict]) -> coco_t:
    """Append annotations with ids continuing from the current max."""
    next_id = max((a["id"] for a in dataset["annotations"]), default=0) + 1
    added = [{**a, "id": next_id + i} for i, a in enumerate(annotations)]
    return {**dataset, "annotations": dataset["annotations"] + added}


def remove_annotation(dataset: coco_t, annotation_id: int) -> coco_t:
    return {**dataset, "annotations": [a for a in dataset["annotations"] if a["id"] != annotation_id]}


def replace_annotation(dataset: coco_t, annotation: dict) -> coco_t:
    annotations = [annotation if a["id"] == annotation["id"] else a
                   for a in dataset["annotations"]]
    return {**dataset, "annotations": annotations}


def stale_annotation(dataset: coco_t, annotation_id: int, image_name: str, bbox_xyxy: list[float]) -> bool:
    """Whether a client-held annotation reference no longer matches dataset."""
    target = next((a for a in dataset["annotations"] if a["id"] == annotation_id), None)
    if target is None:
        return True
    img = next(i for i in dataset["images"] if i["id"] == target["image_id"])
    if pathlib.Path(img["file_name"]).name != image_name:
        return True
    x, y, w, h = target["bbox"]
    drift = max(abs(a - b) for a, b in zip(bbox_xyxy, [x, y, x + w, y + h]))
    return drift > CocoLayout.BBOX_TOL


class CocoStore:
    """One COCO file per image directory, cached against its (mtime, size)."""

    __slots__ = '_path', '_sig', '_data', 'lock'
    _path: pathlib.Path | None
    _sig: tuple[int, int] | None
    _data: coco_t
    # held by callers across load->modify->store
    lock: threading.RLock

    def __init__(self) -> None:
        self._path, self._sig, self._data = None, None, empty_coco()
        self.lock = threading.RLock()

    @staticmethod
    def _signature(path: pathlib.Path) -> tuple[int, int]:
        st = path.stat()
        return (st.st_mtime_ns, st.st_size)

    def load(self, path: pathlib.Path) -> coco_t:
        with self.lock:
            sig = self._signature(path)
            if self._path != path or self._sig != sig:
                self._path, self._sig, self._data = path, sig, load_coco(path)
            return self._data

    def store(self, dataset: coco_t, path: pathlib.Path) -> None:
        with self.lock:
            save_coco(dataset, path)
            self._path, self._sig, self._data = path, self._signature(path), dataset
