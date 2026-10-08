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

"""masker.execution.search_index: index-file layout and upsert,
image-fingerprint statuses; plus the
/record_searches -> /load round trip through the web app."""
from __future__ import annotations

import os

import PIL.Image

import masker_support
import masker.execution.search_index


def _sweep_record(prompt="pallet", **over):
    """The record a completed web sweep posts, prompt already normalized."""
    rec = {"prompt": prompt, "completed": True, "instances": 3,
           "params": masker_support.sweep_defaults()}
    rec.update(over)
    return rec


def _prompts(index):
    return [r["prompt"] for r in index["searches"]]


def test_upsert_creates_index_file_per_file_name(image_path, tmp_path):
    index = masker.execution.search_index.upsert_searches(
        image_path, [_sweep_record()], width=64, height=48)
    assert (tmp_path / "search_index" / "img.png.json").is_file()
    assert index["image"]["file_name"] == "img.png"
    assert index["image"]["width"] == 64
    assert _prompts(index) == ["pallet"]

    # same stem, other suffix: a separate index file
    other = tmp_path / "img.jpg"
    other.write_bytes(b"jpg bytes")
    masker.execution.search_index.upsert_searches(other, [_sweep_record("crate")])
    assert (tmp_path / "search_index" / "img.jpg.json").is_file()
    assert _prompts(masker.execution.search_index.load_index(image_path)) == ["pallet"]
    assert _prompts(masker.execution.search_index.load_index(other)) == ["crate"]


def test_upsert_replaces_same_key_appends_new(image_path):
    masker.execution.search_index.upsert_searches(
        image_path, [_sweep_record(completed=False)])
    index = masker.execution.search_index.upsert_searches(image_path, [_sweep_record()])
    assert len(index["searches"]) == 1
    assert index["searches"][0]["completed"] is True

    index = masker.execution.search_index.upsert_searches(
        image_path, [_sweep_record("crate"), _sweep_record()])
    assert _prompts(index) == ["pallet", "crate"]


def test_statuses(image_path):
    assert masker.execution.search_index.load_with_status(image_path) == (None, "missing")

    masker.execution.search_index.upsert_searches(image_path, [_sweep_record()])
    _, status = masker.execution.search_index.load_with_status(image_path)
    assert status == "valid"

    # mtime churn without a content change revalidates the fingerprint, in
    # memory only: a scan never writes, so a rescan revalidates again
    os.utime(image_path, (1234567890, 1234567890))
    index, status = masker.execution.search_index.load_with_status(image_path)
    assert status == "revalidated"
    assert index["image"]["mtime"] == image_path.stat().st_mtime
    _, status = masker.execution.search_index.load_with_status(image_path)
    assert status == "revalidated"

    # the next upsert persists the revalidated fingerprint
    masker.execution.search_index.upsert_searches(image_path, [_sweep_record()])
    _, status = masker.execution.search_index.load_with_status(image_path)
    assert status == "valid"

    # content change invalidates and the next upsert resets the records
    PIL.Image.new("RGB", (64, 48), color=(200, 0, 0)).save(image_path)
    index, status = masker.execution.search_index.load_with_status(image_path)
    assert status == "invalid"
    assert _prompts(index) == ["pallet"]  # records kept until overwritten
    index = masker.execution.search_index.upsert_searches(image_path, [_sweep_record("crate")])
    assert _prompts(index) == ["crate"]


def test_record_searches_roundtrip(client, image_path):
    """A posted sweep record is echoed and /load reports it valid."""
    sweep = _sweep_record()
    code, body = masker_support.post_json(
        client, "/record_searches",
        {"image_path": str(image_path), "width": 64, "height": 48, "searches": [sweep]},
    )
    assert code == 200
    assert body["index"]["image"]["width"] == 64
    (rec,) = body["index"]["searches"]
    assert rec["prompt"] == "pallet"
    assert rec["params"] == sweep["params"] and rec["instances"] == 3

    code, body = masker_support.post_json(client, "/load", {"image_path": str(image_path)})
    assert code == 200 and body["search_index_status"] == "valid"
    assert len(body["search_index"]["searches"]) == 1
