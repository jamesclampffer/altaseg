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

"""basic utils"""

from __future__ import annotations

import datetime
import functools
import json
import os
import pathlib
import tempfile


def atomic_write_json(path: pathlib.Path, obj: dict[str, object]) -> None:
    fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=lambda v: v.to_json()))
    os.replace(tmp, path)


def utc_now() -> str:
    """timestamp e.g. 2026-01-31T12:34:56Z."""
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


@functools.lru_cache(maxsize=None)
def load_table(path: pathlib.Path) -> dict[str, object]:
    return json.loads(path.read_text())
