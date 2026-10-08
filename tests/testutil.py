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

"""Shared test data path and skip helper importable from any test directory."""

from __future__ import annotations

import pathlib

import pytest

DATA_DIR = pathlib.Path(__file__).parent / "data"


def image_or_skip(filename: str) -> pathlib.Path:
    path = DATA_DIR / filename
    if not path.exists():
        pytest.skip("test image not present: {}".format(filename))
    return path
