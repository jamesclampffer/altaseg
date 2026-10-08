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

"""Fixtures shared across the masker test suite.

The session-scoped chromium fixture (one Chromium for every browser suite)
lives in tests/conftest.py.
"""
from __future__ import annotations

import contextlib

import PIL.Image
import pytest

import masker_support
import masker.execution.model
import masker.web
import masker.web.evalweb


@pytest.fixture
def client(tmp_path):
    """Flask test client over the web app with the stub masker."""
    return masker_support.make_client(tmp_path, masker_support.StubMaskGenerator())


@pytest.fixture
def bench_client(tmp_path):
    """Flask test client over the eval dashboard on the synthetic benchmark."""
    masker_support.build_benchmark(tmp_path)
    return masker.web.evalweb.create_app(tmp_path).test_client()


@pytest.fixture
def image_path(tmp_path):
    """A 64x48 solid PNG; the default single test image."""
    path = tmp_path / "img.png"
    PIL.Image.new("RGB", (64, 48), color=(30, 90, 30)).save(path)
    return path


@pytest.fixture(autouse=True, scope="session")
def single_sam3_instance():
    """make sure weights aren't loaded twice due to vram overhead"""
    real = masker.execution.model.build_masker

    def build(**kwargs):
        masker_support.unload_sam3()
        return real(**kwargs)

    masker.execution.model.build_masker = build
    yield
    masker.execution.model.build_masker = real


@pytest.fixture
def serve_app():
    """serve_app(generator=None, **create_app kwargs) -> base URL of a served
    web app; the stub generator by default."""
    with contextlib.ExitStack() as stack:
        def start(generator=None, **kwargs):
            app = masker.web.create_app(
                generator=generator or masker_support.StubMaskGenerator(), **kwargs)
            return stack.enter_context(masker_support.serve(app))
        yield start


@pytest.fixture
def served_app(serve_app):
    """The web app with the stub generator for browser-driven tests."""
    return serve_app()


@pytest.fixture
def page(chromium):
    """fail on chromium console or page error"""
    errors = []
    p = chromium.new_page(viewport={"width": 1400, "height": 900})

    def on_console(msg):
        # external resources (fonts, favicon) may be unreachable offline;
        # only script errors fail the test
        if msg.type == "error" and "Failed to load resource" not in msg.text:
            errors.append(msg.text)

    p.on("console", on_console)
    p.on("pageerror", lambda e: errors.append(str(e)))
    yield p
    p.close()
    assert errors == []


@pytest.fixture
def sam3_masker():
    """singleton"""
    return masker_support.shared_sam3()
