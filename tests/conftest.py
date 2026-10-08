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

import pytest


@pytest.fixture(scope="session")
def chromium():
    """One Chromium shared by every browser suite: a second sync_playwright
    context cannot start while another is active in the same thread."""
    playwright_api = pytest.importorskip("playwright.sync_api")
    with playwright_api.sync_playwright() as playwright:
        try:
            chromium = playwright.chromium.launch()
        except Exception as exc:  # noqa: BLE001 - browser binary not installed
            pytest.skip("chromium unavailable: {}".format(exc))
        yield chromium
        chromium.close()
