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

"""webui_draw.js in headless Chromium: op geometry (polygon, rect, line and
brush commits, cancelShape, undo/redo) and canvas output (composite paint,
erase, base-mask blit, downscale cap, outline), asserted through md.ops()
and drawPreview onto a probe canvas via getImageData.
"""
from __future__ import annotations

import pathlib

import pytest

pytest.importorskip("playwright.sync_api")

WEB = pathlib.Path(__file__).resolve().parents[2] / "masker" / "web"
VIEWER_JS = WEB / "webui_viewer.js"  # overlay primitives the draw library calls
DRAW_JS = WEB / "webui_draw.js"

# probe canvas + px(x, y) sampler; start(maxSide) opens a fresh draw on the
# 200x200 frame in red, base(w) is a solid white w x w canvas
SETUP = """
() => {
  const cv = document.createElement('canvas');
  cv.width = 200; cv.height = 200;
  document.body.appendChild(cv);
  window.ctx = cv.getContext('2d', { willReadFrequently: true });
  window.px = (x, y) => Array.from(ctx.getImageData(x, y, 1, 1).data);
  window.render = () => {
    ctx.clearRect(0, 0, 200, 200);
    md.drawPreview(ctx);
  };
  window.start = (maxSide) => {
    window.md = createMaskDraw(maxSide || 2016, () => {});
    md.begin({ rect: [0, 0, 200, 200], base: null, color: '#f00', scale: 1 });
  };
  window.base = (w) => {
    const b = document.createElement('canvas');
    b.width = w; b.height = w;
    const g = b.getContext('2d');
    g.fillStyle = '#fff'; g.fillRect(0, 0, w, w);
    return b;
  };
}
"""


@pytest.fixture
def draw_page(page):
    page.set_content("<body></body>")
    page.add_script_tag(path=str(VIEWER_JS))
    page.add_script_tag(path=str(DRAW_JS))
    page.evaluate(SETUP)
    return page


def painted(rgba):
    # red tint at 0.55 alpha; antialiased edges stay well below
    return rgba[3] > 100 and rgba[0] > rgba[1]


def test_polygon_fill_erase_undo_redo(draw_page):
    p = draw_page
    p.evaluate("""() => {
      start();
      md.setTool('polygon');
      md.pointerDown(20, 20); md.pointerDown(120, 20); md.pointerDown(120, 120);
      md.finishShape();
      render();
    }""")
    assert painted(p.evaluate("px(100, 60)"))       # inside the triangle
    assert p.evaluate("px(40, 100)")[3] == 0        # outside

    p.evaluate("""() => {
      md.setTool('brush'); md.setErase(true); md.setDiameter(20);
      md.pointerDown(100, 60); md.pointerUp(100, 60);
      render();
    }""")
    assert p.evaluate("px(100, 60)")[3] == 0        # erased hole
    assert painted(p.evaluate("px(115, 30)"))       # rest of the triangle intact

    p.evaluate("() => { md.undo(); render(); }")
    assert painted(p.evaluate("px(100, 60)"))       # undo repaints from replay
    p.evaluate("() => { md.redo(); render(); }")
    assert p.evaluate("px(100, 60)")[3] == 0


def test_base_mask_blit_swap_and_erase(draw_page):
    p = draw_page
    p.evaluate("""() => {
      window.md = createMaskDraw(2016, () => {});
      md.begin({ rect: [0, 0, 200, 200], base: { img: base(50), rect: [50, 50, 100, 100] },
                 color: '#f00', scale: 1 });
      render();
    }""")
    assert painted(p.evaluate("px(75, 75)"))        # base blitted at its rect
    assert p.evaluate("px(45, 75)")[3] == 0         # nothing outside the rect

    p.evaluate("() => { md.setBase(null); render(); }")
    assert p.evaluate("px(75, 75)")[3] == 0         # base dropped
    p.evaluate("() => { md.setBase({ img: base(50), rect: [50, 50, 100, 100] }); render(); }")
    assert painted(p.evaluate("px(75, 75)"))        # swapped back in

    p.evaluate("""() => {
      md.setTool('brush'); md.setErase(true); md.setDiameter(16);
      md.pointerDown(75, 75); md.pointerUp(75, 75);
      render();
    }""")
    assert p.evaluate("px(75, 75)")[3] == 0         # erase cuts into the base
    assert painted(p.evaluate("px(54, 54)"))        # base corner survives


def test_composite_cap_downscales_preview_only(draw_page):
    p = draw_page
    p.evaluate("""() => {
      start(50);
      md.setTool('polygon');
      md.pointerDown(40, 40); md.pointerDown(160, 40);
      md.pointerDown(160, 160); md.pointerDown(40, 160);
      md.finishShape();
      render();
    }""")
    assert painted(p.evaluate("px(100, 100)"))      # capped composite stretches back
    assert p.evaluate("px(20, 20)")[3] == 0
    # wire geometry is untouched by the preview cap
    assert p.evaluate("md.ops()[0].points") == [[40, 40], [160, 40], [160, 160], [40, 160]]


def test_outline_rings_the_composite(draw_page):
    p = draw_page
    p.evaluate("""() => {
      start();
      md.setTool('rect');
      md.pointerDown(50, 50); md.pointerMove(150, 150); md.pointerUp(150, 150);
      md.setOutline(true);
      render();
    }""")
    ring = p.evaluate("px(48, 100)")
    assert ring[3] > 100 and ring[0] > ring[1]   # 2px ring just outside the edge
    assert p.evaluate("px(100, 100)")[3] == 0    # interior is clear
    assert p.evaluate("px(40, 100)")[3] == 0     # beyond the ring
    p.evaluate("() => { md.setOutline(false); render(); }")
    assert painted(p.evaluate("px(100, 100)"))   # fill is back


def test_line_commit_and_cancel_shape(draw_page):
    p = draw_page
    p.evaluate("""() => {
      start();
      md.setTool('line'); md.setDiameter(10);
      md.pointerDown(20, 100); md.pointerMove(100, 40); md.pointerUp(180, 100);
      md.pointerDown(50, 160); md.pointerUp(50, 160);
      render();
    }""")
    # a drag commits start -> release (the mid-drag point is dropped); a click
    # commits one point
    assert p.evaluate("md.ops()") == [
        {"kind": "stroke", "erase": False, "points": [[20, 100], [180, 100]], "diameter": 10},
        {"kind": "stroke", "erase": False, "points": [[50, 160]], "diameter": 10},
    ]
    assert painted(p.evaluate("px(100, 100)"))      # along the line
    assert p.evaluate("px(60, 70)")[3] == 0         # on the mid-drag path
    assert painted(p.evaluate("px(50, 160)"))       # the click dot

    # an open brush stroke paints the composite as it goes; cancel replays
    # the committed ops without it
    p.evaluate("""() => {
      md.setTool('brush');
      md.pointerDown(150, 40); md.pointerMove(170, 40);
      render();
    }""")
    assert painted(p.evaluate("px(160, 40)"))
    assert p.evaluate("md.cancelShape()") is True
    assert p.evaluate("md.cancelShape()") is False  # nothing left to drop
    p.evaluate("render()")
    assert p.evaluate("px(160, 40)")[3] == 0
    assert len(p.evaluate("md.ops()")) == 2         # committed ops kept
    assert painted(p.evaluate("px(100, 100)"))
