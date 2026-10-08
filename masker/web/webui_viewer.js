/*
 * Copyright 2026 Jim Clampffer. Created 2026-10-07.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/* Reusable pan/zoom canvas viewer for the tool web pages.
 *
 * Coords are normalized to absolute pixels relative to the full sized input
 * in all client <-> server.
 *
 * opts (all optional):
 *   dragMode()               -> 'pan' | 'box', sampled at mousedown.
 *                               'box' rubber-bands a dashed rectangle.
 *   onBoxDrawn([x0,y0,x1,y1]) box-mode drag released (normalized, clamped).
 *   onClick(x, y)            mouseup without meaningful movement, any mode.
 *   onHover(x, y | null)     mousemove / mouseleave.
 *   onViewChange(viewportRect) debounced ~300ms after pan/zoom settles.
 */
'use strict';

const OUTLINE_DIRS = [[1, 0], [-1, 0], [0, 1], [0, -1], [1, 1], [1, -1], [-1, 1], [-1, -1]];

function dashedRect(ctx, scale, x, y, w, h) {
  ctx.lineWidth = 1 / scale;
  ctx.strokeStyle = '#000';
  ctx.setLineDash([4 / scale, 3 / scale]);
  ctx.strokeRect(x, y, w, h);
  ctx.setLineDash([]);
}

function drawAlpha(ctx, alpha, ...args) {
  ctx.globalAlpha = alpha;
  ctx.drawImage(...args);
  ctx.globalAlpha = 1;
}

let outlineScratch = null;

// Builds the ring at screen resolution under ctx's current transform: the
// image stamped at eight offsets, the image itself punched out, then tinted.
// Touches only the image's screen rect grown by the ring width.
function drawOutline(ctx, px, color, img, x, y, w, h) {
  const t = ctx.getTransform();
  const c = outlineScratch || (outlineScratch = document.createElement('canvas'));
  if (c.width !== ctx.canvas.width || c.height !== ctx.canvas.height) {
    c.width = ctx.canvas.width; c.height = ctx.canvas.height;
  }
  const bx = Math.max(0, Math.floor(t.a * x + t.e - px));
  const by = Math.max(0, Math.floor(t.d * y + t.f - px));
  const bw = Math.min(c.width, Math.ceil(t.a * (x + w) + t.e + px)) - bx;
  const bh = Math.min(c.height, Math.ceil(t.d * (y + h) + t.f + px)) - by;
  if (bw <= 0 || bh <= 0) return;
  const g = c.getContext('2d');
  g.clearRect(bx, by, bw, bh);
  g.save();
  g.beginPath();
  g.rect(bx, by, bw, bh);
  g.clip();
  for (const [dx, dy] of OUTLINE_DIRS) {
    g.setTransform(t.a, t.b, t.c, t.d, t.e + dx * px, t.f + dy * px);
    g.drawImage(img, x, y, w, h);
  }
  g.setTransform(t);
  g.globalCompositeOperation = 'destination-out';
  g.drawImage(img, x, y, w, h);
  g.setTransform(1, 0, 0, 1, 0, 0);
  g.globalCompositeOperation = 'source-in';
  g.fillStyle = color;
  g.fillRect(bx, by, bw, bh);
  g.restore();
  ctx.save();
  ctx.setTransform(1, 0, 0, 1, 0, 0);
  ctx.drawImage(c, bx, by, bw, bh, bx, by, bw, bh);
  ctx.restore();
}

function createViewer(canvas, opts = {}) {
  const ctx = canvas.getContext('2d');
  let img = null, origW = 0, origH = 0;
  let zoom = 1;            // 1 = fit-to-canvas; > 1 = zoomed in
  let panX = 0, panY = 0;  // canvas-px position of the image origin
  let overlayFn = null;
  let patch = null;        // {img, rect}: hi-res crop drawn over the base
  let drag = null;         // {mode, x0, y0, x, y, moved} in canvas px
  let viewTimer = 0;

  const CLICK_SLOP = 4;    // canvas px of movement below which mouseup is a click
  const MAX_SCALE = 8;     // zoom-in cap: canvas px per original px

  function fitScale() {
    if (!origW || !origH) return 1;
    return Math.min(canvas.width / origW, canvas.height / origH);
  }
  function totalScale() { return fitScale() * zoom; }

  function clampPan() {
    // keep the image covering the canvas; center along axes it can't fill
    const s = totalScale();
    const w = origW * s, h = origH * s;
    panX = w <= canvas.width ? (canvas.width - w) / 2
      : Math.min(0, Math.max(canvas.width - w, panX));
    panY = h <= canvas.height ? (canvas.height - h) / 2
      : Math.min(0, Math.max(canvas.height - h, panY));
  }

  function canvasPos(evt) {
    // CSS may scale the canvas element; map client px onto the backing store
    const r = canvas.getBoundingClientRect();
    return { x: (evt.clientX - r.left) * canvas.width / r.width,
             y: (evt.clientY - r.top) * canvas.height / r.height };
  }
  function toOriginal(p) {
    const s = totalScale();
    return [(p.x - panX) / s, (p.y - panY) / s];
  }

  function viewportRect() {
    const s = totalScale();
    return [
      Math.max(0, -panX / s),
      Math.max(0, -panY / s),
      Math.min(origW, (canvas.width - panX) / s),
      Math.min(origH, (canvas.height - panY) / s),
    ];
  }

  function rectsIntersect(a, b) {
    return a[0] < b[2] && b[0] < a[2] && a[1] < b[3] && b[1] < a[3];
  }

  function redraw() {
    ctx.setTransform(1, 0, 0, 1, 0, 0);
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if (!img) return;
    const s = totalScale();
    ctx.setTransform(s, 0, 0, s, panX, panY);
    ctx.drawImage(img, 0, 0, origW, origH);
    if (patch) {
      const [x0, y0, x1, y1] = patch.rect;
      ctx.drawImage(patch.img, x0, y0, x1 - x0, y1 - y0);
    }
    if (overlayFn) overlayFn(ctx, viewer);
    if (drag && drag.mode === 'box' && drag.moved) {
      const [ax, ay] = toOriginal({ x: drag.x0, y: drag.y0 });
      const [bx, by] = toOriginal({ x: drag.x, y: drag.y });
      dashedRect(ctx, s, ax, ay, bx - ax, by - ay);
    }
    ctx.setTransform(1, 0, 0, 1, 0, 0);
  }

  function viewChanged() {
    clampPan();
    // a hi-res patch is only valid where it was fetched; drop it once the
    // view leaves it entirely (a fresh one arrives via onViewChange)
    if (patch && !rectsIntersect(viewportRect(), patch.rect)) patch = null;
    redraw();
    if (opts.onViewChange) {
      clearTimeout(viewTimer);
      viewTimer = setTimeout(() => opts.onViewChange(viewportRect()), 300);
    }
  }

  function maxZoom() { return Math.max(1, MAX_SCALE / fitScale()); }

  // zoom to z (clamped to [1, maxZoom]) keeping the image point under
  // canvas point (cx, cy) in place
  function zoomAbout(z, cx, cy) {
    const anchor = toOriginal({ x: cx, y: cy });
    zoom = Math.min(maxZoom(), Math.max(1, z));
    panX = cx - anchor[0] * totalScale();
    panY = cy - anchor[1] * totalScale();
    viewChanged();
  }

  canvas.addEventListener('wheel', (evt) => {
    if (!img) return;
    evt.preventDefault();
    const p = canvasPos(evt);
    zoomAbout(zoom * Math.exp(-evt.deltaY * 0.0015), p.x, p.y);
  }, { passive: false });

  canvas.addEventListener('mousedown', (evt) => {
    if (!img || evt.button !== 0) return;
    const p = canvasPos(evt);
    const mode = opts.dragMode ? opts.dragMode() : 'pan';
    drag ={ mode: mode, x0: p.x, y0: p.y, x: p.x, y: p.y, moved: false };
  });

  canvas.addEventListener('mousemove', (evt) => {
    if (!img) return;
    const p = canvasPos(evt);
    if (drag) {
      if (Math.abs(p.x - drag.x0) + Math.abs(p.y - drag.y0) >= CLICK_SLOP) {
        drag.moved = true;
      }
      if (drag.mode === 'pan') {
        panX += p.x - drag.x;
        panY += p.y - drag.y;
        drag.x = p.x; drag.y = p.y;
        viewChanged();
      } else {
        drag.x = p.x; drag.y = p.y;
        redraw();
      }
      return;
    }
    if (opts.onHover) {
      const [x, y] = toOriginal(p);
      opts.onHover(x, y);
    }
  });

  canvas.addEventListener('mouseup', (evt) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    const p = canvasPos(evt);
    if (!d.moved) {
      redraw();
      if (opts.onClick) {
        const [x, y] = toOriginal(p);
        opts.onClick(x, y);
      }
      return;
    }
    if (d.mode === 'box') {
      redraw();  // drop the rubber band
      if (opts.onBoxDrawn) {
        const [ax, ay] = toOriginal({ x: d.x0, y: d.y0 });
        const [bx, by] = toOriginal(p);
        opts.onBoxDrawn([
          Math.max(0, Math.min(ax, bx)),
          Math.max(0, Math.min(ay, by)),
          Math.min(origW, Math.max(ax, bx)),
          Math.min(origH, Math.max(ay, by)),
        ]);
      }
    }
  });

  canvas.addEventListener('mouseleave', () => {
    if (drag) { drag = null; redraw(); }  // cancel: no click/box fires
    if (opts.onHover) opts.onHover(null);
  });

  const viewer = {
    get zoom() { return zoom; },
    get scale() { return totalScale(); },
    get maxZoom() { return maxZoom(); },
    get patch() { return patch; },
    setImage(imgElement, originalW, originalH) {
      img = imgElement;
      origW = originalW;
      origH = originalH;
      patch = null;
      clearTimeout(viewTimer);
      canvas.width = imgElement.naturalWidth;
      canvas.height = imgElement.naturalHeight;
      viewer.resetView();
    },
    setOverlay(drawFn) { overlayFn = drawFn; },
    setHiResPatch(imgElement, rectXyxy) {
      patch = { img: imgElement, rect: rectXyxy };
      redraw();
    },
    redraw: redraw,
    resetView() {
      zoom = 1;
      panX = 0; panY = 0;
      clampPan();
      redraw();
    },
    viewportRect: viewportRect,
    // Backing-store size from the page (setImage defaults it to the bitmap
    // size). Keeps zoom; re-clamps the pan.
    resize(w, h) {
      canvas.width = Math.max(1, Math.round(w));
      canvas.height = Math.max(1, Math.round(h));
      if (img) { clampPan(); redraw(); }
    },
    // Programmatic zoom about the canvas center, same clamps as the wheel.
    zoomTo(z) { if (img) zoomAbout(z, canvas.width / 2, canvas.height / 2); },
    zoomBy(factor) { viewer.zoomTo(zoom * factor); },
    // Original-image px -> backing-store canvas px.
    imageToCanvas(x, y) {
      const s = totalScale();
      return [x * s + panX, y * s + panY];
    },
    eventToImage(evt) { return toOriginal(canvasPos(evt)); },
  };
  return viewer;
}
