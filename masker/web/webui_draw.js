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

/* Binary-mask drawing primitives for the masker page.
 *
 * paint/erase + undo mechanics. coords are in source pixels space. edits run
 * on crop isolated at begin()
 *
 * onChange() invoked when the op list or history changes. A rect commits as
 * a 4-point polygon op.
 */
'use strict';

function createMaskDraw(maxSide, onChange) {
  let tool = 'brush', erase = false, diameter = 16;  // diameter in image px
  let outline = false;
  let rect = null, base = null, color = null, scale = 1;
  let ops = [], undone = [];
  let shape = null;   // in-progress {kind: tool, erase, points, diameter}
  let hover = null;   // cursor [x, y] for the rubber band / brush ring
  let comp = null, cctx = null;  // rect-sized composite in `color`
  let cs = 1;         // composite px per image px

  const r2 = (v) => Math.round(v * 100) / 100;
  const nearFirst = (x, y) => shape.points.length >= 3
    && Math.hypot(x - shape.points[0][0], y - shape.points[0][1]) < 8 / scale;

  function trace(c, p) {
    c.beginPath();
    c.moveTo(p[0][0], p[0][1]);
    for (const q of p.slice(1)) c.lineTo(q[0], q[1]);
  }

  function paintOp(op) {
    const p = op.points;
    cctx.globalCompositeOperation = op.erase ? 'destination-out' : 'source-over';
    if (p.length === 1) {
      cctx.beginPath();
      cctx.arc(p[0][0], p[0][1], op.diameter / 2, 0, 2 * Math.PI);
      cctx.fill();
      return;
    }
    trace(cctx, p);
    if (op.kind === 'polygon') cctx.fill('evenodd');  // the server's fill rule
    else { cctx.lineWidth = op.diameter; cctx.stroke(); }
  }

  function replay() {
    cctx.reset();
    cctx.setTransform(cs, 0, 0, cs, -rect[0] * cs, -rect[1] * cs);
    cctx.fillStyle = cctx.strokeStyle = color;
    cctx.lineCap = cctx.lineJoin = 'round';
    if (base) {
      const r = base.rect;
      cctx.drawImage(base.img, r[0], r[1], r[2] - r[0], r[3] - r[1]);
      cctx.globalCompositeOperation = 'source-in';
      cctx.fillRect(rect[0], rect[1], comp.width / cs, comp.height / cs);
    }
    for (const op of ops) paintOp(op);
  }

  function push(op) {
    ops.push(op);
    undone.length = 0;
    paintOp(op);
    onChange();
  }

  function commit(s, kind, points) {
    const op = { kind: kind, erase: s.erase,
                 points: points.map((p) => [r2(p[0]), r2(p[1])]) };
    if (kind === 'stroke') op.diameter = r2(s.diameter);
    push(op);
  }

  function finishShape() {
    if (!shape || shape.kind !== 'polygon') return;
    const s = shape;
    shape = null;
    if (s.points.length >= 3) commit(s, 'polygon', s.points);  // else silent discard
  }

  function cancelShape() {
    if (!shape) return false;
    shape = null;
    replay();
    return true;
  }

  return {
    begin(spec) {
      ({ rect, base, color, scale } = spec);
      ops = []; undone = [];
      shape = null; hover = null;
      const w = rect[2] - rect[0], h = rect[3] - rect[1];
      cs = Math.min(1, maxSide / Math.max(w, h));
      comp = document.createElement('canvas');
      comp.width = Math.round(w * cs); comp.height = Math.round(h * cs);
      cctx = comp.getContext('2d');
      replay();
      onChange();
    },
    end() {
      rect = null; base = null;
      ops = []; undone = [];
      shape = null; hover = null;
      comp = null; cctx = null;
      onChange();
    },
    setBase(b) { base = b; replay(); },
    setTool(t) { if (t !== tool) cancelShape(); tool = t; },
    setErase(b) { erase = b; },
    setDiameter(screenPx) { diameter = screenPx / scale; },
    setOutline(b) { outline = b; },

    pointerDown(x, y) {
      if (tool === 'polygon' && shape) {
        if (nearFirst(x, y)) finishShape();  // close, no new vertex
        else shape.points.push([x, y]);
        return;
      }
      shape = { kind: tool, erase: erase, points: [[x, y]], diameter: diameter };
      if (tool === 'brush') paintOp(shape);
    },
    pointerMove(x, y) {
      hover = [x, y];
      if (!shape || shape.kind === 'polygon') return;
      const p = shape.points;
      if (shape.kind !== 'brush') { p[1] = hover; return; }
      const last = p[p.length - 1];
      if (Math.hypot(x - last[0], y - last[1]) >= 1 / scale) {  // decimate below one screen px
        p.push(hover);
        paintOp({ ...shape, points: [last, hover] });
      }
    },
    pointerUp(x, y) {
      if (!shape || shape.kind === 'polygon') return;
      const s = shape, p = s.points, [x0, y0] = p[0];
      shape = null;
      if (s.kind === 'rect') {  // corners normalized, clockwise from top-left
        const xa = Math.min(x0, x), xb = Math.max(x0, x);
        const ya = Math.min(y0, y), yb = Math.max(y0, y);
        if (xa < xb && ya < yb) commit(s, 'polygon', [[xa, ya], [xb, ya], [xb, yb], [xa, yb]]);
      } else if (s.kind === 'line') {
        commit(s, 'stroke', x === x0 && y === y0 ? [p[0]] : [p[0], [x, y]]);
      } else {
        commit(s, 'stroke', p);
      }
    },
    pointerOut() { hover = null; },

    finishShape: finishShape,
    cancelShape: cancelShape,
    undo() {
      if (!ops.length) return;
      undone.push(ops.pop());
      replay();
      onChange();
    },
    redo() {
      if (!undone.length) return;
      const op = undone.pop();
      ops.push(op);
      paintOp(op);
      onChange();
    },
    canUndo() { return ops.length > 0; },
    canRedo() { return undone.length > 0; },
    ops() { return ops; },

    drawPreview(ctx) {
      const lw = 1 / scale;
      const w = rect[2] - rect[0], h = rect[3] - rect[1];
      if (outline) drawOutline(ctx, 2, color, comp, rect[0], rect[1], w, h);
      else drawAlpha(ctx, 0.55, comp, rect[0], rect[1], w, h);
      dashedRect(ctx, scale, rect[0], rect[1], w, h);
      if (shape) {
        const p = shape.points;
        if (shape.kind === 'polygon') {
          ctx.lineWidth = lw;
          ctx.strokeStyle = ctx.fillStyle = '#fff';
          trace(ctx, hover ? p.concat([hover]) : p);
          ctx.stroke();
          const closing = hover && nearFirst(hover[0], hover[1]);
          ctx.beginPath();
          ctx.arc(p[0][0], p[0][1], (closing ? 6 : 3) * lw, 0, 2 * Math.PI);
          if (closing) ctx.stroke(); else ctx.fill();  // ring = click here to close
        } else if (shape.kind === 'rect' && p[1]) {
          const x = Math.min(p[0][0], p[1][0]), y = Math.min(p[0][1], p[1][1]);
          const bw = Math.abs(p[1][0] - p[0][0]), bh = Math.abs(p[1][1] - p[0][1]);
          ctx.globalAlpha = 0.3;
          ctx.fillStyle = color;
          ctx.fillRect(x, y, bw, bh);
          ctx.globalAlpha = 1;
          dashedRect(ctx, scale, x, y, bw, bh);
        } else if (shape.kind === 'line' && p[1]) {
          ctx.save();
          ctx.globalAlpha = 0.4;
          ctx.strokeStyle = color;
          ctx.lineWidth = shape.diameter;
          ctx.lineCap = 'round';
          trace(ctx, p);
          ctx.stroke();
          ctx.restore();
        }
      }
      if (hover && (tool === 'brush' || tool === 'line')) {
        ctx.strokeStyle = '#fff';
        ctx.lineWidth = lw;
        ctx.beginPath();
        ctx.arc(hover[0], hover[1], diameter / 2, 0, 2 * Math.PI);
        ctx.stroke();
        ctx.strokeStyle = '#000';
        ctx.beginPath();
        ctx.arc(hover[0], hover[1], diameter / 2 + lw, 0, 2 * Math.PI);
        ctx.stroke();
      }
    },
  };
}
