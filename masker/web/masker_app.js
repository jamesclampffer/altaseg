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

    // Masker workspace app. Runs inside masker_page.html with CFG and the
    // shared viewer (webui_viewer.js) already loaded. Every canvas coordinate
    // crossing the viewer API is in original-image pixels.
    const elById = new Map();  // id -> element, looked up once
    const $ = (id) => elById.get(id)
      || elById.set(id, document.getElementById(id)).get(id);
    const MODEL_SIDE = CFG.MODEL_SIDE;
    const SWEEP_DEFAULTS = CFG.SWEEP_DEFAULTS;
    // label palette: webui.css --pal-N custom properties, read once
    const rootCss = getComputedStyle(document.documentElement);
    const PALETTE = [];
    for (let v; (v = rootCss.getPropertyValue('--pal-' + PALETTE.length).trim()); ) {
      PALETTE.push(v);
    }
    // bbox stroke by the sweep pass that found the instance while the sweep
    // runs; outside PALETTE, which masks keep using by label
    const PASS_COLORS = { segment_contents: '#00e5ff', reprompt: '#ffea00', exemplar: '#ff00e5' };
    const HISTORY_KEY = 'masker.prompt_history';
    const ROW_CAP = 200;      // Instances rows rendered per class; rest counted
    const MIN_BOX = 12;       // px: smaller drawn boxes are ignored
    const ZOOMED_IN = 1.01;   // zoom epsilon over fit view
    const ZOOM_STEPS = 1000;  // zoom slider resolution; position is log-scale in zoom
    const FILMSTRIP_MAX = 12; // thumbnails in the filmstrip window, loaded image included
    const DEDUP_IOU = 0.5;    // batch anchor-dedup IoU default
    const MASK_ALPHA = 0.25;  // mask fill over the image

    const state = {
      imagePath: null, imgW: 0, imgH: 0, previewImg: null, cocoFile: null,
      saved: [],    // {annotationId, label, bboxXyxy, maskImg|null, visible}
      pending: [],  // {tmpId, label, score, bboxXyxy, maskImg, rle, visible}
      labelOrder: [],
      labelCats: [],   // /load categories with supercategory + total/in_image counts
      labelTree: [],   // nested category hierarchy (supercategory links)
      prev: null, next: null,  // neighboring image paths in the directory
      viewOnly: false,  // loaded image is a Zephyr mask: no prompts, no save
      searchIndex: null, searchIndexStatus: 'missing',  // per-image search index
    };
    const ui = {
      mode: 'text',       // text | exemplar | point | reprompt | erase
      page: 'work',       // work | review
      hideMasks: false, showPresence: false,
      solo: false,        // only the selected instance renders
      outline: false,     // masks draw as 2px rings instead of fills
      locked: false,      // manual mask edit: viewport frozen, draw tools own the canvas
      modelView: true,    // the model input float; a corner pill when collapsed
    };
    const hiddenLabels = new Set();  // per-class eye: display-only hide
    const outlineLabels = new Map();  // per-class draw mode 'ring' or 'both'; absent = fill
    let tmpCounter = 0;
    let hover = null;    // instance under the mouse (the canvas halo follows this)
    let pinned = null;   // sticky: keeps its panel row highlighted after the mouse moves on
    let selected = null; // click-selected instance (popover + bright fill)
    let pinTimer = null;
    const rowByInst = new Map();    // instance -> its persistent panel row
    const hdrByLabel = new Map();   // label -> its persistent group header
    const moreByLabel = new Map();  // label -> its "N more" row
    let busy = false;
    const collapsedLabels = new Set();  // folded Instances groups (session-only)
    let sweepCancel = false;
    let sweepAbort = null;
    let searchLog = [];  // completed-search records awaiting a successful save
    // per-image unsaved work, stashed on navigation and restored on return
    const imgStore = new Map();  // path -> {pending, stagedIds, searchLog}

    const MODES = ['text', 'exemplar', 'point', 'reprompt', 'erase'];
    const MENUS = ['exportmenu', 'maskmenu', 'touchmenu'];
    const OVERLAY_OWED_MS = 500;
    // redraws coalesce while streaming
    const STREAM_REDRAW_MS = 100;
    // draw tool -> its bar button id
    const DRAW_TOOLS = { polygon: 'dpoly', line: 'dline', brush: 'dbrush', rect: 'drect' };
    // touch-up entry: the selection popover's dropdown; [item id, tool, erase]
    const TOUCH_ITEMS = [['tbrushadd', 'brush', false], ['tbrushdel', 'brush', true],
                         ['trectadd', 'rect', false], ['trectdel', 'rect', true]];
    // touch-up tools; the draw lock offers every tool
    const TOUCHUP_TOOLS = ['brush', 'rect'];
    const LOAD_CACHE_MAX = 4;
    const PACE_ALPHA = 0.3;

    function children(node) { return [...node.childNodes].filter(n => n.tagName); }
    function el(tag, cls, text, parent) {
      const node = document.createElement(tag);
      if (cls) node.className = cls;
      if (text != null) node.textContent = text;
      if (parent) parent.appendChild(node);
      return node;
    }
    const sleep = ms => new Promise(r => setTimeout(r, ms));
    // display toggle by id or node; kind is the shown display where the
    // stylesheet's is none
    function show(target, on, kind = '') {
      (typeof target === 'string' ? $(target) : target).style.display = on ? kind : 'none';
    }
    function setText(node, s) { if (node.textContent !== s) node.textContent = s; }

    // --- toasts + status ----------------------------------------------------

    let toastTimer = 0;
    function toast(msg, ms) {
      $('toast').textContent = msg;
      show('toast', true, 'block');
      clearTimeout(toastTimer);
      toastTimer = setTimeout(() => show('toast', false), ms || 2600);
    }
    function say(msg) { $('status').textContent = msg; }
    function sayError(msg) { say(msg); toast(msg, 4000); }
    // refusal guards: true (with a status line) when the action cannot run now
    function noImageBlock() {
      if (state.imagePath) return false;
      say('Load an image first.');
      return true;
    }
    function busyBlock() {
      if (!busy) return false;
      say('Wait for the current run to finish first.');
      return true;
    }

    // First segmentation request blocks while the server builds the model;
    // poll /model_status and surface it. No-op after the first success.
    let modelReady = false;
    function startModelWatch(activeMsg) {
      if (modelReady) return () => {};
      let stopped = false, sawLoading = false;
      const stop = () => { stopped = true; clearInterval(timer); };
      const timer = setInterval(async () => {
        try {
          const { ok, json } = await getJson('/model_status');
          if (stopped || !ok) return;
          if (json.loaded) {
            modelReady = true;
            if (sawLoading) say(activeMsg);
            stop();
          } else {
            sawLoading = true;
            say('Waiting for model to load...');
          }
        } catch (err) { /* transient; keep polling */ }
      }, 500);
      return stop;
    }

    function applyPin(inst, scrollRow) {
      if (!isLive(inst)) return;  // deleted while the dwell timer ran
      pinned = inst;
      for (const [i, row] of rowByInst) row.classList.toggle('hov', i === pinned);
      if (scrollRow) scrollToRow(inst);
    }

    function scrollToRow(inst) {
      const row = rowByInst.get(inst);
      if (!row || row.offsetTop == null || !$('instances').clientHeight) return;
      $('instances').scrollTop = row.offsetTop - $('instances').clientHeight / 2;
    }

    function setHover(inst, scrollRow) {
      if (hover !== inst) { hover = inst; scheduleRedraw(); }
      // null (the mouse left a bbox) keeps the current pin; the dwell delay
      // ignores bboxes crossed on the way to the panel
      if (!inst || inst === pinned) { clearTimeout(pinTimer); pinTimer = null; return; }
      clearTimeout(pinTimer);
      pinTimer = setTimeout(() => { pinTimer = null; applyPin(inst, scrollRow); }, 150);
    }

    function instanceAt(x, y) {
      // smallest visible bbox under the cursor
      let best = null, bestArea = Infinity;
      for (const inst of allInstances()) {
        if (!shown(inst)) continue;
        const [x0, y0, x1, y1] = inst.bboxXyxy;
        if (x < x0 || x > x1 || y < y0 || y > y1) continue;
        const area = (x1 - x0) * (y1 - y0);
        if (area < bestArea) { best = inst; bestArea = area; }
      }
      return best;
    }

    // Guarded parse: a non-JSON body yields {error}.
    async function parsedJson(resp) {
      let json = null;
      try { json = await resp.json(); } catch (err) { /* non-JSON body */ }
      return json != null ? json : { error: 'HTTP error ' + resp.status };
    }

    async function postJson(url, body) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      return { ok: resp.ok, code: resp.status, json: await parsedJson(resp) };
    }

    async function getJson(url) {
      const resp = await fetch(url);
      return { ok: resp.ok, code: resp.status, json: await parsedJson(resp) };
    }
    function getPath(route, p) { return getJson(route + '?path=' + encodeURIComponent(p)); }

    // POST and unpack; a failed request reports and yields null. 409 (stale
    // annotation reference) runs onStale, then resyncs from disk.
    async function call(url, body, onStale) {
      let resp;
      try {
        resp = await postJson(url, body);
      } catch (err) {
        sayError('Error: ' + err);
        return null;
      }
      if (resp.code === 409) {
        if (onStale) onStale();
        await resyncStale(resp.json);
        return null;
      }
      if (!resp.ok) { sayError('Error: ' + resp.json.error); return null; }
      return resp.json;
    }

    // POST body to url and feed each NDJSON line of the response to onEvent.
    // Returns the error string from a non-ok response, else null at stream end.
    async function streamNdjson(url, body, onEvent, signal) {
      const resp = await fetch(url, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
        signal: signal,
      });
      if (!resp.ok) {
        // error body: JSON {error} when parseable, else the response text
        let text = null;
        try { text = await resp.text(); } catch (err) { /* unreadable */ }
        if (text == null) return (await parsedJson(resp)).error;
        try {
          const err = JSON.parse(text).error;
          if (err != null) return err;
        } catch (err) { /* non-JSON body */ }
        return text.trim() || ('HTTP error ' + resp.status);
      }
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffered = '';
      for (;;) {
        const { value, done: eof } = await reader.read();
        if (value) buffered += decoder.decode(value, { stream: true });
        const lines = buffered.split('\n');
        buffered = eof ? '' : lines.pop();
        for (const line of lines) {
          if (line.trim()) onEvent(JSON.parse(line));
        }
        if (eof) break;
      }
      return null;
    }

    // One streamed model call: model watch, coalesced redraws, focus cleared
    // and a full render at the end. Returns the failure string (non-ok
    // response, error event, or thrown error) or null; an abort under
    // sweepCancel is not a failure.
    async function runStream(url, body, onEvent, watchMsg, signal) {
      const stopModelWatch = startModelWatch(watchMsg);
      streamBegin();
      let failed = null;
      try {
        const httpError = await streamNdjson(url, body,
          (ev) => { if (ev.type === 'error') failed = ev.error; else onEvent(ev); }, signal);
        return httpError != null ? httpError : failed;
      } catch (err) {
        return signal && sweepCancel ? null : String(err);
      } finally {
        streamEnd();
        stopModelWatch();
        sweepFocus = null;
        flushRenderAll();
      }
    }

    // popover menus: one open at a time, any document click closes them
    function closeMenus() { for (const id of MENUS) show(id, false); }
    function toggleMenu(menu, e) {
      e.stopPropagation();
      const open = menu.style.display === 'block';
      closeMenus();
      show(menu, !open, 'block');
    }
    document.addEventListener('click', closeMenus);

    function showModal(id, on) { show(id, on, 'flex'); }

    function collapsible(toggleId, bodyId, label, onOpen) {
      $(toggleId).addEventListener('click', () => {
        const open = $(bodyId).style.display !== 'block';
        show(bodyId, open, 'block');
        $(toggleId).textContent = label + (open ? '^' : 'v');
        if (open && onOpen) onOpen();
      });
    }

    function tintChip(node, color) {
      node.style.color = color;
      node.style.background = color + '1f';
      node.style.border = '1px solid ' + color + '59';
    }

    // General-purpose image fetch (GET /image): a server-local path at a
    // resolution (longest side, never upscaled) becomes an <img> src URL.
    const ImageProxy = {
      url(path, resolution) {
        return '/image?path=' + encodeURIComponent(path)
          + '&resolution=' + Math.round(resolution);
      },
    };

    // the stored form of a prompt in the search index
    function normPrompt(s) { return s.trim().toLowerCase().replace(/\s+/g, ' '); }

    function labelColor(name) {
      if (!state.labelOrder.includes(name)) state.labelOrder.push(name);
      return PALETTE[state.labelOrder.indexOf(name) % PALETTE.length];
    }

    function allInstances() { return state.saved.concat(state.pending); }
    // still on the image (async work may outlive a delete)
    function isLive(inst) { return state.pending.includes(inst) || state.saved.includes(inst); }

    function labelShown(inst) { return !hiddenLabels.has(inst.label); }
    function shown(inst) { return inst.visible && !inst.pendingDelete && labelShown(inst); }
    function outlined(inst) { return ui.outline || outlineLabels.has(inst.label); }

    // Take instances matching pred off this image: pending dropped, saved
    // staged for delete (Save commits). Returns the counts.
    function dropInstances(pred) {
      const dropped = state.pending.filter(pred).length;
      state.pending = state.pending.filter(p => !pred(p));
      let staged = 0;
      for (const s of state.saved) {
        if (!s.pendingDelete && pred(s)) { s.pendingDelete = true; staged++; }
      }
      return { dropped, staged };
    }

    function liveCount(label, list = allInstances()) {
      return list.filter(i => i.label === label && !i.pendingDelete).length;
    }

    function centerInBox(inst, box) {
      const [x0, y0, x1, y1] = inst.bboxXyxy;
      let cx = (x0 + x1) / 2, cy = (y0 + y1) / 2;
      if ($('sweepcentroid').value === 'mask' && inst.centroid) [cx, cy] = inst.centroid;
      return box[0] <= cx && cx <= box[2] && box[1] <= cy && cy <= box[3];
    }

    function loadMaskImage(dataUrl, onReady, onFail) {
      if (!dataUrl) { onReady(null); return; }
      const img = new Image();
      img.onload = () => onReady(img);
      img.onerror = () => { if (onFail) onFail(); };
      img.src = dataUrl;
    }

    // Decode a mask data URL onto the instance; a failed decode marks it for
    // retry on its next select.
    let maskFailSaidAt = -Infinity;
    function decodeMaskInto(entry, dataUrl) {
      loadMaskImage(dataUrl, img => {
        if (!isLive(entry)) return;  // deleted while decoding
        entry.maskImg = img;
        markOverlayAdd(entry);
      }, () => {
        if (!isLive(entry)) return;
        entry.maskFailed = dataUrl;  // retry source for the next select
        if (performance.now() - maskFailSaidAt > 4000) {  // one toast per burst
          maskFailSaidAt = performance.now();
          toast('A mask image failed to decode - select the instance to retry.', 4000);
        }
      });
    }

    function basename(p) { return p.split('\\').pop().split('/').pop(); }
    function dirOf(p) {
      return p.slice(0, Math.max(p.lastIndexOf('/'), p.lastIndexOf('\\')));
    }
    // name inside a listing (/fsmeta, /dir_counts) as a server-side path
    function listedPath(m, name) { return m.dir + m.sep + name; }

    // --- canvas rendering ---------------------------------------------------

    // Masks composite once into a preview-resolution offscreen canvas;
    // markOverlayAdd paints new masks onto it, markOverlayDirty forces a
    // rebuild. Bboxes and tags draw per-frame at screen-constant size.
    let overlay = null, overlayFull = true, rafId = 0;
    let sweepFocus = null;  // recrop rects of the sweep batch in flight
    // pass-1 grid + per-recrop telemetry of the latest sweep; kept after
    // the sweep ends
    let sweepGrid = null;   // {rects: [[x0,y0,x1,y1]...], stats: Map(index -> recrop_stats)}
    // reprompt-by-bbox queue: refine boxes waiting or in flight
    const rq = { queued: [], inflight: [], draining: false };  // {bbox, label}
    const overlayNew = [];  // instances to paint onto the existing overlay
    let streaming = 0, overlayOwedTimer = 0;

    function scheduleRedraw() {
      if (rafId) return;
      rafId = 1;
      if (streaming) {
        setTimeout(() => { rafId = 0; viewer.redraw(); }, STREAM_REDRAW_MS);
      } else {
        requestAnimationFrame(() => { rafId = 0; viewer.redraw(); });
      }
    }
    function flushOverlayOwed() {
      overlayOwedTimer = 0;
      overlayFull = true;
      scheduleRedraw();
    }
    function markOverlayDirty() {
      lockOverlay = null;
      if (streaming) {
        if (!overlayOwedTimer) {
          overlayOwedTimer = setTimeout(flushOverlayOwed, OVERLAY_OWED_MS);
        }
        return;
      }
      overlayFull = true;
      scheduleRedraw();
    }
    function streamBegin() { streaming++; }
    function streamEnd() {
      streaming--;
      if (!streaming && overlayOwedTimer) {
        clearTimeout(overlayOwedTimer);
        flushOverlayOwed();
      }
    }
    function markOverlayAdd(inst) {
      lockOverlay = null;
      if (!overlayFull) overlayNew.push(inst);
      scheduleRedraw();
    }

    // Mask canvases (overlay composite, sprites) render at the image fitted
    // to the server's mask side, not the page preview's.
    function maskCanvasDims() {
      const scale = Math.min(1, CFG.MASK_SIDE / Math.max(state.imgW, state.imgH));
      return [Math.max(1, Math.round(state.imgW * scale)),
              Math.max(1, Math.round(state.imgH * scale))];
    }

    // colour the opaque pixels drawn so far; the fill spans (0, 0, w, h)
    function tintFill(ctx2, color, w, h) {
      ctx2.globalCompositeOperation = 'source-in';
      ctx2.fillStyle = color;
      ctx2.fillRect(0, 0, w, h);
    }

    // tinted sprite cropped to the bbox (mask-canvas coords), cached on the
    // instance; the mask PNG covers its tight rect (original px)
    function tintedSprite(inst, color, w, h) {
      if (inst.tint && inst.tintColor === color
          && inst.tintForW === w && inst.tintForH === h) return inst.tint;
      const sx = w / state.imgW, sy = h / state.imgH;
      const [x0, y0, x1, y1] = inst.bboxXyxy;
      const px = Math.max(0, Math.floor(x0 * sx) - 2);
      const py = Math.max(0, Math.floor(y0 * sy) - 2);
      const pw = Math.min(w, Math.ceil(x1 * sx) + 2) - px;
      const ph = Math.min(h, Math.ceil(y1 * sy) + 2) - py;
      const c = document.createElement('canvas');
      c.width = Math.max(1, pw); c.height = Math.max(1, ph);
      const ctx2 = c.getContext('2d');
      ctx2.translate(-px, -py);
      const [rx0, ry0, rx1, ry1] = inst.maskRect;
      ctx2.drawImage(inst.maskImg, rx0 * sx, ry0 * sy,
                     (rx1 - rx0) * sx, (ry1 - ry0) * sy);
      tintFill(ctx2, color, w, h);
      inst.tint = c; inst.tintColor = color;
      inst.tintForW = w; inst.tintForH = h;
      inst.tintX = px; inst.tintY = py;
      return c;
    }

    function paintOverlayInstance(octx, inst) {
      if (!shown(inst) || !inst.maskImg || outlineLabels.get(inst.label) === 'ring') return;
      const sprite = tintedSprite(inst, labelColor(inst.label),
                                  octx.canvas.width, octx.canvas.height);
      drawAlpha(octx, MASK_ALPHA, sprite, inst.tintX, inst.tintY);
    }

    function updateOverlay() {
      const [mw, mh] = maskCanvasDims();
      let toPaint;
      if (overlayFull || !overlay
          || overlay.width !== mw || overlay.height !== mh) {
        overlay = document.createElement('canvas');
        overlay.width = mw; overlay.height = mh;
        overlayFull = false;
        overlayNew.length = 0;
        toPaint = allInstances();
      } else if (overlayNew.length) {
        // drop anything deleted between markOverlayAdd and this repaint
        toPaint = overlayNew.splice(0).filter(isLive);
      } else {
        return;
      }
      const octx = overlay.getContext('2d');
      for (const inst of toPaint) paintOverlayInstance(octx, inst);
    }

    // Tint of one instance's mask PNG at the PNG's own resolution, for the
    // emphasize overlay; cached per (instance, color).
    const hiTints = [];
    function hiResTint(inst, color) {
      for (const t of hiTints) {
        if (t.inst === inst && t.color === color) return t.canvas;
      }
      const c = document.createElement('canvas');
      c.width = Math.max(1, inst.maskImg.width);
      c.height = Math.max(1, inst.maskImg.height);
      const x = c.getContext('2d');
      x.drawImage(inst.maskImg, 0, 0);
      tintFill(x, color, c.width, c.height);
      hiTints.push({ inst: inst, color: color, canvas: c });
      if (hiTints.length > 4) hiTints.shift();
      return c;
    }
    // an in-place mask change invalidates the instance's cached tints
    function dropTints(inst) {
      inst.tint = null;
      for (let i = hiTints.length - 1; i >= 0; i--) {
        if (hiTints[i].inst === inst) hiTints.splice(i, 1);
      }
    }

    // white halo + bright tint for the hovered/selected instance; the PNG
    // draws at native resolution over its rect
    function emphasize(ctx, v, inst) {
      const [x0, y0, x1, y1] = inst.bboxXyxy;
      const color = labelColor(inst.label);
      if (inst.maskImg) {
        const [rx0, ry0, rx1, ry1] = inst.maskRect;
        const tint = hiResTint(inst, color);
        if (outlined(inst)) drawOutline(ctx, 2, color, tint, rx0, ry0, rx1 - rx0, ry1 - ry0);
        else drawAlpha(ctx, 0.65, tint, rx0, ry0, rx1 - rx0, ry1 - ry0);
      }
      ctx.lineWidth = 5 / v.scale;
      ctx.strokeStyle = '#fff';
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      ctx.lineWidth = 3 / v.scale;
      ctx.strokeStyle = color;
      ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
    }

    // Runs inside viewer.redraw() with ctx transformed to original-image
    // coordinates; on-screen-constant stroke widths divide by v.scale.
    function drawOverlay(ctx, v) {
      if (!state.previewImg) return;
      // solo: with a selected (or edited) instance, nothing else renders;
      // outline: the per-instance loop below rings each mask itself
      const solo = ui.solo && (ui.locked ? drawHide : selected);
      if (ui.hideMasks || solo || ui.outline) {
      } else if (ui.locked) {
        if (!lockOverlay) buildLockOverlay();
        ctx.drawImage(lockOverlay, drawRect[0], drawRect[1],
                      drawRect[2] - drawRect[0], drawRect[3] - drawRect[1]);
      } else {
        updateOverlay();
        ctx.drawImage(overlay, 0, 0, state.imgW, state.imgH);
      }
      // per-recrop presence of the latest sweep (presence toggle):
      // fill runs green (high) to red (low); the model blanks a recrop whose
      // presence sigmoid is at or below the score threshold
      if (ui.showPresence && sweepGrid) {
        ctx.lineWidth = 2 / v.scale;
        ctx.font = (12 / v.scale) + 'px sans-serif';
        for (let i = 0; i < sweepGrid.rects.length; i++) {
          const [gx0, gy0, gx1, gy1] = sweepGrid.rects[i];
          const s = sweepGrid.stats.get(i);
          const p = s && s.presence != null ? s.presence : null;
          if (p != null) {
            const r = Math.round(220 * (1 - p)), g = Math.round(200 * p);
            ctx.fillStyle = 'rgba(' + r + ',' + g + ',60,0.18)';
            ctx.fillRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
            ctx.strokeStyle = 'rgb(' + r + ',' + g + ',60)';
          } else {
            ctx.strokeStyle = 'rgba(160,160,160,0.8)';
          }
          ctx.strokeRect(gx0, gy0, gx1 - gx0, gy1 - gy0);
          if (s) {
            ctx.fillStyle = '#fff';
            const tag = (p != null ? 'p ' + p.toFixed(2) + ' | ' : '')
              + 'n ' + s.count;
            ctx.fillText(tag, gx0 + 4 / v.scale, gy0 + 14 / v.scale);
          }
        }
      }
      // live sweep highlight: the recrops of the forward pass in flight
      if (sweepFocus) {
        ctx.fillStyle = 'rgba(80, 180, 255, 0.12)';
        ctx.strokeStyle = '#50b4ff';
        ctx.lineWidth = 3 / v.scale;
        for (const [sx0, sy0, sx1, sy1] of sweepFocus) {
          ctx.fillRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
          ctx.strokeRect(sx0, sy0, sx1 - sx0, sy1 - sy0);
        }
      }
      if (rq.queued.length || rq.inflight.length) {
        ctx.strokeStyle = '#1e90ff';
        ctx.lineWidth = 2 / v.scale;
        ctx.setLineDash([6 / v.scale, 4 / v.scale]);
        for (const q of rq.queued) {
          ctx.strokeRect(q.bbox[0], q.bbox[1], q.bbox[2] - q.bbox[0], q.bbox[3] - q.bbox[1]);
        }
        ctx.setLineDash([]);
        for (const q of rq.inflight) {
          ctx.strokeRect(q.bbox[0], q.bbox[1], q.bbox[2] - q.bbox[0], q.bbox[3] - q.bbox[1]);
        }
      }
      if (!ui.hideMasks) {
        // boxes and labels draw per-frame at screen-constant size (/ v.scale);
        // off-view instances skip
        const [vx0, vy0, vx1, vy1] = v.viewportRect();
        ctx.lineWidth = 1.5 / v.scale;
        ctx.font = (12 / v.scale) + 'px sans-serif';
        for (const inst of solo ? [solo] : allInstances()) {
          if (!shown(inst)) continue;
          const [x0, y0, x1, y1] = inst.bboxXyxy;
          if (x1 < vx0 || y1 < vy0 || x0 > vx1 || y0 > vy1) continue;
          if (outlined(inst) && inst.maskImg && inst !== drawHide) {
            const [rx0, ry0, rx1, ry1] = inst.maskRect;
            drawOutline(ctx, 2, labelColor(inst.label), inst.maskImg,
                        rx0, ry0, rx1 - rx0, ry1 - ry0);
          }
          const color = PASS_COLORS[inst.origin] || labelColor(inst.label);
          ctx.strokeStyle = color;
          ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
          ctx.fillStyle = color;
          const tag = inst.label + (inst.score != null ? ' ' + inst.score.toFixed(2) : '');
          ctx.fillText(tag, x0, Math.max(12 / v.scale, y0 - 3 / v.scale));
        }
        if (selected && !selected.pendingDelete && !ui.locked) emphasize(ctx, v, selected);
        if (hover && hover !== selected && !hover.pendingDelete && !ui.locked && !solo) {
          emphasize(ctx, v, hover);
        }
      }
      if (ui.locked) md.drawPreview(ctx);
      updateChrome();
    }

    // --- viewer wiring (all callback coordinates are original-image px) -----

    const viewer = createViewer($('view'), {
      dragMode: () => (ui.mode === 'exemplar' || ui.mode === 'reprompt'
                       || ui.mode === 'erase' ? 'box' : 'pan'),
      onClick: (x, y) => {
        if (ui.mode === 'point') {
          const label = currentLabel();
          if (label) segment({ prompt_type: 'point', point: [x, y] }, label);
          return;
        }
        setSelected(instanceAt(x, y));
      },
      onBoxDrawn: (box) => {
        if (box[2] - box[0] < MIN_BOX || box[3] - box[1] < MIN_BOX) {
          say('Box too small; drag a larger one.');
          return;
        }
        if (ui.mode === 'erase') { eraseInBox(box); return; }
        if (ui.mode === 'reprompt') { refineBox(box); return; }
        const label = currentLabel();
        // the box and the prompt text prompt together
        if (label) segment({ prompt_type: 'exemplar', boxes: [box], text: label }, label);
      },
      onHover: (x, y) => {
        if (x == null) { setHover(null); return; }
        setHover(instanceAt(x, y), true);
      },
      onViewChange: (rect) => { fetchCropPreview(rect); },
    });
    viewer.setOverlay(drawOverlay);

    // Recompute the backing-store size only; zoom and pan survive (resize
    // re-clamps the pan). resetView runs on image change (via setImage) or
    // the explicit FIT action, never on a resize or page round-trip.
    function sizeCanvas() {
      if (ui.locked) return;  // the edit rect is frozen
      const w = $('canvaswrap').clientWidth, h = $('canvaswrap').clientHeight;
      if (w && h) viewer.resize(w, h);
    }

    function updateChrome() {
      const mz = viewer.maxZoom;
      $('zslider').disabled = ui.locked || mz <= 1;
      $('zslider').value = mz > 1
        ? Math.round(Math.log(viewer.zoom) / Math.log(mz) * ZOOM_STEPS) : 0;
      $('zoomtext').textContent = Math.round(viewer.scale * 100) + '%';
      drawModelView();
      positionSelPop();
    }

    // center of the model input, 1 model px = 1 screen px
    function drawModelView() {
      // hidden while locked: the panel would cover part of the edit rect
      const on = state.previewImg && !ui.locked;
      show('modelview', on && ui.modelView, 'block');
      show('modelpill', on && !ui.modelView, 'block');
      if (!on || !ui.modelView) return;
      const rect = promptRect() || [0, 0, state.imgW, state.imgH];
      const w = rect[2] - rect[0], h = rect[3] - rect[1];
      const s = Math.min(1, MODEL_SIDE / Math.max(w, h));
      const cv = $('modelcv'), mctx = cv.getContext('2d');
      mctx.setTransform(1, 0, 0, 1, 0, 0);
      mctx.fillStyle = '#222';
      mctx.fillRect(0, 0, cv.width, cv.height);
      mctx.setTransform(s, 0, 0, s, cv.width / 2 - (rect[0] + w / 2) * s,
                        cv.height / 2 - (rect[1] + h / 2) * s);
      mctx.drawImage(state.previewImg, 0, 0, state.imgW, state.imgH);
      const patch = viewer.patch;
      if (patch) {
        const [x0, y0, x1, y1] = patch.rect;
        mctx.drawImage(patch.img, x0, y0, x1 - x0, y1 - y0);
      }
      $('modeltext').textContent = 'model input ' + Math.round(w * s) + 'x'
        + Math.round(h * s) + ' | ' + s.toFixed(2) + 'x';
    }

    $('zin').addEventListener('click', () => { if (!ui.locked) viewer.zoomBy(1.4); });
    $('zout').addEventListener('click', () => { if (!ui.locked) viewer.zoomBy(1 / 1.4); });
    $('zfit').addEventListener('click',
      () => { if (!ui.locked) viewer.resetView(); });
    $('zslider').addEventListener('input',
      () => viewer.zoomTo(Math.pow(viewer.maxZoom, $('zslider').value / ZOOM_STEPS)));
    function toggleModelView() { ui.modelView = !ui.modelView; drawModelView(); }
    $('modelhide').addEventListener('click', toggleModelView);
    $('modelpill').addEventListener('click', toggleModelView);

    // Latest-wins image fetch: POST body to url, load json[key] as an <img>,
    // and hand it to onImage unless slot.n has moved on (a newer call or a
    // slot.n++ elsewhere). Failures and stale responses drop silently.
    async function fetchLatestImage(slot, url, body, key, onImage) {
      const token = ++slot.n;
      try {
        const { ok, json } = await postJson(url, body);
        if (!ok || token !== slot.n) return;
        loadMaskImage(json[key], img => { if (token === slot.n) onImage(img, json); });
      } catch (err) { /* the current image stands */ }
    }

    // When zoomed past fit, swap in a crop of the original image; the
    // page-load preview is capped at PREVIEW_MAX_SIDE.
    const cropReq = { n: 0 };  // bumped when the view moves on
    function fetchCropPreview(rect) {
      if (!state.imagePath || viewer.zoom <= ZOOMED_IN) return;
      // an image within MODEL_SIDE is already at full resolution
      if (Math.max(state.imgW, state.imgH) <= MODEL_SIDE) return;
      fetchLatestImage(cropReq, '/crop_preview',
        { image_path: state.imagePath, rect_xyxy: rect }, 'preview',
        (img, json) => { viewer.setHiResPatch(img, json.rect_xyxy); drawModelView(); });
    }

    // --- selection ----------------------------------------------------------

    function setSelected(inst) {
      selected = inst || null;
      if (selected) {
        if (selected.maskFailed && !selected.maskImg) {
          const url = selected.maskFailed;  // re-request the failed mask
          selected.maskFailed = null;
          decodeMaskInto(selected, url);
        }
        scrollToRow(selected);
      }
      renderInstances();
      scheduleRedraw();
    }

    function positionSelPop() {
      if (!selected || ui.page !== 'work' || ui.locked || !isLive(selected)) {
        show('selpop', false);
        return;
      }
      const [x0, , x1, y1] = selected.bboxXyxy;
      const [cx, cy] = viewer.imageToCanvas((x0 + x1) / 2, y1);
      show('selpop', true, 'flex');
      $('sellbl').textContent = selected.label + ' | ' + instText(selected);
      $('sellbl').style.color = labelColor(selected.label);
      const w = $('canvaswrap').clientWidth || 0, h = $('canvaswrap').clientHeight || 0;
      $('selpop').style.left = Math.max(4, Math.min((w || cx + 130) - 130, cx - 60)) + 'px';
      $('selpop').style.top = Math.max(4, Math.min((h || cy + 40) - 40, cy + 10)) + 'px';
    }

    $('seldrop').addEventListener('click', () => {
      if (selected) deleteInstance(selected);
    });
    $('selrep').addEventListener('click', () => {
      if (!selected) return;
      queueReprompt(selected.bboxXyxy.slice(), selected.label);
      say('Re-prompting around the selected ' + selected.label + '...');
      setSelected(null);
    });

    // --- manual mask edit: frozen viewport + createMaskDraw -----------------

    let md = null;          // createMaskDraw instance, built on first lock
    let drawRect = null;  // frozen [x0,y0,x1,y1] while locked
    let drawHide = null;    // instance the edit replaces; its normal render is suppressed
    let drawTool = 'brush';
    // draw: Apply stages a replacement instance (EDIT). touchup: Save
    // manual updates writes the selected instance's mask in place.
    let lockKind = 'draw';

    function lockBlock() {
      if (!ui.locked) return false;
      say(lockKind === 'touchup'
        ? 'Touch-up in progress - Save manual updates or cancel first.'
        : 'Manual edit in progress - Apply or cancel first.');
      return true;
    }

    function syncDrawBar() {
      $('dundo').disabled = !md.canUndo();
      $('dredo').disabled = !md.canRedo();
      $('dapply').disabled = !md.canUndo();
      $('dsave').disabled = !md.canUndo();
    }

    function setDrawTool(t) {
      drawTool = t;
      md.setTool(t);
      for (const [name, id] of Object.entries(DRAW_TOOLS)) {
        $(id).classList.toggle('on', name === t);
      }
      scheduleRedraw();
    }
    for (const [name, id] of Object.entries(DRAW_TOOLS)) {
      $(id).addEventListener('click', () => setDrawTool(name));
    }

    function setDrawErase(b) {
      md.setErase(b);
      $('derase').classList.toggle('danger', b);
    }
    $('derase').addEventListener('click',
      () => setDrawErase(!$('derase').classList.contains('danger')));

    function applyDiameter() {
      const px = parseInt($('dsize').value, 10);
      $('dsizeval').textContent = px;
      md.setDiameter(px);
      scheduleRedraw();
    }
    $('dsize').addEventListener('input', applyDiameter);

    // While a stroke is down, document-level capture keeps it alive off-canvas.
    let drawStroking = false;
    function onDrawDocMove(evt) {
      evt.stopPropagation();
      md.pointerMove(...viewer.eventToImage(evt));
      scheduleRedraw();
    }
    function onDrawDocUp(evt) {
      document.removeEventListener('mousemove', onDrawDocMove, true);
      document.removeEventListener('mouseup', onDrawDocUp, true);
      drawStroking = false;
      evt.stopPropagation();
      md.pointerUp(...viewer.eventToImage(evt));
      scheduleRedraw();
      syncDrawBar();
    }

    // Capture-phase listeners on the wrap run before the viewer's canvas
    // listeners; the lock swallows pan/zoom here. Events not targeting the
    // canvas pass through.
    $('canvaswrap').addEventListener('wheel', (evt) => {
      if (!ui.locked) return;
      evt.preventDefault();
      evt.stopPropagation();
    }, { capture: true, passive: false });
    $('canvaswrap').addEventListener('mousedown', (evt) => {
      if (!ui.locked || evt.target !== $('view') || evt.button !== 0) return;
      evt.stopPropagation();
      evt.preventDefault();
      drawStroking = true;
      document.addEventListener('mousemove', onDrawDocMove, true);
      document.addEventListener('mouseup', onDrawDocUp, true);
      md.pointerDown(...viewer.eventToImage(evt));
      scheduleRedraw();
      syncDrawBar();
    }, { capture: true });
    $('canvaswrap').addEventListener('mousemove', (evt) => {
      if (!ui.locked || drawStroking) return;
      if (evt.target !== $('view')) md.pointerOut();  // cursor over chrome: drop the ring
      else { evt.stopPropagation(); md.pointerMove(...viewer.eventToImage(evt)); }
      scheduleRedraw();
    }, { capture: true });
    $('canvaswrap').addEventListener('contextmenu', (evt) => {
      if (!ui.locked || evt.target !== $('view')) return;
      evt.preventDefault();
      evt.stopPropagation();
      md.finishShape();  // right-click closes the polygon
      scheduleRedraw();
    }, { capture: true });
    $('canvaswrap').addEventListener('mouseleave', () => {
      if (ui.locked) { md.pointerOut(); scheduleRedraw(); }
    });

    // Visible region as an integer rect clamped to the image; null when degenerate.
    function viewRect() {
      const r = viewer.viewportRect();
      const rect = [Math.max(0, Math.round(r[0])), Math.max(0, Math.round(r[1])),
                    Math.min(state.imgW, Math.round(r[2])),
                    Math.min(state.imgH, Math.round(r[3]))];
      return (rect[2] - rect[0] >= 2 && rect[3] - rect[1] >= 2) ? rect : null;
    }

    // Draw base for the edit rect: a native-resolution crop replaces the
    // instance's PNG (capped over its whole tight rect) once it arrives.
    const cropBaseReq = { n: 0 };  // bumped on unlock
    function fetchDrawBase(inst, rect) {
      fetchLatestImage(cropBaseReq, '/mask_crop',
        { image_path: state.imagePath, rle: inst.rle, rect: rect }, 'mask_png',
        (img, json) => { md.setBase({ img: img, rect: json.rect }); scheduleRedraw(); });
    }

    // While locked, masks render from a composite of the instances' PNGs
    // covering the edit rect at native resolution (capped like the draw
    // composite).
    let lockOverlay = null;  // rebuilt lazily after markOverlayDirty/Add
    function buildLockOverlay() {
      const [x0, y0, x1, y1] = drawRect;
      const w = x1 - x0, h = y1 - y0;
      const cs = Math.min(1, CFG.MASK_SIDE / Math.max(w, h));
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(w * cs));
      c.height = Math.max(1, Math.round(h * cs));
      const octx = c.getContext('2d');
      octx.globalAlpha = MASK_ALPHA;
      for (const inst of allInstances()) {
        if (!shown(inst) || !inst.maskImg || inst === drawHide) continue;
        const r = inst.maskRect;
        if (r[2] <= x0 || r[0] >= x1 || r[3] <= y0 || r[1] >= y1) continue;
        octx.drawImage(hiResTint(inst, labelColor(inst.label)),
                       (r[0] - x0) * cs, (r[1] - y0) * cs,
                       (r[2] - r[0]) * cs, (r[3] - r[1]) * cs);
      }
      lockOverlay = c;
    }

    // opts: {kind: 'draw'|'touchup', tool, erase}; touchup needs a selected
    // instance whose mask the edits land in
    function setLocked(on, opts = {}) {
      if (on === ui.locked) return;
      if (on) {
        const kind = opts.kind || 'draw';
        if (noImageBlock() || viewOnlyBlock() || busy || ui.page !== 'work') return;
        if (kind === 'touchup' && !selected) { say('Select an instance to touch up.'); return; }
        if (selected && (!selected.maskImg || !selected.rle)) {
          say(selected.rle ? 'Mask still loading - try again.' : 'Instance has no editable mask.');
          return;
        }
        const rect = viewRect();
        if (!rect) return;
        md = md || createMaskDraw(CFG.MASK_SIDE, syncDrawBar);
        md.setOutline(ui.outline);
        lockKind = kind;
        drawRect = rect;
        drawHide = selected || null;
        md.begin({
          rect: rect,
          base: drawHide ? { img: drawHide.maskImg, rect: drawHide.maskRect } : null,
          color: drawHide ? labelColor(drawHide.label) : '#0e7fc1',
          scale: viewer.scale,
        });
        ui.locked = true;
        if (drawHide) fetchDrawBase(drawHide, rect);
        const touch = kind === 'touchup';
        for (const [name, id] of Object.entries(DRAW_TOOLS)) {
          show(id, !touch || TOUCHUP_TOOLS.includes(name));
        }
        show('dapply', !touch);
        show('dsave', touch);
        setDrawTool(opts.tool || (touch && !TOUCHUP_TOOLS.includes(drawTool) ? 'brush' : drawTool));
        setDrawErase(!!opts.erase);
        applyDiameter();
        hover = null;
        closeMenus();
        show('drawbar', true, 'flex');
        drawModelView();
        $('zlock').classList.add('on');
        $('canvaswrap').classList.add('locked');
        show('modehint', true, 'block');
        $('modehint').textContent = touch
          ? 'Touching up "' + drawHide.label + ' | ' + instText(drawHide)
            + '" - brush / rect, add or remove; Save manual updates writes the mask'
          : drawHide
            ? 'Editing "' + drawHide.label + '" - draw, then Apply'
            : 'Drawing a new instance - label comes from the prompt field on Apply';
        $('canvaswrap').style.cursor = 'crosshair';
        markOverlayDirty();
      } else {
        ui.locked = false;
        cropBaseReq.n++;
        md.end();
        drawRect = null;
        drawHide = null;
        drawStroking = false;
        show('drawbar', false);
        $('zlock').classList.remove('on');
        $('canvaswrap').classList.remove('locked');
        setMode(ui.mode);  // restores the hint and cursor
        drawModelView();
        markOverlayDirty();
      }
    }
    $('zlock').addEventListener('click', () => setLocked(!ui.locked));
    $('dcancel').addEventListener('click', () => setLocked(false));
    // the menu opens below the popup row, above it when that would leave
    // the canvas
    $('seltouch').addEventListener('click', (e) => {
      const menu = $('touchmenu');
      toggleMenu(menu, e);
      if (menu.style.display !== 'block') return;
      menu.classList.remove('up');
      const wrap = $('canvaswrap').getBoundingClientRect();
      menu.classList.toggle('up', menu.getBoundingClientRect().bottom > wrap.bottom);
    });
    for (const [id, tool, erase] of TOUCH_ITEMS) {
      $(id).addEventListener('click', (e) => {
        e.stopPropagation();
        closeMenus();
        setLocked(true, { kind: 'touchup', tool: tool, erase: erase });
      });
    }
    const drawUndo = () => { md.undo(); scheduleRedraw(); };
    const drawRedo = () => { md.redo(); scheduleRedraw(); };
    $('dundo').addEventListener('click', drawUndo);
    $('dredo').addEventListener('click', drawRedo);

    async function submitDraw(btnId, fn) {
      const ops = md.ops();
      if (!ops.length) { say('Nothing drawn.'); return; }
      $(btnId).disabled = true;
      try { await fn(ops); } finally { syncDrawBar(); }
    }
    const drawMaskBody = (ops, baseRle) =>
      ({ image_path: state.imagePath, rect: drawRect, base_rle: baseRle, ops: ops });

    async function applyDraw(ops) {
      const src = drawHide;
      const label = src ? src.label : currentLabel();
      if (!label) return;
      const json = await call('/draw_mask', drawMaskBody(ops, src ? src.rle : null));
      if (!json) return;
      if (json.empty && !src) { say('Drawing erased everything - nothing to add.'); return; }
      let replacement = null;
      if (!json.empty) replacement = addPending(label, json, src ? src.origin : null);
      if (src) {
        // the replacement supersedes the source; /save carries the delete
        if (src.annotationId != null) src.pendingDelete = true;
        else state.pending = state.pending.filter(p => p !== src);
      }
      setLocked(false);
      renderAll();
      setSelected(replacement);
      say(json.empty ? '"' + label + '" fully erased - staged for delete.'
                     : 'Mask drawn onto "' + label + '" - review, then save.');
    }
    $('dapply').addEventListener('click', () => submitDraw('dapply', applyDraw));

    // Touch-up save: the ops rasterize over the instance's own mask and the
    // instance keeps its identity. A saved annotation is rewritten on disk
    // by /update_annotation; a pending one goes through /draw_mask and only
    // its in-memory fields change (Save commits it).
    function replaceMask(inst, json) {
      inst.bboxXyxy = json.bbox_xyxy;
      inst.rle = json.rle;
      inst.maskRect = json.mask_rect;
      inst.centroid = json.centroid || null;
      inst.maskImg = null;
      dropTints(inst);
      decodeMaskInto(inst, json.mask_png);
    }
    async function saveTouchup(ops) {
      const inst = drawHide;
      let json;
      if (inst.annotationId != null) {
        // a stale reference unlocks before the resync (loadImage refuses while locked)
        json = await call('/update_annotation', {
          image_path: state.imagePath, coco_file: state.cocoFile,
          annotation_id: inst.annotationId, bbox_xyxy: inst.bboxXyxy,
          rect: drawRect, ops: ops,
        }, () => setLocked(false));
      } else {
        json = await call('/draw_mask', drawMaskBody(ops, inst.rle));
        if (json && json.empty) {
          say('Touch-up erased the whole mask - delete the instance instead.');
          return;
        }
      }
      if (!json) return;
      if (json.coco_file) { state.cocoFile = json.coco_file; invalidateLoadCache(); }
      replaceMask(inst, json);
      setLocked(false);
      renderDirty();
      say(inst.annotationId != null
        ? 'Manual updates saved to "' + inst.label + '" #' + inst.annotationId + ' - ' + state.cocoFile
        : 'Manual updates applied to pending "' + inst.label + '" - Save commits.');
    }
    $('dsave').addEventListener('click', () => submitDraw('dsave', saveTouchup));

    // --- instances panel (keyed reconciler) ---------------------------------

    function instText(inst) {
      return inst.annotationId != null
        ? '#' + inst.annotationId
        : 'p' + inst.tmpId + ' ' + (inst.score != null ? inst.score.toFixed(2) : '-');
    }
    function instDims(inst) {
      const [x0, y0, x1, y1] = inst.bboxXyxy;
      return Math.round(x1 - x0) + 'x' + Math.round(y1 - y0);
    }

    // static structure + handlers, once per instance; handlers read live
    // state at event time (inst.pendingDelete flips the del button's action)
    function buildRow(inst) {
      const row = el('div', 'irow');
      const box = el('input', '', null, row);
      box.type = 'checkbox';
      box.title = "show this instance's mask in the overlay";
      box.addEventListener('change', () => {
        inst.visible = box.checked;
        markOverlayDirty();
        renderInstances();
      });
      const text = el('span', 'ell', null, row);
      const dims = el('span', 'dims', null, row);
      const del = el('button', 'del x', null, row);
      del.addEventListener('click', (e) => {
        e.preventDefault(); e.stopPropagation();
        if (inst.pendingDelete) {
          inst.pendingDelete = false; renderDirty();
        } else {
          deleteInstance(inst);
        }
      });
      row.addEventListener('click', (e) => {
        if (e.target === box || e.target === del) return;
        setSelected(inst);
      });
      row.addEventListener('mouseenter', () => setHover(inst));
      row.addEventListener('mouseleave', () => { if (hover === inst) setHover(null); });
      row._ui = { box, text, dims, del };
      return row;
    }

    function updateRow(row, inst) {
      const uiRef = row._ui;
      if (uiRef.box.checked !== inst.visible) uiRef.box.checked = inst.visible;
      const dis = !!inst.pendingDelete;
      if (uiRef.box.disabled !== dis) uiRef.box.disabled = dis;
      setText(uiRef.text, instText(inst));
      setText(uiRef.dims, instDims(inst));
      setText(uiRef.del, dis ? 'undo' : 'x');
      uiRef.del.title = dis ? 'undo staged delete' : 'delete instance';
      row.classList.toggle('pdel', dis);
      row.classList.toggle('sel', inst === selected);
      row.classList.toggle('hov', inst === pinned && inst !== selected);
      const opacity = labelShown(inst) ? '' : '0.45';  // class eye off on canvas
      if (row.style.opacity !== opacity) row.style.opacity = opacity;
    }

    function buildHeader(label) {
      const hdr = el('div', 'grouphdr');
      const caret = el('span', 'caret', null, hdr);
      const dot = el('span', 'dot9', null, hdr);
      el('span', 'name', label, hdr);
      const cnt = el('span', 'mono-dim', null, hdr);
      el('span', 'grow', null, hdr);
      const redo = el('span', 'redo x', 'redo', hdr);
      redo.title = 'redo this class from scratch on this image';
      const ring = el('span', 'ring x', null, hdr);
      ring.title = 'draw this class as fills, rings or both on the canvas';
      const eye = el('span', 'eye x', null, hdr);
      eye.title = 'show/hide this class on the canvas';
      const del = el('span', 'del x', 'x', hdr);
      del.title = 'delete every instance of this class on this image '
        + '(saved ones staged until Save)';
      const cycleRing = () => {
        const mode = outlineLabels.get(label);
        if (mode === 'both') outlineLabels.delete(label);
        else outlineLabels.set(label, mode ? 'both' : 'ring');
      };
      const toggleEye = () => {
        if (hiddenLabels.has(label)) hiddenLabels.delete(label);
        else hiddenLabels.add(label);
      };
      hdr.addEventListener('click', (e) => {
        if (e.target === redo) { openRedo(label); return; }
        if (e.target === del) { deleteLabelInstances(label); return; }
        if (e.target === ring || e.target === eye) {
          (e.target === ring ? cycleRing : toggleEye)();
          markOverlayDirty();
          renderInstances();
          return;
        }
        if (collapsedLabels.has(label)) collapsedLabels.delete(label);
        else collapsedLabels.add(label);
        renderInstances();
      });
      hdr._ui = { caret, dot, cnt, ring, eye };
      return hdr;
    }

    function updateHeader(hdr, label, count) {
      const u = hdr._ui;
      const shown = !hiddenLabels.has(label);
      setText(u.caret, collapsedLabels.has(label) ? '>' : 'v');
      u.dot.style.background = labelColor(label);
      u.dot.style.opacity = shown ? '' : '0.3';
      setText(u.cnt, 'x' + count);
      setText(u.ring, outlineLabels.get(label) || 'fill');
      setText(u.eye, shown ? 'on' : 'off');
    }

    function renderInstances() {
      if (hover && !isLive(hover)) hover = null;
      if (pinned && !isLive(pinned)) pinned = null;
      if (selected && !isLive(selected)) selected = null;
      const groups = new Map();
      for (const inst of allInstances()) {
        if (!groups.has(inst.label)) groups.set(inst.label, []);
        groups.get(inst.label).push(inst);
      }
      if (!groups.size) {
        rowByInst.clear();
        hdrByLabel.clear();
        moreByLabel.clear();
        $('instances').textContent =
          state.imagePath ? 'No instances yet.' : 'Load an image first.';
      } else {
        // clear the placeholder text
        if (!rowByInst.size && !hdrByLabel.size) $('instances').textContent = '';
        const desired = [];
        for (const [label, insts] of groups) {
          let hdr = hdrByLabel.get(label);
          if (!hdr) { hdr = buildHeader(label); hdrByLabel.set(label, hdr); }
          updateHeader(hdr, label, insts.length);
          desired.push(hdr);
          if (collapsedLabels.has(label)) continue;  // hides only the rows
          for (const inst of insts.slice(0, ROW_CAP)) {
            let row = rowByInst.get(inst);
            if (!row) { row = buildRow(inst); rowByInst.set(inst, row); }
            updateRow(row, inst);
            desired.push(row);
          }
          if (insts.length > ROW_CAP) {
            let more = moreByLabel.get(label);
            if (!more) { more = el('div', 'more'); moreByLabel.set(label, more); }
            more.textContent = '... ' + (insts.length - ROW_CAP)
              + ' more - click a mask to jump';
            desired.push(more);
          }
        }
        const want = new Set(desired);
        for (const [inst, row] of rowByInst) if (!want.has(row)) rowByInst.delete(inst);
        for (const [label, hdr] of hdrByLabel) if (!want.has(hdr)) hdrByLabel.delete(label);
        for (const [label, m] of moreByLabel) if (!want.has(m)) moreByLabel.delete(label);
        let child = $('instances').firstChild;
        while (child) {
          const next = child.nextSibling;
          if (!want.has(child)) $('instances').removeChild(child);
          child = next;
        }
        let cursor = $('instances').firstChild;
        for (const node of desired) {
          if (node === cursor) { cursor = cursor.nextSibling; continue; }
          $('instances').insertBefore(node, cursor);
        }
      }
      const total = allInstances().filter(i => !i.pendingDelete).length;
      $('totalcnt').textContent = total ? '| ' + total : '';
      const nAdd = state.pending.filter(p => p.visible).length;
      const nDel = state.saved.filter(s => s.pendingDelete).length;
      $('save').textContent = 'Save | ' + nAdd + ' new' + (nDel ? ', ' + nDel + ' del' : '');
      $('save').disabled = (!state.pending.length && !nDel) || !state.imagePath;
      $('saved').textContent = (nAdd || nDel)
        ? '* ' + (nAdd + nDel) + ' unsaved' : '* saved';
      $('saved').className = (nAdd || nDel) ? 'dirty' : '';
    }

    function renderChips() {
      const items = getHistory();
      $('chips').innerHTML = '';
      for (const term of items) {
        const chip = el('div', 'chip', null, $('chips'));
        tintChip(chip, labelColor(term));
        el('span', '', term, chip);
        const x = el('span', 'x', 'x', chip);
        x.title = 'remove the class and its instances on this image';
        chip.addEventListener('click', (e) => {
          if (e.target === x) { removeClass(term); return; }
          $('promptText').value = term;
          setMode('text');
        });
      }
      $('runall').textContent = 'Run all ' + (items.length || '') + ' on image';
      $('runall').disabled = !state.imagePath || !items.length;
      $('batchsub').textContent =
        'run ' + (items.length || 'your') + ' prompt' + (items.length === 1 ? '' : 's')
        + ' over the directory';
    }

    function dropAndReport(pred, verb) {
      const { dropped, staged } = dropInstances(pred);
      renderDirty();
      toast(verb + ' ' + dropped + ' pending'
        + (staged ? ' | staged ' + staged + ' saved for delete (Save commits)' : ''));
      return dropped + staged > 0;
    }

    // header delete button: the class's instances off this image; the label
    // and its chip stay
    function deleteLabelInstances(label) {
      if (!liveCount(label)) { say('No "' + label + '" instances to delete.'); return; }
      dropAndReport(i => i.label === label, 'Deleted "' + label + '":');
    }

    // chip x: forget the prompt and take its instances off this image
    function removeClass(term) {
      setHistory(getHistory().filter(t => t !== term));
      dropAndReport(i => i.label === term, 'Class "' + term + '" removed:');
    }

    // does not invalidate the mask overlay
    function renderAll() {
      renderInstances(); renderChips(); scheduleRedraw();
    }
    function renderDirty() { markOverlayDirty(); renderAll(); }

    let renderTimer = 0;
    function throttledRenderAll() {
      if (renderTimer) return;
      renderTimer = setTimeout(() => { renderTimer = 0; renderAll(); }, 200);
    }
    function flushRenderAll() {
      if (renderTimer) { clearTimeout(renderTimer); renderTimer = 0; }
      renderAll();
    }

    // --- history (browser-side prompt list; drives the class chips) ---------

    function getHistory() {
      try { return JSON.parse(localStorage.getItem(HISTORY_KEY)) || []; }
      catch (e) { return []; }
    }
    function setHistory(items) {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(items));
      renderChips();
    }
    function rememberPrompt(term) {
      const items = getHistory().filter(t => t !== term);
      items.unshift(term);
      setHistory(items.slice(0, 30));
    }

    // --- label management (dataset-wide category edits) ---------------------

    const lblCollapsed = new Set();  // folded tree nodes (session-only)
    function lblCat(id) { return state.labelCats.find(c => c.id === id); }

    function lblRollup(node, key) {  // own count + descendants'
      const cat = node.id != null ? lblCat(node.id) : null;
      let sum = cat ? cat[key] : 0;
      for (const child of node.children) sum += lblRollup(child, key);
      return sum;
    }

    function lblSubtreeNames(node, out) {
      out.push(node.name);
      for (const child of node.children) lblSubtreeNames(child, out);
      return out;
    }

    // shared parent-option list, rebuilt per data load; a row's select fills
    // from it on first open
    let lblParentOptions = [];
    function buildParentOptions() {
      lblParentOptions = [{ text: '(no parent)', value: '' }];
      for (const root of state.labelTree) {  // group headings only ever sit at root
        if (root.id == null) lblParentOptions.push({ text: '# ' + root.name, value: root.name, heading: true });
      }
      for (const c of state.labelCats) lblParentOptions.push({ text: '^ ' + c.name, value: c.name });
    }
    function parentDisplay(cat) {
      const superc = cat.supercategory;
      if (!superc || superc === cat.name) return { text: '(no parent)', value: '' };
      const isLabel = state.labelCats.some(k => k.name === superc);
      return { text: (isLabel ? '^ ' : '# ') + superc, value: superc };
    }

    function lblMatches(node, needle) {
      if (!needle) return true;
      if (node.name.toLowerCase().includes(needle)) return true;
      return node.children.some(c => lblMatches(c, needle));
    }

    function lblSortNodes(nodes) {
      const out = nodes.slice();
      out.sort($('lblsort').value === 'count'
        ? (a, b) => lblRollup(b, 'total') - lblRollup(a, 'total') || a.name.localeCompare(b.name)
        : (a, b) => a.name.localeCompare(b.name));
      return out;
    }

    function renderLabelNodes(nodes, into, needle) {
      for (const node of lblSortNodes(nodes.filter(n => lblMatches(n, needle)))) {
        into.appendChild(renderLabelNode(node, needle));
      }
    }

    function renderLabelNode(node, needle) {
      const wrap = el('div');
      const row = el('div', 'hrow lrow', null, wrap);
      const key = (node.id == null ? 'g:' : 'c:') + node.name;
      const twist = el('span', 'twist x', null, row);
      if (node.children.length) {
        twist.textContent = lblCollapsed.has(key) ? '>' : 'v';
        twist.addEventListener('click', () => {
          if (lblCollapsed.has(key)) lblCollapsed.delete(key); else lblCollapsed.add(key);
          renderLabelTree();
        });
      }
      if (node.id == null) {  // group heading: no category behind it
        el('span', 'name', node.name, row);
      } else {
        const cat = lblCat(node.id);
        const box = el('input', 'pick', null, row);
        box.type = 'checkbox';
        box.value = node.name;
        const swatch = el('span', 'dot9', null, row);
        swatch.style.background = labelColor(node.name);
        let counts = cat.in_image + ' in image, ' + cat.total + ' total';
        if (node.children.length) counts += ', ' + lblRollup(node, 'total') + ' in subtree';
        el('span', 'ell', node.name + ' - ' + counts, row);
        // parent picker: shows the current parent (label or group heading);
        // the shared option list is filled in on first open
        const sel = el('select', 'parent', null, row);
        sel.title = 'parent label or group';
        const cur = parentDisplay(cat);
        sel.appendChild(new Option(cur.text, cur.value));
        sel.value = cur.value;
        let filled = false;
        const fill = () => {
          if (filled) return;
          filled = true;
          const banned = lblSubtreeNames(node, []);  // headings never cycle
          sel.innerHTML = '';
          for (const o of lblParentOptions) {
            if (!o.heading && o.value && o.value !== cur.value && banned.includes(o.value)) continue;
            sel.appendChild(new Option(o.text, o.value));
          }
          sel.value = cur.value;
        };
        sel.addEventListener('focus', fill);
        sel.addEventListener('mousedown', fill);
        sel.addEventListener('change', () => setLabelParent(node.name, sel.value));
      }
      if (node.children.length && !lblCollapsed.has(key)) {
        const kids = el('div', 'kids', null, wrap);
        renderLabelNodes(node.children, kids, needle);
      }
      return wrap;
    }

    function renderLabelTree() {
      $('labeltree').innerHTML = '';
      if (!state.imagePath) { $('labeltree').textContent = 'Load an image first.'; return; }
      if (!state.labelCats.length) {
        $('labeltree').textContent = 'No labels in this dataset yet.';
        return;
      }
      renderLabelNodes(state.labelTree, $('labeltree'), $('lblfilter').value.trim().toLowerCase());
    }

    function pickedLabels() {
      return [...$('labeltree').querySelectorAll('input.pick:checked')].map(b => b.value);
    }

    function updateLabelPanel() {
      const off = !state.cocoFile || state.viewOnly;
      $('lbldrop').disabled = $('lblmerge').disabled = $('lbladd').disabled = off;
      renderLabelTree();
    }

    // POST one dataset-wide label edit (/edit_labels op), then resync from
    // disk (pending kept); the response lists the categories left
    async function labelEdit(body, doneMsg) {
      if (busyBlock()) return;
      const json = await call('/edit_labels', { coco_file: state.cocoFile, ...body });
      if (!json) return;
      state.cocoFile = json.coco_file;
      invalidateLoadCache();
      await loadImage({ keepPending: true });
      say(doneMsg(json));
    }

    function setLabelParent(name, parent) {
      return labelEdit({ op: 'parent', name: name, parent: parent },
        () => parent
          ? '"' + name + '" parented under "' + parent + '".'
          : '"' + name + '" moved to the root.');
    }

    $('lbldrop').addEventListener('click', () => {
      const picked = pickedLabels();
      if (!picked.length) { say('Tick the labels to remove first.'); return; }
      if (!confirm('Remove ' + picked.join(', ') + ' and all their bboxes dataset-wide?'
        + " Children re-root to the removed label's parent.")) return;
      labelEdit({ op: 'drop', labels: picked },
        j => 'Removed ' + j.removed_annotations + ' bbox(es); labels left: '
          + (j.categories.map(c => c.name).join(', ') || '(none)'));
    });

    $('lblmerge').addEventListener('click', () => {
      const picked = pickedLabels();
      const name = $('lblnewname').value.trim();
      if (!picked.length) { say('Tick the label(s) to merge or rename first.'); return; }
      if (!name) { say('Enter the surviving name.'); return; }
      if (!confirm((picked.length > 1 ? 'Merge ' : 'Rename ') + picked.join(', ')
        + ' -> "' + name + '" across the whole dataset?')) return;
      labelEdit({ op: 'merge', labels: picked, new_name: name },
        () => { $('lblnewname').value = ''; return 'Merged ' + picked.join(', ') + ' -> "' + name + '".'; });
    });

    $('lbladd').addEventListener('click', () => {
      const label = $('lblnew').value.trim();
      if (!label) { say('Enter a label name to add.'); return; }
      labelEdit({ op: 'add', label: label },
        j => {
          $('lblnew').value = '';
          const added = j.categories[j.categories.length - 1];
          return 'Added label "' + added.name + '" (category id ' + added.id + ')';
        });
    });

    $('lblfilter').addEventListener('input', renderLabelTree);
    $('lblsort').addEventListener('change', renderLabelTree);
    collapsible('lbltoggle', 'lblbody', '');

    // --- search metadata (which prompts ran with which knobs) ---------------

    function indexRecords() {
      return (state.searchIndexStatus !== 'invalid' && state.searchIndex
              && state.searchIndex.searches) || [];
    }

    async function recordSearches(records) {
      if (!records.length) return true;
      const forImage = state.imagePath;
      const { ok, json } = await postJson('/record_searches',
        { image_path: forImage, width: state.imgW, height: state.imgH,
          searches: records });
      if (ok) {
        invalidateLoadCache();
        if (state.imagePath === forImage) {  // may have navigated mid-flight
          state.searchIndex = json.index;
          state.searchIndexStatus = 'valid';
        }
      }
      return ok;
    }

    // zero-kept searches record immediately; ones that kept results wait for
    // a successful save
    function queueSearchRecords(res) {
      if (!res || !res.records || !res.records.length) return Promise.resolve();
      if (res.kept === 0) return recordSearches(res.records);
      searchLog.push(...res.records);
      return Promise.resolve();
    }

    async function flushSearchLog() {
      const recs = searchLog.splice(0);
      if (recs.length && !(await recordSearches(recs))) searchLog.unshift(...recs);
    }

    // --- loading ------------------------------------------------------------

    // /load responses for the directory neighbors, fetched in the background
    // once the current image settles. Entries are single-use; any annotation
    // mutation clears the cache.
    const loadCache = new Map();  // image path -> /load response json

    // /fsmeta of the loaded image's directory; best-effort, may be null.
    let dirMeta = null;
    async function fetchDirMeta(p) {
      try {
        const r = await getPath('/fsmeta', p || '');
        return r.ok ? r.json : null;
      } catch (err) { return null; }
    }

    // Zephyr mask files are listed apart from the images; the topbar pill
    // lists them, click opens one view-only.
    function isZephyrMaskName(n) { return /_masked?\.[^.]+$/i.test(n); }
    function viewOnlyBlock() {
      if (!state.viewOnly) return false;
      say('Zephyr mask - view only.');
      return true;
    }
    function renderMaskPill(m) {
      const masks = (m && m.masks) || [];
      show('maskwrap', masks.length > 0, 'flex');
      show('maskmenu', false);
      $('maskpill').textContent = masks.length + ' mask file(s)';
      $('maskmenu').innerHTML = '';
      for (const name of masks) {
        const row = el('div', 'item mono', name, $('maskmenu'));
        row.addEventListener('click', () => {
          show('maskmenu', false);
          $('path').value = listedPath(m, name);
          loadImage();
        });
      }
    }
    $('maskpill').addEventListener('click', (e) => toggleMenu($('maskmenu'), e));

    // window of neighbors around the loaded image in directory order, the
    // loaded one centred except at either end; click loads one
    function renderFilmstrip(m) {
      const strip = $('filmstrip');
      strip.innerHTML = '';
      const on = m && m.index != null && m.count > 1;
      show(strip, on, 'flex');
      if (!on) return;
      const lo = Math.max(0, Math.min(m.index - Math.floor((FILMSTRIP_MAX - 1) / 2),
                                      m.count - FILMSTRIP_MAX));
      for (let i = lo; i < Math.min(m.count, lo + FILMSTRIP_MAX); i++) {
        const p = listedPath(m, m.images[i]);
        const img = el('img', i === m.index ? 'on' : '', null, strip);
        img.src = ImageProxy.url(p, 512);
        img.title = m.images[i];
        img.addEventListener('click', () => navigate(p));
      }
    }
    function invalidateLoadCache() { loadCache.clear(); }
    async function prefetchLoad(p) {
      if (!p || loadCache.has(p)) return;
      try {
        const { ok, json } = await postJson('/load', { image_path: p });
        if (!ok) return;
        loadCache.set(p, json);
        while (loadCache.size > LOAD_CACHE_MAX) {
          loadCache.delete(loadCache.keys().next().value);
        }
      } catch (err) { /* prefetch is best-effort */ }
    }

    // unsaved work survives navigation: stashed per image, restored on return
    function stashImageState() {
      if (!state.imagePath) return;
      imgStore.set(state.imagePath, {
        pending: state.pending,
        stagedIds: state.saved.filter(s => s.pendingDelete).map(s => s.annotationId),
        searchLog: searchLog,
      });
    }

    let loadReq = 0;  // only the newest load's preview may install
    async function loadImage(opts = {}) {
      if (busyBlock() || lockBlock()) return false;
      // Explorer's "Copy as path" wraps the path in double quotes
      const p = $('path').value.trim().replace(/^"(.*)"$/s, '$1').trim();
      if (p !== $('path').value) $('path').value = p;
      // keepPending: same-image resync that reloads saved state from disk
      const keep = !!opts.keepPending && p === state.imagePath;
      if (!p) { say('Enter an image path.'); return false; }
      if (p !== state.imagePath) stashImageState();
      say('Loading...');
      const token = ++loadReq;  // claimed before the first await
      try {
        let json = loadCache.get(p);
        if (json) {
          loadCache.delete(p);  // single-use; the neighbor prefetch refills it
        } else {
          json = await call('/load', { image_path: p });
          if (!json) return false;
        }
        if (token !== loadReq) return false;  // a newer load owns the state
        const stash = keep ? null : imgStore.get(p);
        state.imagePath = p;
        state.viewOnly = isZephyrMaskName(basename(p));
        state.imgW = json.width; state.imgH = json.height;
        state.cocoFile = json.coco_file;
        if (!keep) state.pending = stash ? stash.pending : [];
        state.labelOrder = json.categories.map(c => c.name);
        state.labelCats = json.categories;
        state.labelTree = json.tree || [];
        buildParentOptions();
        const byId = new Map(json.categories.map(c => [c.id, c.name]));
        state.saved = json.saved.map(a => ({
          annotationId: a.annotation_id,
          label: byId.get(a.category_id) || ('category ' + a.category_id),
          bboxXyxy: a.bbox_xyxy, maskImg: null, score: null, visible: true,
          maskRect: a.mask_rect,
          rle: a.rle || null,  // null for polygons: no touch-up, no dedupe
        }));
        if (stash && stash.stagedIds.length) {
          const stagedIds = new Set(stash.stagedIds);
          for (const s of state.saved) {
            if (stagedIds.has(s.annotationId)) s.pendingDelete = true;
          }
        }
        if (!keep) searchLog = stash ? stash.searchLog : [];
        markOverlayDirty();  // the old image's composited masks are invalid now
        selected = null;
        json.saved.forEach((a, i) => decodeMaskInto(state.saved[i], a.mask_png));
        state.prev = json.prev; state.next = json.next;
        state.searchIndex = json.search_index || null;
        state.searchIndexStatus = json.search_index_status || 'missing';
        $('prev').disabled = !state.prev;
        $('next').disabled = !state.next;
        sweepFocus = null;
        sweepGrid = null;
        rq.queued = [];  // in-flight boxes stay counted until their stream ends
        updateRepromptUi();
        $('dims').textContent = state.imgW + ' x ' + state.imgH + ' | '
          + (state.imgW * state.imgH / 1e6).toFixed(1) + ' MP';
        $('imgidx').textContent = '';
        fetchDirMeta(p).then(m => {
          if (token !== loadReq) return;  // a newer load owns the counter
          dirMeta = m;
          if (m && m.index != null) {
            $('imgidx').textContent = (m.index + 1) + ' / ' + m.count;
          }
          renderMaskPill(m);
          renderFilmstrip(m);
          updateBatchUi();  // the start-button image count follows the listing
        });
        prefetchLoad(json.prev);
        prefetchLoad(json.next);
        cropReq.n++;  // drops a crop response still in flight
        const img = new Image();
        img.onload = () => {
          if (token !== loadReq) return;  // a newer load owns the canvas
          state.previewImg = img;
          viewer.setImage(img, state.imgW, state.imgH);
          sizeCanvas();
          renderAll();
        };
        img.src = json.preview;
        say(state.viewOnly
          ? 'Zephyr mask - view only; prompts and saving are disabled.'
          : (state.cocoFile
            ? 'Annotations: ' + state.cocoFile + ' - ' + json.saved.length + ' saved on this image'
            : 'No COCO annotations in this directory; saving creates ' + json.new_coco_name + '.')
            + (state.searchIndexStatus === 'invalid'
               ? ' - image content changed; previous searches ignored.' : ''));
        renderAll();
        updateLabelPanel();
        return true;
      } catch (err) {
        sayError('Error: ' + err);
        return false;
      }
    }

    async function navigate(target) {
      if (!target || busy || lockBlock()) return;
      $('path').value = target;
      await loadImage();
    }

    // --- model pace + sweep strip progress ----------------------------------

    // EWMA of a duration sample, persisted across page loads.
    function makePace(key) {
      const estimate = () => parseFloat(localStorage.getItem(key)) || 0;
      const record = (seconds) => {
        if (!(seconds > 0)) return;
        const prev = estimate();
        const next = prev ? PACE_ALPHA * seconds + (1 - PACE_ALPHA) * prev : seconds;
        localStorage.setItem(key, next.toFixed(3));
      };
      return { estimate, record };
    }
    // EWMA of one segmentation pass's model time
    const passPace = makePace('masker.pass_seconds_ewma');
    // one batch image's wall time (load + sweeps + dedupe + save)
    const imagePace = makePace('masker.image_seconds_ewma');

    function etaLeft() {
      const est = passPace.estimate();
      if (!est || !prog.active) return '';
      const left = Math.max(0, (prog.total - prog.done) * est);
      return ' | ~' + Math.round(left) + 's left';
    }

    // Best-effort seconds per batch image: the wall-time EWMA once any image
    // has completed; before that, the per-forward EWMA times the running
    // sweep's queued forwards times the prompt count.
    function perImageSeconds() {
      const wall = imagePace.estimate();
      if (wall) return wall;
      const pass = passPace.estimate();
      if (pass && prog.active && prog.total) {
        return pass * prog.total * Math.max(1, getHistory().length);
      }
      return 0;
    }
    function batchEtaText() {
      const per = perImageSeconds();
      if (!per) return 'estimating...';
      const left = Math.max(0, (batch.queue.length - batch.pos) * per);
      return '~' + Math.round(left) + 's left';
    }

    const prog ={ active: false, total: 0, done: 0, passStart: 0, raf: 0 };
    function progressFrame() {
      const est = passPace.estimate();
      const inflight = est
        ? Math.min((performance.now() - prog.passStart) / 1000 / est, 0.98) : 0;
      const frac = Math.min(1, (prog.done + inflight) / prog.total);
      $('sweepbar').style.width = (frac * 100).toFixed(1) + '%';
      prog.raf = requestAnimationFrame(progressFrame);
    }
    function progressBegin(total) {
      prog.active = true;
      prog.total = total;
      prog.done = 0;
      prog.passStart = performance.now();
      show('sweepstrip', true, 'flex');
      progressFrame();
    }
    function progressPass() {
      if (!prog.active) return;
      prog.done++;
      prog.passStart = performance.now();
    }
    // Mid-run total revision: a sweep's pass-2 forward count is unknown
    // until its regroup event.
    function progressExtend(n) {
      if (!prog.active || !(n > 0)) return;
      prog.total += n;
    }
    function progressEnd() {
      if (!prog.active) return;
      prog.active = false;
      cancelAnimationFrame(prog.raf);
      show('sweepstrip', false);
      $('sweeptext').textContent = '';
      $('sweepcounts').textContent = '';
      $('sweepbar').style.width = '0';
    }

    function cancelSweep() {
      sweepCancel = true;  // also stops a run-all loop between sweeps
      if (sweepAbort) sweepAbort.abort();
    }
    $('sweepcancel').addEventListener('click', cancelSweep);

    // --- prompting ----------------------------------------------------------

    function addPending(label, inst, origin) {
      const entry = {
        tmpId: ++tmpCounter, label: label, score: inst.score,
        bboxXyxy: inst.bbox_xyxy, maskImg: null, rle: inst.rle, visible: true,
        centroid: inst.centroid || null,  // mask centroid (original px)
        maskRect: inst.mask_rect,  // rect the mask PNG spans (original px)
        origin: origin || null,  // sweep pass (segment_contents/reprompt/exemplar)
      };
      decodeMaskInto(entry, inst.mask_png);
      state.pending.push(entry);
      return entry;
    }

    // A stream event's instances land as pending under label unless the
    // page moved on from forImage; the pace records either way. Returns
    // the entries added.
    function landInstances(ev, label, origin, forImage) {
      passPace.record(ev.seconds);
      if (state.imagePath !== forImage) return [];
      const entries = ev.instances.map(inst => addPending(label, inst, origin));
      throttledRenderAll();
      return entries;
    }

    // Zoomed box/point prompts see only the visible region: the server crops
    // it from the original at up to full resolution and returns results in
    // original-image coordinates.
    function promptRect() {
      return viewer.zoom <= ZOOMED_IN ? null : viewRect();
    }

    async function segment(body, label) {
      if (viewOnlyBlock() || lockBlock() || busy) return;
      busy = true;
      const rect = promptRect();
      const runningMsg = 'Segmenting (' + body.prompt_type + ': ' + label + ')'
        + (rect ? ' in the visible region' : '') + '...';
      say(runningMsg);
      const stopModelWatch = startModelWatch(runningMsg);
      try {
        const req = { image_path: state.imagePath, ...body };
        if (rect) req.recrop = rect;
        const json = await call('/segment', req);
        if (!json) return;
        passPace.record(json.stats.generation_seconds);
        for (const inst of json.instances) addPending(label, inst);
        say(json.instances.length + ' instance(s) of "' + label + '" in '
          + json.stats.generation_seconds + 's'
          + (rect ? ' (visible region)' : '') + ' - review, then save.');
        renderAll();
      } catch (err) {
        sayError('Error: ' + err);
      } finally {
        stopModelWatch();
        busy = false;
      }
    }

    function currentLabel() {
      const label = $('promptText').value.trim();
      if (!label) say('Enter a prompt/label first.');
      return label;
    }

    $('run').addEventListener('click', () => {
      if (ui.mode !== 'text') { say('Box/point modes run from the image itself.'); return; }
      const label = currentLabel();
      if (!label) return;
      runSweep(label, { interactive: true }).then(queueSearchRecords);
    });
    // Stage the term as a chip without a sweep; chips feed Run all and the
    // batch job.
    $('addprompt').addEventListener('click', () => {
      if (ui.mode !== 'text') return;
      const label = currentLabel();
      if (!label) return;
      rememberPrompt(label);
      $('promptText').value = '';
      say('"' + label + '" staged - appears in Run all and batch.');
    });
    $('promptText').addEventListener('keydown', (e) => {
      if (e.key !== 'Enter') return;
      if (e.shiftKey) $('addprompt').click();
      else $('run').click();
    });

    async function runAllPrompts() {
      const terms = getHistory();
      if (!terms.length || !state.imagePath) return;
      for (const term of terms) {
        queueSearchRecords(await runSweep(term));
        if (sweepCancel) break;  // cancel stops the batch too
      }
    }
    $('runall').addEventListener('click', runAllPrompts);

    // --- sweep: full-image recrop segmentation (/sweep) ---------------------

    // one reader per input type; a blank, unparsable, or out-of-server-range
    // value falls back to the served default
    const knob = (el, def, ok, parse = parseFloat) => {
      const v = parse(el.value);
      return Number.isFinite(v) && ok(v) ? v : def;
    };
    const parseIntBase10 = s => parseInt(s, 10);
    const intKnob = (el, def, ok) => knob(el, def, ok, parseIntBase10);
    const unit = v => v > 0 && v <= 1;  // the server's (0, 1] knobs

    const bandOn = () => {
      const lo = parseFloat($('sweepbandlo').value), hi = parseFloat($('sweepbandhi').value);
      return 0 < lo && lo < hi && hi <= 1;
    };

    // one row per knob: el, payload key, seed (served default; absent stays
    // blank), hint (title suffix; derived from seed unless given), val
    // (payload value), gate (checkbox graying el)
    const D = SWEEP_DEFAULTS;
    const num = (el, key, ok, gate, parse) =>
      ({ el, key, gate, seed: D[key], val: () => knob(el, D[key], ok, parse) });
    const int = (el, key, ok, gate) => num(el, key, ok, gate, parseIntBase10);
    const flag = (el, key) => ({ el, key, seed: D[key], val: () => el.checked });
    const KNOBS = [
      num($('sweepdownscale'), 'downscale', v => v > 0),
      num($('sweepoverlap'), 'overlap', v => v >= 0 && v < 1),
      { el: $('sweeppack'), key: 'pack', hint: 'off',
        val: () => ($('sweeppack').value ? parseInt($('sweeppack').value, 10) : D.pack) },
      num($('sweepexpand'), 'reprompt_expand', v => v >= 1),
      flag($('sweepiosgate'), 'ios_gate'),
      { el: $('sweepbandlo'), key: 'band_lo',
        val: () => (bandOn() ? parseFloat($('sweepbandlo').value) : D.band_lo) },
      { el: $('sweepbandhi'), key: 'band_hi',
        val: () => (bandOn() ? parseFloat($('sweepbandhi').value) : D.band_hi) },
      int($('sweepbackoff'), 'backoff_retries', v => v >= 0),
      num($('sweepbackoffscale'), 'backoff_scale', v => v > 0),
      num($('sweepdupios'), 'dup_ios', unit, $('sweepiosgate')),
      num($('sweepdupiou'), 'dup_iou', unit),
      flag($('sweepnovel'), 'novel'),
      { el: $('sweepnovelgap'), key: 'novel_gap_px', gate: $('sweepnovel'),
        val: () => knob($('sweepnovelgap'), D.novel_gap_px, v => v >= 0) },
      { el: $('sweepcentroid'), key: 'centroid', seed: D.centroid,
        val: () => $('sweepcentroid').value || D.centroid },
      // the toggle gates the count: unchecked sends 0 (off)
      { el: $('sweeprepromptexemplar'), key: 'reprompt_exemplars',
        seed: D.reprompt_exemplars > 0,
        val: () => ($('sweeprepromptexemplar').checked ? intKnob(
          $('sweeprepromptexemplarmax'), D.reprompt_exemplars || 1, v => v >= 1) : 0) },
      { el: $('sweeprepromptexemplarmax'), seed: D.reprompt_exemplars || 1,
        gate: $('sweeprepromptexemplar') },
      flag($('sweepexemplar'), 'exemplar'),
      num($('sweepexemplarfrac'), 'exemplar_frac', v => v > 0 && v < 1, $('sweepexemplar')),
      int($('sweepexemplarmax'), 'exemplar_max', v => v >= 1, $('sweepexemplar')),
      int($('sweepexemplarbackoff'), 'exemplar_backoff', v => v >= 0, $('sweepexemplar')),
      { el: $('sweepexemplarscales'), key: 'exemplar_scales', gate: $('sweepexemplar'),
        seed: D.exemplar_scales.join(','),
        val: () => {
          const s = $('sweepexemplarscales').value.split(',').map(Number);
          return s.every(v => Number.isFinite(v) && v > 0) && s.length
            ? s : D.exemplar_scales;
        } },
      // the server accepts 9 variant offsets
      int($('sweepexemplarvariants'), 'exemplar_variants', v => v >= 1 && v <= 9,
          $('sweepexemplar')),
      num($('sweepexemplarcover'), 'exemplar_cover', unit, $('sweepexemplar')),
    ];

    function sweepParams() {
      const out = {};
      for (const k of KNOBS) if (k.key) out[k.key] = k.val();
      return out;
    }

    // a completed sweep record whose stored params equal these knobs; a
    // record swept with other knobs never blocks a run. Stored params come
    // back with sorted keys, so both sides canonicalize.
    const canonParams = (o) => JSON.stringify(o, Object.keys(o).sort());
    function findSweepRecord(label, params) {
      const p = normPrompt(label), want = canonParams(params);
      return indexRecords().find(r => r.prompt === p && r.completed === true
        && canonParams(r.params) === want) || null;
    }

    // Full-image text sweep; returns null when refused, else
    // { records, kept } for queueSearchRecords.
    async function runSweep(label, opts = {}) {
      if (!state.imagePath || busy) return null;
      if (viewOnlyBlock() || lockBlock()) return null;
      const params = sweepParams();
      // replace = redo: recorded searches don't block a sweep meant to
      // supersede their annotations; an interactive run confirms the re-run
      const replaceOn = opts.replace || $('replaceexisting').checked;
      if (!$('ignoreprev').checked && !replaceOn
          && findSweepRecord(label, params)
          && !(opts.interactive
               && confirm('A sweep with these settings already ran for "'
                          + label + '".\nRe-run it anyway?'))) {
        say('Sweep already recorded for "' + label
          + '" - tick "ignore previous searches" to re-run.');
        return { records: [], kept: 0 };
      }
      busy = true;  // before the first await
      sweepCancel = false;
      $('run').classList.add('busy');
      $('run').textContent = 'Running...';
      const forImage = state.imagePath;  // late stream events must not land elsewhere
      let added = 0, done = 0, total = 0, ambiguous = 0, clusters = 0,
          reprompted = 0, exRecrops = 0, exDone = 0,
          sawDone = false, completed = false;
      // packed pass 2 sends pack^2 recrops per forward: progress counts canvases
      const per = params.pack ? params.pack * params.pack : 1;
      const body = { image_path: forImage, text: label, params };
      // replace = redo: a completed sweep supersedes saved instances of this label;
      // only this run's staging is restored on cancel or failure
      const replacing = [];
      if (replaceOn) {
        for (const s of state.saved) {
          if (s.label === label && s.annotationId != null && !s.pendingDelete) {
            s.pendingDelete = true;
            replacing.push(s);
          }
        }
        if (replacing.length) renderDirty();
      }
      // entries keyed 'kind:index' (segment_contents/reprompt/exemplar);
      // replace events retract by key
      const landed = new Map();
      const land = (ev, key) => {
        const entries = landInstances(ev, label, key.split(':')[0], forImage);
        added += entries.length;
        landed.set(key, entries);
      };
      const retract = (origins) => {
        for (const [kind, i, pos] of origins) {
          const entry = (landed.get(kind + ':' + i) || [])[pos];
          if (!entry) continue;
          const at = state.pending.indexOf(entry);
          if (at >= 0) { state.pending.splice(at, 1); added--; }
        }
        markOverlayDirty();
        throttledRenderAll();
      };
      const strip = (text, counts) => {
        $('sweeptext').textContent = label + ' | ' + text;
        if (counts != null) $('sweepcounts').textContent = counts + etaLeft();
      };
      const handleEvent = (ev) => {
        if (ev.type === 'start') {
          total = ev.total;
          sweepGrid = ev.recrops ? { rects: ev.recrops, stats: new Map() } : null;
          // priced at pass 1 only; regroup revises when pass 2 exists
          progressBegin(total);
        } else if (ev.type === 'recrop_stats') {
          if (sweepGrid) sweepGrid.stats.set(ev.index, ev);
          if (ui.showPresence) scheduleRedraw();
        } else if (ev.type === 'batch' || ev.type === 'rebatch') {
          sweepFocus = ev.recrops;
          scheduleRedraw();
        } else if (ev.type === 'segment_contents') {
          land(ev, 'segment_contents:' + ev.index);
          progressPass();
          done++; ambiguous += ev.ambiguous;
          strip('recrop ' + done + '/' + total,
                added + ' accepted | ' + ambiguous + ' ambiguous');
        } else if (ev.type === 'regroup') {
          clusters = ev.clusters;
          // the cluster count is known now: revise the total upward by the
          // pass-2 forwards (canvases when packed; zero clusters, no change)
          progressExtend(Math.ceil(clusters / per));
          strip(clusters ? 're-prompting ' + clusters + ' cluster(s)...' : 'nothing ambiguous');
        } else if (ev.type === 'tighten' || ev.type === 'tighten3') {
          // a forward finished oversized; its shrunken retry is one more
          progressExtend(1);
          progressPass();
          strip(ev.type === 'tighten'
            ? 'tightening recrop for cluster ' + (ev.cluster + 1) + '...'
            : 'tightening exemplar recrop ' + (ev.index + 1) + '...');
        } else if (ev.type === 'reprompt') {
          land(ev, 'reprompt:' + ev.cluster);
          reprompted++;
          // a pass-2 forward completes every per clusters (+ the ragged tail)
          if (reprompted % per === 0 || reprompted === clusters) progressPass();
          strip('cluster ' + reprompted + '/' + clusters, added + ' instance(s) total');
        } else if (ev.type === 'replace' || ev.type === 'replace3') {
          // a later pass replaced fragments an earlier pass kept; the
          // replacement landed with the preceding reprompt/exemplar event
          retract(ev.origins);
        } else if (ev.type === 'explan') {
          exRecrops = ev.recrops.length;
          progressExtend(exRecrops);
          sweepFocus = ev.recrops;
          scheduleRedraw();
          strip(exRecrops ? 'exemplar pass: ' + exRecrops + ' recrop(s)...'
                          : 'exemplar pass: nothing to prompt');
        } else if (ev.type === 'exemplar') {
          land(ev, 'exemplar:' + ev.index);
          exDone++;
          progressPass();
          strip('exemplar recrop ' + exDone + '/' + exRecrops, added + ' instance(s) total');
        } else if (ev.type === 'done') {
          sawDone = true;
        }
      };
      try {
        sweepAbort = new AbortController();
        const failed = await runStream('/sweep', body, handleEvent,
                                       'Sweeping "' + label + '"...', sweepAbort.signal);
        completed = failed === null && sawDone && !sweepCancel;
        if (failed !== null) {
          sayError('Error: ' + failed);
        } else {
          rememberPrompt(label);
          say(added + ' pending "' + label + '" instance(s) from the '
            + (completed ? 'sweep' : 'cancelled sweep')
            + (replacing.length && completed
               ? '; replacing ' + replacing.length + ' saved (deleted on save)'
               : '')
            + ' - review, then save.');
        }
      } finally {
        busy = false;
        sweepAbort = null;
        $('run').classList.remove('busy');
        $('run').textContent = 'Run';
        progressEnd();
        // pass colors are a live-progress aid; afterwards boxes take the label color
        for (const entries of landed.values()) {
          for (const e of entries) e.origin = null;
        }
        scheduleRedraw();
        if (replacing.length && !completed) {
          // cancelled or failed: restore the staging this run applied
          for (const s of replacing) s.pendingDelete = false;
          renderDirty();
          say('Replace undone; saved "' + label + '" annotations kept.');
        }
      }
      // only a full pass (done event, no error, not cancelled) records the
      // sweep; partial sweeps are not resumable
      const records = completed
        ? [{ prompt: normPrompt(label), completed: true, params: params, instances: added }]
        : [];
      return { records, kept: added };
    }

    // --- refine (reprompt by bbox) ------------------------------------------

    // Boxes queue without blocking busy; a drain sends every queued box of
    // one label and its detections replace the instances centered there.
    function updateRepromptUi() {
      if (rq.queued.length || rq.inflight.length) {
        say('refine queue: ' + rq.queued.length + ' queued, '
          + rq.inflight.length + ' in flight');
      }
      scheduleRedraw();
    }

    // a refine box reuses the dominant class inside it; the prompt field is
    // the fallback for boxes over unlabeled ground
    function dominantLabelIn(box) {
      const counts = new Map();
      for (const inst of allInstances()) {
        if (inst.pendingDelete || !inst.visible) continue;
        if (centerInBox(inst, box)) counts.set(inst.label, (counts.get(inst.label) || 0) + 1);
      }
      let best = null, bestN = 0;
      for (const [label, n] of counts) if (n > bestN) { best = label; bestN = n; }
      return best;
    }

    function refineBox(box) {
      const label = dominantLabelIn(box) || $('promptText').value.trim();
      if (!label) { say('No instances in the box and no prompt to re-run.'); return; }
      queueReprompt(box, label);
    }

    function queueReprompt(box, label) {
      rq.queued.push({ bbox: box, label: label });
      updateRepromptUi();
      drainRepromptQueue();
    }

    async function sendRepromptBatch() {
      const label = rq.queued[0].label;
      const batchBoxes = rq.queued.filter(q => q.label === label);
      rq.queued = rq.queued.filter(q => q.label !== label);
      rq.inflight = rq.inflight.concat(batchBoxes);
      updateRepromptUi();
      const forImage = state.imagePath;  // late stream events must not land elsewhere
      const body = { image_path: forImage, text: label,
                     bboxes: batchBoxes.map(q => q.bbox),
                     params: sweepParams() };  // advanced knobs govern Refine too
      let added = 0;
      const handleEvent = (ev) => {
        if (ev.type === 'rebatch') {
          sweepFocus = ev.recrops;
          scheduleRedraw();
        } else if (ev.type === 'reprompt') {
          const q = batchBoxes[ev.index];
          rq.inflight = rq.inflight.filter(x => x !== q);
          updateRepromptUi();
          if (state.imagePath === forImage) {
            // the instances centered in the box give way to the re-detections
            dropInstances(i => i.label === label && centerInBox(i, q.bbox));
            markOverlayDirty();
          }
          added += landInstances(ev, label, null, forImage).length;
        }
      };
      const failed = await runStream('/reprompt_bboxes', body, handleEvent,
                                     'Refining "' + label + '"...');
      // boxes the stream never resolved are dropped unreplaced; redraw them
      rq.inflight = rq.inflight.filter(x => !batchBoxes.includes(x));
      updateRepromptUi();
      if (failed !== null) {
        sayError('Refine error: ' + failed);
      } else if (state.imagePath === forImage) {
        say('Refine: ' + added + ' "' + label
          + '" instance(s) replaced the boxed fragments - review, then save.');
      }
    }

    async function drainRepromptQueue() {
      if (rq.draining) return;
      rq.draining = true;
      try {
        while (rq.queued.length) {
          if (busy) {  // a sweep or prompt owns the model; retry shortly
            await sleep(300);
            continue;
          }
          busy = true;
          try { await sendRepromptBatch(); } finally { busy = false; }
        }
      } finally {
        rq.draining = false;
        updateRepromptUi();
      }
    }

    // --- erase --------------------------------------------------------------

    // Erase box: pending instances centered inside are dropped, saved ones
    // staged for delete (struck through in the list; Save commits).
    function eraseInBox(box) {
      selected = null;
      if (!dropAndReport(i => centerInBox(i, box) && i.visible && labelShown(i), 'Erased')) {
        say('No instances in the box.');
      }
    }

    // --- instance delete / save / export ------------------------------------

    // 409: the annotation reference went stale; resync saved state from disk
    async function resyncStale(json) {
      sayError(json.error + ' - reloading.');
      invalidateLoadCache();
      await loadImage({ keepPending: true });
    }

    async function deleteInstance(inst) {
      if (lockBlock()) return;
      if (inst.annotationId == null) {
        state.pending = state.pending.filter(p => p !== inst);
        renderDirty();
        return;
      }
      const json = await call('/delete_annotation',
        { coco_file: state.cocoFile, annotation_id: inst.annotationId,
          image_path: state.imagePath, bbox_xyxy: inst.bboxXyxy });
      if (!json) return;
      state.cocoFile = json.coco_file;
      state.saved = state.saved.filter(s => s !== inst);
      invalidateLoadCache();
      say('Deleted annotation #' + inst.annotationId + '.');
      renderDirty();
    }

    // Commits ticked pending and staged deletes; true on success (a no-op counts).
    let saveInFlight = false;
    async function saveNow() {
      if (viewOnlyBlock() || lockBlock()) return false;
      if (busy) {  // replace/redo staging is live while a sweep runs
        say('Wait for the current run to finish before saving.');
        return false;
      }
      if (saveInFlight) {
        say('Save already in progress.');
        return false;
      }
      const sent = state.pending.filter(p => p.visible);
      const deletes = state.saved.filter(s => s.pendingDelete)
        .map(s => ({ annotation_id: s.annotationId, bbox_xyxy: s.bboxXyxy }));
      if (!sent.length && !deletes.length) {
        if (state.pending.length) {  // everything unticked: local discard
          const n = state.pending.length;
          state.pending = [];
          renderDirty();
          say('Discarded ' + n + ' unticked pending instance(s).');
        }
        return true;
      }
      saveInFlight = true;
      try {
        const json = await call('/save', {
          image_path: state.imagePath,
          coco_file: state.cocoFile,
          instances: sent.map(p => ({ label: p.label, rle: p.rle })),
          deletes: deletes,
        });
        if (!json) return false;
        state.cocoFile = json.coco_file;
        json.saved.forEach((a, i) => {
          // the same object moves pending -> saved; rows, the pin and decodes keep it
          const p = sent[i];
          p.annotationId = a.annotation_id;
          p.bboxXyxy = a.bbox_xyxy;
          p.score = null;
          delete p.tmpId;
          state.saved.push(p);
        });
        state.saved = state.saved.filter(s => !s.pendingDelete);
        const discarded = state.pending.length - sent.length;
        state.pending = [];
        imgStore.delete(state.imagePath);
        invalidateLoadCache();
        renderDirty();  // discarded and deleted instances leave the overlay
        await flushSearchLog();
        say('Saved ' + sent.length + ' instance(s)'
          + (deletes.length ? ', deleted ' + deletes.length : '')
          + (discarded ? ', discarded ' + discarded + ' unticked' : '')
          + ' - ' + state.cocoFile);
        return true;
      } catch (err) {
        sayError('Error: ' + err);
        return false;
      } finally {
        saveInFlight = false;
      }
    }
    $('save').addEventListener('click', saveNow);

    $('export').addEventListener('click', (e) => toggleMenu($('exportmenu'), e));

    // --- Zephyr mask export (union of chosen labels -> _mask.tiff) ----------

    async function exportZephyr(labels, missingOk, imagePath, force) {
      const json = await call('/export_zephyr',
        { image_path: imagePath || state.imagePath, labels: labels,
          missing_ok: !!missingOk, force: !!force });
      if (!json) return;
      toast('Zephyr masks: ' + json.written + ' written, ' + json.skipped + ' skipped');
      say('Zephyr masks written beside the images in ' + json.dir + '.');
    }
    $('exportzephyr').addEventListener('click', async () => {
      closeMenus();
      if (noImageBlock()) return;
      const r = await getPath('/dir_counts', state.imagePath);
      if (!r.ok) { sayError('Error: ' + r.json.error); return; }
      if (!r.json.categories.length) { say('No saved annotations in this directory.'); return; }
      const box = $('zephyrlabels');
      box.innerHTML = '';  // per-run selection: rebuilt on every open
      for (const c of r.json.categories) {
        const lab = el('label', '', null, box);
        const cb = el('input', '', null, lab);
        cb.type = 'checkbox';
        cb.value = c.name;
        lab.appendChild(document.createTextNode(' ' + c.name));
      }
      showModal('zephyrmodal', true);
    });
    $('zephyrcancel').addEventListener('click', () => showModal('zephyrmodal', false));
    $('zephyrok').addEventListener('click', () => {
      const labels = [...$('zephyrlabels').querySelectorAll('input:checked')]
        .map(i => i.value);
      if (!labels.length) { say('Check at least one label.'); return; }
      showModal('zephyrmodal', false);
      exportZephyr(labels, false, null, $('zephyrforce').checked);
    });

    // --- redo modal (per-class replace on this image) -----------------------

    let redoLabel = null;
    function openRedo(label) {
      redoLabel = label;
      const nSaved = liveCount(label, state.saved);
      const nPend = liveCount(label, state.pending);
      $('redotitle').textContent =
        'Redo "' + label + '" from scratch';
      $('redobody').textContent =
        'Drops ' + nPend + ' pending and stages ' + nSaved + ' saved "' + label
        + '" instance(s) on ' + (state.imagePath ? basename(state.imagePath) : 'this image')
        + ', then re-runs the prompt as a full-image sweep.';
      showModal('redomodal', true);
    }
    function closeRedo() {
      redoLabel = null;
      showModal('redomodal', false);
    }
    $('redocancel').addEventListener('click', closeRedo);
    $('redook').addEventListener('click', () => {
      const label = redoLabel;
      closeRedo();
      if (!label || busyBlock()) return;
      state.pending = state.pending.filter(p => p.label !== label);
      renderDirty();
      // replace stages the saved instances and bypasses recorded searches;
      // cancel or failure restores the staging
      runSweep(label, { replace: true }).then(queueSearchRecords);
    });

    // --- batch over the directory -------------------------------------------

    // Image-major loop over a queue snapshot: per image, sweep the
    // selected prompts, dedupe, save, record, advance.
    const batch = {
      state: 'idle',  // idle | running | pausing | cancelling | paused | done
      visible: false, dir: null,
      queue: [],    // absolute image paths, snapshotted at start
      pos: 0,       // queue index of the image being processed
      pickedMeta: null,  // /fsmeta of a Browse-picked directory
      counts: {},   // prompt -> instances kept across the run
      images: [],   // {path, name, counts, thumb} per completed image
      zephyr: false,  // write Zephyr masks when the run completes
      terms: [],    // selected-prompt snapshot at start; the Zephyr label selection
    };
    // inverse selection: unticked prompts
    const batchDesel = new Set();
    function batchSelectedTerms() {
      return getHistory().filter(t => !batchDesel.has(t));
    }

    // Server-supplied and typed paths may spell one Windows file two ways;
    // compare with slashes and case normalized.
    function samePath(a, b) {
      return !!a && !!b
        && a.replace(/\//g, '\\').toLowerCase() === b.replace(/\//g, '\\').toLowerCase();
    }

    // Extension filter: "arw, .JPG" -> lowercase dotted suffixes; null = all.
    function batchExtFilter() {
      const parts = $('bext').value.trim().split(/[\s,]+/).filter(Boolean);
      return parts.length
        ? new Set(parts.map(e => '.' + e.replace(/^\./, '').toLowerCase())) : null;
    }
    // Extension filter then the MAX IMAGES cap; blank cap = all.
    function batchSelectNames(names) {
      const exts = batchExtFilter();
      const kept = exts
        ? names.filter(n => exts.has(n.slice(n.lastIndexOf('.')).toLowerCase())) : names;
      const cap = parseInt($('bmax').value, 10);
      return cap > 0 ? kept.slice(0, cap) : kept;
    }

    async function buildBatchQueue() {
      if (batch.pickedMeta) {
        const m = batch.pickedMeta;
        return batchSelectNames(m.images).map(n => listedPath(m, n));
      }
      // fresh listing: dirMeta can lag the loaded image
      const m = await fetchDirMeta(state.imagePath);
      if (!m || m.index == null) return null;
      const from = $('bwholedir').checked ? 0 : m.index;
      return batchSelectNames(m.images.slice(from)).map(n => listedPath(m, n));
    }

    // Per label, drop pending that duplicate a saved anchor and union
    // overlapping pending groups.
    async function batchDedupe() {
      const byLabel = new Map();
      for (const p of state.pending) {
        if (!p.rle || !p.visible) continue;
        if (!byLabel.has(p.label)) byLabel.set(p.label, []);
        byLabel.get(p.label).push(p);
      }
      for (const [label, cands] of byLabel) {
        try {
          const anchors = state.saved.filter(
            s => s.label === label && s.rle && !s.pendingDelete);
          const candidates = cands.filter(c => state.pending.includes(c));
          if (candidates.length + anchors.length < 2) continue;
          const all = anchors.concat(candidates);
          const d = await postJson('/dedup_instances',
            { image_path: state.imagePath,
              rles: all.map(i => i.rle),
              // saved anchors have no score; the server treats null as top
              // priority
              scores: all.map(i => i.score != null ? i.score : null),
              iou: knob($('batchdedupiou'), DEDUP_IOU, unit) });
          if (!d.ok) continue;
          const nA = anchors.length;
          for (const g of d.json.groups) {
            const pend = g.indices.filter(i => i >= nA).map(i => candidates[i - nA]);
            if (!pend.length) continue;
            // a group with a saved anchor drops its pending (never the saved);
            // a pending-only group of two or more becomes its union
            const anchored = g.indices.some(i => i < nA);
            if (!anchored && g.indices.length < 2) continue;
            const members = new Set(pend);
            state.pending = state.pending.filter(p => !members.has(p));
            if (!anchored) {
              const score = Math.max(...pend.map(m => m.score != null ? m.score : 0));
              addPending(label, { score: score, bbox_xyxy: g.bbox_xyxy,
                                  rle: g.rle, mask_png: g.mask_png,
                                  mask_rect: g.mask_rect });
            }
          }
        } catch (err) { /* best-effort; raw detections stay */ }
      }
      renderDirty();
    }

    // run id, bumped on Start; awaiting code from an older run exits on mismatch
    let batchRun = 0;

    // Distinguishes a killed server (Ctrl+C in its terminal) from a failing
    // request; any reachable server resolves fetch, even on an error status.
    async function serverGone() {
      try { await fetch('/model_status'); return false; }
      catch (err) { return true; }
    }

    // Ctrl+C on the server reads as a cancel: completed images' saves are on
    // disk; the current image's pending stays in the browser for review.
    function batchServerLost() {
      batch.state = 'idle';
      toast('Server stopped - batch cancelled; saved annotations are kept, '
        + "this image's pending left for review");
    }

    async function batchProcessCurrentImage(run = batchRun) {
      const img = state.imagePath;
      const name = basename(img);
      const perImage = {};
      for (const term of batch.terms) {
        if (batch.state !== 'running' || batchRun !== run) break;  // pausing/superseded
        try {
          let res = null;
          for (;;) {
            while (busy) {  // an interactive prompt owns the model; wait it out
              await sleep(300);
              if (batch.state !== 'running' || batchRun !== run) break;
            }
            if (batch.state !== 'running' || batchRun !== run) break;
            res = await runSweep(term);
            // null with busy still held: drainRepromptQueue grabbed the model
            // first; wait and retry
            if (res || !busy) break;
          }
          if (!res) continue;  // refused (view-only or no image), or stopping
          batch.counts[term] = (batch.counts[term] || 0) + res.kept;
          perImage[term] = (perImage[term] || 0) + res.kept;
          await queueSearchRecords(res);
        } catch (err) { /* one prompt's failure must not stop the image */ }
      }
      if (batch.state === 'idle' || batchRun !== run) return true;  // superseded: nothing saved
      if (state.pending.length) await batchDedupe();
      // runs even when pausing or cancelling: save what landed, record the
      // searches
      if (await saveNow()) {
        await flushSearchLog();  // covers the nothing-pending case
      } else {
        if (await serverGone()) { batchServerLost(); return false; }
        // stop without advancing; the pending instances stay on this image
        toast(name + ': save failed', 4000);
        return false;
      }
      if (batchRun !== run) return true;  // a new run owns batch.images now
      batch.images.push({ path: img, name: name, counts: perImage,
                          thumb: ImageProxy.url(img, 256) });
      if (batch.visible) renderBatchRun();
      updatePill();
      return true;
    }

    async function batchLoop(run = batchRun) {
      updateBatchUi();
      while (batch.state === 'running' && batchRun === run
             && batch.pos < batch.queue.length) {
        const target = batch.queue[batch.pos];
        if (!samePath(state.imagePath, target)) {
          if (busy) {  // an interactive run owns the model; loadImage refuses
            await sleep(300);
            continue;
          }
          $('path').value = target;
          const ok = await loadImage();
          if (batchRun !== run) return;  // cancelled/superseded during the load
          if (!ok) {
            if (busy) continue;  // refused: lost the model race; retry, not skip
            if (await serverGone()) { batchServerLost(); break; }
            toast(basename(target) + ': failed to load - skipped', 4000);
            batch.pos++;
            continue;
          }
        }
        const imageStart = performance.now();
        try {
          const cont = await batchProcessCurrentImage(run);
          if (batchRun !== run) return;  // a new run owns the queue position
          if (!cont) {  // save failed; a lost server has already gone idle
            if (batch.state !== 'idle') batch.state = 'paused';  // Resume retries
            break;
          }
          if (batch.state === 'running') {  // pausing cut the image short
            imagePace.record((performance.now() - imageStart) / 1000);
          }
        } catch (err) { /* one image's failure must not stop the batch */ }
        batch.pos++;
      }
      if (batchRun !== run) return;
      if (batch.state === 'running') {
        batch.state = 'done';
        toast('Batch done - ' + batch.images.length + ' image(s)');
        if (batch.zephyr && batch.queue.length) {
          // every annotated label in the directory, not the prompt snapshot;
          // forced rewrite
          const r = await getPath('/dir_counts', batch.queue[0]);
          const labels = r.ok ? r.json.categories.map(c => c.name) : batch.terms;
          await exportZephyr(labels, true, batch.queue[0], true);
        }
      }
      if (batch.state === 'cancelling') {
        batch.state = 'idle';
        toast('Batch cancelled - annotations saved so far are kept');
      }
      if (batch.state === 'pausing') batch.state = 'paused';
      sweepCancel = false;  // don't leak the cancel flag into interactive use
      updateBatchUi();
      updatePill();
      if (batch.visible) renderBatchModal();
    }

    // --- batch modal + pill -------------------------------------------------

    // pause is offered while running or pausing; start waits for cancelling
    // to finish too; setup shows before a run and while paused
    const batchPausable = () => batch.state === 'running' || batch.state === 'pausing';
    const batchLooping = () => batchPausable() || batch.state === 'cancelling';
    const batchSetup = () => batch.state === 'idle' || batch.state === 'paused';

    function openBatch() {
      batch.visible = true;
      renderBatchModal();
      showModal('batchmodal', true);
    }
    function hideBatchModal() {
      batch.visible = false;
      showModal('batchmodal', false);
      updatePill();
    }
    $('batchcard').addEventListener('click', openBatch);
    $('batchpill').addEventListener('click', openBatch);
    $('bclose').addEventListener('click', hideBatchModal);
    $('bcancelsetup').addEventListener('click', hideBatchModal);
    $('bhide').addEventListener('click', hideBatchModal);

    function updatePill() {
      show('batchpill', batch.state !== 'idle' && !batch.visible, 'flex');
      const progress = batch.images.length + '/' + batch.queue.length;
      $('batchpilltext').textContent =
        batch.state === 'done' ? 'batch done'
        : batch.state === 'paused' ? 'batch paused | ' + progress
        : batch.state === 'cancelling' ? 'batch cancelling...'
        : 'batch | ' + progress + ' | ' + batchEtaText();
    }

    // Queued-image count for the start button: the picked directory's total,
    // else current-image-to-end from the loaded directory's listing.
    function startCountSuffix() {
      if (batch.pickedMeta) {
        return ' | ' + batchSelectNames(batch.pickedMeta.images).length + ' images';
      }
      if (dirMeta && dirMeta.index != null && state.imagePath
          && dirMeta.name === basename(state.imagePath)) {
        const from = $('bwholedir').checked ? 0 : dirMeta.index;
        return ' | ' + batchSelectNames(dirMeta.images.slice(from)).length + ' images';
      }
      return '';
    }

    function updateBatchUi() {
      $('batchstart').textContent = batch.state === 'paused'
        ? 'Resume batch' : 'Start batch' + startCountSuffix();
      $('batchstart').disabled = batchLooping()
        || (!state.imagePath && !batch.pickedMeta)
        || !(batch.state === 'paused' ? batch.terms : batchSelectedTerms()).length;
      show('batchpause', batchPausable());
      $('batchpause').textContent = batch.state === 'pausing' ? 'pausing...' : 'Pause';
    }

    function renderBatchModal() {
      const setup = batchSetup();
      show('bsetup', setup, 'flex');
      show('bfoot', setup, 'flex');
      show('brun', !setup, 'flex');
      show('bfootrun', !setup, 'flex');
      $('btitle').textContent =
        setup ? (batch.state === 'paused' ? 'Batch paused' : 'Start batch job')
          : batch.state === 'done' ? 'Batch complete' : 'Batch running';
      $('bsub').textContent = setup
        ? 'Text prompts only - box & point prompts are per-image tools'
        : (batch.dir ? basename(batch.dir) : '') + ' | ' + batch.terms.length + ' prompt(s)';
      if (setup) renderBatchSetup(); else renderBatchRun();
      updateBatchUi();
    }

    function renderBatchSetup() {
      const picked = batch.pickedMeta;
      const dir = picked ? picked.dir
        : state.imagePath ? dirOf(state.imagePath) : '';
      $('bdir').textContent =
        dir || 'load an image or browse to a directory';
      const whole = $('bwholedir').checked;
      $('bdirstats').textContent = picked
        ? batchSelectNames(picked.images).length + ' image(s) | the whole directory'
        : whole ? 'the whole directory'
        : 'from the current image to the end of the directory';
      show('bwholelabel', !picked);
      // Resume continues the snapshotted queue; repointing it is not possible
      $('bbrowse').disabled = batch.state === 'paused';
      $('bwholedir').disabled = batch.state === 'paused';
      $('bext').disabled = batch.state === 'paused';
      $('bmax').disabled = batch.state === 'paused';
      const bp = $('bprompts');
      bp.innerHTML = '';
      const terms = getHistory();
      const selected = batchSelectedTerms();
      const untested = [];
      for (const term of terms) {
        const on = !batchDesel.has(term);
        const n = liveCount(term);
        const recorded = indexRecords().some(
          r => r.completed && r.prompt === normPrompt(term));
        if (on && !n && !recorded) untested.push(term);
        const row = el('div', 'prow', null, bp);
        row.style.opacity = on ? 1 : 0.45;
        const cb = el('input', '', null, row);
        cb.type = 'checkbox';
        cb.checked = on;
        cb.addEventListener('change', () => {
          if (cb.checked) batchDesel.delete(term); else batchDesel.add(term);
          refreshBatchSetup();
        });
        el('span', 'dot9', null, row).style.background = labelColor(term);
        el('span', '', term, row).style.fontWeight = '600';
        el('span', 'ptag ' + ((n || recorded) ? 'ok' : 'warn'),
           n ? 'tested | ' + n + ' hits here' : recorded ? 'recorded' : 'untested',
           row);
      }
      $('bplan').textContent = !terms.length
        ? 'Run at least one text prompt in the workspace first.'
        : !selected.length ? 'Select at least one prompt.'
        : 'Plan: for each image ' + (picked ? 'in ' + basename(picked.dir)
            : whole ? 'in the directory'
            : 'from here to the end of the directory') + ', sweep '
          + selected.join(', ') + ', drop duplicates of saved annotations, save to the '
          + 'COCO file, and record the searches. Recorded searches '
          + 'are skipped, so a resumed batch continues where it stopped.'
          + (untested.length ? ' ! untested: ' + untested.join(', ') : '');
    }

    function refreshBatchSetup() { renderBatchSetup(); updateBatchUi(); }
    $('bselall').addEventListener('click', () => {
      batchDesel.clear();
      refreshBatchSetup();
    });
    $('bselnone').addEventListener('click', () => {
      for (const t of getHistory()) batchDesel.add(t);
      refreshBatchSetup();
    });
    function addBatchPrompt() {
      const term = $('bnewprompt').value.trim();
      if (!term) return;
      rememberPrompt(term);
      batchDesel.delete(term);
      $('bnewprompt').value = '';
      refreshBatchSetup();
    }
    $('bnewadd').addEventListener('click', addBatchPrompt);
    $('bnewprompt').addEventListener('keydown', (e) => {
      if (e.key === 'Enter') addBatchPrompt();
    });
    $('bwholedir').addEventListener('change', refreshBatchSetup);
    $('bext').addEventListener('input', refreshBatchSetup);
    $('bmax').addEventListener('input', refreshBatchSetup);

    function renderBatchRun() {
      const terms = batch.terms;
      const qtotal = batch.queue.length;
      const qdone = Math.min(batch.pos, qtotal);
      $('bpct').textContent = Math.round(qdone / qtotal * 100) + '%';
      $('bprogfill').style.width = (qdone / qtotal * 100).toFixed(1) + '%';
      $('beta').textContent =
        batch.state === 'done' ? 'done'
        : batch.state === 'pausing' ? 'pausing...'
        : batch.state === 'paused' ? 'paused'
        : batchEtaText() + ' | ' + qdone + '/' + qtotal;
      const total = Object.values(batch.counts).reduce((a, x) => a + x, 0);
      $('bline').textContent =
        batch.state === 'done'
          ? 'all images complete | ' + total + ' instances'
          : 'image ' + (batch.images.length + 1) + ' | '
            + (state.imagePath ? basename(state.imagePath) : '');
      // composition bar (each class's share of the instances so far) + stat cards
      const bar = $('bbar');
      bar.innerHTML = '';
      const st = $('bstats');
      st.innerHTML = '';
      terms.forEach(term => {
        const n = batch.counts[term];
        const share = el('div', '', null, bar);
        share.style.flex = '0 0 ' + ((n || 0) / (total || 1) * 100) + '%';
        share.style.background = labelColor(term);
        const d = el('div', 'card', null, st);
        d.style.opacity = n ? 1 : 0.55;
        const h = el('div', 'h', null, d);
        el('span', 'dot9', null, h).style.background = labelColor(term);
        el('span', '', term, h);
        el('div', 'v', n > 0 ? String(n) : '-', d);
        el('div', 's', n
          ? '~' + Math.round(n / Math.max(1, batch.images.length)) + ' / image'
          : 'queued', d);
      });
      const th = $('bthumbs');
      th.innerHTML = '';
      for (const entry of batch.images) {
        const d = el('div', 'bthumb', null, th);
        if (entry.thumb) el('img', '', null, d).src = entry.thumb;
        el('span', 'nm', entry.name, d);
        el('span', 'ct',
           terms.map(t => entry.counts[t] != null ? entry.counts[t] : '-').join(' '), d);
      }
      show('bcancel', batchPausable());
      show('bopenreview', batch.state === 'done');
      show('bdismiss', batch.state === 'done');
    }
    $('bdismiss').addEventListener('click', () => {
      if (batch.state !== 'done') return;
      batch.state = 'idle';
      batch.queue = []; batch.images = []; batch.counts = {}; batch.pos = 0;
      renderBatchModal();
    });

    $('batchstart').addEventListener('click', async () => {
      if (batchLooping()) return;
      // busy does not refuse the start: the loop defers until it clears
      if (busy) say('Batch started - waiting for the current run to finish...');
      const resume = batch.state === 'paused';
      if (!(resume ? batch.terms : batchSelectedTerms()).length
          || (!state.imagePath && !batch.pickedMeta)) return;
      batch.state = 'running';
      if (!resume) {
        const queue = await buildBatchQueue();
        if (!queue || !queue.length) {
          batch.state = 'idle';
          sayError('Cannot start: directory listing unavailable or empty.');
          updateBatchUi();
          return;
        }
        batch.queue = queue;
        batch.pos = 0;
        batch.dir = batch.pickedMeta ? batch.pickedMeta.dir : dirOf(queue[0]);
        batch.images = [];
        batch.zephyr = $('bzephyr').checked;
        batch.terms = batchSelectedTerms();
        // one count per term, zero-hit included
        batch.counts = {};
        for (const t of batch.terms) batch.counts[t] = 0;
      }
      renderBatchModal();
      batchLoop(++batchRun);
    });
    $('batchpause').addEventListener('click', () => {
      if (batch.state !== 'running') return;
      batch.state = 'pausing';
      cancelSweep();
      renderBatchModal();
    });
    $('bcancel').addEventListener('click', () => {
      if (!batchPausable()) return;
      // the in-flight sweep aborts, but the loop still dedupes + saves +
      // records what landed on the current image before going idle
      batch.state = 'cancelling';
      cancelSweep();
      hideBatchModal();
    });
    $('bopenreview').addEventListener('click', () => {
      hideBatchModal();
      setPage('review');
    });

    // --- directory picker ---------------------------------------------------

    // Use pins the listing on the batch as pickedMeta.
    const pick = { meta: null, req: 0 };
    async function pickInto(path) {
      const token = ++pick.req;
      const m = await fetchDirMeta(path);
      if (token !== pick.req) return;  // a newer navigation owns the modal
      if (!m) { sayError('Cannot list ' + (path || 'the server root') + '.'); return; }
      pick.meta = m;
      $('pickdir').textContent = m.dir;
      $('pickup').disabled = !m.parent;
      $('pickcount').textContent = m.count + ' image(s)';
      const list = $('picklist');
      list.innerHTML = '';
      if (!m.subdirs.length) list.textContent = 'no subdirectories';
      for (const name of m.subdirs) {
        const row = el('div', 'pickrow', name, list);
        row.addEventListener('click', () => pickInto(listedPath(m, name)));
      }
    }
    function openPicker() {
      showModal('pickmodal', true);
      pickInto(batch.pickedMeta ? batch.pickedMeta.dir
        : state.imagePath ? dirOf(state.imagePath) : '');
    }
    function closePicker() {
      pick.req++;  // drop any in-flight listing
      showModal('pickmodal', false);
    }
    $('bbrowse').addEventListener('click', openPicker);
    $('pickup').addEventListener('click', () => {
      if (pick.meta && pick.meta.parent) pickInto(pick.meta.parent);
    });
    $('pickcancel').addEventListener('click', closePicker);
    $('pickuse').addEventListener('click', () => {
      if (pick.meta) {
        batch.pickedMeta = pick.meta;
        batch.dir = pick.meta.dir;
      }
      closePicker();
      renderBatchModal();
    });

    // --- review grid --------------------------------------------------------

    function setPage(page) {
      ui.page = page;
      const work = page === 'work';
      show('stage', work, 'flex');
      show('rail', work, 'flex');
      show('review', !work, 'block');
      $('main').style.gridTemplateColumns = work ? '1fr 336px' : '1fr';
      $('reviewbtn').textContent = work ? 'Review batch >' : '< Workspace';
      if (!work) renderReview();
      else { sizeCanvas(); scheduleRedraw(); }
    }
    $('reviewbtn').addEventListener('click',
      () => setPage(ui.page === 'work' ? 'review' : 'work'));
    $('backwork').addEventListener('click', (e) => {
      e.preventDefault();
      setPage('work');
    });

    // Per-image class counts from this session's batch, else from the COCO
    // file on disk (/dir_counts); zero-hit and >2-sigma tiles are flagged.
    let reviewReq = 0;  // last render wins; a stale /dir_counts response drops
    function renderReview() {
      const token = ++reviewReq;
      if (batch.images.length) {
        renderReviewGrid(batch.images, Object.keys(batch.counts),
          batch.dir ? basename(batch.dir) : '');
        renderReviewStats(batch.dir || '');
        return;
      }
      const dirPath = batch.dir
        || (batch.pickedMeta && batch.pickedMeta.dir)
        || state.imagePath || '';
      renderReviewGrid([], [], '');
      renderReviewStats(dirPath);
      if (!dirPath) return;
      getPath('/dir_counts', dirPath).then(r => {
        if (token !== reviewReq || ui.page !== 'review'
            || batch.images.length || !r.ok) return;
        const m = r.json;
        const images = m.images.map(e => ({
          path: listedPath(m, e.name), name: e.name, counts: e.counts,
          thumb: ImageProxy.url(listedPath(m, e.name), 256),
        }));
        renderReviewGrid(images, m.categories.map(c => c.name), basename(m.dir));
      }).catch(() => {});
    }

    // tri-state grid filter: label -> 'req' | 'excl'; absent = neutral
    const reviewFilter = new Map();
    let reviewData = null;  // last-rendered {images, terms, dirLabel}
    function reviewShown(images) {
      return images.filter(e => [...reviewFilter].every(([t, m]) =>
        m === 'req' ? (e.counts[t] || 0) > 0 : !(e.counts[t] || 0)));
    }
    function renderReviewFilter(terms) {
      const box = $('reviewfilter');
      box.innerHTML = '';
      for (const term of terms) {
        const mode = reviewFilter.get(term);
        const chip = el('span', 'fchip' + (mode === 'excl' ? ' excl warn' : ''),
          (mode === 'req' ? '+ ' : mode === 'excl' ? '- ' : '') + term, box);
        chip.title = 'require -> exclude -> off';
        if (mode === 'req') tintChip(chip, labelColor(term));
        chip.addEventListener('click', () => {
          if (mode === 'req') reviewFilter.set(term, 'excl');
          else if (mode === 'excl') reviewFilter.delete(term);
          else reviewFilter.set(term, 'req');
          if (reviewData) {
            renderReviewGrid(reviewData.images, reviewData.terms,
                             reviewData.dirLabel);
          }
          for (const [label, sec] of histSecByLabel) {
            sec.style.opacity = reviewFilter.get(label) === 'excl' ? 0.45 : 1;
          }
        });
      }
    }

    function renderReviewGrid(images, terms, dirLabel) {
      reviewData = { images, terms, dirLabel };
      for (const k of [...reviewFilter.keys()]) {
        if (!terms.includes(k)) reviewFilter.delete(k);
      }
      renderReviewFilter(images.length ? terms : []);
      $('reviewtitle').textContent =
        'Review' + (dirLabel ? ' | ' + dirLabel : '');
      show('reviewempty', !images.length, 'block');
      const grid = $('reviewgrid');
      grid.innerHTML = '';
      if (!images.length) {
        $('reviewsummary').textContent = '';
        $('reviewmeans').textContent = '';
        return;
      }
      // flags and means stay dataset-wide while the filter narrows the tiles
      const stats = {};
      for (const t of terms) {
        const vals = images.map(e => e.counts[t] || 0);
        const mu = vals.reduce((a, x) => a + x, 0) / vals.length;
        const sd = Math.sqrt(vals.reduce((a, x) => a + (x - mu) * (x - mu), 0) / vals.length);
        stats[t] = { mu, sd };
      }
      const shown = reviewFilter.size ? reviewShown(images) : images;
      let total = 0;
      for (const entry of shown) {
        let flag = null;
        for (const t of terms) {
          const v = entry.counts[t] || 0;
          total += v;
          if (v === 0) flag = '0 ' + t;
          else if (stats[t].sd > 0 && Math.abs(v - stats[t].mu) > 2 * stats[t].sd) {
            flag = v + ' ' + t;
          }
        }
        const tile = el('div', 'tile' + (flag ? ' flag' : ''), null, grid);
        const thumb = el('div', 'thumb', null, tile);
        if (entry.thumb) el('img', '', null, thumb).src = entry.thumb;
        if (flag) el('span', 'fl', '! ' + flag, thumb);
        const meta = el('div', 'meta', null, tile);
        el('span', 'n', entry.name, meta);
        el('span', 'grow', null, meta);
        for (const t of terms) {
          const v = el('span', '', String(entry.counts[t] || 0), meta);
          v.style.color = entry.counts[t] ? labelColor(t) : 'var(--yellow)';
        }
        tile.addEventListener('click', () => {
          $('path').value = entry.path;
          setPage('work');
          loadImage();
        });
      }
      $('reviewsummary').textContent =
        (reviewFilter.size ? shown.length + '/' + images.length : String(images.length))
        + ' images | ' + total + ' instances';
      $('reviewmeans').textContent =
        terms.map(t => t + ' mu=' + Math.round(stats[t].mu)).join(' | ');
    }

    // --- review size histograms (directory-wide, from /dir_stats) -----------

    let statsReq = 0;  // last render wins, like reviewReq
    const histSecByLabel = new Map();
    function renderReviewStats(dirPath) {
      const token = ++statsReq;
      show('reviewstats', false);
      histSecByLabel.clear();
      if (!dirPath) return;
      getPath('/dir_stats', dirPath).then(r => {
        if (token !== statsReq || ui.page !== 'review' || !r.ok
            || !r.json.labels.length) return;
        buildStatsPanel(r.json);
      }).catch(() => {});
    }

    function buildStatsPanel(data) {
      const panel = $('reviewstats');
      panel.innerHTML = '';
      // one shared axis across labels, trimmed to the occupied bin range
      let lo = data.edges.length - 1, hi = 0;
      for (const l of data.labels) {
        l.area.bins.forEach((n, i) => {
          if (n) { lo = Math.min(lo, i); hi = Math.max(hi, i); }
        });
      }
      if (hi < lo) return;
      const side = a => Math.round(Math.sqrt(a));  // axis in px side length
      for (const l of data.labels) {
        const sec = el('div', 'card', null, panel);
        histSecByLabel.set(l.name, sec);
        const h = el('div', 'h', null, sec);
        el('span', 'dot9', null, h).style.background = labelColor(l.name);
        el('span', '', l.name, h);
        el('span', 'grow', null, h);
        el('span', 'mono-dim', l.count + ' instances', h);
        el('div', 's', 'side ' + side(l.area.min) + '-' + side(l.area.max)
          + ' px | bbox w ' + Math.round(l.bbox.w_min) + '-' + Math.round(l.bbox.w_max)
          + ' | h ' + Math.round(l.bbox.h_min) + '-' + Math.round(l.bbox.h_max), sec);
        const bins = l.area.bins.slice(lo, hi + 1);
        const max = Math.max(...bins, 1);
        const bar = el('div', 'hist', null, sec);
        bins.forEach((n, i) => {
          const d = el('div', '', null, bar);
          d.style.height = n ? Math.max(2, Math.round(n / max * 34)) + 'px' : '1px';
          d.style.background = labelColor(l.name);
          d.title = side(data.edges[lo + i]) + '-' + side(data.edges[lo + i + 1])
            + ' px side | ' + n;
        });
        const ax = el('div', 'hax', null, sec);
        for (let i = 0; i < bins.length; i++) {
          el('span', '', i % 4 === 0 ? String(side(data.edges[lo + i])) : '', ax);
        }
        sec.style.opacity = reviewFilter.get(l.name) === 'excl' ? 0.45 : 1;
      }
      show(panel, true, 'grid');
    }

    // --- modes / page chrome ------------------------------------------------

    function setMode(m) {
      ui.mode = m;
      const tabs = children($('tabs'));
      tabs.forEach((tab, i) => {
        const on = MODES[i] === m;
        tab.className = on
          ? 'on' + (m === 'reprompt' ? ' warn' : m === 'erase' ? ' danger' : '')
          : '';
      });
      const boxy = m === 'reprompt' || m === 'erase';
      show('promptrow', !boxy, 'flex');
      show('modenote', boxy, 'block');
      if (m === 'reprompt') {
        $('modenote').textContent = 'No prompt needed - drag a box over an ambiguous '
          + 'cluster on the image; its dominant label is re-prompted on a context '
          + 'recrop and the fragments inside are replaced. Boxes queue without blocking.';
      } else if (m === 'erase') {
        $('modenote').textContent = 'No prompt needed - drag a box on the image; every '
          + 'instance centered inside is deleted (saved ones staged until Save).';
      }
      show('run', m === 'text');
      show('addprompt', m === 'text');
      $('promptText').placeholder =
        m === 'text' ? "describe a class... e.g. 'pallet'"
        : m === 'exemplar' ? 'label for exemplar matches...'
        : 'label for clicked object...';
      show('modehint', m !== 'text', 'block');
      if (m !== 'text') {
        $('modehint').textContent =
          m === 'exemplar' ? 'Drag a box around one example - similar objects get the label from the prompt field'
          : m === 'point' ? 'Click an object - its mask gets the label from the prompt field'
          : m === 'reprompt' ? 'Drag a box over an ambiguous cluster - fragments are re-prompted into clean instances'
          : 'Drag a box - every instance centered inside is deleted';
      }
      $('canvaswrap').style.cursor = (m === 'text' || m === 'point') ? 'grab' : 'crosshair';
    }
    children($('tabs')).forEach((tab, i) => {
      tab.addEventListener('click', () => setMode(MODES[i]));
    });

    collapsible('adv', 'advpanel', 'Advanced ');

    function toggleHideMasks() {
      ui.hideMasks = !ui.hideMasks;
      $('eyeall').textContent = ui.hideMasks ? 'masks hidden' : 'hide masks';
      $('eyeall').classList.toggle('warn', ui.hideMasks);
      scheduleRedraw();
    }
    $('eyeall').addEventListener('click', toggleHideMasks);

    $('solo').addEventListener('click', () => {
      ui.solo = !ui.solo;
      $('solo').classList.toggle('warn', ui.solo);
      scheduleRedraw();
    });
    $('outline').addEventListener('click', () => {
      ui.outline = !ui.outline;
      $('outline').classList.toggle('warn', ui.outline);
      if (md) md.setOutline(ui.outline);
      scheduleRedraw();
    });

    function togglePresence() {
      ui.showPresence = !ui.showPresence;
      $('presence').classList.toggle('warn', ui.showPresence);
      scheduleRedraw();
    }
    $('presence').addEventListener('click', togglePresence);

    // --- image nav ----------------------------------------------------------

    $('path').addEventListener('keydown', (e) => { if (e.key === 'Enter') loadImage(); });
    $('loadbtn').addEventListener('click', () => loadImage());
    $('prev').addEventListener('click', () => navigate(state.prev));
    $('next').addEventListener('click', () => navigate(state.next));

    // --- embed-cache rail panel (polls only while expanded) -----------------

    let cachePolling = false;
    async function pollCacheStats() {
      if ($('cachebody').style.display !== 'block' || ui.page !== 'work'
          || document.hidden || cachePolling) return;
      cachePolling = true;
      try {
        const r = await getJson('/cache_stats');
        if (!r.ok) return;
        const s = r.json;
        const gb = b => (b / 2 ** 30).toFixed(1);
        $('cachedisk').textContent =
          s.present ? gb(s.disk_bytes) + ' / ' + gb(s.disk_budget) + ' GB' : '-';
        $('cachebudget').textContent = s.present ? '' : 'model not loaded';
        $('cacheimgs').textContent =
          s.present && s.images_left != null ? String(s.images_left) : '-';
        const lookups = s.present ? s.hits + s.misses : 0;
        $('cachehit').textContent = lookups ? Math.round(s.hit_rate * 100) + '%' : '-';
        $('cachecounts').textContent = lookups ? s.hits + '/' + lookups : '';
        $('cachetensors').textContent = s.present ? String(s.disk_tensors) : '-';
        $('cacheratio').textContent =
          s.present && s.ratio != null ? s.ratio.toFixed(2) + ':1' : '-';
      } catch (err) { /* display only */
      } finally {
        cachePolling = false;
      }
    }
    collapsible('cachetoggle', 'cachebody', '', pollCacheStats);
    setInterval(pollCacheStats, 2000);

    // --- init ---------------------------------------------------------------

    // knob inputs seeded from the served defaults (no-seed knobs stay
    // blank), served defaults appended to the hover hints
    for (const k of KNOBS) {
      const el = k.el;
      if (k.seed !== undefined) {
        if (el.type === 'checkbox') el.checked = k.seed; else el.value = k.seed;
      }
      const def = 'hint' in k ? k.hint
        : el.type === 'checkbox' ? (k.seed ? 'on' : 'off') : k.seed;
      if (def !== undefined && el.title) el.title += ' (default ' + def + ')';
      // wheel over a focused number input scrolls the panel, not the value
      if (el.type === 'number') {
        el.addEventListener('wheel', (e) => {
          if (document.activeElement === el) { e.preventDefault(); el.blur(); }
        });
      }
    }
    const knobState = el => (el.type === 'checkbox' ? el.checked : String(el.value));
    const allKnobs = KNOBS.map(k => k.el)
      .concat([$('ignoreprev'), $('replaceexisting')]);
    const knobSeeds = new Map(allKnobs.map(el => [el, knobState(el)]));
    // gates gray the knobs they disable; markers flag values off the seeds
    function refreshKnobUi() {
      for (const k of KNOBS) if (k.gate) k.el.disabled = !k.gate.checked;
      // the band pair reads as off until both edges are set and ordered
      const on = bandOn();
      $('sweepbandlo').classList.toggle('off', !on);
      $('sweepbandhi').classList.toggle('off', !on);
      for (const el of allKnobs) {
        el.classList.toggle('tweaked', knobState(el) !== knobSeeds.get(el));
      }
    }
    for (const el of allKnobs) el.addEventListener('input', refreshKnobUi);
    refreshKnobUi();

    new ResizeObserver(sizeCanvas).observe($('canvaswrap'));
    setMode('text');
    renderChips();
    renderInstances();
    updateBatchUi();
