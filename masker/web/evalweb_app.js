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

    // Benchmark dashboard
    const $ = (id) => document.getElementById(id);
    // report mode/threshold controls and readouts
    const ctl = { mode: $('mode'), matchiou: $('matchiou'), downscale: $('downscale'),
                  reload: $('reload'), status: $('loadstatus'), paths: $('paths') };
    // summary section: report (stat tiles, charts), coverage tables
    const sum = { root: $('summary'), empty: $('empty'), report: $('report'),
                  tiles: $('tiles'), band: $('bandchart'), cats: $('cattable'),
                  imgs: $('imgtable'), coverage: $('coverage') };
    // per-image detail: viewer canvas plus layer toggles
    const det = { root: $('detail'), name: $('detailname'), back: $('back'),
                  prev: $('prevbtn'), next: $('nextbtn'), cat: $('catfilter'),
                  wrap: $('viewwrap'), canvas: $('view'), status: $('detailstatus'),
                  boxes: { tp: $('layertp'), fn: $('layerfn'), fp: $('layerfp') },
                  counts: { tp: $('tpcount'), fn: $('fncount'), fp: $('fpcount') } };
    const tip = $('tip');

    // layer tints: webui.css --tp/--fn/--fp custom properties, read once
    const rootCss = getComputedStyle(document.documentElement);
    const LAYER_COLORS = Object.fromEntries(
      ['tp', 'fn', 'fp'].map((k) => [k, rootCss.getPropertyValue('--' + k).trim()]));
    const LAYER_ALPHA = 0.5;
    const KIND_LABELS = { tp: 'matched', fn: 'missed reference', fp: 'spurious' };

    const state = { summary: null, image: null, layers: {}, hover: null, mouse: [0, 0] };

    async function getJson(url) {
      const resp = await fetch(url);
      let json = null;
      try { json = await resp.json(); } catch (e) { /* non-JSON body */ }
      return { ok: resp.ok, json };
    }

    const pct = (v) => (v == null ? '-' : (v * 100).toFixed(1) + '%');
    const f3 = (v) => (v == null ? '-' : Number(v).toFixed(3));

    function showTip(text, x, y) {
      tip.textContent = text;
      tip.hidden = false;
      tip.style.left = Math.min(x + 14, window.innerWidth - tip.offsetWidth - 8) + 'px';
      tip.style.top = (y + 14) + 'px';
    }
    function hideTip() { tip.hidden = true; }

    function reportQuery() {
      const q = new URLSearchParams();
      q.set('mode', ctl.mode.value);
      q.set('match_iou', ctl.matchiou.value || CFG.MATCH_IOU);
      q.set('downscale', ctl.downscale.value || 1);
      return q;
    }

    // --- stat tiles ---
    function tile(value, label, sub) {
      const el = document.createElement('div');
      el.className = 'card';
      const h = document.createElement('div');
      h.className = 'h';
      h.textContent = label;
      const v = document.createElement('div');
      v.className = 'v';
      v.textContent = value;
      el.appendChild(h);
      el.appendChild(v);
      if (sub) {
        const s = document.createElement('div');
        s.className = 's';
        s.textContent = sub;
        el.appendChild(s);
      }
      return el;
    }

    function renderTiles(report) {
      sum.tiles.replaceChildren();
      const o = report.overall;
      if (report.mode === 'instance') {
        sum.tiles.appendChild(tile(pct(o.precision), 'precision'));
        sum.tiles.appendChild(tile(pct(o.recall), 'recall'));
        sum.tiles.appendChild(tile(pct(o.f1), 'F1'));
        sum.tiles.appendChild(tile(pct(o.pq), 'panoptic quality', 'SQ ' + pct(o.sq) + ' x RQ ' + pct(o.rq)));
        sum.tiles.appendChild(tile(String(o.tp), 'matched (TP)', o.fp + ' FP | ' + o.fn + ' FN'));
        sum.tiles.appendChild(tile(String(o.false_splits), 'false splits', o.false_merges + ' false merges'));
      } else {
        sum.tiles.appendChild(tile(pct(o.iou), 'IoU'));
        sum.tiles.appendChild(tile(pct(o.dice), 'Dice'));
        sum.tiles.appendChild(tile(pct(o.precision), 'pixel precision'));
        sum.tiles.appendChild(tile(pct(o.recall), 'pixel recall'));
        sum.tiles.appendChild(tile((o.ref_px / 1e6).toFixed(1) + ' MP', 'reference px', (o.pred_px / 1e6).toFixed(1) + ' MP predicted'));
      }
    }

    // --- recall-by-band column chart (single series, one hue) ---
    function renderBands(report) {
      sum.band.replaceChildren();
      const o = report.overall;
      const instance = report.mode === 'instance';
      sum.band.parentElement.style.display = instance ? '' : 'none';
      if (!instance) return;
      o.recall_by_band.forEach((r, i) => {
        const col = document.createElement('div');
        col.className = 'band';
        const v = document.createElement('div');
        v.className = 'bv mono-dim';
        v.textContent = r == null ? '-' : pct(r);
        const bar = document.createElement('div');
        bar.className = 'bb';
        bar.style.height = r == null ? '0' : Math.max(r * 130, 2) + 'px';
        const label = document.createElement('div');
        label.className = 'bl mono-dim';
        label.textContent = CFG.BANDS[i];
        const n = document.createElement('div');
        n.className = 'mono-dim';
        n.textContent = 'n=' + o.ref_by_band[i];
        col.appendChild(v);
        col.appendChild(bar);
        col.appendChild(label);
        col.appendChild(n);
        col.addEventListener('mousemove', (e) => showTip(
          CFG.BANDS[i] + ' px: recall ' + pct(r) + ' (' + o.hit_by_band[i] + ' of ' + o.ref_by_band[i] + ')',
          e.clientX, e.clientY));
        col.addEventListener('mouseleave', hideTip);
        sum.band.appendChild(col);
      });
    }

    // --- tables ---
    // first cell left-aligned, the rest right
    function row(cells) {
      const tr = document.createElement('tr');
      cells.forEach((cell, i) => {
        const td = document.createElement(typeof cell === 'object' && cell.th ? 'th' : 'td');
        if (i >= 1) td.className = 'right';
        if (typeof cell === 'object') {
          if (cell.th) td.textContent = cell.th;
          else td.appendChild(cell);
        } else {
          td.textContent = cell;
        }
        tr.appendChild(td);
      });
      return tr;
    }

    function barCell(value, text) {
      const wrap = document.createElement('span');
      const bar = document.createElement('span');
      bar.className = 'cellbar';
      bar.style.width = Math.max((value || 0) * 60, 1) + 'px';
      wrap.appendChild(bar);
      wrap.appendChild(document.createTextNode(text));
      return wrap;
    }

    function renderCategories(report) {
      sum.cats.replaceChildren();
      const instance = report.mode === 'instance';
      const keyMetric = instance ? 'f1' : 'iou';
      const names = Object.keys(report.categories)
        .sort((a, b) => report.categories[a][keyMetric] - report.categories[b][keyMetric]);
      const head = instance
        ? ['category', 'F1', 'P', 'R', 'PQ', 'SQ', 'TP', 'FP', 'FN', 'split', 'merge']
        : ['category', 'IoU', 'Dice', 'P', 'R', 'pred px', 'ref px'];
      sum.cats.appendChild(row(head.map((h) => ({ th: h }))));
      for (const name of names.concat(['OVERALL'])) {
        const m = name === 'OVERALL' ? report.overall : report.categories[name];
        const cells = instance
          ? [name, barCell(m.f1, f3(m.f1)), f3(m.precision), f3(m.recall), f3(m.pq), f3(m.sq),
             m.tp, m.fp, m.fn, m.false_splits, m.false_merges]
          : [name, barCell(m.iou, f3(m.iou)), f3(m.dice), f3(m.precision), f3(m.recall),
             m.pred_px, m.ref_px];
        sum.cats.appendChild(row(cells));
      }
    }

    // Per-image aggregates from the report's per-category entries.
    function imageMetrics(report, name) {
      const entry = report && report.per_image[name];
      if (!entry) return null;
      const cats = Object.values(entry);
      if (report.mode === 'instance') {
        const t = { tp: 0, fp: 0, fn: 0 };
        cats.forEach((m) => { t.tp += m.tp; t.fp += m.fp; t.fn += m.fn; });
        const p = t.tp + t.fp ? t.tp / (t.tp + t.fp) : 0;
        const r = t.tp + t.fn ? t.tp / (t.tp + t.fn) : 0;
        return { a: p, b: r, c: p + r ? 2 * p * r / (p + r) : 0 };
      }
      const mean = (k) => cats.reduce((s, m) => s + m[k], 0) / (cats.length || 1);
      return { a: mean('iou'), b: mean('dice'), c: null };
    }

    function renderImages(payload) {
      sum.imgs.replaceChildren();
      const instance = !payload.report || payload.report.mode === 'instance';
      const head = ['image', 'reference instances', 'categories', 'predicted']
        .concat(instance ? ['P', 'R', 'F1'] : ['IoU', 'Dice']);
      sum.imgs.appendChild(row(head.map((h) => ({ th: h }))));
      const annotated = payload.images.filter((i) => i.ref > 0).length;
      sum.coverage.textContent = annotated + ' of ' + payload.images.length
        + ' images hand-annotated';
      for (const img of payload.images) {
        const m = imageMetrics(payload.report, img.name);
        const metrics = m === null ? (instance ? ['-', '-', '-'] : ['-', '-'])
          : instance ? [f3(m.a), f3(m.b), f3(m.c)] : [f3(m.a), f3(m.b)];
        const tr = row([img.name, img.ref || 'none', img.categories, img.predicted]
          .concat(metrics));
        tr.className = 'click';
        tr.title = 'Open the overlay view';
        tr.addEventListener('click', () => openDetail(img.name));
        sum.imgs.appendChild(tr);
      }
    }

    async function loadSummary() {
      ctl.status.textContent = 'loading...';
      let ok, json;
      try {
        ({ ok, json } = await getJson('/summary?' + reportQuery()));
      } catch (err) {
        ctl.status.textContent = 'Error: ' + (err && err.message ? err.message : err);
        return;
      }
      if (!ok) {
        ctl.status.textContent = 'Error: ' + ((json && json.error) || 'summary failed');
        return;
      }
      ctl.status.textContent = '';
      state.summary = json;
      ctl.paths.textContent = json.root
        + ' | reference: ' + (json.reference || 'none')
        + ' | predictions: ' + (json.predictions || 'none');
      sum.root.hidden = false;
      const haveReport = json.report !== null;
      sum.report.hidden = !haveReport;
      sum.empty.hidden = haveReport;
      if (!haveReport) {
        sum.empty.textContent = json.reference === null
          ? 'No reference found: annotate images in the masker web UI so '
            + 'a COCO file appears beside them.'
          : 'No predictions found: sweep this directory to produce '
            + 'a prediction COCO.';
      } else {
        renderTiles(json.report);
        renderBands(json.report);
        renderCategories(json.report);
      }
      renderImages(json);
      if (state.image) openDetail(state.image.name);
    }

    // --- per-image detail ---
    const viewer = createViewer(det.canvas, {
      onHover: (x, y) => hoverAt(x, y),
    });
    det.canvas.addEventListener('mousemove', (e) => { state.mouse = [e.clientX, e.clientY]; });

    function visibleKinds() {
      return ['fp', 'tp', 'fn'].filter((k) => det.boxes[k].checked);
    }

    function drawOverlay(ctx, v) {
      for (const kind of visibleKinds()) {
        const layer = state.layers[kind];
        if (!layer || !state.image) continue;
        drawAlpha(ctx, LAYER_ALPHA, layer, 0, 0, state.image.width, state.image.height);
      }
      if (state.hover) {
        const [x0, y0, x1, y1] = state.hover.bbox_xyxy;
        ctx.strokeStyle = LAYER_COLORS[state.hover.kind];
        ctx.lineWidth = 2 / v.scale;
        ctx.strokeRect(x0, y0, x1 - x0, y1 - y0);
      }
    }
    viewer.setOverlay(drawOverlay);

    function hoverAt(x, y) {
      let best = null;
      if (x !== null && state.image) {
        const shown = new Set(visibleKinds());
        for (const inst of state.image.instances) {
          const [x0, y0, x1, y1] = inst.bbox_xyxy;
          if (!shown.has(inst.kind) || x < x0 || x > x1 || y < y0 || y > y1) continue;
          // missing area sorts as +inf: never favored over a sized instance
          if (best === null || (inst.area ?? Infinity) < (best.area ?? Infinity)) best = inst;
        }
      }
      if (best !== state.hover) {
        state.hover = best;
        viewer.redraw();
      }
      if (best === null) { hideTip(); return; }
      let text = best.category + ' | ' + KIND_LABELS[best.kind];
      if (best.iou != null) text += ' | IoU ' + f3(best.iou);
      if (best.score != null) text += ' | score ' + f3(best.score);
      if (best.area != null) text += ' | ' + Math.round(best.area) + ' px^2';
      showTip(text, state.mouse[0], state.mouse[1]);
    }

    function loadImageEl(src) {
      return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => resolve(img);
        img.onerror = () => reject(new Error('image load failed: ' + src));
        img.src = src;
      });
    }

    // Full-frame white mask -> layer color, tinted once at load.
    function tinted(img, color) {
      const c = document.createElement('canvas');
      c.width = img.naturalWidth;
      c.height = img.naturalHeight;
      const g = c.getContext('2d');
      g.drawImage(img, 0, 0);
      g.globalCompositeOperation = 'source-in';
      g.fillStyle = color;
      g.fillRect(0, 0, c.width, c.height);
      return c;
    }

    function fitCanvas() {
      const width = det.wrap.clientWidth || 800;
      viewer.resize(width, Math.max(420, window.innerHeight - 220));
    }
    window.addEventListener('resize', () => { if (!det.root.hidden) { fitCanvas(); viewer.redraw(); } });

    async function openDetail(name, category) {
      det.root.hidden = false;
      det.name.textContent = name;
      det.status.textContent = 'loading...';
      fitCanvas();
      try {
        const q = new URLSearchParams({ name, match_iou: ctl.matchiou.value || CFG.MATCH_IOU });
        if (category) q.set('category', category);
        const [evalRes, baseImg] = await Promise.all([
          getJson('/image_eval?' + q),
          loadImageEl('/image?name=' + encodeURIComponent(name) + '&resolution=' + CFG.PREVIEW_SIDE),
        ]);
        if (!evalRes.ok) {
          det.status.textContent = 'Error: ' + ((evalRes.json && evalRes.json.error) || 'load failed');
          return;
        }
        state.image = evalRes.json;
        state.hover = null;
        state.layers = {};
        const loads = [];
        for (const kind of ['tp', 'fn', 'fp']) {
          const url = state.image.layers[kind];
          if (url) loads.push(loadImageEl(url).then((img) => { state.layers[kind] = tinted(img, LAYER_COLORS[kind]); }));
          det.counts[kind].textContent = String(state.image.counts[kind]);
        }
        await Promise.all(loads);
        det.cat.replaceChildren(new Option('all', ''));
        for (const cat of state.image.categories) {
          det.cat.appendChild(new Option(cat, cat, false, cat === (category || '')));
        }
        det.status.textContent = state.image.instances.length
          ? '' : 'no annotations or predictions for this image';
        viewer.setImage(baseImg, state.image.width, state.image.height);
        det.root.scrollIntoView({ block: 'nearest' });
      } catch (err) {
        // a rejected image or layer load must not leave "loading..." stuck
        det.status.textContent = 'Error: ' + (err && err.message ? err.message : err);
      }
    }

    function step(delta) {
      if (!state.summary || !state.image) return;
      const names = state.summary.images.map((i) => i.name);
      const next = names.indexOf(state.image.name) + delta;
      if (next >= 0 && next < names.length) openDetail(names[next], det.cat.value || undefined);
    }

    det.back.addEventListener('click', () => {
      det.root.hidden = true;
      state.image = null;
      sum.root.scrollIntoView({ block: 'nearest' });
    });
    det.prev.addEventListener('click', () => step(-1));
    det.next.addEventListener('click', () => step(1));
    det.cat.addEventListener('change', () => openDetail(state.image.name, det.cat.value || undefined));
    for (const kind of ['tp', 'fn', 'fp']) {
      det.boxes[kind].addEventListener('change', () => viewer.redraw());
    }

    ctl.matchiou.value = CFG.MATCH_IOU;
    ctl.reload.addEventListener('click', loadSummary);
    ctl.mode.addEventListener('change', loadSummary);
    loadSummary();
