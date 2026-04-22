'use strict';

// ── State ──────────────────────────────────────────────────────────────────────
const S = {
  elements:  [],   // [{id, vertices:[[x,y]...], type, class, label, color, opacity, strokeWidth}]
  rooms:     [],
  imgObj:    null, // fabric.Image reference
  polyScale: 1,    // model-px → canvas-px
  imgOffX:   0,
  imgOffY:   0,
  activeEntry: null,
  history:   [],   // undo stack [{elements:[], rooms:[]}]
  showElements: true,
  showRooms:    true,
  showImage:    true,
};

// ── Fabric canvas ──────────────────────────────────────────────────────────────
const fc = new fabric.Canvas('c', {
  selection:              true,
  preserveObjectStacking: true,
  fireRightClick:         false,
});

function fitCanvasToContainer() {
  const wrap = document.getElementById('canvas-wrap');
  fc.setWidth(wrap.clientWidth);
  fc.setHeight(wrap.clientHeight);
  fc.renderAll();
}
window.addEventListener('resize', fitCanvasToContainer);
fitCanvasToContainer();

// ── Coordinate helpers ─────────────────────────────────────────────────────────
function m2c([x, y]) {
  return { x: S.imgOffX + x * S.polyScale, y: S.imgOffY + y * S.polyScale };
}
function c2m({ x, y }) {
  return [(x - S.imgOffX) / S.polyScale, (y - S.imgOffY) / S.polyScale];
}

// ── Geometry helpers ───────────────────────────────────────────────────────────
function bbox(verts) {
  const xs = verts.map(v => v[0]);
  const ys = verts.map(v => v[1]);
  return { minX: Math.min(...xs), minY: Math.min(...ys), maxX: Math.max(...xs), maxY: Math.max(...ys) };
}
function centerOf(verts) {
  const b = bbox(verts);
  return [(b.minX + b.maxX) / 2, (b.minY + b.maxY) / 2];
}

// ── Absolute canvas vertices from a transformed fabric.Polygon ─────────────────
function fabricAbsVerts(poly) {
  const mat = poly.calcTransformMatrix();
  return poly.points.map(p => {
    const pt = fabric.util.transformPoint(
      { x: p.x - poly.pathOffset.x, y: p.y - poly.pathOffset.y },
      mat,
    );
    return [pt.x, pt.y];
  });
}

// ── Create a Fabric polygon from an entry ──────────────────────────────────────
function makePoly(entry, groupVisible = true) {
  const pts   = entry.vertices.map(v => m2c(v));
  const hex   = entry.color || '#ffffff';
  const alpha = Math.round((entry.opacity ?? 0.3) * 255).toString(16).padStart(2, '0');
  const show  = groupVisible && (entry.visible !== false);

  const poly = new fabric.Polygon(pts, {
    fill:               hex + alpha,
    stroke:             hex,
    strokeWidth:        entry.strokeWidth ?? 1.5,
    objectCaching:      false,
    visible:            show,
    selectable:         show,
    evented:            show,
    hasBorders:         true,
    hasControls:        true,
    perPixelTargetFind: false,
    data:               entry,
  });

  entry._obj = poly;
  return poly;
}

// ── Render all polygons (clear old, add from state) ───────────────────────────
function renderPolygons() {
  fc.getObjects().forEach(obj => { if (obj !== S.imgObj) fc.remove(obj); });
  S.rooms.forEach(e    => fc.add(makePoly(e, S.showRooms)));
  S.elements.forEach(e => fc.add(makePoly(e, S.showElements)));
  fc.renderAll();
  renderLayers();
}

// ── Apply group-level visibility without recreating objects ───────────────────
function applyGroupVisibility() {
  S.rooms.forEach(e => {
    const show = S.showRooms && (e.visible !== false);
    if (e._obj) e._obj.set({ visible: show, selectable: show, evented: show });
  });
  S.elements.forEach(e => {
    const show = S.showElements && (e.visible !== false);
    if (e._obj) e._obj.set({ visible: show, selectable: show, evented: show });
  });
  fc.renderAll();
}

// ── Layers panel ───────────────────────────────────────────────────────────────
function renderLayers() {
  renderLayerGroup('elements', S.elements, S.showElements);
  renderLayerGroup('rooms',    S.rooms,    S.showRooms);
}

function renderLayerGroup(group, entries, groupVisible) {
  document.getElementById(`lg-${group}-count`).textContent = entries.length;
  document.getElementById(`lg-${group}`).checked = groupVisible;

  const list = document.getElementById(`layer-list-${group}`);
  list.innerHTML = '';

  entries.forEach((entry, idx) => {
    const isVisible = entry.visible !== false;
    const row = document.createElement('div');
    row.className = 'layer-row' + (S.activeEntry === entry ? ' active' : '');
    row.dataset.group = group;
    row.dataset.idx   = idx;

    row.innerHTML = `
      <input type="checkbox" class="layer-vis" ${isVisible ? 'checked' : ''}
             title="Toggle visibility" ${groupVisible ? '' : 'disabled'}>
      <span class="layer-swatch" style="background:${entry.color || '#888'}"></span>
      <span class="layer-label">${entry.label || 'Polygon'}</span>
      <span class="type-badge ${entry.type}">${entry.type}</span>
    `;

    const cb = row.querySelector('.layer-vis');
    cb.addEventListener('change', ev => {
      ev.stopPropagation();
      entry.visible = cb.checked;
      const show = groupVisible && entry.visible;
      if (entry._obj) {
        entry._obj.set({ visible: show, selectable: show, evented: show });
        fc.renderAll();
      }
    });

    row.addEventListener('click', ev => {
      if (ev.target === cb) return;
      if (entry._obj && entry._obj.visible) {
        fc.setActiveObject(entry._obj);
        fc.renderAll();
        onSelect(entry._obj);
        switchTab('props');
      }
    });

    list.appendChild(row);
  });
}

// ── Tab switching ─────────────────────────────────────────────────────────────
function switchTab(name) {
  document.querySelectorAll('.panel-tab').forEach(btn => {
    btn.classList.toggle('active', btn.dataset.tab === name);
  });
  document.querySelectorAll('.tab-content').forEach(el => {
    el.classList.toggle('hidden', el.id !== `tab-${name}`);
  });
}

document.querySelectorAll('.panel-tab').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

// ── Recreate just one polygon, re-selecting it ────────────────────────────────
function reRenderEntry(entry) {
  if (entry._obj) fc.remove(entry._obj);
  const poly = makePoly(entry);
  fc.add(poly);
  fc.setActiveObject(poly);
  fc.renderAll();
}

// ── Push current state to undo history ────────────────────────────────────────
function pushHistory() {
  const snap = {
    elements: JSON.parse(JSON.stringify(S.elements.map(stripObj))),
    rooms:    JSON.parse(JSON.stringify(S.rooms.map(stripObj))),
  };
  S.history.push(snap);
  if (S.history.length > 50) S.history.shift();
}
function stripObj(e) { const c = { ...e }; delete c._obj; return c; }

// ── Upload + inference ─────────────────────────────────────────────────────────
document.getElementById('file-input').addEventListener('change', async e => {
  const file = e.target.files[0];
  if (!file) return;
  e.target.value = '';

  document.getElementById('hint').classList.add('hidden');
  document.getElementById('loading').classList.remove('hidden');
  document.getElementById('status-text').textContent = 'Uploading…';

  // 1. Show image immediately
  const url = URL.createObjectURL(file);
  await new Promise(resolve => {
    fabric.Image.fromURL(url, img => {
      fc.clear();
      S.elements = [];
      S.rooms    = [];
      S.history  = [];
      clearPanel();

      const origW = img.width;
      const origH = img.height;
      const longSide  = Math.max(origW, origH);
      const fitScale  = Math.min(fc.width / origW, fc.height / origH);
      const dispW     = origW * fitScale;
      const dispH     = origH * fitScale;
      const imgLeft   = (fc.width  - dispW) / 2;
      const imgTop    = (fc.height - dispH) / 2;

      img.set({
        scaleX: fitScale, scaleY: fitScale,
        left: imgLeft, top: imgTop,
        selectable: false, evented: false,
        hasBorders: false, hasControls: false,
      });

      // scale: (model 512-space px) → canvas px
      // inference resizes longest side to 512
      const resizeFactor = 512 / longSide;
      S.polyScale = fitScale / resizeFactor;
      S.imgOffX   = imgLeft;
      S.imgOffY   = imgTop;
      S.imgObj    = img;

      fc.add(img);
      fc.renderAll();
      resolve();
    }, { crossOrigin: null });
  });

  // 2. Call backend
  document.getElementById('status-text').textContent = 'Running inference…';
  const form = new FormData();
  form.append('image', file);

  let data;
  try {
    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(await resp.text());
    data = await resp.json();
  } catch (err) {
    document.getElementById('status-text').textContent = `Error: ${err.message}`;
    document.getElementById('loading').classList.add('hidden');
    return;
  }

  document.getElementById('loading').classList.add('hidden');
  document.getElementById('status-text').textContent =
    `${data.elements.length} elements · ${data.rooms.length} rooms`;

  data.elements.forEach(e => { e.opacity = 0.35; e.strokeWidth = 1.5; e.visible = true; });
  data.rooms.forEach(r    => { r.opacity = 0.20; r.strokeWidth = 1.0; r.visible = true; });

  S.elements = data.elements;
  S.rooms    = data.rooms;

  renderPolygons();
});

// ── Fabric selection events ────────────────────────────────────────────────────
fc.on('selection:created', e  => onSelect(e.selected?.[0]));
fc.on('selection:updated', e  => onSelect(e.selected?.[0]));
fc.on('selection:cleared',  () => clearPanel());

fc.on('object:modified', e => {
  const obj = e.target;
  if (!obj?.data) return;
  pushHistory();
  const entry = obj.data;
  const absCV = fabricAbsVerts(obj);
  entry.vertices = absCV.map(([cx, cy]) => c2m({ x: cx, y: cy }));
  populatePanel(obj);
});

fc.on('object:moving', e => {
  const obj = e.target;
  if (!obj?.data) return;
  const entry = obj.data;
  const absCV = fabricAbsVerts(obj);
  entry.vertices = absCV.map(([cx, cy]) => c2m({ x: cx, y: cy }));
  refreshTransformFields(entry);
  refreshVertexFields(entry);
});

// ── Panel population ───────────────────────────────────────────────────────────
function onSelect(obj) {
  if (!obj?.data) { clearPanel(); return; }
  S.activeEntry = obj.data;
  populatePanel(obj);
  // highlight the matching row in the layers panel (no full re-render needed)
  document.querySelectorAll('.layer-row').forEach(row => row.classList.remove('active'));
  const group = S.elements.includes(obj.data) ? 'elements' : 'rooms';
  const idx   = (group === 'elements' ? S.elements : S.rooms).indexOf(obj.data);
  const activeRow = document.querySelector(`#layer-list-${group} .layer-row[data-idx="${idx}"]`);
  if (activeRow) activeRow.classList.add('active');
}

function clearPanel() {
  S.activeEntry = null;
  document.getElementById('no-sel-msg').classList.remove('hidden');
  document.getElementById('sel-props').classList.add('hidden');
}

function populatePanel(obj) {
  const e = obj.data;
  document.getElementById('no-sel-msg').classList.add('hidden');
  document.getElementById('sel-props').classList.remove('hidden');

  document.getElementById('ph-title').textContent = e.label || 'Polygon';
  const badge = document.getElementById('ph-type-badge');
  badge.textContent = e.type;
  badge.className = `type-badge ${e.type}`;

  _suppress = true;
  document.getElementById('p-label').value    = e.label    ?? '';
  document.getElementById('p-color').value    = e.color    || '#ffffff';
  const pct = Math.round((e.opacity ?? 0.3) * 100);
  document.getElementById('p-opacity').value  = pct;
  document.getElementById('p-opacity-val').textContent = pct + '%';
  document.getElementById('p-stroke-w').value = e.strokeWidth ?? 1.5;
  document.getElementById('p-type').value     = e.type     ?? '';
  document.getElementById('p-class').value    = e.class    ?? 0;

  refreshTransformFields(e);
  refreshVertexFields(e);
  _suppress = false;

  // relative position info (reset to default)
  document.getElementById('rel-pos-info').textContent =
    'Shift+Click another polygon to compare.';
}

function refreshTransformFields(e) {
  const b = bbox(e.vertices);
  document.getElementById('p-x').value   = b.minX.toFixed(1);
  document.getElementById('p-y').value   = b.minY.toFixed(1);
  document.getElementById('p-w').value   = (b.maxX - b.minX).toFixed(1);
  document.getElementById('p-h').value   = (b.maxY - b.minY).toFixed(1);
  // rotation is tracked on the fabric obj
  const obj = e._obj;
  document.getElementById('p-rot').value = obj ? (obj.angle ?? 0).toFixed(1) : '0';
}

function refreshVertexFields(e) {
  const list = document.getElementById('verts-list');
  list.innerHTML = '';
  e.vertices.forEach(([x, y], i) => {
    const row = document.createElement('div');
    row.className = 'vert-row';
    row.dataset.vi = i;
    row.innerHTML = `
      <span class="vert-idx">${i}</span>
      <input type="number" class="vx" value="${x.toFixed(1)}" step="0.5">
      <input type="number" class="vy" value="${y.toFixed(1)}" step="0.5">
      <button class="vert-del" title="Remove vertex">&#215;</button>
    `;
    row.querySelector('.vx').addEventListener('change', onVertexEdit);
    row.querySelector('.vy').addEventListener('change', onVertexEdit);
    row.querySelector('.vert-del').addEventListener('click', onVertexDelete);
    list.appendChild(row);
  });
}

// ── Panel → canvas sync ────────────────────────────────────────────────────────
let _suppress = false;

function withHistory(fn) {
  pushHistory();
  fn();
}

document.getElementById('p-label').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  S.activeEntry.label = document.getElementById('p-label').value;
  document.getElementById('ph-title').textContent = S.activeEntry.label || 'Polygon';
  renderLayers(); // keep layer-row label in sync
});

document.getElementById('p-class').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  S.activeEntry.class = parseInt(document.getElementById('p-class').value) || 0;
});

function applyAppearance() {
  if (_suppress || !S.activeEntry) return;
  const e   = S.activeEntry;
  const hex = document.getElementById('p-color').value;
  const pct = parseInt(document.getElementById('p-opacity').value);
  const sw  = parseFloat(document.getElementById('p-stroke-w').value) || 1;
  e.color       = hex;
  e.opacity     = pct / 100;
  e.strokeWidth = sw;
  document.getElementById('p-opacity-val').textContent = pct + '%';
  const alpha = Math.round(e.opacity * 255).toString(16).padStart(2, '0');
  const obj = e._obj;
  if (obj) {
    obj.set({ fill: hex + alpha, stroke: hex, strokeWidth: sw });
    fc.renderAll();
  }
}

document.getElementById('p-color').addEventListener('input', applyAppearance);
document.getElementById('p-opacity').addEventListener('input', applyAppearance);
document.getElementById('p-stroke-w').addEventListener('change', applyAppearance);

// Position / size → translate/scale vertices
document.getElementById('p-x').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  withHistory(() => {
    const e = S.activeEntry;
    const b = bbox(e.vertices);
    const dx = parseFloat(document.getElementById('p-x').value) - b.minX;
    e.vertices = e.vertices.map(([x, y]) => [x + dx, y]);
    reRenderEntry(e);
    refreshVertexFields(e);
  });
});

document.getElementById('p-y').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  withHistory(() => {
    const e = S.activeEntry;
    const b = bbox(e.vertices);
    const dy = parseFloat(document.getElementById('p-y').value) - b.minY;
    e.vertices = e.vertices.map(([x, y]) => [x, y + dy]);
    reRenderEntry(e);
    refreshVertexFields(e);
  });
});

document.getElementById('p-w').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  withHistory(() => {
    const e   = S.activeEntry;
    const b   = bbox(e.vertices);
    const curW = b.maxX - b.minX;
    const newW = parseFloat(document.getElementById('p-w').value);
    if (curW <= 0) return;
    const sx = newW / curW;
    const cx = (b.minX + b.maxX) / 2;
    e.vertices = e.vertices.map(([x, y]) => [cx + (x - cx) * sx, y]);
    reRenderEntry(e);
    refreshVertexFields(e);
  });
});

document.getElementById('p-h').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  withHistory(() => {
    const e    = S.activeEntry;
    const b    = bbox(e.vertices);
    const curH = b.maxY - b.minY;
    const newH = parseFloat(document.getElementById('p-h').value);
    if (curH <= 0) return;
    const sy = newH / curH;
    const cy = (b.minY + b.maxY) / 2;
    e.vertices = e.vertices.map(([x, y]) => [x, cy + (y - cy) * sy]);
    reRenderEntry(e);
    refreshVertexFields(e);
  });
});

document.getElementById('p-rot').addEventListener('change', () => {
  if (_suppress || !S.activeEntry) return;
  withHistory(() => {
    const e     = S.activeEntry;
    const angle = parseFloat(document.getElementById('p-rot').value) || 0;
    const [cx, cy] = centerOf(e.vertices);
    const rad   = (angle * Math.PI) / 180;
    const cos   = Math.cos(rad);
    const sin   = Math.sin(rad);
    e.vertices  = e.vertices.map(([x, y]) => {
      const dx = x - cx, dy = y - cy;
      return [cx + dx * cos - dy * sin, cy + dx * sin + dy * cos];
    });
    reRenderEntry(e);
    refreshTransformFields(e);
    refreshVertexFields(e);
  });
});

// ── Vertex editing ─────────────────────────────────────────────────────────────
function onVertexEdit() {
  if (!S.activeEntry) return;
  withHistory(() => {
    const e   = S.activeEntry;
    const rows = document.querySelectorAll('#verts-list .vert-row');
    e.vertices = Array.from(rows).map(row => [
      parseFloat(row.querySelector('.vx').value),
      parseFloat(row.querySelector('.vy').value),
    ]);
    reRenderEntry(e);
    refreshTransformFields(e);
  });
}

function onVertexDelete(ev) {
  if (!S.activeEntry) return;
  const row = ev.target.closest('.vert-row');
  const vi  = parseInt(row.dataset.vi);
  if (S.activeEntry.vertices.length <= 3) return; // minimum triangle
  withHistory(() => {
    S.activeEntry.vertices.splice(vi, 1);
    reRenderEntry(S.activeEntry);
    refreshTransformFields(S.activeEntry);
    refreshVertexFields(S.activeEntry);
  });
}

document.getElementById('btn-add-vert').addEventListener('click', () => {
  if (!S.activeEntry) return;
  const e = S.activeEntry;
  const last = e.vertices[e.vertices.length - 1];
  const first = e.vertices[0];
  withHistory(() => {
    e.vertices.push([(last[0] + first[0]) / 2, (last[1] + first[1]) / 2]);
    reRenderEntry(e);
    refreshTransformFields(e);
    refreshVertexFields(e);
  });
});

// ── Delete polygon ─────────────────────────────────────────────────────────────
document.getElementById('btn-delete').addEventListener('click', () => {
  const e = S.activeEntry;
  if (!e) return;
  pushHistory();
  if (e._obj) fc.remove(e._obj);
  S.elements = S.elements.filter(x => x !== e);
  S.rooms    = S.rooms.filter(x => x !== e);
  clearPanel();
  fc.renderAll();
  renderLayers();
});

// ── Undo ──────────────────────────────────────────────────────────────────────
document.getElementById('btn-undo').addEventListener('click', () => {
  if (S.history.length === 0) return;
  const snap = S.history.pop();
  S.elements = snap.elements;
  S.rooms    = snap.rooms;
  clearPanel();
  renderPolygons(); // already calls renderLayers()
});

// ── Toolbar: zoom ─────────────────────────────────────────────────────────────
document.getElementById('btn-zoom-in').addEventListener('click',  () => setZoom(fc.getZoom() * 1.2));
document.getElementById('btn-zoom-out').addEventListener('click', () => setZoom(fc.getZoom() / 1.2));
document.getElementById('btn-100').addEventListener('click', () => {
  fc.setViewportTransform([1, 0, 0, 1, 0, 0]);
  fc.renderAll();
});
document.getElementById('btn-fit').addEventListener('click', () => {
  fc.setViewportTransform([1, 0, 0, 1, 0, 0]);
  fc.renderAll();
});

function setZoom(z) {
  z = Math.min(Math.max(z, 0.05), 20);
  fc.setZoom(z);
  fc.renderAll();
}

// Wheel zoom (centered on cursor)
document.getElementById('canvas-wrap').addEventListener('wheel', ev => {
  ev.preventDefault();
  const z = fc.getZoom() * (ev.deltaY > 0 ? 0.9 : 1.1);
  const pt = fc.getPointer(ev);
  fc.zoomToPoint(new fabric.Point(pt.x, pt.y), Math.min(Math.max(z, 0.05), 20));
  fc.renderAll();
}, { passive: false });

// Alt+drag to pan
let _panning = false, _panX = 0, _panY = 0;
fc.on('mouse:down', ev => {
  if (!ev.e.altKey) return;
  _panning = true;
  _panX = ev.e.clientX;
  _panY = ev.e.clientY;
  fc.selection = false;
});
fc.on('mouse:move', ev => {
  if (!_panning) return;
  fc.relativePan(new fabric.Point(ev.e.clientX - _panX, ev.e.clientY - _panY));
  _panX = ev.e.clientX;
  _panY = ev.e.clientY;
});
fc.on('mouse:up', () => { _panning = false; fc.selection = true; });

// ── Toolbar: layer toggles (kept in sync with Layers-panel group checkboxes) ──
function setGroupVisible(group, visible) {
  if (group === 'elements') S.showElements = visible;
  else                      S.showRooms    = visible;
  // sync toolbar and layers-panel checkboxes
  document.getElementById(`tog-${group}`).checked = visible;
  document.getElementById(`lg-${group}`).checked  = visible;
  applyGroupVisibility();
  renderLayers(); // refresh disabled state on individual checkboxes
}

document.getElementById('tog-elements').addEventListener('change', e =>
  setGroupVisible('elements', e.target.checked));
document.getElementById('tog-rooms').addEventListener('change', e =>
  setGroupVisible('rooms', e.target.checked));
document.getElementById('tog-image').addEventListener('change', e => {
  S.showImage = e.target.checked;
  if (S.imgObj) S.imgObj.set('visible', S.showImage);
  fc.renderAll();
});

// ── Layers-panel: group toggles ───────────────────────────────────────────────
document.getElementById('lg-elements').addEventListener('change', e =>
  setGroupVisible('elements', e.target.checked));
document.getElementById('lg-rooms').addEventListener('change', e =>
  setGroupVisible('rooms', e.target.checked));

// ── Layers-panel: All / None bulk buttons ─────────────────────────────────────
function setBulkVisibility(group, visible) {
  const entries = group === 'elements' ? S.elements : S.rooms;
  const groupVisible = group === 'elements' ? S.showElements : S.showRooms;
  entries.forEach(e => {
    e.visible = visible;
    const show = groupVisible && visible;
    if (e._obj) e._obj.set({ visible: show, selectable: show, evented: show });
  });
  fc.renderAll();
  renderLayers();
}

document.getElementById('lg-elements-all').addEventListener('click',  () => setBulkVisibility('elements', true));
document.getElementById('lg-elements-none').addEventListener('click', () => setBulkVisibility('elements', false));
document.getElementById('lg-rooms-all').addEventListener('click',     () => setBulkVisibility('rooms', true));
document.getElementById('lg-rooms-none').addEventListener('click',    () => setBulkVisibility('rooms', false));

// ── Shift+click: relative position between two selected polygons ───────────────
let _firstSel = null;
fc.on('mouse:down', ev => {
  if (!ev.e.shiftKey) return;
  const obj = fc.findTarget(ev.e);
  if (!obj?.data) { _firstSel = null; return; }

  if (!_firstSel) {
    _firstSel = obj.data;
  } else if (_firstSel !== obj.data) {
    const a = _firstSel, b = obj.data;
    const ca = centerOf(a.vertices);
    const cb = centerOf(b.vertices);
    const dx = (cb[0] - ca[0]).toFixed(1);
    const dy = (cb[1] - ca[1]).toFixed(1);
    const dist = Math.sqrt((cb[0]-ca[0])**2 + (cb[1]-ca[1])**2).toFixed(1);
    document.getElementById('rel-pos-info').innerHTML =
      `<b>${a.label}</b> → <b>${b.label}</b><br>` +
      `ΔX: ${dx} px &nbsp; ΔY: ${dy} px<br>Distance: ${dist} px`;
    _firstSel = null;
  }
});

// ── Export JSON ───────────────────────────────────────────────────────────────
document.getElementById('btn-export').addEventListener('click', () => {
  const out = {
    elements: S.elements.map(stripObj),
    rooms:    S.rooms.map(stripObj),
  };
  const blob = new Blob([JSON.stringify(out, null, 2)], { type: 'application/json' });
  const a    = document.createElement('a');
  a.href     = URL.createObjectURL(blob);
  a.download = 'floorplan_polygons.json';
  a.click();
  URL.revokeObjectURL(a.href);
});
