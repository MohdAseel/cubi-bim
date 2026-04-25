'use strict';

// ── Color helper: convert #rrggbb + 0-1 opacity to rgba() ─────────────────────
function hexAlpha(hex, opacity) {
  const r = parseInt(hex.slice(1, 3), 16);
  const g = parseInt(hex.slice(3, 5), 16);
  const b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${opacity})`;
}

console.log("CubiCasa: Initializing canvas extensions...");

// ── Labels on Canvas ──────────────────────────────────────────────────────────
// Override Polygon rendering to draw the label text inside the shape
// We do this globally before any objects are instantiated.
fabric.Polygon.prototype._render = (function(render) {
  return function(ctx) {
    render.call(this, ctx); // Draw the shape first
    if (!this.data || !this.data.label) return;

    ctx.save();

    // ── Compute available space inside the polygon ──────────────────────────
    const pts = this.points;
    const xs  = pts.map(p => p.x - this.pathOffset.x);
    const ys  = pts.map(p => p.y - this.pathOffset.y);
    const polyW = Math.max(...xs) - Math.min(...xs);
    const polyH = Math.max(...ys) - Math.min(...ys);
    const shortSide = Math.min(polyW, polyH);

    // ── Font size: 18% of short side, clamped 8–16 px (canvas units) ───────
    const fontSize = Math.max(8, Math.min(16, shortSide * 0.18));
    ctx.font = `bold ${fontSize}px 'Segoe UI', Roboto, sans-serif`;
    ctx.textAlign    = 'center';
    ctx.textBaseline = 'middle';

    // ── Split label into lines that fit ────────────────────────────────────
    const label   = this.data.label;
    const maxW    = polyW * 0.85;
    const words   = label.split(' ');
    const lines   = [];
    let current   = '';
    for (const word of words) {
      const test = current ? `${current} ${word}` : word;
      if (ctx.measureText(test).width <= maxW || !current) {
        current = test;
      } else {
        lines.push(current);
        current = word;
      }
    }
    if (current) lines.push(current);

    // ── Skip if text still too wide (tiny polygon) ─────────────────────────
    const maxLineW = Math.max(...lines.map(l => ctx.measureText(l).width));
    if (maxLineW > polyW || lines.length * fontSize > polyH * 0.9) {
      ctx.restore();
      return;
    }

    // ── Draw pill background ───────────────────────────────────────────────
    const lineH   = fontSize * 1.35;
    const totalH  = lines.length * lineH;
    const padX    = fontSize * 0.55;
    const padY    = fontSize * 0.3;
    const pillW   = maxLineW + padX * 2;
    const pillH   = totalH   + padY * 2;
    const rx      = Math.min(pillW / 2, fontSize * 0.6);
    const px      = -pillW / 2;
    const py      = -pillH / 2;

    ctx.beginPath();
    ctx.moveTo(px + rx, py);
    ctx.lineTo(px + pillW - rx, py);
    ctx.quadraticCurveTo(px + pillW, py, px + pillW, py + rx);
    ctx.lineTo(px + pillW, py + pillH - rx);
    ctx.quadraticCurveTo(px + pillW, py + pillH, px + pillW - rx, py + pillH);
    ctx.lineTo(px + rx, py + pillH);
    ctx.quadraticCurveTo(px, py + pillH, px, py + pillH - rx);
    ctx.lineTo(px, py + rx);
    ctx.quadraticCurveTo(px, py, px + rx, py);
    ctx.closePath();
    ctx.fillStyle = 'rgba(15, 23, 42, 0.72)';
    ctx.fill();

    // ── Draw text lines ────────────────────────────────────────────────────
    ctx.fillStyle = '#ffffff';
    const startY  = -(lines.length - 1) * lineH / 2;
    lines.forEach((line, i) => {
      ctx.fillText(line, 0, startY + i * lineH);
    });

    ctx.restore();
  };
})(fabric.Polygon.prototype._render);

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
  redoHistory: [], // redo stack
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
  const show  = groupVisible && (entry.visible !== false);

  const poly = new fabric.Polygon(pts, {
    fill:               hexAlpha(hex, entry.opacity ?? 0.3),
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
        // stay on Layers tab — don't force-switch to Properties
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
  S.redoHistory = []; // Any new edit clears redo history
}
function stripObj(e) { const c = { ...e }; delete c._obj; return c; }

const TARGET_SIZE = 512;

// ── Preprocessing ─────────────────────────────────────────────────────────────

/**
 * Samples the edges of the image bitmap to determine a suitable background 
 * color for padding (e.g., white for most floorplans).
 */
function detectBackgroundColor(bitmap) {
  const s = 16;
  const canvas = document.createElement('canvas');
  canvas.width = canvas.height = s;
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  ctx.drawImage(bitmap, 0, 0, s, s);
  
  const data = ctx.getImageData(0, 0, s, s).data;
  const samples = [[0,0], [s-1,0], [0,s-1], [s-1,s-1], [s/2,0], [0,s/2]];
  
  let r=0, g=0, b=0;
  samples.forEach(([x, y]) => {
    const i = (Math.floor(y) * s + Math.floor(x)) * 4;
    r += data[i]; g += data[i+1]; b += data[i+2];
  });
  
  const n = samples.length;
  return `rgb(${Math.round(r/n)}, ${Math.round(g/n)}, ${Math.round(b/n)})`;
}

/**
 * Resizes the longest side of the image to 512px and pads it to 512x512.
 * Uses intelligent background color detection for padding.
 */
async function preprocessForModel(file) {
  try {
    const bitmap = await createImageBitmap(file);
    const canvas = document.createElement('canvas');
    canvas.width  = TARGET_SIZE;
    canvas.height = TARGET_SIZE;
    const ctx = canvas.getContext('2d');

    // 1. Detect background color from the edges
    const bgColor = detectBackgroundColor(bitmap);

    // 2. Fill with detected color (padding)
    ctx.fillStyle = bgColor;
    ctx.fillRect(0, 0, TARGET_SIZE, TARGET_SIZE);

    const w = bitmap.width;
    const h = bitmap.height;
    const longSide = Math.max(w, h);
    const scale = TARGET_SIZE / longSide;
    const nw = Math.round(w * scale);
    const nh = Math.round(h * scale);

    // 3. Draw image resized at (0,0) (top-left placement, matching inference.py)
    ctx.drawImage(bitmap, 0, 0, nw, nh);
    bitmap.close();

    return new Promise(resolve => {
      canvas.toBlob(blob => resolve({ blob, width: nw, height: nh }), 'image/jpeg', 0.9);
    });
  } catch (err) {
    throw new Error('Failed to preprocess image: ' + err.message);
  }
}

// ── Upload + inference ─────────────────────────────────────────────────────────

async function handleFile(file) {
  if (!file || !file.type.startsWith('image/')) return;

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
      S.redoHistory = [];
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

      // scale: (model-space px) → canvas px
      // inference operates in a TARGET_SIZE longest-side space
      const resizeFactor = TARGET_SIZE / longSide;
      S.polyScale = fitScale / resizeFactor;
      S.imgOffX   = imgLeft;
      S.imgOffY   = imgTop;
      S.imgObj    = img;

      fc.add(img);
      fc.renderAll();
      resolve();
    }, { crossOrigin: null });
  });

  // 2. Preprocess and Call backend
  let data;
  try {
    document.getElementById('status-text').textContent = 'Preprocessing image…';
    const preprocessed = await preprocessForModel(file);

    document.getElementById('status-text').textContent = 'Running inference…';
    const form = new FormData();
    form.append('image', preprocessed.blob, 'preprocessed.jpg');

    const resp = await fetch('/predict', { method: 'POST', body: form });
    if (!resp.ok) throw new Error(await resp.text());
    data = await resp.json();
  } catch (err) {
    console.error(err);
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
}

document.getElementById('file-input').addEventListener('change', e => {
  const file = e.target.files[0];
  if (file) {
    handleFile(file);
    e.target.value = '';
  }
});

// Drag & Drop
const wrap = document.getElementById('canvas-wrap');
wrap.addEventListener('dragover', e => {
  e.preventDefault();
  wrap.classList.add('drag-over');
});
wrap.addEventListener('dragleave', e => {
  e.preventDefault();
  wrap.classList.remove('drag-over');
});
wrap.addEventListener('drop', e => {
  e.preventDefault();
  wrap.classList.remove('drag-over');
  const file = e.dataTransfer.files[0];
  if (file) handleFile(file);
});

// ── Fabric selection events ────────────────────────────────────────────────────
fc.on('selection:created', e  => onSelect(e.selected?.[0]));
fc.on('selection:updated', e  => onSelect(e.selected?.[0]));
fc.on('selection:cleared',  () => clearPanel());

// Snapshot captured on mouse:down so undo restores the state BEFORE the drag
let _preModSnap = null;
fc.on('mouse:down', ev => {
  if (!ev.e.altKey && ev.target?.data) {
    _preModSnap = {
      elements: JSON.parse(JSON.stringify(S.elements.map(stripObj))),
      rooms:    JSON.parse(JSON.stringify(S.rooms.map(stripObj))),
    };
  }
});

fc.on('object:modified', e => {
  const obj = e.target;
  if (!obj?.data) return;
  if (_preModSnap) {
    S.history.push(_preModSnap);
    if (S.history.length > 50) S.history.shift();
    _preModSnap = null;
  }
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
  const obj = e._obj;
  if (obj) {
    obj.set({ fill: hexAlpha(hex, e.opacity), stroke: hex, strokeWidth: sw });
    fc.renderAll();
  }
}

// ── Hover Tooltip ──────────────────────────────────────────────────────────────

/**
 * Shoelace formula to compute polygon area from vertices array [[x,y], ...].
 */
function polyArea(verts) {
  let area = 0;
  const n = verts.length;
  for (let i = 0; i < n; i++) {
    const [x1, y1] = verts[i];
    const [x2, y2] = verts[(i + 1) % n];
    area += x1 * y2 - x2 * y1;
  }
  return Math.abs(area / 2);
}

fc.on('mouse:over', e => {
  const obj = e.target;
  if (!obj || !obj.data) return;

  const entry = obj.data;
  const b   = bbox(entry.vertices);
  const w   = (b.maxX - b.minX).toFixed(1);
  const h   = (b.maxY - b.minY).toFixed(1);
  const area  = polyArea(entry.vertices).toFixed(1);
  const verts = entry.vertices.length;
  const typeClass = entry.type || 'unknown';

  // Badge color class
  const badgeMap = { room: 'room', icon: 'icon', wall: 'wall' };
  const badgeCls = badgeMap[typeClass] || '';

  const tt = document.getElementById('tooltip');
  tt.innerHTML = `
    <div class="tt-title">${entry.label || 'Polygon'}</div>
    <span class="tt-badge ${badgeCls}">${typeClass}</span>
    <div class="tt-grid">
      <span class="tt-key">Class</span><span class="tt-val">${entry.class ?? '—'}</span>
      <span class="tt-key">Width</span><span class="tt-val">${w} px</span>
      <span class="tt-key">Height</span><span class="tt-val">${h} px</span>
      <span class="tt-key">Area</span><span class="tt-val">${area} px²</span>
      <span class="tt-key">Vertices</span><span class="tt-val">${verts}</span>
    </div>
  `;
  tt.classList.remove('hidden');
});

fc.on('mouse:move', e => {
  const tt = document.getElementById('tooltip');
  if (tt.classList.contains('hidden')) return;
  // Keep tooltip inside the viewport
  const tw  = tt.offsetWidth;
  const th  = tt.offsetHeight;
  const vw  = window.innerWidth;
  const vh  = window.innerHeight;
  let lx = e.e.clientX + 18;
  let ly = e.e.clientY + 14;
  if (lx + tw > vw) lx = e.e.clientX - tw - 10;
  if (ly + th > vh) ly = e.e.clientY - th - 10;
  tt.style.left = lx + 'px';
  tt.style.top  = ly + 'px';
});

fc.on('mouse:out', () => {
  document.getElementById('tooltip').classList.add('hidden');
});

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

// ── Undo / Redo ──────────────────────────────────────────────────────────────
function getSnap() {
  return {
    elements: JSON.parse(JSON.stringify(S.elements.map(stripObj))),
    rooms:    JSON.parse(JSON.stringify(S.rooms.map(stripObj))),
  };
}

function performUndo() {
  if (S.history.length === 0) return;
  S.redoHistory.push(getSnap());
  const snap = S.history.pop();
  S.elements = snap.elements;
  S.rooms    = snap.rooms;
  clearPanel();
  renderPolygons();
}

function performRedo() {
  if (S.redoHistory.length === 0) return;
  S.history.push(getSnap());
  const snap = S.redoHistory.pop();
  S.elements = snap.elements;
  S.rooms    = snap.rooms;
  clearPanel();
  renderPolygons();
}

document.getElementById('btn-undo').addEventListener('click', performUndo);
document.getElementById('btn-redo').addEventListener('click', performRedo);

document.addEventListener('keydown', ev => {
  const isZ = ev.key.toLowerCase() === 'z';
  const isY = ev.key.toLowerCase() === 'y';
  const ctrl = ev.ctrlKey || ev.metaKey;
  const shift = ev.shiftKey;

  if (ctrl && (isZ || isY)) {
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'TEXTAREA') return;
    ev.preventDefault();

    if (isY || (isZ && shift)) {
      performRedo();
    } else if (isZ) {
      performUndo();
    }
  }
});

// ── Toolbar: zoom ─────────────────────────────────────────────────────────────
document.getElementById('btn-zoom-in').addEventListener('click',  () => setZoom(fc.getZoom() * 1.2));
document.getElementById('btn-zoom-out').addEventListener('click', () => setZoom(fc.getZoom() / 1.2));
document.getElementById('btn-100').addEventListener('click', () => {
  const cx = fc.width / 2;
  const cy = fc.height / 2;
  fc.zoomToPoint(new fabric.Point(cx, cy), 1);
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
