import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const $ = (id) => document.getElementById(id);
const el = (tag, cls, text) => {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (text !== undefined) e.textContent = text;
  return e;
};
const fmt = (v) => {
  if (v === null || v === undefined) return '-';
  if (typeof v !== 'number') return String(v);
  if (v === 0) return '0';
  const a = Math.abs(v);
  return a >= 1e5 || a < 1e-3 ? v.toExponential(3) : String(+v.toPrecision(5));
};

let st = null;
let highlightId = null;
let picked = null;
const R = { mesh: null, frames: new Map(), index: 0, field: 'U', scale: 1, playing: null, auto: true };

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) {
    o.method = o.method || 'POST';
    o.headers = { 'Content-Type': 'application/json' };
    o.body = JSON.stringify(o.json);
  }
  const r = await fetch(path, o);
  let data = null;
  try {
    data = await r.json();
  } catch (e) {
    data = null;
  }
  if (!r.ok) throw new Error((data && data.detail) || `요청 실패 (${r.status})`);
  return data;
}

const canvas = $('gl');
const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(40, 1, 0.1, 1e7);
camera.up.set(0, 0, 1);
camera.position.set(300, -400, 250);
const controls = new OrbitControls(camera, canvas);
controls.enableDamping = true;
controls.dampingFactor = 0.12;
scene.add(new THREE.HemisphereLight(0xffffff, 0x334455, 1.15));
const sun = new THREE.DirectionalLight(0xffffff, 1.4);
scene.add(sun);
const G = { model: new THREE.Group(), edges: new THREE.Group(), symbols: new THREE.Group(), result: new THREE.Group(), pick: new THREE.Group(), rigid: new THREE.Group() };
Object.values(G).forEach((g) => scene.add(g));
let diag = 100;
let center = new THREE.Vector3();

function resize() {
  const r = canvas.parentElement.getBoundingClientRect();
  renderer.setSize(r.width, r.height, false);
  camera.aspect = r.width / Math.max(r.height, 1);
  camera.updateProjectionMatrix();
}
new ResizeObserver(resize).observe(canvas.parentElement);

function clear(g) {
  while (g.children.length) {
    const c = g.children.pop();
    c.traverse((o) => {
      if (o.geometry) o.geometry.dispose();
      if (o.material) o.material.dispose();
    });
  }
}

function drawGizmo() {
  const c = $('gizmo');
  const x = c.getContext('2d');
  x.clearRect(0, 0, 96, 96);
  const q = camera.quaternion.clone().invert();
  const axes = [
    ['X', new THREE.Vector3(1, 0, 0), '#ff5c5c'],
    ['Y', new THREE.Vector3(0, 1, 0), '#57d9a3'],
    ['Z', new THREE.Vector3(0, 0, 1), '#4cc2ff'],
  ].map(([n, v, col]) => [n, v.applyQuaternion(q), col]);
  axes.sort((a, b) => a[1].z - b[1].z);
  x.font = '11px sans-serif';
  x.textAlign = 'center';
  x.textBaseline = 'middle';
  for (const [n, v, col] of axes) {
    const px = 48 + v.x * 30;
    const py = 48 - v.y * 30;
    x.strokeStyle = col;
    x.fillStyle = col;
    x.lineWidth = 2;
    x.beginPath();
    x.moveTo(48, 48);
    x.lineTo(px, py);
    x.stroke();
    x.fillText(n, 48 + v.x * 40, 48 - v.y * 40);
  }
}

function loop() {
  controls.update();
  sun.position.copy(camera.position);
  renderer.render(scene, camera);
  drawGizmo();
  requestAnimationFrame(loop);
}

function fit() {
  if (!st || !st.view) return;
  const b = new THREE.Box3();
  for (const p of st.view.nodes) b.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
  center = b.getCenter(new THREE.Vector3());
  diag = Math.max(b.getSize(new THREE.Vector3()).length(), 1e-6);
  controls.target.copy(center);
  camera.position.copy(center).add(new THREE.Vector3(0.9, -1.2, 0.75).multiplyScalar(diag));
  camera.near = diag / 500;
  camera.far = diag * 100;
  camera.updateProjectionMatrix();
}

function triangulate(faces) {
  const t = [];
  for (const f of faces) {
    t.push(f[0], f[1], f[2]);
    if (f.length === 4) t.push(f[0], f[2], f[3]);
  }
  return t;
}

function buildModel() {
  clear(G.model);
  clear(G.edges);
  if (!st || !st.view) return;
  const v = st.view;
  const pos = new Float32Array(v.nodes.flat());
  const tris = triangulate(v.shells).concat(v.boundary.flat());
  if (tris.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(tris);
    const flat = g.toNonIndexed();
    flat.computeVertexNormals();
    g.dispose();
    const m = new THREE.MeshLambertMaterial({ color: 0x9fb3c8, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 });
    G.model.add(new THREE.Mesh(flat, m));
  }
  const seen = new Set();
  const ed = [];
  const add = (a, b) => {
    const k = a < b ? a * 1e7 + b : b * 1e7 + a;
    if (seen.has(k)) return;
    seen.add(k);
    ed.push(a, b);
  };
  for (const f of v.shells) for (let i = 0; i < f.length; i++) add(f[i], f[(i + 1) % f.length]);
  for (const f of v.boundary) for (let i = 0; i < 3; i++) add(f[i], f[(i + 1) % 3]);
  if (ed.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(ed);
    G.edges.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x2c3948 })));
  }
  if (v.beams.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(v.beams.flat());
    G.model.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xe8eef6 })));
    const pg = new THREE.BufferGeometry();
    const used = [...new Set(v.beams.flat())];
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(used.flatMap((i) => v.nodes[i])), 3));
    G.model.add(new THREE.Points(pg, new THREE.PointsMaterial({ color: 0xe8eef6, size: 4, sizeAttenuation: false })));
  }
  const h = Object.values(v.hinges).flat();
  if (h.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(h.flat());
    G.model.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0xff5fa8, depthTest: false })));
  }
}

function sample(n, max) {
  if (n <= max) return [...Array(n).keys()];
  const out = [];
  for (let i = 0; i < max; i++) out.push(Math.floor((i * n) / max));
  return out;
}

function placeRigid(t) {
  for (const o of G.rigid.children) {
    const b = o.userData.base;
    const m = o.userData.motion;
    if (b) o.position.set(b[0] + t * m[0], b[1] + t * m[1], b[2] + t * m[2]);
  }
}

function buildRigid() {
  clear(G.rigid);
  if (!st || !st.preview || !st.preview.contacts) return;
  for (const c of st.preview.contacts) {
    if (c.type !== 'rigid') continue;
    const hot = c.id === highlightId;
    const mat = new THREE.MeshBasicMaterial({ color: hot ? 0xffc857 : 0x9aa7b8, wireframe: true, transparent: true, opacity: 0.55 });
    let o;
    if (c.shape === 'sphere') {
      o = new THREE.Mesh(new THREE.SphereGeometry(c.radius, 28, 18), mat);
      o.userData.base = c.center;
    } else {
      o = new THREE.Mesh(new THREE.PlaneGeometry(2 * c.size, 2 * c.size, 12, 12), mat);
      o.quaternion.setFromUnitVectors(new THREE.Vector3(0, 0, 1), new THREE.Vector3(...c.normal).normalize());
      const cen = [0, 1, 2].map((k) => (st.view.nodes.reduce((a, p) => Math.min(a, p[k]), Infinity) + st.view.nodes.reduce((a, p) => Math.max(a, p[k]), -Infinity)) / 2);
      const n = new THREE.Vector3(...c.normal).normalize();
      const d = new THREE.Vector3(c.point[0] - cen[0], c.point[1] - cen[1], c.point[2] - cen[2]).dot(n);
      o.userData.base = [cen[0] + d * n.x, cen[1] + d * n.y, cen[2] + d * n.z];
    }
    o.userData.motion = c.motion;
    G.rigid.add(o);
    const mv = new THREE.Vector3(...c.motion);
    if (mv.length() > 0) {
      const len = Math.max(mv.length(), diag * 0.08);
      const a = new THREE.ArrowHelper(mv.clone().normalize(), new THREE.Vector3(0, 0, 0), len, 0x6ee7b7, diag * 0.03, diag * 0.015);
      a.userData.base = o.userData.base;
      a.userData.motion = c.motion;
      G.rigid.add(a);
    }
  }
  placeRigid(0);
}

function buildSymbols() {
  clear(G.symbols);
  buildRigid();
  if (!st || !st.view || !st.preview) return;
  const nodes = st.view.nodes;
  const s = diag * 0.035;
  for (const c of st.preview.contacts || []) {
    if (c.type !== 'pair' || !c.a) continue;
    for (const [ids, col] of [[c.a, 0x6ee7b7], [c.b, 0xf0a35e]]) {
      const pg = new THREE.BufferGeometry();
      const arr = [];
      for (const k of sample(ids.length, 1500)) arr.push(...nodes[ids[k]]);
      pg.setAttribute('position', new THREE.Float32BufferAttribute(arr, 3));
      G.symbols.add(new THREE.Points(pg, new THREE.PointsMaterial({ color: c.id === highlightId ? 0xffc857 : col, size: 4, sizeAttenuation: false, depthTest: false })));
    }
  }
  const AX = [new THREE.Vector3(1, 0, 0), new THREE.Vector3(0, 1, 0), new THREE.Vector3(0, 0, 1)];
  for (const bc of st.preview.bcs) {
    const hot = bc.id === highlightId;
    const seg = [];
    const rot = [];
    for (const k of sample(bc.nodes.length, 600)) {
      const p = nodes[bc.nodes[k]];
      for (const d of bc.dofs) {
        if (d <= 3) {
          const a = AX[d - 1];
          seg.push(p[0] - a.x * s, p[1] - a.y * s, p[2] - a.z * s, p[0] + a.x * s, p[1] + a.y * s, p[2] + a.z * s);
        }
      }
      if (bc.dofs.some((d) => d > 3)) rot.push(p[0], p[1], p[2]);
    }
    const col = hot ? 0xffc857 : bc.value !== 0 ? 0x57d9a3 : 0x4cc2ff;
    if (seg.length) {
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(seg), 3));
      G.symbols.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: col, depthTest: false })));
    }
    const pg = new THREE.BufferGeometry();
    pg.setAttribute('position', new THREE.BufferAttribute(new Float32Array(sample(bc.nodes.length, 3000).flatMap((k) => nodes[bc.nodes[k]])), 3));
    G.symbols.add(new THREE.Points(pg, new THREE.PointsMaterial({ color: col, size: rot.length ? 9 : 6, sizeAttenuation: false, depthTest: false })));
    if (bc.value !== 0) {
      const d = bc.dofs.find((x) => x <= 3);
      if (d) {
        const dir = AX[d - 1].clone().multiplyScalar(Math.sign(bc.value));
        for (const k of sample(bc.nodes.length, 40)) {
          const p = new THREE.Vector3(...nodes[bc.nodes[k]]);
          G.symbols.add(new THREE.ArrowHelper(dir, p.clone().sub(dir.clone().multiplyScalar(s * 3)), s * 3, col, s, s * 0.5));
        }
      }
    }
  }
  for (const ld of st.preview.loads) {
    const hot = ld.id === highlightId;
    const col = hot ? 0xffc857 : ld.type === 'moment' ? 0xc792ea : 0xff5c5c;
    if (ld.type === 'gravity') {
      const dir = new THREE.Vector3(...ld.total);
      if (dir.length() > 0) G.symbols.add(new THREE.ArrowHelper(dir.normalize(), center.clone().sub(dir.clone().multiplyScalar(s * 6)), s * 6, col, s * 1.5, s * 0.8));
      continue;
    }
    let vmax = 0;
    for (const v of ld.vectors) vmax = Math.max(vmax, Math.hypot(v[0], v[1], v[2]));
    if (vmax === 0) continue;
    for (const k of sample(ld.nodes.length, 300)) {
      const v = new THREE.Vector3(...ld.vectors[k]);
      const mag = v.length();
      if (mag === 0) continue;
      const len = s * (1.5 + 3 * (mag / vmax));
      const p = new THREE.Vector3(...nodes[ld.nodes[k]]);
      const dir = v.normalize();
      const a = new THREE.ArrowHelper(dir, p.clone().sub(dir.clone().multiplyScalar(len)), len, col, s * 0.9, s * 0.45);
      a.traverse((o) => {
        if (o.material) o.material.depthTest = false;
      });
      G.symbols.add(a);
    }
  }
}

const RAMP = [
  [0.17, 0.29, 1.0],
  [0.1, 0.76, 0.49],
  [1.0, 0.88, 0.29],
  [1.0, 0.23, 0.19],
];
function ramp(t, out, o) {
  const x = Math.min(Math.max(t, 0), 1) * 3;
  const i = Math.min(Math.floor(x), 2);
  const f = x - i;
  for (let k = 0; k < 3; k++) out[o + k] = RAMP[i][k] + (RAMP[i + 1][k] - RAMP[i][k]) * f;
}

async function loadResult() {
  clear(G.result);
  R.mesh = null;
  R.frames.clear();
  stopPlay();
  if (!st || !st.result) {
    $('resultbar').hidden = true;
    $('legend').hidden = true;
    applyVisibility();
    return;
  }
  R.mesh = await api('/api/result/mesh');
  const n = R.mesh.nodes.length;
  const pos = new Float32Array(n * 3);
  const col = new Float32Array(n * 3);
  if (R.mesh.tris.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(R.mesh.tris.flat());
    G.result.add(new THREE.Mesh(g, new THREE.MeshLambertMaterial({ vertexColors: true, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1 })));
  }
  if (R.mesh.lines.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(R.mesh.lines.flat());
    const dark = R.mesh.tris.length > 0;
    G.result.add(new THREE.LineSegments(g, new THREE.LineBasicMaterial(dark ? { color: 0x10151c } : { vertexColors: true })));
  }
  if (R.mesh.fedges && R.mesh.fedges.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setIndex(R.mesh.fedges.flat());
    const ls = new THREE.LineSegments(g, new THREE.LineBasicMaterial({ color: 0x0b0f14, transparent: true, opacity: 0.55 }));
    ls.userData.meshLines = true;
    G.result.add(ls);
  }
  R.pos = pos;
  R.col = col;
  const frames = st.result.frames;
  $('frame').max = frames.length - 1;
  const kind = st.result.kind;
  R.index = kind === 'static' ? frames.length - 1 : 0;
  $('frame').value = R.index;
  const fsel = $('field');
  fsel.innerHTML = '';
  for (const f of st.result.fields) fsel.append(new Option(f, f));
  R.field = kind === 'static' && st.result.fields.includes('vonMises') ? 'vonMises' : st.result.fields.includes('foldRotation') ? 'foldRotation' : 'U';
  fsel.value = R.field;
  R.auto = true;
  $('resultbar').hidden = false;
  $('showResult').checked = true;
  await showFrame();
  applyVisibility();
}

async function getFrame(i, field) {
  const k = `${i}|${field}`;
  if (!R.frames.has(k)) R.frames.set(k, await api(`/api/result/frame?i=${i}&field=${encodeURIComponent(field)}`));
  return R.frames.get(k);
}

async function showFrame() {
  if (!R.mesh) return;
  const f = await getFrame(R.index, R.field);
  if (R.auto) {
    let umax = 0;
    for (const u of f.U) umax = Math.max(umax, Math.hypot(u[0], u[1], u[2]));
    const nl = st.result.kind === 'static' && (st.result.solver === 'barhinge' || st.result.frames.length > 1);
    R.scale = nl || umax === 0 ? 1 : +((0.08 * diag) / umax).toPrecision(2);
    if (st.result.kind === 'static' && umax > 0.02 * diag) R.scale = 1;
    $('scale').value = R.scale;
    R.auto = false;
  }
  const n = R.mesh.nodes.length;
  const lo = f.min;
  const span = f.max - f.min || 1;
  for (let i = 0; i < n; i++) {
    const p = R.mesh.nodes[i];
    const u = f.U[i];
    R.pos[3 * i] = p[0] + R.scale * u[0];
    R.pos[3 * i + 1] = p[1] + R.scale * u[1];
    R.pos[3 * i + 2] = p[2] + R.scale * u[2];
    ramp((f.values[i] - lo) / span, R.col, 3 * i);
  }
  for (const o of G.result.children) {
    o.geometry.attributes.position.needsUpdate = true;
    if (o.geometry.attributes.color) o.geometry.attributes.color.needsUpdate = true;
    if (o.isMesh) o.geometry.computeVertexNormals();
    o.geometry.computeBoundingSphere();
  }
  placeRigid(st.result.kind === 'static' && $('showResult').checked ? R.scale * f.time : 0);
  $('frameLabel').textContent = `${R.index + 1}/${st.result.frames.length} · ${f.label}`;
  $('legend').hidden = !$('showResult').checked;
  $('legendTitle').textContent = R.field;
  const ticks = $('legendTicks');
  ticks.innerHTML = '';
  for (let k = 4; k >= 0; k--) ticks.append(el('span', '', fmt(lo + (span * k) / 4)));
}

function stopPlay() {
  if (R.playing) clearInterval(R.playing);
  R.playing = null;
  $('play').textContent = '▶';
}

function applyVisibility() {
  const res = !!R.mesh && $('showResult').checked;
  G.result.visible = res;
  G.model.visible = !res;
  G.edges.visible = $('showEdges').checked && !res;
  G.symbols.visible = $('showSymbols').checked;
  G.rigid.visible = $('showSymbols').checked;
  if (!res) placeRigid(0);
  $('legend').hidden = !res;
  $('resultbar').hidden = !R.mesh;
  const ml = $('showMeshLines').checked;
  G.result.children.forEach((c) => {
    if (c.userData.meshLines) c.visible = ml;
  });
  $('meshLinesWrap').hidden = !R.mesh;
}

function describeBC(b) {
  const names = ['ux', 'uy', 'uz', 'rx', 'ry', 'rz'];
  return b.dofs.map((d) => names[d - 1]).join(' ');
}

function ops(list, quiet) {
  return api('/api/ops', { json: { ops: list } })
    .then((s) => refresh(s, { keepResult: true }))
    .catch((e) => {
      if (!quiet) note(e.message, true);
      refresh(st, { keepResult: true });
    });
}

function note(msg, err) {
  const m = $('solveMsg');
  m.textContent = msg || '';
  m.style.color = err ? 'var(--err)' : '';
}

function renderLists() {
  const pv = st.preview;
  const bl = $('bcList');
  bl.innerHTML = '';
  $('bcCount').textContent = st.spec.bcs.length ? `${st.spec.bcs.length}개` : '';
  if (!st.spec.bcs.length) bl.append(el('div', 'empty', '없음. 오른쪽에서 문장으로 지시하세요'));
  for (const b of st.spec.bcs) {
    const p = pv.bcs.find((x) => x.id === b.id);
    const it = el('div', 'item' + (highlightId === b.id ? ' sel' : ''));
    const top = el('div', 'top');
    const nm = el('span');
    nm.append(el('span', 'id', b.id), document.createTextNode(b.name || '구속'));
    const x = el('button', 'x', '✕');
    x.onclick = (e) => {
      e.stopPropagation();
      ops([{ op: 'remove', id: b.id }]);
    };
    top.append(nm, x);
    it.append(top, el('div', 'desc', `${p ? p.desc : '?'} · ${describeBC(b)} · 절점 ${p ? p.nodes.length : 0}개`));
    const ed = el('div', 'edit');
    const inp = el('input');
    inp.type = 'number';
    inp.step = 'any';
    inp.value = b.value;
    inp.title = '지정 변위 (mm), 0이면 고정';
    inp.onclick = (e) => e.stopPropagation();
    inp.onchange = () => ops([{ op: 'update', id: b.id, value: +inp.value || 0 }]);
    ed.append(inp);
    it.append(ed);
    it.onclick = () => {
      highlightId = highlightId === b.id ? null : b.id;
      renderLists();
      buildSymbols();
    };
    bl.append(it);
  }
  const ll = $('ldList');
  ll.innerHTML = '';
  $('ldCount').textContent = st.spec.loads.length ? `${st.spec.loads.length}개` : '';
  if (!st.spec.loads.length) ll.append(el('div', 'empty', '없음'));
  const tn = { force: '힘 (N)', moment: '모멘트 (N·mm)', pressure: '압력 (MPa)', gravity: '중력 (mm/s²)' };
  for (const l of st.spec.loads) {
    const p = pv.loads.find((x) => x.id === l.id);
    const it = el('div', 'item' + (highlightId === l.id ? ' sel' : ''));
    const top = el('div', 'top');
    const nm = el('span');
    nm.append(el('span', 'id', l.id), document.createTextNode(`${l.name && l.name !== l.type ? l.name + ' · ' : ''}${tn[l.type]}`));
    const x = el('button', 'x', '✕');
    x.onclick = (e) => {
      e.stopPropagation();
      ops([{ op: 'remove', id: l.id }]);
    };
    top.append(nm, x);
    it.append(top, el('div', 'desc', `${p ? p.desc : '?'}${p && p.total ? ' · 합 [' + p.total.map(fmt).join(', ') + ']' : ''}${p && p.nodes.length ? ' · 절점 ' + p.nodes.length + '개' : ''}`));
    const ed = el('div', 'edit');
    if (l.vector) {
      const ins = l.vector.map((v) => {
        const i = el('input');
        i.type = 'number';
        i.step = 'any';
        i.value = v;
        i.onclick = (e) => e.stopPropagation();
        return i;
      });
      ins.forEach((i) => (i.onchange = () => ops([{ op: 'update', id: l.id, vector: ins.map((q) => +q.value || 0) }])));
      ed.append(...ins);
    } else {
      const i = el('input');
      i.type = 'number';
      i.step = 'any';
      i.value = l.value;
      i.onclick = (e) => e.stopPropagation();
      i.onchange = () => ops([{ op: 'update', id: l.id, value: +i.value || 0 }]);
      ed.append(i);
    }
    it.append(ed);
    it.onclick = () => {
      highlightId = highlightId === l.id ? null : l.id;
      renderLists();
      buildSymbols();
    };
    ll.append(it);
  }
  const cl = $('ctList');
  cl.innerHTML = '';
  const cts = st.spec.contacts || [];
  $('ctCount').textContent = cts.length ? `${cts.length}개` : '';
  if (!cts.length) cl.append(el('div', 'empty', '없음'));
  for (const c of cts) {
    const p = (pv.contacts || []).find((x) => x.id === c.id);
    const it = el('div', 'item' + (highlightId === c.id ? ' sel' : ''));
    const top = el('div', 'top');
    const nm = el('span');
    nm.append(el('span', 'id', c.id), document.createTextNode(c.type === 'rigid' ? (c.shape === 'sphere' ? '강체 구' : '강체 평면') : c.type === 'self' ? '자기 접촉' : '접촉 쌍'));
    const x = el('button', 'x', '✕');
    x.onclick = (e) => {
      e.stopPropagation();
      ops([{ op: 'remove', id: c.id }]);
    };
    top.append(nm, x);
    it.append(top, el('div', 'desc', p && p.desc ? p.desc : ''));
    const ed = el('div', 'edit');
    const mu = el('input');
    mu.type = 'number';
    mu.step = 'any';
    mu.min = 0;
    mu.value = c.friction || 0;
    mu.title = '마찰계수';
    mu.onclick = (e) => e.stopPropagation();
    mu.onchange = () => ops([{ op: 'update', id: c.id, friction: Math.max(0, +mu.value || 0) }]);
    ed.append(mu);
    if (c.type === 'rigid') {
      const ins = c.motion.map((v) => {
        const i = el('input');
        i.type = 'number';
        i.step = 'any';
        i.value = v;
        i.title = '강체 이동량 (mm)';
        i.onclick = (e) => e.stopPropagation();
        return i;
      });
      ins.forEach((i) => (i.onchange = () => ops([{ op: 'update', id: c.id, motion: ins.map((q) => +q.value || 0) }])));
      ed.append(...ins);
    }
    it.append(ed);
    it.onclick = () => {
      highlightId = highlightId === c.id ? null : c.id;
      renderLists();
      buildSymbols();
    };
    cl.append(it);
  }
  const w = $('warnings');
  w.innerHTML = '';
  for (const m of st.warnings) w.append(el('div', '', '⚠ ' + m));
  for (const m of pv.errors) w.append(el('div', 'err', '✕ ' + m));
}

function renderControls() {
  const t = st.tools || {};
  $('tools').innerHTML = `<b class="${t.ccx ? 'on' : 'off'}">CalculiX ${t.ccx ? '●' : '○'}</b> <b class="${t.tensormesh ? 'on' : 'off'}">TensorMesh ${t.tensormesh ? '●' : '○'}</b>`;
  if (!st.model) {
    $('modelInfo').textContent = '불러온 모델이 없습니다';
    return;
  }
  const c = st.model.counts;
  const g = Object.entries(st.model.groups)
    .map(([k, v]) => `${k}${v.kind === 'hinge' ? '(힌지)' : v.kind === 'patch' ? (v.normal ? `(면 n=[${v.normal.join(',')}])` : '(곡면)') : ''}`)
    .join(', ');
  const lo = st.model.bbox.min.map(fmt).join(', ');
  const hi = st.model.bbox.max.map(fmt).join(', ');
  $('modelInfo').textContent = `${st.model.source}\n절점 ${c.nodes} · 보 ${c.beams} · 쉘 ${c.shells} · 솔리드 ${c.solids}\n범위 [${lo}] ~ [${hi}]\n그룹: ${g || '없음'}`;
  const s = st.spec;
  $('solver').value = s.solver;
  $('divisions').value = s.divisions;
  $('divisions').disabled = st.model.kind !== 'obj' || s.solver === 'barhinge';
  $('stepType').value = s.step.type;
  $('nlgeom').checked = !!s.step.nlgeom;
  $('increments').value = s.step.increments;
  $('modes').value = s.step.modes;
  $('imperfection').value = s.step.imperfection || 0;
  $('control').value = s.step.control || 'load';
  $('controlDisp').value = s.step.control_disp || 0;
  $('controlKind').value = s.step.control_kind || 'average';
  $('branch').checked = (s.step.branch || 'auto') !== 'primary';
  const dc = s.step.control === 'displacement' && (s.step.type === 'static' || s.step.type === 'arclength');
  $('controlDisp').disabled = !dc;
  $('controlKind').disabled = !dc;
  $('branch').disabled = s.solver !== 'barhinge';
  const dm = s.step.damping || { kind: 'none' };
  $('dampKind').value = dm.kind === 'direct' ? 'rayleigh' : dm.kind || 'none';
  $('dampZeta').value = dm.zeta || 0;
  $('dampTime').value = dm.time || 0;
  $('amplitude').value = s.step.amplitude || 'linear';
  const own = s.solver === 'shell';
  for (const id of ['dampKind', 'amplitude', 'addSelf']) $(id).disabled = !own;
  $('dampZeta').disabled = !own || dm.kind === 'none';
  $('dampTime').disabled = !own || dm.kind === 'none';
  $('addSelf').disabled = !own || (s.contacts || []).some((c) => c.type === 'self');
  const m = s.materials.default;
  $('matE').value = m.E;
  $('matNu').value = m.nu;
  $('matY').value = m.yield ?? '';
  const sec = s.sections['*'];
  $('secT').value = sec.thickness;
  $('secB').value = sec.beam.b ?? '';
  $('secH').value = sec.beam.h ?? '';
  $('hingeK').value = s.hinges['*'].kappa ?? '';
  $('hingeR').value = s.hinges['*'].ratio;
  $('hingePin').checked = !!s.hinges['*'].pin;
  $('hingeK').disabled = !!s.hinges['*'].pin;
  $('hingeR').disabled = !!s.hinges['*'].pin;
  const hg = st.hinge_info || { auto: { mode: 'off' }, groups: [] };
  $('hingeAuto').value = hg.auto.mode;
  $('hingeAuto').disabled = st.model.kind !== 'obj';
  $('hingeInfo').textContent = hg.groups.length ? '힌지: ' + hg.groups.map((g) => `${g.name} ${g.edges}개 엣지 · ${g.length} mm${g.pin ? ' · 핀' : ''}`).join(' / ') : st.model.kind === 'obj' && st.model.counts.shells ? '지정된 힌지 없음 (쉘 이음은 강접, 바-힌지는 모든 내부 엣지가 접힘선)' : '';
  $('undo').disabled = !st.can_undo;
}

function renderSummary() {
  const box = $('summary');
  box.innerHTML = '';
  if (!st || !st.result) return;
  const head = el('span');
  head.innerHTML = `<b>솔버</b>${st.result.solver} · ${st.result.kind}${st.solve_time !== null ? ` · ${st.solve_time}s` : ''}${st.stale ? ' · <em style="color:var(--warn)">조건이 바뀌었습니다. 다시 해석하세요</em>' : ''}`;
  box.append(head);
  for (const w of st.warnings || []) box.append(el('span', 'note warn', '⚠ ' + w));
  for (const [k, v] of Object.entries(st.result.summary)) {
    if (k === '참고') {
      for (const n of v) box.append(el('span', 'note', n));
      continue;
    }
    const s = el('span');
    const b = el('b', '', k);
    s.append(b, document.createTextNode(Array.isArray(v) ? v.map(fmt).join(', ') : fmt(v)));
    box.append(s);
  }
}

const COLORS = ['#4cc2ff', '#ff5fa8', '#57d9a3', '#ffc857', '#c792ea', '#ff8a5c'];
function chart(p) {
  const W = 420;
  const H = 230;
  const m = { l: 58, r: 12, t: 8, b: 34 };
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', `0 0 ${W} ${H}`);
  let x0 = Infinity;
  let x1 = -Infinity;
  let y0 = Infinity;
  let y1 = -Infinity;
  for (const s of p.series) {
    for (const v of s.x) {
      x0 = Math.min(x0, v);
      x1 = Math.max(x1, v);
    }
    for (const v of s.y) {
      y0 = Math.min(y0, v);
      y1 = Math.max(y1, v);
    }
  }
  if (!isFinite(x0)) return svg;
  if (x0 === x1) x1 = x0 + 1;
  if (y0 === y1) {
    y0 -= Math.abs(y0) * 0.1 || 1;
    y1 += Math.abs(y1) * 0.1 || 1;
  }
  const py = (y1 - y0) * 0.06;
  y0 -= py;
  y1 += py;
  const X = (v) => m.l + ((v - x0) / (x1 - x0)) * (W - m.l - m.r);
  const Y = (v) => H - m.b - ((v - y0) / (y1 - y0)) * (H - m.t - m.b);
  const mk = (tag, attrs, text) => {
    const e = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text !== undefined) e.textContent = text;
    svg.append(e);
    return e;
  };
  for (let i = 0; i <= 4; i++) {
    const yv = y0 + ((y1 - y0) * i) / 4;
    const xv = x0 + ((x1 - x0) * i) / 4;
    mk('line', { x1: m.l, x2: W - m.r, y1: Y(yv), y2: Y(yv), stroke: '#2a3340', 'stroke-width': 1 });
    mk('text', { x: m.l - 6, y: Y(yv) + 3, 'text-anchor': 'end', 'font-size': 10, fill: '#8492a6' }, fmt(yv));
    mk('text', { x: X(xv), y: H - m.b + 14, 'text-anchor': 'middle', 'font-size': 10, fill: '#8492a6' }, fmt(xv));
  }
  mk('text', { x: (m.l + W - m.r) / 2, y: H - 4, 'text-anchor': 'middle', 'font-size': 10, fill: '#8492a6' }, p.xlabel);
  p.series.forEach((s, k) => {
    const pts = s.x.map((v, i) => `${X(v).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join(' ');
    mk('polyline', { points: pts, fill: 'none', stroke: COLORS[k % COLORS.length], 'stroke-width': 1.8 });
    if (s.x.length <= 40) s.x.forEach((v, i) => mk('circle', { cx: X(v), cy: Y(s.y[i]), r: 2.2, fill: COLORS[k % COLORS.length] }));
  });
  return svg;
}

async function loadPlots() {
  const box = $('plots');
  box.innerHTML = '';
  if (!st || !st.spec || !st.spec.plots.length) return;
  if (!st.result) {
    box.append(el('div', 'info', `그래프 ${st.spec.plots.length}개가 등록되어 있습니다. 해석을 실행하면 표시됩니다.`));
    return;
  }
  const list = await api('/api/plots');
  for (const p of list) {
    const d = el('div', 'plot');
    const t = el('div', 'ptitle');
    const x = el('button', '', '✕');
    x.onclick = () => ops([{ op: 'remove', id: p.id }]);
    t.append(el('span', '', `${p.id} · ${p.title}`), x);
    d.append(t);
    if (p.error) d.append(el('div', 'perr', p.error));
    else {
      d.append(chart(p));
      const lg = el('div', 'legend');
      p.series.forEach((s, k) => {
        const sp = el('span');
        const i = el('i');
        i.style.background = COLORS[k % COLORS.length];
        sp.append(i, document.createTextNode(s.name));
        lg.append(sp);
      });
      d.append(lg);
    }
    box.append(d);
  }
}

let lastGeoKey = '';
async function refresh(s, opt = {}) {
  const prev = st;
  st = s;
  renderControls();
  if (!st.model) return;
  const key = `${st.model.source}|${st.model.counts.nodes}|${st.spec.solver === 'barhinge'}|${st.spec.divisions}|${JSON.stringify(st.hinge_info || {})}`;
  if (st.view === undefined && prev) st.view = prev.view;
  if (key !== lastGeoKey) {
    lastGeoKey = key;
    buildModel();
    if (!opt.keepCamera && (!prev || !prev.model || prev.model.source !== st.model.source)) fit();
    else {
      const b = new THREE.Box3();
      for (const p of st.view.nodes) b.expandByPoint(new THREE.Vector3(p[0], p[1], p[2]));
      diag = Math.max(b.getSize(new THREE.Vector3()).length(), 1e-6);
      center = b.getCenter(new THREE.Vector3());
    }
  }
  renderLists();
  buildSymbols();
  renderSummary();
  const hadResult = !!R.mesh;
  if (!st.result || !opt.keepResult || !hadResult) await loadResult();
  applyVisibility();
  await loadPlots();
}

function addMsg(role, text, extra) {
  const log = $('chatLog');
  const m = el('div', 'msg ' + (role === 'user' ? 'user' : 'bot'));
  if (extra && extra.engine) m.append(el('span', 'tag', extra.engine === 'llm' ? 'LLM · ' + (extra.model || '') : '규칙 기반'));
  m.append(document.createTextNode(text));
  if (extra && extra.ops && extra.ops.length) m.append(el('div', 'ops', extra.ops.map((o) => JSON.stringify(o)).join('\n')));
  if (extra && extra.note) m.append(el('div', 'notetxt', extra.note));
  if (extra && extra.error) m.append(el('div', 'errtxt', '적용 실패: ' + extra.error));
  log.append(m);
  log.scrollTop = log.scrollHeight;
}

async function sendChat(text) {
  if (!text.trim()) return;
  if (!st || !st.model) {
    addMsg('bot', '먼저 지오메트리를 불러오세요.');
    return;
  }
  addMsg('user', text);
  $('chatInput').value = '';
  const wait = el('div', 'msg bot', '');
  const label = document.createElement('span');
  const stop = el('button', 'danger stop', 'LLM 중단');
  stop.type = 'button';
  stop.onclick = () => api('/api/cancel/llm', { method: 'POST' }).catch(() => {});
  wait.append(label, stop);
  const t0 = Date.now();
  const tick = () => {
    label.textContent = '처리 중… ' + Math.round((Date.now() - t0) / 1000) + '초 (로컬 LLM은 처음 1~2분 걸릴 수 있습니다)';
  };
  tick();
  const timer = setInterval(tick, 1000);
  $('chatLog').append(wait);
  $('chatLog').scrollTop = $('chatLog').scrollHeight;
  try {
    const s = await api('/api/chat', { json: { text } });
    clearInterval(timer);
    wait.remove();
    const c = s.chat;
    addMsg('bot', c.reply || (c.ops.length ? '적용했습니다.' : '적용할 내용이 없습니다.'), c);
    await refresh(s, { keepResult: !!s.result });
    if (c.action === 'solve') $('solve').click();
  } catch (e) {
    clearInterval(timer);
    wait.remove();
    if (e.message.includes('중단')) addMsg('bot', e.message);
    else addMsg('bot', '', { error: e.message });
  }
}

async function loadExamples() {
  const list = await api('/api/examples');
  const sel = $('example');
  for (const e of list) {
    const o = new Option(e.title, e.name);
    o.dataset.note = e.note;
    o.dataset.prompts = JSON.stringify(e.prompts);
    sel.append(o);
  }
}

function chips(prompts) {
  const c = $('chips');
  c.innerHTML = '';
  for (const p of prompts) {
    const b = el('button', '', p);
    b.type = 'button';
    b.onclick = () => sendChat(p);
    c.append(b);
  }
}

async function openExample(preset) {
  const sel = $('example');
  if (!sel.value) return;
  note('불러오는 중…');
  try {
    const s = await api(`/api/examples/${sel.value}?preset=${preset}`, { method: 'POST' });
    lastGeoKey = '';
    $('chatLog').innerHTML = '';
    const o = sel.selectedOptions[0];
    chips(JSON.parse(o.dataset.prompts));
    await refresh(s);
    note(o.dataset.note);
  } catch (e) {
    note(e.message, true);
  }
}

async function llmRefresh(save) {
  const body = save ? { url: $('llmUrl').value, model: $('llmModel').value, mode: $('llmMode').value, ...( $('llmKey').value ? { key: $('llmKey').value } : {}) } : null;
  const r = save ? await api('/api/llm', { json: body }) : await api('/api/llm');
  $('llmUrl').value = r.llm.url;
  $('llmModel').value = r.llm.model;
  $('llmMode').value = r.llm.mode;
  const s = $('llmStatus');
  renderLlmInfo(r);
  if (r.llm.mode === 'rule') {
    s.textContent = '규칙 기반 사용';
    s.className = '';
  } else if (r.llm.mode === 'rule_first') {
    s.textContent = r.status.ok && r.status.has_model ? '규칙 우선 · LLM 대기 중' : '규칙 우선 · LLM 없음';
    s.className = r.status.ok && r.status.has_model ? 'ok' : '';
  } else if (r.status.ok) {
    s.textContent = r.status.has_model ? '연결됨' : `연결됨 · 모델 '${r.llm.model}' 없음`;
    s.className = r.status.has_model ? 'ok' : 'bad';
  } else {
    s.textContent = '연결 안 됨 → 규칙 기반으로 대체';
    s.className = 'bad';
  }
}

function renderLlmInfo(r, test) {
  const box = $('llmInfo');
  const stt = r.status || {};
  const rows = [];
  const add = (k, v, cls) => {
    const d = document.createElement('div');
    const b = document.createElement('b');
    b.textContent = k + ' ';
    const sp = document.createElement('span');
    sp.textContent = v;
    if (cls) sp.className = cls;
    d.append(b, sp);
    rows.push(d);
  };
  add('연결 대상', `${stt.provider || r.provider || ''} · ${stt.host || ''}`);
  add('모델', r.llm.model, stt.ok ? (stt.has_model ? 'good' : 'badt') : '');
  add('API 키', r.llm.has_key ? '입력됨' : '없음');
  if (r.llm.mode === 'rule') add('현재 방식', 'LLM을 부르지 않고 규칙 기반만 사용 중');
  else if (r.llm.mode === 'rule_first') add('현재 방식', '규칙이 이해한 문장은 즉시 처리, 못 알아들은 문장만 LLM으로 보냄');
  else if (stt.ok) add('서버 응답', `정상 (${stt.ms} ms) · 모델 ${stt.models.length}개 목록${stt.has_model ? '에 선택한 모델이 있음' : '에 선택한 모델이 없음'}`, stt.has_model ? 'good' : 'badt');
  else add('서버 응답', '연결 실패: ' + (stt.error || '') + (r.llm.mode === 'rule_first' ? ' (규칙으로 처리되는 문장은 그대로 동작합니다)' : ''), 'badt');
  if (test) {
    if (test.ok) add('실제 응답', `성공 (${(test.ms / 1000).toFixed(1)}초) · 응답한 모델 ${test.model || '?'} · JSON ${test.json_ok ? '정상' : '아님'} · "${test.reply}"`, test.json_ok ? 'good' : 'badt');
    else add('실제 응답', '실패: ' + test.error, 'badt');
  }
  box.replaceChildren(...rows);
}

async function llmTest() {
  const b = $('llmTest');
  b.disabled = true;
  b.textContent = '테스트 중… (로컬 모델은 첫 응답이 느립니다)';
  try {
    await llmRefresh(true);
    const t = await api('/api/llm/test', { method: 'POST' });
    const r = await api('/api/llm');
    renderLlmInfo(r, t.test);
  } catch (e) {
    note(e.message, true);
  }
  b.disabled = false;
  b.textContent = '실제 응답 테스트';
}

function pickNode(ev) {
  if (!st || !st.view) return;
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const my = ev.clientY - rect.top;
  const useRes = G.result.visible && R.mesh;
  const nodes = useRes ? R.mesh.nodes : st.view.nodes;
  const v = new THREE.Vector3();
  let best = -1;
  let bd = 14 * 14;
  for (let i = 0; i < nodes.length; i++) {
    if (useRes) v.set(R.pos[3 * i], R.pos[3 * i + 1], R.pos[3 * i + 2]);
    else v.set(nodes[i][0], nodes[i][1], nodes[i][2]);
    v.project(camera);
    if (v.z < -1 || v.z > 1) continue;
    const dx = (v.x * 0.5 + 0.5) * rect.width - mx;
    const dy = (-v.y * 0.5 + 0.5) * rect.height - my;
    const d = dx * dx + dy * dy;
    if (d < bd) {
      bd = d;
      best = i;
    }
  }
  clear(G.pick);
  const hud = $('hud');
  hud.innerHTML = '';
  picked = null;
  if (best < 0) return;
  const p = nodes[best];
  picked = p;
  const g = new THREE.BufferGeometry();
  const q = useRes ? [R.pos[3 * best], R.pos[3 * best + 1], R.pos[3 * best + 2]] : p;
  g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(q), 3));
  G.pick.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffc857, size: 11, sizeAttenuation: false, depthTest: false })));
  let txt = `절점 (${p.map((c) => fmt(c)).join(', ')})`;
  if (useRes) {
    const f = R.frames.get(`${R.index}|${R.field}`);
    if (f) txt += ` · ${R.field} = ${fmt(f.values[best])} · U = [${f.U[best].map(fmt).join(', ')}]`;
  }
  hud.append(document.createTextNode(txt));
  const b = el('button', '', '지시문에 넣기');
  b.onclick = () => {
    const t = $('chatInput');
    t.value += `${t.value && !t.value.endsWith(' ') ? ' ' : ''}절점 (${p.map((c) => +c.toFixed(4)).join(', ')}) `;
    t.focus();
  };
  hud.append(b);
}

function bind() {
  $('loadPreset').onclick = () => openExample(1);
  $('loadGeo').onclick = () => openExample(0);
  $('file').onchange = async (e) => {
    const f = e.target.files[0];
    if (!f) return;
    const fd = new FormData();
    fd.append('file', f);
    fd.append('size', $('stlSize').value || '0');
    note(f.name.toLowerCase().endsWith('.stl') ? '체적 메쉬 생성 중…' : '불러오는 중…');
    try {
      const s = await api('/api/upload', { method: 'POST', body: fd });
      lastGeoKey = '';
      $('chatLog').innerHTML = '';
      chips(['z 최소 면을 고정해줘', 'z 최대 면에 z 방향으로 -1000 N', '비선형 해석으로 증분 10']);
      await refresh(s);
      note(`${f.name} 을(를) 불러왔습니다. 단위는 mm 로 가정합니다.`);
    } catch (err) {
      note(err.message, true);
    }
    e.target.value = '';
  };
  $('solver').onchange = (e) => ops([{ op: 'set_solver', solver: e.target.value }]);
  $('divisions').onchange = (e) => ops([{ op: 'set_divisions', divisions: +e.target.value || 1 }]);
  $('stepType').onchange = (e) => ops([{ op: 'set_step', type: e.target.value }]);
  $('nlgeom').onchange = (e) => ops([{ op: 'set_step', nlgeom: e.target.checked }]);
  $('increments').onchange = (e) => ops([{ op: 'set_step', increments: +e.target.value || 1 }]);
  $('modes').onchange = (e) => ops([{ op: 'set_step', modes: +e.target.value || 1 }]);
  $('imperfection').onchange = (e) => ops([{ op: 'set_step', imperfection: +e.target.value || 0 }]);
  $('control').onchange = (e) => ops([{ op: 'set_step', control: e.target.value }]);
  $('controlDisp').onchange = (e) => ops([{ op: 'set_step', control: 'displacement', control_disp: Math.abs(+e.target.value) || 0 }]);
  $('controlKind').onchange = (e) => ops([{ op: 'set_step', control_kind: e.target.value }]);
  $('branch').onchange = (e) => ops([{ op: 'set_step', branch: e.target.checked ? 'auto' : 'primary' }]);
  $('dampKind').onchange = (e) => ops([{ op: 'set_step', damping: { kind: e.target.value } }]);
  $('dampZeta').onchange = (e) => ops([{ op: 'set_step', damping: { zeta: Math.max(0, +e.target.value || 0) } }]);
  $('dampTime').onchange = (e) => ops([{ op: 'set_step', damping: { time: Math.max(0, +e.target.value || 0) } }]);
  $('amplitude').onchange = (e) => ops([{ op: 'set_step', amplitude: e.target.value }]);
  $('addSelf').onclick = () => ops([{ op: 'add_contact', type: 'self', friction: 0 }]);
  $('matE').onchange = (e) => ops([{ op: 'set_material', name: 'default', E: +e.target.value }]);
  $('matNu').onchange = (e) => ops([{ op: 'set_material', name: 'default', nu: +e.target.value }]);
  $('matY').onchange = (e) => ops([{ op: 'set_material', name: 'default', yield: e.target.value === '' ? null : +e.target.value }]);
  $('secT').onchange = (e) => ops([{ op: 'set_section', group: '*', thickness: +e.target.value }]);
  $('secB').onchange = (e) => ops([{ op: 'set_section', group: '*', beam: { b: +e.target.value } }]);
  $('secH').onchange = (e) => ops([{ op: 'set_section', group: '*', beam: { h: +e.target.value } }]);
  $('hingeK').onchange = (e) => ops([{ op: 'set_hinge', group: '*', kappa: e.target.value === '' ? null : +e.target.value }]);
  $('hingeR').onchange = (e) => ops([{ op: 'set_hinge', group: '*', ratio: +e.target.value }]);
  $('hingeAuto').onchange = (e) => ops([{ op: 'auto_hinge', mode: e.target.value }]);
  $('hingePin').onchange = (e) => ops([{ op: 'set_hinge', group: '*', pin: e.target.checked }]);
  $('undo').onclick = async () => {
    try {
      await refresh(await api('/api/undo', { method: 'POST' }));
    } catch (e) {
      note(e.message, true);
    }
  };
  $('solve').onclick = async () => {
    if (!st || !st.model) return note('먼저 지오메트리를 불러오세요', true);
    const b = $('solve');
    b.disabled = true;
    b.textContent = '해석 중…';
    $('cancelSolve').hidden = false;
    note('');
    try {
      const s = await api('/api/solve', { method: 'POST' });
      s.view = st.view;
      await refresh(s);
      note(`완료 (${s.solve_time}s)`);
    } catch (e) {
      note(e.message, true);
    }
    $('cancelSolve').hidden = true;
    b.disabled = false;
    b.textContent = '해석 실행';
  };
  $('cancelSolve').onclick = () => api('/api/cancel/solve', { method: 'POST' }).catch(() => {});
  $('fit').onclick = fit;
  for (const id of ['showSymbols', 'showEdges', 'showResult', 'showMeshLines']) $(id).onchange = applyVisibility;
  $('frame').oninput = (e) => {
    R.index = +e.target.value;
    showFrame();
  };
  $('field').onchange = (e) => {
    R.field = e.target.value;
    showFrame();
  };
  $('scale').onchange = (e) => {
    R.scale = +e.target.value || 0;
    showFrame();
  };
  $('autoScale').onclick = () => {
    R.auto = true;
    showFrame();
  };
  $('play').onclick = () => {
    if (R.playing) return stopPlay();
    $('play').textContent = '❚❚';
    R.playing = setInterval(() => {
      R.index = (R.index + 1) % st.result.frames.length;
      $('frame').value = R.index;
      showFrame();
    }, 220);
  };
  $('llmTest').onclick = llmTest;
  $('chatForm').onsubmit = (e) => {
    e.preventDefault();
    sendChat($('chatInput').value);
  };
  $('chatInput').onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) {
      e.preventDefault();
      sendChat($('chatInput').value);
    }
  };
  $('llmSave').onclick = () => llmRefresh(true).catch((e) => note(e.message, true));
  let down = null;
  canvas.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY]));
  canvas.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) pickNode(e);
    down = null;
  });
}

async function start() {
  resize();
  bind();
  loop();
  await loadExamples();
  llmRefresh(false).catch(() => {});
  const s = await api('/api/state');
  if (s.model) await refresh(s);
  else {
    st = s;
    renderControls();
  }
}

start().catch((e) => note(e.message, true));
