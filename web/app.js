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
scene.add(new THREE.HemisphereLight(0xffffff, 0x2a3442, 1.25));
const sun = new THREE.DirectionalLight(0xffffff, 1.9);
scene.add(sun);
const fill = new THREE.DirectionalLight(0xbcd4ff, 0.7);
scene.add(fill);
const surfaceMat = (extra) => new THREE.MeshStandardMaterial({ roughness: 0.62, metalness: 0.08, side: THREE.DoubleSide, polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1, ...extra });
const G = { model: new THREE.Group(), edges: new THREE.Group(), symbols: new THREE.Group(), result: new THREE.Group(), pick: new THREE.Group(), rigid: new THREE.Group(), crease: new THREE.Group() };
Object.values(G).forEach((g) => scene.add(g));
G.crease.renderOrder = 2;
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
  const cr = new THREE.Vector3().crossVectors(camera.getWorldDirection(new THREE.Vector3()), camera.up).normalize();
  sun.position.copy(camera.position).addScaledVector(cr, -0.45 * camera.position.distanceTo(controls.target)).addScaledVector(camera.up, 0.5 * camera.position.distanceTo(controls.target));
  sun.target.position.copy(controls.target);
  sun.target.updateMatrixWorld();
  fill.position.copy(camera.position).addScaledVector(cr, 0.9 * camera.position.distanceTo(controls.target));
  fill.target.position.copy(controls.target);
  fill.target.updateMatrixWorld();
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
    const m = surfaceMat({ color: 0x9fb3c8 });
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
    if (c.shape === 'mesh') {
      const rm = R.mesh && (R.mesh.rigid || []).find((q) => q.id === c.id && q.V);
      if (!rm) continue;
      const g = new THREE.BufferGeometry();
      g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(rm.V.flat()), 3));
      g.setIndex(rm.F.flat());
      o = new THREE.Mesh(g, mat);
      o.userData.base = [0, 0, 0];
      o.userData.resultOnly = true;
    } else if (c.shape === 'sphere') {
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
      a.userData.base = c.shape === 'mesh' ? [0, 1, 2].map((k) => o.geometry.attributes.position.array.filter((_, i) => i % 3 === k).reduce((s, v, _, arr) => s + v / arr.length, 0)) : o.userData.base;
      a.userData.motion = c.motion;
      a.userData.resultOnly = o.userData.resultOnly;
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
  for (const c of st.preview.couplings || []) {
    const hot = c.id === highlightId;
    const col = hot ? 0xffc857 : 0xc084fc;
    const r = new THREE.Vector3(...c.ref);
    const seg = [];
    for (const k of sample(c.nodes.length, 120)) seg.push(r.x, r.y, r.z, ...nodes[c.nodes[k]]);
    const lg = new THREE.BufferGeometry();
    lg.setAttribute('position', new THREE.Float32BufferAttribute(seg, 3));
    G.symbols.add(new THREE.LineSegments(lg, new THREE.LineBasicMaterial({ color: col, transparent: true, opacity: 0.45, depthTest: false })));
    const ball = new THREE.Mesh(new THREE.SphereGeometry(s * 0.35, 16, 12), new THREE.MeshBasicMaterial({ color: col, depthTest: false }));
    ball.position.copy(r);
    G.symbols.add(ball);
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
  const thick = R.mesh.thick && R.mesh.tris.length ? R.mesh.thick : null;
  R.thick = thick;
  R.mid = null;
  R.pos2 = null;
  if (R.mesh.tris.length) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(col, 3));
    g.setIndex(R.mesh.tris.flat());
    const thin = new THREE.Mesh(g, surfaceMat({ vertexColors: true }));
    thin.userData.thin = true;
    G.result.add(thin);
    if (thick) {
      R.mid = g;
      R.pos2 = new Float32Array(n * 6);
      R.col2 = new Float32Array(n * 6);
      const idx = [];
      const cnt = new Map();
      for (const t of R.mesh.tris) {
        idx.push(t[0], t[1], t[2], t[0] + n, t[2] + n, t[1] + n);
        for (let k = 0; k < 3; k++) {
          const a = t[k];
          const b = t[(k + 1) % 3];
          const key = a < b ? a * 1e7 + b : b * 1e7 + a;
          const c = cnt.get(key);
          if (c) c.n += 1;
          else cnt.set(key, { n: 1, a, b });
        }
      }
      for (const c of cnt.values()) if (c.n === 1) idx.push(c.a, c.a + n, c.b, c.b, c.a + n, c.b + n);
      const g2 = new THREE.BufferGeometry();
      g2.setAttribute('position', new THREE.BufferAttribute(R.pos2, 3));
      g2.setAttribute('color', new THREE.BufferAttribute(R.col2, 3));
      g2.setIndex(idx);
      const solid = new THREE.Mesh(g2, surfaceMat({ vertexColors: true }));
      solid.userData.thick = true;
      G.result.add(solid);
    }
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
    ls.userData.thin = true;
    G.result.add(ls);
    if (thick) {
      const g2 = new THREE.BufferGeometry();
      g2.setAttribute('position', new THREE.BufferAttribute(R.pos2, 3));
      const e2 = [];
      for (const e of R.mesh.fedges) e2.push(e[0], e[1], e[0] + n, e[1] + n);
      g2.setIndex(e2);
      const l2 = new THREE.LineSegments(g2, new THREE.LineBasicMaterial({ color: 0x0b0f14, transparent: true, opacity: 0.5 }));
      l2.userData.meshLines = true;
      l2.userData.thick = true;
      G.result.add(l2);
    }
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
  buildRigid();
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
  if (R.mid && R.pos2) {
    R.mid.computeVertexNormals();
    const nr = R.mid.attributes.normal.array;
    for (let i = 0; i < n; i++) {
      const h = 0.5 * R.thick[i];
      for (let k = 0; k < 3; k++) {
        const c = 3 * i + k;
        R.pos2[c] = R.pos[c] + h * nr[c];
        R.pos2[3 * n + c] = R.pos[c] - h * nr[c];
        R.col2[c] = R.col[c];
        R.col2[3 * n + c] = R.col[c];
      }
    }
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

function snapshot(maxW) {
  const sv = G.symbols.visible;
  G.symbols.visible = false;
  renderer.render(scene, camera);
  G.symbols.visible = sv;
  const k = Math.min(1, maxW / canvas.width);
  const c = document.createElement('canvas');
  c.width = Math.round(canvas.width * k);
  c.height = Math.round(canvas.height * k);
  const x = c.getContext('2d');
  x.fillStyle = getComputedStyle(document.body).backgroundColor || '#0f141b';
  x.fillRect(0, 0, c.width, c.height);
  x.drawImage(canvas, 0, 0, c.width, c.height);
  return c.toDataURL('image/png');
}

function download(url, name) {
  const a = document.createElement('a');
  a.href = url;
  a.download = name;
  a.click();
}

async function saveGif() {
  if (!R.mesh) return;
  stopPlay();
  const keep = R.index;
  const total = st.result.frames.length;
  const step = Math.max(1, Math.ceil(total / 80));
  const frames = [];
  try {
    for (let i = 0; i < total; i += step) {
      R.index = i;
      $('frame').value = i;
      await showFrame();
      frames.push(snapshot(900));
    }
    if ((total - 1) % step) {
      R.index = total - 1;
      await showFrame();
      frames.push(snapshot(900));
    }
    const r = await fetch('/api/export/gif', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ frames, fps: 8 }) });
    if (!r.ok) throw new Error(((await r.json().catch(() => null)) || {}).detail || 'GIF 저장 실패');
    download(URL.createObjectURL(await r.blob()), 'femweb.gif');
  } catch (e) {
    note(e.message, true);
  }
  R.index = keep;
  $('frame').value = keep;
  showFrame();
}

async function saveVideo() {
  if (!R.mesh) return;
  if (!window.MediaRecorder || !canvas.captureStream) return note('이 브라우저는 영상 저장을 지원하지 않습니다. GIF 를 쓰세요', true);
  stopPlay();
  const mime = ['video/mp4;codecs=avc1', 'video/webm;codecs=vp9', 'video/webm'].find((m) => MediaRecorder.isTypeSupported(m));
  if (!mime) return note('지원되는 영상 형식이 없습니다. GIF 를 쓰세요', true);
  const keep = R.index;
  const rec = new MediaRecorder(canvas.captureStream(30), { mimeType: mime, videoBitsPerSecond: 8e6 });
  const chunks = [];
  rec.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  const done = new Promise((res) => (rec.onstop = res));
  const sv = G.symbols.visible;
  G.symbols.visible = false;
  rec.start();
  for (let i = 0; i < st.result.frames.length; i++) {
    R.index = i;
    $('frame').value = i;
    await showFrame();
    await new Promise((res) => setTimeout(res, 120));
  }
  await new Promise((res) => setTimeout(res, 500));
  rec.stop();
  await done;
  G.symbols.visible = sv;
  download(URL.createObjectURL(new Blob(chunks, { type: mime })), mime.startsWith('video/mp4') ? 'femweb.mp4' : 'femweb.webm');
  R.index = keep;
  $('frame').value = keep;
  showFrame();
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
  for (const o of G.rigid.children) if (o.userData.resultOnly) o.visible = res;
  if (!res) placeRigid(0);
  $('legend').hidden = !res;
  $('resultbar').hidden = !R.mesh;
  const ml = $('showMeshLines').checked;
  const tk = !!R.thick && $('showThick').checked;
  G.result.children.forEach((c) => {
    c.visible = true;
    if (c.userData.thick) c.visible = tk;
    if (c.userData.thin) c.visible = !tk;
    if (c.userData.meshLines) c.visible = c.visible && ml;
  });
  $('thickWrap').hidden = !R.thick;
  $('meshLinesWrap').hidden = !R.mesh;
  G.crease.visible = !!st && !!st.spec && st.spec.solver === 'bh' && (!res || BH.edit);
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
    nm.append(el('span', 'id', c.id), document.createTextNode(c.type === 'rigid' ? (c.shape === 'sphere' ? '강체 구' : c.shape === 'mesh' ? `강체 부품 ${c.body && c.body.group ? c.body.group : ''}` : '강체 평면') : c.type === 'self' ? '자기 접촉' : '접촉 쌍'));
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
  const cpl = $('cpList');
  cpl.innerHTML = '';
  const cps = st.spec.couplings || [];
  $('cpCount').textContent = cps.length ? `${cps.length}개` : '';
  if (!cps.length) cpl.append(el('div', 'empty', '없음 · 예: "위 끝면을 기준점에 묶어줘"'));
  for (const c of cps) {
    const p = (pv.couplings || []).find((x) => x.id === c.id);
    const it = el('div', 'item' + (highlightId === c.id ? ' sel' : ''));
    const top = el('div', 'top');
    const nm = el('span');
    nm.append(el('span', 'id', c.id), document.createTextNode(c.name || '기준점'));
    const x = el('button', 'x', '✕');
    x.onclick = (e) => {
      e.stopPropagation();
      ops([{ op: 'remove', id: c.id }]);
    };
    top.append(nm, x);
    it.append(top, el('div', 'desc', p && p.desc ? p.desc : ''));
    const row = (label, vals, title, mk) => {
      const ed = el('div', 'edit');
      ed.append(el('span', 'lab', label));
      const ins = vals.map((v, k) => {
        const i = el('input');
        i.type = 'number';
        i.step = 'any';
        i.value = v;
        i.placeholder = '자유';
        i.title = title[k];
        i.onclick = (e) => e.stopPropagation();
        return i;
      });
      ins.forEach((i) => (i.onchange = () => ops([mk(ins)])));
      ed.append(...ins);
      return ed;
    };
    const fx = c.fix || {};
    const dn = ['ux (mm)', 'uy (mm)', 'uz (mm)', 'rx (rad)', 'ry (rad)', 'rz (rad)'];
    const fixRow = (a, b, label) =>
      row(label, [a, b, b + 1].slice(0, 3).map((k) => (fx[String(k)] ?? '')), [a, b, b + 1].map((k) => `${dn[k - 1]} · 비우면 자유, 0 이면 구속`), (ins) => {
        const keys = [a, b, b + 1];
        const fix = {};
        const free = [];
        ins.forEach((q, j) => (q.value === '' ? free.push(keys[j]) : (fix[String(keys[j])] = +q.value || 0)));
        return { op: 'update', id: c.id, fix, free };
      });
    it.append(fixRow(1, 2, '병진'), fixRow(4, 5, '회전'));
    it.append(row('힘 N', c.force || [0, 0, 0], ['Fx', 'Fy', 'Fz'], (ins) => ({ op: 'update', id: c.id, force: ins.map((q) => +q.value || 0) })));
    it.append(row('모멘트 N·mm', c.moment || [0, 0, 0], ['Mx', 'My', 'Mz'], (ins) => ({ op: 'update', id: c.id, moment: ins.map((q) => +q.value || 0) })));
    it.onclick = () => {
      highlightId = highlightId === c.id ? null : c.id;
      renderLists();
      buildSymbols();
    };
    cpl.append(it);
  }
  const w = $('warnings');
  w.innerHTML = '';
  for (const m of st.warnings) w.append(el('div', '', '⚠ ' + m));
  for (const m of pv.errors) w.append(el('div', 'err', '✕ ' + m));
}

function renderControls() {
  const t = st.tools || {};
  $('tools').innerHTML = `<b class="${t.ccx ? 'on' : 'off'}">CalculiX ${t.ccx ? '●' : '○'}</b> <b class="${t.tensormesh ? 'on' : 'off'}">TensorMesh ${t.tensormesh ? '●' : '○'}</b> <b class="${t.gpu ? 'on' : 'off'}" title="${t.gpu || 'GPU 없음'}">GPU ${t.gpu ? '●' : '○'}</b>`;
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
  const sz = st.model.bbox.max.map((v, i) => v - st.model.bbox.min[i]);
  $('modelInfo').textContent = `${st.model.source}\n절점 ${c.nodes} · 보 ${c.beams} · 쉘 ${c.shells} · 솔리드 ${c.solids}\n크기 ${sz.map((v) => v.toFixed(3)).join(' × ')} mm (대각선 ${Math.hypot(...sz).toFixed(3)})\n범위 [${lo}] ~ [${hi}]\n그룹: ${g || '없음'}\n우클릭: 절점·엣지·면 정보`;
  const s = st.spec;
  $('solver').value = s.solver;
  $('divisions').value = s.divisions;
  $('divisions').disabled = st.model.kind !== 'obj' || s.solver === 'barhinge';
  $('stepType').value = s.step.type;
  $('nlgeom').checked = !!s.step.nlgeom || s.solver === 'shell';
  $('nlgeom').disabled = s.solver === 'shell';
  $('nlgeom').parentElement.title = s.solver === 'shell' ? '자체 쉘 솔버는 항상 기하 비선형으로 풉니다' : '';
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
  $('element').value = s.step.element || 'auto';
  $('device').value = s.step.device || 'auto';
  $('precision').value = s.step.precision || 'auto';
  $('threads').value = s.step.threads || 0;
  const own = s.solver === 'shell';
  for (const id of ['dampKind', 'amplitude', 'addSelf', 'element', 'device', 'precision', 'threads']) $(id).disabled = !own;
  $('dampZeta').disabled = !own || dm.kind === 'none';
  $('dampTime').disabled = !own || dm.kind === 'none';
  $('addSelf').disabled = !own || (s.contacts || []).some((c) => c.type === 'self');
  const m = s.materials.default;
  $('matE').value = m.E;
  $('matNu').value = m.nu;
  $('matY').value = m.yield ?? '';
  $('matH').value = m.hardening || '';
  $('matCurve').value = (m.curve || []).map((q) => `${q[1]}@${q[0]}`).join(', ');
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
  renderBH();
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
const REFC = ['#ffffff', '#b0bec5', '#ffd54f', '#80cbc4'];

function refRow(r, k) {
  const row = el('div', 'refrow');
  const sw = el('i');
  sw.style.borderTop = `2px dashed ${REFC[k % REFC.length]}`;
  const m = r.metrics || {};
  const pct = (v) => (v === null || v === undefined ? '-' : `${(v * 100).toFixed(1)}%`);
  const rat = (v) => (v === null || v === undefined ? '-' : `${v.toFixed(3)}배`);
  const txt = m.overlap ? `${r.name} 대비 ${r.against}: 최대값 ${rat(m.peak_ratio)}, 평균 ${rat(m.mean_ratio)}, RMS 차이 ${pct(m.rms)}, 최대 차이 ${pct(m.max_diff)} (구간 ${fmt(m.overlap[0])}~${fmt(m.overlap[1])})` : `${r.name}: 겹치는 X 구간이 없습니다`;
  const flip = el('button', 'mini', r.sy < 0 ? '부호 ×(−1)' : '부호 ×(+1)');
  flip.title = r.flipped ? '자체 결과와 부호가 반대라 자동으로 뒤집었습니다' : '';
  flip.onclick = async () => { await api('/api/refs/update', { json: { id: r.id, sy: -r.sy } }); loadPlots(); };
  const del = el('button', 'mini', '삭제');
  del.onclick = async () => { await api('/api/refs/update', { json: { id: r.id, remove: true } }); loadPlots(); };
  row.append(sw, el('span', '', txt), flip, del);
  return row;
}

function refUpload(p) {
  const lab = el('label', 'mini btn ghost', 'Abaqus 곡선 겹치기');
  const inp = document.createElement('input');
  inp.type = 'file';
  inp.accept = '.csv,.rpt,.txt,.dat';
  inp.style.display = 'none';
  inp.onchange = async () => {
    const f = inp.files[0];
    if (!f) return;
    const text = await f.text();
    try {
      await api('/api/refs', { json: { plot: p.id, text, name: f.name.replace(/\.[^.]+$/, '') } });
    } catch (e) {
      alert(e.message || e);
    }
    loadPlots();
  };
  lab.append(inp);
  return lab;
}
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
  for (const s of p.series.concat(p.refs || [])) {
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
  (p.refs || []).forEach((s, k) => {
    const pts = s.x.map((v, i) => `${X(v).toFixed(1)},${Y(s.y[i]).toFixed(1)}`).join(' ');
    mk('polyline', { points: pts, fill: 'none', stroke: REFC[k % REFC.length], 'stroke-width': 1.8, 'stroke-dasharray': '5 3' });
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
      (p.refs || []).forEach((r, k) => d.append(refRow(r, k)));
      d.append(refUpload(p));
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
  await loadCreases();
  renderLists();
  buildSymbols();
  renderSummary();
  const hadResult = !!R.mesh;
  if (!st.result || !opt.keepResult || !hadResult) await loadResult();
  applyVisibility();
  await loadPlots();
  await loadKin3d();
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
  if (!r.llm.url || !r.llm.model) {
    s.textContent = '규칙 기반 사용 · LLM 미설정';
    s.className = '';
  } else if (r.llm.mode === 'rule') {
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

function viewTopo() {
  const v = st.view;
  if (v._edges) return v;
  const seen = new Map();
  const edges = [];
  const add = (a, b) => {
    const k = a < b ? a * 1e7 + b : b * 1e7 + a;
    if (!seen.has(k)) {
      seen.set(k, edges.length);
      edges.push([Math.min(a, b), Math.max(a, b)]);
    }
  };
  for (const f of v.shells) for (let i = 0; i < f.length; i++) add(f[i], f[(i + 1) % f.length]);
  for (const f of v.boundary) for (let i = 0; i < 3; i++) add(f[i], f[(i + 1) % 3]);
  for (const b of v.beams) add(b[0], b[1]);
  const faces = v.shells.map((f, i) => ({ n: f, kind: '면', id: i, group: (v.shell_group || [])[i] })).concat(v.boundary.map((f, i) => ({ n: f, kind: '솔리드 표면 삼각형', id: i })));
  v._edges = edges;
  v._faces = faces;
  return v;
}

function infoPos(i) {
  const useRes = G.result.visible && R.mesh && R.mesh.nodes.length === st.view.nodes.length;
  return useRes ? [R.pos[3 * i], R.pos[3 * i + 1], R.pos[3 * i + 2]] : st.view.nodes[i];
}

const f3 = (x) => (Math.abs(x) < 5e-4 ? 0 : x).toFixed(3);
const vtxt = (a) => `(${a.map(f3).join(', ')})`;

function pickInfo(ev) {
  if (!st || !st.view) return;
  const v = viewTopo();
  const rect = canvas.getBoundingClientRect();
  const mx = ev.clientX - rect.left;
  const my = ev.clientY - rect.top;
  const n = v.nodes.length;
  const sx = new Float32Array(n);
  const sy = new Float32Array(n);
  const ok = new Uint8Array(n);
  const q = new THREE.Vector3();
  for (let i = 0; i < n; i++) {
    const p = infoPos(i);
    q.set(p[0], p[1], p[2]).project(camera);
    if (q.z < -1 || q.z > 1) continue;
    ok[i] = 1;
    sx[i] = (q.x * 0.5 + 0.5) * rect.width;
    sy[i] = (-q.y * 0.5 + 0.5) * rect.height;
  }
  const ray = new THREE.Raycaster();
  ray.setFromCamera(new THREE.Vector2((mx / rect.width) * 2 - 1, -(my / rect.height) * 2 + 1), camera);
  let hitFace = -1;
  let hitDist = Infinity;
  const tri = new THREE.Triangle();
  const hp = new THREE.Vector3();
  v._faces.forEach((f, k) => {
    const P = f.n.map((i) => new THREE.Vector3(...infoPos(i)));
    for (let j = 1; j < P.length - 1; j++) {
      const h = ray.ray.intersectTriangle(P[0], P[j], P[j + 1], false, hp);
      if (h) {
        const d = h.distanceTo(ray.ray.origin);
        if (d < hitDist) {
          hitDist = d;
          hitFace = k;
        }
      }
    }
  });
  const depth = (i) => new THREE.Vector3(...infoPos(i)).distanceTo(ray.ray.origin);
  const tol = hitDist === Infinity ? Infinity : hitDist + 1e-3 * (diag || 1);
  let best = -1;
  let bd = Infinity;
  for (let i = 0; i < n; i++) {
    if (!ok[i]) continue;
    const d = (sx[i] - mx) ** 2 + (sy[i] - my) ** 2;
    if (d < bd && depth(i) <= tol) {
      bd = d;
      best = i;
    }
  }
  let ls = 0;
  let nl = 0;
  if (best >= 0)
    for (const [a, b] of v._edges)
      if ((a === best || b === best) && ok[a] && ok[b]) {
        ls += Math.hypot(sx[b] - sx[a], sy[b] - sy[a]);
        nl++;
      }
  ls = nl ? ls / nl : 40;
  const rn = Math.min(8, 0.22 * ls);
  const re = Math.min(5, 0.12 * ls);
  if (Math.sqrt(bd) > rn) best = -1;
  let be = -1;
  if (best < 0) {
    let ed = re * re;
    v._edges.forEach(([a, b], k) => {
      if (!ok[a] || !ok[b]) return;
      const ux = sx[b] - sx[a];
      const uy = sy[b] - sy[a];
      const L2 = ux * ux + uy * uy || 1e-9;
      const t = Math.max(0, Math.min(1, ((mx - sx[a]) * ux + (my - sy[a]) * uy) / L2));
      const d = (sx[a] + t * ux - mx) ** 2 + (sy[a] + t * uy - my) ** 2;
      if (d < ed) {
        const pa = infoPos(a);
        const pb = infoPos(b);
        const pm = new THREE.Vector3(pa[0] + t * (pb[0] - pa[0]), pa[1] + t * (pb[1] - pa[1]), pa[2] + t * (pb[2] - pa[2]));
        if (pm.distanceTo(ray.ray.origin) <= tol) {
          ed = d;
          be = k;
        }
      }
    });
  }
  clear(G.pick);
  const hud = $('hud');
  hud.innerHTML = '';
  picked = null;
  const mark = (pts, kind) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(pts.flat()), 3));
    if (kind === 'pt') G.pick.add(new THREE.Points(g, new THREE.PointsMaterial({ color: 0xffc857, size: 11, sizeAttenuation: false, depthTest: false })));
    else G.pick.add(new THREE.LineLoop(g, new THREE.LineBasicMaterial({ color: 0xffc857, depthTest: false })));
  };
  const lines = [];
  let chip = '';
  if (best >= 0) {
    const p = infoPos(best);
    const p0 = v.nodes[best];
    picked = p0;
    mark([p], 'pt');
    lines.push(`절점 #${best + 1}`, `좌표 ${vtxt(p0)}`);
    if (p !== p0) lines.push(`변형 후 ${vtxt(p)}`);
    const deg = v._edges.filter(([a, b]) => a === best || b === best).length;
    lines.push(`연결 엣지 ${deg}개`);
    chip = `절점 (${p0.map((c) => +c.toFixed(4)).join(', ')})`;
  } else if (be >= 0) {
    const [a, b] = v._edges[be];
    const pa = infoPos(a);
    const pb = infoPos(b);
    const d = pb.map((c, i) => c - pa[i]);
    const L = Math.hypot(...d);
    mark([pa, pb], 'ln');
    lines.push(`엣지 #${be + 1} (절점 #${a + 1} → #${b + 1})`, `길이 ${f3(L)} mm`, `벡터 ${vtxt(d)}`, `방향 ${vtxt(d.map((c) => c / (L || 1)))}`);
  } else if (hitFace >= 0) {
    const f = v._faces[hitFace];
    const P = f.n.map((i) => infoPos(i));
    const nv = [0, 0, 0];
    for (let j = 1; j < P.length - 1; j++) {
      const u = P[j].map((c, i) => c - P[0][i]);
      const w = P[j + 1].map((c, i) => c - P[0][i]);
      nv[0] += 0.5 * (u[1] * w[2] - u[2] * w[1]);
      nv[1] += 0.5 * (u[2] * w[0] - u[0] * w[2]);
      nv[2] += 0.5 * (u[0] * w[1] - u[1] * w[0]);
    }
    const A = Math.hypot(...nv);
    const c = [0, 1, 2].map((k) => P.reduce((s, p) => s + p[k], 0) / P.length);
    mark(P, 'ln');
    lines.push(`${f.kind} #${f.id + 1}${f.group ? ` (그룹 ${f.group})` : ''}`, `절점 ${f.n.map((i) => '#' + (i + 1)).join(', ')}`, `면적 ${f3(A)} mm²`, `법선 ${vtxt(nv.map((x) => x / (A || 1)))}`, `중심 ${vtxt(c)}`);
    chip = `점 (${c.map((x) => +x.toFixed(4)).join(', ')})`;
  } else return;
  const box = el('div', 'pickinfo');
  for (const t of lines) box.append(el('div', '', t));
  hud.append(box);
  const cp = el('button', '', '복사');
  cp.onclick = () => navigator.clipboard && navigator.clipboard.writeText(lines.join('\n'));
  hud.append(cp);
  if (chip) {
    const b = el('button', '', '지시문에 넣기');
    b.onclick = () => {
      const t = $('chatInput');
      t.value += `${t.value && !t.value.endsWith(' ') ? ' ' : ''}${chip} `;
      t.focus();
    };
    hud.append(b);
  }
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
  $('element').onchange = (e) => ops([{ op: 'set_step', element: e.target.value }]);
  $('device').onchange = (e) => ops([{ op: 'set_step', device: e.target.value }]);
  $('precision').onchange = (e) => ops([{ op: 'set_step', precision: e.target.value }]);
  $('threads').onchange = (e) => ops([{ op: 'set_step', threads: Math.max(0, Math.round(+e.target.value || 0)) }]);
  $('amplitude').onchange = (e) => ops([{ op: 'set_step', amplitude: e.target.value }]);
  $('addSelf').onclick = () => ops([{ op: 'add_contact', type: 'self', friction: 0 }]);
  $('matE').onchange = (e) => ops([{ op: 'set_material', name: 'default', E: +e.target.value }]);
  $('matNu').onchange = (e) => ops([{ op: 'set_material', name: 'default', nu: +e.target.value }]);
  $('matH').onchange = (e) => ops([{ op: 'set_material', name: 'default', hardening: Math.max(0, +e.target.value || 0) }]);
  $('matCurve').onchange = (e) => {
    const pts = e.target.value.split(/[,;]+/).map((q) => q.trim()).filter(Boolean).map((q) => q.split('@').map(Number)).filter((q) => q.length === 2 && q.every(Number.isFinite)).map((q) => [q[1], q[0]]);
    ops([{ op: 'set_material', name: 'default', curve: pts.length >= 2 ? pts : null }]);
  };
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
    const tm = (s) => (s >= 3600 ? `${Math.floor(s / 3600)}시간 ${Math.round((s % 3600) / 60)}분` : s >= 60 ? `${Math.floor(s / 60)}분 ${Math.round(s % 60)}초` : `${Math.round(s)}초`);
    const poll = setInterval(async () => {
      try {
        const p = await api('/api/progress');
        if (!p.running) return;
        if (p.lam == null) {
          b.textContent = `해석 중… ${tm(p.elapsed)}`;
          note(`${p.stage || '해석 준비 중'} · ${tm(p.elapsed)} 경과`);
          return;
        }
        const pct = (100 * p.lam).toFixed(1);
        b.textContent = `해석 중… ${pct}%`;
        const parts = [`진행 ${pct}%`, `증분 ${p.inc}${p.increments ? ` (기본 ${p.increments})` : ''}`];
        if (p.disp != null) parts.push(`변위 ${p.disp} mm`);
        parts.push(`${tm(p.elapsed)} 경과`);
        if (p.eta != null) parts.push(`남은 약 ${tm(p.eta)}`);
        note(parts.join(' · ') + (p.stage ? `\n${p.stage}` : ''));
      } catch (e) {}
    }, 1000);
    try {
      const s = await api('/api/solve', { method: 'POST' });
      s.view = st.view;
      await refresh(s);
      note(`완료 (${s.solve_time}s)`);
    } catch (e) {
      note(e.message, true);
    }
    clearInterval(poll);
    $('cancelSolve').hidden = true;
    b.disabled = false;
    b.textContent = '해석 실행';
  };
  $('cancelSolve').onclick = () => api('/api/cancel/solve', { method: 'POST' }).catch(() => {});
  $('fit').onclick = fit;
  for (const id of ['showSymbols', 'showEdges', 'showResult', 'showMeshLines', 'showThick']) $(id).onchange = applyVisibility;
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
  $('savePng').onclick = () => download(snapshot(1600), 'femweb.png');
  $('saveGif').onclick = saveGif;
  $('saveVideo').onclick = saveVideo;
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
  bindBH();
  $('llmSave').onclick = () => llmRefresh(true).catch((e) => note(e.message, true));
  let down = null;
  canvas.addEventListener('pointerdown', (e) => (down = [e.clientX, e.clientY, e.button]));
  canvas.addEventListener('pointerup', (e) => {
    if (down && Math.hypot(e.clientX - down[0], e.clientY - down[1]) < 4) {
      if (down[2] === 2) pickInfo(e);
      else if (down[2] === 0 && !BH.edit) pickNode(e);
    }
    down = null;
  });
  canvas.addEventListener('contextmenu', (e) => e.preventDefault());
}

const BH = { edges: [], note: '', sel: new Set(), edit: false, drag: null, k3: null };
const BH_COL = { fold: 0xff6b5e, bend: 0x5ec8ff, cut: 0xffd84a, boundary: 0x7d8a99 };
const BH_NAME = { fold: '접힘', bend: '휨', cut: '자름', boundary: '경계' };
const ekey = (e) => (e[0] < e[1] ? `${e[0]}-${e[1]}` : `${e[1]}-${e[0]}`);

async function loadCreases() {
  const on = !!st && !!st.spec && st.spec.solver === 'bh' && !!st.model && !!st.model.counts.shells;
  if (!on) {
    BH.edges = [];
    clear(G.crease);
    return;
  }
  try {
    const r = await api('/api/creases');
    BH.edges = r.edges || [];
    BH.note = r.note || '';
  } catch (e) {
    BH.edges = [];
    BH.note = e.message;
  }
  const known = new Set(BH.edges.map((c) => ekey(c.edge)));
  for (const k of [...BH.sel]) if (!known.has(k)) BH.sel.delete(k);
  drawCreases();
  renderBH();
}

function drawCreases() {
  clear(G.crease);
  if (!BH.edges.length || !st.view) return;
  const groups = {};
  for (const c of BH.edges) {
    const t = BH.sel.has(ekey(c.edge)) ? 'sel' : c.type;
    (groups[t] = groups[t] || []).push(...st.view.nodes[c.edge[0]], ...st.view.nodes[c.edge[1]]);
  }
  for (const [t, arr] of Object.entries(groups)) {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(arr), 3));
    const m = new THREE.LineBasicMaterial({ color: t === 'sel' ? 0xffffff : BH_COL[t], depthTest: t !== 'boundary', transparent: true, opacity: t === 'boundary' ? 0.6 : 1 });
    const ls = new THREE.LineSegments(g, m);
    ls.renderOrder = t === 'sel' ? 3 : 2;
    G.crease.add(ls);
  }
  applyVisibility();
}

function renderBH() {
  const on = !!st && !!st.spec && st.spec.solver === 'bh';
  $('bhPanel').hidden = !on;
  if (!on) {
    if (BH.edit) setEdit(false);
    return;
  }
  const c = st.spec.bh || {};
  $('bhFoldMode').value = c.fold_mode || 'auto';
  $('bhLratio').value = c.Lstar_ratio ?? 3;
  const mode = c.fold_mode || 'auto';
  $('bhPctBox').firstChild.textContent = mode === 'value' ? 'k (N·mm/rad/mm)' : mode === 'percent' ? 'B 대비 %' : 'L* (mm)';
  $('bhPct').value = (mode === 'value' ? c.fold_k : mode === 'percent' ? c.fold_percent ?? 10 : c.Lstar) ?? '';
  $('bhLratio').disabled = mode !== 'auto';
  $('bhBarY').value = c.bar_yield ?? '';
  $('bhBarH').value = c.bar_H || '';
  $('bhFoldY').value = c.fold_yield ?? '';
  $('bhFoldH').value = c.fold_H || '';
  $('bhBendY').value = c.bend_yield ?? '';
  $('bhBendH').value = c.bend_H || '';
  $('bhTol').value = c.tol ?? 3;
  $('bhContact').checked = !!c.contact;
  $('bhGap').value = c.gap ?? '';
  $('bhGap').disabled = !c.contact;
  $('bhCycles').value = c.cycles || '';
  $('bhCycLow').value = c.cycle_low || '';
  $('bhArea').value = c.bar_area ?? '';
  $('bhMass').checked = c.use_mass !== false;
  if (document.activeElement !== $('bhTargets')) $('bhTargets').value = (c.targets || []).join(', ');
  const cnt = {};
  for (const e of BH.edges) cnt[e.type] = (cnt[e.type] || 0) + 1;
  $('bhCount').textContent = Object.entries(cnt).map(([k, v]) => `${BH_NAME[k] || k} ${v}`).join(' · ');
  const sel = [...BH.sel];
  const lines = [];
  lines.push(sel.length ? `선택 ${sel.length}개: ${sel.slice(0, 12).join(', ')}${sel.length > 12 ? ' …' : ''}` : BH.edit ? '드래그로 엣지를 고르세요 (왼→오 완전 포함, 오→왼 걸침, Shift 추가, Ctrl 제외, Esc 해제)' : '주름 편집 모드를 켜면 상자 선택으로 여러 엣지의 종류를 바꿀 수 있습니다');
  const man = Object.keys(st.spec.creases || {}).length;
  if (man) lines.push(`직접 지정 ${man}개`);
  if ((c.drivers || []).length) lines.push(`구동: ${c.drivers.map((d, i) => `${d}${c.targets && c.targets[i] !== undefined ? ' → ' + c.targets[i] + '°' : ''}`).join(', ')}`);
  if ((c.plot3 || []).length) lines.push(`3D 해공간: ${c.plot3.join(', ')}`);
  if ((c.watch || []).length) lines.push(`이력: ${c.watch.join(', ')}`);
  if (BH.note) lines.push(BH.note);
  $('bhSel').textContent = lines.join('\n');
  $('bhSel').style.whiteSpace = 'pre-line';
}

function setEdit(on) {
  BH.edit = on;
  $('bhEdit').textContent = on ? '주름 편집 모드 켬 (Esc 해제)' : '주름 편집 모드 끔';
  $('bhEdit').classList.toggle('on', on);
  controls.mouseButtons.LEFT = on ? null : THREE.MOUSE.ROTATE;
  applyVisibility();
  renderBH();
}

function screenEdges() {
  const rect = canvas.getBoundingClientRect();
  const q = new THREE.Vector3();
  const pts = new Map();
  const P = (i) => {
    if (pts.has(i)) return pts.get(i);
    const p = st.view.nodes[i];
    q.set(p[0], p[1], p[2]).project(camera);
    const r = q.z < -1 || q.z > 1 ? null : [(q.x * 0.5 + 0.5) * rect.width + rect.left, (-q.y * 0.5 + 0.5) * rect.height + rect.top];
    pts.set(i, r);
    return r;
  };
  return BH.edges.filter((c) => c.type !== 'boundary').map((c) => ({ key: ekey(c.edge), a: P(c.edge[0]), b: P(c.edge[1]) })).filter((s) => s.a && s.b);
}

function segHitsRect(a, b, x0, y0, x1, y1) {
  let t0 = 0;
  let t1 = 1;
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const p = [-dx, dx, -dy, dy];
  const qq = [a[0] - x0, x1 - a[0], a[1] - y0, y1 - a[1]];
  for (let i = 0; i < 4; i++) {
    if (p[i] === 0) {
      if (qq[i] < 0) return false;
    } else {
      const r = qq[i] / p[i];
      if (p[i] < 0) t0 = Math.max(t0, r);
      else t1 = Math.min(t1, r);
      if (t0 > t1) return false;
    }
  }
  return true;
}

function segDist(a, b, x, y) {
  const dx = b[0] - a[0];
  const dy = b[1] - a[1];
  const L = dx * dx + dy * dy || 1e-12;
  const t = Math.max(0, Math.min(1, ((x - a[0]) * dx + (y - a[1]) * dy) / L));
  return Math.hypot(a[0] + t * dx - x, a[1] + t * dy - y);
}

function applySel(keys, ev) {
  if (ev.ctrlKey || ev.metaKey) keys.forEach((k) => BH.sel.delete(k));
  else {
    if (!ev.shiftKey) BH.sel.clear();
    keys.forEach((k) => BH.sel.add(k));
  }
  drawCreases();
  renderBH();
}

function bhPointerDown(e) {
  if (!BH.edit || e.button !== 0) return;
  BH.drag = { x: e.clientX, y: e.clientY };
}

function bhPointerMove(e) {
  if (!BH.drag) return;
  const b = $('selBox');
  const x0 = Math.min(BH.drag.x, e.clientX);
  const y0 = Math.min(BH.drag.y, e.clientY);
  if (Math.hypot(e.clientX - BH.drag.x, e.clientY - BH.drag.y) < 4) return;
  b.style.display = 'block';
  b.className = e.clientX >= BH.drag.x ? 'win' : 'cross';
  b.style.left = x0 + 'px';
  b.style.top = y0 + 'px';
  b.style.width = Math.abs(e.clientX - BH.drag.x) + 'px';
  b.style.height = Math.abs(e.clientY - BH.drag.y) + 'px';
}

function bhPointerUp(e) {
  if (!BH.drag) return;
  const d = BH.drag;
  BH.drag = null;
  $('selBox').style.display = 'none';
  const segs = screenEdges();
  if (Math.hypot(e.clientX - d.x, e.clientY - d.y) < 4) {
    let best = null;
    let bd = 8;
    for (const s of segs) {
      const dd = segDist(s.a, s.b, e.clientX, e.clientY);
      if (dd < bd) {
        bd = dd;
        best = s.key;
      }
    }
    applySel(best ? [best] : [], e);
    return;
  }
  const x0 = Math.min(d.x, e.clientX);
  const x1 = Math.max(d.x, e.clientX);
  const y0 = Math.min(d.y, e.clientY);
  const y1 = Math.max(d.y, e.clientY);
  const inside = (p) => p[0] >= x0 && p[0] <= x1 && p[1] >= y0 && p[1] <= y1;
  const win = e.clientX >= d.x;
  applySel(segs.filter((s) => (win ? inside(s.a) && inside(s.b) : segHitsRect(s.a, s.b, x0, y0, x1, y1))).map((s) => s.key), e);
}

function bhNum(v) {
  const s = String(v).trim();
  return s === '' ? null : +s;
}

function bindBH() {
  const box = el('div');
  box.id = 'selBox';
  document.body.append(box);
  canvas.addEventListener('pointerdown', bhPointerDown);
  window.addEventListener('pointermove', bhPointerMove);
  window.addEventListener('pointerup', bhPointerUp);
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && BH.edit && !['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName)) {
      if (BH.sel.size) applySel([], { shiftKey: false });
      else setEdit(false);
    }
  });
  $('bhEdit').onclick = () => setEdit(!BH.edit);
  document.querySelectorAll('.bhType').forEach((b) => {
    b.onclick = () => {
      const es = [...BH.sel].map((k) => k.split('-').map(Number));
      if (!es.length) return note('먼저 주름 편집 모드에서 엣지를 선택하세요', true);
      ops([{ op: 'set_crease', edges: es, type: b.dataset.t }]);
    };
  });
  const setbh = (o) => ops([{ op: 'set_bh', ...o }]);
  $('bhFoldMode').onchange = (e) => setbh({ fold_mode: e.target.value });
  $('bhLratio').onchange = (e) => setbh({ Lstar_ratio: bhNum(e.target.value) ?? 3 });
  $('bhPct').onchange = (e) => {
    const m = $('bhFoldMode').value;
    const v = bhNum(e.target.value);
    setbh(m === 'value' ? { fold_k: v } : m === 'percent' ? { fold_percent: v ?? 10 } : { Lstar: v });
  };
  const num = { bhBarY: 'bar_yield', bhBarH: 'bar_H', bhFoldY: 'fold_yield', bhFoldH: 'fold_H', bhBendY: 'bend_yield', bhBendH: 'bend_H', bhGap: 'gap', bhTol: 'tol' };
  for (const [id, f] of Object.entries(num)) $(id).onchange = (e) => setbh({ [f]: bhNum(e.target.value) ?? (f.endsWith('_H') ? 0 : f === 'tol' ? 3 : null) });
  $('bhContact').onchange = (e) => setbh({ contact: e.target.checked });
  $('bhMass').onchange = (e) => setbh({ use_mass: e.target.checked });
  $('bhCycles').onchange = (e) => setbh({ cycles: Math.max(0, Math.round(bhNum(e.target.value) || 0)) });
  $('bhCycLow').onchange = (e) => setbh({ cycle_low: bhNum(e.target.value) || 0 });
  $('bhArea').onchange = (e) => setbh({ bar_area: bhNum(e.target.value) });
  const selFolds = () => [...BH.sel].filter((k) => (BH.edges.find((c) => ekey(c.edge) === k) || {}).type === 'fold');
  const targets = (n) => {
    const t = $('bhTargets').value.split(/[,\s]+/).filter(Boolean).map(Number);
    if (t.some((v) => !Number.isFinite(v))) return null;
    if (t.length === 1 && n > 1) return Array(n).fill(t[0]);
    return t;
  };
  $('bhDrivers').onclick = () => {
    const d = selFolds();
    if (!d.length) return note('구동 주름은 접힘(fold) 주름이어야 합니다', true);
    const t = targets(d.length);
    if (!t || t.length !== d.length) return note(`목표 이면각을 ${d.length}개 (또는 한 개) 넣어 주세요. 평면은 180°`, true);
    setbh({ drivers: d, targets: t });
  };
  $('bhTargets').onchange = () => {
    const c = st.spec.bh || {};
    const n = (c.drivers || []).length;
    const t = targets(n);
    if (n && t && t.length === n) setbh({ targets: t });
  };
  $('bhClearDrv').onclick = () => setbh({ drivers: [], targets: [] });
  $('bhPlot3').onclick = () => {
    const d = selFolds();
    if (d.length !== 3) return note(`접힘 주름을 정확히 3개 선택하세요 (지금 ${d.length}개)`, true);
    setbh({ plot3: d });
  };
  $('bhWatch').onclick = () => {
    const d = [...BH.sel];
    setbh({ watch: d });
  };
}

function textSprite(text, color) {
  const c = document.createElement('canvas');
  c.width = 256;
  c.height = 64;
  const g = c.getContext('2d');
  g.font = '28px sans-serif';
  g.fillStyle = color || '#dfe7f1';
  g.textBaseline = 'middle';
  g.fillText(text, 4, 32);
  const sp = new THREE.Sprite(new THREE.SpriteMaterial({ map: new THREE.CanvasTexture(c), depthTest: false }));
  sp.scale.set(0.7, 0.175, 1);
  sp.center.set(0, 0.5);
  return sp;
}

async function loadKin3d() {
  if (BH.k3) {
    BH.k3.stop = true;
    BH.k3.renderer.dispose();
    BH.k3 = null;
  }
  if (!st || !st.result || !st.spec || st.spec.solver !== 'bh' || st.spec.step.type !== 'kinematic') return;
  let k;
  try {
    k = await api('/api/result/kin3d');
  } catch (e) {
    return;
  }
  if (!k || !k.edges) return;
  const all = [...(k.paths || []).flat(), ...(k.main || [])];
  if (!all.length) return;
  const lo = [0, 1, 2].map((i) => Math.min(...all.map((p) => p[i])));
  const hi = [0, 1, 2].map((i) => Math.max(...all.map((p) => p[i])));
  const nrm = (p) => p.map((v, i) => (hi[i] - lo[i] > 1e-9 ? (v - lo[i]) / (hi[i] - lo[i]) : 0.5) - 0.5);
  const d = el('div', 'plot kin3d');
  const t = el('div', 'ptitle');
  t.append(el('span', '', `기구 해공간 · 자유도 ${k.dof} · 이면각 (도)`));
  d.append(t);
  const cv = document.createElement('canvas');
  d.append(cv);
  const lg = el('div', 'legend');
  lg.innerHTML = `<span><i style="background:#ff6b5e"></i>X ${k.edges[0]}</span><span><i style="background:#7fe07f"></i>Y ${k.edges[1]}</span><span><i style="background:#5ec8ff"></i>Z ${k.edges[2]}</span><span><i style="background:#ffd84a"></i>해석 경로</span><span><i style="background:#9fb3c8"></i>방향별 경로</span>`;
  d.append(lg);
  $('plots').append(d);
  const w = cv.clientWidth || 420;
  const h = 320;
  const rd = new THREE.WebGLRenderer({ canvas: cv, antialias: true, alpha: true });
  rd.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  rd.setSize(w, h, false);
  const sc = new THREE.Scene();
  const cam = new THREE.PerspectiveCamera(35, w / h, 0.01, 100);
  cam.up.set(0, 0, 1);
  cam.position.set(1.6, -2.0, 1.3);
  const oc = new OrbitControls(cam, cv);
  oc.enableDamping = true;
  const axes = [[0xff6b5e, [1, 0, 0]], [0x7fe07f, [0, 1, 0]], [0x5ec8ff, [0, 0, 1]]];
  axes.forEach(([col, a], i) => {
    const g = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(-0.5 + a[0], -0.5 + a[1], -0.5 + a[2])]);
    sc.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: col })));
    const s0 = textSprite(lo[i].toFixed(1), '#' + col.toString(16));
    s0.position.set(-0.5 + a[0] * 0.12, -0.5 + a[1] * 0.12, -0.5 + a[2] * 0.12 - 0.05);
    const s1 = textSprite(hi[i].toFixed(1), '#' + col.toString(16));
    s1.position.set(-0.5 + a[0], -0.5 + a[1], -0.5 + a[2]);
    sc.add(s0, s1);
  });
  sc.add(new THREE.Box3Helper(new THREE.Box3(new THREE.Vector3(-0.5, -0.5, -0.5), new THREE.Vector3(0.5, 0.5, 0.5)), 0x334152));
  for (const path of k.paths || []) {
    if (path.length < 2) continue;
    const g = new THREE.BufferGeometry().setFromPoints(path.map((p) => new THREE.Vector3(...nrm(p))));
    sc.add(new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0x9fb3c8, transparent: true, opacity: 0.7 })));
  }
  if ((k.main || []).length > 1) {
    const g = new THREE.BufferGeometry().setFromPoints(k.main.map((p) => new THREE.Vector3(...nrm(p))));
    const ml = new THREE.Line(g, new THREE.LineBasicMaterial({ color: 0xffd84a, depthTest: false }));
    ml.renderOrder = 2;
    sc.add(ml);
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.025), new THREE.MeshBasicMaterial({ color: 0xffd84a }));
    dot.position.set(...nrm(k.main[0]));
    sc.add(dot);
  }
  const ctx = { renderer: rd, stop: false };
  BH.k3 = ctx;
  const tick = () => {
    if (ctx.stop || !cv.isConnected) return;
    oc.update();
    rd.render(sc, cam);
    requestAnimationFrame(tick);
  };
  tick();
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
