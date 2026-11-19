import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';

const C = { bg: '#0b0f15', grid: '#1f2833', axis: '#5d6b7e', text: '#dbe3ee', mute: '#8492a6', a: '#4cc2ff', b: '#ff5fa8', ok: '#57d9a3', w: '#ffc857', e: '#ff6b6b' };
const $ = (s, r = document) => r.querySelector(s);
const fmt = (v, d = 3) => (Math.abs(v) >= 1e4 || (Math.abs(v) < 1e-3 && v !== 0) ? v.toExponential(2) : (+v.toFixed(d)).toString());

function el(tag, cls, html) {
  const e = document.createElement(tag);
  if (cls) e.className = cls;
  if (html !== undefined) e.innerHTML = html;
  return e;
}

function slider(box, label, min, max, step, val, unit, on, log = false) {
  const w = el('label', 'sl');
  const t = el('span', 'k', label);
  const v = el('span', 'v');
  const i = el('input');
  i.type = 'range';
  i.min = log ? Math.log10(min) : min;
  i.max = log ? Math.log10(max) : max;
  i.step = log ? 0.01 : step;
  i.value = log ? Math.log10(val) : val;
  const get = () => (log ? 10 ** +i.value : +i.value);
  const show = () => (v.textContent = `${fmt(get(), 4)}${unit ? ' ' + unit : ''}`);
  i.oninput = () => {
    show();
    on && on(get());
  };
  show();
  w.append(t, i, v);
  box.append(w);
  return { get, set: (x) => { i.value = log ? Math.log10(x) : x; show(); }, input: i };
}

function button(box, label, on, cls = '') {
  const b = el('button', 'lb ' + cls, label);
  b.onclick = on;
  box.append(b);
  return b;
}

function canvas(box, h = 300) {
  const c = el('canvas', 'lc');
  c.style.height = h + 'px';
  box.append(c);
  const ctx = c.getContext('2d');
  const fit = () => {
    const r = c.getBoundingClientRect();
    const d = window.devicePixelRatio || 1;
    c.width = Math.max(10, r.width * d);
    c.height = r.height * d;
    ctx.setTransform(d, 0, 0, d, 0, 0);
    return { w: r.width, h: r.height };
  };
  return { c, ctx, fit };
}

function plot(ctx, W, H, box, opt) {
  const { x0, x1, y0, y1, xl, yl } = opt;
  const L = 54, R = 14, T = 16, B = 34;
  const X = (x) => L + ((x - x0) / (x1 - x0)) * (W - L - R);
  const Y = (y) => H - B - ((y - y0) / (y1 - y0)) * (H - T - B);
  ctx.fillStyle = C.bg;
  ctx.fillRect(0, 0, W, H);
  ctx.strokeStyle = C.grid;
  ctx.lineWidth = 1;
  ctx.font = '11px system-ui';
  ctx.fillStyle = C.mute;
  const tick = (a, b, n) => Array.from({ length: n + 1 }, (_, k) => a + ((b - a) * k) / n);
  const tf = (v, a, b, n) => { const st = Math.abs(b - a) / n; const m = Math.max(Math.abs(a), Math.abs(b)); if (m < 1e-2 || m >= 1e5) return v === 0 ? '0' : v.toExponential(1); const d = Math.max(0, Math.min(6, 1 - Math.floor(Math.log10(st)))); return (+v.toFixed(d)).toString(); };
  for (const x of tick(x0, x1, 5)) {
    ctx.beginPath(); ctx.moveTo(X(x), T); ctx.lineTo(X(x), H - B); ctx.stroke();
    ctx.textAlign = 'center'; ctx.fillText(tf(x, x0, x1, 5), X(x), H - B + 14);
  }
  for (const y of tick(y0, y1, 4)) {
    ctx.beginPath(); ctx.moveTo(L, Y(y)); ctx.lineTo(W - R, Y(y)); ctx.stroke();
    ctx.textAlign = 'right'; ctx.fillText(tf(y, y0, y1, 4), L - 6, Y(y) + 4);
  }
  ctx.strokeStyle = C.axis;
  if (y0 < 0 && y1 > 0) { ctx.beginPath(); ctx.moveTo(L, Y(0)); ctx.lineTo(W - R, Y(0)); ctx.stroke(); }
  if (x0 < 0 && x1 > 0) { ctx.beginPath(); ctx.moveTo(X(0), T); ctx.lineTo(X(0), H - B); ctx.stroke(); }
  ctx.fillStyle = C.text;
  ctx.textAlign = 'right';
  ctx.fillText(xl || '', W - R, H - 4);
  ctx.textAlign = 'left';
  ctx.fillText(yl || '', L + 4, T + 10);
  return { X, Y };
}

function line(ctx, pts, color, w = 2, dash) {
  if (!pts.length) return;
  ctx.strokeStyle = color;
  ctx.lineWidth = w;
  ctx.setLineDash(dash || []);
  ctx.beginPath();
  pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y)));
  ctx.stroke();
  ctx.setLineDash([]);
}

function dot(ctx, x, y, color, r = 5) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, 7);
  ctx.fill();
}

function label(ctx, s, x, y, color = C.text, align = 'left', size = 12) {
  ctx.fillStyle = color;
  ctx.font = `${size}px system-ui`;
  ctx.textAlign = align;
  ctx.fillText(s, x, y);
}

function readout(box) {
  const r = el('div', 'ro');
  box.append(r);
  return (items) => (r.innerHTML = items.map(([k, v, c]) => `<span><i>${k}</i><b style="color:${c || C.text}">${v}</b></span>`).join(''));
}

function frame(id, title, note) {
  const host = document.querySelector(`[data-lab="${id}"]`);
  if (!host) return null;
  host.classList.add('lab');
  host.innerHTML = '';
  const h = el('div', 'lh', `<b>${title}</b><span>${note || ''}</span>`);
  const body = el('div', 'lbody');
  const ctl = el('div', 'lctl');
  const view = el('div', 'lview');
  body.append(view, ctl);
  host.append(h, body);
  return { host, ctl, view };
}

const sub = (a, b) => [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
const cross = (a, b) => [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
const dotv = (a, b) => a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
const norm = (a) => Math.hypot(a[0], a[1], a[2]);

function dihedral(xi, xj, xk, xl) {
  const rij = sub(xi, xj), rkj = sub(xk, xj), rkl = sub(xk, xl);
  const m = cross(rij, rkj), n = cross(rkj, rkl);
  const c = Math.max(-1, Math.min(1, dotv(m, n) / (norm(m) * norm(n))));
  let eta = Math.sign(dotv(m, rkl)) || 1;
  let th = eta * Math.acos(c);
  th = ((th % (2 * Math.PI)) + 2 * Math.PI) % (2 * Math.PI);
  const L = norm(rkj);
  return { th, gi: m.map((q) => (q * L) / dotv(m, m)), gl: n.map((q) => (-q * L) / dotv(n, n)) };
}

function labDihedral() {
  const f = frame('dihedral', '이면각 θ 와 그 경사 ∇θ', '두 삼각형이 주름 j–k 를 공유합니다. 슬라이더로 접고, 마우스로 돌려 보세요');
  if (!f) return;
  const w = 640, h = 340;
  const r = new THREE.WebGLRenderer({ antialias: true });
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  f.view.append(r.domElement);
  r.domElement.className = 'l3';
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.bg);
  const cam = new THREE.PerspectiveCamera(40, w / h, 0.01, 100);
  cam.position.set(2.2, -2.6, 2.0);
  cam.up.set(0, 0, 1);
  const oc = new OrbitControls(cam, r.domElement);
  oc.target.set(0, 0, 0.2);
  scene.add(new THREE.AmbientLight(0xffffff, 0.55));
  const dl = new THREE.DirectionalLight(0xffffff, 1.2);
  dl.position.set(2, -3, 4);
  scene.add(dl);
  const grid = new THREE.GridHelper(4, 16, 0x2a3340, 0x1f2833);
  grid.rotation.x = Math.PI / 2;
  scene.add(grid);
  const geoA = new THREE.BufferGeometry(), geoB = new THREE.BufferGeometry();
  const mA = new THREE.Mesh(geoA, new THREE.MeshStandardMaterial({ color: 0x4cc2ff, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }));
  const mB = new THREE.Mesh(geoB, new THREE.MeshStandardMaterial({ color: 0xffc857, side: THREE.DoubleSide, transparent: true, opacity: 0.85 }));
  scene.add(mA, mB);
  const axis = new THREE.Line(new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(-1, 0, 0), new THREE.Vector3(1, 0, 0)]), new THREE.LineBasicMaterial({ color: 0xff5fa8 }));
  scene.add(axis);
  const ai = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0x57d9a3, 0.12, 0.07);
  const al = new THREE.ArrowHelper(new THREE.Vector3(0, 0, 1), new THREE.Vector3(), 1, 0xff6b6b, 0.12, 0.07);
  for (const a of [ai, al]) a.traverse((o) => { if (o.material) { o.material.depthTest = false; o.material.transparent = true; o.renderOrder = 9; } });
  scene.add(ai, al);
  const ro = readout(f.ctl);
  const s = slider(f.ctl, '접힘 ρ = π − θ', -175, 175, 1, 60, '°', draw);
  const sg = slider(f.ctl, '경사 화살표 배율', 0.1, 1, 0.05, 0.5, '', draw);
  const hint = el('p', 'lnote', '초록 화살표: ∂θ/∂x<sub>i</sub>, 빨강: ∂θ/∂x<sub>l</sub>. 날개를 화살표 방향으로 밀면 θ 가 커집니다. θ 는 [0, 2π) 에서 정의되고 평면이 π 입니다. 접힘이 ±180° 근처로 가면 패널이 겹치는데, 이것을 막는 것이 기하 접촉의 "힌지각 장벽" 입니다.');
  f.ctl.append(hint);
  function draw() {
    const rho = (s.get() * Math.PI) / 180;
    const xj = [-1, 0, 0], xk = [1, 0, 0];
    const xi = [0.2, 1.1, 0];
    const xl = [-0.1, -1.1 * Math.cos(rho), 1.1 * Math.sin(rho)];
    geoA.setAttribute('position', new THREE.Float32BufferAttribute([...xi, ...xj, ...xk], 3));
    geoB.setAttribute('position', new THREE.Float32BufferAttribute([...xl, ...xk, ...xj], 3));
    geoA.computeVertexNormals();
    geoB.computeVertexNormals();
    const d = dihedral(xi, xj, xk, xl);
    const sc = sg.get();
    const put = (a, p, g) => {
      const v = new THREE.Vector3(...g);
      const L = v.length();
      a.position.set(...p);
      a.setDirection(v.normalize());
      a.setLength(Math.max(0.05, L * sc), 0.12, 0.07);
    };
    put(ai, xi, d.gi);
    put(al, xl, d.gl);
    ro([
      ['θ', `${fmt((d.th * 180) / Math.PI, 1)}°`, C.a],
      ['ρ = π − θ', `${fmt(180 - (d.th * 180) / Math.PI, 1)}°`, C.b],
      ['|∂θ/∂xᵢ|', fmt(norm(d.gi), 3), C.ok],
      ['|∂θ/∂xₗ|', fmt(norm(d.gl), 3), C.e],
    ]);
  }
  const loop = () => {
    const rr = r.domElement.getBoundingClientRect();
    if (rr.width > 0 && (r.domElement.width !== Math.round(rr.width * r.getPixelRatio()))) {
      r.setSize(rr.width, rr.height, false);
      cam.aspect = rr.width / rr.height;
      cam.updateProjectionMatrix();
    }
    oc.update();
    r.render(scene, cam);
    requestAnimationFrame(loop);
  };
  draw();
  loop();
}

function solveDense(K, F) {
  const n = F.length;
  const A = K.map((r, i) => [...r, F[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(A[r][c]) > Math.abs(A[p][c])) p = r;
    [A[c], A[p]] = [A[p], A[c]];
    const d = A[c][c];
    if (Math.abs(d) < 1e-14) return null;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const q = A[r][c] / d;
      for (let k = c; k <= n; k++) A[r][k] -= q * A[c][k];
    }
  }
  return A.map((r, i) => r[n] / r[i]);
}

function truss2(nodes, bars, E, fixed, presc) {
  const n = nodes.length * 2;
  const K = Array.from({ length: n }, () => new Array(n).fill(0));
  for (const [a, b, A] of bars) {
    const dx = nodes[b][0] - nodes[a][0], dy = nodes[b][1] - nodes[a][1];
    const L = Math.hypot(dx, dy), c = dx / L, s = dy / L, k = (E * A) / L;
    const m = [[c * c, c * s], [c * s, s * s]];
    const id = [2 * a, 2 * a + 1, 2 * b, 2 * b + 1];
    for (let i = 0; i < 4; i++) for (let j = 0; j < 4; j++) K[id[i]][id[j]] += (i < 2 === j < 2 ? 1 : -1) * k * m[i % 2][j % 2];
  }
  const known = new Map(fixed.map((d) => [d, 0]));
  for (const [d, v] of presc) known.set(d, v);
  const free = [...Array(n).keys()].filter((d) => !known.has(d));
  const u = new Array(n).fill(0);
  for (const [d, v] of known) u[d] = v;
  const Kff = free.map((i) => free.map((j) => K[i][j]));
  const F = free.map((i) => -[...known].reduce((s, [d, v]) => s + K[i][d] * v, 0));
  const x = solveDense(Kff, F);
  if (!x) return null;
  free.forEach((d, q) => (u[d] = x[q]));
  const R = K.map((r) => r.reduce((s, kij, j) => s + kij * u[j], 0));
  return { u, R };
}

function labN5B8() {
  const f = frame('n5b8', 'N5B8 패널: 두께로 정한 바 면적과 면내 강성 확인', '패널 치수와 ν 를 바꾸면 바 면적이 다시 계산되고, 5절점 트러스를 바로 풀어 연속체 이론값과 비교합니다');
  if (!f) return;
  const cv = canvas(f.view, 320);
  const ro = readout(f.ctl);
  const sW = slider(f.ctl, '폭 W', 5, 40, 0.5, 20, 'mm', draw);
  const sH = slider(f.ctl, '높이 H', 5, 40, 0.5, 20, 'mm', draw);
  const sN = slider(f.ctl, '푸아송비 ν', 0, 0.49, 0.01, 0.333, '', draw);
  const st = slider(f.ctl, '두께 t', 0.1, 2, 0.05, 0.5, 'mm', draw);
  let mode = 'tension';
  const bx = el('div', 'lrow');
  f.ctl.append(bx);
  const b1 = button(bx, '인장 시험', () => { mode = 'tension'; draw(); }, 'on');
  const b2 = button(bx, '단순 전단', () => { mode = 'shear'; draw(); });
  f.ctl.append(el('p', 'lnote', 'Filipov 2017 식 5 는 인장은 모든 ν 에서, 단순 전단은 ν = 1/3 에서 정확히 맞게 만든 식입니다. ν 를 1/3 에서 벗어나게 하면 전단 비율이 1 에서 멀어지는 것을 볼 수 있습니다. 바 굵기는 면적에 비례하게 그렸습니다.'));
  window.addEventListener('resize', draw);
  function draw() {
    b1.classList.toggle('on', mode === 'tension');
    b2.classList.toggle('on', mode === 'shear');
    const W = sW.get(), H = sH.get(), nu = sN.get(), t = st.get(), E = 1000;
    const AX = Math.max((t * (H * H - nu * W * W)) / (2 * H * (1 - nu * nu)), 0.05 * t * H);
    const AY = Math.max((t * (W * W - nu * H * H)) / (2 * W * (1 - nu * nu)), 0.05 * t * W);
    const AD = Math.max((t * nu * (H * H + W * W) ** 1.5) / (2 * H * W * (1 - nu * nu)), 0.05 * t * Math.min(W, H));
    const N = [[0, 0], [W, 0], [W, H], [0, H], [W / 2, H / 2]];
    const bars = [[0, 1, AX], [1, 2, AY], [2, 3, AX], [3, 0, AY], [0, 4, AD], [1, 4, AD], [2, 4, AD], [3, 4, AD]];
    const d = 1e-4 * H;
    let ratio, ref, sol;
    if (mode === 'tension') {
      sol = truss2(N, bars, E, [0, 1, 3], [[5, d], [7, d]]);
      ref = (E * W * t) / H;
      ratio = sol ? (sol.R[5] + sol.R[7]) / d / ref : NaN;
    } else {
      const G = E / (2 * (1 + nu));
      sol = truss2(N, bars, E, [0, 1, 2, 3, 5, 7], [[4, d], [6, d]]);
      ref = (G * W * t) / H;
      ratio = sol ? (sol.R[4] + sol.R[6]) / d / ref : NaN;
    }
    const { w, h } = cv.fit();
    const ctx = cv.ctx;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const sc = Math.min((w * 0.55) / W, (h - 60) / H);
    const ox = 40, oy = h - 30;
    const amp = (0.3 * H) / d;
    const P = (i, def) => {
      const u = def && sol ? [sol.u[2 * i] * amp, sol.u[2 * i + 1] * amp] : [0, 0];
      return [ox + (N[i][0] + u[0]) * sc, oy - (N[i][1] + u[1]) * sc];
    };
    const Amax = Math.max(AX, AY, AD);
    for (const [a, b, A] of bars) line(ctx, [P(a, false), P(b, false)], '#2a3340', 1, [4, 4]);
    for (const [a, b, A] of bars) line(ctx, [P(a, true), P(b, true)], A === AD ? C.b : A === AX ? C.a : C.ok, 1 + (7 * A) / Amax);
    for (let i = 0; i < 5; i++) dot(ctx, ...P(i, true), C.text, 3.5);
    const tx = ox + W * sc + 30;
    label(ctx, '변형 (과장)', tx, 30, C.mute);
    label(ctx, `A_X = ${fmt(AX, 4)} mm²`, tx, 56, C.a);
    label(ctx, `A_Y = ${fmt(AY, 4)} mm²`, tx, 76, C.ok);
    label(ctx, `A_D = ${fmt(AD, 4)} mm²`, tx, 96, C.b);
    label(ctx, mode === 'tension' ? '아래 고정, 위 y 로 당김' : '아래 고정, 위 x 로 밀기 (y 구속)', tx, 126, C.mute);
    ro([
      ['트러스 강성 / 이론', fmt(ratio, 5), Math.abs(ratio - 1) < 1e-3 ? C.ok : C.w],
      ['이론', mode === 'tension' ? 'E W t / H' : 'G W t / H', C.mute],
    ]);
  }
  draw();
}

function labFold() {
  const f = frame('foldk', '접힘 강성 K_F: L* 와 주름 길이의 영향', 'K_ℓ (주름 길이 척도) 와 K_m (국부 굽힘) 이 직렬로 이어집니다');
  if (!f) return;
  const cv = canvas(f.view, 300);
  const ro = readout(f.ctl);
  const sE = slider(f.ctl, 'E', 100, 210000, 100, 3000, 'MPa', draw, true);
  const sT = slider(f.ctl, 't', 0.05, 3, 0.05, 0.5, 'mm', draw);
  const sNu = slider(f.ctl, 'ν', 0, 0.49, 0.01, 0.3, '', draw);
  const sLF = slider(f.ctl, '주름 길이 L_F', 5, 200, 1, 50, 'mm', draw);
  const sL = slider(f.ctl, 'L*', 1, 2000, 1, 150, 'mm', draw, true);
  f.ctl.append(el('p', 'lnote', '세로축은 주름 길이당 강성 K_F/L_F (N·mm/rad/mm). L* 가 작으면 K_m 이 한계가 되어 평평해지고, 크면 K_ℓ ∝ 1/L* 를 따라 내려갑니다. 빈칸이면 L* = 3·L_F (점선).'));
  window.addEventListener('resize', draw);
  function draw() {
    const E = sE.get(), t = sT.get(), nu = sNu.get(), LF = sLF.get(), Ls = sL.get();
    const B = (E * t ** 3) / (12 * (1 - nu * nu));
    const Km = 0.55 * B * Math.cbrt(LF / t);
    const kf = (L) => 1 / (L / (LF * B) + 1 / Km) / LF;
    const { w, h } = cv.fit();
    const xs = Array.from({ length: 200 }, (_, i) => 10 ** (0 + (i / 199) * 3.4));
    const ys = xs.map(kf);
    const lx = (x) => Math.log10(x);
    const ymax = Math.log10(Math.max(...ys) * 2), ymin = Math.log10(Math.min(...ys) / 2);
    const { X, Y } = plot(cv.ctx, w, h, null, { x0: 0, x1: 3.4, y0: ymin, y1: ymax, xl: 'log₁₀ L* (mm)', yl: 'log₁₀ K_F/L_F' });
    line(cv.ctx, xs.map((x, i) => [X(lx(x)), Y(Math.log10(ys[i]))]), C.a, 2.5);
    line(cv.ctx, xs.map((x) => [X(lx(x)), Y(Math.log10(B / x))]), C.ok, 1.2, [5, 4]);
    line(cv.ctx, [[X(0), Y(Math.log10(Km / LF))], [X(3.4), Y(Math.log10(Km / LF))]], C.b, 1.2, [5, 4]);
    const d3 = 3 * LF;
    line(cv.ctx, [[X(lx(d3)), Y(ymin)], [X(lx(d3)), Y(ymax)]], C.w, 1, [2, 4]);
    dot(cv.ctx, X(lx(Ls)), Y(Math.log10(kf(Ls))), C.w, 6);
    label(cv.ctx, 'K_ℓ/L_F = B/L*', X(2.4), Y(Math.log10(B / 10 ** 2.4)) - 8, C.ok);
    label(cv.ctx, 'K_m/L_F', X(0.1), Y(Math.log10(Km / LF)) - 8, C.b);
    ro([
      ['B = Et³/12(1−ν²)', `${fmt(B, 4)} N·mm`],
      ['K_ℓ = L_F B / L*', `${fmt((LF * B) / Ls, 4)} N·mm/rad`, C.ok],
      ['K_m = 0.55 B (L_F/t)^⅓', `${fmt(Km, 4)} N·mm/rad`, C.b],
      ['K_F', `${fmt(kf(Ls) * LF, 4)} N·mm/rad`, C.a],
      ['L*/L_F', fmt(Ls / LF, 3), C.w],
    ]);
  }
  draw();
}

function labBar() {
  const f = frame('bar', '바 탄소성: 로그 변형률 반환 사상', '변형률을 직접 움직이거나 반복 재생을 눌러 보세요. 항복 후 제하하면 소성 변형률이 남습니다');
  if (!f) return;
  const cv = canvas(f.view, 320);
  const ro = readout(f.ctl);
  const sE = slider(f.ctl, 'E', 500, 10000, 100, 3000, 'MPa', reset);
  const sY = slider(f.ctl, '항복 응력 σ_y', 5, 100, 1, 30, 'MPa', reset);
  const sH = slider(f.ctl, '경화 H', 0, 3000, 10, 300, 'MPa', reset);
  const sEps = slider(f.ctl, '바 신장비 λ = L/L₀', 0.95, 1.05, 0.0005, 1, '', (v) => step(v));
  const bx = el('div', 'lrow');
  f.ctl.append(bx);
  const bp = button(bx, '반복 재생', () => (play = !play));
  button(bx, '처음으로', reset);
  f.ctl.append(el('p', 'lnote', '파랑: 지금까지 지나온 τ–ε 경로, 분홍 점선: 현재 항복면 ±(σ_y + Hα). 탄성 구간 기울기는 E, 소성 구간은 EH/(E+H). 반복할수록 항복면이 넓어지는 것이 등방 경화입니다.'));
  let st, path, play = false, tt = 0;
  function reset() {
    st = { ep: 0, a: 0 };
    path = [[0, 0]];
    sEps.set(1);
    draw(0, 0);
  }
  function step(lam) {
    const E = sE.get(), Y = sY.get(), H = sH.get();
    const eps = Math.log(lam);
    let tau = E * (eps - st.ep);
    const fy = Math.abs(tau) - (Y + H * st.a);
    if (fy > 0) {
      const dg = fy / (E + H);
      st.ep += dg * Math.sign(tau);
      st.a += dg;
      tau = E * (eps - st.ep);
    }
    path.push([eps, tau]);
    if (path.length > 3000) path.shift();
    draw(eps, tau);
  }
  function draw(eps, tau) {
    const E = sE.get(), Y = sY.get(), H = sH.get();
    const { w, h } = cv.fit();
    const ym = Math.max(Y * 2.2, Y + H * st.a + 10);
    const { X, Y: YY } = plot(cv.ctx, w, h, null, { x0: -0.05, x1: 0.05, y0: -ym, y1: ym, xl: 'ε = ln λ', yl: 'τ (MPa)' });
    const yy = Y + H * st.a;
    line(cv.ctx, [[X(-0.05), YY(yy)], [X(0.05), YY(yy)]], C.b, 1, [6, 4]);
    line(cv.ctx, [[X(-0.05), YY(-yy)], [X(0.05), YY(-yy)]], C.b, 1, [6, 4]);
    line(cv.ctx, path.map(([e, t]) => [X(e), YY(t)]), C.a, 2);
    dot(cv.ctx, X(eps), YY(tau), C.w, 6);
    line(cv.ctx, [[X(st.ep), YY(0)], [X(st.ep), YY(0) + 10]], C.e, 3);
    label(cv.ctx, 'ε_p', X(st.ep), YY(0) + 24, C.e, 'center');
    ro([
      ['τ', `${fmt(tau, 3)} MPa`, C.w],
      ['ε_e', fmt(eps - st.ep, 5)],
      ['ε_p', fmt(st.ep, 5), C.e],
      ['α (누적)', fmt(st.a, 5), C.b],
      ['접선', Math.abs(tau) >= Y + H * st.a - 1e-9 && st.a > 0 ? `EH/(E+H) = ${fmt((E * H) / (E + H), 4)}` : `E = ${fmt(E, 4)}`],
    ]);
  }
  const anim = () => {
    if (play) {
      tt += 0.02;
      const amp = 0.03 + 0.012 * Math.floor(tt / (2 * Math.PI));
      const lam = Math.exp(amp * Math.sin(tt));
      sEps.set(lam);
      step(lam);
    }
    bp.classList.toggle('on', play);
    requestAnimationFrame(anim);
  };
  window.addEventListener('resize', () => draw(...(path[path.length - 1] || [0, 0])));
  reset();
  anim();
}

function labHinge() {
  const f = frame('hinge', '소성 힌지: 단면 응력과 M–θ', '회전을 움직이면 주름 단면의 응력이 탄성(삼각형) → 부분 소성 → 완전 소성(직사각형)으로 바뀌고, 힌지는 M_p 에서 일정 모멘트로 돕니다');
  if (!f) return;
  const cv = canvas(f.view, 340);
  const ro = readout(f.ctl);
  const sY = slider(f.ctl, '항복 응력 σ_y', 5, 300, 1, 30, 'MPa', reset);
  const sT = slider(f.ctl, '두께 t', 0.1, 2, 0.05, 0.5, 'mm', reset);
  const sB = slider(f.ctl, '주름선 약화 β', 0.1, 1, 0.05, 1, '', reset);
  const sK = slider(f.ctl, 'K_F / L_F', 0.05, 50, 0.05, 2, 'N·mm/rad/mm', reset, true);
  const sR = slider(f.ctl, '회전 θ − θ₀', -1.2, 1.2, 0.005, 0, 'rad', (v) => step(v));
  const bx = el('div', 'lrow');
  f.ctl.append(bx);
  const bp = button(bx, '반복 재생', () => (play = !play));
  button(bx, '처음으로', reset);
  f.ctl.append(el('p', 'lnote', '왼쪽: 주름 측면(두 패널), 가운데: 주름선 단면 응력(폭 1), 오른쪽: 길이당 M–θ. 단면 그림은 판 굽힘 이론의 탄성-완전소성 분포로, m_e = σ_y t²/6 에서 표면이 항복하고 m_p = σ_y t²/4 에서 전 단면이 항복합니다. BH-web 의 힌지는 이 m_p 를 항복 모멘트로 쓰고 그 전 구간은 K_F 로 선형입니다.'));
  let st, path, play = false, tt = 0;
  function reset() {
    st = { tp: 0 };
    path = [[0, 0]];
    sR.set(0);
    draw(0, 0);
  }
  function step(th) {
    const mp = (sB.get() * sY.get() * sT.get() ** 2) / 4, k = sK.get();
    let M = k * (th - st.tp);
    if (Math.abs(M) > mp) {
      st.tp += (Math.abs(M) - mp) / k * Math.sign(M);
      M = Math.sign(M) * mp;
    }
    path.push([th, M]);
    if (path.length > 3000) path.shift();
    draw(th, M);
  }
  function draw(th, M) {
    const sy = sY.get(), t = sT.get(), mp = (sB.get() * sy * t * t) / 4, me = (sB.get() * sy * t * t) / 6;
    const { w, h } = cv.fit();
    const ctx = cv.ctx;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const w1 = w * 0.26, w2 = w * 0.22;
    const cx = w1 * 0.5, cy = h * 0.55, L = w1 * 0.42;
      line(ctx, [[cx - L, cy], [cx, cy]], C.a, 6);
    line(ctx, [[cx, cy], [cx + L * Math.cos(th), cy - L * Math.sin(th)]], C.w, 6);
    dot(ctx, cx, cy, M === 0 ? C.text : Math.abs(M) >= mp - 1e-12 ? C.e : C.ok, 7);
    label(ctx, Math.abs(M) >= mp - 1e-12 && Math.abs(M) > 0 ? '항복 중' : '탄성', cx, cy + 34, Math.abs(M) >= mp - 1e-12 && Math.abs(M) > 0 ? C.e : C.ok, 'center');
    label(ctx, `θ_p = ${fmt(st.tp, 3)} rad`, cx, 26, C.b, 'center');
    const sx = w1 + w2 * 0.5, sh = h * 0.6, s0 = h * 0.2;
    ctx.strokeStyle = C.axis;
    ctx.strokeRect(sx - 20, s0, 40, sh);
    label(ctx, '단면 (폭 1)', sx, s0 - 10, C.mute, 'center');
    const r = Math.min(Math.abs(M) / mp, 1);
    let c;
    if (Math.abs(M) <= me) c = 1;
    else c = Math.sqrt(3 * (1 - Math.abs(M) / mp));
    const sg = Math.sign(M) || 1;
    const prof = [];
    for (let i = 0; i <= 60; i++) {
      const z = -1 + (2 * i) / 60;
      let s;
      if (Math.abs(M) <= me) s = (Math.abs(M) / me) * z;
      else s = Math.abs(z) <= c ? z / c : Math.sign(z);
      prof.push([sx + 20 + sg * s * w2 * 0.33, s0 + sh * (1 - (z + 1) / 2)]);
    }
    ctx.fillStyle = 'rgba(255,95,168,.35)';
    ctx.beginPath();
    ctx.moveTo(sx + 20, s0);
    prof.slice().reverse().forEach(([x, y]) => ctx.lineTo(x, y));
    ctx.lineTo(sx + 20, s0 + sh);
    ctx.closePath();
    ctx.fill();
    line(ctx, prof, C.b, 2);
    line(ctx, [[sx + 20 + w2 * 0.33, s0], [sx + 20 + w2 * 0.33, s0 + sh]], C.e, 1, [3, 3]);
    line(ctx, [[sx + 20 - w2 * 0.33, s0], [sx + 20 - w2 * 0.33, s0 + sh]], C.e, 1, [3, 3]);
    label(ctx, '+σ_y', sx + 20 + w2 * 0.33, s0 + sh + 16, C.e, 'center');
    label(ctx, '−σ_y', sx + 20 - w2 * 0.33, s0 + sh + 16, C.e, 'center');
    label(ctx, Math.abs(M) <= me ? '탄성 (삼각형)' : Math.abs(M) < mp ? '부분 소성' : '완전 소성 (직사각형)', sx, s0 + sh + 34, C.text, 'center');
    ctx.save();
    ctx.translate(w1 + w2, 0);
    const pw = w - w1 - w2;
    const yM = mp * 1.6;
    const { X, Y } = plot(ctx, pw, h, null, { x0: -1.2, x1: 1.2, y0: -yM, y1: yM, xl: 'θ − θ₀ (rad)', yl: 'M/L (N·mm/mm)' });
    line(ctx, [[X(-1.2), Y(mp)], [X(1.2), Y(mp)]], C.e, 1, [6, 4]);
    line(ctx, [[X(-1.2), Y(-mp)], [X(1.2), Y(-mp)]], C.e, 1, [6, 4]);
    line(ctx, [[X(-1.2), Y(me)], [X(1.2), Y(me)]], C.w, 1, [2, 4]);
    label(ctx, 'm_p', X(1.15), Y(mp) - 5, C.e, 'right');
    label(ctx, 'm_e (탄성 한계, 참고)', X(-1.15), Y(me) - 5, C.w, 'left');
    line(ctx, path.map(([a, b]) => [X(a), Y(b)]), C.a, 2);
    dot(ctx, X(th), Y(M), C.w, 6);
    ctx.restore();
    ro([
      ['m_p = βσ_y t²/4', `${fmt(mp, 4)} N·mm/mm`, C.e],
      ['M/L', `${fmt(M, 4)} N·mm/mm`, C.w],
      ['M / m_p', fmt(r, 3)],
      ['소성 회전 θ_p', `${fmt(st.tp, 4)} rad`, C.b],
      ['탄성 한계 회전 m_p/K', `${fmt(mp / sK.get(), 4)} rad`],
    ]);
  }
  const anim = () => {
    if (play) {
      tt += 0.02;
      const th = 1.0 * Math.sin(tt);
      sR.set(th);
      step(th);
    }
    bp.classList.toggle('on', play);
    requestAnimationFrame(anim);
  };
  window.addEventListener('resize', () => draw(...(path[path.length - 1] || [0, 0])));
  reset();
  anim();
}

function barrier(s, sh) {
  if (s >= sh || s <= 0) return [s <= 0 ? Infinity : 0, 0];
  const r = s - sh;
  return [-(r * r) * Math.log(s / sh), -2 * r * Math.log(s / sh) - (r * r) / s];
}

function labBarrier() {
  const f = frame('barrier', 'IPC 장벽: 거리 → 에너지 → 접촉력', '두 패널 사이 거리를 움직여 보세요. 장벽 범위 밖에서는 힘이 정확히 0 이고, 최소 간격에 다가갈수록 무한대로 커집니다');
  if (!f) return;
  const cv = canvas(f.view, 320);
  const ro = readout(f.ctl);
  const sD = slider(f.ctl, '면 사이 거리 d', 0.0102, 0.04, 0.0001, 0.025, 'mm', draw);
  const sH = slider(f.ctl, '장벽 범위 d̂', 0.002, 0.02, 0.0005, 0.01, 'mm', draw);
  const sK = slider(f.ctl, 'κ (크기 조정)', 0.1, 100, 0.1, 10, '', draw, true);
  f.ctl.append(el('p', 'lnote', 'd_min = 0.01 mm 로 두었습니다(BH-web 기본은 평균 바 길이의 0.1 %, d̂ = d_min). 위: 측면에서 본 두 패널과 장벽 힘(화살표), 아래: 장벽 에너지 κb(d) 와 힘 −κb′(d). κ 를 바꿔도 "뚫리지 않는다" 는 결론은 같고, 힘이 커지기 시작하는 거리만 조금 바뀝니다.'));
  window.addEventListener('resize', draw);
  function draw() {
    const d = sD.get(), sh = sH.get(), k = sK.get(), dmin = 0.01;
    const { w, h } = cv.fit();
    const ctx = cv.ctx;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const top = h * 0.36;
    const gx = (v) => 60 + (v / 0.04) * (w - 90);
    const yb = top * 0.75;
    line(ctx, [[40, yb], [w - 30, yb]], C.a, 8);
    const yt = yb - 18 - ((d - dmin) / 0.03) * (top * 0.5);
    line(ctx, [[40, yt], [w - 30, yt]], C.w, 8);
    const [b, b1] = barrier(d - dmin, sh);
    const F = -k * b1;
    const ar = Math.min(60, Math.log10(1 + F * 1e3) * 15);
    if (F > 0) {
      ctx.strokeStyle = C.b; ctx.lineWidth = 3;
      for (const xx of [w * 0.3, w * 0.5, w * 0.7]) {
        line(ctx, [[xx, yt - 6], [xx, yt - 6 - ar]], C.b, 3);
        line(ctx, [[xx - 6, yt - ar], [xx, yt - 8 - ar], [xx + 6, yt - ar]], C.b, 3);
      }
    }
    label(ctx, `d = ${fmt(d, 4)} mm`, w - 34, (yb + yt) / 2 + 4, C.text, 'right');
    label(ctx, F > 0 ? '장벽 활성: 밀어냄' : '장벽 밖: 힘 0', 44, 22, F > 0 ? C.b : C.ok);
    ctx.save();
    ctx.translate(0, top);
    const ph = h - top;
    const xs = Array.from({ length: 300 }, (_, i) => dmin + 1e-5 + (i / 299) * 0.03);
    const E = xs.map((x) => k * barrier(x - dmin, sh)[0]);
    const Fs = xs.map((x) => -k * barrier(x - dmin, sh)[1]);
    const ym = Math.max(k * barrier(sh * 0.05, sh)[0], 1e-9);
    const { X, Y } = plot(ctx, w, ph, null, { x0: dmin, x1: dmin + 0.03, y0: 0, y1: ym, xl: '거리 d (mm)', yl: 'κ b(d)' });
    line(ctx, xs.map((x, i) => [X(x), Y(Math.min(E[i], ym * 1.1))]), C.a, 2.5);
    const fm = Math.max(-k * barrier(sh * 0.05, sh)[1], 1e-9);
    line(ctx, xs.map((x, i) => [X(x), Y((Math.min(Fs[i], fm * 1.1) / fm) * ym)]), C.b, 2, [6, 4]);
    line(ctx, [[X(dmin + sh), Y(0)], [X(dmin + sh), Y(ym)]], C.w, 1, [2, 4]);
    label(ctx, 'd_min + d̂', X(dmin + sh) + 4, Y(ym) + 14, C.w);
    dot(ctx, X(d), Y(Math.min(k * b, ym)), C.w, 6);
    label(ctx, '에너지', X(dmin + 0.027), Y(ym * 0.85), C.a, 'right');
    label(ctx, '접촉력 (상대 크기)', X(dmin + 0.027), Y(ym * 0.72), C.b, 'right');
    ctx.restore();
    ro([
      ['κ b(d)', fmt(k * b, 4), C.a],
      ['접촉력 −κ b′(d)', fmt(F, 4), C.b],
      ['장벽 활성', d - dmin < sh ? '예' : '아니오', d - dmin < sh ? C.b : C.ok],
    ]);
  }
  draw();
}

function labCCD() {
  const f = frame('ccd', '연속 충돌 검사(CCD) 와 힌지각 장벽', '뉴턴 한 걸음이 너무 크면 장벽을 건너뛰어 버립니다. CCD 는 처음 부딪히는 시점의 90 % 까지만 가게 걸음을 자릅니다');
  if (!f) return;
  const cv = canvas(f.view, 380);
  const ro = readout(f.ctl);
  let ccd = true, mode = 'face';
  const bx = el('div', 'lrow');
  f.ctl.append(bx);
  const b1 = button(bx, 'CCD 켬', () => { ccd = !ccd; b1.textContent = ccd ? 'CCD 켬' : 'CCD 끔'; run(); }, 'on');
  const b2 = button(bx, '떨어진 면', () => { mode = 'face'; run(); }, 'on');
  const b3 = button(bx, '이웃 패널(힌지)', () => { mode = 'hinge'; run(); });
  const sS = slider(f.ctl, '뉴턴 걸음 크기', 0.2, 3, 0.05, 1.6, '× 남은 거리', run);
  button(f.ctl, '다시 재생', run);
  f.ctl.append(el('p', 'lnote', '떨어진 면: 노란 패널 꼭짓점이 파란 패널 쪽으로 한 걸음에 이동합니다. 이웃 패널: 주름을 사이에 둔 두 패널이 닫히는 방향으로 한 걸음 접힙니다. 이웃 패널은 꼭짓점을 공유해 거리 계산에서 빠지므로, 날개 높이 h 와 접힘각 θ 로 거리 h·θ 를 대신 쓰고 각도를 이분법으로 잘라 냅니다. CCD 를 끄면 걸음이 클 때 반대편으로 뚫고 나갑니다.'));
  let t0 = 0, raf;
  window.addEventListener('resize', () => draw(1));
  function run() {
    b1.classList.toggle('on', ccd);
    b2.classList.toggle('on', mode === 'face');
    b3.classList.toggle('on', mode === 'hinge');
    t0 = performance.now();
    cancelAnimationFrame(raf);
    const tick = () => {
      const s = Math.min(1, (performance.now() - t0) / 1400);
      draw(s);
      if (s < 1) raf = requestAnimationFrame(tick);
    };
    tick();
  }
  function draw(s) {
    const { w, h } = cv.fit();
    const ctx = cv.ctx;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const step = sS.get();
    if (mode === 'face') {
      const ys = 70, gap = (h - 100) / 3, y0 = ys + gap;
      line(ctx, [[w * 0.15, y0], [w * 0.85, y0]], C.a, 8);
      const full = gap * step;
      const tHit = Math.min(1, gap / full);
      const a = ccd ? Math.min(1, 0.9 * tHit) : 1;
      const yy = ys + full * a * s;
      line(ctx, [[w * 0.3, ys], [w * 0.3, ys + full]], '#3a4554', 1, [4, 4]);
      dot(ctx, w * 0.3, ys + full, C.mute, 4);
      label(ctx, '뉴턴 예측 위치', w * 0.3 + 8, ys + full + 4, C.mute);
      const pen = yy > y0;
      ctx.fillStyle = pen ? 'rgba(255,107,107,.65)' : 'rgba(255,200,87,.8)';
      ctx.beginPath();
      ctx.moveTo(w * 0.3, yy);
      ctx.lineTo(w * 0.62, yy - 30);
      ctx.lineTo(w * 0.5, yy - 60);
      ctx.closePath();
      ctx.fill();
      dot(ctx, w * 0.3, yy, pen ? C.e : C.w, 6);
      label(ctx, pen ? '관통!' : s >= 1 ? '정지 (관통 없음)' : '이동 중', w * 0.3 - 12, yy, pen ? C.e : C.ok, 'right', 14);
      ro([
        ['처음 닿는 시점 t', fmt(tHit, 3)],
        ['실제 걸음 a', fmt(a, 3), ccd ? C.ok : C.e],
        ['남은 거리', `${fmt(Math.max(0, y0 - yy), 1)} px`, pen ? C.e : C.text],
      ]);
    } else {
      const cx = w * 0.3, cy = h * 0.75, L = Math.min(w * 0.5, h * 0.6);
      const th0 = 0.9;
      const full = th0 * step;
      const tHit = Math.min(1, th0 / full);
      const a = ccd ? Math.min(1, 0.9 * tHit) : 1;
      const th = th0 - full * a * s;
      line(ctx, [[cx, cy], [cx + L, cy]], C.a, 8);
      line(ctx, [[cx, cy], [cx + L * Math.cos(th), cy - L * Math.sin(th)]], th < 0 ? C.e : C.w, 8);
      dot(ctx, cx, cy, C.text, 6);
      ctx.strokeStyle = C.b; ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(cx, cy, 60, -Math.max(th, 0), 0); ctx.stroke();
      label(ctx, `θ = ${fmt((th * 180) / Math.PI, 1)}°`, cx + 70, cy - 10, C.b);
      line(ctx, [[cx + L * Math.cos(Math.max(th, 0)), cy - L * Math.sin(Math.max(th, 0))], [cx + L * Math.cos(Math.max(th, 0)), cy]], C.ok, 1.5, [4, 4]);
      label(ctx, 'h·θ', cx + L * Math.cos(Math.max(th, 0)) + 6, cy - (L * Math.sin(Math.max(th, 0))) / 2, C.ok);
      label(ctx, th < 0 ? '패널이 서로 통과!' : s >= 1 ? '정지 (접힘각 > 0)' : '접히는 중', cx, 30, th < 0 ? C.e : C.ok, 'left', 14);
      ro([
        ['처음 닿는 시점 t', fmt(tHit, 3)],
        ['실제 걸음 a', fmt(a, 3), ccd ? C.ok : C.e],
        ['남은 각 θ', `${fmt((th * 180) / Math.PI, 2)}°`, th < 0 ? C.e : C.text],
      ]);
    }
  }
  run();
}

function labKernel() {
  const f = frame('kernel', '에너지 컨투어: 요소 에너지를 요소 주위로 퍼뜨리기', '바나 주름을 클릭해 에너지를 주고, 퍼짐 폭 σ 를 바꿔 보세요. 컨투어를 면적분하면 넣은 에너지 합과 같습니다');
  if (!f) return;
  const cv = canvas(f.view, 340);
  const ro = readout(f.ctl);
  const sS = slider(f.ctl, 'σ / √(패널 면적)', 0.05, 0.6, 0.01, 0.2, '', draw);
  button(f.ctl, '모두 지우기', () => { E.fill(0); draw(); });
  f.ctl.append(el('p', 'lnote', '정사각 패널 두 장(N5B8)과 가운데 접힘 주름입니다. 클릭할 때마다 가장 가까운 요소 에너지가 1 씩 늘어납니다. BH-web 은 σ = 0.2√A 를 씁니다. 패널 경계를 넘어가지 않고, 두 패널이 공유하는 요소(공유 변 바, 접힘 주름)는 양쪽 패널에 걸쳐 퍼집니다.'));
  const P = [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0.5, 0.5], [1.5, 0.5]];
  const els = [[0, 1, 'bar', [0]], [3, 4, 'bar', [0]], [0, 3, 'bar', [0]], [1, 2, 'bar', [1]], [4, 5, 'bar', [1]], [2, 5, 'bar', [1]], [0, 6, 'bar', [0]], [1, 6, 'bar', [0]], [3, 6, 'bar', [0]], [4, 6, 'bar', [0]], [1, 7, 'bar', [1]], [2, 7, 'bar', [1]], [4, 7, 'bar', [1]], [5, 7, 'bar', [1]], [1, 4, 'fold', [0, 1]]];
  const E = new Array(els.length).fill(0);
  E[14] = 3;
  E[6] = 1;
  let geo;
  cv.c.addEventListener('click', (ev) => {
    if (!geo) return;
    const r = cv.c.getBoundingClientRect();
    const x = (ev.clientX - r.left - geo.ox) / geo.sc, y = (geo.oy - (ev.clientY - r.top)) / geo.sc;
    let best = -1, bd = 1e9;
    els.forEach(([a, b], i) => {
      const d = segd([x, y], P[a], P[b]);
      if (d < bd) { bd = d; best = i; }
    });
    if (best >= 0 && bd < 0.15) { E[best] += 1; draw(); }
  });
  function segd(p, a, b) {
    const dx = b[0] - a[0], dy = b[1] - a[1];
    const t = Math.max(0, Math.min(1, ((p[0] - a[0]) * dx + (p[1] - a[1]) * dy) / (dx * dx + dy * dy)));
    return Math.hypot(a[0] + t * dx - p[0], a[1] + t * dy - p[1]);
  }
  window.addEventListener('resize', draw);
  function draw() {
    const { w, h } = cv.fit();
    const ctx = cv.ctx;
    ctx.fillStyle = C.bg;
    ctx.fillRect(0, 0, w, h);
    const sc = Math.min((w - 160) / 2, h - 40), ox = 30, oy = h - 20;
    geo = { sc, ox, oy };
    const n = 80;
    const sg = sS.get();
    const cell = 1 / n;
    const field = [];
    let tot = 0;
    const kern = els.map(([a, b, , pan]) => {
      const vals = [];
      let s = 0;
      for (const p of pan) for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
        const x = p + (i + 0.5) * cell, y = (j + 0.5) * cell;
        const v = Math.exp(-((segd([x, y], P[a], P[b]) / sg) ** 2));
        vals.push([p * n + i, j, v]);
        s += v * cell * cell;
      }
      return vals.map(([i, j, v]) => [i, j, v / s]);
    });
    const F = Array.from({ length: 2 * n }, () => new Array(n).fill(0));
    kern.forEach((ks, e) => { if (E[e]) for (const [i, j, v] of ks) F[i][j] += E[e] * v; });
    let mx = 0;
    for (const r of F) for (const v of r) { mx = Math.max(mx, v); tot += v * cell * cell; }
    const cmap = (t) => {
      const st = [[13, 8, 135], [126, 3, 168], [204, 71, 120], [248, 149, 64], [240, 249, 33]];
      const x = Math.min(0.9999, Math.max(0, t)) * (st.length - 1);
      const k = Math.floor(x), u = x - k;
      return st[k].map((c, q) => Math.round(c + (st[k + 1][q] - c) * u));
    };
    for (let i = 0; i < 2 * n; i++) for (let j = 0; j < n; j++) {
      const [r, g, b] = cmap(mx ? F[i][j] / mx : 0);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(ox + i * cell * sc, oy - (j + 1) * cell * sc, cell * sc + 1, cell * sc + 1);
    }
    els.forEach(([a, b, k], i) => line(ctx, [[ox + P[a][0] * sc, oy - P[a][1] * sc], [ox + P[b][0] * sc, oy - P[b][1] * sc]], k === 'fold' ? C.e : 'rgba(255,255,255,.55)', k === 'fold' ? 3 : 1 + Math.min(E[i], 4)));
    const lx = ox + 2 * sc + 30;
    for (let q = 0; q <= 50; q++) {
      const [r, g, b] = cmap(q / 50);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(lx, oy - 30 - q * ((h - 80) / 50), 18, (h - 80) / 50 + 1);
    }
    label(ctx, fmt(mx, 3), lx + 24, 44, C.text);
    label(ctx, '0', lx + 24, oy - 26, C.text);
    label(ctx, '에너지 밀도', lx - 4, 26, C.mute);
    ro([
      ['넣은 에너지 합', fmt(E.reduce((a, b) => a + b, 0), 3)],
      ['컨투어 면적분', fmt(tot, 3), C.ok],
      ['σ', `${fmt(sg, 3)} × √A`],
    ]);
  }
  draw();
}

function labMiura() {
  const f = frame('miura', '미우라 강체 접힘: 패널은 그대로, 주름만 돈다', '접힘 정도를 움직이면 3×3 미우라가 강체로 접힙니다. 곧은 주름과 지그재그 주름 각의 관계가 cos γ 로 고정됩니다');
  if (!f) return;
  const r = new THREE.WebGLRenderer({ antialias: true });
  r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  r.domElement.className = 'l3';
  const two = el('div', 'ltwo');
  f.view.append(two);
  two.append(r.domElement);
  const cv = canvas(two, 300);
  const scene = new THREE.Scene();
  scene.background = new THREE.Color(C.bg);
  const cam = new THREE.PerspectiveCamera(35, 1.4, 0.1, 1000);
  cam.up.set(0, 0, 1);
  cam.position.set(40, -120, 110);
  const oc = new OrbitControls(cam, r.domElement);
  oc.target.set(40, 60, 10);
  scene.add(new THREE.AmbientLight(0xffffff, 0.5));
  const dl = new THREE.DirectionalLight(0xffffff, 1.3);
  dl.position.set(50, -80, 150);
  scene.add(dl);
  const geo = new THREE.BufferGeometry();
  const mesh = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0x9fc3e6, side: THREE.DoubleSide, flatShading: true }));
  scene.add(mesh);
  const lines = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true }));
  scene.add(lines);
  const ro = readout(f.ctl);
  const sT = slider(f.ctl, '접힘 정도 θ (Schenk–Guest)', 1, 89, 0.5, 45, '°', draw);
  const sG = slider(f.ctl, '섹터 각 γ', 30, 80, 1, 60, '°', draw);
  f.ctl.append(el('p', 'lnote', '빨강: 곧은 주름(펼친 전개도에서 일직선), 파랑: 지그재그 주름. 오른쪽 그래프는 θ 를 0 에서 90° 까지 움직일 때 두 주름의 접힘각 ρ 와, tan(ρ곧은/2) / tan(ρ지그재그/2) 비(초록)입니다. 이 비는 접힘 정도와 상관없이 cos γ 로 일정합니다. BH-web 기구 해석은 이 관계를 1e-8 이하 오차로 따라갑니다.'));
  const nx = 3, ny = 3, a = 40, b = 40;
  function nodes(th, g) {
    const H = a * Math.sin(th) * Math.sin(g);
    const q = Math.sqrt(1 + Math.cos(th) ** 2 * Math.tan(g) ** 2);
    const S = (b * Math.cos(th) * Math.tan(g)) / q;
    const Lp = a * Math.sqrt(1 - Math.sin(th) ** 2 * Math.sin(g) ** 2);
    const V = b / q;
    const X = [];
    for (let j = 0; j <= ny; j++) for (let i = 0; i <= nx; i++) X.push([i * S, j * Lp + (i % 2 ? V : 0), j % 2 ? H : 0]);
    return X;
  }
  const id = (i, j) => j * (nx + 1) + i;
  function draw() {
    const th = (sT.get() * Math.PI) / 180, g = (sG.get() * Math.PI) / 180;
    const X = nodes(th, g);
    const pos = [];
    for (let j = 0; j < ny; j++) for (let i = 0; i < nx; i++) {
      const A = X[id(i, j)], B = X[id(i + 1, j)], Cc = X[id(i + 1, j + 1)], D = X[id(i, j + 1)];
      pos.push(...A, ...B, ...Cc, ...A, ...Cc, ...D);
    }
    geo.setAttribute('position', new THREE.Float32BufferAttribute(pos, 3));
    geo.computeVertexNormals();
    const lp = [], lc = [];
    const push = (p, q, c) => { lp.push(...p, ...q); const cc = new THREE.Color(c); lc.push(cc.r, cc.g, cc.b, cc.r, cc.g, cc.b); };
    for (let j = 1; j < ny; j++) for (let i = 0; i < nx; i++) push(X[id(i, j)], X[id(i + 1, j)], 0x4cc2ff);
    for (let i = 1; i < nx; i++) for (let j = 0; j < ny; j++) push(X[id(i, j)], X[id(i, j + 1)], 0xff6b6b);
    lines.geometry.setAttribute('position', new THREE.Float32BufferAttribute(lp, 3));
    lines.geometry.setAttribute('color', new THREE.Float32BufferAttribute(lc, 3));
    const ang = (X) => {
      const fa = (p, q, wi, wl) => Math.abs(Math.PI - dihedral(X[wi], X[p], X[q], X[wl]).th);
      return { str: fa(id(1, 0), id(1, 1), id(0, 0), id(2, 0)), zig: fa(id(0, 1), id(1, 1), id(0, 0), id(0, 2)) };
    };
    const curve = [];
    for (let k = 1; k <= 89; k += 1) {
      const A = ang(nodes((k * Math.PI) / 180, g));
      curve.push([k, A.str, A.zig, Math.tan(A.str / 2) / Math.tan(A.zig / 2)]);
    }
    const cur = ang(X);
    const { w, h } = cv.fit();
    const { X: PX, Y: PY } = plot(cv.ctx, w, h, null, { x0: 0, x1: 90, y0: 0, y1: 180, xl: 'θ (°)', yl: '접힘각 ρ (°)' });
    line(cv.ctx, curve.map((c) => [PX(c[0]), PY((c[1] * 180) / Math.PI)]), C.e, 2.5);
    line(cv.ctx, curve.map((c) => [PX(c[0]), PY((c[2] * 180) / Math.PI)]), C.a, 2.5);
    line(cv.ctx, curve.map((c) => [PX(c[0]), PY(c[3] * 180)]), C.ok, 2, [5, 4]);
    line(cv.ctx, [[PX(0), PY(Math.cos(g) * 180)], [PX(90), PY(Math.cos(g) * 180)]], C.w, 1, [2, 4]);
    dot(cv.ctx, PX(sT.get()), PY((cur.str * 180) / Math.PI), C.e, 5);
    dot(cv.ctx, PX(sT.get()), PY((cur.zig * 180) / Math.PI), C.a, 5);
    label(cv.ctx, '초록: tan 비 × 180,  노랑: cos γ × 180', PX(4), PY(Math.cos(g) * 180) - 8, C.ok);
    const ratio = Math.tan(cur.str / 2) / Math.tan(cur.zig / 2);
    ro([
      ['ρ 곧은', `${fmt((cur.str * 180) / Math.PI, 2)}°`, C.e],
      ['ρ 지그재그', `${fmt((cur.zig * 180) / Math.PI, 2)}°`, C.a],
      ['tan 비', fmt(ratio, 6), C.ok],
      ['cos γ', fmt(Math.cos(g), 6), C.w],
    ]);
  }
  const loop = () => {
    const rr = r.domElement.getBoundingClientRect();
    if (rr.width > 0 && r.domElement.width !== Math.round(rr.width * r.getPixelRatio())) {
      r.setSize(rr.width, rr.height, false);
      cam.aspect = rr.width / rr.height;
      cam.updateProjectionMatrix();
    }
    oc.update();
    r.render(scene, cam);
    requestAnimationFrame(loop);
  };
  window.addEventListener('resize', draw);
  draw();
  loop();
}

for (const fn of [labDihedral, labN5B8, labFold, labBar, labHinge, labBarrier, labCCD, labKernel, labMiura]) {
  try {
    fn();
  } catch (e) {
    console.error(fn.name, e);
  }
}
