import numpy as np
from .select import resolve

TET_FACES = ((0, 1, 2), (0, 3, 1), (1, 3, 2), (2, 3, 0))


def contacts_of(spec, with_self=False):
    return [c for c in spec.get("contacts") or [] if with_self or c["type"] != "self"]


def unit(v):
    v = np.array(v, dtype=float)
    n = np.linalg.norm(v)
    if n <= 0:
        raise ValueError("방향 벡터의 크기가 0 입니다")
    return v / n


def shell_offset(model, spec):
    if not model.shells:
        return 0.0
    from .spec import section_for

    return 0.5 * max(float(section_for(spec, g)["thickness"]) for g in set(model.shell_group))


def place(model, spec, item):
    d = unit(item["motion"]) if np.linalg.norm(item.get("motion", [0, 0, 0])) > 0 else np.array([0.0, 0.0, -1.0])
    lo, hi = model.bbox()
    c = 0.5 * (lo + hi)
    proj = (model.nodes - c) @ d
    reach = float(-proj.min()) + shell_offset(model, spec)
    if item["shape"] == "sphere":
        if item.get("center") is None:
            base = c if item.get("at") is None else np.array(item["at"], dtype=float)
            base = base + d * ((c - base) @ d)
            lat = model.nodes - base
            lat = lat - np.outer(lat @ d, d)
            col = np.linalg.norm(lat, axis=1) <= max(0.25 * float(item["radius"]), 0.02 * model.diag())
            if col.any():
                reach = float(-proj[col].min()) + shell_offset(model, spec)
            item["center"] = [round(float(x), 6) for x in base - d * (reach + float(item["radius"]))]
    else:
        if item.get("normal") is None:
            item["normal"] = [round(float(x), 6) for x in d]
        if item.get("point") is None:
            n = unit(item["normal"])
            pr = (model.nodes - c) @ n
            item["point"] = [round(float(x), 6) for x in c + n * (float(pr.min()) - shell_offset(model, spec))]
    item.pop("at", None)
    return item


def bodies(model):
    t = model.solids
    if len(t) == 0:
        return []
    parent = np.arange(len(model.nodes))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for a, b, c, d in t.tolist():
        ra = find(a)
        for x in (b, c, d):
            rx = find(x)
            if rx != ra:
                parent[rx] = ra
    roots = {}
    for i in np.unique(t):
        roots.setdefault(find(int(i)), []).append(int(i))
    out = sorted(roots.values(), key=lambda ids: tuple(model.nodes[ids].mean(axis=0).round(6)[::-1]))
    return out


def tag_bodies(model):
    bs = bodies(model)
    if len(bs) > 1:
        for k, ids in enumerate(bs):
            model.patches[f"body_{k + 1}"] = {"nodes": sorted(ids), "area": None, "normal": None}
    return model


def solid_faces(model):
    t = model.solids
    if len(t) == 0:
        return []
    faces = np.vstack([t[:, list(f)] for f in TET_FACES])
    owner = np.tile(np.arange(len(t)), 4)
    local = np.repeat(np.arange(4), len(t))
    key = np.sort(faces, axis=1)
    _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
    keep = idx[cnt == 1]
    return [(int(owner[i]), int(local[i]), [int(x) for x in faces[i]]) for i in keep]


def surface(model, sel=None, own=None):
    ids = None if not sel or sel.get("all") else set(resolve(model, sel).tolist())
    out = []
    if own is None:
        own = solid_faces(model)
    for e, lf, f in own:
        if ids is None or all(i in ids for i in f):
            out.append(("solid", e, lf, f))
    for fi, s in enumerate(model.shells):
        if ids is None or all(i in ids for i in s):
            out.append(("shell", fi, None, list(s)))
    return out


def face_geo(nodes, f):
    p = nodes[f]
    c = np.cross(p[1] - p[0], p[2] - p[0]) if len(f) == 3 else np.cross(p[2] - p[0], p[3] - p[1])
    n = np.linalg.norm(c)
    return p.mean(axis=0), (c / n if n > 0 else np.zeros(3))


def icosphere(level=3):
    t = (1.0 + 5.0**0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t), (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    v = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2), (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11), (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    for _ in range(level):
        cache = {}

        def mid(a, b):
            k = (min(a, b), max(a, b))
            if k not in cache:
                p = v[a] + v[b]
                v.append(p / np.linalg.norm(p))
                cache[k] = len(v) - 1
            return cache[k]

        nf = []
        for a, b, c in f:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        f = nf
    return np.array(v), np.array(f, dtype=int)


def plane_basis(n):
    n = unit(n)
    a = np.array([1.0, 0.0, 0.0]) if abs(n[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
    t1 = unit(np.cross(n, a))
    t2 = np.cross(n, t1)
    return t1, t2, n


def view(model, spec):
    out = []
    for c in contacts_of(spec, True):
        if c["type"] == "self":
            out.append({"id": c["id"], "type": "self", "friction": c.get("friction", 0.0)})
            continue
        if c["type"] != "rigid":
            item = {"id": c["id"], "type": "pair", "friction": c.get("friction", 0.0)}
            try:
                item["a"] = resolve(model, c["a"]).tolist()
                item["b"] = resolve(model, c["b"]).tolist()
            except Exception as e:
                item["error"] = str(e)
            out.append(item)
            continue
        item = {"id": c["id"], "type": "rigid", "shape": c["shape"], "motion": c.get("motion", [0, 0, 0]), "friction": c.get("friction", 0.0)}
        if c["shape"] == "sphere":
            item["center"] = c["center"]
            item["radius"] = c["radius"]
        else:
            item["point"] = c["point"]
            item["normal"] = c["normal"]
            item["size"] = 0.75 * model.diag()
        out.append(item)
    return out


def describe(c):
    mu = f", 마찰 {c.get('friction', 0.0):g}"
    if c["type"] == "rigid":
        mv = ", 이동 [" + ", ".join(f"{v:g}" for v in c.get("motion", [0, 0, 0])) + "] mm"
        if c["shape"] == "sphere":
            return f"강체 구 R={c['radius']:g} 중심 (" + ", ".join(f"{v:g}" for v in c["center"]) + ")" + mv + mu
        return "강체 평면 점 (" + ", ".join(f"{v:g}" for v in c["point"]) + ") 법선 (" + ", ".join(f"{v:g}" for v in c["normal"]) + ")" + mv + mu
    if c["type"] == "self":
        return "자기 접촉 (모든 면끼리)" + mu
    from .select import describe as dsel

    return f"접촉 쌍 {dsel(c['a'])} ↔ {dsel(c['b'])}" + mu


def default_stiffness(spec):
    return 10.0 * float(spec["materials"]["default"]["E"])


def ccx_build(model, spec, nodes, eid, shell_eids, solid_eids, static):
    cs = contacts_of(spec)
    out = {"elements": [], "model": [], "step": [], "print": [], "hide": [], "refs": {}, "notes": [], "eid": eid, "pairs": {}}
    if not cs:
        return None
    if not static:
        out["notes"].append("접촉은 정적 스텝에서만 적용됩니다 (좌굴·모달에서는 무시)")
        return out
    own = solid_faces(model)
    P = np.array(nodes[: len(model.nodes)])
    E = model.nodes
    span = model.diag()

    def label(kind, e, lf, f, toward=None, against=None):
        if kind == "solid":
            return f"{solid_eids[e]}, S{lf + 1}"
        cen, n = face_geo(E, f)
        s = float((toward - cen) @ n) if toward is not None else -float(against @ n)
        return f"{shell_eids[e]}, {'SPOS' if s >= 0 else 'SNEG'}"

    for k, c in enumerate(cs):
        tag = f"C{k + 1}"
        K = float(c.get("stiffness") or default_stiffness(spec))
        mu = float(c.get("friction") or 0.0)
        if c["type"] == "rigid":
            tgt = surface(model, c.get("target"), own)
            if not tgt:
                raise ValueError(f"접촉 {c['id']} 의 대상 면이 없습니다")
            n0 = len(nodes)
            if c["shape"] == "sphere":
                cen = np.array(c["center"], dtype=float)
                R = float(c["radius"])
                v, f = icosphere(3)
                nv = len(v)
                em = {}
                vm = []
                for a, b, d in f:
                    for i, j in ((a, b), (b, d), (d, a)):
                        key = (min(i, j), max(i, j))
                        if key not in em:
                            q = v[i] + v[j]
                            vm.append(q / np.linalg.norm(q))
                            em[key] = len(vm) - 1
                vm = np.array(vm)
                nm = len(vm)
                for sc in (0.8, 1.0):
                    for p in v:
                        nodes.append(cen + sc * R * p)
                    for p in vm:
                        nodes.append(cen + sc * R * p)
                for p in v:
                    nodes.append(cen + 0.9 * R * p)
                lay = nv + nm
                rows = []
                faces = []
                for a, b, d in f:
                    out["eid"] += 1
                    e3 = [nv + em[(min(i, j), max(i, j))] for i, j in ((a, b), (b, d), (d, a))]
                    ids = [a, b, d, a + lay, b + lay, d + lay] + e3 + [x + lay for x in e3] + [2 * lay + a, 2 * lay + b, 2 * lay + d]
                    rows.append(f"{out['eid']}, " + ", ".join(str(n0 + i + 1) for i in ids[:10]))
                    rows[-1] += ",\n" + ", ".join(str(n0 + i + 1) for i in ids[10:])
                    faces.append(f"{out['eid']}, S2")
                    out["hide"].append(out["eid"])
                out["elements"] += [f"*ELEMENT, TYPE=C3D15, ELSET=ERIG{tag}"] + rows
                dep = [label(kd, e, lf, f2, toward=cen) for kd, e, lf, f2 in tgt]
                refp = cen
            else:
                p0 = np.array(c["point"], dtype=float)
                t1, t2, n = plane_basis(c["normal"])
                h = 0.75 * span
                p0 = p0 + t1 * ((0.5 * (E.min(axis=0) + E.max(axis=0)) - p0) @ t1) + t2 * ((0.5 * (E.min(axis=0) + E.max(axis=0)) - p0) @ t2)
                for z in (-0.1 * h, 0.0):
                    for sx, sy in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
                        nodes.append(p0 + n * z + t1 * sx * h + t2 * sy * h)
                out["eid"] += 1
                out["elements"] += [f"*ELEMENT, TYPE=C3D8, ELSET=ERIG{tag}", f"{out['eid']}, " + ", ".join(str(n0 + i + 1) for i in range(8))]
                faces = [f"{out['eid']}, S2"]
                out["hide"].append(out["eid"])
                dep = [label(kd, e, lf, f2, against=n) for kd, e, lf, f2 in tgt]
                refp = p0
            n1 = len(nodes)
            nodes.append(np.array(refp, dtype=float))
            nodes.append(np.array(refp, dtype=float))
            ref, rot = n1, n1 + 1
            out["refs"][c["id"]] = ref
            M = out["model"]
            M += [f"*MATERIAL, NAME=MRIG{tag}", "*ELASTIC", f"{float(spec['materials']['default']['E']):.9g}, 0.3", "*DENSITY", "1e-20", f"*SOLID SECTION, ELSET=ERIG{tag}, MATERIAL=MRIG{tag}"]
            M += [f"*NSET, NSET=NRIG{tag}, GENERATE", f"{n0 + 1}, {n1}, 1", f"*NSET, NSET=NREF{tag}", f"{ref + 1}"]
            M += [f"*RIGID BODY, NSET=NRIG{tag}, REF NODE={ref + 1}, ROT NODE={rot + 1}", "*BOUNDARY", f"{ref + 1}, 1, 3, 0.", f"{rot + 1}, 1, 3, 0."]
            mv = c.get("motion", [0, 0, 0])
            rows = [f"{ref + 1}, {d + 1}, {d + 1}, {float(mv[d]):.9g}" for d in range(3) if float(mv[d]) != 0.0]
            if rows:
                out["step"] += ["*BOUNDARY"] + rows
            out["print"] += [f"*NODE PRINT, NSET=NREF{tag}, TOTALS=ONLY", "RF"]
            ind = faces
        else:
            A = surface(model, c["a"], own)
            B = surface(model, c["b"], own)
            if not A or not B:
                raise ValueError(f"접촉 {c['id']} 의 면을 찾지 못했습니다 (선택 영역 안에 요소 면이 있어야 합니다)")
            from scipy.spatial import cKDTree

            ca = np.array([face_geo(E, f)[0] for *_, f in A])
            cb = np.array([face_geo(E, f)[0] for *_, f in B])
            ia = cKDTree(cb).query(ca)[1]
            ib = cKDTree(ca).query(cb)[1]
            dep = [label(kd, e, lf, f, toward=cb[ia[i]]) for i, (kd, e, lf, f) in enumerate(A)]
            ind = [label(kd, e, lf, f, toward=ca[ib[i]]) for i, (kd, e, lf, f) in enumerate(B)]
        M = out["model"]
        M += [f"*SURFACE, NAME=SDEP{tag}, TYPE=ELEMENT"] + dep + [f"*SURFACE, NAME=SIND{tag}, TYPE=ELEMENT"] + ind
        M += [f"*CONTACT PAIR, INTERACTION=INT{tag}, TYPE=SURFACE TO SURFACE", f"SDEP{tag}, SIND{tag}"]
        M += [f"*SURFACE INTERACTION, NAME=INT{tag}", "*SURFACE BEHAVIOR, PRESSURE-OVERCLOSURE=LINEAR", f"{K:.9g}"]
        if mu > 0:
            M += ["*FRICTION", f"{mu:.9g}, {0.1 * K:.9g}"]
        out["pairs"][c["id"]] = (f"SDEP{tag}", f"SIND{tag}")
        out["print"] += [f"*CONTACT PRINT, SLAVE=SDEP{tag}, MASTER=SIND{tag}", "CF"]
        out["notes"].append(f"접촉 {c['id']}: {describe(c)}, 페널티 강성 {K:.4g} MPa/mm")
    return out


def record(res, spec, times, forces):
    cs = contacts_of(spec)
    first = None
    for c in cs:
        f = forces.get(c["id"])
        if f is None or len(f) != len(times):
            continue
        f = np.array(f, dtype=float).reshape(len(times), 3)
        if c["type"] == "rigid" and np.linalg.norm(c.get("motion", [0, 0, 0])) > 0:
            mag = f @ unit(c["motion"])
        else:
            mag = np.linalg.norm(f, axis=1)
        res.history[f"contact.{c['id']}.force"] = mag.tolist()
        for k, nm in enumerate("xyz"):
            res.history[f"contact.{c['id']}.f{nm}"] = f[:, k].tolist()
        if first is None:
            first = c
            res.history["contact.force"] = mag.tolist()
            amp = float(np.linalg.norm(c.get("motion", [0, 0, 0]))) if c["type"] == "rigid" else 0.0
            res.history["contact.disp"] = [float(t) * amp for t in times] if amp > 0 else [float(t) for t in times]
        res.summary[f"접촉력 {c['id']} (N)"] = round(float(mag[-1]), 4)
        res.summary[f"접촉력 {c['id']} 벡터 (N)"] = [round(float(v), 4) for v in f[-1]]
    res.rigid = [v for v in view_spec(spec) if v["type"] == "rigid"]


def view_spec(spec):
    out = []
    for c in contacts_of(spec):
        if c["type"] == "rigid":
            item = {"id": c["id"], "type": "rigid", "shape": c["shape"], "motion": c.get("motion", [0, 0, 0])}
            for k in ("center", "radius", "point", "normal"):
                if k in c:
                    item[k] = c[k]
            out.append(item)
    return out


def _closest(p, a, b, c):
    ab, ac, ap = b - a, c - a, p - a
    d1 = (ab * ap).sum(-1)
    d2 = (ac * ap).sum(-1)
    bp = p - b
    d3 = (ab * bp).sum(-1)
    d4 = (ac * bp).sum(-1)
    cp = p - c
    d5 = (ab * cp).sum(-1)
    d6 = (ac * cp).sum(-1)
    va = d3 * d6 - d5 * d4
    vb = d5 * d2 - d1 * d6
    vc = d1 * d4 - d3 * d2
    den = va + vb + vc
    den = np.where(np.abs(den) < 1e-300, 1e-300, den)
    v = vb / den
    w = vc / den
    u = 1 - v - w
    inside = (u >= 0) & (v >= 0) & (w >= 0)

    def seg(p0, p1, i0, i1):
        e = p1 - p0
        t = np.clip(((p - p0) * e).sum(-1) / np.maximum((e * e).sum(-1), 1e-300), 0, 1)
        q = p0 + t[..., None] * e
        bw = np.zeros(t.shape + (3,))
        bw[..., i0] = 1 - t
        bw[..., i1] = t
        return ((p - q) ** 2).sum(-1), bw

    best = np.stack([u, v, w], axis=-1)
    q = u[..., None] * a + v[..., None] * b + w[..., None] * c
    dist = np.where(inside, ((p - q) ** 2).sum(-1), np.inf)
    for p0, p1, i0, i1 in ((a, b, 0, 1), (b, c, 1, 2), (c, a, 2, 0)):
        d, bw = seg(p0, p1, i0, i1)
        use = d < dist
        dist = np.where(use, d, dist)
        best = np.where(use[..., None], bw, best)
    return np.sqrt(dist), best


class TMContact:
    def __init__(self, model, spec, pts, mids, torch, dtype):
        self.torch = torch
        self.dt = dtype
        self.items = []
        self.pts = torch.tensor(pts, dtype=dtype)
        self.pts_np = np.asarray(pts, dtype=float)
        n = len(pts)
        tri = model.solid_boundary()
        sub = []
        for a, b, c in tri.tolist():
            ab, bc, ca = mids[(min(a, b), max(a, b))], mids[(min(b, c), max(b, c))], mids[(min(c, a), max(c, a))]
            sub += [(a, ab, ca), (ab, b, bc), (ca, bc, c), (ab, bc, ca)]
        sub = np.array(sub, dtype=int).reshape(-1, 3)
        corner = np.repeat(tri, 4, axis=0)
        p = self.pts_np[sub]
        area = 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), axis=1)
        self.h = float(np.sqrt(area.mean() * 4 / 3**0.5)) if len(area) else 1.0
        self.eps = 1e-3 * self.h
        self.notes = []

        def pick(sel):
            if not sel or sel.get("all"):
                return np.ones(len(sub), dtype=bool)
            ids = np.zeros(len(model.nodes), dtype=bool)
            ids[resolve(model, sel)] = True
            return ids[corner].all(axis=1)

        def weights(keep):
            w = np.zeros(n)
            for k in range(3):
                np.add.at(w, sub[keep][:, k], area[keep] / 3)
            idx = np.where(w > 0)[0]
            return idx, w[idx]

        for c in contacts_of(spec):
            K = float(c.get("stiffness") or default_stiffness(spec))
            it = {"c": c, "K": K, "mu": float(c.get("friction") or 0.0)}
            if c["type"] == "rigid":
                keep = pick(c.get("target"))
                if not keep.any():
                    raise ValueError(f"접촉 {c['id']} 의 대상 면이 없습니다")
                idx, w = weights(keep)
                it["mv"] = torch.tensor(np.array(c.get("motion", [0, 0, 0]), dtype=float), dtype=dtype)
                if c["shape"] == "sphere":
                    it["cen"] = torch.tensor(np.array(c["center"], dtype=float), dtype=dtype)
                else:
                    it["p0"] = torch.tensor(np.array(c["point"], dtype=float), dtype=dtype)
                    it["nrm"] = torch.tensor(unit(c["normal"]), dtype=dtype)
            else:
                ka, kb = pick(c["a"]), pick(c["b"])
                if not ka.any() or not kb.any():
                    raise ValueError(f"접촉 {c['id']} 의 면을 찾지 못했습니다 (선택 영역 안에 요소 면이 있어야 합니다)")
                if (ka & kb).any():
                    raise ValueError(f"접촉 {c['id']} 의 두 영역이 같은 면을 포함합니다")
                idx, w = weights(ka)
                it["mtri"] = sub[kb]
            it["idx"] = torch.tensor(idx)
            it["idx_np"] = idx
            it["w"] = torch.tensor(w, dtype=dtype)
            self.items.append(it)
            self.notes.append(f"접촉 {c['id']}: {describe(c)}, 페널티 강성 {K:.4g} MPa/mm")
        self.prev = torch.zeros((n, 3), dtype=dtype)
        self.lam_prev = 0.0

    def commit(self, U, lam):
        self.prev = self.torch.tensor(np.asarray(U, dtype=float).reshape(-1, 3), dtype=self.dt)
        self.lam_prev = float(lam)

    def _one(self, it, u, lam):
        torch = self.torch
        c = it["c"]
        x = self.pts[it["idx"]] + u[it["idx"]]
        du = u[it["idx"]] - self.prev[it["idx"]]
        if c["type"] == "rigid":
            if c["shape"] == "sphere":
                r = x - (it["cen"] + lam * it["mv"])
                dist = torch.sqrt((r * r).sum(-1) + 1e-300)
                nrm = (r / dist[:, None]).detach()
                gap = dist - float(c["radius"])
            else:
                nrm = it["nrm"].expand(len(x), 3)
                gap = ((x - (it["p0"] + lam * it["mv"])) * nrm).sum(-1)
            slip = du - (lam - self.lam_prev) * it["mv"]
        else:
            from scipy.spatial import cKDTree

            xa = x.detach().numpy()
            m = it["mtri"]
            pm = self.pts_np[m] + u.detach().numpy()[m]
            k = min(8, len(m))
            _, cand = cKDTree(pm.mean(axis=1)).query(xa, k=k)
            cand = cand.reshape(len(xa), k)
            tri = pm[cand]
            d, bw = _closest(xa[:, None, :], tri[:, :, 0], tri[:, :, 1], tri[:, :, 2])
            j = np.argmin(d, axis=1)
            rows = np.arange(len(xa))
            sel = cand[rows, j]
            bw = bw[rows, j]
            t3 = pm[sel]
            nn = np.cross(t3[:, 1] - t3[:, 0], t3[:, 2] - t3[:, 0])
            nn = nn / np.maximum(np.linalg.norm(nn, axis=1), 1e-300)[:, None]
            near = torch.tensor(d[rows, j] < 2.0 * self.h)
            mi = torch.tensor(m[sel])
            bwt = torch.tensor(bw, dtype=self.dt)
            xm = self.pts[mi] + u[mi]
            q = (bwt[:, :, None] * xm).sum(1)
            nrm = torch.tensor(nn, dtype=self.dt)
            gap = ((x - q) * nrm).sum(-1)
            gap = torch.where(near, gap, torch.ones_like(gap))
            slip = du - (bwt[:, :, None] * (u[mi] - self.prev[mi])).sum(1)
        pen = torch.clamp(-gap, min=0.0)
        en = 0.5 * it["K"] * (it["w"] * pen * pen).sum()
        fn = (it["K"] * it["w"] * pen).detach()
        if it["mu"] > 0:
            st = slip - (slip * nrm).sum(-1, keepdim=True) * nrm
            en = en + it["mu"] * (fn * (torch.sqrt((st * st).sum(-1) + self.eps**2) - self.eps)).sum()
        return en

    def energy(self, u, lam):
        en = 0.0
        for it in self.items:
            en = en + self._one(it, u, lam)
        return en

    def forces(self, U, lam):
        torch = self.torch
        out = {}
        for it in self.items:
            u = torch.tensor(np.asarray(U, dtype=float).reshape(-1, 3), dtype=self.dt, requires_grad=True)
            en = self._one(it, u, lam)
            if not torch.is_tensor(en) or not en.requires_grad:
                out[it["c"]["id"]] = [0.0, 0.0, 0.0]
                continue
            (g,) = torch.autograd.grad(en, u)
            out[it["c"]["id"]] = (-g[it["idx"]].sum(0)).tolist()
        return out
