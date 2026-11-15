import numpy as np
from .model import Model


def _idx(tok, n):
    i = int(tok.split("/")[0])
    return i - 1 if i > 0 else n + i


def parse_obj(text, name="model.obj", weld_tol=None):
    verts = []
    lines = []
    faces = []
    group = "default"
    for raw in text.splitlines():
        s = raw.strip()
        if not s or s[0] == "#":
            continue
        p = s.split()
        k = p[0]
        if k == "v" and len(p) >= 4:
            verts.append([float(p[1]), float(p[2]), float(p[3])])
        elif k in ("g", "o"):
            group = "_".join(p[1:]) if len(p) > 1 else "default"
        elif k == "l" and len(p) >= 3:
            ids = [_idx(t, len(verts)) for t in p[1:]]
            for a, b in zip(ids[:-1], ids[1:]):
                lines.append((a, b, group))
        elif k == "f" and len(p) >= 4:
            ids = [_idx(t, len(verts)) for t in p[1:]]
            if len(ids) <= 4:
                faces.append((ids, group))
            else:
                for i in range(1, len(ids) - 1):
                    faces.append(([ids[0], ids[i], ids[i + 1]], group))
    if not verts:
        raise ValueError("OBJ에 정점(v)이 없습니다")
    v = np.array(verts, dtype=float)
    span = float(np.linalg.norm(v.max(axis=0) - v.min(axis=0))) or 1.0
    tol = weld_tol if weld_tol else span * 1e-6
    key = np.round(v / tol).astype(np.int64)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    inv = np.asarray(inv).reshape(-1)
    order = np.argsort(first)
    rank = np.empty(len(order), dtype=int)
    rank[order] = np.arange(len(order))
    remap = rank[inv]
    m = Model()
    m.nodes = v[first[order]]
    m.source = name
    m.kind = "obj"
    hinge_lines = []
    seen = set()
    for a, b, g in lines:
        a, b = int(remap[a]), int(remap[b])
        if a == b:
            continue
        if g.lower().startswith("hinge"):
            hinge_lines.append((a, b, g))
            continue
        for e in _split_on_nodes(m.nodes, a, b, tol * 100):
            if e in seen:
                continue
            seen.add(e)
            m.beams.append(e)
            m.beam_group.append(g)
    for ids, g in faces:
        ids = [int(remap[i]) for i in ids]
        clean = []
        for i in ids:
            if i not in clean:
                clean.append(i)
        if len(clean) >= 3:
            m.shells.append(clean)
            m.shell_group.append(g)
    if hinge_lines:
        edges = m.shell_edges()
        for a, b, g in hinge_lines:
            for e in _split_on_nodes(m.nodes, a, b, tol * 100):
                if e in edges:
                    lst = m.hinges.setdefault(g, [])
                    if e not in lst:
                        lst.append(e)
    used = set()
    for b in m.beams:
        used.update(b)
    for s in m.shells:
        used.update(s)
    if len(used) < len(m.nodes):
        _compact(m, sorted(used))
    if not m.beams and not m.shells:
        raise ValueError("OBJ에 선(l)이나 면(f) 요소가 없습니다")
    return m


def _split_on_nodes(nodes, a, b, tol):
    pa, pb = nodes[a], nodes[b]
    d = pb - pa
    L = np.linalg.norm(d)
    if L == 0:
        return []
    t = (nodes - pa) @ d / (L * L)
    dist = np.linalg.norm(nodes - (pa + np.outer(t, d)), axis=1)
    on = np.where((dist < tol) & (t > -1e-9) & (t < 1 + 1e-9))[0]
    on = on[np.argsort(t[on])]
    return [(int(min(i, j)), int(max(i, j))) for i, j in zip(on[:-1], on[1:])]


def _compact(m, keep):
    remap = {old: new for new, old in enumerate(keep)}
    m.nodes = m.nodes[keep]
    m.beams = [(remap[a], remap[b]) for a, b in m.beams]
    m.shells = [[remap[i] for i in s] for s in m.shells]
    m.hinges = {k: [(min(remap[a], remap[b]), max(remap[a], remap[b])) for a, b in v] for k, v in m.hinges.items()}


def refine(model, k):
    k = int(max(1, k))
    if k == 1 or model.kind != "obj":
        return model.copy()
    m = Model()
    m.source = model.source
    m.kind = model.kind
    pts = [p for p in model.nodes]
    chains = {}

    def chain(a, b):
        key = (min(a, b), max(a, b))
        if key not in chains:
            lo, hi = key
            ids = [lo]
            for i in range(1, k):
                pts.append(model.nodes[lo] + (model.nodes[hi] - model.nodes[lo]) * i / k)
                ids.append(len(pts) - 1)
            ids.append(hi)
            chains[key] = ids
        ids = chains[key]
        return ids if a < b else ids[::-1]

    for (a, b), g in zip(model.beams, model.beam_group):
        c = chain(a, b)
        for i in range(k):
            m.beams.append((c[i], c[i + 1]))
            m.beam_group.append(g)
    for s, g in zip(model.shells, model.shell_group):
        if len(s) == 4:
            c0, c1, c2, c3 = s
            bottom = chain(c0, c1)
            right = chain(c1, c2)
            top = chain(c3, c2)
            left = chain(c0, c3)
            grid = [[None] * (k + 1) for _ in range(k + 1)]
            for i in range(k + 1):
                grid[0][i] = bottom[i]
                grid[k][i] = top[i]
                grid[i][0] = left[i]
                grid[i][k] = right[i]
            p0, p1, p2, p3 = (model.nodes[c] for c in s)
            for j in range(1, k):
                for i in range(1, k):
                    u, v = i / k, j / k
                    pts.append((1 - u) * (1 - v) * p0 + u * (1 - v) * p1 + u * v * p2 + (1 - u) * v * p3)
                    grid[j][i] = len(pts) - 1
            for j in range(k):
                for i in range(k):
                    m.shells.append([grid[j][i], grid[j][i + 1], grid[j + 1][i + 1], grid[j + 1][i]])
                    m.shell_group.append(g)
        else:
            a, b, c = s
            ab, ac, bc = chain(a, b), chain(a, c), chain(b, c)
            row = {}
            pa, pb, pc = model.nodes[a], model.nodes[b], model.nodes[c]
            for i in range(k + 1):
                for j in range(k + 1 - i):
                    if j == 0:
                        row[(i, j)] = ab[i]
                    elif i == 0:
                        row[(i, j)] = ac[j]
                    elif i + j == k:
                        row[(i, j)] = bc[j]
                    else:
                        pts.append(pa + (pb - pa) * i / k + (pc - pa) * j / k)
                        row[(i, j)] = len(pts) - 1
            for i in range(k):
                for j in range(k - i):
                    m.shells.append([row[(i, j)], row[(i + 1, j)], row[(i, j + 1)]])
                    m.shell_group.append(g)
                    if i + j < k - 1:
                        m.shells.append([row[(i + 1, j)], row[(i + 1, j + 1)], row[(i, j + 1)]])
                        m.shell_group.append(g)
    for name, edges in model.hinges.items():
        out = []
        for a, b in edges:
            c = chain(a, b)
            for i in range(k):
                out.append((min(c[i], c[i + 1]), max(c[i], c[i + 1])))
        m.hinges[name] = out
    m.nodes = np.array(pts, dtype=float)
    return m
