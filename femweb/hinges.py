import json

import numpy as np

AUTO = "hinge_auto"
MODES = ("off", "auto", "groups", "crease", "all")


def _normal(model, s):
    p = model.nodes[s]
    n = np.cross(p[1] - p[0], p[2] - p[0]) if len(s) == 3 else np.cross(p[2] - p[0], p[3] - p[1])
    L = np.linalg.norm(n)
    return n / L if L > 0 else n


def _fold_angle(model, e, f1, f2):
    s1, s2 = model.shells[f1], model.shells[f2]
    n1, n2 = _normal(model, s1), _normal(model, s2)
    i = s1.index(e[0])
    same = s1[(i + 1) % len(s1)] == e[1]
    j = s2.index(e[0])
    same2 = s2[(j + 1) % len(s2)] == e[1]
    if same == same2:
        n2 = -n2
    return float(np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0))))


def auto_edges(model, mode="auto", angle=5.0):
    if mode == "off" or not model.shells:
        return []
    taken = {e for es in model.hinges.values() for e in es}
    out = []
    for e, fl in model.shell_edges().items():
        if len(fl) != 2 or e in taken:
            continue
        diff = model.shell_group[fl[0]] != model.shell_group[fl[1]]
        if mode == "all":
            ok = True
        elif mode == "groups":
            ok = diff
        else:
            crease = _fold_angle(model, e, fl[0], fl[1]) > angle
            ok = crease if mode == "crease" else diff or crease
        if ok:
            out.append(e)
    return sorted(out)


def auto_cfg(spec):
    c = {"mode": "off", "angle": 5.0}
    c.update(spec.get("hinge_auto") or {})
    return c


def auto_key(spec):
    return json.dumps(auto_cfg(spec), sort_keys=True)


def with_auto(model, spec):
    c = auto_cfg(spec)
    if c["mode"] == "off" or model.kind != "obj":
        return model
    es = auto_edges(model, c["mode"], float(c["angle"]))
    if not es:
        return model
    m = model.copy()
    m.hinges[AUTO] = es
    return m


def is_pin(spec, name):
    h = dict(spec["hinges"].get("*", {}))
    h.update(spec["hinges"].get(name, {}))
    if h.get("pin"):
        return True
    return h.get("kappa") is not None and float(h["kappa"]) == 0.0


def split_pins(model, spec):
    pins = {}
    for name, es in model.hinges.items():
        if is_pin(spec, name):
            for e in es:
                pins[e] = name
    if not pins:
        return model, [], [], []
    m = model.copy()
    m.hinges = {k: v for k, v in m.hinges.items() if not is_pin(spec, k)}
    at = {}
    for fi, s in enumerate(model.shells):
        for v in s:
            at.setdefault(v, []).append(fi)
    nodes = [p for p in model.nodes]
    ties = []
    for v in sorted({i for e in pins for i in e}):
        fs = at.get(v, [])
        parent = {f: f for f in fs}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by_edge = {}
        for f in fs:
            s = model.shells[f]
            i = s.index(v)
            for w in (s[i - 1], s[(i + 1) % len(s)]):
                by_edge.setdefault((min(v, w), max(v, w)), []).append(f)
        for e, lst in by_edge.items():
            if e in pins:
                continue
            for f in lst[1:]:
                parent[find(f)] = find(lst[0])
        comps = {}
        for f in fs:
            comps.setdefault(find(f), []).append(f)
        keys = sorted(comps, key=lambda k: min(comps[k]))
        for k in keys[1:]:
            nodes.append(model.nodes[v].copy())
            new = len(nodes) - 1
            ties.append((v, new))
            for f in comps[k]:
                m.shells[f] = [new if i == v else i for i in m.shells[f]]
    m.nodes = np.array(nodes, dtype=float)
    pairs = []
    faces_of = model.shell_edges()
    for e in pins:
        fl = faces_of.get(e, [])
        keyed = []
        for f in fl:
            s0, s1 = model.shells[f], m.shells[f]
            a, b = s1[s0.index(e[0])], s1[s0.index(e[1])]
            keyed.append((min(a, b), max(a, b)))
        for k in keyed[1:]:
            if k != keyed[0]:
                pairs.append((keyed[0], k))
    return m, ties, pairs, sorted(set(pins.values()))
