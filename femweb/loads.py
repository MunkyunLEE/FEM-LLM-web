import numpy as np
from .select import describe, resolve


def _faces(model):
    if model.kind == "stl":
        return [list(map(int, f)) for f in model.solid_boundary()]
    return [list(s) for s in model.shells]


def _face_area_normal(nodes, f):
    p = nodes[f]
    if len(f) == 3:
        c = np.cross(p[1] - p[0], p[2] - p[0])
    else:
        c = np.cross(p[2] - p[0], p[3] - p[1])
    a = 0.5 * np.linalg.norm(c)
    return a, (c / (2 * a) if a > 0 else np.zeros(3))


def _edges(model):
    out = set()
    for f in _faces(model):
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            out.add((min(a, b), max(a, b)))
    for a, b in model.beams:
        out.add((min(a, b), max(a, b)))
    return out


def assemble(model, spec):
    res = {"node": [], "edge": [], "face": [], "moment": [], "gravity": None, "total": np.zeros(3)}
    faces = None
    edges = None
    for ld in spec["loads"]:
        t = ld["type"]
        if t == "gravity":
            res["gravity"] = np.array(ld["vector"], dtype=float)
            continue
        ids = resolve(model, ld.get("select"))
        if len(ids) == 0:
            raise ValueError(f"하중 {ld['id']} 의 대상 절점이 0개입니다")
        sel = set(ids.tolist())
        if t == "moment":
            v = np.array(ld["vector"], dtype=float) / len(ids)
            res["moment"] += [(int(i), v) for i in ids]
            continue
        if faces is None:
            faces = _faces(model)
        hit = [f for f in faces if all(i in sel for i in f)]
        if t == "pressure":
            if not hit:
                raise ValueError(f"압력 {ld['id']} 을(를) 가할 면이 선택 영역 안에 없습니다")
            d = ld.get("direction")
            for f in hit:
                a, n = _face_area_normal(model.nodes, f)
                if d is not None:
                    dv = np.array(d, dtype=float)
                    dv = dv / (np.linalg.norm(dv) or 1.0)
                else:
                    dv = -n
                res["face"].append((f, ld["value"] * a * dv))
                res["total"] += ld["value"] * a * dv
            continue
        v = np.array(ld["vector"], dtype=float)
        res["total"] += v
        if hit:
            areas = np.array([_face_area_normal(model.nodes, f)[0] for f in hit])
            for f, a in zip(hit, areas):
                res["face"].append((f, v * a / areas.sum()))
            continue
        if edges is None:
            edges = _edges(model)
        eh = [e for e in edges if e[0] in sel and e[1] in sel]
        if eh:
            L = np.array([np.linalg.norm(model.nodes[a] - model.nodes[b]) for a, b in eh])
            for e, l in zip(eh, L):
                res["edge"].append((e, v * l / L.sum()))
            continue
        res["node"] += [(int(i), v / len(ids)) for i in ids]
    return res


def lumped(model, spec):
    a = assemble(model, spec)
    F = np.zeros((len(model.nodes), 3))
    M = np.zeros((len(model.nodes), 3))
    for i, v in a["node"]:
        F[i] += v
    for (i, j), v in a["edge"]:
        F[i] += v / 2
        F[j] += v / 2
    for f, v in a["face"]:
        for i in f:
            F[i] += v / len(f)
    for i, v in a["moment"]:
        M[i] += v
    return F, M, a


def bc_table(model, spec):
    table = {}
    for bc in spec["bcs"]:
        ids = resolve(model, bc["select"])
        if len(ids) == 0:
            raise ValueError(f"구속 {bc['id']} 의 대상 절점이 0개입니다")
        for i in ids:
            for d in bc["dofs"]:
                table[(int(i), int(d))] = float(bc.get("value", 0.0))
    return table


def preview(model, spec):
    out = {"bcs": [], "loads": [], "errors": []}
    for bc in spec["bcs"]:
        try:
            ids = resolve(model, bc["select"])
            out["bcs"].append({"id": bc["id"], "nodes": ids.tolist(), "dofs": bc["dofs"], "value": bc.get("value", 0.0), "desc": describe(bc["select"])})
        except Exception as e:
            out["errors"].append(f"{bc['id']}: {e}")
    for ld in spec["loads"]:
        try:
            if ld["type"] == "gravity":
                out["loads"].append({"id": ld["id"], "type": "gravity", "nodes": [], "vectors": [], "total": ld["vector"], "desc": "전체"})
                continue
            one = dict(spec)
            one["loads"] = [ld]
            F, M, a = lumped(model, one)
            V = M if ld["type"] == "moment" else F
            ids = np.where(np.abs(V).sum(axis=1) > 0)[0]
            out["loads"].append({"id": ld["id"], "type": ld["type"], "nodes": ids.tolist(), "vectors": V[ids].round(8).tolist(), "total": (V.sum(axis=0).round(8) + 0.0).tolist(), "desc": describe(ld.get("select"))})
        except Exception as e:
            out["errors"].append(f"{ld['id']}: {e}")
    from .contact import contacts_of, describe as cdesc, view

    out["contacts"] = view(model, spec)
    for c, v in zip(contacts_of(spec), out["contacts"]):
        v["desc"] = cdesc(c)
        if v.get("error"):
            out["errors"].append(f"{c['id']}: {v['error']}")
    return out
