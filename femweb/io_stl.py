import math
import os
import tempfile
import numpy as np
from .model import Model


def stl_to_solid(data, name="model.stl", size=None, angle=40.0):
    import gmsh

    fd, path = tempfile.mkstemp(suffix=".stl")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    started = not gmsh.isInitialized()
    if started:
        gmsh.initialize(interruptible=False)
    try:
        gmsh.option.setNumber("General.Terminal", 0)
        gmsh.model.add("stl")
        gmsh.merge(path)
        gmsh.model.mesh.removeDuplicateNodes()
        gmsh.model.mesh.classifySurfaces(angle * math.pi / 180.0, True, False, math.pi)
        gmsh.model.mesh.createGeometry()
        surfs = [s[1] for s in gmsh.model.getEntities(2)]
        if not surfs:
            raise ValueError("STL에서 표면을 찾지 못했습니다")
        loop = gmsh.model.geo.addSurfaceLoop(surfs)
        gmsh.model.geo.addVolume([loop])
        gmsh.model.geo.synchronize()
        box = gmsh.model.getBoundingBox(-1, -1)
        diag = math.dist(box[:3], box[3:])
        h = float(size) if size else diag / 14.0
        gmsh.option.setNumber("Mesh.MeshSizeMin", h * 0.5)
        gmsh.option.setNumber("Mesh.MeshSizeMax", h)
        gmsh.option.setNumber("Mesh.Algorithm3D", 1)
        gmsh.option.setNumber("Mesh.Optimize", 1)
        gmsh.model.mesh.generate(3)
        tags, coords, _ = gmsh.model.mesh.getNodes()
        coords = np.array(coords).reshape(-1, 3)
        et, _, en = gmsh.model.mesh.getElements(3)
        tets = None
        for t, n in zip(et, en):
            if t == 4:
                tets = np.array(n, dtype=np.int64).reshape(-1, 4)
        if tets is None or len(tets) == 0:
            raise ValueError("체적 메쉬 생성 실패: STL이 닫힌 표면인지 확인하세요")
        lut = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        lut[np.array(tags, dtype=np.int64)] = np.arange(len(tags))
        used = np.unique(tets)
        new = np.full(int(tags.max()) + 1, -1, dtype=np.int64)
        new[used] = np.arange(len(used))
        m = Model()
        m.kind = "stl"
        m.source = name
        m.nodes = coords[lut[used]]
        m.solids = new[tets]
        _fix_orientation(m)
        for s in surfs:
            ft, _, fn = gmsh.model.mesh.getElements(2, s)
            tri = None
            for t, n in zip(ft, fn):
                if t == 2:
                    tri = np.array(n, dtype=np.int64).reshape(-1, 3)
            if tri is None:
                continue
            tri = new[tri]
            p = m.nodes[tri]
            cr = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            area = 0.5 * np.linalg.norm(cr, axis=1)
            nrm = cr.sum(axis=0)
            ln = np.linalg.norm(nrm)
            cen = p.mean(axis=(0, 1))
            cvol = m.nodes.mean(axis=0)
            if ln > 0:
                nrm = nrm / ln
                planar = bool(ln / (2 * area.sum()) > 0.999)
                if planar and np.dot(nrm, cen - cvol) < 0:
                    nrm = -nrm
            m.patches[f"face_{len(m.patches) + 1}"] = {
                "nodes": sorted(set(int(i) for i in tri.ravel())),
                "area": round(float(area.sum()), 4),
                "normal": [round(float(x), 3) for x in nrm] if ln > 0 and planar else None,
            }
        return m
    finally:
        try:
            gmsh.model.remove()
        except Exception:
            pass
        if started:
            gmsh.finalize()
        os.unlink(path)


def _fix_orientation(m):
    p = m.nodes[m.solids]
    vol = np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])
    neg = vol < 0
    m.solids[neg] = m.solids[neg][:, [0, 2, 1, 3]]


def check_stl(data):
    import struct

    tris = None
    if len(data) >= 84:
        n = struct.unpack("<I", data[80:84])[0]
        if 84 + 50 * n == len(data):
            a = np.frombuffer(data, dtype=np.dtype([("n", "<f4", 3), ("v", "<f4", (3, 3)), ("a", "<u2")]), count=n, offset=84)
            tris = a["v"].astype(float)
    if tris is None:
        vs = []
        for line in data.decode("utf-8", "ignore").splitlines():
            s = line.split()
            if len(s) == 4 and s[0] == "vertex":
                vs.append([float(s[1]), float(s[2]), float(s[3])])
        if len(vs) < 3 or len(vs) % 3:
            raise ValueError("STL 형식을 읽을 수 없습니다")
        tris = np.array(vs).reshape(-1, 3, 3)
    span = float(np.linalg.norm(tris.reshape(-1, 3).max(axis=0) - tris.reshape(-1, 3).min(axis=0))) or 1.0
    key = np.round(tris.reshape(-1, 3) / (span * 1e-6)).astype(np.int64)
    _, inv = np.unique(key, axis=0, return_inverse=True)
    f = np.asarray(inv).reshape(-1, 3)
    e = np.sort(np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    _, cnt = np.unique(e, axis=0, return_counts=True)
    return {"triangles": int(len(f)), "open_edges": int((cnt == 1).sum()), "nonmanifold_edges": int((cnt > 2).sum()), "watertight": bool((cnt == 2).all())}
