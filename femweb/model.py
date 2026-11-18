import numpy as np


class Model:
    def __init__(self):
        self.nodes = np.zeros((0, 3))
        self.beams = []
        self.beam_group = []
        self.shells = []
        self.shell_group = []
        self.solids = np.zeros((0, 4), dtype=int)
        self.hinges = {}
        self.patches = {}
        self.source = ""
        self.kind = "obj"

    def copy(self):
        m = Model()
        m.nodes = self.nodes.copy()
        m.beams = [tuple(b) for b in self.beams]
        m.beam_group = list(self.beam_group)
        m.shells = [list(s) for s in self.shells]
        m.shell_group = list(self.shell_group)
        m.solids = self.solids.copy()
        m.hinges = {k: [tuple(e) for e in v] for k, v in self.hinges.items()}
        m.patches = {k: dict(v) for k, v in self.patches.items()}
        m.source = self.source
        m.kind = self.kind
        return m

    def bbox(self):
        if len(self.nodes) == 0:
            return np.zeros(3), np.zeros(3)
        return self.nodes.min(axis=0), self.nodes.max(axis=0)

    def diag(self):
        lo, hi = self.bbox()
        d = float(np.linalg.norm(hi - lo))
        return d if d > 0 else 1.0

    def group_names(self):
        names = []
        for g in self.beam_group + self.shell_group:
            if g not in names:
                names.append(g)
        for g in self.hinges:
            if g not in names:
                names.append(g)
        for g in self.patches:
            if g not in names:
                names.append(g)
        return names

    def group_nodes(self, name):
        ids = set()
        for b, g in zip(self.beams, self.beam_group):
            if g == name:
                ids.update(b)
        for s, g in zip(self.shells, self.shell_group):
            if g == name:
                ids.update(s)
        if name in self.hinges:
            for e in self.hinges[name]:
                ids.update(e)
        if name in self.patches:
            ids.update(self.patches[name]["nodes"])
        return sorted(ids)

    def solid_boundary(self):
        if len(self.solids) == 0:
            return np.zeros((0, 3), dtype=int)
        t = self.solids
        f = np.vstack([t[:, [0, 2, 1]], t[:, [0, 1, 3]], t[:, [1, 2, 3]], t[:, [2, 0, 3]]])
        key = np.sort(f, axis=1)
        _, idx, cnt = np.unique(key, axis=0, return_index=True, return_counts=True)
        return f[idx[cnt == 1]]

    def shell_edges(self):
        edges = {}
        for fi, s in enumerate(self.shells):
            n = len(s)
            for i in range(n):
                a, b = s[i], s[(i + 1) % n]
                edges.setdefault((min(a, b), max(a, b)), []).append(fi)
        return edges

    def summary(self):
        lo, hi = self.bbox()
        groups = {}
        for name in self.group_names():
            ids = self.group_nodes(name)
            if not ids:
                continue
            p = self.nodes[ids]
            kind = "hinge" if name in self.hinges else "patch" if name in self.patches else "element"
            info = {"kind": kind, "nodes": len(ids), "min": p.min(axis=0).round(4).tolist(), "max": p.max(axis=0).round(4).tolist()}
            if name in self.patches:
                info["normal"] = self.patches[name].get("normal")
                info["area"] = self.patches[name].get("area")
            groups[name] = info
        return {
            "kind": self.kind,
            "source": self.source,
            "counts": {"nodes": int(len(self.nodes)), "beams": len(self.beams), "shells": len(self.shells), "solids": int(len(self.solids))},
            "bbox": {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()},
            "groups": groups,
        }

    def to_view(self):
        return {
            "nodes": self.nodes.round(6).tolist(),
            "beams": [list(map(int, b)) for b in self.beams],
            "shells": [list(map(int, s)) for s in self.shells],
            "shell_group": list(self.shell_group),
            "boundary": self.solid_boundary().tolist(),
            "hinges": {k: [list(map(int, e)) for e in v] for k, v in self.hinges.items()},
        }
