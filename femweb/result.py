import numpy as np


class Result:
    def __init__(self, solver, kind):
        self.solver = solver
        self.kind = kind
        self.nodes = np.zeros((0, 3))
        self.tris = np.zeros((0, 3), dtype=int)
        self.lines = np.zeros((0, 2), dtype=int)
        self.fedges = np.zeros((0, 2), dtype=int)
        self.frames = []
        self.history = {}
        self.summary = {}
        self.log = ""

    def add_frame(self, label, time, U, fields=None):
        self.frames.append({"label": label, "time": float(time), "U": np.asarray(U, dtype=float), "fields": {k: np.asarray(v, dtype=float) for k, v in (fields or {}).items()}})

    def field_names(self):
        names = ["U", "Ux", "Uy", "Uz"]
        for f in self.frames:
            for k in f["fields"]:
                if k not in names:
                    names.append(k)
        return names

    def field(self, i, name):
        f = self.frames[i]
        if name == "U":
            return np.linalg.norm(f["U"], axis=1)
        if name in ("Ux", "Uy", "Uz"):
            return f["U"][:, "xyz".index(name[1])]
        if name in f["fields"]:
            return f["fields"][name]
        return np.zeros(len(self.nodes))

    def meta(self):
        return {
            "solver": self.solver,
            "kind": self.kind,
            "frames": [{"label": f["label"], "time": f["time"]} for f in self.frames],
            "fields": self.field_names(),
            "summary": self.summary,
            "history": sorted(self.history.keys()),
            "log": self.log[-6000:],
        }

    def mesh(self):
        return {"nodes": np.round(self.nodes, 6).tolist(), "tris": self.tris.tolist(), "lines": self.lines.tolist(), "fedges": self.fedges.tolist()}

    def frame(self, i, name):
        f = self.frames[i]
        v = self.field(i, name)
        return {"index": i, "label": f["label"], "time": f["time"], "field": name, "U": _sig(f["U"]).tolist(), "values": _sig(v).tolist(), "min": float(v.min()) if len(v) else 0.0, "max": float(v.max()) if len(v) else 0.0}


def _sig(a, digits=6):
    a = np.asarray(a, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        mag = np.where(a != 0, np.floor(np.log10(np.abs(a))), 0)
    s = 10.0 ** (digits - 1 - mag)
    return np.round(a * s) / s


def von_mises(S):
    S = np.asarray(S, dtype=float)
    xx, yy, zz, xy, yz, zx = (S[:, i] for i in range(6))
    return np.sqrt(0.5 * ((xx - yy) ** 2 + (yy - zz) ** 2 + (zz - xx) ** 2) + 3 * (xy**2 + yz**2 + zx**2))
