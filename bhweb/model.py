import numpy as np

from .geom import dihedral

INF = float("inf")

DEFAULTS = {
    "E": 1.0e3,
    "nu": 1.0 / 3.0,
    "t": 0.1,
    "rho": 1.0e-9,
    "fold_mode": "auto",
    "Lstar": None,
    "Lstar_ratio": 3.0,
    "fold_percent": 10.0,
    "fold_k": None,
    "bend_k": None,
    "long_factor": 100.0,
    "bar_area": None,
    "bar_yield": INF,
    "bar_H": 0.0,
    "fold_yield": INF,
    "fold_H": 0.0,
    "bend_yield": INF,
    "bend_H": 0.0,
    "bend_rest": "asis",
    "tri_bars": "woodruff",
}

FOLD, PANEL, BEND = 0, 1, 2


def edges_of(faces):
    out = {}
    for fi, f in enumerate(faces):
        n = len(f)
        for i in range(n):
            a, b = int(f[i]), int(f[(i + 1) % n])
            out.setdefault((min(a, b), max(a, b)), []).append(fi)
    return out


def _normal(P):
    if len(P) == 3:
        n = np.cross(P[1] - P[0], P[2] - P[0])
    else:
        n = np.cross(P[2] - P[0], P[3] - P[1])
    L = np.linalg.norm(n)
    return n / L if L > 0 else n


def fold_angle_deg(nodes, faces, e, fl):
    f1, f2 = faces[fl[0]], faces[fl[1]]
    n1, n2 = _normal(nodes[f1]), _normal(nodes[f2])
    i = list(f1).index(e[0])
    s1 = f1[(i + 1) % len(f1)] == e[1]
    j = list(f2).index(e[0])
    s2 = f2[(j + 1) % len(f2)] == e[1]
    if s1 == s2:
        n2 = -n2
    return float(np.degrees(np.arccos(np.clip(n1 @ n2, -1.0, 1.0))))


def classify(nodes, faces, crease_edges=None, tol=3.0):
    nodes = np.asarray(nodes, float)
    eds = edges_of(faces)
    out = {}
    angles = {}
    for e, fl in eds.items():
        if len(fl) == 1:
            out[e] = "boundary"
        elif len(fl) == 2:
            angles[e] = fold_angle_deg(nodes, faces, e, fl)
        else:
            out[e] = "fold"
    marked = {tuple(sorted(map(int, e))) for e in (crease_edges or [])}
    if marked:
        for e, a in angles.items():
            out[e] = "fold" if e in marked or a >= tol else "bend"
        return out, angles, "주름선이 지정되어 있어 지정 엣지와 꺾인 엣지는 접힘, 나머지는 휨으로 판정"
    if angles and max(angles.values()) < tol:
        for e in angles:
            out[e] = "fold"
        return out, angles, "평평한 전개도라 내부 엣지를 모두 접힘으로 판정 (휨으로 바꿀 엣지는 직접 골라 주세요)"
    for e, a in angles.items():
        out[e] = "fold" if a >= tol else "bend"
    return out, angles, f"꺾임 {tol:g}도 이상은 접힘, 미만은 휨으로 판정"


def split_cuts(nodes, faces, cuts):
    nodes = [np.asarray(p, float) for p in nodes]
    faces = [list(map(int, f)) for f in faces]
    cuts = {tuple(sorted(map(int, e))) for e in cuts}
    orig = list(range(len(nodes)))
    if not cuts:
        return np.array(nodes), faces, np.array(orig)
    at = {}
    for fi, f in enumerate(faces):
        for v in f:
            at.setdefault(v, []).append(fi)
    old = [list(f) for f in faces]
    for v in sorted({i for e in cuts for i in e}):
        fs = at.get(v, [])
        parent = {f: f for f in fs}

        def find(a):
            while parent[a] != a:
                parent[a] = parent[parent[a]]
                a = parent[a]
            return a

        by = {}
        for f in fs:
            s = old[f]
            i = s.index(v)
            for w in (s[i - 1], s[(i + 1) % len(s)]):
                by.setdefault((min(v, w), max(v, w)), []).append(f)
        for e, lst in by.items():
            if e in cuts:
                continue
            for f in lst[1:]:
                parent[find(f)] = find(lst[0])
        comps = {}
        for f in fs:
            comps.setdefault(find(f), []).append(f)
        keys = sorted(comps, key=lambda k: min(comps[k]))
        for k in keys[1:]:
            nodes.append(nodes[v].copy())
            orig.append(orig[v])
            nv = len(nodes) - 1
            for f in comps[k]:
                faces[f] = [nv if i == v else i for i in faces[f]]
    return np.array(nodes), faces, np.array(orig)


class BHModel:
    def __init__(self, nodes, faces, types=None, params=None, crease_edges=None, tol=3.0):
        p = dict(DEFAULTS)
        p.update({k: v for k, v in (params or {}).items() if v is not None or k in ("Lstar", "fold_k", "bend_k", "bar_area")})
        self.p = p
        nodes = np.asarray(nodes, float)
        faces = [list(map(int, f)) for f in faces]
        for f in faces:
            if len(f) not in (3, 4):
                raise ValueError("바-힌지 패널은 삼각형과 사각형만 받습니다")
        auto, self.edge_angles, self.note = classify(nodes, faces, crease_edges, tol)
        self.auto_types = dict(auto)
        tp = dict(auto)
        for e, t in (types or {}).items():
            e = tuple(sorted(map(int, e)))
            if e in tp and tp[e] != "boundary":
                if t not in ("fold", "bend", "cut"):
                    raise ValueError(f"엣지 종류는 fold, bend, cut 중 하나여야 합니다: {t}")
                tp[e] = t
        self.types = tp
        cuts = [e for e, t in tp.items() if t == "cut"]
        X, F, orig = split_cuts(nodes, faces, cuts)
        self.n_input = len(nodes)
        self.faces = F
        self.orig = orig
        self._build(X, F)

    def _build(self, X0, F):
        p = self.p
        E, nu, t, rho = float(p["E"]), float(p["nu"]), float(p["t"]), float(p["rho"])
        B = E * t**3 / (12 * (1 - nu * nu))
        self.B = B
        X = [x for x in X0]
        bar = {}
        spokes = []
        H = []
        hk = []
        hkind = []
        hedge = []
        hlen = []
        tris = []
        wing = {}
        mass = np.zeros(len(X0) + sum(1 for f in F if len(f) == 4))
        widths = []
        area = {}
        center = {}
        for fi, f in enumerate(F):
            P = X0[f]
            if len(f) == 4:
                W = 0.5 * (np.linalg.norm(P[1] - P[0]) + np.linalg.norm(P[2] - P[3]))
                Hh = 0.5 * (np.linalg.norm(P[2] - P[1]) + np.linalg.norm(P[3] - P[0]))
                widths.append(0.5 * (W + Hh))
                S = 0.5 * np.linalg.norm(np.cross(P[2] - P[0], P[3] - P[1]))
                area[fi] = S
                AX = t * (Hh * Hh - nu * W * W) / (2 * Hh * (1 - nu * nu))
                AY = t * (W * W - nu * Hh * Hh) / (2 * W * (1 - nu * nu))
                AD = t * nu * (Hh * Hh + W * W) ** 1.5 / (2 * Hh * W * (1 - nu * nu))
                AX = max(AX, 0.05 * t * Hh)
                AY = max(AY, 0.05 * t * W)
                AD = max(AD, 0.05 * t * min(W, Hh))
                for i, A in enumerate((AX, AY, AX, AY)):
                    a, b = f[i], f[(i + 1) % 4]
                    k = (min(a, b), max(a, b))
                    bar[k] = bar.get(k, 0.0) + A
                X.append(P.mean(axis=0))
                c = len(X) - 1
                center[fi] = c
                m = rho * t * S / 5.0
                mass[f] += m
                mass[c] += m
                d0 = np.linalg.norm(P[0] - P[2])
                d1 = np.linalg.norm(P[1] - P[3])
                short = 0 if d0 <= d1 else 1
                ends = (0, 2) if short == 0 else (1, 3)
                ang = 0.0
                for v in ends:
                    u1 = P[(v + 1) % 4] - P[v]
                    u2 = P[(v - 1) % 4] - P[v]
                    ang += np.pi - np.arccos(np.clip(u1 @ u2 / (np.linalg.norm(u1) * np.linalg.norm(u2)), -1, 1))
                ang = float(np.clip(ang, np.pi / 2, np.pi))
                DS = min(d0, d1)
                KBs = (0.55 - 0.42 * ang / np.pi) * B * (DS / t) ** (1.0 / 3.0)
                if p.get("bend_k") is not None:
                    KBs = float(p["bend_k"]) * DS
                for i in range(4):
                    spokes.append((c, f[i], AD))
                    diag = i % 2
                    K = 0.5 * KBs * (1.0 if diag == short else float(p["long_factor"]))
                    H.append((f[(i - 1) % 4], c, f[i], f[(i + 1) % 4]))
                    hk.append(K)
                    hkind.append(PANEL)
                    hedge.append(None)
                    hlen.append(np.linalg.norm(P[i] - P.mean(axis=0)))
                    tris.append((c, f[i], f[(i + 1) % 4]))
            else:
                S = 0.5 * np.linalg.norm(np.cross(P[1] - P[0], P[2] - P[0]))
                area[fi] = S
                per = sum(np.linalg.norm(P[i] - P[(i + 1) % 3]) for i in range(3))
                widths.append(per / 3.0)
                Ls = [float(np.linalg.norm(P[(i + 1) % 3] - P[i])) for i in range(3)]
                if p.get("tri_bars") == "merlin2":
                    As = [2 * S * t / ((1 - nu) * per)] * 3
                else:
                    jd = int(np.argmax(Ls))
                    o = [Ls[q] for q in range(3) if q != jd]
                    ar = max(o) / max(min(o), 1e-12)
                    As = [S * t / Ls[i] * ((ar * ar / 2 + 1.25) if i == jd else 1.0) for i in range(3)]
                for i in range(3):
                    a, b = f[i], f[(i + 1) % 3]
                    k = (min(a, b), max(a, b))
                    bar[k] = bar.get(k, 0.0) + As[i]
                    wing[(fi, k)] = f[(i + 2) % 3]
                mass[f] += rho * t * S / 3.0
                tris.append(tuple(f))
        self.wbar = float(np.mean(widths)) if widths else 1.0
        if p.get("Lstar") is not None:
            Ls = float(p["Lstar"])
        else:
            Ls = float(p["Lstar_ratio"]) * self.wbar
        self.Lstar = Ls
        eds = edges_of(F)
        self.fold_edges = []
        for e, fl in eds.items():
            if len(fl) != 2:
                continue
            oe = (int(self.orig[e[0]]), int(self.orig[e[1]]))
            oe = (min(oe), max(oe))
            ty = self.types.get(oe, "fold")
            ws = [center[f] if f in center else wing[(f, e)] for f in fl]
            LF = float(np.linalg.norm(X0[e[0]] - X0[e[1]]))
            if ty == "bend":
                K = 2.0 * B * LF * LF / (area[fl[0]] + area[fl[1]])
                if p.get("bend_k") is not None:
                    K = float(p["bend_k"]) * LF
                kind = BEND
            else:
                mode = p["fold_mode"]
                if mode == "percent":
                    K = float(p["fold_percent"]) / 100.0 * B / self.wbar * LF
                elif mode == "value":
                    K = float(p["fold_k"]) * LF
                else:
                    Kl = LF * B / Ls
                    Km = 0.55 * B * (LF / t) ** (1.0 / 3.0)
                    K = 1.0 / (1.0 / Kl + 1.0 / Km)
                kind = FOLD
            H.append((ws[0], e[0], e[1], ws[1]))
            hk.append(K)
            hkind.append(kind)
            hedge.append(oe)
            hlen.append(LF)
            if kind == FOLD:
                self.fold_edges.append(oe)
        bars = list(bar.keys())
        A = [bar[k] for k in bars]
        for c, i, a in spokes:
            bars.append((c, i))
            A.append(a)
        self.X = np.array(X, float)
        self.bars = np.array(bars, int).reshape(-1, 2)
        A = np.array(A, float)
        if p.get("bar_area") is not None:
            A = np.full(len(A), float(p["bar_area"]))
        self.A = A
        self.L0 = np.linalg.norm(self.X[self.bars[:, 1]] - self.X[self.bars[:, 0]], axis=1)
        self.Eb = np.full(len(A), E)
        self.Yb = np.full(len(A), float(p["bar_yield"]))
        self.Hb = np.full(len(A), float(p["bar_H"]))
        self.h = np.array(H, int).reshape(-1, 4)
        self.hk = np.array(hk, float)
        self.hkind = np.array(hkind, int)
        self.hedge = hedge
        self.hlen = np.array(hlen, float)
        yl = np.where(self.hkind == FOLD, float(p["fold_yield"]), float(p["bend_yield"]))
        hl = np.where(self.hkind == FOLD, float(p["fold_H"]), float(p["bend_H"]))
        self.hY = yl * self.hlen
        self.hH = hl * self.hlen
        x = self.X
        self.th_init = dihedral(x[self.h[:, 0]], x[self.h[:, 1]], x[self.h[:, 2]], x[self.h[:, 3]]) if len(self.h) else np.zeros(0)
        self.th0 = self.th_init.copy()
        if p.get("bend_rest") == "flat":
            self.th0[self.hkind != FOLD] = np.pi
        self.tris = np.array(tris, int).reshape(-1, 3)
        self.mass = mass[: len(self.X)]
        self.nn = len(self.X)
        self.ndof = 3 * self.nn

    def fold_index(self, edge):
        e = tuple(sorted(map(int, edge)))
        for i, he in enumerate(self.hedge):
            if he == e:
                return i
        raise KeyError(f"엣지 {e} 에 힌지가 없습니다")

    def summary(self):
        return {
            "nodes": int(self.nn),
            "input_nodes": int(self.n_input),
            "bars": int(len(self.bars)),
            "folds": int((self.hkind == FOLD).sum()),
            "panel_bends": int((self.hkind == PANEL).sum()),
            "edge_bends": int((self.hkind == BEND).sum()),
            "B": self.B,
            "Lstar": self.Lstar,
            "wbar": self.wbar,
        }
