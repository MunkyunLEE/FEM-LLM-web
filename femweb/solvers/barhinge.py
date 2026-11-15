import numpy as np
from ..loads import bc_table, lumped
from ..result import Result
from ..spec import material_for, section_for
from .calculix import hinge_kappa


class Truss:
    def __init__(self, X, bars, EA, hinges=None, K=None, kinds=None):
        self.X = np.asarray(X, dtype=float)
        self.bars = np.asarray(bars, dtype=int).reshape(-1, 2)
        self.EA = np.asarray(EA, dtype=float)
        d = self.X[self.bars[:, 1]] - self.X[self.bars[:, 0]]
        self.L0 = np.linalg.norm(d, axis=1)
        self.h = np.asarray(hinges if hinges is not None else np.zeros((0, 4)), dtype=int).reshape(-1, 4)
        self.K = np.asarray(K if K is not None else np.zeros(0), dtype=float)
        self.kinds = np.asarray(kinds if kinds is not None else np.zeros(len(self.h)), dtype=int)
        self.th0 = self.angles(self.X) if len(self.h) else np.zeros(0)
        self.ref = self.th0.copy()

    def angles(self, x, grad=False):
        return self._ang(x[self.h[:, 0]], x[self.h[:, 1]], x[self.h[:, 2]], x[self.h[:, 3]], grad)

    @staticmethod
    def _ang(xi, xj, xk, xl, grad=False):
        rij = xi - xj
        rkj = xk - xj
        rkl = xk - xl
        m = np.cross(rij, rkj)
        n = np.cross(rkj, rkl)
        lk = np.linalg.norm(rkj, axis=1)
        mm = np.einsum("ij,ij->i", m, m)
        nn = np.einsum("ij,ij->i", n, n)
        c = np.einsum("ij,ij->i", m, n) / np.sqrt(mm * nn)
        s = np.einsum("ij,ij->i", m, rkl)
        th = np.arccos(np.clip(c, -1.0, 1.0))
        th = np.where(s < 0, 2 * np.pi - th, th)
        if not grad:
            return th
        di = (lk / mm)[:, None] * m
        dl = -(lk / nn)[:, None] * n
        a = (np.einsum("ij,ij->i", rij, rkj) / lk**2)[:, None]
        b = (np.einsum("ij,ij->i", rkl, rkj) / lk**2)[:, None]
        dj = (a - 1) * di - b * dl
        dk = (b - 1) * dl - a * di
        return th, (di, dj, dk, dl)

    def _hinge_local(self, xl):
        th, (di, dj, dk, dl) = self._ang(xl[:, 0], xl[:, 1], xl[:, 2], xl[:, 3], True)
        M = (self.K * (self._unwrap(th) - self.th0))[:, None]
        return np.stack([M * di, M * dj, M * dk, M * dl], axis=1).reshape(len(xl), 12)

    def _unwrap(self, th):
        d = th - self.ref
        return th - 2 * np.pi * np.round(d / (2 * np.pi))

    def energy(self, x, parts=False):
        d = x[self.bars[:, 1]] - x[self.bars[:, 0]]
        L = np.linalg.norm(d, axis=1)
        eb = 0.5 * self.EA / self.L0 * (L - self.L0) ** 2
        if len(self.h):
            th = self._unwrap(self.angles(x))
            eh = 0.5 * self.K * (th - self.th0) ** 2
        else:
            eh = np.zeros(0)
        if parts:
            return float(eb.sum()), float(eh[self.kinds == 0].sum()), float(eh[self.kinds == 1].sum())
        return float(eb.sum() + eh.sum())

    def gradient(self, x):
        g = np.zeros_like(x)
        d = x[self.bars[:, 1]] - x[self.bars[:, 0]]
        L = np.linalg.norm(d, axis=1)
        f = (self.EA / self.L0 * (L - self.L0) / L)[:, None] * d
        np.add.at(g, self.bars[:, 1], f)
        np.add.at(g, self.bars[:, 0], -f)
        if len(self.h):
            th, (di, dj, dk, dl) = self.angles(x, grad=True)
            M = (self.K * (self._unwrap(th) - self.th0))[:, None]
            np.add.at(g, self.h[:, 0], M * di)
            np.add.at(g, self.h[:, 1], M * dj)
            np.add.at(g, self.h[:, 2], M * dk)
            np.add.at(g, self.h[:, 3], M * dl)
        return g

    def hessian(self, x, free):
        from scipy import sparse

        nd = x.size
        rows = []
        cols = []
        vals = []
        d = x[self.bars[:, 1]] - x[self.bars[:, 0]]
        L = np.linalg.norm(d, axis=1)
        n = d / L[:, None]
        k = self.EA / self.L0
        nn = np.einsum("ij,ik->ijk", n, n)
        kb = k[:, None, None] * (nn + ((L - self.L0) / L)[:, None, None] * (np.eye(3)[None] - nn))
        ia = 3 * self.bars[:, 0][:, None] + np.arange(3)[None]
        ib = 3 * self.bars[:, 1][:, None] + np.arange(3)[None]
        for r, c, sg in ((ia, ia, 1), (ib, ib, 1), (ia, ib, -1), (ib, ia, -1)):
            rows.append(np.repeat(r[:, :, None], 3, axis=2).ravel())
            cols.append(np.repeat(c[:, None, :], 3, axis=1).ravel())
            vals.append((sg * kb).ravel())
        if len(self.h):
            xl = x[self.h]
            scale = float(np.mean(self.L0))
            e = 1e-6 * scale
            Hl = np.zeros((len(self.h), 12, 12))
            for c in range(12):
                xp = xl.copy()
                xm = xl.copy()
                xp[:, c // 3, c % 3] += e
                xm[:, c // 3, c % 3] -= e
                Hl[:, :, c] = (self._hinge_local(xp) - self._hinge_local(xm)) / (2 * e)
            Hl = 0.5 * (Hl + Hl.transpose(0, 2, 1))
            idx = (3 * self.h[:, :, None] + np.arange(3)[None, None]).reshape(len(self.h), 12)
            rows.append(np.repeat(idx[:, :, None], 12, axis=2).ravel())
            cols.append(np.repeat(idx[:, None, :], 12, axis=1).ravel())
            vals.append(Hl.ravel())
        H = sparse.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(nd, nd)).tocsr()
        return H[free][:, free].tocsc()

    def strain(self, x):
        d = x[self.bars[:, 1]] - x[self.bars[:, 0]]
        return (np.linalg.norm(d, axis=1) - self.L0) / self.L0


def _solve(H, r):
    import warnings
    from scipy import sparse
    from scipy.sparse.linalg import splu

    d = abs(H.diagonal()).max() if H.shape[0] else 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            u = splu(H).solve(r)
            if np.all(np.isfinite(u)) and np.linalg.norm(H @ u - r) <= 1e-6 * max(np.linalg.norm(r), 1e-30):
                return u, False
        except Exception:
            pass
        u = splu((H + 1e-8 * d * sparse.identity(H.shape[0])).tocsc()).solve(r)
    return u, True


def newton(tr, x, free, fext, tol=1e-8, itmax=60):
    flat = x.ravel().copy()
    fnorm = max(np.linalg.norm(fext), 1e-12)
    soft = False
    scale = float(np.mean(tr.L0))
    ref = None
    for it in range(itmax):
        g = tr.gradient(flat.reshape(-1, 3)).ravel()
        r = (fext - g)[free]
        rn = np.linalg.norm(r)
        if ref is None:
            ref = max(rn, fnorm, 1e-12)
        if rn <= tol * ref or rn < 1e-10:
            return flat.reshape(-1, 3), it, True, soft
        H = tr.hessian(flat.reshape(-1, 3), free)
        du, s = _solve(H, r)
        soft = soft or s
        mx = np.abs(du).max()
        if mx > 0.5 * scale:
            du *= 0.5 * scale / mx
        a = 1.0
        for _ in range(12):
            trial = flat.copy()
            trial[free] += a * du
            rt = (fext - tr.gradient(trial.reshape(-1, 3)).ravel())[free]
            if np.linalg.norm(rt) < rn or a < 1e-3:
                break
            a *= 0.5
        flat = trial
    return flat.reshape(-1, 3), itmax, False, soft


def run_static(tr, free, pres, pval, F0, n):
    x = tr.X.copy()
    frames = []
    msgs = []
    lam_done = 0.0
    step = 1.0 / n
    lam = 0.0
    cuts = 0
    while lam_done < 1.0 - 1e-12:
        lam = min(1.0, lam_done + step)
        trial = x.copy().ravel()
        trial[pres] = tr.X.ravel()[pres] + lam * pval
        xn, it, ok, soft = newton(tr, trial.reshape(-1, 3), free, lam * F0.ravel())
        if not ok:
            cuts += 1
            step *= 0.5
            if cuts > 12:
                msgs.append(f"하중계수 {lam_done:.4g} 이후 수렴 실패")
                break
            continue
        x = xn
        tr.ref = tr._unwrap(tr.angles(x)) if len(tr.h) else tr.ref
        lam_done = lam
        frames.append((lam, x.copy()))
        if soft and "특이" not in " ".join(msgs):
            msgs.append("강성행렬이 특이에 가깝습니다 (기구 또는 구속 부족 가능)")
        if step < 1.0 / n:
            step = min(1.0 / n, step * 2)
    return frames, msgs


def run_arclength(tr, free, F0, n, dlam0):
    x = tr.X.copy().ravel()
    f = F0.ravel()
    frames = []
    msgs = []
    lam = 0.0
    H = tr.hessian(x.reshape(-1, 3), free)
    ut, _ = _solve(H, f[free])
    dl = dlam0 * np.linalg.norm(ut)
    prev = None
    for stepi in range(n):
        ok = False
        for attempt in range(8):
            H = tr.hessian(x.reshape(-1, 3), free)
            ut, _ = _solve(H, f[free])
            sgn = 1.0 if prev is None else np.sign(prev @ ut) or 1.0
            dlam = sgn * dl / np.linalg.norm(ut)
            du = dlam * ut
            xi = x.copy()
            good = False
            for it in range(40):
                xi = x.copy()
                xi[free] += du
                g = tr.gradient(xi.reshape(-1, 3)).ravel()
                r = ((lam + dlam) * f - g)[free]
                if np.linalg.norm(r) <= 1e-8 * max(np.linalg.norm(f), 1e-12):
                    good = True
                    break
                H = tr.hessian(xi.reshape(-1, 3), free)
                ur, _ = _solve(H, r)
                ut, _ = _solve(H, f[free])
                a = ut @ ut
                b = 2 * (du + ur) @ ut
                c = (du + ur) @ (du + ur) - dl * dl
                disc = b * b - 4 * a * c
                if disc < 0:
                    break
                r1 = (-b + np.sqrt(disc)) / (2 * a)
                r2 = (-b - np.sqrt(disc)) / (2 * a)
                c1 = (du + ur + r1 * ut) @ du
                c2 = (du + ur + r2 * ut) @ du
                dd = r1 if c1 >= c2 else r2
                du = du + ur + dd * ut
                dlam += dd
            if good:
                ok = True
                break
            dl *= 0.5
        if not ok:
            msgs.append(f"호장법 {stepi + 1}번째 증분에서 수렴 실패")
            break
        x = xi
        lam += dlam
        prev = du
        tr.ref = tr._unwrap(tr.angles(x.reshape(-1, 3))) if len(tr.h) else tr.ref
        frames.append((lam, x.reshape(-1, 3).copy()))
    return frames, msgs


def build(model, spec):
    X = [p for p in model.nodes]
    edges = model.shell_edges()
    bar_area = {}
    spokes = []
    hinges = []
    K = []
    kinds = []
    names = []
    center = {}
    opp = {}
    tris = []
    for fi, (s, g) in enumerate(zip(model.shells, model.shell_group)):
        mat = material_for(spec, g)
        t = section_for(spec, g)["thickness"]
        nu = mat["nu"]
        E = mat["E"]
        p = model.nodes[s]
        if len(s) == 4:
            W = 0.5 * (np.linalg.norm(p[1] - p[0]) + np.linalg.norm(p[2] - p[3]))
            H = 0.5 * (np.linalg.norm(p[2] - p[1]) + np.linalg.norm(p[3] - p[0]))
            AX = t * (H * H - nu * W * W) / (2 * H * (1 - nu * nu))
            AY = t * (W * W - nu * H * H) / (2 * W * (1 - nu * nu))
            AD = t * nu * (H * H + W * W) ** 1.5 / (2 * H * W * (1 - nu * nu))
            AX = max(AX, 0.05 * t * H)
            AY = max(AY, 0.05 * t * W)
            AD = max(AD, 0.05 * t * min(W, H))
            for i, A in enumerate((AX, AY, AX, AY)):
                e = (min(s[i], s[(i + 1) % 4]), max(s[i], s[(i + 1) % 4]))
                bar_area[e] = bar_area.get(e, 0.0) + A * E
            X.append(p.mean(axis=0))
            c = len(X) - 1
            center[fi] = c
            D = E * t**3 / (12 * (1 - nu * nu))
            for i in range(4):
                spokes.append((c, s[i], AD * E))
                diag = np.linalg.norm(p[i] - p[(i + 2) % 4])
                kb = 0.441 * D * (diag / t) ** (1 / 3) * 0.5
                hinges.append((s[(i - 1) % 4], c, s[i], s[(i + 1) % 4]))
                K.append(kb)
                kinds.append(0)
                names.append("")
                tris.append((c, s[i], s[(i + 1) % 4]))
        else:
            for i in range(3):
                a, b, o = s[i], s[(i + 1) % 3], s[(i + 2) % 3]
                L = np.linalg.norm(model.nodes[a] - model.nodes[b])
                area = 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
                e = (min(a, b), max(a, b))
                bar_area[e] = bar_area.get(e, 0.0) + E * t * (area / L) / (1 - nu)
                opp[(fi, e)] = o
            tris.append(tuple(s))
    hname = {}
    for name, es in model.hinges.items():
        for e in es:
            hname[e] = name
    nfold = 0
    for e, fl in edges.items():
        if len(fl) != 2:
            continue
        wings = []
        for fi in fl:
            wings.append(center[fi] if fi in center else opp[(fi, e)])
        g = model.shell_group[fl[0]]
        L = np.linalg.norm(model.nodes[e[0]] - model.nodes[e[1]])
        if model.hinges and e not in hname:
            mat = material_for(spec, g)
            t = section_for(spec, g)["thickness"]
            k = mat["E"] * t**3 / (12 * (1 - mat["nu"] ** 2)) * L / max(0.5 * L, 1e-9)
            kind = 0
            nm = ""
        else:
            nm = hname.get(e, "*")
            k = hinge_kappa(spec, nm, model, g) * L
            kind = 1
            nfold += 1
        hinges.append((wings[0], e[0], e[1], wings[1]))
        K.append(k)
        kinds.append(kind)
        names.append(nm)
    bars = []
    EA = []
    for e, ea in bar_area.items():
        bars.append(e)
        EA.append(ea)
    nedge = len(bars)
    for c, i, ea in spokes:
        bars.append((c, i))
        EA.append(ea)
    tr = Truss(np.array(X), bars, EA, hinges, K, kinds)
    return tr, {"tris": np.array(tris, dtype=int), "lines": np.array(bars[:nedge], dtype=int), "names": names, "nfold": nfold}


def solve(model, spec, progress=None):
    if not model.shells:
        raise ValueError("바-힌지 솔버에는 면(f) 패널이 필요합니다")
    tr, info = build(model, spec)
    n0 = len(model.nodes)
    ntot = len(tr.X)
    table = bc_table(model, spec)
    pres = []
    pval = []
    for (n, d), v in table.items():
        if d <= 3:
            pres.append(3 * n + d - 1)
            pval.append(v)
    pres = np.array(pres, dtype=int)
    pval = np.array(pval, dtype=float)
    free = np.setdiff1d(np.arange(3 * ntot), pres)
    F0 = np.zeros((ntot, 3))
    F, M, _ = lumped(model, spec)
    F0[:n0] = F
    st = spec["step"]
    n = max(1, int(st.get("increments", 25)))
    msgs = []
    if np.abs(M).sum() > 0:
        msgs.append("바-힌지 모델에는 회전 자유도가 없어 모멘트 하중은 무시했습니다")
    if st["type"] == "arclength":
        if np.abs(F0).sum() == 0:
            raise ValueError("호장법에는 힘 하중이 필요합니다")
        if np.abs(pval).sum() > 0:
            msgs.append("호장법에서는 지정 변위를 0으로 고정했습니다")
        frames, m2 = run_arclength(tr, free, F0, n, 1.0 / max(4, n // 4))
    elif st["type"] in ("static",):
        frames, m2 = run_static(tr, free, pres, pval, F0, n)
    else:
        raise ValueError("바-힌지 솔버는 static 또는 arclength 스텝만 지원합니다")
    msgs += m2
    if not frames:
        raise RuntimeError("바-힌지 해석이 수렴하지 않았습니다. 구속을 확인하세요\n" + "\n".join(msgs))
    res = Result("barhinge", "static")
    res.nodes = tr.X.copy()
    res.tris = info["tris"]
    res.lines = info["lines"]
    hist = {k: [] for k in ("time", "load_factor", "energy.bar", "energy.bend", "energy.fold", "energy.total", "reaction.x", "reaction.y", "reaction.z", "u.max", "fold.mean", "strain.max")}
    fold = tr.kinds == 1
    names = np.array(info["names"])
    gnames = sorted({nm for nm in names[fold] if nm and nm != "*"})
    for nm in gnames:
        hist[f"fold.{nm}"] = []
    for k, (lam, x) in enumerate(frames):
        U = x - tr.X
        eb, ebend, efold = tr.energy(x, parts=True)
        g = tr.gradient(x)
        r = np.zeros(3)
        if len(pres):
            gp = g.ravel()[pres]
            mv = pval != 0 if np.abs(pval).sum() > 0 else np.ones(len(pres), dtype=bool)
            for c in range(3):
                r[c] = gp[(pres % 3 == c) & mv].sum()
        eps = np.abs(tr.strain(x))
        nodal = np.zeros(ntot)
        cnt = np.zeros(ntot)
        np.add.at(nodal, tr.bars[:, 0], eps)
        np.add.at(nodal, tr.bars[:, 1], eps)
        np.add.at(cnt, tr.bars[:, 0], 1)
        np.add.at(cnt, tr.bars[:, 1], 1)
        th = tr._unwrap(tr.angles(x)) if len(tr.h) else np.zeros(0)
        rot = np.zeros(ntot)
        rc = np.zeros(ntot)
        if fold.any():
            dth = np.degrees(np.abs(th - tr.th0))[fold]
            for c in (1, 2):
                np.add.at(rot, tr.h[fold][:, c], dth)
                np.add.at(rc, tr.h[fold][:, c], 1)
        res.add_frame(f"하중계수 {lam:.4g}", lam, U, {"barStrain": nodal / np.maximum(cnt, 1), "foldRotation": rot / np.maximum(rc, 1)})
        hist["time"].append(float(lam))
        hist["load_factor"].append(float(lam))
        hist["energy.bar"].append(eb)
        hist["energy.bend"].append(ebend)
        hist["energy.fold"].append(efold)
        hist["energy.total"].append(eb + ebend + efold)
        for c, nm in enumerate("xyz"):
            hist[f"reaction.{nm}"].append(float(r[c]))
        hist["u.max"].append(float(np.linalg.norm(U, axis=1).max()))
        hist["strain.max"].append(float(eps.max()))
        hist["fold.mean"].append(float(np.degrees(np.abs(th[fold] - np.pi)).mean()) if fold.any() else 0.0)
        for nm in gnames:
            hist[f"fold.{nm}"].append(float(np.degrees(np.abs(th[fold & (names == nm)] - np.pi)).mean()))
    res.history = hist
    tot = hist["energy.total"][-1]
    res.summary = {
        "절점 수 (중심 절점 포함)": ntot,
        "바 수": int(len(tr.bars)),
        "접힘 힌지 수": int(info["nfold"]),
        "최대 변위 (mm)": hist["u.max"][-1],
        "총 에너지 (N·mm)": tot,
        "에너지 비율 바/굽힘/접힘": [round(hist[k][-1] / tot, 4) if tot > 0 else 0.0 for k in ("energy.bar", "energy.bend", "energy.fold")],
        "평균 접힘각 처음→끝 (도, 평면=0)": [round(float(np.degrees(np.abs(tr.th0[fold] - np.pi)).mean()), 3) if fold.any() else 0.0, round(hist["fold.mean"][-1], 3)],
        "최대 바 변형률": hist["strain.max"][-1],
        "프레임 수": len(frames),
    }
    if msgs:
        res.summary["참고"] = msgs
    res.log = "\n".join(msgs)
    return res
