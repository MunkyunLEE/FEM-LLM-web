import numpy as np
from scipy import linalg

from .core import Assembler, State, evaluate
from .geom import dihedral, dihedral_grad, unwrap
from .model import FOLD


def mass_matrix(m):
    return np.repeat(m.mass, 3)


def modal(m, x=None, state=None, fixed=None, n=12, use_mass=True):
    x = m.X.copy() if x is None else np.asarray(x, float)
    st = state or State(m)
    _, K, _ = evaluate(m, Assembler(m), x, st, True)
    K = K.toarray()
    K = 0.5 * (K + K.T)
    fixed = np.zeros(0, int) if fixed is None else np.unique(np.asarray(fixed, int))
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    Kf = K[np.ix_(free, free)]
    if use_mass:
        Md = mass_matrix(m)[free]
        if np.any(Md <= 0):
            raise ValueError("질량이 0 인 자유도가 있습니다. 밀도를 확인하세요")
        s = 1.0 / np.sqrt(Md)
        lam, v = linalg.eigh(Kf * s[:, None] * s[None, :])
        v = v * s[:, None]
    else:
        lam, v = linalg.eigh(Kf)
    k = min(n, len(lam))
    shapes = np.zeros((k, m.ndof))
    shapes[:, free] = v[:, :k].T
    mx = np.abs(shapes).max(axis=1, keepdims=True)
    shapes = shapes / np.where(mx > 0, mx, 1.0)
    rigid = int((np.abs(lam) < 1e-12 * max(np.abs(lam).max(), 1e-30)).sum())
    out = {"eig": lam[:k], "shapes": shapes.reshape(k, -1, 3), "rigid": rigid, "use_mass": use_mass}
    if use_mass:
        out["freq"] = np.sqrt(np.clip(lam[:k], 0, None)) / (2 * np.pi)
    return out


def _constraints(m, x, rigid_panels=True):
    d = x[m.bars[:, 1]] - x[m.bars[:, 0]]
    L = np.linalg.norm(d, axis=1)
    n = d / L[:, None]
    nb = len(m.bars)
    rows = [np.repeat(np.arange(nb), 6)]
    cols = [np.concatenate([3 * m.bars[:, :1] + np.arange(3), 3 * m.bars[:, 1:] + np.arange(3)], axis=1).ravel()]
    vals = [np.concatenate([-n, n], axis=1).ravel()]
    g = [L - m.L0]
    hid = np.where(m.hkind != FOLD)[0] if rigid_panels else np.zeros(0, int)
    if len(hid):
        hh = m.h[hid]
        xs = [x[hh[:, q]] for q in range(4)]
        th = unwrap(dihedral(*xs), m.th0[hid])
        G = dihedral_grad(*xs)
        r0 = nb
        rows.append(np.repeat(np.arange(len(hid)) + r0, 12))
        cols.append((3 * hh[:, :, None] + np.arange(3)).reshape(len(hid), 12).ravel())
        vals.append(G.ravel())
        g.append(th - m.th0[hid])
    nr = nb + len(hid)
    J = np.zeros((nr, m.ndof))
    np.add.at(J, (np.concatenate(rows), np.concatenate(cols)), np.concatenate(vals))
    return np.concatenate(g), J


def _rigid_fix(m):
    f = m.faces[0]
    a, b, c = (int(i) for i in f[:3])
    return np.array([3 * a, 3 * a + 1, 3 * a + 2, 3 * b + 1, 3 * b + 2, 3 * c + 2])


def dof_analysis(m, x=None, fixed=None, rigid_panels=True, rtol=1e-7):
    x = m.X.copy() if x is None else np.asarray(x, float)
    _, J = _constraints(m, x, rigid_panels)
    fixed = np.zeros(0, int) if fixed is None else np.unique(np.asarray(fixed, int))
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    Jf = J[:, free]
    s = linalg.svdvals(Jf) if Jf.size else np.zeros(0)
    tol = rtol * (s.max() if len(s) else 1.0)
    r = int((s > tol).sum())
    null = len(free) - r
    rigid = 6 if len(fixed) == 0 else 0
    return {"constraints": int(J.shape[0]), "free_dofs": int(len(free)), "rank": r, "nullity": int(null), "mechanisms": int(max(null - rigid, 0)), "self_stress": int(J.shape[0] - r), "rigid_removed": rigid}


def _fold_grads(m, x, idx):
    hh = m.h[idx]
    xs = [x[hh[:, q]] for q in range(4)]
    G = dihedral_grad(*xs)
    A = np.zeros((len(idx), m.ndof))
    for r in range(len(idx)):
        A[r, (3 * hh[r][:, None] + np.arange(3)).ravel()] = G[r]
    return A


def fold_angles(m, x, ref=None):
    xs = [x[m.h[:, q]] for q in range(4)]
    th = dihedral(*xs)
    return unwrap(th, m.th_init if ref is None else ref)


def rigid_fold_path(m, drivers, targets, steps=100, fixed=None, x0=None, tol=1e-10, itmax=30):
    x = (m.X if x0 is None else x0).copy()
    drivers = [int(i) for i in drivers]
    fixed = _rigid_fix(m) if fixed is None or len(fixed) == 0 else np.unique(np.asarray(fixed, int))
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    th_start = fold_angles(m, x)[drivers]
    targets = np.asarray(targets, float)
    path = [fold_angles(m, x)]
    xs = [x.copy()]
    err = []
    ref = path[0].copy()
    tol = tol * max(1.0, float(np.mean(m.L0)))
    for s in range(1, steps + 1):
        goal = th_start + (targets - th_start) * s / steps
        _, J = _constraints(m, x)
        A = _fold_grads(m, x, drivers)
        cur = fold_angles(m, x, ref)[drivers]
        M = np.vstack([J[:, free], A[:, free]])
        rhs = np.concatenate([np.zeros(J.shape[0]), goal - cur])
        v = np.linalg.lstsq(M, rhs, rcond=None)[0]
        xv = x.ravel().copy()
        xv[free] += v
        x = xv.reshape(-1, 3)
        rprev = np.inf
        for _ in range(itmax):
            g, J = _constraints(m, x)
            A = _fold_grads(m, x, drivers)
            cur = fold_angles(m, x, ref)[drivers]
            r = np.concatenate([g, cur - goal])
            rn = np.abs(r).max()
            if rn < tol or (_ > 1 and rn < 1e3 * tol and rn > 0.5 * rprev):
                break
            rprev = rn
            M = np.vstack([J[:, free], A[:, free]])
            dx = np.linalg.lstsq(M, -r, rcond=None)[0]
            xv = x.ravel().copy()
            xv[free] += dx
            x = xv.reshape(-1, 3)
        ref = fold_angles(m, x, ref)
        path.append(ref.copy())
        xs.append(x.copy())
        err.append(float(np.abs(r).max()))
    return {"angles": np.array(path), "x": np.array(xs), "err": np.array(err)}


def null_modes(m, x=None, fixed=None):
    x = m.X.copy() if x is None else np.asarray(x, float)
    _, J = _constraints(m, x)
    fixed = _rigid_fix(m) if fixed is None or len(fixed) == 0 else np.unique(np.asarray(fixed, int))
    free = np.setdiff1d(np.arange(m.ndof), fixed)
    U, s, Vt = linalg.svd(J[:, free])
    tol = 1e-7 * (s.max() if len(s) else 1.0)
    r = int((s > tol).sum())
    N = Vt[r:].T
    modes = np.zeros((N.shape[1], m.ndof))
    modes[:, free] = N.T
    fi = np.where(m.hkind == FOLD)[0]
    rates = np.array([_fold_grads(m, x, fi) @ v for v in modes]) if len(modes) else np.zeros((0, len(fi)))
    return modes, rates, fi


def solution_space(m, edges3, ndir=24, amp=np.pi * 0.9, steps=60, fixed=None):
    modes, rates, fi = null_modes(m, fixed=fixed)
    k = len(modes)
    if k == 0:
        return {"dof": 0, "paths": []}
    hid = [m.fold_index(e) for e in edges3]
    col = [list(fi).index(h) for h in hid]
    paths = []
    rng = np.random.default_rng(0)
    if k == 1:
        dirs = [np.array([1.0]), np.array([-1.0])]
    else:
        dirs = [d / np.linalg.norm(d) for d in rng.normal(size=(ndir, k))]
    th0 = fold_angles(m, m.X)
    for d in dirs:
        rate = d @ rates
        j = int(np.argmax(np.abs(rate)))
        drv = [fi[j]]
        if k > 1:
            order = np.argsort(-np.abs(rates).sum(axis=0))
            drv = []
            for c in order:
                cand = drv + [fi[c]]
                sub = rates[:, [list(fi).index(h) for h in cand]]
                if np.linalg.matrix_rank(sub) == len(cand):
                    drv = cand
                if len(drv) == k:
                    break
        dr_rates = np.array([d @ rates[:, list(fi).index(h)] for h in drv])
        if np.abs(dr_rates).max() == 0:
            continue
        tgt = th0[drv] + amp * dr_rates / np.abs(dr_rates).max()
        tgt = np.clip(tgt, 1e-3, 2 * np.pi - 1e-3)
        try:
            p = rigid_fold_path(m, drv, tgt, steps=steps, fixed=fixed)
        except Exception:
            continue
        ok = p["err"] < 1e-6
        n_ok = int(np.argmin(ok)) if not ok.all() else len(ok)
        paths.append(p["angles"][: n_ok + 1][:, hid])
    return {"dof": k, "paths": paths, "edges": [tuple(e) for e in edges3]}
