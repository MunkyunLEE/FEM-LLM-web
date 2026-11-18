import time

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import splu

from .core import Assembler, State, commit, energies, evaluate
from .model import FOLD


def lin_solve(K, r, dense_max=4000):
    n = K.shape[0]
    if n == 0:
        return np.zeros(0)
    if n <= dense_max:
        A = K.toarray() if sparse.issparse(K) else K
        try:
            return np.linalg.solve(A, r)
        except np.linalg.LinAlgError:
            return np.linalg.lstsq(A, r, rcond=None)[0]
    try:
        return splu(sparse.csc_matrix(K), permc_spec="MMD_AT_PLUS_A").solve(r)
    except RuntimeError:
        from scipy.sparse.linalg import lsqr

        return lsqr(K, r)[0]


class Problem:
    def __init__(self, m, fixed=None, disp=None, force=None, rest=None, contact=None, path=None):
        self.m = m
        self.path = path
        nd = m.ndof
        self.fixed = np.zeros(0, int) if fixed is None else np.unique(np.asarray(fixed, int))
        self.up = np.zeros(nd)
        for dof, v in (disp or {}).items():
            self.up[int(dof)] = float(v)
        pdofs = np.array([int(k) for k in (disp or {})] + ([int(k) for k in path[0]] if path else []), int)
        self.pres = np.unique(np.concatenate([self.fixed, pdofs])).astype(int)
        self.free = np.setdiff1d(np.arange(nd), self.pres)
        self.F = np.zeros(nd)
        for dof, v in (force or {}).items():
            self.F[int(dof)] += float(v)
        self.dth0 = None
        if rest:
            self.dth0 = np.zeros(len(m.h))
            for i, v in rest.items():
                self.dth0[int(i)] = float(v)
        self.contact = contact
        self.asm = Assembler(m)


class Recorder:
    def __init__(self, pb, watch=None):
        self.pb = pb
        self.watch = list(watch or [])
        self.lam = []
        self.U = []
        self.react = []
        self.E = []
        self.th = []
        self.M = []
        self.iters = []
        self.extwork = [0.0]
        self.contact = []

    def add(self, lam, x, st, out, T, it, cinfo=None):
        m = self.pb.m
        u = (x - m.X).ravel()
        if self.U:
            du = u - self.U[-1]
            f_prev = self._fext(self.lam[-1], self.react[-1])
            f_now = self._fext(lam, T.ravel())
            self.extwork.append(self.extwork[-1] + 0.5 * float((f_prev + f_now) @ du))
        self.lam.append(float(lam))
        self.U.append(u.copy())
        self.react.append(T.ravel().copy())
        e = energies(m, out, st)
        if cinfo:
            e["contact"] = float(cinfo.get("energy", 0.0))
        self.E.append(e)
        self.th.append(out.get("th", np.zeros(0)).copy())
        self.M.append(out.get("M", np.zeros(0)).copy())
        self.iters.append(int(it))
        self.contact.append(cinfo or {})

    def _fext(self, lam, T):
        f = lam * self.pb.F.copy()
        f[self.pb.pres] = T[self.pb.pres]
        return f


def _residual(pb, x, st, lam, need_K=True):
    T, K, out = evaluate(pb.m, pb.asm, x, st, need_K, lam, pb.dth0)
    T = T.ravel()
    cinfo = None
    if pb.contact is not None:
        g, Hc, cinfo = pb.contact.terms(x, need_K)
        T = T + g
        if need_K:
            K = K + Hc
    return T, K, out, cinfo


def _potential(pb, x, st, lam):
    _, _, out = evaluate(pb.m, pb.asm, x, st, False, lam, pb.dth0)
    e = float(out["Ub"].sum() + out["wpb"].sum())
    if "Uh" in out:
        e += float(out["Uh"].sum() + out["wph"].sum())
    if pb.contact is not None:
        e += pb.contact.terms(x, False)[2]["energy"]
    e -= lam * float(pb.F @ (x - pb.m.X).ravel())
    return e


def _set_prescribed(pb, x, lam):
    u = (x - pb.m.X).ravel()
    u[pb.pres] = lam * pb.up[pb.pres]
    if pb.path is not None:
        u[np.asarray(pb.path[0], int)] = np.asarray(pb.path[1](lam), float)
    return (pb.m.X.ravel() + u).reshape(-1, 3)


def solve_displacement(m, pb=None, steps=50, tol=1e-8, kmax=30, beta_min=1e-4, c_min=0.2, max_fail=40, progress=None, watch=None, **kw):
    pb = pb or Problem(m, **kw)
    st = State(m)
    x = m.X.copy()
    rec = Recorder(pb, watch)
    T, _, out, ci = _residual(pb, x, st, 0.0, False)
    rec.add(0.0, x, st, out, T, 0, ci)
    lam = 0.0
    h0 = 1.0 / max(1, steps)
    beta = 1.0
    c = 1.0
    fails = 0
    v_prev = None
    t0 = time.time()
    scale = float(np.abs(pb.F).max() + 1.0)
    notes = []
    while lam < 1.0 - 1e-12:
        h = min(beta * h0, 1.0 - lam)
        lam1 = lam + h
        xi = _set_prescribed(pb, x, lam1)
        if v_prev is not None:
            xi = xi.ravel()
            xi[pb.free] += v_prev[pb.free] * h
            xi = xi.reshape(-1, 3)
            if pb.contact is not None:
                a = pb.contact.stepsize(x, xi)
                if a < 1.0:
                    xi = (x.ravel() + 0.8 * a * (xi.ravel() - x.ravel())).reshape(-1, 3)
                    xi = _set_prescribed(pb, xi, lam1)
        ok = False
        k = 0
        for k in range(1, kmax + 1):
            T, K, out, ci = _residual(pb, xi, st, lam1)
            R = T - lam1 * pb.F
            Rf = R[pb.free]
            if len(pb.free) == 0:
                ok = True
                break
            Kff = K[pb.free][:, pb.free]
            du = lin_solve(Kff, -Rf)
            if not np.all(np.isfinite(du)):
                break
            step = c * du
            xn = xi.ravel().copy()
            xn[pb.free] += step
            sz = pb.contact.stepsize(xi, xn.reshape(-1, 3)) if pb.contact is not None else 1.0
            a0 = 1.0 if sz >= 1.0 else 0.9 * sz
            a = a0
            slope = float(Rf @ step)
            if slope < 0:
                e0 = _potential(pb, xi, st, lam1)
                for _ in range(14):
                    xt = xi.ravel() + a * (xn - xi.ravel())
                    if _potential(pb, xt.reshape(-1, 3), st, lam1) <= e0 + 1e-4 * a * slope + 1e-9 * abs(e0):
                        break
                    a *= 0.5
                else:
                    a = a0
            step = a * step
            xn = xi.ravel() + a * (xn - xi.ravel())
            xi = xn.reshape(-1, 3)
            ref = max(float(np.abs(xi - m.X).max()), float(np.abs(m.X).max()) * 1e-6, 1e-12)
            rn = float(np.abs(Rf).max()) if len(Rf) else 0.0
            if np.abs(step).max() <= tol * ref * 1e3 and rn <= tol * max(scale, float(np.abs(T).max())) * 1e2:
                ok = True
                break
            if np.abs(du).max() <= tol * ref:
                ok = True
                break
        if ok:
            T, _, out, ci = _residual(pb, xi, st, lam1, False)
            st = commit(st, out)
            if pb.contact is not None:
                pb.contact.commit(xi)
            v_prev = (xi.ravel() - x.ravel()) / h
            x = xi
            lam = lam1
            rec.add(lam, x, st, out, T, k, ci)
            fails = 0
            c = 1.0
            beta = min(1.1 * beta, 1.0) if beta < 1 else max(0.9 * beta, 1.0)
            if progress:
                progress(lam, k)
        else:
            fails += 1
            if fails >= max_fail:
                notes.append(f"λ={lam:.4g} 에서 수렴하지 못해 멈췄습니다")
                break
            beta = max(beta / 2, beta_min)
            c = max(0.75 * c, c_min)
            v_prev = None if fails > 2 else v_prev
    rec.time = time.time() - t0
    rec.notes = notes
    rec.state = st
    rec.x = x
    return rec


def solve_force(m, pb=None, increments=100, dlam=0.05, tol=1e-8, kmax=30, lam_max=None, stop=None, progress=None, **kw):
    pb = pb or Problem(m, **kw)
    st = State(m)
    x = m.X.copy()
    rec = Recorder(pb)
    T, _, out, ci = _residual(pb, x, st, 0.0, False)
    rec.add(0.0, x, st, out, T, 0, ci)
    lam = 0.0
    free = pb.free
    F = pb.F[free]
    du1 = None
    du_hat_prev = None
    sgn = 1.0
    t0 = time.time()
    notes = []
    radius = 1.0
    for i in range(1, increments + 1):
        xi = x.ravel().copy()
        lam_i = lam
        dlam_tot = 0.0
        ok = False
        for k in range(1, kmax + 1):
            T, K, out, ci = _residual(pb, xi.reshape(-1, 3), st, lam_i)
            R = lam_i * pb.F - T
            Kff = K[free][:, free]
            dh = lin_solve(Kff, F)
            dr = lin_solve(Kff, R[free])
            if k == 1:
                if du1 is None:
                    dl = dlam * radius
                    du1 = dh.copy()
                else:
                    sgn = np.sign(du_hat_prev @ dh) if du_hat_prev is not None else 1.0
                    dl = sgn * dlam * radius * np.sqrt(abs((du1 @ du1) / (dh @ dh)))
                dh1 = dh.copy()
            else:
                dl = -(dh1 @ dr) / (dh1 @ dh)
            dd = dl * dh + dr
            if not np.all(np.isfinite(dd)):
                break
            if pb.contact is not None:
                xn = xi.copy()
                xn[free] += dd
                a = pb.contact.stepsize(xi.reshape(-1, 3), xn.reshape(-1, 3))
                if a < 1.0:
                    dd = 0.8 * a * dd
                    dl = 0.8 * a * dl
            xi[free] += dd
            lam_i += dl
            dlam_tot += dl
            ref = max(float(np.abs(xi.reshape(-1, 3) - m.X).max()), 1e-12)
            if k > 1 and np.abs(dr).max() <= tol * ref:
                ok = True
                break
        if not ok:
            radius *= 0.5
            if radius < 1e-4:
                notes.append(f"{i} 번째 증분에서 수렴하지 못해 멈췄습니다")
                break
            continue
        du_hat_prev = dh1
        T, _, out, ci = _residual(pb, xi.reshape(-1, 3), st, lam_i, False)
        st = commit(st, out)
        if pb.contact is not None:
            pb.contact.commit(xi)
        x = xi.reshape(-1, 3)
        lam = lam_i
        rec.add(lam, x, st, out, T, k, ci)
        if k <= 3:
            radius = min(radius * 1.25, 4.0)
        elif k > 8:
            radius *= 0.7
        if progress:
            progress(i / increments, k)
        if lam_max is not None and abs(lam) >= lam_max:
            break
        if stop is not None and stop(x, lam):
            break
    rec.time = time.time() - t0
    rec.notes = notes
    rec.state = st
    rec.x = x
    return rec


def fold_angles(rec, m):
    idx = np.where(m.hkind == FOLD)[0]
    return np.array([t[idx] for t in rec.th]), [m.hedge[i] for i in idx]
