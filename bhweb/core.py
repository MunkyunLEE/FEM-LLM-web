import numpy as np
from scipy import sparse

from .geom import dihedral, dihedral_grad, dihedral_hess, unwrap
from .model import FOLD


class State:
    def __init__(self, m):
        self.ep = np.zeros(len(m.bars))
        self.ab = np.zeros(len(m.bars))
        self.thp = m.th0.copy()
        self.ah = np.zeros(len(m.h))
        self.th = m.th_init.copy()
        self.wpb = np.zeros(len(m.bars))
        self.wph = np.zeros(len(m.h))

    def copy(self):
        s = State.__new__(State)
        for k, v in self.__dict__.items():
            setattr(s, k, v.copy())
        return s


class Assembler:
    def __init__(self, m):
        self.m = m
        nd = m.ndof
        bi = np.concatenate([3 * m.bars[:, :1] + np.arange(3), 3 * m.bars[:, 1:] + np.arange(3)], axis=1)
        hi = (3 * m.h[:, :, None] + np.arange(3)).reshape(len(m.h), 12)
        r = [np.repeat(bi, 6, axis=1).ravel(), np.repeat(hi, 12, axis=1).ravel()]
        c = [np.tile(bi, 6).ravel(), np.tile(hi, 12).ravel()]
        rows = np.concatenate(r)
        cols = np.concatenate(c)
        key = rows.astype(np.int64) * nd + cols
        uk, inv = np.unique(key, return_inverse=True)
        self.inv = inv
        self.ur = (uk // nd).astype(np.int64)
        self.uc = (uk % nd).astype(np.int64)
        self.nu = len(uk)
        self.bi = bi
        self.hi = hi
        self.nd = nd

    def matrix(self, vals):
        data = np.bincount(self.inv, weights=vals, minlength=self.nu)
        return sparse.csr_matrix((data, (self.ur, self.uc)), shape=(self.nd, self.nd))


def bar_response(m, x, st):
    d = x[m.bars[:, 1]] - x[m.bars[:, 0]]
    L = np.linalg.norm(d, axis=1)
    lam = L / m.L0
    eps = np.log(lam)
    ee = eps - st.ep
    tau = m.Eb * ee
    f = np.abs(tau) - (m.Yb + m.Hb * st.ab)
    pl = f > 0
    dg = np.where(pl, f / (m.Eb + m.Hb), 0.0)
    sg = np.sign(tau)
    ee = ee - dg * sg
    tau = m.Eb * ee
    Cb = np.where(pl, np.maximum(m.Eb * m.Hb / (m.Eb + m.Hb), 1e-3 * m.Eb), m.Eb)
    new = (eps - ee, st.ab + dg)
    S = tau / lam**2
    C = (Cb - 2 * tau) / lam**4
    U = m.A * m.L0 * 0.5 * m.Eb * ee**2
    wp = m.A * m.L0 * np.abs(tau) * dg
    return d, S, C, new, U, wp, pl


def hinge_response(m, x, st, lam=0.0, dth0=None):
    xi, xj, xk, xl = (x[m.h[:, q]] for q in range(4))
    th = unwrap(dihedral(xi, xj, xk, xl), st.th)
    th0 = m.th0 if dth0 is None else m.th0 + lam * dth0
    thp = st.thp if dth0 is None else st.thp + lam * dth0
    Mtr = m.hk * (th - thp)
    f = np.abs(Mtr) - (m.hY + m.hH * st.ah)
    pl = f > 0
    dg = np.where(pl, f / (m.hk + m.hH), 0.0)
    the = th - thp + th0 - dg * np.sign(Mtr)
    M = m.hk * (the - th0)
    Ch = np.where(pl, np.maximum(m.hk * m.hH / (m.hk + m.hH), 1e-3 * m.hk), m.hk)
    new_thp = th - the + th0
    if dth0 is not None:
        new_thp = new_thp - lam * dth0
    U = 0.5 * m.hk * (the - th0) ** 2
    wp = np.abs(M) * dg
    return th, M, Ch, (new_thp, st.ah + dg), U, wp, pl


def evaluate(m, asm, x, st, need_K=True, lam=0.0, dth0=None):
    T = np.zeros_like(x)
    d, S, C, bnew, Ub, wpb, plb = bar_response(m, x, st)
    fb = (m.A * S / m.L0)[:, None] * d
    np.add.at(T, m.bars[:, 1], fb)
    np.add.at(T, m.bars[:, 0], -fb)
    out = {"bar": bnew, "Ub": Ub, "wpb": wpb, "plb": plb}
    if len(m.h):
        xi, xj, xk, xl = (x[m.h[:, q]] for q in range(4))
        th, M, Ch, hnew, Uh, wph, plh = hinge_response(m, x, st, lam, dth0)
        g = dihedral_grad(xi, xj, xk, xl)
        fh = (M[:, None] * g).reshape(-1, 4, 3)
        for q in range(4):
            np.add.at(T, m.h[:, q], fh[:, q])
        out.update(th=th, M=M, hinge=hnew, Uh=Uh, wph=wph, plh=plh)
    K = None
    if need_K:
        nn = d / np.linalg.norm(d, axis=1)[:, None]
        L = np.linalg.norm(d, axis=1)
        kb = (m.A * m.L0 * C / m.L0**4 * L**2)[:, None, None] * np.einsum("ni,nj->nij", nn, nn) + (m.A * S / m.L0)[:, None, None] * np.eye(3)[None]
        Kb = np.zeros((len(d), 6, 6))
        Kb[:, :3, :3] = kb
        Kb[:, 3:, 3:] = kb
        Kb[:, :3, 3:] = -kb
        Kb[:, 3:, :3] = -kb
        vals = [Kb.ravel()]
        if len(m.h):
            Hh = dihedral_hess(xi, xj, xk, xl)
            Kh = Ch[:, None, None] * np.einsum("ni,nj->nij", g, g) + M[:, None, None] * Hh
            vals.append(Kh.ravel())
        else:
            vals.append(np.zeros(0))
        K = asm.matrix(np.concatenate(vals))
    return T, K, out


def commit(st, out):
    s = st.copy()
    s.ep, s.ab = out["bar"]
    s.wpb = st.wpb + out["wpb"]
    if "hinge" in out:
        s.thp, s.ah = out["hinge"]
        s.th = out["th"]
        s.wph = st.wph + out["wph"]
    return s


def energies(m, out, st):
    fold = m.hkind == FOLD
    Uh = out.get("Uh", np.zeros(0))
    return {
        "stretch": float(out["Ub"].sum()),
        "fold": float(Uh[fold].sum()) if len(Uh) else 0.0,
        "bend": float(Uh[~fold].sum()) if len(Uh) else 0.0,
        "plastic_bar": float(st.wpb.sum()),
        "plastic_fold": float(st.wph[fold].sum()) if len(st.wph) else 0.0,
        "plastic_bend": float(st.wph[~fold].sum()) if len(st.wph) else 0.0,
    }
