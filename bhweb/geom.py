import numpy as np

_E = np.eye(3)


def _seed(p, q):
    d = np.zeros((3, 12))
    for c in range(3):
        d[c, 3 * p + c] += 1.0
        if q is not None:
            d[c, 3 * q + c] -= 1.0
    return d


_DIJ = _seed(0, 1)
_DKJ = _seed(2, 1)
_DKL = _seed(2, 3)


def _cross(a, da, b, db):
    v = np.cross(a, b)
    d = np.cross(da, b[:, :, None], axis=1) + np.cross(a[:, :, None], db, axis=1)
    return v, d


def _dot(a, da, b, db):
    v = np.einsum("nc,nc->n", a, b)
    d = np.einsum("nc,ncq->nq", b, da) + np.einsum("nc,ncq->nq", a, db)
    return v, d


def _sv(s, ds, v, dv):
    return s[:, None] * v, ds[:, None, :] * v[:, :, None] + s[:, None, None] * dv


def dihedral(xi, xj, xk, xl):
    rij, rkj, rkl = xi - xj, xk - xj, xk - xl
    m = np.cross(rij, rkj)
    n = np.cross(rkj, rkl)
    lk = np.linalg.norm(rkj, axis=1)
    s = np.einsum("ij,ij->i", m, rkl)
    th = np.arctan2(lk * s, np.einsum("ij,ij->i", m, n))
    return np.where(th < 0, th + 2 * np.pi, th)


def dihedral_grad(xi, xj, xk, xl):
    rij, rkj, rkl = xi - xj, xk - xj, xk - xl
    m = np.cross(rij, rkj)
    n = np.cross(rkj, rkl)
    lk2 = np.einsum("ij,ij->i", rkj, rkj)
    lk = np.sqrt(lk2)
    mm = np.einsum("ij,ij->i", m, m)
    nn = np.einsum("ij,ij->i", n, n)
    di = (lk / mm)[:, None] * m
    dl = -(lk / nn)[:, None] * n
    a = (np.einsum("ij,ij->i", rij, rkj) / lk2)[:, None]
    b = (np.einsum("ij,ij->i", rkl, rkj) / lk2)[:, None]
    dj = (a - 1) * di - b * dl
    dk = (b - 1) * dl - a * di
    return np.stack([di, dj, dk, dl], axis=1).reshape(len(xi), 12)


def dihedral_hess(xi, xj, xk, xl):
    N = len(xi)
    rij, rkj, rkl = xi - xj, xk - xj, xk - xl
    d_rij = np.broadcast_to(_DIJ, (N, 3, 12))
    d_rkj = np.broadcast_to(_DKJ, (N, 3, 12))
    d_rkl = np.broadcast_to(_DKL, (N, 3, 12))
    m, dm = _cross(rij, d_rij, rkj, d_rkj)
    n, dn = _cross(rkj, d_rkj, rkl, d_rkl)
    lk2, dlk2 = _dot(rkj, d_rkj, rkj, d_rkj)
    lk = np.sqrt(lk2)
    dlk = dlk2 / (2 * lk[:, None])
    mm, dmm = _dot(m, dm, m, dm)
    nn, dnn = _dot(n, dn, n, dn)
    ci = lk / mm
    dci = dlk / mm[:, None] - (lk / mm**2)[:, None] * dmm
    cl = -lk / nn
    dcl = -dlk / nn[:, None] + (lk / nn**2)[:, None] * dnn
    di, ddi = _sv(ci, dci, m, dm)
    dl, ddl = _sv(cl, dcl, n, dn)
    pa, dpa = _dot(rij, d_rij, rkj, d_rkj)
    pb, dpb = _dot(rkl, d_rkl, rkj, d_rkj)
    a = pa / lk2
    da = dpa / lk2[:, None] - (pa / lk2**2)[:, None] * dlk2
    b = pb / lk2
    db = dpb / lk2[:, None] - (pb / lk2**2)[:, None] * dlk2
    t1, dt1 = _sv(a - 1, da, di, ddi)
    t2, dt2 = _sv(b, db, dl, ddl)
    ddj = dt1 - dt2
    t3, dt3 = _sv(b - 1, db, dl, ddl)
    t4, dt4 = _sv(a, da, di, ddi)
    ddk = dt3 - dt4
    H = np.concatenate([ddi, ddj, ddk, ddl], axis=1)
    return 0.5 * (H + H.transpose(0, 2, 1))


def unwrap(th, ref):
    return th - 2 * np.pi * np.round((th - ref) / (2 * np.pi))
