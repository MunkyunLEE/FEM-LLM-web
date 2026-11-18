import numpy as np
from scipy import sparse

from .geom import dihedral, dihedral_grad, dihedral_hess, unwrap


def _bar(s, sh):
    s = np.maximum(s, 1e-300)
    r = s - sh
    b = -(r**2) * np.log(s / sh)
    b1 = -2 * r * np.log(s / sh) - r**2 / s
    b2 = -2 * np.log(s / sh) - 4 * r / s + r**2 / s**2
    on = s < sh
    return np.where(on, b, 0.0), np.where(on, b1, 0.0), np.where(on, b2, 0.0)


class GeometricContact:
    def __init__(self, m, gap=None, kappa=None, rel_gap=1e-3):
        import ipctk

        self.ipc = ipctk
        self.m = m
        Lbar = float(np.mean(m.L0))
        self.gap = float(gap) if gap else rel_gap * Lbar
        self.dhat = self.gap
        self.dmin = self.gap
        F = np.asarray(m.tris, np.int32)
        E = ipctk.edges(F)
        self.mesh = ipctk.CollisionMesh(np.asfortranarray(m.X), E, F)
        k = float(np.median(m.hk)) if len(m.hk) else float(np.median(m.A * m.Eb / m.L0)) * Lbar**2
        self.kappa = float(kappa) if kappa else 1e3 * k / Lbar**2
        self.B = ipctk.BarrierPotential(self.dhat, self.kappa, True)
        self.ccd = ipctk.AdditiveCCD(10000, 0.9)
        c = ipctk.NormalCollisions()
        V = np.asfortranarray(m.X)
        c.build(self.mesh, V, self.dhat, self.dmin)
        self.rest_pairs = len(c)
        self._key = None
        self._c = None
        X = m.X
        hh = m.h
        ax = X[hh[:, 2]] - X[hh[:, 1]]
        ax = ax / np.linalg.norm(ax, axis=1)[:, None]

        def height(w):
            v = X[w] - X[hh[:, 1]]
            return np.linalg.norm(v - np.einsum("ij,ij->i", v, ax)[:, None] * ax, axis=1)

        self.hh = np.minimum(height(hh[:, 0]), height(hh[:, 3])) if len(hh) else np.zeros(0)
        self.th_ref = m.th_init.copy()

    def _cols(self, V):
        key = V.tobytes()
        if key != self._key:
            c = self.ipc.NormalCollisions()
            c.build(self.mesh, V, self.dhat, self.dmin)
            self._key, self._c = key, c
        return self._c

    def _hinge_d(self, x):
        xs = [x[self.m.h[:, q]] for q in range(4)]
        th = unwrap(dihedral(*xs), self.th_ref)
        lo = self.hh * th
        hi = self.hh * (2 * np.pi - th)
        return xs, th, lo, hi

    def hinge_terms(self, x, need_K=True):
        m = self.m
        nd = m.ndof
        if not len(m.h):
            return np.zeros(nd), (sparse.csr_matrix((nd, nd)) if need_K else None), 0.0, 0, float("inf")
        xs, th, lo, hi = self._hinge_d(x)
        dmin, sh = self.dmin, self.dhat
        b_lo, b1_lo, b2_lo = _bar(lo - dmin, sh)
        b_hi, b1_hi, b2_hi = _bar(hi - dmin, sh)
        act = (b1_lo != 0) | (b1_hi != 0)
        e = float(self.kappa * (b_lo.sum() + b_hi.sum()))
        g = np.zeros(nd)
        H = None
        idx = np.where(act)[0]
        if len(idx):
            G = dihedral_grad(*(q[idx] for q in xs))
            d1 = self.kappa * (b1_lo[idx] - b1_hi[idx]) * self.hh[idx]
            d2 = self.kappa * (b2_lo[idx] + b2_hi[idx]) * self.hh[idx] ** 2
            dof = (3 * m.h[idx][:, :, None] + np.arange(3)).reshape(len(idx), 12)
            np.add.at(g, dof.ravel(), (d1[:, None] * G).ravel())
            if need_K:
                Hh = dihedral_hess(*(q[idx] for q in xs))
                Kh = d2[:, None, None] * np.einsum("ni,nj->nij", G, G) + d1[:, None, None] * Hh
                H = sparse.csr_matrix((Kh.ravel(), (np.repeat(dof, 12, axis=1).ravel(), np.tile(dof, 12).ravel())), shape=(nd, nd))
        if need_K and H is None:
            H = sparse.csr_matrix((nd, nd))
        dmn = float(np.minimum(lo, hi).min())
        return g, H, e, int(len(idx)), dmn

    def terms(self, x, need_K=True):
        x = np.asarray(x, float).reshape(-1, 3)
        V = np.asfortranarray(x)
        c = self._cols(V)
        nd = V.size
        gh, Hh, eh, nh, dh = self.hinge_terms(x, need_K)
        info = {"pairs": int(len(c)), "hinge_pairs": nh, "energy": eh, "min_dist": float("inf"), "hinge_min": dh}
        if not len(c):
            return gh, Hh, info
        g = self.B.gradient(c, self.mesh, V) + gh
        H = (sparse.csr_matrix(self.B.hessian(c, self.mesh, V)) + Hh) if need_K else None
        info["energy"] = eh + float(self.B(c, self.mesh, V))
        info["min_dist"] = float(np.sqrt(max(c.compute_minimum_distance(self.mesh, V), 0.0)))
        info["force"] = float(0.5 * np.linalg.norm(g.reshape(-1, 3), axis=1).sum())
        return g, H, info

    def commit(self, x):
        xs = [np.asarray(x, float).reshape(-1, 3)[self.m.h[:, q]] for q in range(4)]
        self.th_ref = unwrap(dihedral(*xs), self.th_ref)

    def hinge_step(self, x0, x1):
        if not len(self.m.h):
            return 1.0
        x0 = np.asarray(x0, float).reshape(-1, 3)
        x1 = np.asarray(x1, float).reshape(-1, 3)

        def ok(a):
            _, th, lo, hi = self._hinge_d(x0 + a * (x1 - x0))
            return np.minimum(lo, hi).min() > self.dmin

        if ok(1.0):
            n = 8
            for k in range(1, n):
                if not ok(k / n):
                    break
            else:
                return 1.0
        lo_a, hi_a = 0.0, 1.0
        for _ in range(30):
            mid = 0.5 * (lo_a + hi_a)
            if ok(mid):
                lo_a = mid
            else:
                hi_a = mid
        return lo_a

    def stepsize(self, x0, x1):
        V0 = np.asfortranarray(np.asarray(x0, float).reshape(-1, 3))
        V1 = np.asfortranarray(np.asarray(x1, float).reshape(-1, 3))
        a = float(self.ipc.compute_collision_free_stepsize(self.mesh, V0, V1, self.dmin, narrow_phase_ccd=self.ccd))
        return min(a, self.hinge_step(V0, V1))

    def intersects(self, x):
        V = np.asfortranarray(np.asarray(x, float).reshape(-1, 3))
        return bool(self.ipc.has_intersections(self.mesh, V))
