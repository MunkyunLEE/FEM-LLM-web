import copy
import os

import numpy as np
import scipy.sparse as sp
import torch

from .shell import GAUSS5, Hardening, _return_map, set_threads

GP = 1.0 / 3.0**0.5
XI = np.array([[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]])


def _shape(xi, eta):
    N = 0.25 * (1 + XI[:, 0] * xi) * (1 + XI[:, 1] * eta)
    dN = np.stack([0.25 * XI[:, 0] * (1 + XI[:, 1] * eta), 0.25 * XI[:, 1] * (1 + XI[:, 0] * xi)], axis=1)
    return N, dN


def _rot(th, v):
    t2 = (th * th).sum(-1, keepdim=True)
    a = 1 - t2 / 6 * (1 - t2 / 20 * (1 - t2 / 42 * (1 - t2 / 72 * (1 - t2 / 110 * (1 - t2 / 156)))))
    b = 0.5 * (1 - t2 / 12 * (1 - t2 / 30 * (1 - t2 / 56 * (1 - t2 / 90 * (1 - t2 / 132 * (1 - t2 / 182))))))
    c1 = torch.linalg.cross(th, v, dim=-1)
    return v + a * c1 + b * torch.linalg.cross(th, c1, dim=-1)


class QuadShell:
    def __init__(self, X, quads, E, nu, t, rot_fixed=None, plastic=None, kink_deg=20.0):
        set_threads()
        X = np.asarray(X, float)
        Q = np.asarray(quads, int)
        ne = len(Q)
        self.n = len(X)
        self.X = X
        self.quads = Q
        self.nx = 3 * self.n
        self.ndof = 6 * self.n
        tt = np.ones(ne) * np.asarray(t, float)
        P = X[Q]
        d1 = P[:, 2] - P[:, 0]
        d2 = P[:, 3] - P[:, 1]
        nel = np.cross(d1, d2)
        nel /= np.linalg.norm(nel, axis=1)[:, None]
        navg = np.zeros((self.n, 3))
        np.add.at(navg, Q.ravel(), np.repeat(nel, 4, axis=0))
        ln = np.linalg.norm(navg, axis=1)
        navg = navg / np.maximum(ln, 1e-30)[:, None]
        cosk = np.cos(np.radians(kink_deg))
        mind = np.ones(self.n)
        np.minimum.at(mind, Q.ravel(), (np.repeat(nel, 4, axis=0) * navg[Q.ravel()]).sum(1))
        smooth = mind >= cosk
        self.smooth = smooth
        D = np.where(smooth[Q][:, :, None], navg[Q], nel[:, None, :])
        cornerN = np.zeros((ne, 4, 3))
        for i in range(4):
            e1 = P[:, (i + 1) % 4] - P[:, i]
            e2 = P[:, (i + 3) % 4] - P[:, i]
            c = np.cross(e1, e2)
            cornerN[:, i] = c / np.linalg.norm(c, axis=1)[:, None]
        D = np.where(smooth[Q][:, :, None], navg[Q], cornerN)
        self.D0 = D.copy()
        gps = [(-GP, -GP), (GP, -GP), (GP, GP), (-GP, GP)]
        Ng = np.zeros((4, 4))
        dNg = np.zeros((4, 4, 2))
        for g, (a, b) in enumerate(gps):
            Ng[g], dNg[g] = _shape(a, b)
        Nc, dNc = _shape(0.0, 0.0)

        def frame(dN):
            A = np.einsum("ia,eik->eak", dN, P)
            nrm = np.cross(A[:, 0], A[:, 1])
            det = np.linalg.norm(nrm, axis=1)
            e1 = A[:, 0] / np.linalg.norm(A[:, 0], axis=1)[:, None]
            n3 = nrm / det[:, None]
            e2 = np.cross(n3, e1)
            J = np.stack([np.stack([(A[:, 0] * e1).sum(1), (A[:, 0] * e2).sum(1)], 1), np.stack([(A[:, 1] * e1).sum(1), (A[:, 1] * e2).sum(1)], 1)], 1)
            return A, det, np.linalg.inv(J)

        Jinv = np.zeros((ne, 4, 2, 2))
        wA = np.zeros((ne, 4))
        Acov = np.zeros((ne, 4, 2, 2))
        Bref = np.zeros((ne, 4, 2, 2))
        for g in range(4):
            A, det, Ji = frame(dNg[g])
            Jinv[:, g] = Ji
            wA[:, g] = det
            Acov[:, g] = np.einsum("eak,ebk->eab", A, A)
            Dd = np.einsum("ia,eik->eak", dNg[g], D)
            AD = np.einsum("eak,ebk->eab", A, Dd)
            Bref[:, g] = 0.5 * (AD + AD.transpose(0, 2, 1))
        Ac, detc, Jic = frame(dNc)
        self.area = wA.sum(1)
        tie = self._tie(P, D)
        tn = torch.tensor
        self.Ng = tn(Ng)
        self.dNg = tn(dNg)
        self.dNc = tn(dNc)
        self.Jinv = tn(Jinv)
        self.Jic = tn(Jic)
        self.wA = tn(wA)
        self.Acov = tn(Acov)
        self.Acov_c = tn(np.einsum("eak,ebk->eab", Ac, Ac))
        self.Bref = tn(Bref)
        self.tie0 = tn(tie)
        self.te = tn(tt)
        self.Ee = tn(np.ones(ne) * float(E))
        self.nue = tn(np.ones(ne) * float(nu))
        self.d = tn(D.copy())
        self.pc = tn(P.copy())
        self.smw = tn(smooth[Q].astype(float))
        xs = np.array([-GP, GP, GP, -GP])
        es = np.array([-GP, -GP, GP, GP])
        self.sx = tn(np.stack([0.5 * (1 - es), 0.5 * (1 + es)], 1))
        self.sy = tn(np.stack([0.5 * (1 - xs), 0.5 * (1 + xs)], 1))
        dof = np.concatenate([(Q[:, :, None] * 3 + np.arange(3)).reshape(ne, 12), (self.nx + Q[:, :, None] * 3 + np.arange(3)).reshape(ne, 12)], 1)
        self.dof = dof
        self.rows = np.repeat(dof, 24, axis=1).ravel()
        self.cols = np.tile(dof, (1, 24)).ravel()
        tris = []
        tth = []
        for e, q in enumerate(Q):
            if np.linalg.norm(X[q[0]] - X[q[2]]) <= np.linalg.norm(X[q[1]] - X[q[3]]):
                tris += [[q[0], q[1], q[2]], [q[0], q[2], q[3]]]
            else:
                tris += [[q[0], q[1], q[3]], [q[1], q[2], q[3]]]
            tth += [tt[e], tt[e]]
        self.tris = np.array(tris, int)
        self.thick = np.array(tth)
        TP = X[self.tris]
        self.A0 = 0.5 * np.linalg.norm(np.cross(TP[:, 1] - TP[:, 0], TP[:, 2] - TP[:, 0]), axis=1)
        self.E = self.Ee
        self.nu = self.nue
        self.t = tn(self.thick)
        used = np.zeros(self.n, bool)
        used[Q.ravel()] = True
        fx = {self.nx + 3 * int(i) + c for i in np.where(~used)[0] for c in range(3)}
        for i, c in rot_fixed or []:
            fx.add(self.nx + 3 * int(i) + int(c))
        self.fixed_phi = np.array(sorted(fx), int)
        self.plastic = None
        if plastic:
            self.plastic = Hardening(plastic, float(E))
            tg = self.te[:, None].expand(ne, 4).reshape(-1)
            self.zk = 0.5 * tg[:, None] * tn(GAUSS5[0])[None, :]
            self.wk = self.wA.reshape(-1)[:, None] * 0.5 * tg[:, None] * tn(GAUSS5[1])[None, :]
            self.ep = torch.zeros(ne * 4, 5, 3)
            self.al = torch.zeros(ne * 4, 5)
            self.sig = torch.zeros(ne * 4, 5, 3)
            self.wp = 0.0
        self._cache = None
        self.floor = 1e-13 * float(E) * float(tt.mean()) * float(np.sqrt(self.area.mean())) * np.sqrt(self.ndof)

    @staticmethod
    def _tie(P, D):
        def g(i, j, k, l):
            return ((P[:, j] - P[:, i]) * 0.5 * 0.5 * (D[:, k] + D[:, l])).sum(-1)

        return np.stack([g(0, 1, 0, 1), g(3, 2, 3, 2), g(0, 3, 0, 3), g(1, 2, 1, 2)], 1)

    def x0(self, X=None):
        x = np.zeros(self.ndof)
        x[: self.nx] = np.asarray(self.X if X is None else X, float).ravel()
        return x

    def reset(self, x):
        x = x.copy()
        x[self.nx :] = 0.0
        return x

    def _u(self, x):
        dt = getattr(self, "_dt", torch.float64)
        v = np.asarray(x, float)[self.dof]
        return torch.tensor(v.astype(np.float32) if dt == torch.float32 else v, dtype=dt, device=getattr(self, "_dev", "cpu"))

    def _hshadow(self):
        if getattr(self, "_is_shadow", False):
            return None
        dev = os.environ.get("FEMWEB_DEVICE", "cpu").lower()
        prec = os.environ.get("FEMWEB_HESS", "float64" if dev == "cpu" else "float32").lower()
        if dev == "cpu" and prec == "float64":
            return None
        if getattr(self, "_hs", None) is None:
            dt = torch.float32 if prec == "float32" else torch.float64
            s = copy.copy(self)
            s._is_shadow = True
            s._hs = None
            s._cache = None
            for k, v in vars(self).items():
                if torch.is_tensor(v):
                    setattr(s, k, v.to(device=dev, dtype=dt) if v.is_floating_point() else v.to(device=dev))
            if self.plastic is not None:
                hp = copy.copy(self.plastic)
                for k in ("ep", "sy", "sl"):
                    setattr(hp, k, getattr(self.plastic, k).to(device=dev, dtype=dt))
                s.plastic = hp
            s._dev, s._dt = dev, dt
            self._hs = s
        return self._hs

    def _kin(self, u):
        ne = u.shape[0]
        p = u[:, :12].reshape(ne, 4, 3)
        th = u[:, 12:].reshape(ne, 4, 3)
        d = _rot(th, self.d)
        a = torch.einsum("gia,eik->egak", self.dNg, p)
        dd = torch.einsum("gia,eik->egak", self.dNg, d)
        ecov = 0.5 * (torch.einsum("egak,egbk->egab", a, a) - self.Acov)
        ad = torch.einsum("egak,egbk->egab", a, dd)
        kcov = 0.5 * (ad + ad.transpose(2, 3)) - self.Bref
        Ji = self.Jinv
        em = Ji @ ecov @ Ji.transpose(2, 3)
        kp = Ji @ kcov @ Ji.transpose(2, 3)
        ac = torch.einsum("ia,eik->eak", self.dNc, p)
        ec = self.Jic @ (0.5 * (torch.einsum("eak,ebk->eab", ac, ac) - self.Acov_c)) @ self.Jic.transpose(1, 2)
        s12 = ec[:, 0, 1][:, None].expand(ne, 4)
        em = torch.stack([torch.stack([em[..., 0, 0], s12], -1), torch.stack([s12, em[..., 1, 1]], -1)], -2)

        def tg(i, j, k, l):
            return ((p[:, j] - p[:, i]) * 0.25 * (d[:, k] + d[:, l])).sum(-1)

        tie = torch.stack([tg(0, 1, 0, 1), tg(3, 2, 3, 2), tg(0, 3, 0, 3), tg(1, 2, 1, 2)], 1) - self.tie0
        gx = tie[:, None, 0] * self.sx[None, :, 0] + tie[:, None, 1] * self.sx[None, :, 1]
        gy = tie[:, None, 2] * self.sy[None, :, 0] + tie[:, None, 3] * self.sy[None, :, 1]
        gam = torch.einsum("egab,egb->ega", Ji, torch.stack([gx, gy], -1))
        def unit(v):
            return v / torch.linalg.norm(v, dim=-1, keepdim=True)

        nxt = [1, 2, 3, 0]
        prv = [3, 0, 1, 2]
        psi = 0.0
        for idx in (nxt, prv):
            m = _rot(th, unit(self.pc[:, idx] - self.pc))
            psi = psi + (d * torch.linalg.cross(m, unit(p[:, idx] - p), dim=-1)).sum(-1)
        dr = (psi**2 * self.smw).sum(1)
        return em, kp, gam, dr

    def _extra_energy(self, gam, dr):
        G = self.Ee / (2 * (1 + self.nue))
        es = 0.5 * (5.0 / 6.0) * (G * self.te)[:, None] * (gam * gam).sum(-1) * self.wA
        return es.sum(1) + 1e-3 * G * self.te * self.wA.sum(1) * 0.25 * dr

    def _energy_t(self, u):
        em, kp, gam, dr = self._kin(u)
        nu = self.nue[:, None]

        def quad(e):
            return (1.0 - nu) * (e * e).sum((2, 3)) + nu * (e[..., 0, 0] + e[..., 1, 1]) ** 2

        c = (self.Ee / (1.0 - self.nue**2))[:, None]
        t = self.te[:, None]
        mb = (self.wA * c * (0.5 * t * quad(em) + t**3 / 24.0 * quad(kp))).sum(1)
        return mb + self._extra_energy(gam, dr)

    def _eps(self, em, kp):
        ne = em.shape[0]
        e = em.reshape(ne * 4, 1, 2, 2) + self.zk[:, :, None, None] * kp.reshape(ne * 4, 1, 2, 2)
        return torch.stack([e[..., 0, 0], e[..., 1, 1], 2.0 * e[..., 0, 1]], dim=-1)

    def _rm(self, eps):
        E4 = self.Ee[:, None].expand(-1, 4).reshape(-1)[:, None]
        n4 = self.nue[:, None].expand(-1, 4).reshape(-1)[:, None]
        return _return_map(eps, self.ep, self.al, E4, n4, self.plastic)

    def _stress(self, x):
        with torch.no_grad():
            em, kp, gam, dr = self._kin(self._u(x))
            eps = self._eps(em, kp)
            return (eps,) + self._rm(eps) + (gam, dr)

    def energy(self, x):
        with torch.no_grad():
            if self.plastic is None:
                return float(self._energy_t(self._u(x)).sum())
            eps, sig, ep, al, gam, dr = self._stress(x)
            return float(0.5 * (self.wk[:, :, None] * sig * (eps - ep)).sum() + self._extra_energy(gam, dr).sum())

    def _gh(self, x, second):
        key = np.asarray(x, float).tobytes()
        c = self._cache
        if c is not None and c[0] == key and (c[2] is not None or not second):
            return c[1], c[2]
        hs = self._hshadow() if second else None
        if hs is not None:
            g = self._gh(x, False)[0]
            H = hs._gh(x, True)[1].astype(float)
            self._cache = (key, g, H)
            return g, H
        u = self._u(x).clone().requires_grad_(True)
        if self.plastic is None:
            g = torch.autograd.grad(self._energy_t(u).sum(), u, create_graph=second)[0]
        else:
            em, kp, gam, dr = self._kin(u)
            eps = self._eps(em, kp)
            sig, _, _ = self._rm(eps)
            g = torch.autograd.grad(eps, u, grad_outputs=self.wk[:, :, None] * sig, create_graph=second, retain_graph=True)[0]
            g = g + torch.autograd.grad(self._extra_energy(gam, dr).sum(), u, create_graph=second)[0]
        H = None
        if second:
            H = torch.stack([torch.autograd.grad(g[:, i].sum(), u, retain_graph=True)[0] for i in range(24)], 1).detach().cpu().numpy()
            H = 0.5 * (H + H.transpose(0, 2, 1))
        g = g.detach().cpu().numpy()
        self._cache = (key, g, H)
        return g, H

    def grad(self, x):
        g = self._gh(x, False)[0]
        out = np.zeros(self.ndof)
        np.add.at(out, self.dof.ravel(), g.ravel())
        return out

    def hess(self, x):
        h = self._gh(x, True)[1].reshape(-1)
        return sp.coo_matrix((h, (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()

    def commit(self, x):
        u = self._u(x)
        with torch.no_grad():
            if self.plastic is not None:
                eps, sig, ep, al, _, _ = self._stress(x)
                sy, _ = self.plastic(0.5 * (al + self.al))
                self.wp += float((self.wk * sy * (al - self.al)).sum())
                self.ep, self.al, self.sig = ep, al, sig
            d = _rot(u[:, 12:].reshape(-1, 4, 3), self.d)
            self.d = d / torch.linalg.norm(d, dim=-1, keepdim=True)
            self.pc = u[:, :12].reshape(-1, 4, 3).clone()
        self._cache = None
        self._hs = None

    def lumped_area(self):
        a = np.zeros(self.n)
        np.add.at(a, self.quads.ravel(), np.repeat(self.area / 4.0, 4))
        return a

    def _to_nodes(self, val):
        nod = np.zeros(self.n)
        cnt = np.zeros(self.n)
        np.add.at(nod, self.quads.ravel(), np.repeat(val, 4))
        np.add.at(cnt, self.quads.ravel(), 1.0)
        return nod / np.maximum(cnt, 1.0)

    def vm_nodal(self, x):
        if self.plastic is not None:
            sg = self.sig
            vm = torch.sqrt(torch.clamp(sg[..., 0] ** 2 + sg[..., 1] ** 2 - sg[..., 0] * sg[..., 1] + 3 * sg[..., 2] ** 2, min=0)).max(1).values
            return self._to_nodes(vm.reshape(-1, 4).max(1).values.numpy())
        with torch.no_grad():
            em, kp, _, _ = self._kin(self._u(x))
            best = torch.zeros(em.shape[0])
            c = (self.Ee / (1 - self.nue**2))[:, None]
            nu = self.nue[:, None]
            for sgn in (-1.0, 1.0):
                e = em + sgn * 0.5 * self.te[:, None, None, None] * kp
                tr = e[..., 0, 0] + e[..., 1, 1]
                s11 = c * ((1 - nu) * e[..., 0, 0] + nu * tr)
                s22 = c * ((1 - nu) * e[..., 1, 1] + nu * tr)
                s12 = c * (1 - nu) * e[..., 0, 1]
                vm = torch.sqrt(torch.clamp(s11**2 + s22**2 - s11 * s22 + 3 * s12**2, min=0)).max(1).values
                best = torch.maximum(best, vm)
            return self._to_nodes(best.numpy())

    def peeq_nodal(self):
        return self._to_nodes(self.al.max(1).values.reshape(-1, 4).max(1).values.numpy())

    def peeq_max(self):
        return float(self.al.max())


def qgrid(L, b, nx, ny):
    xs = np.linspace(0, L, nx + 1)
    ys = np.linspace(0, b, ny + 1)
    X = np.array([[x, y, 0.0] for y in ys for x in xs])
    Q = [[j * (nx + 1) + i, j * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i + 1, (j + 1) * (nx + 1) + i] for j in range(ny) for i in range(nx)]
    return X, np.array(Q)


def qtube(a, L, npa, nz):
    ring = []
    for side in range(4):
        for i in range(npa):
            s = a * i / npa
            ring.append([(s, 0.0), (a, s), (a - s, a), (0.0, a - s)][side])
    m = len(ring)
    X = np.array([[x, y, L * k / nz] for k in range(nz + 1) for x, y in ring])
    Q = [[k * m + i, k * m + (i + 1) % m, (k + 1) * m + (i + 1) % m, (k + 1) * m + i] for k in range(nz) for i in range(m)]
    return X, np.array(Q)
