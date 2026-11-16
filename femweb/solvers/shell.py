import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from torch.func import grad, hessian, vmap

torch.set_default_dtype(torch.float64)


def _quad(e, nu):
    return (1.0 - nu) * (e * e).sum() + nu * (e[0, 0] + e[1, 1]) ** 2


def _patch6(p, R, kind):
    q = []
    for i in range(3):
        j, k = (i + 1) % 3, (i + 2) % 3
        free = p[j] + p[k] - p[i]
        cl = p[j] + R[i] @ (p[i] - p[j])
        q.append(torch.where(kind[i] == 0, p[3 + i], torch.where(kind[i] == 1, free, cl)))
    return torch.cat([p[:3], torch.stack(q)])


def _angle(n, pj, pk, q):
    e = pk - pj
    m = torch.linalg.cross(pj - pk, q - pk)
    s = torch.dot(torch.linalg.cross(n, m), e) / torch.linalg.norm(e)
    return torch.atan2(s, torch.dot(n, m))


def _angles(p, R, kind):
    y = _patch6(p, R, kind)
    n = torch.linalg.cross(y[1] - y[0], y[2] - y[0])
    return torch.stack([_angle(n, y[1], y[2], y[3]), _angle(n, y[2], y[0], y[4]), _angle(n, y[0], y[1], y[5])])


def _strains(u, Dmi, B, R, kind, th0, w, sg):
    p = u[:18].reshape(6, 3)
    phi = u[18:]
    Ds = torch.stack([p[1] - p[0], p[2] - p[0]], dim=1)
    F = Ds @ Dmi
    Em = 0.5 * (F.T @ F - torch.eye(2))
    dth = (_angles(p, R, kind) - th0) * w + 2.0 * sg * phi
    K = (dth[:, None, None] * B).sum(0)
    return Em, K


def _energy(u, Dmi, A0, B, R, kind, th0, w, sg, E, nu, t):
    Em, K = _strains(u, Dmi, B, R, kind, th0, w, sg)
    c = E / (1.0 - nu * nu)
    return A0 * c * (0.5 * t * _quad(Em, nu) + t**3 / 24.0 * _quad(K, nu))


_vE = vmap(_energy)
_vG = vmap(grad(_energy))
_vH = vmap(hessian(_energy))
_vS = vmap(_strains)
_vA = vmap(_angles)


class Shell:
    def __init__(self, X, tris, E, nu, t, clamped=None):
        X = np.asarray(X, float)
        tris = np.asarray(tris, int)
        self.n = len(X)
        self.tris = tris
        ne = len(tris)
        edges = {}
        for e, tr in enumerate(tris):
            for i in range(3):
                a, b = int(tr[(i + 1) % 3]), int(tr[(i + 2) % 3])
                edges.setdefault((min(a, b), max(a, b)), []).append((e, i))
        self.edges = edges
        self.nedge = len(edges)
        clamped = set(int(v) for v in (clamped if clamped is not None else []))
        P = X[tris]
        e1 = P[:, 1] - P[:, 0]
        e2 = P[:, 2] - P[:, 0]
        nrm = np.cross(e1, e2)
        A2 = np.linalg.norm(nrm, axis=1)
        a1 = e1 / np.linalg.norm(e1, axis=1)[:, None]
        a3 = nrm / A2[:, None]
        a2 = np.cross(a3, a1)
        Q = np.zeros((ne, 3, 2))
        Q[:, 1, 0] = (e1 * a1).sum(1)
        Q[:, 2, 0] = (e2 * a1).sum(1)
        Q[:, 2, 1] = (e2 * a2).sum(1)
        Dm = np.stack([Q[:, 1], Q[:, 2]], axis=2)
        opp = tris.copy()
        kind = np.ones((ne, 3), int)
        w = np.zeros((ne, 3))
        sg = np.ones((ne, 3))
        eid = np.zeros((ne, 3), int)
        R = np.tile(np.eye(3), (ne, 3, 1, 1))
        fixphi = []
        for n_e, ((a, b), lst) in enumerate(edges.items()):
            for e0, i0 in lst:
                eid[e0, i0] = n_e
            if len(lst) == 2:
                (e0, i0), (e1_, i1) = lst
                opp[e0, i0] = tris[e1_, i1]
                opp[e1_, i1] = tris[e0, i0]
                kind[e0, i0] = kind[e1_, i1] = 0
                w[e0, i0] = w[e1_, i1] = 1.0
                sg[e1_, i1] = -1.0
            elif len(lst) == 1 and a in clamped and b in clamped:
                e0, i0 = lst[0]
                kind[e0, i0] = 2
                w[e0, i0] = 1.0
                fixphi.append(n_e)
                j, k = (i0 + 1) % 3, (i0 + 2) % 3
                d2 = Q[e0, k] - Q[e0, j]
                nu2 = np.array([d2[1], -d2[0]]) / np.linalg.norm(d2)
                m = nu2[0] * a1[e0] + nu2[1] * a2[e0]
                R[e0, i0] = np.eye(3) - 2.0 * np.outer(m, m)
            elif len(lst) > 2:
                fixphi.append(n_e)
        B = np.zeros((ne, 3, 2, 2))
        for i in range(3):
            d = Q[:, (i + 2) % 3] - Q[:, (i + 1) % 3]
            pr = np.stack([d[:, 1], -d[:, 0]], axis=1)
            ln = np.linalg.norm(d, axis=1)
            B[:, i] = pr[:, :, None] * pr[:, None, :] / (A2 * ln)[:, None, None]
        self.A0 = A2 / 2.0
        self.frame = np.stack([a1, a2, a3], axis=1)
        tt = torch.tensor
        self.Dmi = tt(np.linalg.inv(Dm))
        self.tA0 = tt(self.A0)
        self.B = tt(B)
        self.R = tt(R)
        self.kind = tt(kind)
        self.w = tt(w)
        self.sg = tt(sg)
        ones = np.ones(ne)
        self.E = tt(ones * E)
        self.nu = tt(ones * nu)
        self.t = tt(ones * np.asarray(t, float))
        self.thick = self.t.numpy()
        self.X = X
        self.nx = 3 * self.n
        self.ndof = self.nx + self.nedge
        self.fixed_phi = self.nx + np.array(fixphi, int)
        nd = (np.hstack([tris, opp])[:, :, None] * 3 + np.arange(3)[None, None, :]).reshape(ne, 18)
        dof = np.hstack([nd, self.nx + eid])
        self.dof = dof
        self.rows = np.repeat(dof, 21, axis=1).ravel()
        self.cols = np.tile(dof, (1, 21)).ravel()
        self.th0 = _vA(tt(X[np.hstack([tris, opp])]), self.R, self.kind)
        self.floor = 1e-13 * float(E) * float(np.mean(self.thick)) * float(np.sqrt(self.A0.mean())) * np.sqrt(self.ndof)

    def x0(self, X=None):
        x = np.zeros(self.ndof)
        x[: self.nx] = np.asarray(self.X if X is None else X, float).ravel()
        return x

    def _args(self, x):
        u = torch.tensor(np.asarray(x, float)[self.dof])
        return (u, self.Dmi, self.tA0, self.B, self.R, self.kind, self.th0, self.w, self.sg, self.E, self.nu, self.t)

    def energy(self, x):
        return float(_vE(*self._args(x)).sum())

    def grad(self, x):
        g = _vG(*self._args(x)).numpy()
        out = np.zeros(self.ndof)
        np.add.at(out, self.dof.ravel(), g.ravel())
        return out

    def hess(self, x):
        h = _vH(*self._args(x)).numpy().reshape(-1)
        return sp.coo_matrix((h, (self.rows, self.cols)), shape=(self.ndof, self.ndof)).tocsr()

    def strains(self, x):
        Em, K = _vS(*self._args(x)[:1], self.Dmi, self.B, self.R, self.kind, self.th0, self.w, self.sg)
        return Em.numpy(), K.numpy()

    def lumped_area(self):
        a = np.zeros(self.n)
        np.add.at(a, self.tris.ravel(), np.repeat(self.A0 / 3.0, 3))
        return a


def _solve(H, r):
    try:
        return spla.spsolve(H.tocsc(), r)
    except Exception:
        return spla.lsqr(H, r)[0]


def newton(sh, x, free, fext, extra=None, tol=1e-8, itmax=40):
    x = x.copy()
    fscale = max(np.linalg.norm(fext[free]), 1e-12)
    info = {"iters": 0, "converged": False}

    def total(xx):
        e = sh.energy(xx) - fext @ xx
        if extra is not None:
            e += extra.energy(xx)
        return e

    def resid(xx):
        g = sh.grad(xx) - fext
        if extra is not None:
            g = g + extra.grad(xx)
        return g

    g = resid(x)
    ref = None
    for it in range(itmax):
        r = g[free]
        nr = np.linalg.norm(r)
        if ref is None:
            ref = max(nr, fscale, 1e-9)
        info["iters"] = it
        info["res"] = nr / ref
        if nr <= max(tol * ref, sh.floor):
            info["converged"] = True
            break
        H = sh.hess(x)
        if extra is not None:
            H = H + extra.hess(x)
        Hf = H[free][:, free]
        dx = _solve(Hf, -r)
        if not np.all(np.isfinite(dx)) or dx @ r > 0:
            dx = _solve(Hf + sp.identity(len(free)) * (abs(Hf.diagonal()).mean() * 1e-3), -r)
        if extra is not None and hasattr(extra, "limit"):
            dx = dx * extra.limit(x, free, dx)
        e0 = total(x)
        a = 1.0
        ok = False
        for _ in range(12):
            xn = x.copy()
            xn[free] += a * dx
            gn = resid(xn)
            if np.isfinite(gn).all() and (np.linalg.norm(gn[free]) < nr or total(xn) < e0):
                ok = True
                break
            a *= 0.5
        if not ok:
            xn = x.copy()
            xn[free] += a * dx
            gn = resid(xn)
        x, g = xn, gn
    return x, info


def run_static(sh, X, fixed, pres_dofs, pres_vals, fext, nsteps, extra=None, on_step=None, tol=1e-8):
    ndof = sh.ndof
    x = sh.x0(X)
    x0 = x.copy()
    con = np.zeros(ndof, bool)
    con[np.asarray(fixed, int)] = True
    con[np.asarray(pres_dofs, int)] = True
    con[sh.fixed_phi] = True
    free = np.where(~con)[0]
    hist = []
    lam_prev = 0.0
    k = 0
    dl = 1.0 / nsteps
    lam = 0.0
    cuts = 0
    while lam < 1.0 - 1e-12:
        lt = min(1.0, lam + dl)
        xt = x.copy()
        if len(pres_dofs):
            xt[pres_dofs] = x0[pres_dofs] + lt * np.asarray(pres_vals)
        if extra is not None and hasattr(extra, "begin"):
            extra.begin(x)
        xs, info = newton(sh, xt, free, fext * lt, extra=extra, tol=tol)
        if not info["converged"]:
            cuts += 1
            dl *= 0.5
            if cuts > 8:
                hist.append({"lam": lam, "failed": True})
                break
            continue
        cuts = 0
        x = xs
        lam = lt
        k += 1
        g = sh.grad(x)
        if extra is not None:
            g = g + extra.grad(x)
        rec = {"lam": lam, "x": x.copy(), "reaction": g - fext * lam, "iters": info["iters"]}
        hist.append(rec)
        if on_step:
            on_step(rec)
        if info["iters"] <= 5:
            dl = min(dl * 1.5, 1.0 / nsteps)
    return hist


def grid(L, b, nx, ny, alt=False):
    xs = np.linspace(0, L, nx + 1)
    ys = np.linspace(0, b, ny + 1)
    X = np.array([[x, y, 0.0] for y in ys for x in xs])
    T = []
    for j in range(ny):
        for i in range(nx):
            a = j * (nx + 1) + i
            c = a + nx + 1
            if alt and (i + j) % 2:
                T += [[a, a + 1, c], [a + 1, c + 1, c]]
            else:
                T += [[a, a + 1, c + 1], [a, c + 1, c]]
    return X, np.array(T)


def tube(a, L, npa, nz):
    ring = []
    for side in range(4):
        for i in range(npa):
            s = a * i / npa
            ring.append([(s, 0.0), (a, s), (a - s, a), (0.0, a - s)][side])
    m = len(ring)
    X = np.array([[x, y, L * k / nz] for k in range(nz + 1) for x, y in ring])
    T = []
    for k in range(nz):
        for i in range(m):
            p, q = k * m + i, k * m + (i + 1) % m
            if (i + k) % 2:
                T += [[p, q, p + m], [q, q + m, p + m]]
            else:
                T += [[p, q, q + m], [p, q + m, p + m]]
    return X, np.array(T)


class SelfContact:
    def __init__(self, sh, stiffness=None, ring=1, margin=None):
        from scipy.spatial import cKDTree

        self.sh = sh
        self.tree = cKDTree
        tris = sh.tris
        nt = np.zeros(sh.n)
        cnt = np.zeros(sh.n)
        np.add.at(nt, tris.ravel(), np.repeat(sh.thick, 3))
        np.add.at(cnt, tris.ravel(), 1.0)
        self.tn = nt / np.maximum(cnt, 1.0)
        adj = [set([i]) for i in range(sh.n)]
        for tr in tris:
            for a in tr:
                adj[int(a)].update(int(b) for b in tr)
        for _ in range(ring - 1):
            adj = [set().union(*[adj[j] for j in s]) for s in adj]
        self.adj = adj
        P = sh.X[tris]
        self.rad = float(np.linalg.norm(P - P.mean(1, keepdims=True), axis=2).max())
        self.hmax = float(sh.thick.max())
        self.margin = self.hmax * 2.0 if margin is None else margin
        self.k = float(sh.E.mean()) * float(sh.thick.mean()) if stiffness is None else stiffness
        self.pn = np.zeros(0, int)
        self.pt = np.zeros(0, int)
        self.s0 = np.zeros(0)

    def begin(self, x):
        sh = self.sh
        xx = x[: sh.nx].reshape(-1, 3)
        cen = xx[sh.tris].mean(1)
        tree = self.tree(cen)
        lst = tree.query_ball_point(xx, self.rad + self.hmax + self.margin)
        pn, pt = [], []
        for i, cand in enumerate(lst):
            ad = self.adj[i]
            for e in cand:
                tr = sh.tris[e]
                if int(tr[0]) in ad or int(tr[1]) in ad or int(tr[2]) in ad:
                    continue
                pn.append(i)
                pt.append(e)
        self.pn = np.array(pn, int)
        self.pt = np.array(pt, int)
        if len(pn):
            v, nrm, _, _ = self._geom(xx)
            self.s0 = np.where((v * nrm).sum(1) >= 0, 1.0, -1.0)
        else:
            self.s0 = np.zeros(0)

    def _geom(self, xx):
        from ..contact import _closest

        tr = self.sh.tris[self.pt]
        a, b, c = xx[tr[:, 0]], xx[tr[:, 1]], xx[tr[:, 2]]
        p = xx[self.pn]
        out = _closest(p, a, b, c)
        bw = out[-1] if out[-1].shape[-1] == 3 and out[-1].ndim == 2 else out[1]
        q = bw[:, :1] * a + bw[:, 1:2] * b + bw[:, 2:3] * c
        nrm = np.cross(b - a, c - a)
        nrm /= np.linalg.norm(nrm, axis=1)[:, None]
        return p - q, nrm, bw, tr

    def _gap(self, x):
        sh = self.sh
        xx = x[: sh.nx].reshape(-1, 3)
        v, nrm, bw, tr = self._geom(xx)
        d = np.linalg.norm(v, axis=1)
        side = np.where((v * nrm).sum(1) >= 0, 1.0, -1.0)
        same = side == self.s0
        dirn = np.where(d[:, None] > 1e-12, v / np.maximum(d, 1e-12)[:, None], nrm * self.s0[:, None])
        dirn = np.where(same[:, None], dirn, -dirn)
        h = 0.5 * (self.tn[self.pn] + sh.thick[self.pt])
        g = np.where(same, d, -d) - h
        return g, dirn, bw, tr

    def energy(self, x):
        if not len(self.pn):
            return 0.0
        g = self._gap(x)[0]
        return float(0.5 * self.k * (np.minimum(g, 0.0) ** 2).sum())

    def _vec(self, x):
        g, dirn, bw, tr = self._gap(x)
        act = g < 0
        g, dirn, bw, tr, pn = g[act], dirn[act], bw[act], tr[act], self.pn[act]
        nodes = np.column_stack([pn, tr])
        coef = np.column_stack([np.ones(len(g)), -bw])
        V = coef[:, :, None] * dirn[:, None, :]
        dof = nodes[:, :, None] * 3 + np.arange(3)[None, None, :]
        return g, V.reshape(len(g), 12), dof.reshape(len(g), 12)

    def grad(self, x):
        out = np.zeros(self.sh.ndof)
        if not len(self.pn):
            return out
        g, V, dof = self._vec(x)
        np.add.at(out, dof.ravel(), (self.k * g[:, None] * V).ravel())
        return out

    def hess(self, x):
        n = self.sh.ndof
        if not len(self.pn):
            return sp.csr_matrix((n, n))
        g, V, dof = self._vec(x)
        h = self.k * V[:, :, None] * V[:, None, :]
        return sp.coo_matrix((h.ravel(), (np.repeat(dof, 12, axis=1).ravel(), np.tile(dof, (1, 12)).ravel())), shape=(n, n)).tocsr()

    def force(self, x):
        g = self._gap(x)[0] if len(self.pn) else np.zeros(0)
        return float(-self.k * np.minimum(g, 0.0).sum()), int((g < 0).sum())
