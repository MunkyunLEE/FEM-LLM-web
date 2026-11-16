import copy

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
    da = _angles(p, R, kind) - th0
    dth = torch.atan2(torch.sin(da), torch.cos(da)) * w + 2.0 * sg * phi
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


def newton(sh, x, free, fext, extra=None, tol=1e-8, itmax=25):
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
    con = np.ones(len(x), bool)
    con[free] = False
    bad = 0
    for it in range(itmax):
        r = g[free]
        nr = np.linalg.norm(r)
        if not np.isfinite(nr):
            break
        ref = max(fscale, np.linalg.norm(g[con]), 1e-9)
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
            bad += 1
            if bad >= 2:
                break
            xn = x.copy()
            xn[free] += a * dx
            gn = resid(xn)
        x, g = xn, gn
    return x, info


def run_static(sh, X, fixed, pres_dofs, pres_vals, fext, nsteps, extra=None, on_step=None, tol=1e-8, time=1.0, maxcuts=12, x_init=None, amp=None):
    ndof = sh.ndof
    x = sh.x0(X) if x_init is None else x_init.copy()
    x0 = sh.x0(X)
    pres_dofs = np.asarray(pres_dofs, int)
    con = np.zeros(ndof, bool)
    con[np.asarray(fixed, int)] = True
    con[pres_dofs] = True
    con[sh.fixed_phi] = True
    free = np.where(~con)[0]
    hist = []
    dl = 1.0 / nsteps
    lam = 0.0
    cuts = 0
    amp = amp or (lambda v: v)
    while lam < 1.0 - 1e-12:
        lt = min(1.0, lam + dl)
        xt = x.copy()
        if len(pres_dofs):
            xt[pres_dofs] = x0[pres_dofs] + amp(lt) * np.asarray(pres_vals)
        if extra is not None and hasattr(extra, "set_factor"):
            extra.set_factor(amp(lam))
        if extra is not None and hasattr(extra, "begin"):
            if isinstance(extra, Extras):
                extra.begin(x, (lt - lam) * time)
            else:
                extra.begin(x)
        if extra is not None and hasattr(extra, "set_factor"):
            extra.set_factor(amp(lt))
        if extra is not None and hasattr(extra, "fraction"):
            a = extra.fraction(x, xt)
            if a < 1.0:
                lt = lam + a * (lt - lam)
                if len(pres_dofs):
                    xt[pres_dofs] = x0[pres_dofs] + amp(lt) * np.asarray(pres_vals)
                extra.set_factor(amp(lt))
        if lt - lam < 1e-12:
            hist.append({"lam": lam, "failed": True})
            break
        xs, info = newton(sh, xt, free, fext * amp(lt), extra=extra, tol=tol)
        if not info["converged"]:
            cuts += 1
            dl *= 0.5
            if cuts > maxcuts:
                hist.append({"lam": lam, "failed": True})
                break
            continue
        cuts = 0
        x = xs
        lam = lt
        if extra is not None and hasattr(extra, "commit"):
            extra.commit(x)
        g = sh.grad(x)
        if extra is not None:
            g = g + (extra.phys_grad(x) if hasattr(extra, "phys_grad") else extra.grad(x))
        rec = {"lam": lam, "x": x.copy(), "reaction": g - fext * amp(lam), "iters": info["iters"]}
        hist.append(rec)
        if on_step:
            on_step(rec)
        if info["iters"] <= 6:
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


class Damping:
    dissipative = True

    def __init__(self, sh, rho, alpha=0.0, beta=0.0, inertia=True, K0=None):
        self.sh = sh
        m = np.zeros(sh.ndof)
        np.add.at(m, (sh.tris[:, :, None] * 3 + np.arange(3)[None, None, :]).ravel(), np.repeat(rho * sh.thick * sh.A0 / 3.0, 9))
        self.m = m
        self.M = sp.diags(m).tocsr()
        self.alpha = alpha
        self.beta = beta
        self.inertia = inertia
        self.K0 = K0 if K0 is not None else (sh.hess(sh.x0()) if beta else None)
        self.C = self.M * alpha
        if beta:
            self.C = self.C + self.K0 * beta
        self.vp = np.zeros(sh.ndof)
        self.xp = None
        self.dt = 1.0
        self.dissipated = 0.0

    def begin(self, x, dt=1.0):
        self.xp = x.copy()
        self.dt = dt

    def energy(self, x):
        dx = x - self.xp
        e = 0.5 / self.dt * dx @ (self.C @ dx)
        if self.inertia:
            r = dx - self.dt * self.vp
            e += 0.5 / self.dt**2 * r @ (self.m * r)
        return float(e)

    def grad(self, x):
        dx = x - self.xp
        g = self.C @ dx / self.dt
        if self.inertia:
            g = g + self.m * (dx - self.dt * self.vp) / self.dt**2
        return g

    def hess(self, x):
        H = self.C / self.dt
        if self.inertia:
            H = H + self.M / self.dt**2
        return H

    def commit(self, x):
        dx = x - self.xp
        self.dissipated += float(dx @ (self.C @ dx)) / self.dt
        self.vp = dx / self.dt

    def kinetic(self):
        return float(0.5 * self.vp @ (self.m * self.vp))


def frequencies(sh, rho, fixed, k=2):
    con = np.zeros(sh.ndof, bool)
    con[np.asarray(fixed, int)] = True
    con[sh.fixed_phi] = True
    fr = np.where(~con)[0]
    d = Damping(sh, rho)
    K = sh.hess(sh.x0())[fr][:, fr].tocsc()
    M = d.M[fr][:, fr].tocsc()
    vals = spla.eigsh(K, k=k, M=M, sigma=0.0, which="LM")[0]
    return np.sqrt(np.maximum(np.sort(vals), 0.0))


def rayleigh(zeta, w1, w2, kind="rayleigh"):
    if kind == "mass":
        return 2.0 * zeta * w1, 0.0
    if kind == "stiffness":
        return 0.0, 2.0 * zeta / w1
    return 2.0 * zeta * w1 * w2 / (w1 + w2), 2.0 * zeta / (w1 + w2)


class Extras:
    def __init__(self, *parts):
        self.parts = [p for p in parts if p is not None]

    def begin(self, x, dt=1.0):
        for p in self.parts:
            if getattr(p, "dissipative", False):
                p.begin(x, dt)
            else:
                p.begin(x)

    def commit(self, x):
        for p in self.parts:
            if hasattr(p, "commit"):
                p.commit(x)

    def limit(self, x, free, dx):
        a = 1.0
        for p in self.parts:
            if hasattr(p, "limit"):
                a = min(a, p.limit(x, free, dx))
        return a

    def set_factor(self, lam):
        for p in self.parts:
            if hasattr(p, "set_factor"):
                p.set_factor(lam)

    def fraction(self, x0, x1):
        a = 1.0
        for p in self.parts:
            if hasattr(p, "fraction"):
                a = min(a, p.fraction(x0, x1))
        return a

    def phys_grad(self, x):
        return sum(p.grad(x) for p in self.parts if not getattr(p, "dissipative", False))

    def energy(self, x):
        return sum(p.energy(x) for p in self.parts)

    def grad(self, x):
        return sum(p.grad(x) for p in self.parts)

    def hess(self, x):
        H = self.parts[0].hess(x)
        for p in self.parts[1:]:
            H = H + p.hess(x)
        return H


class IPCContact:
    def __init__(self, sh, mu=0.0, dhat=None, stiffness=None, rings=1, groups=None, self_contact=True, planes=None, eps_v=None, obstacles=None):
        import ipctk

        self.ipc = ipctk
        self.sh = sh
        self.n = sh.n
        obstacles = list(obstacles or [])
        tris = sh.tris
        Vr = sh.X.copy()
        F = tris.copy()
        ext = float(np.linalg.norm(sh.X.max(0) - sh.X.min(0))) * 3.0
        for pl in planes or []:
            o = np.asarray(pl["point"], float)
            nn = np.asarray(pl["normal"], float)
            nn = nn / np.linalg.norm(nn)
            a = np.cross(nn, [1.0, 0.0, 0.0])
            if np.linalg.norm(a) < 0.5:
                a = np.cross(nn, [0.0, 1.0, 0.0])
            a = a / np.linalg.norm(a)
            b = np.cross(nn, a)
            o = o + 0.37 * ext * a * 0.01 + 0.23 * ext * b * 0.01
            q = np.array([o - ext * a - ext * b, o + ext * a - ext * b, o + ext * a + ext * b, o - ext * a + ext * b])
            obstacles.append({"V": q, "F": np.array([[0, 1, 2], [0, 2, 3]]), "motion": pl.get("motion", [0.0, 0.0, 0.0])})
        self.planes = obstacles
        self.pl_nodes = []
        for ob in obstacles:
            k = len(Vr)
            Vr = np.vstack([Vr, np.asarray(ob["V"], float)])
            F = np.vstack([F, np.asarray(ob["F"], int) + k])
            self.pl_nodes.append(np.arange(k, k + len(ob["V"])))
        self.nv = len(Vr)
        self.rest = np.asfortranarray(Vr)
        F = np.asfortranarray(F.astype(np.int32))
        E = ipctk.edges(F)
        self.mesh = ipctk.CollisionMesh(self.rest, E, F)
        tmean = float(sh.thick.mean())
        self.dmin = tmean
        self.dhat = 0.25 * tmean if dhat is None else float(dhat)
        adj = [set([i]) for i in range(sh.n)]
        for tr in tris:
            for a_ in tr:
                adj[int(a_)].update(int(b_) for b_ in tr)
        base = adj
        for _ in range(rings - 1):
            adj = [set().union(*[base[j] for j in s_]) for s_ in adj]
        blocked = {}
        for i, s_ in enumerate(adj):
            for j in s_:
                if i < j:
                    blocked[(i, j)] = False
        filt = ipctk.make_sparse_filter(blocked, True)
        gid = np.arange(self.nv, dtype=np.int32)
        if groups is not None:
            gid[: sh.n] = np.asarray(groups, np.int32) + self.nv
        elif not self_contact:
            gid[: sh.n] = self.nv
        for k, nodes in enumerate(self.pl_nodes):
            gid[nodes] = 2 * self.nv + 10
        filt = filt & ipctk.make_vertex_patches_filter(gid)
        self.mesh.can_collide = filt
        self.kappa = 2e-3 * float(sh.E.mean()) * tmean if stiffness is None else float(stiffness)
        self.B = ipctk.BarrierPotential(self.dhat, self.kappa, True)
        self.mu = float(mu)
        self.eps_v = 1e-3 * tmean if eps_v is None else eps_v
        self.Fr = ipctk.FrictionPotential(self.eps_v)
        self.ccd = ipctk.AdditiveCCD(10000, 0.9)
        self.tc = None
        self.Vp = None
        self.lam = 0.0
        self._key = None
        self._col = None

    def set_factor(self, lam):
        self.lam = lam

    def _V(self, x):
        V = np.empty((self.nv, 3), order="F")
        V[: self.n] = x[: self.sh.nx].reshape(-1, 3)
        for pl, nodes in zip(self.planes, self.pl_nodes):
            V[nodes] = self.rest[nodes] + self.lam * np.asarray(pl.get("motion", [0.0, 0.0, 0.0]), float)
        return V

    def _cols(self, V):
        key = V.tobytes()
        if key != self._key:
            c = self.ipc.NormalCollisions()
            c.build(self.mesh, V, self.dhat, self.dmin)
            self._key, self._col = key, c
        return self._col

    def begin(self, x):
        V = self._V(x)
        self.Vp = V.copy()
        self.tc = None
        if self.mu > 0:
            c = self._cols(V)
            if len(c):
                tc = self.ipc.TangentialCollisions()
                tc.build(self.mesh, V, c, self.B, self.mu)
                self.tc = tc

    def energy(self, x):
        V = self._V(x)
        c = self._cols(V)
        e = self.B(c, self.mesh, V) if len(c) else 0.0
        if self.tc is not None:
            e += self.Fr(self.tc, self.mesh, np.asfortranarray(V - self.Vp))
        return float(e)

    def _pad(self, g):
        out = np.zeros(self.sh.ndof)
        out[: self.sh.nx] = g[: self.sh.nx]
        return out

    def grad(self, x):
        V = self._V(x)
        c = self._cols(V)
        g = np.zeros(3 * self.nv)
        if len(c):
            g = g + self.B.gradient(c, self.mesh, V)
        if self.tc is not None:
            g = g + self.Fr.gradient(self.tc, self.mesh, np.asfortranarray(V - self.Vp))
        self._g = g
        return self._pad(g)

    def hess(self, x):
        V = self._V(x)
        c = self._cols(V)
        nd = self.sh.ndof
        H = sp.csr_matrix((3 * self.nv, 3 * self.nv))
        if len(c):
            H = H + self.B.hessian(c, self.mesh, V)
        if self.tc is not None:
            H = H + self.Fr.hessian(self.tc, self.mesh, np.asfortranarray(V - self.Vp))
        H = sp.csr_matrix(H)[: self.sh.nx, : self.sh.nx].tocoo()
        return sp.coo_matrix((H.data, (H.row, H.col)), shape=(nd, nd)).tocsr()

    def limit(self, x, free, dx):
        x1 = x.copy()
        x1[free] += dx
        a = self.ipc.compute_collision_free_stepsize(self.mesh, self._V(x), self._V(x1), self.dmin, narrow_phase_ccd=self.ccd)
        return a

    def fraction(self, x0, x1):
        if self.Vp is None:
            return 1.0
        return self.ipc.compute_collision_free_stepsize(self.mesh, self.Vp, self._V(x1), self.dmin, narrow_phase_ccd=self.ccd)

    def force(self, x):
        V = self._V(x)
        c = self._cols(V)
        if not len(c):
            return 0.0, 0, float("inf")
        g = self.B.gradient(c, self.mesh, V).reshape(-1, 3)
        return float(0.5 * np.linalg.norm(g, axis=1).sum()), len(c), float(np.sqrt(c.compute_minimum_distance(self.mesh, V)))

    def plane_force(self, x, k=0):
        V = self._V(x)
        c = self._cols(V)
        if not len(c):
            return np.zeros(3)
        g = self.B.gradient(c, self.mesh, V).reshape(-1, 3)
        if self.tc is not None:
            g = g + self.Fr.gradient(self.tc, self.mesh, np.asfortranarray(V - self.Vp)).reshape(-1, 3)
        return -g[self.pl_nodes[k]].sum(0)


def triangulate(model):
    X = model.nodes
    T, G = [], []
    for f, g in zip(model.shells, model.shell_group):
        f = [int(i) for i in f]
        if len(f) == 3:
            parts = [f]
        elif len(f) == 4:
            if np.linalg.norm(X[f[0]] - X[f[2]]) <= np.linalg.norm(X[f[1]] - X[f[3]]):
                parts = [[f[0], f[1], f[2]], [f[0], f[2], f[3]]]
            else:
                parts = [[f[0], f[1], f[3]], [f[1], f[2], f[3]]]
        else:
            parts = [[f[0], f[i], f[i + 1]] for i in range(1, len(f) - 1)]
        for q in parts:
            if np.linalg.norm(np.cross(X[q[1]] - X[q[0]], X[q[2]] - X[q[0]])) > 1e-14:
                T.append(q)
                G.append(g)
    return np.array(T, int), G


def linear_buckling(sh, fixed, pres, pval, f, k=1):
    con = np.zeros(sh.ndof, bool)
    con[np.asarray(fixed, int)] = True
    con[np.asarray(pres, int)] = True
    con[sh.fixed_phi] = True
    fr = np.where(~con)[0]
    x0 = sh.x0()
    K0 = sh.hess(x0)
    u = np.zeros(sh.ndof)
    u[pres] = pval
    rhs = f - K0 @ u
    u[fr] = _solve(K0[fr][:, fr], rhs[fr])
    um = np.abs(u[: sh.nx]).max()
    if um <= 0:
        return None, None
    size = float(np.linalg.norm(sh.X.max(0) - sh.X.min(0)))
    eps = 1e-5 * size / um
    x1, info = newton(sh, x0 + eps * u, fr, f * eps)
    KG = (sh.hess(x1) - K0)[fr][:, fr] / eps
    try:
        vals, vecs = spla.eigsh(-KG.tocsc(), k=k, M=K0[fr][:, fr].tocsc(), which="LA")
    except Exception:
        return None, None
    order = np.argsort(-vals)
    out = []
    for j in order:
        if vals[j] <= 0:
            continue
        m = np.zeros(sh.ndof)
        m[fr] = vecs[:, j]
        m = m[: sh.nx].reshape(-1, 3)
        out.append((1.0 / vals[j], m / np.abs(m).max()))
    if not out:
        return None, None
    return [o[0] for o in out], [o[1] for o in out]


def _vm(sh, x):
    Em, K = sh.strains(x)
    E = sh.E.numpy()[:, None, None]
    nu = sh.nu.numpy()[:, None, None]
    best = np.zeros(len(Em))
    for sgn in (-1.0, 1.0):
        e = Em + sgn * 0.5 * sh.thick[:, None, None] * K
        tr = (e[:, 0, 0] + e[:, 1, 1])[:, None, None]
        sg = E / (1 - nu**2) * ((1 - nu) * e + nu * tr * np.eye(2)[None])
        vm = np.sqrt(np.maximum(sg[:, 0, 0] ** 2 + sg[:, 1, 1] ** 2 - sg[:, 0, 0] * sg[:, 1, 1] + 3 * sg[:, 0, 1] ** 2, 0))
        best = np.maximum(best, vm)
    nod = np.zeros(sh.n)
    cnt = np.zeros(sh.n)
    np.add.at(nod, sh.tris.ravel(), np.repeat(best, 3))
    np.add.at(cnt, sh.tris.ravel(), 1.0)
    return nod / np.maximum(cnt, 1.0)


def _amp(kind):
    if kind == "smooth":
        return lambda s: s * s * (3.0 - 2.0 * s)
    return lambda s: s


def solve(model, spec, progress=None):
    from ..contact import contacts_of, icosphere, plane_basis, unit, record as contact_record, view_spec
    from ..control import disp_control, record as ctl_record
    from ..loads import bc_table, lumped
    from ..result import Result
    from ..select import resolve
    from ..spec import section_for

    if not model.shells:
        raise ValueError("자체 쉘 솔버에는 면(f) 패널이 필요합니다")
    st = spec["step"]
    if st["type"] not in ("static", "arclength"):
        raise ValueError("자체 쉘 솔버는 정적 스텝만 지원합니다 (좌굴·모달은 CalculiX 를 쓰세요)")
    msgs = ["자체 쉘 솔버: 삼각형 쉘 요소(막 + 모서리 회전 휨), 대변형 음해법. 기하 비선형은 항상 켜져 있습니다"]
    if st["type"] == "arclength":
        msgs.append("자체 쉘 솔버에는 호장법이 없어 변위·하중 증분 정적 해석으로 풀었습니다 (감쇠를 켜면 급락 구간을 지나기 쉽습니다)")
    if model.beams:
        msgs.append("보 요소는 자체 쉘 솔버에서 무시했습니다")
    if any(model.hinges.values()):
        msgs.append("핀 조인트 힌지는 자체 쉘 솔버에서 지원하지 않아 강접으로 풀었습니다")
    orig = spec
    spec, ctl = disp_control(model, spec)
    if ctl and ctl["kind"] == "average":
        s2 = copy.deepcopy(orig)
        s2["step"]["control_kind"] = "rigid"
        spec, ctl = disp_control(model, s2)
        msgs.append("자체 쉘 솔버의 변위 제어는 강체 압판 방식만 지원해 그 방식으로 풀었습니다")
    if ctl:
        msgs += ctl["notes"]
    T, G = triangulate(model)
    X = model.nodes.copy()
    used = np.zeros(len(X), bool)
    used[T.ravel()] = True
    secs = {g: section_for(spec, g) for g in set(G)}
    mats = spec["materials"]
    tt = np.array([float(secs[g]["thickness"]) for g in G])
    mat0 = mats.get(secs[G[0]].get("material", "default"), mats["default"])
    if any(mats.get(secs[g].get("material", "default"), mats["default"]) is not mat0 for g in G):
        msgs.append("자체 쉘 솔버는 재료를 하나만 써서 첫 그룹의 재료로 풀었습니다")
    if mat0.get("yield"):
        msgs.append("자체 쉘 솔버의 소성은 아직 구현 전이라 탄성으로 풀었습니다")
    E, nu, rho = float(mat0["E"]), float(mat0["nu"]), float(mat0.get("rho") or 0.0)
    table = bc_table(model, spec)
    rotn = {i for (i, d) in table if d > 3}
    fixed, pres, pval = [], [], []
    for (i, d), v in table.items():
        if d <= 3 and used[i]:
            if v == 0:
                fixed.append(3 * i + d - 1)
            else:
                pres.append(3 * i + d - 1)
                pval.append(v)
    free_nodes = np.where(~used)[0]
    fixed += [3 * int(i) + c for i in free_nodes for c in range(3)]
    fixed = np.array(sorted(set(fixed)), int)
    pres = np.array(pres, int)
    pval = np.array(pval, float)
    if rotn:
        msgs.append("회전 구속이 걸린 절점을 잇는 자유 모서리는 고정단(기울기 0)으로 처리했습니다")
    F, M, _ = lumped(model, spec)
    if np.abs(M).sum() > 0:
        msgs.append("모멘트 하중은 자체 쉘 솔버에서 무시했습니다")
    imp = float(st.get("imperfection") or 0)
    if imp > 0:
        sh0 = Shell(X, T, E, nu, tt, clamped=rotn)
        f0 = np.zeros(sh0.ndof)
        f0[: sh0.nx] = lumped(model, orig)[0].ravel()
        tb0 = bc_table(model, orig)
        fx0 = np.array(sorted({3 * i + d - 1 for (i, d) in tb0 if d <= 3} | set(fixed.tolist())), int)
        pv = np.zeros(0)
        pd = np.zeros(0, int)
        if np.abs(f0).sum() == 0 and len(pres):
            fx0 = np.setdiff1d(fx0, pres)
            pd, pv = pres, pval
        fac, modes = linear_buckling(sh0, fx0, pd, pv, f0, k=1)
        if modes:
            X = X + imp * modes[0]
            msgs.append(f"초기 결함: 1차 좌굴 모드(선형 좌굴 계수 {fac[0]:.5g}) x {imp:g} mm")
        else:
            msgs.append("초기 결함용 좌굴 모드를 구하지 못해 결함 없이 풀었습니다")
    sh = Shell(X, T, E, nu, tt, clamped=rotn)
    fext = np.zeros(sh.ndof)
    fext[: sh.nx] = F.ravel()
    fext[fixed] = 0.0
    n = max(1, int(st.get("increments", 25)))
    parts = []
    cs = contacts_of(spec, True)
    ipc = None
    rigid_ids = []
    if cs:
        tmean = float(tt.mean())
        dhat = 0.25 * tmean
        obstacles = []
        groups = None
        selfc = any(c["type"] == "self" for c in cs)
        for c in cs:
            if c["type"] == "rigid":
                off = 0.5 * tmean + dhat
                mv = np.asarray(c.get("motion", [0, 0, 0]), float)
                if c["shape"] == "plane":
                    t1, t2, nn = plane_basis(c["normal"])
                    o = np.asarray(c["point"], float) - nn * off
                    ext = model.diag() * 3.0
                    o = o + 0.0037 * ext * t1 + 0.0023 * ext * t2
                    q = np.array([o - ext * t1 - ext * t2, o + ext * t1 - ext * t2, o + ext * t1 + ext * t2, o - ext * t1 + ext * t2])
                    obstacles.append({"V": q, "F": np.array([[0, 1, 2], [0, 2, 3]]), "motion": mv})
                else:
                    v, f = icosphere(3)
                    r = max(float(c["radius"]) - off, 1e-6)
                    obstacles.append({"V": np.asarray(c["center"], float) + r * v, "F": f, "motion": mv})
                rigid_ids.append(c["id"])
            elif c["type"] == "pair" and not selfc:
                if groups is None:
                    groups = np.zeros(len(X), int)
                    nxt = 1
                for nm in ("a", "b"):
                    ids = resolve(model, c[nm])
                    groups[ids] = nxt
                    nxt += 1
        mu = max(float(c.get("friction") or 0.0) for c in cs)
        if len({float(c.get("friction") or 0.0) for c in cs}) > 1:
            msgs.append(f"접촉마다 마찰계수가 달라 가장 큰 값 {mu:g} 을 모두에 적용했습니다")
        only_rigid = not selfc and groups is None
        ipc = IPCContact(sh, mu=mu, dhat=dhat, groups=groups, self_contact=not only_rigid, obstacles=obstacles)
        parts.append(ipc)
        msgs.append(f"접촉: IPC 장벽법(ipctk), 최소 간격 = 두께 {tmean:g} mm, 장벽 범위 {dhat:g} mm" + (", 자기 접촉 포함" if selfc else "") + (f", 마찰 {mu:g}" if mu > 0 else ""))
        if rigid_ids:
            msgs.append(f"강체 면은 장벽 범위만큼 물러난 위치에서 시작하므로 접촉이 약 {dhat:g} mm 늦게 시작합니다")
    dmp = None
    dspec = st.get("damping") or {}
    kind = dspec.get("kind", "none")
    total_time = 1.0
    if kind != "none":
        if rho <= 0:
            raise ValueError("감쇠를 쓰려면 재료 밀도가 필요합니다")
        allc = np.concatenate([fixed, pres]).astype(int)
        w = frequencies(sh, rho, allc, k=2)
        if kind == "direct":
            al, be = float(dspec.get("alpha") or 0.0), float(dspec.get("beta") or 0.0)
        else:
            al, be = rayleigh(float(dspec.get("zeta") or 0.0) / 100.0, w[0], w[1], kind)
        total_time = float(dspec.get("time") or 0.0)
        auto_t = total_time <= 0
        if auto_t:
            total_time = 100.0 * 2 * np.pi / w[0]
        dmp = Damping(sh, rho, al, be, inertia=bool(dspec.get("inertia", True)))
        parts.append(dmp)
        msgs.append(f"감쇠({ {'mass': '질량 비례', 'stiffness': '강성 비례', 'rayleigh': '레일리', 'direct': '직접 입력'}[kind] }): 1·2차 고유진동수 {w[0] / 2 / np.pi:.4g}, {w[1] / 2 / np.pi:.4g} Hz, α = {al:.4g} 1/s, β = {be:.4g} s, 스텝 시간 {total_time:.4g} s" + (" (자동: 1차 주기의 100배)" if auto_t else "") + ". 음해법 후방 오일러 시간 적분")
    extra = Extras(*parts) if parts else None
    amp = _amp(st.get("amplitude", "linear"))
    res = Result("shell", "static")
    res.nodes = X.copy()
    res.tris = T
    names = ("time", "load_factor", "reaction.x", "reaction.y", "reaction.z", "u.max", "stress.max", "energy.strain", "energy.damping", "energy.kinetic", "contact.count", "contact.gap")
    hist = {k: [] for k in names}
    cforce, cdisp, rig = [], [], {i: [] for i in rigid_ids}
    cdof = np.array([3 * i + ctl["axis"] for i in ctl["nodes"]], int) if ctl else np.zeros(0, int)
    moving = np.concatenate([fixed, pres]).astype(int)
    if len(pres):
        moving = pres

    def on_step(r):
        lam = r["lam"]
        x = r["x"]
        U = x[: sh.nx].reshape(-1, 3) - X
        vm = _vm(sh, x)
        res.add_frame(f"진행률 {lam:.4g}", lam, U, {"vonMises": vm})
        hist["time"].append(float(lam))
        hist["load_factor"].append(float(lam))
        g = r["reaction"]
        for c, nm in enumerate("xyz"):
            hist[f"reaction.{nm}"].append(float(g[moving[moving % 3 == c]].sum()))
        hist["u.max"].append(float(np.linalg.norm(U, axis=1).max()))
        hist["stress.max"].append(float(vm.max()))
        hist["energy.strain"].append(sh.energy(x))
        hist["energy.damping"].append(dmp.dissipated if dmp else 0.0)
        hist["energy.kinetic"].append(dmp.kinetic() if dmp else 0.0)
        if ipc is not None:
            _, cnt, gap = ipc.force(x)
            hist["contact.count"].append(float(cnt))
            hist["contact.gap"].append(float(gap) if np.isfinite(gap) else 0.0)
            for k, cid in enumerate(rigid_ids):
                rig[cid].append((-ipc.plane_force(x, k)).tolist())
        else:
            hist["contact.count"].append(0.0)
            hist["contact.gap"].append(0.0)
        if ctl:
            cforce.append(ctl["sign"] * float(g[cdof].sum()))
            cdisp.append(lam * ctl["amp"])
        if progress:
            progress(lam)

    h = run_static(sh, X, fixed, pres, pval, fext, n, extra=extra, on_step=on_step, time=total_time, amp=amp)
    if not res.frames:
        raise RuntimeError("자체 쉘 해석이 첫 증분에서 수렴하지 않았습니다. 구속과 하중을 확인하세요\n" + "\n".join(msgs))
    if h and h[-1].get("failed"):
        msgs.append(f"진행률 {h[-1]['lam']:.4g} 에서 수렴하지 못해 그 직전까지의 결과만 담았습니다. 증분 수를 늘리거나 감쇠를 켜 보세요")
    res.history = hist
    res.summary = {
        "절점 수": int(used.sum()),
        "삼각형 요소 수": int(len(T)),
        "최대 변위 (mm)": hist["u.max"][-1],
        "최대 von Mises 응력 (MPa, 표면)": hist["stress.max"][-1],
        "변형 에너지 (N·mm)": hist["energy.strain"][-1],
        "프레임 수": len(res.frames),
    }
    if dmp:
        ratio = dmp.dissipated / max(hist["energy.strain"][-1], 1e-30)
        res.summary["감쇠 소산 에너지 (N·mm)"] = dmp.dissipated
        res.summary["감쇠 소산 / 변형 에너지"] = round(ratio, 5)
        msgs.append(f"감쇠가 소산한 에너지는 변형 에너지의 {100 * ratio:.3g}% 입니다" + (". 5%를 넘으면 감쇠가 결과에 영향을 주므로 스텝 시간을 늘리거나 감쇠비를 줄이세요" if ratio > 0.05 else ""))
    if ipc is not None:
        res.summary["접촉 쌍 수 (끝)"] = int(hist["contact.count"][-1])
        if rigid_ids:
            contact_record(res, spec, hist["time"], rig)
    if ctl:
        ctl_record(res, ctl, None, cforce, cdisp)
    res.summary["참고"] = msgs
    res.log = "\n".join(msgs)
    return res
