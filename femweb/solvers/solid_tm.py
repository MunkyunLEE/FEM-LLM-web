import warnings
import numpy as np
from ..loads import assemble, bc_table
from ..result import Result, von_mises

MIDS = {(0, 1): 4, (1, 2): 5, (0, 2): 6, (0, 3): 7, (1, 3): 8, (2, 3): 9}


def quadratic(model):
    pts = [p for p in model.nodes]
    mids = {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in mids:
            pts.append(0.5 * (pts[a] + pts[b]))
            mids[k] = len(pts) - 1
        return mids[k]

    t10 = [[a, b, c, d, mid(a, b), mid(b, c), mid(a, c), mid(a, d), mid(b, d), mid(c, d)] for a, b, c, d in model.solids.tolist()]
    return np.array(pts), np.array(t10, dtype=np.int64), mids


def nodal_forces(model, spec, pts, t10, mids, rho):
    n = len(pts)
    F = np.zeros((n, 3))
    a = assemble(model, spec)
    for i, v in a["node"]:
        F[i] += v
    for (i, j), v in a["edge"]:
        k = (min(i, j), max(i, j))
        if k in mids:
            F[i] += v / 6
            F[j] += v / 6
            F[mids[k]] += v * 2 / 3
        else:
            F[i] += v / 2
            F[j] += v / 2
    for f, v in a["face"]:
        for q in range(3):
            k = (min(f[q], f[(q + 1) % 3]), max(f[q], f[(q + 1) % 3]))
            F[mids[k]] += v / 3
    if a["gravity"] is not None:
        p = pts[t10[:, :4]]
        vol = np.abs(np.einsum("ij,ij->i", np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]), p[:, 3] - p[:, 0])) / 6.0
        w = (rho * vol)[:, None] * a["gravity"][None]
        for c in range(4):
            np.add.at(F, t10[:, c], -w / 20)
        for c in range(4, 10):
            np.add.at(F, t10[:, c], w / 5)
    return F, a


def corner_gradients(pts, t10, U):
    M = len(t10)
    X = pts[t10[:, :4]]
    A = np.concatenate([np.ones((M, 4, 1)), X], axis=2)
    inv = np.linalg.inv(A)
    gL = inv[:, 1:4, :].transpose(0, 2, 1)
    u = U[t10]
    n = int(t10[:, :4].max()) + 1
    G = np.zeros((n, 3, 3))
    cnt = np.zeros(n)
    for c in range(4):
        g = 3 * np.einsum("mi,mj->mij", u[:, c], gL[:, c])
        for j in range(4):
            if j == c:
                continue
            g -= np.einsum("mi,mj->mij", u[:, j], gL[:, j])
            g += 4 * np.einsum("mi,mj->mij", u[:, MIDS[(min(c, j), max(c, j))]], gL[:, j])
        np.add.at(G, t10[:, c], g)
        np.add.at(cnt, t10[:, c], 1)
    return G / np.maximum(cnt, 1)[:, None, None]


def stress_fields(G, E, nu, finite=False):
    lam = E * nu / ((1 + nu) * (1 - 2 * nu))
    mu = E / (2 * (1 + nu))
    I = np.eye(3)[None]
    if finite:
        F = I + G
        J = np.clip(np.linalg.det(F), 1e-9, None)
        sig = (mu / J)[:, None, None] * (F @ F.transpose(0, 2, 1) - I) + (lam * np.log(J) / J)[:, None, None] * I
    else:
        eps = 0.5 * (G + G.transpose(0, 2, 1))
        sig = lam * np.trace(eps, axis1=1, axis2=2)[:, None, None] * I + 2 * mu * eps
    S = np.stack([sig[:, 0, 0], sig[:, 1, 1], sig[:, 2, 2], sig[:, 0, 1], sig[:, 1, 2], sig[:, 2, 0]], axis=1)
    out = {"vonMises": von_mises(S)}
    for c, nm in enumerate(("Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx")):
        out[nm] = S[:, c]
    return out


def solve(model, spec, progress=None):
    if len(model.solids) == 0:
        raise ValueError("TensorMesh 솔버에는 솔리드(STL) 모델이 필요합니다")
    st = spec["step"]
    if st["type"] in ("buckle", "modal", "arclength"):
        raise ValueError("TensorMesh 어댑터는 현재 정적 해석(선형, 기하비선형)만 지원합니다. 좌굴·모달은 CalculiX 솔버를 선택하세요")
    mat = spec["materials"].get(spec["sections"]["*"].get("material", "default"), spec["materials"]["default"])
    if mat.get("yield"):
        raise ValueError("TensorMesh 어댑터의 소성 해석은 아직 연결되지 않았습니다. CalculiX 솔버를 선택하세요")
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            import meshio
            import torch
            import tensormesh as tm
            from tensormesh import Condenser
            from tensormesh.assemble import LinearElasticityElementAssembler, NeoHookeanModel
    except Exception as e:
        raise RuntimeError(f"TensorMesh를 불러올 수 없습니다 (pip install torch tensormesh-fem): {e}")
    E, nu = float(mat["E"]), float(mat["nu"])
    pts, t10, mids = quadratic(model)
    n = len(pts)
    n0 = len(model.nodes)
    mesh = tm.Mesh(meshio.Mesh(pts, [("tetra10", t10)]), reorder=True)
    F, a = nodal_forces(model, spec, pts, t10, mids, float(mat["rho"]))
    table = bc_table(model, spec)
    mask = np.zeros(n * 3, dtype=bool)
    val = np.zeros(n * 3)
    for (i, d), v in table.items():
        if d <= 3:
            mask[3 * i + d - 1] = True
            val[3 * i + d - 1] = v
    for (i, j), m in mids.items():
        for d in range(3):
            if mask[3 * i + d] and mask[3 * j + d]:
                mask[3 * m + d] = True
                val[3 * m + d] = 0.5 * (val[3 * i + d] + val[3 * j + d])
    if not mask.any():
        raise ValueError("구속이 없습니다")
    asm = LinearElasticityElementAssembler.from_mesh(mesh, E=E, nu=nu)
    K = asm(mesh.points)
    dt = mesh.dtype
    f = torch.tensor(F.ravel(), dtype=dt)
    res = Result("tensormesh", "static")
    res.nodes = model.nodes.copy()
    res.tris = model.solid_boundary()
    hist = {k: [] for k in ("time", "u.max", "vonMises.max", "energy.total", "reaction.x", "reaction.y", "reaction.z")}
    moving = np.abs(val).sum() > 0
    rsel = (val != 0) if moving else mask
    notes = []

    def record(lam, U, energy, reac, finite):
        fields = {k: v[:n0] for k, v in stress_fields(corner_gradients(pts, t10, U), E, nu, finite).items()}
        res.add_frame(f"하중계수 {lam:.4g}", lam, U[:n0], fields)
        hist["time"].append(float(lam))
        hist["u.max"].append(float(np.linalg.norm(U[:n0], axis=1).max()))
        hist["vonMises.max"].append(float(fields["vonMises"].max()))
        hist["energy.total"].append(float(energy))
        r = np.where(rsel, reac, 0.0).reshape(-1, 3).sum(axis=0)
        for c, nm in enumerate("xyz"):
            hist[f"reaction.{nm}"].append(float(r[c]))

    if not st.get("nlgeom"):
        cond = Condenser(torch.tensor(mask), torch.tensor(val, dtype=dt))
        Ki, fi = cond(K, f)
        u = cond.recover(Ki.solve(fi))
        Ku = K @ u
        record(1.0, u.detach().numpy().reshape(n, 3), 0.5 * float(u @ Ku), (Ku - f).detach().numpy(), False)
    else:
        from scipy.sparse.linalg import LinearOperator, cg, splu

        model_nh = NeoHookeanModel.from_mesh(mesh, E=E, nu=nu)
        free = np.where(~mask)[0]
        pre = splu(K.to_scipy_coo().tocsr()[free][:, free].tocsc())
        M = LinearOperator((len(free), len(free)), matvec=pre.solve)
        ft = torch.tensor(free)
        u = torch.zeros(n * 3, dtype=dt)
        inc = max(1, int(st.get("increments", 25)))
        fnorm = max(float(np.linalg.norm(F)), 1e-12)

        def grad(uv, graph=False):
            x = uv.reshape(n, 3).clone().requires_grad_(True)
            en = model_nh.energy(x)
            (g,) = torch.autograd.grad(en, x, create_graph=graph)
            return en, g, x

        lam_done = 0.0
        step = 1.0 / inc
        cuts = 0
        while lam_done < 1 - 1e-12:
            lam = min(1.0, lam_done + step)
            trial = u.clone()
            trial[torch.tensor(np.where(mask)[0])] = torch.tensor(lam * val[mask], dtype=dt)
            ok = False
            ref = None
            for it in range(30):
                en, g, x = grad(trial, graph=True)
                r = (lam * f - g.reshape(-1))[ft]
                rn = float(r.norm())
                if not np.isfinite(rn):
                    break
                if ref is None:
                    ref = max(rn, fnorm * lam, 1e-12)
                if rn <= 1e-6 * ref:
                    ok = True
                    break

                def hv(v):
                    w = torch.zeros(n * 3, dtype=dt)
                    w[ft] = torch.tensor(np.asarray(v).ravel(), dtype=dt)
                    (h,) = torch.autograd.grad((g.reshape(-1) * w).sum(), x, retain_graph=True)
                    return h.reshape(-1)[ft].numpy()

                du, _ = cg(LinearOperator((len(free), len(free)), matvec=hv), r.detach().numpy(), M=M, rtol=1e-8, maxiter=200)
                s = 1.0
                for _ in range(8):
                    t2 = trial.clone()
                    t2[ft] += s * torch.tensor(du, dtype=dt)
                    _, g2, _ = grad(t2)
                    r2 = float((lam * f - g2.reshape(-1))[ft].norm())
                    if np.isfinite(r2) and r2 < rn:
                        break
                    s *= 0.5
                trial = t2.detach()
            if not ok:
                cuts += 1
                step *= 0.5
                if cuts > 8:
                    notes.append(f"하중계수 {lam_done:.4g} 이후 수렴 실패")
                    break
                continue
            u = trial.detach()
            lam_done = lam
            en, g, _ = grad(u)
            record(lam, u.numpy().reshape(n, 3), float(en), (g.reshape(-1) - lam * f).detach().numpy(), True)
        if not res.frames:
            raise RuntimeError("TensorMesh 비선형 해석이 수렴하지 않았습니다")
        notes.append("기하비선형: Neo-Hookean 모델, 선형 강성을 전처리기로 쓴 Newton-CG")
    res.history = hist
    res.summary = {
        "절점 수 (2차 사면체)": int(n),
        "요소 수": int(len(t10)),
        "최대 변위 (mm)": hist["u.max"][-1],
        "최대 von Mises (MPa)": hist["vonMises.max"][-1],
        "변형에너지 (N·mm)": hist["energy.total"][-1],
        "반력 합 (N)": [round(hist[f"reaction.{c}"][-1], 4) for c in "xyz"],
        "프레임 수": len(res.frames),
    }
    if notes:
        res.summary["참고"] = notes
    res.log = "\n".join(notes)
    return res
