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


def _skew(r):
    return np.array([[0.0, -r[2], r[1]], [r[2], 0.0, -r[0]], [-r[1], r[0], 0.0]])


def _coupled_linear(model, spec, cps, pts, mids, K, F, mask, val, E, n):
    import scipy.sparse as sp
    from scipy.sparse.linalg import splu

    from ..select import resolve

    Ks = K.to_scipy_coo().tocsr()
    N0 = 3 * n
    N = N0 + 6 * len(cps)
    edges = np.array(list(mids.keys()))
    h = float(np.linalg.norm(pts[edges[:, 0]] - pts[edges[:, 1]], axis=1).mean()) if len(edges) else 1.0
    k = 1e3 * E * h
    rows, cols, vals = [], [], []

    def blk(ri, ci, M):
        rows.append(np.repeat(ri, len(ci)))
        cols.append(np.tile(ci, len(ri)))
        vals.append(np.asarray(M).ravel())

    f = np.zeros(N)
    f[:N0] = F.ravel()
    m2 = np.zeros(N, bool)
    m2[:N0] = mask
    v2 = np.zeros(N)
    v2[:N0] = val
    rp = {}
    I = np.eye(3)
    for j, cp in enumerate(cps):
        sel = {int(i) for i in resolve(model, cp["select"])}
        ids = sorted(sel | {m for (a, b), m in mids.items() if a in sel and b in sel})
        ref = np.asarray(cp["ref"], float)
        base = N0 + 6 * j
        U = np.arange(base, base + 3)
        Q = np.arange(base + 3, base + 6)
        S = np.zeros((3, 3))
        SS = np.zeros((3, 3))
        for i in ids:
            Rk = _skew(pts[i] - ref)
            X3 = 3 * i + np.arange(3)
            blk(X3, X3, k * I)
            blk(X3, U, -k * I)
            blk(U, X3, -k * I)
            blk(X3, Q, k * Rk)
            blk(Q, X3, k * Rk.T)
            S += Rk
            SS += Rk.T @ Rk
        blk(U, U, k * len(ids) * I)
        blk(U, Q, -k * S)
        blk(Q, U, -k * S.T)
        blk(Q, Q, k * SS)
        f[base : base + 3] = cp.get("force") or [0.0, 0.0, 0.0]
        f[base + 3 : base + 6] = cp.get("moment") or [0.0, 0.0, 0.0]
        for d, v in (cp.get("fix") or {}).items():
            m2[base + int(d) - 1] = True
            v2[base + int(d) - 1] = float(v)
        rp[cp["id"]] = base
    Kc = sp.coo_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N)).tocsr()
    Ka = sp.bmat([[Ks, None], [None, sp.csr_matrix((N - N0, N - N0))]]).tocsr() + Kc
    free = np.where(~m2)[0]
    u = v2.copy()
    rhs = f - Ka @ u
    u[free] = splu(Ka[free][:, free].tocsc()).solve(rhs[free])
    g = Ka @ u
    reac = g - f
    out = {"k": k, "f": f[N0:], "rp": {cid: (base, g[base : base + 6]) for cid, base in rp.items()}}
    return u, reac, out


def solve(model, spec, progress=None):
    if len(model.solids) == 0:
        raise ValueError("TensorMesh 솔버에는 솔리드(STL) 모델이 필요합니다")
    from ..control import disp_control, record as ctl_record

    from ..spec import solid_rotations

    spec, rot_notes = solid_rotations(model, spec)
    cps = spec.get("couplings") or []
    orig = spec
    spec, ctl = disp_control(model, spec)
    st = dict(spec["step"])
    if cps and (st.get("nlgeom") or st["type"] == "arclength"):
        from . import calculix

        if not calculix.find_ccx():
            raise ValueError("기준점(커플링)이 있는 솔리드의 기하비선형 해석은 CalculiX 가 필요합니다")
        r = calculix.solve(model, model, orig)
        r.summary.setdefault("참고", []).insert(0, "TensorMesh 의 기하비선형 경로에는 기준점이 없어 CalculiX(C3D10 + *RIGID BODY)로 풀었습니다")
        return r
    if st["type"] in ("buckle", "modal"):
        raise ValueError("TensorMesh 어댑터는 현재 정적 해석(선형, 기하비선형)만 지원합니다. 좌굴·모달은 CalculiX 솔버를 선택하세요")
    pre_notes = list(rot_notes)
    if st["type"] == "arclength":
        st["nlgeom"] = True
        if not ctl:
            pre_notes.append("TensorMesh 어댑터에는 호장법이 없어 하중 제어 비선형 정적 해석으로 풀었습니다. 극한점 이후 거동은 변위 제어를 켜야 따라갑니다")
    if ctl:
        pre_notes += ctl["notes"]
        if not st.get("nlgeom"):
            pre_notes.append("기하비선형이 꺼져 있어 좌굴 이후 거동은 나오지 않습니다 (선형 변위 제어)")
    from ..contact import contacts_of

    has_contact = bool(contacts_of(spec))
    if has_contact and not st.get("nlgeom"):
        st["nlgeom"] = True
        pre_notes.append("접촉이 있어 비선형(Neo-Hookean) 경로로 풀었습니다")
    imp = float(st.get("imperfection") or 0)
    if st.get("nlgeom") and imp > 0:
        from . import calculix

        mode = calculix.buckle_mode(model, model, orig) if calculix.find_ccx() else None
        if mode is not None:
            model = model.copy()
            model.nodes = model.nodes + mode * imp
            pre_notes.append(f"초기 결함: 1차 좌굴 모드(CalculiX로 계산) x {imp:g} mm")
        else:
            pre_notes.append("초기 결함용 좌굴 모드를 구하지 못해 결함 없이 풀었습니다 (CalculiX 필요)")
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
    if not mask.any() and not any(cp.get("fix") for cp in cps):
        raise ValueError("구속이 없습니다")
    asm = LinearElasticityElementAssembler.from_mesh(mesh, E=E, nu=nu)
    K = asm(mesh.points)
    dt = mesh.dtype
    f = torch.tensor(F.ravel(), dtype=dt)
    res = Result("tensormesh", "static")
    res.nodes = model.nodes.copy()
    res.tris = model.solid_boundary()
    _e = {(min(a, b), max(a, b)) for t in res.tris.tolist() for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0]))}
    res.fedges = np.array(sorted(_e), dtype=int).reshape(-1, 2)
    hist = {k: [] for k in ("time", "u.max", "vonMises.max", "energy.total", "reaction.x", "reaction.y", "reaction.z")}
    moving = np.abs(val).sum() > 0
    rsel = (val != 0) if moving else mask
    notes = list(pre_notes)
    cdof = np.zeros(0, dtype=int)
    cforce = []
    fc = None
    cdisp = []
    if ctl and ctl["kind"] == "rigid":
        cn = set(ctl["nodes"])
        ids = list(cn) + [m for (i, j), m in mids.items() if i in cn and j in cn]
        cdof = np.array([3 * i + ctl["axis"] for i in ids if mask[3 * i + ctl["axis"]]], dtype=int)
    elif ctl:
        fcn = nodal_forces(model, ctl["one"], pts, t10, mids, float(mat["rho"]))[0].ravel().copy()
        fcn[mask] = 0.0
        if np.abs(fcn).sum() == 0:
            raise ValueError("변위 제어 하중이 걸린 자유도가 모두 구속되어 있습니다")
        fc = torch.tensor(fcn, dtype=dt)
        cvec = fcn / ctl["N"]

    lamc = [0.0]
    tmc = None
    cfh = {}
    if has_contact:
        from ..contact import TMContact

        tmc = TMContact(model, spec, pts, mids, torch, dt)
        notes += tmc.notes

    def record(lam, U, energy, reac, finite):
        if tmc is not None:
            for cid, v in tmc.forces(U, lam).items():
                cfh.setdefault(cid, []).append(v)
            tmc.commit(U, lam)
        fields = {k: v[:n0] for k, v in stress_fields(corner_gradients(pts, t10, U), E, nu, finite).items()}
        res.add_frame(f"하중계수 {lam:.4g}", lam, U[:n0], fields)
        hist["time"].append(float(lam))
        hist["u.max"].append(float(np.linalg.norm(U[:n0], axis=1).max()))
        hist["vonMises.max"].append(float(fields["vonMises"].max()))
        hist["energy.total"].append(float(energy))
        r = np.where(rsel, reac, 0.0).reshape(-1, 3).sum(axis=0)
        if ctl and fc is None:
            cforce.append(ctl["sign"] * float(np.asarray(reac)[cdof].sum()))
            cdisp.append(float(lam) * ctl["amp"])
        elif ctl:
            cforce.append(float(lamc[0]) * ctl["N"])
            cdisp.append(float(cvec @ np.asarray(U).ravel()))
        for c, nm in enumerate("xyz"):
            hist[f"reaction.{nm}"].append(float(r[c]))

    if cps and not st.get("nlgeom"):
        u, reac_all, cp_out = _coupled_linear(model, spec, cps, pts, mids, K, F, mask, val, E, n)
        record(1.0, u[: 3 * n].reshape(n, 3), 0.5 * float(u @ (reac_all + np.concatenate([F.ravel(), cp_out["f"]]))), reac_all[: 3 * n], False)
        for cid, (base, vals) in cp_out["rp"].items():
            for k, nm in enumerate(("ux", "uy", "uz", "rx", "ry", "rz")):
                hist[f"rp.{cid}.{nm}"] = [float(u[base + k])]
            for k, nm in enumerate(("fx", "fy", "fz", "mx", "my", "mz")):
                hist[f"rp.{cid}.{nm}"] = [float(vals[k])]
        pre_notes.append(f"기준점 커플링 {len(cps)}개: 묶인 절점의 병진을 기준점의 강체 운동(소변형)에 벌칙 강성 {cp_out['k']:.3g} N/mm 로 묶었습니다 (선형)")
        notes[:] = list(pre_notes)
    elif not st.get("nlgeom"):
        cond = Condenser(torch.tensor(mask), torch.tensor(val, dtype=dt))
        Ki, fi = cond(K, f)
        u = cond.recover(Ki.solve(fi))
        ftot = f
        if fc is not None:
            c0 = Condenser(torch.tensor(mask), torch.zeros(n * 3, dtype=dt))
            K2, f2 = c0(K, fc)
            u2 = c0.recover(K2.solve(f2))
            ct = torch.tensor(cvec, dtype=dt)
            lamc[0] = float((ctl["amp"] - ct @ u) / (ct @ u2))
            u = u + lamc[0] * u2
            ftot = f + lamc[0] * fc
        Ku = K @ u
        record(1.0, u.detach().numpy().reshape(n, 3), 0.5 * float(u @ Ku), (Ku - ftot).detach().numpy(), False)
    else:
        from scipy.sparse.linalg import LinearOperator, cg, splu

        model_nh = NeoHookeanModel.from_mesh(mesh, E=E, nu=nu)
        free = np.where(~mask)[0]
        Kff = K.to_scipy_coo().tocsr()[free][:, free].tocsc()
        if tmc is not None:
            from scipy.sparse import identity

            Kff = Kff + identity(len(free), format="csc") * (1e-7 * float(Kff.diagonal().mean()))
        pre = splu(Kff)
        M = LinearOperator((len(free), len(free)), matvec=pre.solve)
        ft = torch.tensor(free)
        u = torch.zeros(n * 3, dtype=dt)
        inc = max(1, int(st.get("increments", 25)))
        fnorm = max(float(np.linalg.norm(F)), 1e-12)

        cur = [0.0]

        def grad(uv, graph=False):
            x = uv.reshape(n, 3).clone().requires_grad_(True)
            en = model_nh.energy(x)
            if tmc is not None:
                en = en + tmc.energy(x, cur[0])
            (g,) = torch.autograd.grad(en, x, create_graph=graph)
            return en, g, x

        lam_done = 0.0
        step = 1.0 / inc
        cuts = 0
        lc_done = 0.0
        ct = torch.tensor(cvec, dtype=dt) if fc is not None else None
        fcn_free = fc[ft].numpy() if fc is not None else None
        while lam_done < 1 - 1e-12:
            lam = min(1.0, lam_done + step)
            cur[0] = lam
            trial = u.clone()
            trial[torch.tensor(np.where(mask)[0])] = torch.tensor(lam * val[mask], dtype=dt)
            ok = False
            ref = None
            lc = lc_done
            for it in range(30):
                en, g, x = grad(trial, graph=True)
                r = (lam * f - g.reshape(-1))[ft] if fc is None else (lam * f + lc * fc - g.reshape(-1))[ft]
                rn = float(r.norm())
                if not np.isfinite(rn):
                    break
                qe = 0.0 if fc is None else lam * ctl["amp"] - float(ct @ trial)
                if ref is None:
                    ref = max(rn, fnorm * lam, abs(lc) * float(fc.norm()) if fc is not None else 0.0, 1e-12)
                if fc is not None:
                    ref = max(ref, abs(lc) * float(fc.norm()))
                if rn <= 1e-6 * ref and abs(qe) <= 1e-8 * ctl["amp"] if fc is not None else rn <= 1e-6 * ref:
                    ok = True
                    break

                def hv(v):
                    w = torch.zeros(n * 3, dtype=dt)
                    w[ft] = torch.tensor(np.asarray(v).ravel(), dtype=dt)
                    (h,) = torch.autograd.grad((g.reshape(-1) * w).sum(), x, retain_graph=True)
                    return h.reshape(-1)[ft].numpy()

                Hop = LinearOperator((len(free), len(free)), matvec=hv)
                du, _ = cg(Hop, r.detach().numpy(), M=M, rtol=1e-8, maxiter=200)
                if fc is not None:
                    dut, _ = cg(Hop, fcn_free, M=M, rtol=1e-8, maxiter=200)
                    cf = cvec[free]
                    den = float(cf @ dut)
                    if abs(den) < 1e-300:
                        break
                    dlc = (qe - float(cf @ du)) / den
                    t2 = trial.clone()
                    t2[ft] += torch.tensor(du + dlc * dut, dtype=dt)
                    trial = t2.detach()
                    lc += dlc
                    continue
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
            lc_done = lc
            lamc[0] = lc
            en, g, _ = grad(u)
            record(lam, u.numpy().reshape(n, 3), float(en), (g.reshape(-1) - lam * f - (lc * fc if fc is not None else 0.0)).detach().numpy(), True)
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
    for cp in cps:
        c = cp["id"]
        if f"rp.{c}.ux" in hist:
            res.summary[f"기준점 {c} 변위 (mm)"] = [round(hist[f"rp.{c}.u{a}"][-1], 6) for a in "xyz"]
            res.summary[f"기준점 {c} 회전 (rad)"] = [round(hist[f"rp.{c}.r{a}"][-1], 6) for a in "xyz"]
            res.summary[f"기준점 {c} 힘 (N)"] = [round(hist[f"rp.{c}.f{a}"][-1], 4) for a in "xyz"]
            res.summary[f"기준점 {c} 모멘트 (N·mm)"] = [round(hist[f"rp.{c}.m{a}"][-1], 3) for a in "xyz"]
    if tmc is not None:
        from ..contact import record as crecord

        crecord(res, spec, hist["time"], cfh)
    if ctl:
        ctl_record(res, ctl, None, cforce, cdisp)
    if notes:
        res.summary["참고"] = notes
    res.log = "\n".join(notes)
    return res
