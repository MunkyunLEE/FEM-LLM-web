import numpy as np

from bhweb.analysis import dof_analysis, fold_angles, modal, null_modes, rigid_fold_path, solution_space
from bhweb.contact import GeometricContact
from bhweb.model import BEND, FOLD, PANEL, BHModel
from bhweb.solve import Problem, solve_displacement, solve_force

from ..loads import bc_table, lumped
from ..result import Result
from ..spec import material_for, section_for

BH_DEFAULT = {
    "fold_mode": "auto",
    "Lstar": None,
    "Lstar_ratio": 3.0,
    "fold_percent": 10.0,
    "fold_k": None,
    "long_factor": 100.0,
    "bar_area": None,
    "tri_bars": "woodruff",
    "bar_yield": None,
    "bar_H": 0.0,
    "fold_yield": None,
    "fold_H": 0.0,
    "bend_yield": None,
    "bend_H": 0.0,
    "contact": False,
    "gap": None,
    "tol": 3.0,
    "cycles": 0,
    "cycle_low": 0.0,
    "use_mass": True,
    "rest": {},
    "drivers": [],
    "targets": [],
    "plot3": [],
    "watch": [],
}


def ekey(e):
    a, b = sorted(int(v) for v in e)
    return f"{a}-{b}"


def parse_key(k):
    a, b = (int(v) for v in str(k).split("-"))
    return (min(a, b), max(a, b))


def cfg(spec):
    c = dict(BH_DEFAULT)
    c.update(spec.get("bh") or {})
    return c


def build(model, spec):
    if not model.shells:
        raise ValueError("BH-web 솔버에는 면(f) 패널이 필요합니다")
    c = cfg(spec)
    g = model.shell_group[0]
    mat = material_for(spec, g)
    t = float(section_for(spec, g)["thickness"])
    inf = float("inf")
    params = {
        "E": float(mat["E"]),
        "nu": float(mat["nu"]),
        "t": t,
        "rho": float(mat.get("rho") or 1e-9),
        "fold_mode": c["fold_mode"],
        "Lstar": c["Lstar"],
        "Lstar_ratio": c["Lstar_ratio"],
        "fold_percent": c["fold_percent"],
        "fold_k": c["fold_k"],
        "long_factor": c["long_factor"],
        "bar_area": c["bar_area"],
        "tri_bars": c["tri_bars"],
        "bar_yield": inf if c["bar_yield"] is None else float(c["bar_yield"]),
        "bar_H": float(c["bar_H"] or 0.0),
        "fold_yield": inf if c["fold_yield"] is None else float(c["fold_yield"]),
        "fold_H": float(c["fold_H"] or 0.0),
        "bend_yield": inf if c["bend_yield"] is None else float(c["bend_yield"]),
        "bend_H": float(c["bend_H"] or 0.0),
    }
    types = {parse_key(k): v for k, v in (spec.get("creases") or {}).items()}
    crease = [e for es in model.hinges.values() for e in es]
    m = BHModel(model.nodes, model.shells, types=types, params=params, crease_edges=crease, tol=float(c["tol"]))
    return m


def crease_view(model, spec):
    c = cfg(spec)
    from bhweb.model import classify

    crease = [e for es in model.hinges.values() for e in es]
    auto, angles, note = classify(model.nodes, model.shells, crease, float(c["tol"]))
    over = {parse_key(k): v for k, v in (spec.get("creases") or {}).items()}
    out = []
    for e, ty in sorted(auto.items()):
        out.append({"edge": list(e), "auto": ty, "type": over.get(e, ty) if ty != "boundary" else "boundary", "angle": round(angles.get(e, 0.0), 4)})
    return {"edges": out, "note": note}


def _dofs(m, model, spec):
    table = bc_table(model, spec)
    orig = m.orig
    copies = {}
    for i, o in enumerate(orig):
        copies.setdefault(int(o), []).append(i)
    fixed = []
    disp = {}
    for (i, d), v in table.items():
        if d > 3:
            continue
        for j in copies.get(int(i), [int(i)]):
            dof = 3 * j + d - 1
            if v == 0.0:
                fixed.append(dof)
            else:
                disp[dof] = v
    F, M, _ = lumped(model, spec)
    force = {}
    for i in range(len(model.nodes)):
        for d in range(3):
            if F[i, d] != 0.0:
                force[3 * i + d] = float(F[i, d])
    return fixed, disp, force, bool(np.abs(M).sum() > 0)


def _rest(m, c):
    out = {}
    for k, v in (c.get("rest") or {}).items():
        i = m.fold_index(parse_key(k))
        out[i] = np.radians(float(v)) - m.th0[i]
    return out


def _frame_fields(m, x, st):
    d = x[m.bars[:, 1]] - x[m.bars[:, 0]]
    eps = np.abs(np.log(np.linalg.norm(d, axis=1) / m.L0))
    nodal = np.zeros(m.nn)
    cnt = np.zeros(m.nn)
    for q in (0, 1):
        np.add.at(nodal, m.bars[:, q], eps)
        np.add.at(cnt, m.bars[:, q], 1)
    th = fold_angles(m, x, st.th if st is not None else None)
    fold = m.hkind == FOLD
    rot = np.zeros(m.nn)
    rc = np.zeros(m.nn)
    if fold.any():
        dth = np.degrees(np.abs(th - m.th_init))[fold]
        for q in (1, 2):
            np.add.at(rot, m.h[fold][:, q], dth)
            np.add.at(rc, m.h[fold][:, q], 1)
    f = {"barStrain": nodal / np.maximum(cnt, 1), "foldRotation": rot / np.maximum(rc, 1)}
    if st is not None:
        pp = np.zeros(m.nn)
        np.add.at(pp, m.h[:, 1], np.degrees(np.abs(st.thp - m.th0)))
        np.add.at(pp, m.h[:, 2], np.degrees(np.abs(st.thp - m.th0)))
        hc = np.zeros(m.nn)
        np.add.at(hc, m.h[:, 1], 1)
        np.add.at(hc, m.h[:, 2], 1)
        f["hingePlastic"] = pp / np.maximum(hc, 1)
        bp = np.zeros(m.nn)
        for q in (0, 1):
            np.add.at(bp, m.bars[:, q], np.abs(st.ep))
        f["barPlastic"] = bp / np.maximum(cnt, 1)
    return f


def _base(m, kind):
    res = Result("bh", kind)
    res.nodes = m.X.copy()
    res.tris = m.tris.copy()
    fe = [(int(m.h[i, 1]), int(m.h[i, 2])) for i in range(len(m.h)) if m.hkind[i] == FOLD]
    eds = {}
    for f in m.faces:
        n = len(f)
        for i in range(n):
            a, b = f[i], f[(i + 1) % n]
            eds[(min(a, b), max(a, b))] = 1
    res.lines = np.array(list(eds.keys()), int).reshape(-1, 2)
    res.fedges = np.array(fe, int).reshape(-1, 2)
    return res


def _summary_model(m):
    s = m.summary()
    return {
        "절점 수 (중심 절점 포함)": s["nodes"],
        "바 수": s["bars"],
        "접힘 / 패널 휨 / 휨 엣지 힌지": [s["folds"], s["panel_bends"], s["edge_bends"]],
        "판 굽힘강성 B = Et³/12(1-ν²) (N·mm)": s["B"],
        "L* (mm)": s["Lstar"],
        "자동 판정": m.note,
    }


def cycle_amp(s, r=0.0):
    s = s % 1.0
    if r == 0:
        return 1.0 - abs(1.0 - 2.0 * s)
    if s < 0.25:
        return s / 0.25
    if s < 0.75:
        return 1.0 - (1.0 - r) * (s - 0.25) / 0.5
    return r * (1.0 - (s - 0.75) / 0.25)


def solve(model, spec, progress=None):
    st = spec["step"]
    kind = st["type"]
    c = cfg(spec)
    m = build(model, spec)
    fixed, disp, force, has_moment = _dofs(m, model, spec)
    msgs = []
    if has_moment:
        msgs.append("바-힌지 모델에는 회전 자유도가 없어 모멘트 하중은 무시했습니다")
    allfix = sorted(set(fixed) | set(disp))
    if kind == "modal":
        return _modal(m, allfix, int(st.get("modes", 12)), msgs, bool(c.get("use_mass", True)))
    if kind == "kinematic":
        if disp:
            msgs.append(f"기구 해석은 구동 주름 각도로 움직이므로 0 이 아닌 지정 변위 {len(disp)}개 자유도는 쓰지 않았습니다 (0 구속만 사용)")
        return _kinematic(m, sorted(set(fixed)), c, int(st.get("increments", 60)), msgs)
    cb = (lambda lam, k: progress(lam)) if progress else None
    ct = GeometricContact(m, gap=c["gap"]) if c["contact"] else None
    if ct is not None:
        msgs.append(f"기하 접촉: 최소 간격 {ct.gap:.4g} mm (마찰·감쇠 없음), 처음 형상에서 이미 장벽 안에 든 쌍 {ct.rest_pairs}개")
    rest = _rest(m, c)
    n = max(1, int(st.get("increments", 25)))
    cyc = int(c.get("cycles") or 0)
    path = None
    if cyc > 0 and kind != "arclength":
        if not disp:
            raise ValueError("반복 하중(cycles)은 지정 변위가 있어야 합니다")
        dd = sorted(disp)
        vv = np.array([disp[q] for q in dd], float)
        r = float(c.get("cycle_low") or 0.0)
        path = (dd, lambda lam, vv=vv, cyc=cyc, r=r: vv * cycle_amp(lam * cyc, r) if lam < 1.0 else 0.0 * vv)
        disp = {}
        n = n * (2 if r == 0 else 4) * cyc
        msgs.append(f"지정 변위를 0 → 목표 → {'0' if r == 0 else f'목표×{r:g}'} → 0 으로 {cyc}회 반복합니다 (증분 {n})")
    pb = Problem(m, fixed=fixed, disp=disp, force=force, rest=rest or None, contact=ct, path=path)
    if kind == "arclength":
        if not force:
            raise ValueError("호장법(MGDCM)에는 힘 하중이 필요합니다")
        rec = solve_force(m, pb, increments=n, dlam=1.0 / max(4, n // 4), progress=cb)
    else:
        if not disp and not force and not rest and path is None:
            raise ValueError("하중, 지정 변위, 주름 목표각 중 하나가 필요합니다")
        rec = solve_displacement(m, pb, steps=n, progress=cb)
    msgs += rec.notes
    return _static(m, rec, pb, c, msgs, kind)


def _static(m, rec, pb, c, msgs, kind):
    res = _base(m, "static")
    keys = ("time", "load_factor", "energy.stretch", "energy.fold", "energy.bend", "energy.plastic", "energy.contact", "energy.total", "energy.external", "reaction.x", "reaction.y", "reaction.z", "reaction.control", "u.control", "u.max", "fold.mean", "contact.pairs", "iterations")
    hist = {k: [] for k in keys}
    watch = [parse_key(k) for k in (c.get("watch") or [])]
    widx = []
    for e in watch:
        try:
            widx.append((e, m.fold_index(e)))
        except KeyError:
            msgs.append(f"관찰 엣지 {ekey(e)} 에 힌지가 없습니다")
    for e, _ in widx:
        hist[f"fold.e{ekey(e)}"] = []
        hist[f"moment.e{ekey(e)}"] = []
    fold = m.hkind == FOLD
    ctrl = np.where(pb.up != 0)[0]
    if pb.path is not None:
        ctrl = np.unique(np.concatenate([ctrl, np.asarray(pb.path[0], int)]))
    pres = ctrl if len(ctrl) else pb.pres
    cdir = np.zeros(m.ndof)
    if len(ctrl):
        ref = pb.up.copy()
        if pb.path is not None:
            ref[np.asarray(pb.path[0], int)] = np.asarray(pb.path[1](0.5 / max(int(c.get("cycles") or 1), 1)), float)
        cdir[ctrl] = np.sign(ref[ctrl])
    for k in range(len(rec.lam)):
        U = rec.U[k].reshape(-1, 3)
        x = m.X + U
        lam = rec.lam[k]
        E = rec.E[k]
        T = rec.react[k]
        r = [float(T[pres[pres % 3 == q]].sum()) for q in range(3)]
        th = rec.th[k]
        f = _frame_fields(m, x, None)
        res.add_frame(f"λ {lam:.4g}", lam, U, f)
        hist["time"].append(lam)
        hist["load_factor"].append(lam)
        hist["energy.stretch"].append(E["stretch"])
        hist["energy.fold"].append(E["fold"])
        hist["energy.bend"].append(E["bend"])
        pl = E["plastic_bar"] + E["plastic_fold"] + E["plastic_bend"]
        hist["energy.plastic"].append(pl)
        hist["energy.contact"].append(E.get("contact", 0.0))
        hist["energy.total"].append(E["stretch"] + E["fold"] + E["bend"] + pl + E.get("contact", 0.0))
        hist["energy.external"].append(rec.extwork[k])
        for q, nm in enumerate("xyz"):
            hist[f"reaction.{nm}"].append(r[q])
        hist["u.control"].append(float((rec.U[k] * cdir)[ctrl].mean()) if len(ctrl) else 0.0)
        hist["reaction.control"].append(float((rec.react[k] * cdir)[ctrl].sum()) if len(ctrl) else 0.0)
        hist["u.max"].append(float(np.linalg.norm(U, axis=1).max()))
        hist["fold.mean"].append(float(np.degrees(np.abs(th[fold] - m.th_init[fold])).mean()) if fold.any() else 0.0)
        hist["contact.pairs"].append(float(rec.contact[k].get("pairs", 0) + rec.contact[k].get("hinge_pairs", 0)) if rec.contact[k] else 0.0)
        hist["iterations"].append(float(rec.iters[k]))
        for e, i in widx:
            hist[f"fold.e{ekey(e)}"].append(float(np.degrees(th[i])))
            hist[f"moment.e{ekey(e)}"].append(float(rec.M[k][i]))
    sf = rec.state
    res.frames[-1]["fields"].update({kk: v for kk, v in _frame_fields(m, rec.x, sf).items() if kk in ("hingePlastic", "barPlastic")})
    res.history = hist
    tot = hist["energy.total"][-1]
    ext = hist["energy.external"][-1]
    s = _summary_model(m)
    s.update({
        "해법": "MGDCM 호장법 (힘 하중)" if kind == "arclength" else "적응 감쇠 뉴턴 (변위·하중·목표각)",
        "증분 수 / 평균 반복": [len(rec.lam) - 1, round(float(np.mean(rec.iters[1:])) if len(rec.iters) > 1 else 0.0, 2)],
        "계산 시간 (s)": round(rec.time, 3),
        "최대 변위 (mm)": hist["u.max"][-1],
        "에너지 (늘림/접힘/휨/소성/접촉, N·mm)": [float(f"{hist[k][-1]:.6g}") for k in ("energy.stretch", "energy.fold", "energy.bend", "energy.plastic", "energy.contact")],
        "외부 일 (N·mm)": ext,
        "에너지 균형 오차": round(abs(tot - ext) / max(abs(ext), 1e-30), 6) if abs(ext) > 0 else None,
    })
    if pb.contact is not None and rec.contact[-1]:
        ci = rec.contact[-1]
        s["접촉 (쌍 / 힌지 / 최소 거리)"] = [ci.get("pairs", 0), ci.get("hinge_pairs", 0), min(ci.get("min_dist", float("inf")), ci.get("hinge_min", float("inf")))]
        s["접촉 에너지 / 전체"] = round(hist["energy.contact"][-1] / max(tot, 1e-30), 6)
        s["관통 검사"] = "없음" if not pb.contact.intersects(rec.x) else "관통 있음"
    if msgs:
        s["참고"] = msgs
    res.summary = s
    res.log = "\n".join(msgs)
    return res


def _modal(m, fixed, n, msgs, want_mass=True):
    use_mass = want_mass and bool(np.all(m.mass > 0))
    r = modal(m, fixed=fixed, n=n + (0 if fixed else 6), use_mass=use_mass)
    res = _base(m, "modal")
    span = float(np.ptp(m.X, axis=0).max())
    hist = {"mode": [], "eigenvalue": []}
    if use_mass:
        hist["frequency"] = []
    for k in range(len(r["eig"])):
        sh = r["shapes"][k] * 0.1 * span
        lab = f"모드 {k + 1} ({r['freq'][k]:.5g} Hz)" if use_mass else f"모드 {k + 1} (λ {r['eig'][k]:.5g})"
        res.add_frame(lab, float(r["freq"][k] if use_mass else r["eig"][k]), sh, _frame_fields(m, m.X + sh, None))
        hist["mode"].append(k + 1)
        hist["eigenvalue"].append(float(r["eig"][k]))
        if use_mass:
            hist["frequency"].append(float(r["freq"][k]))
    res.history = hist
    da = dof_analysis(m, fixed=fixed)
    s = _summary_model(m)
    s.update({
        "모드 해석": "K v = λ M v (집중 질량: 사각형 1/5, 삼각형 1/3)" if use_mass else "K v = λ v (질량 없음)",
        "경계조건": "free-free (강체 모드 6개 포함)" if not fixed else f"구속 자유도 {len(fixed)}개",
        "0 고유치 수": r["rigid"],
        "강체 접힘 기구 수 (패널 강체 가정)": da["mechanisms"],
        "자기 응력 수": da["self_stress"],
    })
    if use_mass:
        s["고유진동수 (Hz)"] = [round(float(f), 6) for f in r["freq"]]
    if msgs:
        s["참고"] = msgs
    res.summary = s
    return res


def _kinematic(m, fixed, c, steps, msgs):
    da = dof_analysis(m, fixed=fixed)
    if da["mechanisms"] == 0:
        raise ValueError("패널을 강체로 둔 강체 접힘 기구가 없습니다 (자유도 0). 주름 지정과 구속을 확인하세요")
    fi = np.where(m.hkind == FOLD)[0]
    drv = [m.fold_index(parse_key(k)) for k in c.get("drivers") or []]
    tg = [np.radians(float(v)) for v in c.get("targets") or []]
    if not drv:
        modes, rates, _ = null_modes(m, fixed=fixed)
        j = int(np.argmax(np.abs(rates[0])))
        drv = [int(fi[j])]
        s0 = np.sign(rates[0][j])
        th = m.th_init[drv[0]]
        tg = [th + s0 * 0.95 * (th if s0 < 0 else 2 * np.pi - th)]
        msgs.append(f"구동 주름을 지정하지 않아 기구 모드에서 회전 비율이 가장 큰 주름 {ekey(m.hedge[drv[0]])} 을 골랐습니다")
    if len(tg) != len(drv):
        raise ValueError("구동 주름 수와 목표각 수가 같아야 합니다")
    if len(drv) < da["mechanisms"]:
        msgs.append(f"기구 자유도 {da['mechanisms']} 보다 구동 주름이 적어 최소 노름 해로 나머지를 정했습니다")
    p = rigid_fold_path(m, drv, tg, steps=steps, fixed=fixed or None)
    res = _base(m, "static")
    hist = {"time": [], "constraint.error": [], "energy.fold": [], "energy.total": []}
    for i in fi:
        hist[f"fold.e{ekey(m.hedge[i])}"] = []
    n = len(p["x"])
    for k in range(n):
        x = p["x"][k]
        res.add_frame(f"단계 {k}/{n - 1}", k / max(n - 1, 1), x - m.X, _frame_fields(m, x, None))
        hist["time"].append(k / max(n - 1, 1))
        hist["constraint.error"].append(float(p["err"][k - 1]) if k else 0.0)
        ef = float(0.5 * np.sum(m.hk[fi] * (p["angles"][k][fi] - m.th0[fi]) ** 2))
        hist["energy.fold"].append(ef)
        hist["energy.total"].append(ef)
        for i in fi:
            hist[f"fold.e{ekey(m.hedge[i])}"].append(float(np.degrees(p["angles"][k][i])))
    res.history = hist
    plot3 = [parse_key(k) for k in c.get("plot3") or []]
    if len(plot3) == 3:
        try:
            sp = solution_space(m, plot3, fixed=fixed or None)
            res.kin3d = {"edges": [ekey(e) for e in plot3], "dof": sp["dof"], "paths": [np.degrees(q).round(5).tolist() for q in sp["paths"]], "main": np.degrees(p["angles"][:, [m.fold_index(e) for e in plot3]]).round(5).tolist()}
        except Exception as e:
            msgs.append(f"3D 해공간 계산 실패: {e}")
    s = _summary_model(m)
    s.update({
        "강체 접힘 기구 수": da["mechanisms"],
        "자기 응력 수": da["self_stress"],
        "구동 주름 → 목표각 (도)": [f"{ekey(m.hedge[i])} → {np.degrees(t):.3f}" for i, t in zip(drv, tg)],
        "최대 구속 오차": float(p["err"].max()) if len(p["err"]) else 0.0,
    })
    if msgs:
        s["참고"] = msgs
    res.summary = s
    return res
