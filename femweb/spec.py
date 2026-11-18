import copy
from .select import resolve

SOLVERS = ("calculix", "barhinge", "tensormesh", "shell")
DAMPING = ("none", "mass", "stiffness", "rayleigh", "direct")
STEP_TYPES = ("static", "buckle", "modal", "arclength")
BC_TYPES = {
    "fixed": [1, 2, 3, 4, 5, 6],
    "pinned": [1, 2, 3],
    "xsymm": [1, 5, 6],
    "ysymm": [2, 4, 6],
    "zsymm": [3, 4, 5],
    "xasymm": [2, 3, 4],
    "yasymm": [1, 3, 5],
    "zasymm": [1, 2, 6],
}
LOAD_TYPES = ("force", "moment", "pressure", "gravity")


def default_spec(model=None):
    solver = "calculix"
    if model is not None and model.kind == "stl":
        solver = "tensormesh"
    return {
        "solver": solver,
        "divisions": 1,
        "materials": {"default": {"E": 210000.0, "nu": 0.3, "rho": 7.85e-9, "yield": None, "hardening": 0.0}},
        "sections": {"*": {"material": "default", "thickness": 1.0, "beam": {"shape": "rect", "b": 10.0, "h": 10.0}}},
        "hinges": {"*": {"kappa": None, "ratio": 0.1}},
        "hinge_auto": {"mode": "off", "angle": 5.0},
        "bcs": [],
        "loads": [],
        "contacts": [],
        "couplings": [],
        "step": {"type": "static", "nlgeom": False, "increments": 25, "modes": 5, "imperfection": 0.0, "imperfection_modes": [], "control": "load", "control_disp": 0.0, "control_load": None, "control_kind": "average", "branch": "auto", "amplitude": "linear", "element": "auto", "device": "auto", "precision": "auto", "threads": 0, "damping": {"kind": "none", "zeta": 0.0, "alpha": 0.0, "beta": 0.0, "time": 0.0}},
        "plots": [],
    }


def imperfections(st):
    lst = [[int(n), float(v)] for n, v in (st.get("imperfection_modes") or []) if float(v) != 0.0]
    if lst:
        return lst
    imp = float(st.get("imperfection") or 0.0)
    return [[1, imp]] if imp > 0 else []


def section_for(spec, group):
    base = copy.deepcopy(spec["sections"]["*"])
    over = spec["sections"].get(group)
    if over:
        for k, v in over.items():
            if k == "beam" and isinstance(v, dict):
                base["beam"].update(v)
            else:
                base[k] = v
    return base


def material_for(spec, group):
    name = section_for(spec, group).get("material", "default")
    return spec["materials"].get(name, spec["materials"]["default"])


def _next_id(items, prefix):
    used = {i["id"] for i in items}
    k = 1
    while f"{prefix}{k}" in used:
        k += 1
    return f"{prefix}{k}"


def _num(v, name):
    try:
        return float(v)
    except Exception:
        raise ValueError(f"{name} 값이 숫자가 아닙니다: {v}")


def _vec(v, name):
    if not isinstance(v, (list, tuple)) or len(v) != 3:
        raise ValueError(f"{name} 은(는) [x, y, z] 형식이어야 합니다")
    return [_num(x, name) for x in v]


def _fix(v):
    if v is None:
        return {}
    if isinstance(v, (list, tuple)):
        v = {int(d): 0.0 for d in v}
    out = {}
    for d, x in dict(v).items():
        d = int(d)
        if d < 1 or d > 6:
            raise ValueError("기준점 자유도는 1~6 (ux, uy, uz, rx, ry, rz) 이어야 합니다")
        out[str(d)] = _num(x, "fix")
    return out


def _coupling_fields(item, op, model, merge=False):
    if "select" in op:
        item["select"] = op["select"]
        if model is not None and len(resolve(model, item["select"])) < 2:
            raise ValueError("기준점에 묶을 절점이 2개 이상 필요합니다")
    if "ref" in op:
        item["ref"] = None if op["ref"] is None else _vec(op["ref"], "ref")
    if item.get("ref") is None and model is not None:
        ids = resolve(model, item["select"])
        item["ref"] = [round(float(c), 9) for c in model.nodes[ids].mean(0)]
    if "fix" in op:
        item["fix"] = {**item.get("fix", {}), **_fix(op["fix"])} if merge else _fix(op["fix"])
    if "free" in op:
        for d in op["free"] or []:
            item.setdefault("fix", {}).pop(str(int(d)), None)
    for f in ("force", "moment"):
        if f in op:
            item[f] = _vec(op[f] if op[f] is not None else [0, 0, 0], f)
    if "name" in op:
        item["name"] = str(op["name"])
    if op.get("rigid"):
        item["rigid"] = True


def apply_ops(spec, ops, model=None):
    spec = copy.deepcopy(spec)
    spec.setdefault("contacts", [])
    spec["step"].setdefault("amplitude", "linear")
    spec["step"].setdefault("element", "auto")
    spec["step"].setdefault("damping", {"kind": "none", "zeta": 0.0, "alpha": 0.0, "beta": 0.0, "time": 0.0})
    log = []
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            raise ValueError(f"잘못된 수정 지시: {op}")
        k = op["op"]
        if k == "add_bc":
            dofs = op.get("dofs")
            btype = None
            if not dofs:
                t = str(op.get("type", "fixed")).lower()
                btype = t
                if t not in BC_TYPES:
                    raise ValueError(f"알 수 없는 구속 종류: {t}")
                dofs = BC_TYPES[t]
            dofs = sorted({int(d) for d in dofs})
            if any(d < 1 or d > 6 for d in dofs):
                raise ValueError("자유도는 1~6 이어야 합니다")
            sel = op.get("select") or {"all": True}
            if model is not None and len(resolve(model, sel)) == 0:
                raise ValueError(f"구속 대상 절점이 0개입니다: {sel}")
            item = {"id": _next_id(spec["bcs"], "bc"), "name": op.get("name") or op.get("type") or "구속", "select": sel, "dofs": dofs, "value": _num(op.get("value", 0.0), "value")}
            if btype:
                item["type"] = btype
            spec["bcs"].append(item)
            log.append(f"구속 추가 {item['id']}")
        elif k == "add_load":
            t = str(op.get("type", "force")).lower()
            if t not in LOAD_TYPES:
                raise ValueError(f"알 수 없는 하중 종류: {t}")
            item = {"id": _next_id(spec["loads"], "ld"), "type": t, "name": op.get("name") or t}
            if t in ("force", "moment"):
                item["vector"] = _vec(op.get("vector"), "vector")
                item["select"] = op.get("select") or {"all": True}
                if t == "moment" and op.get("mode"):
                    if op["mode"] not in ("auto", "nodal", "distributed"):
                        raise ValueError("모멘트 mode 는 auto, nodal, distributed 중 하나여야 합니다")
                    item["mode"] = op["mode"]
            elif t == "pressure":
                item["value"] = _num(op.get("value"), "value")
                item["select"] = op.get("select") or {"all": True}
                if op.get("direction") is not None:
                    item["direction"] = _vec(op["direction"], "direction")
            else:
                item["vector"] = _vec(op.get("vector", [0, 0, -9810.0]), "vector")
            if model is not None and "select" in item and len(resolve(model, item["select"])) == 0:
                raise ValueError(f"하중 대상 절점이 0개입니다: {item['select']}")
            spec["loads"].append(item)
            log.append(f"하중 추가 {item['id']}")
        elif k == "add_contact":
            t = str(op.get("type", "rigid")).lower()
            item = {"id": _next_id(spec["contacts"], "ct"), "type": t, "friction": _num(op.get("friction") or 0.0, "friction")}
            if item["friction"] < 0:
                raise ValueError("마찰계수는 0 이상이어야 합니다")
            if op.get("stiffness") is not None:
                item["stiffness"] = _num(op["stiffness"], "stiffness")
                if item["stiffness"] <= 0:
                    raise ValueError("접촉 강성은 0보다 커야 합니다")
            if t == "rigid":
                shape = str(op.get("shape", "sphere")).lower()
                if shape not in ("sphere", "plane", "mesh"):
                    raise ValueError("강체 형상은 sphere, plane, mesh 중 하나여야 합니다")
                item["shape"] = shape
                item["motion"] = _vec(op.get("motion", [0, 0, 0]), "motion")
                if op.get("target"):
                    item["target"] = op["target"]
                if shape == "mesh":
                    body = op.get("body")
                    if isinstance(body, str):
                        body = {"group": body}
                    if not body:
                        raise ValueError("강체 메쉬 접촉에는 강체로 쓸 부품(body, 예: {\"group\": \"body_2\"})이 필요합니다")
                    item["body"] = body
                    if model is not None:
                        from .contact import rigid_faces

                        if not rigid_faces(model, body):
                            raise ValueError(f"강체 부품 안에 면 요소가 없습니다: {body}")
                elif shape == "sphere":
                    item["radius"] = _num(op.get("radius"), "radius")
                    if item["radius"] <= 0:
                        raise ValueError("구 반지름은 0보다 커야 합니다")
                    item["center"] = _vec(op["center"], "center") if op.get("center") is not None else None
                    if op.get("at") is not None:
                        item["at"] = _vec(op["at"], "at")
                else:
                    item["point"] = _vec(op["point"], "point") if op.get("point") is not None else None
                    item["normal"] = _vec(op["normal"], "normal") if op.get("normal") is not None else None
                if shape != "mesh" and (item.get("center") is None and item.get("point") is None or shape == "plane" and item.get("normal") is None):
                    if model is None:
                        raise ValueError("강체 위치(center 또는 point)를 자동으로 정하려면 모델이 필요합니다")
                    from .contact import place

                    place(model, spec, item)
                item.pop("at", None)
                if model is not None and item.get("target") and len(resolve(model, item["target"])) == 0:
                    raise ValueError("접촉 대상 절점이 0개입니다")
            elif t == "pair":
                if not op.get("a") or not op.get("b"):
                    raise ValueError("접촉 쌍에는 a, b 두 영역이 필요합니다")
                item["a"] = op["a"]
                item["b"] = op["b"]
                if model is not None:
                    from .contact import surface

                    for nm in ("a", "b"):
                        if not surface(model, item[nm]):
                            raise ValueError(f"접촉 영역 {nm} 안에 요소 면이 없습니다: {item[nm]}")
            elif t == "self":
                if any(c["type"] == "self" for c in spec["contacts"]):
                    raise ValueError("자기 접촉은 이미 있습니다 (마찰을 바꾸려면 수정하세요)")
            else:
                raise ValueError("접촉 종류는 rigid, pair, self 중 하나여야 합니다")
            spec["contacts"].append(item)
            if t == "rigid" and not any(s2.get("quantity") == "contact.force" for pl in spec["plots"] for s2 in pl.get("series", [])):
                spec["plots"].append({"id": _next_id(spec["plots"], "pl"), "title": "접촉력-강체 이동량", "x": {"quantity": "contact.disp"}, "series": [{"quantity": "contact.force"}]})
            log.append(f"접촉 추가 {item['id']}")
        elif k == "add_coupling":
            spec.setdefault("couplings", [])
            if not op.get("select"):
                raise ValueError("기준점 커플링에는 묶을 영역(select)이 필요합니다")
            item = {"id": _next_id(spec["couplings"], "cp"), "name": op.get("name") or "기준점", "select": None, "ref": None, "fix": {}, "force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0]}
            _coupling_fields(item, {**op, "select": op["select"]}, model)
            spec["couplings"].append(item)
            log.append(f"기준점 커플링 추가 {item['id']}")
        elif k == "remove":
            i = op.get("id")
            spec.setdefault("couplings", [])
            keys = ("bcs", "loads", "plots", "contacts", "couplings")
            n = sum(len(spec[key]) for key in keys)
            for key in keys:
                spec[key] = [x for x in spec[key] if x["id"] != i]
            if n == sum(len(spec[key]) for key in keys):
                raise ValueError(f"id '{i}' 를 찾을 수 없습니다")
            log.append(f"삭제 {i}")
        elif k == "clear":
            what = op.get("what", "all")
            for key in ("bcs", "loads", "plots", "contacts", "couplings"):
                if what in (key, "all"):
                    spec[key] = []
            log.append(f"초기화 {what}")
        elif k == "update":
            i = op.get("id")
            hit = None
            for key in ("bcs", "loads", "plots", "contacts", "couplings"):
                for x in spec.get(key, []):
                    if x["id"] == i:
                        hit = x
            if hit is None:
                raise ValueError(f"id '{i}' 를 찾을 수 없습니다")
            if str(i).startswith("cp"):
                _coupling_fields(hit, {f: v for f, v in op.items() if f not in ("op", "id")}, model, merge=True)
                log.append(f"수정 {i}")
                continue
            for f, v in op.items():
                if f in ("op", "id"):
                    continue
                if f in ("vector", "direction", "motion", "center", "point", "normal"):
                    v = _vec(v, f)
                elif f in ("friction", "radius", "stiffness"):
                    v = None if v is None else _num(v, f)
                    if v is not None and (v < 0 or f == "radius" and v <= 0):
                        raise ValueError(f"{f} 값 범위 오류")
                elif f == "value":
                    v = _num(v, f)
                elif f == "dofs":
                    v = sorted({int(d) for d in v})
                hit[f] = v
            log.append(f"수정 {i}")
        elif k == "set_material":
            name = op.get("name", "default")
            mat = spec["materials"].setdefault(name, copy.deepcopy(spec["materials"]["default"]))
            for f in ("E", "nu", "rho", "yield", "hardening"):
                if f in op:
                    mat[f] = None if op[f] is None else _num(op[f], f)
            if op.get("power") is not None:
                kk, mm = (_num(v, "power") for v in op["power"])
                y0 = op.get("yield", mat.get("yield"))
                if not y0 or kk <= 0 or not (0 < mm <= 1):
                    raise ValueError("멱법칙 경화에는 항복 응력과 k>0, 0<m<=1 이 필요합니다")
                eps = [0.0, 0.001, 0.002, 0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.2, 0.3, 0.5, 0.75, 1.0]
                op = dict(op, curve=[[e, float(y0) + kk * e**mm] for e in eps])
                mat["power"] = [kk, mm]
            elif "curve" in op:
                mat["power"] = None
            if "curve" in op:
                cv = op["curve"]
                if not cv:
                    mat["curve"] = None
                else:
                    pts = sorted([_num(a, "curve"), _num(b, "curve")] for a, b in cv)
                    if len(pts) < 2 or pts[0][0] < 0 or any(q[1] <= 0 for q in pts) or any(pts[i + 1][0] <= pts[i][0] for i in range(len(pts) - 1)):
                        raise ValueError("소성 곡선은 [소성변형률, 응력] 점 두 개 이상이며 변형률이 증가해야 합니다")
                    if any(pts[i + 1][1] < pts[i][1] for i in range(len(pts) - 1)):
                        raise ValueError("소성 곡선의 응력은 줄어들 수 없습니다 (진응력-소성변형률로 넣으세요)")
                    if pts[0][0] > 0:
                        pts.insert(0, [0.0, pts[0][1]])
                    mat["curve"] = pts
                    mat["yield"] = pts[0][1]
            elif "yield" in op and op["yield"] is None:
                mat["curve"] = None
            if mat["E"] <= 0 or not (-1 < mat["nu"] < 0.5):
                raise ValueError("재료값 범위 오류 (E>0, -1<nu<0.5)")
            log.append(f"재료 {name}")
        elif k == "set_section":
            g = op.get("group", "*")
            sec = spec["sections"].setdefault(g, {})
            if "thickness" in op:
                sec["thickness"] = _num(op["thickness"], "thickness")
                if sec["thickness"] <= 0:
                    raise ValueError("두께는 0보다 커야 합니다")
            if "material" in op:
                sec["material"] = op["material"]
            if isinstance(op.get("beam"), dict):
                b = sec.setdefault("beam", {})
                for f, v in op["beam"].items():
                    b[f] = v if f == "shape" else _num(v, f)
            log.append(f"단면 {g}")
        elif k == "set_hinge":
            g = op.get("group", "*")
            h = spec["hinges"].setdefault(g, {"kappa": None, "ratio": 0.1})
            if "kappa" in op:
                h["kappa"] = None if op["kappa"] is None else _num(op["kappa"], "kappa")
            if "ratio" in op:
                h["ratio"] = _num(op["ratio"], "ratio")
            if "kappa" in op and op["kappa"] is not None:
                if h["kappa"] < 0:
                    raise ValueError("힌지 강성은 0 이상이어야 합니다")
                h["pin"] = h["kappa"] == 0.0
            if op.get("method") is not None:
                if op["method"] not in ("spring", "strip"):
                    raise ValueError("hinge method 는 spring 또는 strip 이어야 합니다")
                spec["hinge_method"] = op["method"]
            if "pin" in op:
                h["pin"] = bool(op["pin"])
                if h["pin"]:
                    h["kappa"] = None
            log.append(f"힌지 {g}")
        elif k == "auto_hinge":
            mode = str(op.get("mode", "auto")).lower()
            if mode not in ("off", "auto", "groups", "crease", "all"):
                raise ValueError("auto_hinge mode 는 off, auto, groups, crease, all 중 하나여야 합니다")
            cur = dict(spec.get("hinge_auto") or {"mode": "off", "angle": 5.0})
            cur["mode"] = mode
            if op.get("angle") is not None:
                cur["angle"] = max(0.0, _num(op["angle"], "angle"))
            spec["hinge_auto"] = cur
            log.append(f"자동 힌지 {mode}")
        elif k == "set_step":
            st = spec["step"]
            if "type" in op:
                if op["type"] not in STEP_TYPES:
                    raise ValueError(f"스텝 종류는 {STEP_TYPES} 중 하나여야 합니다")
                st["type"] = op["type"]
            if "nlgeom" in op:
                st["nlgeom"] = bool(op["nlgeom"])
            for f in ("increments", "modes"):
                if f in op:
                    st[f] = max(1, int(op[f]))
            if "imperfection" in op:
                st["imperfection"] = _num(op["imperfection"], "imperfection")
                if "imperfection_modes" not in op:
                    st["imperfection_modes"] = []
            if "imperfection_modes" in op:
                lst = []
                for it in op["imperfection_modes"] or []:
                    n, v = int(it[0]), _num(it[1], "imperfection_modes")
                    if n < 1 or n > 50:
                        raise ValueError("결함 모드 번호는 1~50 이어야 합니다")
                    lst.append([n, v])
                st["imperfection_modes"] = lst
            if "control" in op:
                c = str(op["control"] or "load").lower()
                if c not in ("load", "displacement"):
                    raise ValueError("control 은 load 또는 displacement 여야 합니다")
                st["control"] = c
            if "control_disp" in op:
                st["control_disp"] = abs(_num(op["control_disp"] or 0.0, "control_disp"))
                if "control" not in op and st["control_disp"] > 0:
                    st["control"] = "displacement"
            if "control_kind" in op:
                if op["control_kind"] not in ("average", "rigid"):
                    raise ValueError("control_kind 는 average 또는 rigid 여야 합니다")
                st["control_kind"] = op["control_kind"]
            if "control_load" in op:
                st["control_load"] = op["control_load"] or None
            if "element" in op:
                if op["element"] not in ("auto", "tri", "quad"):
                    raise ValueError("element 는 auto, tri, quad 중 하나여야 합니다")
                st["element"] = op["element"]
            if "device" in op:
                if op["device"] not in ("auto", "cpu", "cuda", "mps"):
                    raise ValueError("device 는 auto, cpu, cuda, mps 중 하나여야 합니다")
                st["device"] = op["device"]
            if "precision" in op:
                if op["precision"] not in ("auto", "float64", "float32"):
                    raise ValueError("precision 은 auto, float64, float32 중 하나여야 합니다")
                st["precision"] = op["precision"]
            if "threads" in op:
                st["threads"] = max(0, int(op["threads"] or 0))
            if "amplitude" in op:
                if op["amplitude"] not in ("linear", "smooth"):
                    raise ValueError("amplitude 는 linear 또는 smooth 여야 합니다")
                st["amplitude"] = op["amplitude"]
            if "damping" in op:
                d = dict(st.get("damping") or {"kind": "none", "zeta": 0.0, "alpha": 0.0, "beta": 0.0, "time": 0.0})
                src = op["damping"] if isinstance(op["damping"], dict) else {"kind": op["damping"]}
                if src.get("kind") is not None:
                    if src["kind"] not in DAMPING:
                        raise ValueError(f"감쇠 종류는 {DAMPING} 중 하나여야 합니다")
                    d["kind"] = src["kind"]
                for f in ("zeta", "alpha", "beta", "time"):
                    if src.get(f) is not None:
                        d[f] = max(0.0, _num(src[f], f))
                if d["kind"] in ("mass", "stiffness", "rayleigh") and d["zeta"] <= 0:
                    d["zeta"] = 5.0
                st["damping"] = d
            if "branch" in op:
                if op["branch"] not in ("auto", "primary"):
                    raise ValueError("branch 는 auto 또는 primary 여야 합니다")
                st["branch"] = op["branch"]
            if st.get("control") == "displacement" and not any(s2.get("quantity") == "control.force" for pl in spec["plots"] for s2 in pl.get("series", [])):
                spec["plots"].append({"id": _next_id(spec["plots"], "pl"), "title": "하중-변위 곡선 (변위 제어)", "x": {"quantity": "control.disp"}, "series": [{"quantity": "control.force"}]})
            if st.get("control") != "displacement":
                spec["plots"] = [pl for pl in spec["plots"] if not any(s2.get("quantity") == "control.force" for s2 in pl.get("series", []))]
            log.append("스텝 설정")
        elif k == "set_solver":
            if op.get("solver") not in SOLVERS:
                raise ValueError(f"솔버는 {SOLVERS} 중 하나여야 합니다")
            spec["solver"] = op["solver"]
            log.append(f"솔버 {op['solver']}")
        elif k == "set_divisions":
            spec["divisions"] = max(1, min(64, int(op.get("divisions", 1))))
            log.append(f"분할수 {spec['divisions']}")
        elif k == "add_plot":
            item = {"id": _next_id(spec["plots"], "pl"), "title": op.get("title") or "그래프", "x": op.get("x") or {"quantity": "time"}, "series": op.get("series") or []}
            if not item["series"]:
                raise ValueError("그래프에 series 가 없습니다")
            spec["plots"].append(item)
            log.append(f"그래프 추가 {item['id']}")
        else:
            raise ValueError(f"알 수 없는 op: {k}")
    return spec, log


def solid_rotations(model, spec):
    from .loads import solid_only

    if model is None or not solid_only(model):
        return spec, []
    rot = [b for b in spec["bcs"] if any(d > 3 for d in b["dofs"]) and b.get("type") not in BC_TYPES]
    if not rot:
        return spec, []
    out = copy.deepcopy(spec)
    out.setdefault("couplings", [])
    notes = []
    bcs = []
    for b in out["bcs"]:
        rd = [d for d in b["dofs"] if d > 3]
        if not rd or b.get("type") in BC_TYPES:
            bcs.append(b)
            continue
        ids = resolve(model, b["select"])
        cp = {"id": f"cp_{b['id']}", "name": f"{b['id']} 회전 구속", "select": b["select"], "ref": [float(c) for c in model.nodes[ids].mean(0)], "fix": {str(d): float(b.get("value", 0.0)) for d in rd}, "force": [0.0, 0.0, 0.0], "moment": [0.0, 0.0, 0.0]}
        out["couplings"].append(cp)
        td = [d for d in b["dofs"] if d <= 3]
        if td:
            bcs.append({**b, "dofs": td})
        v = float(b.get("value", 0.0))
        what = "구속" if v == 0.0 else f"{v:g} rad 로 지정"
        names = ", ".join("r" + "xyz"[d - 4] for d in rd)
        notes.append(f"{b['id']}: 솔리드에는 회전 자유도가 없어, 면 절점을 기준점 {cp['id']} 에 묶고 기준점 회전({names})을 {what}했습니다")
    out["bcs"] = bcs
    return out, notes


def warnings_for(spec, model):
    out = []
    if model is None:
        return out
    if model.kind == "stl" and spec["solver"] == "barhinge":
        out.append("솔리드(STL) 모델은 바-힌지 솔버로 풀 수 없습니다")
    if model.kind == "obj" and spec["solver"] == "tensormesh":
        out.append("보·쉘(OBJ) 모델은 TensorMesh(솔리드) 솔버로 풀 수 없습니다")
    if spec["solver"] == "barhinge" and not model.shells:
        out.append("바-힌지 솔버에는 면(f) 패널이 필요합니다")
    if spec["solver"] == "barhinge" and any(d > 3 for b in spec["bcs"] for d in b["dofs"]):
        out.append("바-힌지 모델에는 회전 자유도가 없어 회전 구속은 적용되지 않습니다 (고정은 병진 구속으로만 반영)")
    cps = spec.get("couplings") or []
    if cps and spec["solver"] not in ("shell", "calculix"):
        out.append("기준점 커플링은 자체 쉘·CalculiX 솔버에서만 적용됩니다")
    for cp in cps:
        try:
            if len(resolve(model, cp["select"])) < 2:
                out.append(f"{cp['id']}: 묶을 절점이 2개 미만입니다")
        except Exception:
            out.append(f"{cp['id']}: 묶을 영역을 찾을 수 없습니다")
    if spec["step"]["type"] == "modal" and not (spec["materials"]["default"].get("rho") or 0) > 0:
        out.append("모달 해석에는 밀도가 필요합니다")
    if not spec["bcs"] and not any(cp.get("fix") for cp in cps):
        out.append("구속이 없습니다 (강체 운동 가능)")
    else:
        t = set()
        for b in spec["bcs"]:
            t.update(d for d in b["dofs"] if d <= 3)
        if t != {1, 2, 3}:
            out.append("병진 자유도 중 구속되지 않은 방향이 있습니다: " + ", ".join("xyz"[d - 1] for d in (1, 2, 3) if d not in t))
    if spec["solver"] == "shell":
        if model.kind != "obj" or not model.shells:
            out.append("자체 쉘 솔버에는 OBJ 쉘 패널 모델이 필요합니다")
        if spec["step"]["type"] in ("buckle", "modal"):
            out.append("자체 쉘 솔버는 정적 스텝만 지원합니다")
    dk = (spec["step"].get("damping") or {}).get("kind", "none")
    if dk != "none" and spec["solver"] != "shell":
        out.append("감쇠 설정은 자체 쉘 솔버에서만 적용됩니다")
    if any(c["type"] == "self" for c in spec.get("contacts") or []) and spec["solver"] != "shell":
        out.append("자기 접촉은 자체 쉘 솔버에서만 지원합니다")
    seen = {}
    for b in spec["bcs"]:
        try:
            ids = resolve(model, b["select"])
        except Exception:
            continue
        for i in ids:
            for d in b["dofs"]:
                key = (int(i), int(d))
                v = float(b.get("value", 0.0))
                if key in seen and seen[key][1] != v:
                    out.append(f"구속 {seen[key][0]} 와 {b['id']} 가 같은 절점·자유도에 서로 다른 값을 줍니다 (나중 것이 적용됨)")
                    break
                seen[key] = (b["id"], v)
            else:
                continue
            break
    if spec.get("contacts"):
        if spec["solver"] == "barhinge":
            out.append("바-힌지 솔버는 접촉을 지원하지 않습니다 (CalculiX 또는 TensorMesh 를 선택하세요)")
        if spec["step"]["type"] in ("buckle", "modal"):
            out.append("접촉은 정적 스텝에서만 적용됩니다")
    st = spec["step"]
    if st.get("control") == "displacement":
        if st["type"] in ("buckle", "modal"):
            out.append("변위 제어는 정적·호장법 스텝에서만 쓰입니다 (좌굴·모달에서는 무시)")
        elif not any(ld["type"] in ("force", "pressure") for ld in spec["loads"]):
            out.append("변위 제어에는 기준이 되는 힘 또는 압력 하중이 필요합니다")
    moving = any(abs(b.get("value", 0.0)) > 0 for b in spec["bcs"])
    moving = moving or any(c["type"] == "rigid" and any(float(v) != 0 for v in c.get("motion", [])) for c in spec.get("contacts") or [])
    moving = moving or any(any(float(v) != 0 for v in cp.get("fix", {}).values()) or any(cp.get("force", [0, 0, 0])) or any(cp.get("moment", [0, 0, 0])) for cp in spec.get("couplings") or [])
    if not spec["loads"] and not moving and spec["step"]["type"] != "modal":
        out.append("하중이나 지정 변위가 없습니다")
    return out
