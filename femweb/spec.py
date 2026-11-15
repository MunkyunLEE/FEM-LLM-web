import copy
from .select import resolve

SOLVERS = ("calculix", "barhinge", "tensormesh")
STEP_TYPES = ("static", "buckle", "modal", "arclength")
BC_TYPES = {
    "fixed": [1, 2, 3, 4, 5, 6],
    "pinned": [1, 2, 3],
    "xsymm": [1, 5, 6],
    "ysymm": [2, 4, 6],
    "zsymm": [3, 4, 5],
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
        "step": {"type": "static", "nlgeom": False, "increments": 25, "modes": 5, "imperfection": 0.0},
        "plots": [],
    }


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


def apply_ops(spec, ops, model=None):
    spec = copy.deepcopy(spec)
    log = []
    for op in ops:
        if not isinstance(op, dict) or "op" not in op:
            raise ValueError(f"잘못된 수정 지시: {op}")
        k = op["op"]
        if k == "add_bc":
            dofs = op.get("dofs")
            if not dofs:
                t = str(op.get("type", "fixed")).lower()
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
        elif k == "remove":
            i = op.get("id")
            n = len(spec["bcs"]) + len(spec["loads"]) + len(spec["plots"])
            for key in ("bcs", "loads", "plots"):
                spec[key] = [x for x in spec[key] if x["id"] != i]
            if n == len(spec["bcs"]) + len(spec["loads"]) + len(spec["plots"]):
                raise ValueError(f"id '{i}' 를 찾을 수 없습니다")
            log.append(f"삭제 {i}")
        elif k == "clear":
            what = op.get("what", "all")
            for key in ("bcs", "loads", "plots"):
                if what in (key, "all"):
                    spec[key] = []
            log.append(f"초기화 {what}")
        elif k == "update":
            i = op.get("id")
            hit = None
            for key in ("bcs", "loads", "plots"):
                for x in spec[key]:
                    if x["id"] == i:
                        hit = x
            if hit is None:
                raise ValueError(f"id '{i}' 를 찾을 수 없습니다")
            for f, v in op.items():
                if f in ("op", "id"):
                    continue
                if f in ("vector", "direction"):
                    v = _vec(v, f)
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
    if spec["step"]["type"] == "modal" and not (spec["materials"]["default"].get("rho") or 0) > 0:
        out.append("모달 해석에는 밀도가 필요합니다")
    if not spec["bcs"]:
        out.append("구속이 없습니다 (강체 운동 가능)")
    else:
        t = set()
        for b in spec["bcs"]:
            t.update(d for d in b["dofs"] if d <= 3)
        if t != {1, 2, 3}:
            out.append("병진 자유도 중 구속되지 않은 방향이 있습니다: " + ", ".join("xyz"[d - 1] for d in (1, 2, 3) if d not in t))
    moving = any(abs(b.get("value", 0.0)) > 0 for b in spec["bcs"])
    if not spec["loads"] and not moving and spec["step"]["type"] != "modal":
        out.append("하중이나 지정 변위가 없습니다")
    return out
