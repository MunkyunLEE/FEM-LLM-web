import numpy as np
from .select import describe, resolve

LABELS = {
    "time": "하중계수",
    "load_factor": "하중계수",
    "frame": "프레임",
    "mode": "모드",
    "energy.total": "총 에너지 (N·mm)",
    "energy.bar": "바 에너지 (N·mm)",
    "energy.bend": "패널 굽힘 에너지 (N·mm)",
    "energy.fold": "접힘 에너지 (N·mm)",
    "energy.fold_bar": "접힘 + 바 에너지 (N·mm)",
    "energy.elastic": "접힘 + 바 + 휨 에너지 (N·mm)",
    "energy.stretch": "바 에너지 (N·mm)",
    "step": "스텝",
    "u.control": "제어 변위 (mm)",
    "reaction.control": "제어점 반력 (N)",
    "energy.strain": "변형 에너지 (N·mm)",
    "energy.plastic": "소성 소산 에너지 (N·mm)",
    "peeq.max": "최대 등가 소성 변형률",
    "energy.damping": "감쇠 소산 에너지 (N·mm)",
    "energy.kinetic": "운동 에너지 (N·mm)",
    "energy.friction": "마찰 소산 에너지 (N·mm)",
    "energy.contact": "접촉 장벽 에너지 (N·mm)",
    "energy.external": "외부 일 (N·mm)",
    "energy.coupling": "기준점 결합 에너지 (N·mm)",
    "contact.count": "접촉 쌍 수",
    "contact.gap": "최소 접촉 간격 (mm)",
    "reaction.x": "반력 x (N)",
    "reaction.y": "반력 y (N)",
    "reaction.z": "반력 z (N)",
    "u.max": "최대 변위 (mm)",
    "contact.disp": "강체 이동량 (mm)",
    "contact.force": "접촉력 (N)",
    "contact.normal": "접촉 수직력 합 (N)",
    "contact.friction": "접촉 마찰력 합 (N)",
    "control.disp": "제어 변위 (mm)",
    "control.force": "제어점 하중 (N)",
    "stability.negative": "음의 고유치 수",
    "vonMises.max": "최대 von Mises (MPa)",
    "fold.mean": "평균 접힘각 (도)",
    "strain.max": "최대 바 변형률",
    "buckling.factor": "좌굴 계수",
    "frequency": "고유진동수 (Hz)",
}
NODAL = {"u": "U", "ux": "Ux", "uy": "Uy", "uz": "Uz", "vonmises": "vonMises", "stress": "vonMises"}


def _nodes(result, model, sel):
    if not sel:
        return None
    ids = resolve(model, sel)
    if len(ids) == 0:
        raise ValueError(f"그래프 대상 절점이 0개입니다: {sel}")
    from scipy.spatial import cKDTree

    return np.unique(cKDTree(result.nodes).query(model.nodes[ids])[1])


def _series(result, model, q):
    name = str(q.get("quantity", "time"))
    if name == "frame":
        return np.arange(1, len(result.frames) + 1, dtype=float), "프레임"
    if name in result.history:
        lab = LABELS.get(name)
        if lab is None and name.startswith("rp.") and name.count(".") == 2:
            _, cid, q = name.split(".")
            lab = f"기준점 {cid} " + {"ux": "변위 x (mm)", "uy": "변위 y (mm)", "uz": "변위 z (mm)", "rx": "회전 x (rad)", "ry": "회전 y (rad)", "rz": "회전 z (rad)", "fx": "힘 x (N)", "fy": "힘 y (N)", "fz": "힘 z (N)", "mx": "모멘트 x (N·mm)", "my": "모멘트 y (N·mm)", "mz": "모멘트 z (N·mm)"}.get(q, q)
        if lab is None and name.startswith("reaction.") and name.count(".") == 2:
            _, bid, cc = name.split(".")
            lab = f"반력 {bid} {cc} (N)"
        return np.array(result.history[name], dtype=float), lab or name
    key = name.lower()
    comp = str(q.get("comp", "")).lower()
    if key == "u" and comp in ("x", "y", "z"):
        key = "u" + comp
    field = NODAL.get(key, name)
    if field not in result.field_names():
        raise ValueError(f"'{name}' 은(는) 이 결과에 없습니다. 사용 가능: {', '.join(sorted(result.history)) + ', ' + ', '.join(result.field_names())}")
    ids = _nodes(result, model, q.get("select"))
    red = q.get("reduce") or ("mean" if field in ("Ux", "Uy", "Uz") else "max")
    out = []
    for i in range(len(result.frames)):
        v = result.field(i, field)
        if ids is not None:
            v = v[ids]
        if red == "mean":
            out.append(float(v.mean()))
        elif red == "min":
            out.append(float(v.min()))
        elif red == "absmax":
            out.append(float(np.abs(v).max()))
        else:
            out.append(float(v.max()))
    unit = "MPa" if field.startswith("S") or field in ("vonMises", "CPRESS") else "mm" if field.startswith("U") else "N" if field in ("RF", "CFORCE") else ""
    where = describe(q["select"]) if q.get("select") else "전체"
    return np.array(out), f"{field} {red} @ {where}" + (f" ({unit})" if unit else "")


def evaluate(plot, result, model):
    x, xl = _series(result, model, plot.get("x") or {"quantity": "time"})
    series = []
    for s in plot.get("series", []):
        y, yl = _series(result, model, s)
        n = min(len(x), len(y))
        series.append({"name": s.get("name") or yl, "x": x[:n].tolist(), "y": y[:n].tolist()})
    return {"id": plot.get("id"), "title": plot.get("title", "그래프"), "xlabel": xl, "series": series}


def bh_defaults(result):
    h = result.history
    out = []
    if "energy.elastic" in h:
        ser = [{"quantity": "energy.fold", "name": "접힘"}]
        if "energy.fold_bar" in h:
            ser.append({"quantity": "energy.fold_bar", "name": "접힘 + 바"})
        ser.append({"quantity": "energy.elastic", "name": "접힘 + 바 + 휨 (전체)"})
        if "energy.plastic" in h:
            ser.append({"quantity": "energy.plastic", "name": "소성"})
        out.append({"id": "auto_energy", "title": "스텝-에너지", "x": {"quantity": "step"}, "series": ser, "auto": True})
    if "u.control" in h and any(abs(v) > 0 for v in h["u.control"]):
        out.append({"id": "auto_ld", "title": "하중-변위 (제어 자유도)", "x": {"quantity": "u.control"}, "series": [{"quantity": "reaction.control", "name": "반력"}], "auto": True})
    elif "load_factor" in h and "u.max" in h:
        out.append({"id": "auto_ld", "title": "하중계수-최대 변위", "x": {"quantity": "u.max"}, "series": [{"quantity": "load_factor", "name": "하중계수"}], "auto": True})
    return out
