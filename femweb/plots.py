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
    "reaction.x": "반력 x (N)",
    "reaction.y": "반력 y (N)",
    "reaction.z": "반력 z (N)",
    "u.max": "최대 변위 (mm)",
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
        return np.array(result.history[name], dtype=float), LABELS.get(name, name)
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
    unit = "MPa" if field.startswith("S") or field == "vonMises" else "mm" if field.startswith("U") else ""
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
