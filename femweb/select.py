import numpy as np

AX = {"x": 0, "y": 1, "z": 2}


def resolve(model, sel):
    n = len(model.nodes)
    if not sel or sel.get("all"):
        return np.arange(n)
    if "and" in sel:
        out = None
        for s in sel["and"]:
            r = set(resolve(model, s).tolist())
            out = r if out is None else out & r
        return np.array(sorted(out or []), dtype=int)
    if "or" in sel:
        out = set()
        for s in sel["or"]:
            out |= set(resolve(model, s).tolist())
        return np.array(sorted(out), dtype=int)
    parts = []
    if "group" in sel:
        ids = model.group_nodes(sel["group"])
        if not ids and sel["group"] not in model.group_names():
            raise ValueError(f"그룹 '{sel['group']}' 이(가) 없습니다. 사용 가능: {', '.join(model.group_names()) or '없음'}")
        parts.append(set(ids))
    if "nodes" in sel:
        parts.append(set(int(i) for i in sel["nodes"] if 0 <= int(i) < n))
    if "plane" in sel:
        p = sel["plane"]
        a = AX[str(p.get("axis", "z")).lower()]
        c = model.nodes[:, a]
        at = p.get("at", "min")
        v = c.min() if at == "min" else c.max() if at == "max" else c.mean() if at in ("mid", "center") else float(at)
        tol = float(p.get("tol", model.diag() * 1e-4))
        parts.append(set(np.where(np.abs(c - v) <= tol)[0].tolist()))
    if "box" in sel:
        lo = np.array(sel["box"]["min"], dtype=float)
        hi = np.array(sel["box"]["max"], dtype=float)
        tol = model.diag() * 1e-6
        ok = np.all((model.nodes >= lo - tol) & (model.nodes <= hi + tol), axis=1)
        parts.append(set(np.where(ok)[0].tolist()))
    if "nearest" in sel:
        d = np.linalg.norm(model.nodes - np.array(sel["nearest"], dtype=float), axis=1)
        parts.append({int(np.argmin(d))})
    if not parts:
        raise ValueError(f"선택 조건을 해석할 수 없습니다: {sel}")
    out = parts[0]
    for s in parts[1:]:
        out &= s
    return np.array(sorted(out), dtype=int)


def describe(sel):
    if not sel or sel.get("all"):
        return "전체"
    if "and" in sel:
        return " ∩ ".join(describe(s) for s in sel["and"])
    if "or" in sel:
        return " ∪ ".join(describe(s) for s in sel["or"])
    out = []
    if "group" in sel:
        out.append(f"그룹 {sel['group']}")
    if "plane" in sel:
        p = sel["plane"]
        out.append(f"{p.get('axis', 'z')}={p.get('at', 'min')}")
    if "box" in sel:
        out.append("박스 영역")
    if "nearest" in sel:
        out.append("절점 (" + ", ".join(f"{float(v):g}" for v in sel["nearest"]) + ")")
    if "nodes" in sel:
        out.append(f"절점 {len(sel['nodes'])}개")
    return " ∩ ".join(out)
