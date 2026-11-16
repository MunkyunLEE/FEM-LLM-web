import copy
import numpy as np
from .loads import bc_table, lumped

CTL_ID = "ctl"


def is_disp(spec):
    st = spec["step"]
    return st.get("control") == "displacement" and st["type"] in ("static", "arclength")


def pick_load(spec):
    want = spec["step"].get("control_load")
    cands = [ld for ld in spec["loads"] if ld["type"] in ("force", "pressure")]
    if want:
        for ld in cands:
            if ld["id"] == want:
                return ld
        raise ValueError(f"변위 제어 대상 하중 '{want}' 을(를) 찾을 수 없습니다 (힘 또는 압력 하중이어야 합니다)")
    if not cands:
        raise ValueError("변위 제어에는 기준이 되는 힘 또는 압력 하중이 하나 필요합니다. 하중을 먼저 추가하세요")
    return cands[0]


def disp_control(model, spec):
    if not is_disp(spec):
        return spec, None
    ld = pick_load(spec)
    one = dict(spec)
    one["loads"] = [ld]
    F = lumped(model, one)[0]
    total = F.sum(axis=0)
    amp = float(spec["step"].get("control_disp") or 0.0)
    auto = amp <= 0
    if auto:
        amp = 0.05 * model.diag()
    tail = " (목표 변위 미지정이라 모델 크기의 5%)" if auto else ""
    out = copy.deepcopy(spec)
    out["loads"] = [x for x in out["loads"] if x["id"] != ld["id"]]
    rest = "나머지 하중과 지정 변위는 진행률에 비례해 함께 증가합니다" if out["loads"] or any(abs(b.get("value", 0.0)) > 0 for b in out["bcs"]) else None
    if spec["step"].get("control_kind", "average") != "rigid":
        N = float(np.linalg.norm(F, axis=1).sum())
        if N <= 1e-12:
            raise ValueError(f"하중 {ld['id']} 의 크기가 0 이라 변위 제어를 할 수 없습니다")
        notes = [f"변위 제어(하중 분포 유지): 하중 {ld['id']} 의 분포는 그대로 두고, 하중 방향 평균 변위가 {amp:.4g} mm 가 될 때까지 하중 크기를 자동으로 조절했습니다{tail}"]
        if rest:
            notes.append(rest)
        return out, {"kind": "average", "load": ld["id"], "one": one, "amp": amp, "N": N, "notes": notes}
    if np.linalg.norm(total) <= 1e-12:
        raise ValueError(f"하중 {ld['id']} 의 합력이 0 이라 강체 압판 방향을 정할 수 없습니다")
    axis = int(np.argmax(np.abs(total)))
    sign = 1.0 if total[axis] > 0 else -1.0
    hit = np.where(np.abs(F[:, axis]) > 1e-12 * np.abs(F).max())[0]
    fixed = bc_table(model, spec) if spec["bcs"] else {}
    nodes = [int(i) for i in hit if (int(i), axis + 1) not in fixed]
    if not nodes:
        raise ValueError(f"하중 {ld['id']} 이(가) 걸린 절점이 모두 같은 방향으로 구속되어 있어 변위 제어를 할 수 없습니다")
    out["bcs"] = [{"id": CTL_ID, "name": "변위 제어", "select": {"nodes": nodes}, "dofs": [axis + 1], "value": sign * amp}] + out["bcs"]
    notes = [f"변위 제어(강체 압판): 하중 {ld['id']} 이(가) 걸린 절점 {len(nodes)}개를 {'-' if sign < 0 else '+'}{'xyz'[axis]} 방향으로 똑같이 {amp:.4g} mm 까지 밀었습니다{tail}. 하중면이 기울지 못하므로 끝단 회전이 막힌 조건이 됩니다"]
    d = total / np.linalg.norm(total)
    if abs(d[axis]) < 0.999:
        notes.append(f"하중 방향이 축과 나란하지 않아 가장 큰 성분인 {'xyz'[axis]} 방향만 제어했습니다")
    if rest:
        notes.append(rest)
    return out, {"kind": "rigid", "load": ld["id"], "nodes": nodes, "axis": axis, "sign": sign, "amp": amp, "notes": notes}


def record(res, info, times, forces, disps=None):
    d = [float(t) * info["amp"] for t in times] if disps is None else [float(v) for v in disps]
    f = [float(v) for v in forces]
    res.history["control.disp"] = d
    res.history["control.force"] = f
    res.summary["제어 변위 끝값 (mm)"] = round(d[-1], 6) if d else 0.0
    res.summary["제어점 하중 끝값 (N)"] = round(f[-1], 4) if f else 0.0
    if not f:
        return
    k = next((i for i in range(1, len(f) - 1) if f[i] > f[i - 1] and f[i] > f[i + 1] and f[i] > 0), None)
    if k is None and len(f) > 1 and f[0] > f[1] and f[0] > 0:
        k = 0
    if k is not None:
        res.summary["첫 극한 하중 (N) @ 변위 (mm)"] = [round(f[k], 4), round(d[k], 6)]
        res.summary["극한점"] = f"하중이 변위 {d[k]:.4g} mm 에서 {f[k]:.5g} N 을 찍고 내려갑니다 (좌굴 또는 스냅스루). 증분 사이 값이라 실제 극한 하중은 이보다 조금 클 수 있습니다"
    else:
        m = int(np.argmax(f))
        res.summary["제어점 하중 최댓값 (N) @ 변위 (mm)"] = [round(f[m], 4), round(d[m], 6)]
