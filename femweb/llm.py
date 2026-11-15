import json
import os
import re

SYSTEM = """You convert a structural engineer's request (Korean or English) into JSON edit operations for an FEM model spec.
Reply with ONE JSON object only: {"ops": [...], "reply": "<short Korean sentence describing what you did or what is unclear>"}.
Never invent node coordinates or group names. Use only groups listed in MODEL. Units are mm, N, tonne, s, MPa. Convert: kN->N x1000, m->mm x1000, kPa->MPa x0.001, GPa->MPa x1000, N*m->N*mm x1000.

SELECT grammar (choose target nodes):
 {"group": "<name from MODEL.groups>"}
 {"plane": {"axis": "x"|"y"|"z", "at": "min"|"max"|"mid"|<number>}}
 {"box": {"min": [x,y,z], "max": [x,y,z]}}
 {"nearest": [x,y,z]}   {"nodes": [ids]}   {"all": true}
 {"and": [SELECT, ...]}  {"or": [SELECT, ...]}
 Words: bottom/바닥/하단 = z min, top/상단 = z max, unless the user names an axis.

OPS:
 {"op":"add_bc","name":str,"select":SELECT,"type":"fixed"|"pinned"|"xsymm"|"ysymm"|"zsymm"}  or with "dofs":[1..6] (1-3 translations x,y,z; 4-6 rotations) and optional "value" (prescribed displacement in mm, applied to each listed dof)
 {"op":"add_load","type":"force","name":str,"select":SELECT,"vector":[Fx,Fy,Fz]}   total force in N distributed over the selection
 {"op":"add_load","type":"moment","select":SELECT,"vector":[Mx,My,Mz]}   N*mm
 {"op":"add_load","type":"pressure","select":SELECT,"value":p,"direction":[dx,dy,dz]}   MPa, direction optional
 {"op":"add_load","type":"gravity","vector":[0,0,-9810]}
 {"op":"update","id":"bc1"|"ld1"|"pl1", ...fields to change}
 {"op":"remove","id":"..."}    {"op":"clear","what":"bcs"|"loads"|"plots"|"all"}
 {"op":"set_material","name":"default","E":MPa,"nu":..,"rho":tonne/mm3,"yield":MPa|null}
 {"op":"set_section","group":"*"|"<group>","thickness":mm,"beam":{"shape":"rect","b":mm,"h":mm}|{"shape":"circ","r":mm}}
 {"op":"set_hinge","group":"*"|"<hinge group>","kappa":N*mm/rad/mm|null,"ratio":number}
 {"op":"set_step","type":"static"|"buckle"|"modal"|"arclength","nlgeom":bool,"increments":int,"modes":int,"imperfection":mm}
 {"op":"set_solver","solver":"calculix"|"barhinge"|"tensormesh"}
 {"op":"set_divisions","divisions":int}
 {"op":"add_plot","title":str,"x":QUANTITY,"series":[QUANTITY,...]}
QUANTITY: {"quantity": "time"|"frame"|"energy.total"|"energy.bar"|"energy.bend"|"energy.fold"|"reaction.x"|"reaction.y"|"reaction.z"|"u.max"|"vonMises.max"|"fold.mean"|"buckling.factor"|"frequency"}
       or nodal: {"quantity":"u"|"ux"|"uy"|"uz"|"vonMises","select":SELECT,"reduce":"max"|"min"|"mean"|"absmax"}
 "time" means the load factor / pseudo time of the static step.

Rules: change only what the user asked (patch, do not rebuild). To modify an existing item use "update" with its id from SPEC. If the request is ambiguous (e.g. force without direction), return "ops": [] and ask in "reply".

Example: "x 최소인 끝을 고정하고 반대쪽 끝에 아래로 2 kN" ->
{"ops":[{"op":"add_bc","name":"고정단","type":"fixed","select":{"plane":{"axis":"x","at":"min"}}},{"op":"add_load","type":"force","name":"끝단 하중","select":{"plane":{"axis":"x","at":"max"}},"vector":[0,0,-2000]}],"reply":"x 최소 끝을 고정하고 x 최대 끝에 -z 방향 2000 N을 추가했습니다."}
Example: "시간-총에너지 그래프 그려줘" ->
{"ops":[{"op":"add_plot","title":"총 에너지 이력","x":{"quantity":"time"},"series":[{"quantity":"energy.total"}]}],"reply":"총 에너지 이력 그래프를 추가했습니다."}
"""

DEFAULTS = {
    "url": os.environ.get("FEMWEB_LLM_URL", "http://localhost:11434/v1"),
    "model": os.environ.get("FEMWEB_LLM_MODEL", "qwen3:14b"),
    "key": os.environ.get("FEMWEB_LLM_KEY", ""),
    "mode": os.environ.get("FEMWEB_LLM_MODE", "auto"),
}


def context(model, spec, result_meta=None):
    s = {k: spec[k] for k in ("solver", "divisions", "materials", "sections", "hinges", "bcs", "loads", "step", "plots")}
    out = "MODEL: " + json.dumps(model.summary(), ensure_ascii=False) + "\nSPEC: " + json.dumps(s, ensure_ascii=False)
    if result_meta:
        out += "\nRESULT: " + json.dumps({"kind": result_meta["kind"], "history": result_meta["history"], "fields": result_meta["fields"]}, ensure_ascii=False)
    return out


def _extract(text):
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
    try:
        return json.loads(text)
    except Exception:
        pass
    a, b = text.find("{"), text.rfind("}")
    if a >= 0 and b > a:
        return json.loads(text[a : b + 1])
    raise ValueError("LLM 응답에서 JSON을 찾지 못했습니다")


def ask_llm(cfg, text, ctx, history=None, error=None):
    import httpx

    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}]
    for h in (history or [])[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    msgs.append({"role": "user", "content": text})
    if error:
        msgs.append({"role": "user", "content": f"The previous ops failed validation: {error}\nReturn corrected JSON."})
    headers = {"Content-Type": "application/json"}
    if cfg.get("key"):
        headers["Authorization"] = "Bearer " + cfg["key"]
    body = {"model": cfg["model"], "messages": msgs, "temperature": 0, "stream": False, "response_format": {"type": "json_object"}}
    r = httpx.post(cfg["url"].rstrip("/") + "/chat/completions", json=body, headers=headers, timeout=180)
    r.raise_for_status()
    data = _extract(r.json()["choices"][0]["message"]["content"])
    ops = data.get("ops", [])
    if not isinstance(ops, list):
        raise ValueError("ops 가 배열이 아닙니다")
    return ops, str(data.get("reply", ""))


def llm_status(cfg):
    import httpx

    try:
        r = httpx.get(cfg["url"].rstrip("/") + "/models", timeout=2.5, headers={"Authorization": "Bearer " + cfg["key"]} if cfg.get("key") else {})
        r.raise_for_status()
        names = [m.get("id") for m in r.json().get("data", [])]
        return {"ok": True, "models": names, "has_model": cfg["model"] in names}
    except Exception as e:
        return {"ok": False, "error": str(e)[:200], "models": []}


NUM = r"([-+]?\d+(?:[.,]\d+)*(?:[eE][-+]?\d+)?)"
UNITS = [
    ("moment", r"(?:kN\s*[·*.]?\s*m)(?![A-Za-z])", 1e6),
    ("moment", r"(?:N\s*[·*.]?\s*mm)(?![A-Za-z])", 1.0),
    ("moment", r"(?:N\s*[·*.]?\s*m)(?![A-Za-z])", 1e3),
    ("force", r"kN(?![A-Za-z])", 1e3),
    ("force", r"(?:N|뉴턴)(?![A-Za-z])", 1.0),
    ("pressure", r"GPa", 1e3),
    ("pressure", r"MPa", 1.0),
    ("pressure", r"kPa", 1e-3),
    ("pressure", r"Pa(?![A-Za-z])", 1e-6),
    ("length", r"mm(?![A-Za-z])", 1.0),
    ("length", r"cm(?![A-Za-z])", 10.0),
    ("length", r"m(?![A-Za-z])", 1e3),
]


def _quantities(t):
    out = []
    used = []
    for kind, unit, scale in UNITS:
        for m in re.finditer(NUM + r"\s*" + unit, t):
            if any(m.start() < e and m.end() > s for s, e in used):
                continue
            used.append((m.start(), m.end()))
            out.append((kind, float(m.group(1).replace(",", "")) * scale, m.start()))
    return out


def _axis_dir(t):
    m = re.search(r"([-+]?)\s*([xyzXYZ])\s*(?:축\s*)?(?:방향|direction|쪽으로|축으로)", t)
    if m:
        v = [0.0, 0.0, 0.0]
        v["xyz".index(m.group(2).lower())] = -1.0 if m.group(1) == "-" else 1.0
        return v
    if re.search(r"아래|하방|downward|down\b", t):
        return [0.0, 0.0, -1.0]
    if re.search(r"위로|상방|upward", t):
        return [0.0, 0.0, 1.0]
    return None


def _select(t, model):
    sels = []
    for m in re.finditer(r"([xyzXYZ])\s*(?:좌표)?\s*(?:가|이|=)?\s*(최소|최대|최솟값|최댓값|min(?:imum)?|max(?:imum)?|중간|mid)", t):
        at = m.group(2).lower()
        at = "min" if at.startswith(("최소", "최솟", "min")) else "max" if at.startswith(("최대", "최댓", "max")) else "mid"
        sels.append({"plane": {"axis": m.group(1).lower(), "at": at}})
    for m in re.finditer(r"([xyzXYZ])\s*=\s*" + NUM, t):
        sels.append({"plane": {"axis": m.group(1).lower(), "at": float(m.group(2))}})
    if not sels:
        words = [(r"바닥|하단|아래쪽|아랫|밑면|bottom", "z", "min"), (r"상단|윗면|위쪽|꼭대기|top", "z", "max"), (r"왼쪽|좌측|left", "x", "min"), (r"오른쪽|우측|right", "x", "max"), (r"앞쪽|앞면|front", "y", "min"), (r"뒤쪽|뒷면|back", "y", "max")]
        for pat, ax, at in words:
            if re.search(pat, t):
                sels.append({"plane": {"axis": ax, "at": at}})
    if re.search(r"네\s*변|모든\s*변|둘레|가장자리|테두리|all edges|perimeter", t):
        return {"or": [{"plane": {"axis": a, "at": b}} for a in "xy" for b in ("min", "max")]}
    m = re.search(r"\(\s*" + NUM + r"\s*,\s*" + NUM + r"\s*,\s*" + NUM + r"\s*\)", t)
    if m:
        sels.append({"nearest": [float(m.group(i).replace(",", "")) for i in (1, 2, 3)]})
    m = re.search(r"(?:절점|노드|node)\s*#?\s*(\d+)", t, re.I)
    if m:
        sels.append({"nodes": [int(m.group(1))]})
    for g in sorted(model.group_names(), key=len, reverse=True):
        if g != "default" and re.search(r"(?<![A-Za-z0-9_])" + re.escape(g) + r"(?![A-Za-z0-9_])", t):
            sels.append({"group": g})
            break
    if not sels:
        return None
    return sels[0] if len(sels) == 1 else {"and": sels}


def _dofs(t):
    d = []
    for m in re.finditer(r"([xyzXYZ])(?:\s*,\s*([xyzXYZ]))?(?:\s*,\s*([xyzXYZ]))?\s*(?:축\s*)?(?:방향|direction|변위|병진)", t):
        for g in m.groups():
            if g:
                d.append("xyz".index(g.lower()) + 1)
    for m in re.finditer(r"([xyzXYZ])\s*(?:축\s*)?(?:회전|rotation)", t):
        d.append("xyz".index(m.group(1).lower()) + 4)
    return sorted(set(d))


def _plot(t, model):
    sel = _select(t, model)
    series = []
    title = []
    if re.search(r"에너지|energy", t, re.I):
        hit = False
        for pat, key, nm in ((r"접힘|fold", "energy.fold", "접힘"), (r"바\s*에너지|bar|신장", "energy.bar", "바"), (r"굽힘|bend", "energy.bend", "굽힘")):
            if re.search(pat, t, re.I):
                series.append({"quantity": key})
                hit = True
        if not hit or re.search(r"총|전체|토탈|total", t, re.I):
            series.insert(0, {"quantity": "energy.total"})
        title.append("에너지")
    if re.search(r"반력|reaction", t, re.I):
        ax = re.search(r"([xyzXYZ])\s*(?:축|방향)?\s*반력|반력\s*([xyzXYZ])|reaction\s*([xyz])", t)
        a = next((g for g in (ax.groups() if ax else []) if g), "z").lower()
        series.append({"quantity": f"reaction.{a}"})
        title.append("반력")
    if re.search(r"응력|stress|미세스|mises", t, re.I):
        series.append({"quantity": "vonMises", "select": sel, "reduce": "max"} if sel else {"quantity": "vonMises.max"})
        title.append("응력")
    if re.search(r"변위|처짐|displacement|deflection", t, re.I) and not re.search(r"하중\s*[-–~]\s*변위|force\s*-\s*disp", t, re.I):
        ax = re.search(r"([xyzXYZ])\s*(?:축|방향)?\s*(?:변위|처짐)", t)
        q = "u" + ax.group(1).lower() if ax else "u"
        series.append({"quantity": q, "select": sel, "reduce": "mean" if ax else "max"} if sel else {"quantity": "u.max"})
        title.append("변위")
    if re.search(r"접힘각|fold angle|접힘\s*각", t, re.I):
        series.append({"quantity": "fold.mean"})
        title.append("접힘각")
    if re.search(r"좌굴\s*계수|buckling factor", t, re.I):
        return {"op": "add_plot", "title": "좌굴 계수", "x": {"quantity": "mode"}, "series": [{"quantity": "buckling.factor"}]}
    x = {"quantity": "time"}
    if re.search(r"하중\s*[-–~]\s*변위|force\s*-\s*disp", t, re.I):
        ax = re.search(r"([xyzXYZ])\s*(?:축|방향)", t)
        a = ax.group(1).lower() if ax else "z"
        x = {"quantity": "u" + a, "select": sel, "reduce": "mean"} if sel else {"quantity": "u.max"}
        series = [{"quantity": f"reaction.{a}"}]
        title = ["하중-변위"]
    if not series:
        return None
    return {"op": "add_plot", "title": " · ".join(title) + " 이력", "x": x, "series": series}


def rule_parse(text, model, spec):
    ops = []
    notes = []
    for raw in re.split(r"[\n;]|(?<=[다요줘해])\s*[.,]\s+|\s그리고\s|\s또\s", text):
        t = raw.strip()
        if not t:
            continue
        low = t.lower()
        n0 = len(ops)
        if re.search(r"그래프|플롯|plot|graph|그려", low):
            p = _plot(t, model)
            if p:
                ops.append(p)
            else:
                notes.append("어떤 값을 그릴지 알 수 없습니다 (에너지, 반력, 변위, 응력, 접힘각)")
            continue
        m = re.search(r"\b(bc\d+|ld\d+|pl\d+)\b", low)
        if m and re.search(r"삭제|지워|제거|remove|delete", low):
            ops.append({"op": "remove", "id": m.group(1)})
            continue
        if re.search(r"초기화|모두\s*(?:삭제|지워)|전부\s*(?:삭제|지워)|clear|reset", low):
            what = "bcs" if re.search(r"구속|경계", low) else "loads" if re.search(r"하중", low) else "plots" if re.search(r"그래프", low) else "all"
            ops.append({"op": "clear", "what": what})
            continue
        for pat, name in ((r"calculix|ccx|칼큘릭스", "calculix"), (r"바\s*-?\s*힌지|bar\s*-?\s*(?:and\s*-?\s*)?hinge|barhinge", "barhinge"), (r"tensormesh|텐서메쉬|텐서메시", "tensormesh")):
            if re.search(pat, low) and re.search(r"솔버|solver|로\s*(?:풀|해석|바꿔|변경)|사용", low):
                ops.append({"op": "set_solver", "solver": name})
        st = {}
        if re.search(r"좌굴|buckl", low) and not re.search(r"좌굴\s*계수\s*그래프", low):
            st["type"] = "buckle"
        elif re.search(r"모달|고유\s*진동|modal|frequency", low):
            st["type"] = "modal"
        elif re.search(r"호장|arc\s*-?\s*length|riks", low):
            st["type"] = "arclength"
        elif re.search(r"정적\s*해석|static", low):
            st["type"] = "static"
        if re.search(r"비선형|대변형|nlgeom|nonlinear", low):
            st["nlgeom"] = not re.search(r"끄|off|해제|선형으로", low)
        m = re.search(r"모드\s*(\d+)|(\d+)\s*(?:개\s*)?모드|(\d+)\s*modes?", low)
        if m:
            st["modes"] = int(next(g for g in m.groups() if g))
        m = re.search(r"(?:증분|increments?|스텝\s*수)\s*(\d+)|(\d+)\s*(?:증분|increments?|단계로)", low)
        if m:
            st["increments"] = int(next(g for g in m.groups() if g))
        m = re.search(r"(?:초기\s*결함|imperfection)\s*" + NUM, low)
        if m:
            st["imperfection"] = float(m.group(1))
        if st:
            ops.append({"op": "set_step", **st})
        m = re.search(r"(?:분할|division)s?\s*(?:수)?\s*(?:를|을|=)?\s*(\d+)|(\d+)\s*분할", low)
        if m:
            ops.append({"op": "set_divisions", "divisions": int(next(g for g in m.groups() if g))})
        qs = _quantities(t)
        m = re.search(r"두께\s*(?:를|을|는|=|t)?\s*" + NUM + r"|thickness\s*=?\s*" + NUM, t, re.I)
        if m:
            ops.append({"op": "set_section", "group": "*", "thickness": float(next(g for g in m.groups() if g))})
            qs = [q for q in qs if q[0] != "length"]
        mat = {}
        m = re.search(r"(?:탄성\s*계수|영률|young'?s?\s*modulus|\bE\b)\s*(?:를|을|는|=)?\s*" + NUM + r"\s*(GPa|MPa)?", t)
        if m:
            mat["E"] = float(m.group(1)) * (1e3 if (m.group(2) or "MPa") == "GPa" else 1.0)
        m = re.search(r"(?:포아송\s*비?|푸아송\s*비?|poisson'?s?\s*(?:ratio)?|\bnu\b)\s*(?:를|을|는|=)?\s*" + NUM, t, re.I)
        if m:
            mat["nu"] = float(m.group(1))
        m = re.search(r"(?:항복\s*(?:응력|강도)?|yield)\s*(?:를|을|는|=)?\s*" + NUM, t, re.I)
        if m:
            mat["yield"] = float(m.group(1))
        if mat:
            ops.append({"op": "set_material", "name": "default", **mat})
            qs = [q for q in qs if q[0] != "pressure"]
        m = re.search(r"(?:힌지\s*강성|kappa|κ)\s*(?:를|을|는|=)?\s*" + NUM, t, re.I)
        if m:
            ops.append({"op": "set_hinge", "group": "*", "kappa": float(m.group(1))})
        sel = _select(t, model)
        if re.search(r"중력|자중|gravity", low):
            ops.append({"op": "add_load", "type": "gravity", "vector": [0, 0, -9810.0]})
        kinds = {q[0] for q in qs}
        d = _axis_dir(t)
        if "force" in kinds or "moment" in kinds:
            kind = "moment" if "moment" in kinds else "force"
            val = next(q[1] for q in qs if q[0] == kind)
            if d is None and re.search(r"압축|compress", low) and sel and "plane" in sel:
                ax = "xyz".index(sel["plane"]["axis"])
                d = [0.0, 0.0, 0.0]
                d[ax] = -1.0 if sel["plane"]["at"] == "max" else 1.0
                val = abs(val)
            if d is None:
                notes.append(f"'{t}': 하중 방향을 알려주세요 (예: z 방향으로 -1000 N)")
            elif sel is None:
                notes.append(f"'{t}': 하중을 가할 위치를 알려주세요 (예: x 최대 끝단)")
            else:
                ops.append({"op": "add_load", "type": kind, "select": sel, "vector": [c * val + 0.0 for c in d]})
        elif "pressure" in kinds and re.search(r"압력|pressure|면압|등분포", low):
            val = next(q[1] for q in qs if q[0] == "pressure")
            op = {"op": "add_load", "type": "pressure", "select": sel or {"all": True}, "value": abs(val)}
            if d is not None:
                op["direction"] = [c * (1 if val >= 0 else -1) for c in d]
            ops.append(op)
        elif "length" in kinds and re.search(r"변위|밀어|당겨|이동|누르|눌러|displace|push|pull|move", low):
            val = next(q[1] for q in qs if q[0] == "length")
            if d is None or sel is None:
                notes.append(f"'{t}': 변위의 위치와 방향을 알려주세요 (예: x 최대 변을 x 방향으로 -5 mm)")
            else:
                ax = max(range(3), key=lambda i: abs(d[i]))
                ops.append({"op": "add_bc", "name": "지정 변위", "select": sel, "dofs": [ax + 1], "value": val * d[ax]})
        elif re.search(r"고정|fix|clamp|encast|핀\s*지지|핀으로|pinned|단순\s*지지|구속|지지|막아|constrain|support|대칭|symm", low):
            if sel is None:
                notes.append(f"'{t}': 구속할 위치를 알려주세요 (예: z 최소 면)")
            else:
                dofs = _dofs(t)
                op = {"op": "add_bc", "select": sel}
                m = re.search(r"([xyz])\s*(?:축\s*)?대칭|([xyz])\s*-?\s*symm", low)
                if m:
                    op["type"] = next(g for g in m.groups() if g) + "symm"
                    op["name"] = "대칭"
                elif dofs and not re.search(r"완전\s*고정|전부\s*고정|fully", low):
                    op["dofs"] = dofs
                    op["name"] = "구속 " + "".join("xyz"[k - 1] if k <= 3 else "r" + "xyz"[k - 4] for k in dofs)
                elif re.search(r"핀|pinned|단순\s*지지", low):
                    op["type"] = "pinned"
                    op["name"] = "핀 지지"
                else:
                    op["type"] = "fixed"
                    op["name"] = "고정"
                ops.append(op)
        if len(ops) == n0:
            notes.append(f"'{t}': 해석하지 못했습니다")
    reply = f"{len(ops)}개 수정 지시를 만들었습니다 (규칙 기반 해석)." if ops else "요청을 해석하지 못했습니다."
    if notes:
        reply += " " + " / ".join(notes)
    return ops, reply
