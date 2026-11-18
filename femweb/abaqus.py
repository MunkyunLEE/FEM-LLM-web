import numpy as np

from .contact import contacts_of, icosphere, plane_basis, rigid_faces, unit
from .loads import bc_table, lumped
from .select import resolve
from .spec import BC_TYPES, material_for, section_for, solid_rotations

SYM = {"fixed": "ENCASTRE", "pinned": "PINNED", "xsymm": "XSYMM", "ysymm": "YSYMM", "zsymm": "ZSYMM", "xasymm": "XASYMM", "yasymm": "YASYMM", "zasymm": "ZASYMM"}


def _rows(ids, k=12):
    ids = list(ids)
    return [", ".join(str(int(x)) for x in ids[i : i + k]) for i in range(0, len(ids), k)]


def _name(s):
    out = "".join(c if c.isalnum() or c == "_" else "_" for c in str(s)).upper()
    return out if out and out[0].isalpha() else "S_" + out


def export(model, spec, info=None, element="S4R"):
    from .control import disp_control

    info = info or {}
    spec, rnotes = solid_rotations(model, spec)
    notes = list(rnotes)
    spec, ctl = disp_control(model, spec)
    if ctl and ctl["kind"] == "average":
        notes.append("평균 변위 제어(하중 분포 유지)는 Abaqus 에 그대로 옮길 수 없어 하중 제어로 내보냈습니다. 목표 변위에 맞는 하중 크기는 Abaqus 에서 Riks 나 변위 제어로 다시 정해야 합니다")
        spec["loads"] = spec["loads"] + [l for l in ctl["one"]["loads"]]
    elif ctl:
        notes += ctl["notes"]
    st = spec["step"]
    solver = spec.get("solver", "calculix")
    kind = st.get("type", "static")
    X = np.asarray(info.get("X") if info.get("X") is not None else model.nodes, float)
    if len(X) != len(model.nodes):
        X = np.asarray(model.nodes, float)
    imp = float(st.get("imperfection") or 0.0)
    modes = st.get("imperfection_modes") or []
    if (imp > 0 or modes) and info.get("X") is None:
        notes.append("초기 결함이 지정되어 있지만 해석 결과가 없어 결함 없는 좌표로 내보냈습니다. 자체 쉘로 한 번 풀고 내보내면 같은 결함 좌표가 들어갑니다")
    elif imp > 0 or modes:
        notes.append("절점 좌표는 자체 솔버가 쓴 초기 결함(좌굴 모드 합) 좌표입니다")
    nodes = [p for p in X]
    mesh_rigid = [c for c in contacts_of(spec) if c["type"] == "rigid" and c.get("shape") == "mesh"]
    rigid_face = set()
    for c in mesh_rigid:
        rigid_face.update(rigid_faces(model, c["body"]))
    mids = {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in mids:
            nodes.append(0.5 * (nodes[a] + nodes[b]))
            mids[k] = len(nodes) - 1
        return mids[k]

    els = {}
    elsets = {}
    eid = 0
    stype3 = "S3R" if element.endswith("R") else "S3"
    for (a, b), g in zip(model.beams, model.beam_group):
        eid += 1
        els.setdefault("B31", []).append(f"{eid}, {a + 1}, {b + 1}")
        elsets.setdefault(("B", g), []).append(eid)
    deform = []
    for k, (f, g) in enumerate(zip(model.shells, model.shell_group)):
        if k in rigid_face:
            continue
        eid += 1
        typ = element if len(f) == 4 else stype3
        els.setdefault(typ, []).append(f"{eid}, " + ", ".join(str(int(i) + 1) for i in f))
        elsets.setdefault(("S", g), []).append(eid)
        deform.append(eid)
    for t in model.solids:
        eid += 1
        a, b, c, d = (int(i) for i in t)
        ids = [a, b, c, d, mid(a, b), mid(b, c), mid(a, c), mid(a, d), mid(b, d), mid(c, d)]
        els.setdefault("C3D10", []).append(f"{eid}, " + ", ".join(str(i + 1) for i in ids))
        elsets.setdefault(("V", "solid"), []).append(eid)
        deform.append(eid)
    if model.hinges:
        notes.append("접힘선(힌지)은 Abaqus 로 옮기지 않았습니다. 공유 엣지는 강접으로 이어집니다")
    span = float(np.ptp(np.asarray(model.nodes, float), axis=0).max()) or 1.0
    rigid = []
    for c in contacts_of(spec):
        if c["type"] != "rigid":
            continue
        if c["shape"] == "plane":
            p = np.asarray(c["point"], float)
            n = unit(c["normal"])
            e1, e2, _ = plane_basis(n)
            h = 1.5 * span
            V = np.array([p + h * (s * e1 + t * e2) for s, t in ((-1, -1), (1, -1), (1, 1), (-1, 1))])
            F = np.array([[0, 1, 2, 3]])
            if np.cross(V[1] - V[0], V[2] - V[0]) @ n < 0:
                F = F[:, ::-1]
            ref = p
        elif c["shape"] == "sphere":
            v, f = icosphere(3)
            V = np.asarray(c["center"], float) + float(c["radius"]) * v
            F = np.asarray(f, int)
            ref = np.asarray(c["center"], float)
        else:
            fs = rigid_faces(model, c["body"])
            vid = sorted({int(i) for k in fs for i in model.shells[k]})
            vm = {o: k for k, o in enumerate(vid)}
            V = np.asarray(model.nodes, float)[vid]
            F = [[vm[int(i)] for i in model.shells[k]] for k in fs]
            ref = V.mean(0)
        b = len(nodes)
        nodes += list(V)
        nodes.append(np.asarray(ref, float))
        rp = len(nodes) - 1
        rows = []
        for f in F:
            eid += 1
            rows.append((eid, [b + int(i) for i in f]))
        rigid.append((c, rp, rows))
    cps = spec.get("couplings") or []
    cpn = []
    for cp in cps:
        sel = {int(i) for i in resolve(model, cp["select"])}
        ext = set(sel)
        for (a0, b0), mn in mids.items():
            if a0 in sel and b0 in sel:
                ext.add(mn)
        nodes.append(np.asarray(cp["ref"], float))
        cpn.append((cp, sorted(ext), len(nodes) - 1))
    L = ["*HEADING", f"FEM-LLM-web export ({solver} spec)"]
    L.append("** units mm, N, tonne, s, MPa")
    L.append("** history output map (Abaqus -> FEM-LLM-web):")
    L.append("**   ALLSE -> energy.strain, ALLPD -> energy.plastic, ALLVD -> energy.damping, ALLFD -> energy.friction, ALLKE -> energy.kinetic, ALLWK -> energy.external")
    for c, rp, _ in rigid:
        L.append(f"**   RP_{_name(c['id'])} (node {rp + 1}) RF/U along motion -> contact.{c['id']}.force / contact.disp")
    for cp, _, rp in cpn:
        L.append(f"**   RP_{_name(cp['id'])} (node {rp + 1}) U, UR, RF, RM -> rp.{cp['id']}.*")
    L.append("*NODE, NSET=NALL")
    L += [f"{i + 1}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}" for i, p in enumerate(nodes)]
    for typ, rows in els.items():
        L.append(f"*ELEMENT, TYPE={typ}, ELSET=E_{typ}")
        L += rows
    names = {}
    for key, ids in elsets.items():
        nm = _name(f"{'BEAM' if key[0] == 'B' else 'SOLID' if key[0] == 'V' else 'SHELL'}_{key[1]}")
        names[key] = nm
        L.append(f"*ELSET, ELSET={nm}")
        L += _rows(ids)
    if deform:
        L.append("*ELSET, ELSET=EDEFORM")
        L += _rows(deform)
    for c, rp, rows in rigid:
        nm = _name(c["id"])
        tri = [r for r in rows if len(r[1]) == 3]
        quad = [r for r in rows if len(r[1]) == 4]
        if tri:
            L.append(f"*ELEMENT, TYPE=R3D3, ELSET=ER_{nm}")
            L += [f"{e}, " + ", ".join(str(i + 1) for i in f) for e, f in tri]
        if quad:
            L.append(f"*ELEMENT, TYPE=R3D4, ELSET=ER_{nm}")
            L += [f"{e}, " + ", ".join(str(i + 1) for i in f) for e, f in quad]
        L += [f"*NSET, NSET=RP_{nm}", f"{rp + 1}", f"*RIGID BODY, REF NODE={rp + 1}, ELSET=ER_{nm}"]
    for cp, ext, rp in cpn:
        nm = _name(cp["id"])
        L.append(f"*NSET, NSET=CP_{nm}")
        L += _rows([x + 1 for x in ext])
        L += [f"*NSET, NSET=RP_{nm}", f"{rp + 1}", f"*KINEMATIC COUPLING, REF NODE={rp + 1}", f"CP_{nm}, 1, 3"]
    mats = {}
    for key in elsets:
        g = key[1] if key[0] != "V" else "*"
        mats[section_for(spec, g).get("material", "default")] = material_for(spec, g)
    dsp = st.get("damping") or {}
    dkind = dsp.get("kind", "none")
    al = be = None
    if dkind != "none" and solver == "shell":
        if dkind == "direct":
            al, be = float(dsp.get("alpha") or 0.0), float(dsp.get("beta") or 0.0)
        elif info.get("alpha") is not None:
            al, be = info["alpha"], info["beta"]
            notes.append(f"레일리 감쇠 α = {al:.6g} 1/s, β = {be:.6g} s 는 자체 솔버가 1·2차 고유진동수 {info.get('f1', 0):.4g}, {info.get('f2', 0):.4g} Hz 에서 구한 값입니다")
        else:
            notes.append("감쇠 계수는 해석을 한 번 돌려야 정해집니다. 지금 파일은 감쇠 없이 정적 스텝으로 내보냈습니다")
    for name, m in mats.items():
        L += [f"*MATERIAL, NAME=M_{_name(name)}", "*ELASTIC", f"{m['E']:.9g}, {m['nu']:.9g}", "*DENSITY", f"{float(m.get('rho') or 0.0) or 1e-12:.9g}"]
        if al is not None:
            L += [f"*DAMPING, ALPHA={al:.9g}, BETA={be:.9g}"]
        if m.get("curve"):
            L += ["*PLASTIC"] + [f"{sg:.9g}, {ep:.9g}" for ep, sg in m["curve"]]
        elif m.get("power"):
            k, n = (float(x) for x in m["power"])
            sy = float(m.get("yield") or 0.0)
            pts = [0.0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
            L += ["*PLASTIC"] + [f"{max(sy, k * e ** n) if e > 0 else sy:.9g}, {e:.9g}" for e in pts]
        elif m.get("yield"):
            H = float(m.get("hardening") or 0.0)
            L += ["*PLASTIC", f"{m['yield']:.9g}, 0.", f"{m['yield'] + H:.9g}, 1."]
    for key, nm in names.items():
        kindk, g = key
        sec = section_for(spec, g if kindk != "V" else "*")
        mn = "M_" + _name(sec.get("material", "default"))
        if kindk == "S":
            L += [f"*SHELL SECTION, ELSET={nm}, MATERIAL={mn}", f"{float(sec['thickness']):.9g}, 5"]
        elif kindk == "V":
            L += [f"*SOLID SECTION, ELSET={nm}, MATERIAL={mn}"]
        else:
            b = sec.get("beam") or {}
            if str(b.get("shape", "rect")).lower().startswith("circ"):
                r = float(b.get("r", b.get("d", 10.0) / 2 if "d" in b else 5.0))
                L += [f"*BEAM SECTION, ELSET={nm}, MATERIAL={mn}, SECTION=CIRC", f"{r:.9g}"]
            else:
                L += [f"*BEAM SECTION, ELSET={nm}, MATERIAL={mn}, SECTION=RECT", f"{float(b.get('b', 10)):.9g}, {float(b.get('h', 10)):.9g}"]
    cs = [c for c in contacts_of(spec, True)] if solver == "shell" else contacts_of(spec)
    if cs:
        L.append("*SURFACE, TYPE=ELEMENT, NAME=SDEF")
        L.append("EDEFORM,")
        incl = []
        selfc = any(c["type"] == "self" for c in cs)
        for c, rp, _ in rigid:
            nm = _name(c["id"])
            L += [f"*SURFACE, TYPE=ELEMENT, NAME=SR_{nm}", f"ER_{nm},"]
            incl.append(("SDEF", f"SR_{nm}", c))
        for c in cs:
            if c["type"] == "pair":
                nm = _name(c["id"])
                for side in ("a", "b"):
                    ids = {int(i) for i in resolve(model, c[side])}
                    ee = [e for e, f in _deform_faces(model, rigid_face) if all(int(i) in ids for i in f)]
                    L += [f"*ELSET, ELSET=EP_{nm}_{side.upper()}"] + _rows(ee)
                    L += [f"*SURFACE, TYPE=ELEMENT, NAME=SP_{nm}_{side.upper()}", f"EP_{nm}_{side.upper()},"]
                incl.append((f"SP_{nm}_A", f"SP_{nm}_B", c))
            elif c["type"] == "self":
                incl.append(("SDEF", "SDEF", c))
        for c in {id(x[2]): x[2] for x in incl}.values():
            nm = _name(c["id"])
            L += [f"*SURFACE INTERACTION, NAME=IP_{nm}", "*FRICTION", f"{float(c.get('friction') or 0.0):.6g}"]
        L.append("*CONTACT")
        if selfc and not any(c["type"] == "pair" for c in cs):
            L.append("*CONTACT INCLUSIONS, ALL EXTERIOR")
        else:
            L.append("*CONTACT INCLUSIONS")
            L += [f"{a}, {b}" for a, b, _ in incl]
        L.append("*CONTACT PROPERTY ASSIGNMENT")
        L += [f"{a}, {b}, IP_{_name(c['id'])}" for a, b, c in incl]
        if solver == "shell":
            notes.append("접촉은 Abaqus 일반 접촉(쉘 두께 반영)으로 옮겼습니다. 자체 솔버의 IPC 장벽 범위(0.25 t) 만큼 접촉 시작 위치가 다를 수 있습니다")
    table = bc_table(model, spec)
    btypes = {}
    for bc in spec["bcs"]:
        if bc.get("type") in SYM and len(model.solids) == 0:
            btypes[bc["id"]] = SYM[bc["type"]]
    solid_only = len(model.solids) > 0 and not model.shells and not model.beams
    fixz, mov = [], []
    for (n, d), v in table.items():
        if solid_only and d > 3:
            continue
        (fixz if v == 0.0 else mov).append((n, d, v))
    typed = []
    for bc in spec["bcs"]:
        if bc["id"] in btypes:
            ids = [int(i) for i in resolve(model, bc["select"])]
            typed.append((bc, ids))
    tdofs = {(n, d) for bc, ids in typed for n in ids for d in BC_TYPES[bc["type"]]}
    fixz = [x for x in fixz if (x[0], x[1]) not in tdofs]
    mov = [x for x in mov if (x[0], x[1]) not in tdofs]
    if mids:
        tb = {(n, d): v for n, d, v in fixz + mov}
        for (a, b), mn in mids.items():
            for d in (1, 2, 3):
                if (a, d) in tb and (b, d) in tb:
                    v = 0.5 * (tb[(a, d)] + tb[(b, d)])
                    (fixz if v == 0.0 else mov).append((mn, d, v))
    for cp, ext, rp in cpn:
        ex = set(ext)
        moved = sorted({d for n, d, _ in fixz + mov if n in ex})
        if moved:
            fixz = [x for x in fixz if x[0] not in ex]
            mov = [x for x in mov if x[0] not in ex]
            fx = cp.get("fix") or {}
            for d in moved:
                if str(d) not in fx:
                    fixz.append((rp, d, 0.0))
            notes.append(f"{cp['id']}: 기준점에 묶인 절점의 구속(자유도 {', '.join(map(str, moved))})은 기준점 구속으로 옮겼습니다")
        for dd, v in (cp.get("fix") or {}).items():
            (fixz if float(v) == 0.0 else mov).append((rp, int(dd), float(v)))
    for bc, ids in typed:
        nm = _name(bc["id"])
        L += [f"*NSET, NSET=BC_{nm}"] + _rows([i + 1 for i in ids])
    movers = [bc for bc in spec["bcs"] if float(bc.get("value", 0.0)) != 0.0 and bc["id"] not in btypes]
    for bc in movers:
        ids = {int(i) for i in resolve(model, bc["select"])}
        ids |= {mn for (a0, b0), mn in mids.items() if a0 in ids and b0 in ids}
        L += [f"*NSET, NSET=BC_{_name(bc['id'])}"] + _rows([i + 1 for i in sorted(ids)])
        L.insert(5, f"**   BC_{_name(bc['id'])} sum of RF -> reaction.{bc['id']}.*")
    L.append("*BOUNDARY")
    for bc, ids in typed:
        L.append(f"BC_{_name(bc['id'])}, {btypes[bc['id']]}")
    L += [f"{n + 1}, {d}, {d}, 0." for n, d, _ in fixz]
    for c, rp, _ in rigid:
        mv = np.asarray(c.get("motion", [0, 0, 0]), float)
        L += [f"{rp + 1}, {d}, {d}, 0." for d in range(1, 7) if d > 3 or mv[d - 1] == 0.0]
    inc = max(1, int(st.get("increments", 25)))
    amp = st.get("amplitude", "linear")
    T = float(info.get("time") or dsp.get("time") or 1.0) if al is not None else 1.0
    L += ["*AMPLITUDE, NAME=AMP" + (", DEFINITION=SMOOTH STEP" if amp == "smooth" else ""), f"0., 0., {T:.9g}, 1."]
    nl = solver == "shell" or bool(st.get("nlgeom"))
    if kind == "buckle":
        L += ["*STEP", "*BUCKLE", f"{int(st.get('modes', 5))}, , 30, 300"]
    elif kind == "modal":
        L += ["*STEP", "*FREQUENCY, EIGENSOLVER=LANCZOS", f"{int(st.get('modes', 5))}"]
    elif al is not None:
        L += [f"*STEP, NLGEOM=YES, INC=1000000", "*DYNAMIC, APPLICATION=QUASI-STATIC, INITIAL=NO", f"{T / inc:.6g}, {T:.9g}, {T * 1e-8:.6g}, {T / inc:.6g}"]
    else:
        L += [f"*STEP, NLGEOM={'YES' if nl else 'NO'}, INC=1000000", "*STATIC", f"{1.0 / inc:.6g}, 1., 1e-8, {1.0 / inc:.6g}"]
    ampk = "" if kind in ("buckle", "modal") else ", AMPLITUDE=AMP"
    bmov = list(mov)
    for c, rp, _ in rigid:
        mv = np.asarray(c.get("motion", [0, 0, 0]), float)
        bmov += [(rp, d, float(mv[d - 1])) for d in (1, 2, 3) if mv[d - 1] != 0.0]
    if bmov and kind != "modal":
        L.append(f"*BOUNDARY{ampk}")
        L += [f"{n + 1}, {d}, {d}, {v:.9g}" for n, d, v in bmov]
    if kind != "modal":
        F, M, a = lumped(model, spec)
        rows = []
        for i in range(len(model.nodes)):
            for d in range(3):
                if F[i, d] != 0.0:
                    rows.append(f"{i + 1}, {d + 1}, {F[i, d]:.9g}")
                if M[i, d] != 0.0:
                    rows.append(f"{i + 1}, {d + 4}, {M[i, d]:.9g}")
        for cp, _, rp in cpn:
            for d in range(3):
                if cp.get("force") and cp["force"][d]:
                    rows.append(f"{rp + 1}, {d + 1}, {float(cp['force'][d]):.9g}")
                if cp.get("moment") and cp["moment"][d]:
                    rows.append(f"{rp + 1}, {d + 4}, {float(cp['moment'][d]):.9g}")
        if rows:
            L.append(f"*CLOAD{ampk}")
            L += rows
        if a["gravity"] is not None and float(np.linalg.norm(a["gravity"])) > 0:
            g = np.asarray(a["gravity"], float)
            mag = float(np.linalg.norm(g))
            L += [f"*DLOAD{ampk}", f"EDEFORM, GRAV, {mag:.9g}, {g[0] / mag:.9g}, {g[1] / mag:.9g}, {g[2] / mag:.9g}"]
    if kind in ("buckle", "modal"):
        L += ["*OUTPUT, FIELD", "*NODE OUTPUT", "U"]
    else:
        L += [f"*OUTPUT, FIELD, NUMBER INTERVAL={min(inc, 100)}", "*NODE OUTPUT", "U, RF", "*ELEMENT OUTPUT", "S, PE, PEEQ"]
        if cs:
            L += ["*CONTACT OUTPUT", "CSTRESS, CFORCE"]
        L += ["*OUTPUT, HISTORY, FREQUENCY=1", "*ENERGY OUTPUT", "ALLSE, ALLPD, ALLVD, ALLFD, ALLKE, ALLWK, ETOTAL"]
        for c, rp, _ in rigid:
            L += [f"*NODE OUTPUT, NSET=RP_{_name(c['id'])}", "U, RF"]
        for cp, _, rp in cpn:
            L += [f"*NODE OUTPUT, NSET=RP_{_name(cp['id'])}", "U, UR, RF, RM"]
        for bc in movers:
            L += [f"*NODE OUTPUT, NSET=BC_{_name(bc['id'])}", "U, RF"]
    L.append("*END STEP")
    L[2:2] = ["** " + t for t in notes]
    return "\n".join(L) + "\n", notes


def _deform_faces(model, skip):
    e = 0
    out = []
    for k, f in enumerate(model.shells):
        if k in skip:
            continue
        e += 1
        out.append((e + len(model.beams), f))
    return out
