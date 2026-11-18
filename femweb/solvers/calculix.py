import os
import re
import shutil
import subprocess
import tempfile
import time
import numpy as np
from ..loads import assemble, bc_table
from ..select import resolve
from ..result import Result, von_mises
from ..spec import material_for, section_for


def find_ccx():
    env = os.environ.get("FEMWEB_CCX")
    if env and shutil.which(env):
        return shutil.which(env)
    for name in ("ccx", "ccx_2.23", "ccx_2.22", "ccx_2.21", "ccx_2.20", "ccx_2.19", "ccx.exe"):
        p = shutil.which(name)
        if p:
            return p
    for d in ("/opt/homebrew/bin", "/usr/local/bin"):
        if os.path.isdir(d):
            for f in sorted(os.listdir(d), reverse=True):
                if f.startswith("ccx"):
                    return os.path.join(d, f)
    return None


def hinge_kappa(spec, name, model, group_hint=None):
    h = dict(spec["hinges"]["*"])
    h.update(spec["hinges"].get(name, {}))
    if h.get("pin"):
        return 0.0
    if h.get("kappa") is not None:
        return float(h["kappa"])
    g = group_hint or (model.shell_group[0] if model.shell_group else "*")
    mat = material_for(spec, g)
    t = section_for(spec, g)["thickness"]
    D = mat["E"] * t**3 / (12 * (1 - mat["nu"] ** 2))
    return float(h.get("ratio", 0.1)) * D / panel_width(model)


def panel_width(model):
    if not model.shells:
        return 1.0
    w = []
    for s in model.shells:
        p = model.nodes[s]
        n = len(s)
        w.append(np.mean([np.linalg.norm(p[i] - p[(i + 1) % n]) for i in range(n)]))
    return float(np.mean(w))


def _beam_dir(t):
    z = np.array([0.0, 0.0, 1.0])
    n1 = np.cross(z, t)
    if np.linalg.norm(n1) < 0.2:
        return (1.0, 0.0, 0.0)
    n1 = n1 / np.linalg.norm(n1)
    return tuple(np.round(n1, 3))


def build_inp(model, base, spec, step=None, perturb=None, out2d=False, ctl=None):
    from ..hinges import split_pins

    st = dict(spec["step"])
    if step:
        st.update(step)
    n_orig = len(model.nodes)
    method = os.environ.get("FEMWEB_HINGE", spec.get("hinge_method", "spring"))
    rigid = []
    if method != "strip" and model.hinges and model.shells:
        from ..hinges import is_pin

        wm = panel_width(model)
        for name in list(model.hinges):
            g = model.shell_group[0]
            sec = section_for(spec, g)
            mat = material_for(spec, g)
            D = mat["E"] * sec["thickness"] ** 3 / (12 * (1 - mat["nu"] ** 2))
            if not is_pin(spec, name) and hinge_kappa(spec, name, base, g) * wm / (4 * D) > 0.5:
                rigid.append(name)
        if rigid:
            model = model.copy()
            model.hinges = {k: v for k, v in model.hinges.items() if k not in rigid}
    model, ties, tie_pairs, pin_names, joints = split_pins(model, spec, every=method != "strip")
    if perturb is not None and len(perturb) < len(model.nodes):
        extra = np.array([perturb[a] for a, _ in ties]).reshape(-1, 3)
        perturb = np.vstack([perturb, extra])
    nodes = [p for p in (model.nodes if perturb is None else model.nodes + perturb)]
    mids = {}

    def mid(a, b):
        k = (min(a, b), max(a, b))
        if k not in mids:
            nodes.append(0.5 * (nodes[a] + nodes[b]))
            mids[k] = len(nodes) - 1
        return mids[k]

    L = ["*HEADING", "FEM-LLM-web"]
    elsets = {}
    eid = 0
    etext = {}
    hinge_of = {}
    hinge_w = {}
    if model.hinges:
        hedge = {}
        for name, edges in model.hinges.items():
            for e in edges:
                hedge[e] = name
        for fi, s in enumerate(model.shells):
            n = len(s)
            for i in range(n):
                e = (min(s[i], s[(i + 1) % n]), max(s[i], s[(i + 1) % n]))
                if e in hedge:
                    hinge_of[fi] = hedge[e]
                    p = model.nodes[s]
                    le = np.linalg.norm(model.nodes[e[0]] - model.nodes[e[1]])
                    area = 0.5 * np.linalg.norm(np.cross(p[2] - p[0], p[-1] - p[1])) if n == 4 else 0.5 * np.linalg.norm(np.cross(p[1] - p[0], p[2] - p[0]))
                    hinge_w.setdefault(hedge[e], []).append(area / le)
                    break
    shell_eids = []
    solid_eids = []
    for (a, b), g in zip(model.beams, model.beam_group):
        eid += 1
        t = model.nodes[b] - model.nodes[a]
        t = t / np.linalg.norm(t)
        key = ("B31", g, _beam_dir(t))
        elsets.setdefault(key, []).append(eid)
        etext.setdefault("B31", []).append(f"{eid}, {a + 1}, {b + 1}")
    for fi, (s, g) in enumerate(zip(model.shells, model.shell_group)):
        eid += 1
        n = len(s)
        ids = list(s) + [mid(s[i], s[(i + 1) % n]) for i in range(n)]
        typ = "S8R" if n == 4 else "S6"
        key = ("SH", g, hinge_of.get(fi))
        shell_eids.append(eid)
        elsets.setdefault(key, []).append(eid)
        etext.setdefault(typ, []).append(f"{eid}, " + ", ".join(str(i + 1) for i in ids))
    spring_notes = {}
    nseg = int(os.environ.get("FEMWEB_HSEG", "1"))
    btype = os.environ.get("FEMWEB_HTYPE", "B32R")
    for name, k0, k1, f2 in joints:
        from ..hinges import is_pin

        if is_pin(spec, name) or k0 not in mids or k1 not in mids:
            continue
        g = model.shell_group[f2]
        mat = material_for(spec, g)
        kap = hinge_kappa(spec, name, base, g)
        s2 = model.shells[f2]
        n2 = len(s2)
        i = next(j for j in range(n2) if (min(s2[j], s2[(j + 1) % n2]), max(s2[j], s2[(j + 1) % n2])) == k1)
        far = mid(s2[(i + 2) % n2], s2[(i + 3) % n2]) if n2 == 4 else s2[(i + 2) % n2]
        m1 = mids[k0]
        le = float(np.linalg.norm(nodes[k1[0]] - nodes[k1[1]]))
        w = float(np.linalg.norm(nodes[far] - nodes[m1]))
        hdir = (nodes[k1[1]] - nodes[k1[0]]) / le
        cos2 = float(((nodes[far] - nodes[m1]) / w) @ hdir) ** 2
        I = kap * le * w / (4.0 * mat["E"] * ((1.0 - cos2) + 0.2109 * cos2))
        a = (12.0 * I) ** 0.25
        chain = [m1]
        nn = nseg * (2 if btype.startswith("B32") else 1)
        for q in range(1, nn):
            nodes.append(nodes[m1] + (nodes[far] - nodes[m1]) * q / nn)
            chain.append(len(nodes) - 1)
        chain.append(far)
        t = (nodes[far] - nodes[m1]) / w
        key = ("HB", g, (name, round(a, 5), _beam_dir(t)))
        step = 2 if btype.startswith("B32") else 1
        for q in range(0, len(chain) - 1, step):
            eid += 1
            elsets.setdefault(key, []).append(eid)
            etext.setdefault(btype, []).append(f"{eid}, " + ", ".join(str(x + 1) for x in chain[q : q + step + 1]))
        sn = spring_notes.setdefault(name, [kap, 0, []])
        sn[1] += 1
        sn[2].append(a)
    for t in model.solids:
        eid += 1
        a, b, c, d = (int(i) for i in t)
        ids = [a, b, c, d, mid(a, b), mid(b, c), mid(a, c), mid(a, d), mid(b, d), mid(c, d)]
        solid_eids.append(eid)
        elsets.setdefault(("SO", "solid", None), []).append(eid)
        etext.setdefault("C3D10", []).append(f"{eid}, " + ", ".join(str(i + 1) for i in ids))
    from ..contact import ccx_build

    cx = ccx_build(model, spec, nodes, eid, shell_eids, solid_eids, st["type"] in ("static", "arclength"))
    if cx:
        eid = cx["eid"]
    ref = None
    if ctl and ctl["kind"] == "average" and ctl.get("k"):
        hi = np.array(nodes).max(axis=0)
        far = hi + 0.1 * (hi - np.array(nodes).min(axis=0)) + 1.0
        nodes.append(far)
        nodes.append(far + np.array([1.0, 0.0, 0.0]))
        ref = len(nodes) - 2
        eid += 1
        ctl_eid = eid
    cpn = []
    for cp in spec.get("couplings") or []:
        sel = {int(i) for i in resolve(model, cp["select"])}
        ext = set(sel)
        for (a0, b0), mn in mids.items():
            if a0 in sel and b0 in sel:
                ext.add(mn)
        nodes.append(np.asarray(cp["ref"], float))
        nodes.append(np.asarray(cp["ref"], float))
        cpn.append((cp, sorted(ext), len(nodes) - 2, len(nodes) - 1))
    L.append("*NODE, NSET=NALL")
    for i, p in enumerate(nodes):
        L.append(f"{i + 1}, {p[0]:.9g}, {p[1]:.9g}, {p[2]:.9g}")
    if ref is not None:
        L += ["*NSET, NSET=NREF", f"{ref + 1}", "*NSET, NSET=NREF2", f"{ref + 2}", "*ELEMENT, TYPE=T3D2, ELSET=ECTL", f"{ctl_eid}, {ref + 1}, {ref + 2}"]
    for cp, ext, r1, r2 in cpn:
        tag = cp["id"].upper()
        L.append(f"*NSET, NSET={tag}")
        for i in range(0, len(ext), 12):
            L.append(", ".join(str(x + 1) for x in ext[i : i + 12]))
        L += [f"*NSET, NSET={tag}R", f"{r1 + 1}", f"*NSET, NSET={tag}Q", f"{r2 + 1}", f"*RIGID BODY, NSET={tag}, REF NODE={r1 + 1}, ROT NODE={r2 + 1}"]
    names = {}
    for k, key in enumerate(elsets):
        names[key] = f"ES{k + 1}"
    for typ, rows in etext.items():
        L.append(f"*ELEMENT, TYPE={typ}, ELSET=E{typ}")
        L += rows
    if cx:
        L += cx["elements"]
    for key, ids in elsets.items():
        L.append(f"*ELSET, ELSET={names[key]}")
        for i in range(0, len(ids), 12):
            L.append(", ".join(str(x) for x in ids[i : i + 12]))
    L.append("*ELSET, ELSET=EALL")
    L.append(", ".join(names.values()))
    mats = {}
    for key in elsets:
        g = key[1] if key[0] != "SO" else "*"
        name = section_for(spec, g).get("material", "default")
        mats[name] = spec["materials"].get(name, spec["materials"]["default"])
    if ref is not None:
        L += ["*MATERIAL, NAME=MCTL", "*ELASTIC", f"{ctl['k']:.9g}, 0.", "*DENSITY", "1e-20", "*SOLID SECTION, ELSET=ECTL, MATERIAL=MCTL", "1."]
    plastic = False
    hmats = sorted({section_for(spec, key[1]).get("material", "default") for key in elsets if key[0] == "HB"})
    for name in hmats:
        L += [f"*MATERIAL, NAME=MH_{name}", "*ELASTIC", f"{mats[name]['E']:.9g}, 0.", "*DENSITY", "1e-20"]
    for name, m in mats.items():
        L += [f"*MATERIAL, NAME=M_{name}", "*ELASTIC", f"{m['E']:.9g}, {m['nu']:.9g}", "*DENSITY", f"{m['rho']:.9g}"]
        if m.get("yield"):
            plastic = True
            if m.get("curve"):
                L += ["*PLASTIC"] + [f"{sg:.9g}, {ep:.9g}" for ep, sg in m["curve"]]
            else:
                L += ["*PLASTIC", f"{m['yield']:.9g}, 0."]
                H = m.get("hardening") or 0.0
                L.append(f"{m['yield'] + H * 0.5:.9g}, 0.5")
    notes = []
    for key in elsets:
        kind, g, extra = key
        sec = section_for(spec, g if kind != "SO" else "*")
        mname = "M_" + sec.get("material", "default")
        if kind == "B31":
            b = sec["beam"]
            if str(b.get("shape", "rect")).lower().startswith("circ"):
                r = float(b.get("r", b.get("d", 10.0) / 2 if "d" in b else 5.0))
                L += [f"*BEAM SECTION, ELSET={names[key]}, MATERIAL={mname}, SECTION=CIRC", f"{r:.9g}, {r:.9g}"]
            else:
                L += [f"*BEAM SECTION, ELSET={names[key]}, MATERIAL={mname}, SECTION=RECT", f"{float(b['b']):.9g}, {float(b['h']):.9g}"]
            L.append(", ".join(f"{x:.6g}" for x in extra))
        elif kind == "HB":
            a = extra[1]
            L += [f"*BEAM SECTION, ELSET={names[key]}, MATERIAL=MH_{sec.get('material', 'default')}, SECTION=RECT", f"{a:.9g}, {a:.9g}", ", ".join(f"{x:.6g}" for x in extra[2])]
        elif kind == "SH":
            t = float(sec["thickness"])
            if extra:
                mat = spec["materials"].get(sec.get("material", "default"), spec["materials"]["default"])
                kap = hinge_kappa(spec, extra, base, g)
                w = 2.0 * float(np.mean(hinge_w[extra]))
                th = (12 * (1 - mat["nu"] ** 2) * kap * w / mat["E"]) ** (1 / 3)
                notes.append(f"힌지 {extra}: kappa={kap:.5g} N·mm/rad/mm, 띠 폭 {w:.4g} mm, 띠 두께 {th:.4g} mm (패널 두께 {t:.4g})")
                t = min(t, th)
            L += [f"*SHELL SECTION, ELSET={names[key]}, MATERIAL={mname}", f"{t:.9g}"]
        else:
            L.append(f"*SOLID SECTION, ELSET={names[key]}, MATERIAL={mname}")
    for name, (kap, cnt, aa) in spring_notes.items():
        notes.append(f"힌지 {name}: kappa={kap:.5g} N·mm/rad/mm, 접힘선 절점 분리 + 회전 스프링 보 {cnt}개 (단면 {min(aa):.3g}~{max(aa):.3g} mm)")
    if rigid:
        notes.append(f"힌지 {', '.join(rigid)}: 강성이 요소 폭 기준 패널 휨강성보다 커서 강접으로 처리했습니다 (분할수를 늘리면 스프링으로 반영됩니다)")
    if pin_names:
        notes.append(f"핀 조인트 {', '.join(pin_names)}: 접힘선 절점을 분리하고 병진 자유도만 묶었습니다 (회전 자유, 절점 쌍 {len(ties)}개)")
    if cx:
        L += cx["model"]
        notes += cx["notes"]
    SYM = ("xsymm", "ysymm", "zsymm", "xasymm", "yasymm", "zasymm")
    if any(b.get("type") in SYM for b in spec["bcs"]):
        sp2 = dict(spec)
        sp2["bcs"] = [{**b, "dofs": [d for d in b["dofs"] if d <= 3]} if b.get("type") in SYM else b for b in spec["bcs"]]
        table = bc_table(model, sp2)
        notes.append("대칭·반대칭 조건은 CalculiX 에서 병진 구속만 걸었습니다 (쉘·보는 두께 방향으로 펼쳐 풀기 때문에 대칭면의 병진 구속이 회전도 막습니다. 회전까지 걸면 국부적으로 뻣뻣해집니다)")
    else:
        table = bc_table(model, spec)
    slaves = {b for _, b in ties}
    eq = [(b, a) for a, b in ties]
    for k0, k1 in tie_pairs:
        if k0 in mids and k1 in mids and mids[k0] != mids[k1]:
            eq.append((mids[k1], mids[k0]))
            slaves.add(mids[k1])
    master_of = {b: a for b, a in eq}
    for (n, d) in list(table):
        if n in slaves and d <= 3:
            v = table.pop((n, d))
            table.setdefault((master_of[n], d), v)
    if eq:
        L.append("*EQUATION")
        for b, a in eq:
            for d in (1, 2, 3):
                L += ["2", f"{b + 1}, {d}, 1., {a + 1}, {d}, -1."]

    def nodal(a):
        F = {}

        def add(n, v, off=0):
            for k in range(3):
                if v[k] != 0.0:
                    F[(n, k + 1 + off)] = F.get((n, k + 1 + off), 0.0) + float(v[k])

        for n, v in a["node"]:
            add(n, v)
        for n, v in a["moment"]:
            add(n, v, 3)
        for (i, j), v in a["edge"]:
            k = (min(i, j), max(i, j))
            if k in mids:
                add(i, v / 6)
                add(j, v / 6)
                add(mids[k], v * 2 / 3)
            else:
                add(i, v / 2)
                add(j, v / 2)
        for f, v in a["face"]:
            n = len(f)
            if n == 4:
                for i in f:
                    add(i, -v / 12)
                for i in range(4):
                    add(mid(f[i], f[(i + 1) % 4]), v / 3)
            else:
                for i in range(3):
                    add(mid(f[i], f[(i + 1) % 3]), v / 3)
        for (n, d) in list(F):
            if n in slaves and d <= 3:
                F[(master_of[n], d)] = F.get((master_of[n], d), 0.0) + F.pop((n, d))
        return F

    if ref is not None:
        Fc = {k: v / ctl["N"] for k, v in nodal(assemble(model, ctl["one"])).items() if k not in table and k[1] <= 3}
        if not Fc:
            raise ValueError("변위 제어 하중이 걸린 자유도가 모두 구속되어 있습니다")
        dep = max(Fc, key=lambda k: abs(Fc[k]))
        terms = [(dep[0], dep[1], Fc[dep])] + [(n, d, v) for (n, d), v in Fc.items() if (n, d) != dep] + [(ref, 1, -1.0)]
        L += ["*EQUATION", f"{len(terms)}"]
        for i in range(0, len(terms), 4):
            L.append(", ".join(f"{n + 1}, {d}, {v:.12g}" for n, d, v in terms[i : i + 4]))
        L += ["*BOUNDARY", f"{ref + 1}, 2, 3, 0.", f"{ref + 2}, 2, 3, 0."]
    solid_only = len(model.solids) > 0 and not model.beams and not model.shells
    zero = [(n, d) for (n, d), v in table.items() if v == 0.0 and not (solid_only and d > 3)]
    moving = [(n, d, v) for (n, d), v in table.items() if v != 0.0 and not (solid_only and d > 3)]
    for cp, ext, r1, r2 in cpn:
        ex = set(ext)
        moved = sorted({d for n, d in zero if n in ex} | {d for n, d, _ in moving if n in ex})
        if moved:
            zero = [(n, d) for n, d in zero if n not in ex]
            moving = [(n, d, v) for n, d, v in moving if n not in ex]
            fx = cp.get("fix") or {}
            for d in moved:
                if str(d) not in fx:
                    zero.append((r1, d) if d <= 3 else (r2, d - 3))
            notes.append(f"{cp['id']}: 기준점에 묶인 절점에 걸린 구속(자유도 {', '.join(str(d) for d in moved)})은 CalculiX 에서 기준점 구속으로 옮겼습니다")
        for dd, v in (cp.get("fix") or {}).items():
            j = int(dd)
            nd, k = (r1, j) if j <= 3 else (r2, j - 3)
            if float(v) == 0.0:
                zero.append((nd, k))
            else:
                moving.append((nd, k, float(v)))
    if model.solids is not None and len(model.solids):
        extra_z = []
        for (a, b), mnode in mids.items():
            for d in (1, 2, 3):
                if (a, d) in table and (b, d) in table:
                    if table[(a, d)] == 0.0 and table[(b, d)] == 0.0:
                        extra_z.append((mnode, d))
                    else:
                        moving.append((mnode, d, 0.5 * (table[(a, d)] + table[(b, d)])))
        zero += extra_z
    else:
        for (a, b), mnode in mids.items():
            for d in range(1, 7):
                if (a, d) in table and (b, d) in table:
                    va, vb = table[(a, d)], table[(b, d)]
                    if mnode in slaves and d <= 3:
                        continue
                    if va == 0.0 and vb == 0.0:
                        zero.append((mnode, d))
                    else:
                        moving.append((mnode, d, 0.5 * (va + vb)))
    if zero:
        L.append("*BOUNDARY")
        L += [f"{n + 1}, {d}, {d}, 0." for n, d in zero]
    kind = st["type"]
    nl = bool(st.get("nlgeom")) or plastic
    inc = max(1, int(st.get("increments", 25)))
    if kind == "buckle":
        L += ["*STEP", "*BUCKLE", f"{int(st.get('modes', 5))}"]
    elif kind == "modal":
        L += ["*STEP", "*FREQUENCY", f"{int(st.get('modes', 5))}"]
    elif nl or (moving or ref is not None) and inc > 1 and st.get("nlgeom") or cx and cx["pairs"]:
        L += ["*STEP, NLGEOM, INC=100000" if st.get("nlgeom") else "*STEP, INC=100000", "*STATIC", f"{1.0 / inc:.6g}, 1., 1e-8, {1.0 / inc:.6g}"]
    else:
        L += ["*STEP", "*STATIC"]
    if moving and kind != "modal":
        L.append("*BOUNDARY")
        L += [f"{n + 1}, {d}, {d}, {v:.9g}" for n, d, v in moving]
    if ref is not None:
        L += ["*BOUNDARY", f"{ref + 2}, 1, 1, {ctl['amp'] if kind == 'static' else 0.0:.9g}"]
    if cx and kind == "static":
        L += cx["step"]
    meta_cload = {}
    if kind != "modal":
        a = assemble(model, spec)
        F = nodal(a)
        for cp, ext, r1, r2 in cpn:
            for k in range(3):
                if cp.get("force") and cp["force"][k]:
                    F[(r1, k + 1)] = F.get((r1, k + 1), 0.0) + float(cp["force"][k])
                if cp.get("moment") and cp["moment"][k]:
                    F[(r2, k + 1)] = F.get((r2, k + 1), 0.0) + float(cp["moment"][k])
        meta_cload = dict(F)
        if F:
            L.append("*CLOAD")
            L += [f"{n + 1}, {d}, {v:.9g}" for (n, d), v in F.items()]
        if a["gravity"] is not None:
            g = a["gravity"]
            mag = float(np.linalg.norm(g))
            if mag > 0:
                L += ["*DLOAD", f"EALL, GRAV, {mag:.9g}, {g[0] / mag:.9g}, {g[1] / mag:.9g}, {g[2] / mag:.9g}"]
    o2 = ", OUTPUT=2D" if out2d else ""
    if kind in ("buckle", "modal"):
        L += [f"*NODE FILE{o2}", "U"]
    else:
        L += [f"*NODE FILE{o2}", "U, RF", "*EL FILE", "S", "*EL PRINT, ELSET=EALL, TOTALS=ONLY", "ELSE"]
        if ref is not None:
            L += ["*NODE PRINT, NSET=NREF", "U", "*NODE PRINT, NSET=NREF2", "RF"]
        for cp, ext, r1, r2 in cpn:
            tag = cp["id"].upper()
            L += [f"*NODE PRINT, NSET={tag}R", "U", f"*NODE PRINT, NSET={tag}Q", "U", f"*NODE PRINT, NSET={tag}", "U, RF"]
        if cx:
            L += cx["print"]
    L.append("*END STEP")
    meta = {"nodes": np.array(nodes, dtype=float), "bc": sorted({n for n, _, _ in moving}) if moving else sorted({n for n, _ in zero}), "hinge_elems": [e for k, v in elsets.items() if k[0] == "HB" for e in v]}
    cb = next((x for x in spec["bcs"] if x["id"] == "ctl"), None)
    if cb is not None:
        cn = set(int(i) for i in cb["select"]["nodes"])
        cs = {master_of.get(n, n) for n in cn}
        for (i, j), mnode in mids.items():
            if i in cn and j in cn:
                cs.add(master_of.get(mnode, mnode))
        d0 = cb["dofs"][0]
        meta["ctl"] = sorted(n for n, d, _ in moving if d == d0 and n in cs)
    meta["ref"] = ref
    meta["couplings"] = [(cp["id"], r1, r2) for cp, ext, r1, r2 in cpn]
    meta["cload"] = meta_cload
    meta["contact"] = cx
    if cx:
        meta["hinge_elems"] += cx["hide"]
    if ref is not None:
        meta["hinge_elems"].append(ctl_eid)
    return "\n".join(L) + "\n", notes, meta


def run_ccx(inp, threads=None):
    exe = find_ccx()
    if not exe:
        raise RuntimeError("CalculiX(ccx) 실행 파일을 찾을 수 없습니다. 설치 후 PATH에 두거나 FEMWEB_CCX 환경변수로 경로를 지정하세요")
    d = tempfile.mkdtemp(prefix="femweb_ccx_")
    with open(os.path.join(d, "job.inp"), "w", encoding="utf-8") as f:
        f.write(inp)
    env = dict(os.environ)
    env["OMP_NUM_THREADS"] = str(threads or os.cpu_count() or 1)
    from .. import progress as _pg

    _pg.context("CalculiX 실행 중")
    logf = open(os.path.join(d, "job.log"), "w+", encoding="utf-8", errors="replace")
    p = subprocess.Popen([exe, "job"], cwd=d, env=env, stdout=logf, stderr=subprocess.STDOUT, text=True)
    sta = os.path.join(d, "job.sta")
    while p.poll() is None:
        time.sleep(1.0)
        row = _sta_last(sta)
        if row:
            _pg.stage(f"CalculiX 스텝 {row[0]} 증분 {row[1]} (시도 {row[2]}, 반복 {row[3]}), 스텝 시간 {row[5]:.4g}", force=True)
    logf.seek(0)
    out = logf.read()
    logf.close()
    frd = os.path.join(d, "job.frd")
    dat = os.path.join(d, "job.dat")
    err = [l.strip() for l in out.splitlines() if "*ERROR" in l]
    if not os.path.exists(frd) or os.path.getsize(frd) < 200 or err:
        tail = "\n".join(out.splitlines()[-25:])
        shutil.rmtree(d, ignore_errors=True)
        raise RuntimeError("CalculiX 해석 실패\n" + ("\n".join(err[:5]) + "\n" if err else "") + tail)
    with open(frd) as f:
        frd_text = f.read()
    dat_text = open(dat).read() if os.path.exists(dat) else ""
    shutil.rmtree(d, ignore_errors=True)
    return frd_text, dat_text, out


def _set_blocks(dat, head, tag):
    out = []
    for m in re.finditer(head + r" for set " + tag + r" and time\s+\S+\s*\n", dat):
        blk = {}
        for ln in dat[m.end() :].splitlines():
            t = ln.split()
            if not t:
                if blk:
                    break
                continue
            if len(t) < 4 or not t[0].isdigit():
                break
            blk[int(t[0])] = tuple(float(x) for x in t[1:4])
        out.append(blk)
    return out


def _sta_last(path):
    try:
        with open(path, encoding="utf-8", errors="replace") as f:
            lines = f.read().splitlines()
    except OSError:
        return None
    for ln in reversed(lines):
        t = ln.split()
        if len(t) >= 7 and t[0].isdigit() and t[1].isdigit():
            try:
                return int(t[0]), int(t[1]), t[2], int(t[3]), float(t[4]), float(t[5])
            except ValueError:
                continue
    return None


def parse_frd(text):
    nodes = {}
    elems = []
    blocks = []
    lines = text.split("\n")
    i = 0
    n = len(lines)
    cur_time = 0.0
    while i < n:
        l = lines[i]
        if l.startswith("    2C"):
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                nodes[int(s[3:13])] = (float(s[13:25]), float(s[25:37]), float(s[37:49]))
                i += 1
        elif l.startswith("    3C"):
            i += 1
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -1"):
                    enum = int(s[3:13])
                    typ = int(s[13:18])
                    ids = []
                    i += 1
                    while i < n and lines[i].startswith(" -2"):
                        t = lines[i][3:]
                        ids += [int(t[k : k + 10]) for k in range(0, len(t.rstrip()), 10)]
                        i += 1
                    elems.append((typ, ids, enum))
                else:
                    i += 1
        elif l.startswith("  100C"):
            try:
                cur_time = float(l[12:24])
            except ValueError:
                cur_time = 0.0
            i += 1
            name = lines[i][5:13].strip()
            ncomp = int(lines[i][13:18])
            i += 1
            while i < n and lines[i].startswith(" -5"):
                i += 1
            ids = []
            vals = []
            while i < n and not lines[i].startswith(" -3"):
                s = lines[i]
                if s.startswith(" -1"):
                    ids.append(int(s[3:13]))
                    row = []
                    for k in range(ncomp):
                        c = s[13 + 12 * k : 25 + 12 * k]
                        try:
                            row.append(float(c))
                        except ValueError:
                            row.append(0.0)
                    vals.append(row)
                i += 1
            blocks.append({"name": name, "time": cur_time, "ids": np.array(ids, dtype=np.int64), "values": np.array(vals, dtype=float)})
        i += 1
    return nodes, elems, blocks


HEX = [[0, 3, 2, 1], [4, 5, 6, 7], [0, 1, 5, 4], [1, 2, 6, 5], [2, 3, 7, 6], [3, 0, 4, 7]]
WED_T = [[0, 2, 1], [3, 4, 5]]
WED_Q = [[0, 1, 4, 3], [1, 2, 5, 4], [2, 0, 3, 5]]
TET = [[0, 2, 1], [0, 1, 3], [1, 2, 3], [2, 0, 3]]


def surface(elems):
    faces = {}
    lines = []
    for typ, ids in ((e[0], e[1]) for e in elems):
        if typ in (1, 4):
            fl = [[ids[k] for k in f] for f in HEX]
        elif typ in (2, 5):
            fl = [[ids[k] for k in f] for f in WED_T + WED_Q]
        elif typ in (3, 6):
            fl = [[ids[k] for k in f] for f in TET]
        elif typ in (7, 8):
            fl = [ids[:3], ids[:3][::-1]]
        elif typ in (9, 10):
            fl = [ids[:4], ids[:4][::-1]]
        elif typ in (11, 12):
            lines.append(ids[:2])
            continue
        else:
            continue
        for f in fl:
            k = tuple(sorted(f))
            if k in faces:
                faces[k] = None
            else:
                faces[k] = f
    tris = []
    fe = set()
    for f in faces.values():
        if f is None:
            continue
        tris.append([f[0], f[1], f[2]])
        if len(f) == 4:
            tris.append([f[0], f[2], f[3]])
        for k in range(len(f)):
            a, b = f[k], f[(k + 1) % len(f)]
            fe.add((a, b) if a < b else (b, a))
    surface.fedges = sorted(fe)
    return tris, lines


def _totals(dat):
    out = []
    for m in re.finditer(r"total internal energy[^\n]*\n\s*\n?\s*([-+0-9.Ee]+)", dat):
        out.append(float(m.group(1)))
    return out


def _ref_forces(dat):
    f = [float(m.group(1)) for m in re.finditer(r"forces \(fx,fy,fz\) for set NREF2 and time\s+\S+\s+\d+\s+(\S+)", dat)]
    q = [float(m.group(1)) for m in re.finditer(r"displacements \(vx,vy,vz\) for set NREF and time\s+\S+\s+\d+\s+(\S+)", dat)]
    return f, q


def machine_stiffness(model, base, spec, ctl):
    sp = dict(spec)
    sp["loads"] = ctl["one"]["loads"]
    sp["bcs"] = [dict(b, value=0.0) for b in spec["bcs"]]
    k = None
    try:
        inp, _, meta = build_inp(model, base, sp, step={"type": "static", "nlgeom": False, "increments": 1}, out2d=True)
        frd, _, _ = run_ccx(inp)
        _, _, blocks = parse_frd(frd)
        b = [x for x in blocks if x["name"].startswith("DISP")][-1]
        U = np.zeros((len(meta["nodes"]) + 1, 3))
        ok = b["ids"] <= len(meta["nodes"])
        U[b["ids"][ok]] = b["values"][ok][:, :3]
        q = sum(v * U[n + 1, d - 1] for (n, d), v in meta["cload"].items() if d <= 3) / ctl["N"]
        if np.isfinite(q) and q > 0:
            k = ctl["N"] / q
    except Exception:
        k = None
    if k is None:
        k = ctl["N"] / max(ctl["amp"], 1e-12)
    return 1000.0 * k


def _factors(dat):
    out = []
    m = re.search(r"B U C K L I N G   F A C T O R   O U T P U T(.*?)(?:\n\s*\n\s*\n|\Z)", dat, re.S)
    if m:
        for l in m.group(1).splitlines():
            p = l.split()
            if len(p) == 2 and p[0].isdigit():
                out.append(float(p[1]))
    return out


def _freqs(dat):
    out = []
    m = re.search(r"E I G E N V A L U E   O U T P U T(.*?)(?:P A R T I C I P|\Z)", dat, re.S)
    if m:
        for l in m.group(1).splitlines():
            p = l.split()
            if len(p) >= 4 and p[0].isdigit():
                out.append(float(p[3]))
    return out


def buckle_mode(model, base, spec):
    sp = dict(spec)
    sp["step"] = dict(spec["step"])
    sp["step"].pop("control", None)
    if not sp["loads"]:
        return None
    try:
        inp, _, _ = build_inp(model, base, sp, step={"type": "buckle", "modes": 1}, out2d=True)
        frd, dat, _ = run_ccx(inp)
        _, _, blocks = parse_frd(frd)
    except Exception:
        return None
    disp = [b for b in blocks if b["name"].startswith("DISP")]
    if not disp:
        return None
    b = disp[-1]
    mode = np.zeros((len(model.nodes), 3))
    ok = b["ids"] <= len(model.nodes)
    mode[b["ids"][ok] - 1] = b["values"][ok][:, :3]
    mx = np.abs(mode).max()
    if not np.isfinite(mx) or mx <= 0:
        return None
    return mode / mx


def solve(model, base, spec, progress=None):
    from ..control import disp_control

    from ..spec import solid_rotations

    spec, rnotes = solid_rotations(model, spec)
    orig = spec
    spec, ctl = disp_control(model, spec)
    st = spec["step"]
    kind = "static" if st["type"] == "arclength" else st["type"]
    notes = list(rnotes)
    perturb = None
    step = {"type": kind}
    if ctl:
        notes += ctl["notes"]
        if not st.get("nlgeom"):
            notes.append("기하비선형이 꺼져 있어 좌굴 이후 거동은 나오지 않습니다 (선형 변위 제어)")
    if st["type"] == "arclength":
        step["nlgeom"] = True
        if not ctl:
            notes.append("CalculiX에는 호장법이 없어 하중 제어 비선형 정적 해석으로 풀었습니다. 극한점 이후 거동은 변위 제어를 켜야 따라갑니다")
    nlg = bool(st.get("nlgeom")) or st["type"] == "arclength"
    if ctl and ctl["kind"] == "average":
        ctl["k"] = machine_stiffness(model, base, spec, ctl)
    ref_load = 0.0
    if kind == "buckle":
        from ..loads import lumped

        F = lumped(model, spec)[0]
        net = float(np.linalg.norm(F.sum(axis=0)))
        ref_load = net if net > 1e-9 else float(np.linalg.norm(F, axis=1).sum()) / 2
        if ref_load <= 1e-9:
            raise ValueError("좌굴 해석에는 기준이 되는 압축 하중이 필요합니다. 하중(힘 또는 압력)을 먼저 추가하세요")
    if kind == "static" and nlg and float(st.get("imperfection") or 0) > 0:
        mode = buckle_mode(model, base, orig)
        if mode is not None:
            perturb = mode * float(st["imperfection"])
            notes.append(f"초기 결함: 1차 좌굴 모드 x {float(st['imperfection']):g} mm")
        else:
            notes.append("초기 결함용 좌굴 모드를 구하지 못해 결함 없이 풀었습니다")
    inp, n2, meta = build_inp(model, base, spec, step=step, perturb=perturb, ctl=ctl)
    notes += n2
    try:
        frd, dat, out = run_ccx(inp)
    except RuntimeError as e:
        if "increment size smaller" in str(e) and kind == "static" and not ctl:
            raise RuntimeError("하중 제어 비선형 해석이 극한점(좌굴·스냅스루) 부근에서 더 나아가지 못했습니다. '변위제어'를 켜면 그 이후 거동까지 따라갑니다\n" + str(e))
        raise
    nodes, elems, blocks = parse_frd(frd)
    hide = set(meta.get("hinge_elems", []))
    elems = [e for e in elems if e[2] not in hide]
    tris, lines = surface(elems)
    fedges = surface.fedges
    used = sorted({i for t in tris for i in t} | {i for l in lines for i in l})
    lut = {nid: k for k, nid in enumerate(used)}
    res = Result("calculix", kind)
    res.nodes = np.array([nodes[i] for i in used], dtype=float)
    res.tris = np.array([[lut[i] for i in t] for t in tris], dtype=int).reshape(-1, 3)
    res.lines = np.array([[lut[i] for i in l] for l in lines], dtype=int).reshape(-1, 2)
    res.fedges = np.array([[lut[a], lut[b]] for a, b in fedges], dtype=int).reshape(-1, 2)
    maxid = max(nodes) + 1
    pos = np.full(maxid + 1, -1, dtype=np.int64)
    pos[np.array(used)] = np.arange(len(used))

    def take(b, ncomp):
        arr = np.zeros((len(used), ncomp))
        ok = b["ids"] <= maxid
        p = pos[b["ids"][ok]]
        keep = p >= 0
        arr[p[keep]] = b["values"][ok][keep][:, :ncomp]
        return arr

    disp = [b for b in blocks if b["name"].startswith("DISP")]
    stress = [b for b in blocks if b["name"].startswith("STRESS")]
    forc = [b for b in blocks if b["name"].startswith("FORC")]
    sup = None
    if forc and meta["bc"]:
        from scipy.spatial import cKDTree

        near = cKDTree(meta["nodes"]).query(np.array([nodes[i] for i in forc[0]["ids"]]))[1]
        bcpos = meta["nodes"][np.array(meta["bc"])]
        span = float(np.linalg.norm(meta["nodes"].max(axis=0) - meta["nodes"].min(axis=0))) or 1.0
        sup = cKDTree(bcpos).query(meta["nodes"][near])[0] <= 1e-9 * span
    csel = None
    if ctl and ctl["kind"] == "rigid" and forc and meta.get("ctl"):
        from scipy.spatial import cKDTree

        near = cKDTree(meta["nodes"]).query(np.array([nodes[i] for i in forc[0]["ids"]]))[1]
        span = float(np.linalg.norm(meta["nodes"].max(axis=0) - meta["nodes"].min(axis=0))) or 1.0
        csel = cKDTree(meta["nodes"][np.array(meta["ctl"])]).query(meta["nodes"][near])[0] <= 1e-9 * span
    cforce = []
    factors = _factors(dat) if kind == "buckle" else []
    freqs = _freqs(dat) if kind == "modal" else []
    energy = _totals(dat)
    nmode = len(factors) if kind == "buckle" else len(freqs) if kind == "modal" else 0
    if nmode and len(disp) > nmode:
        disp = disp[-nmode:]
    times = []
    reac = []
    umax = []
    smax = []
    for k, b in enumerate(disp):
        U = take(b, 3)
        fields = {}
        if kind == "buckle":
            label = f"모드 {k + 1} (계수 {factors[k]:.5g})" if k < len(factors) else f"모드 {k + 1}"
            t = factors[k] if k < len(factors) else b["time"]
        elif kind == "modal":
            label = f"모드 {k + 1} ({freqs[k]:.5g} Hz)" if k < len(freqs) else f"모드 {k + 1}"
            t = freqs[k] if k < len(freqs) else b["time"]
        else:
            t = b["time"]
            label = f"하중계수 {t:.4g}"
            if k < len(stress):
                S = take(stress[k], 6)
                fields["vonMises"] = von_mises(S)
                for c, nm in enumerate(("Sxx", "Syy", "Szz", "Sxy", "Syz", "Szx")):
                    fields[nm] = S[:, c]
                smax.append(float(fields["vonMises"].max()))
            if k < len(forc) and sup is not None and len(forc[k]["ids"]) == len(sup):
                reac.append(forc[k]["values"][sup][:, :3].sum(axis=0).tolist())
            if k < len(forc) and csel is not None and len(forc[k]["ids"]) == len(csel):
                cforce.append(ctl["sign"] * float(forc[k]["values"][csel][:, ctl["axis"]].sum()))
        res.add_frame(label, t, U, fields)
        times.append(t)
        umax.append(float(np.linalg.norm(U, axis=1).max()))
    if not res.frames:
        raise RuntimeError("CalculiX 결과에 변위가 없습니다\n" + "\n".join(out.splitlines()[-20:]))
    if kind == "static":
        res.history["time"] = times
        res.history["u.max"] = umax
        if smax:
            res.history["vonMises.max"] = smax
        if len(energy) >= len(times):
            res.history["energy.total"] = energy[-len(times) :]
        if reac:
            r = np.array(reac)
            for c, nm in enumerate("xyz"):
                res.history[f"reaction.{nm}"] = r[:, c].tolist()
        res.summary = {"최대 변위 (mm)": umax[-1], "최대 von Mises (MPa)": smax[-1] if smax else None, "프레임 수": len(times)}
        if reac:
            res.summary["반력 합 (N)"] = [round(x, 4) for x in reac[-1]]
        if energy:
            res.summary["변형에너지 (N·mm)"] = energy[-1]
        for cid, r1, r2 in meta.get("couplings") or []:
            tag = cid.upper()
            vals = {}
            for suf, off in (("R", 0), ("Q", 3)):
                rows = re.findall(r"displacements \(vx,vy,vz\) for set " + tag + suf + r" and time\s+\S+\s*\n(?:\s*\n)?\s*\d+\s+(\S+)\s+(\S+)\s+(\S+)", dat)
                vals[("u", off)] = [[float(x) for x in r] for r in rows[-len(times) :]] if len(rows) >= len(times) else None
            ublk = _set_blocks(dat, "displacements \\(vx,vy,vz\\)", tag)
            fblk = _set_blocks(dat, "forces \\(fx,fy,fz\\)", tag)
            if len(fblk) >= len(times) and len(ublk) >= len(times) and vals.get(("u", 0)):
                fs, ms = [], []
                refp = np.asarray(next(c["ref"] for c in spec["couplings"] if c["id"] == cid), float)
                for k in range(len(times)):
                    fb, ub = fblk[-len(times) + k], ublk[-len(times) + k]
                    ids = np.array(sorted(fb))
                    fv = np.array([fb[i] for i in ids])
                    pv = meta["nodes"][ids - 1] + np.array([ub.get(i, (0.0, 0.0, 0.0)) for i in ids])
                    rc = refp + np.asarray(vals[("u", 0)][k])
                    fs.append(fv.sum(0).tolist())
                    ms.append(np.cross(pv - rc, fv).sum(0).tolist())
                vals[("f", 0)] = fs
                vals[("f", 3)] = ms
            names = {("u", 0): ("ux", "uy", "uz"), ("u", 3): ("rx", "ry", "rz"), ("f", 0): ("fx", "fy", "fz"), ("f", 3): ("mx", "my", "mz")}
            for key, nm in names.items():
                if vals.get(key):
                    arr = np.array(vals[key])
                    for c, n3 in enumerate(nm):
                        res.history[f"rp.{cid}.{n3}"] = arr[:, c].tolist()
            for lab, key in (("변위 (mm)", ("u", 0)), ("회전 (rad)", ("u", 3)), ("힘 (N)", ("f", 0)), ("모멘트 (N·mm)", ("f", 3))):
                if vals.get(key):
                    res.summary[f"기준점 {cid} {lab}"] = [round(v, 6) for v in vals[key][-1]]
        cx = meta.get("contact")
        if cx and cx["pairs"]:
            from ..contact import contacts_of, record as crecord

            cf = {}
            for c in contacts_of(spec):
                if c["id"] in cx["refs"]:
                    tag = cx["pairs"][c["id"]][0].replace("SDEP", "NREF")
                    v = re.findall(r"total force \(fx,fy,fz\) for set " + tag + r" and time\s+\S+\s+(\S+)\s+(\S+)\s+(\S+)", dat)
                else:
                    sd, si = cx["pairs"][c["id"]]
                    v = re.findall(r"statistics for slave set " + sd + r", master set " + si + r" and time\s+\S+\s+total surface force[^\n]*\n\s+(\S+)\s+(\S+)\s+(\S+)", dat)
                if len(v) >= len(times):
                    cf[c["id"]] = [[float(x) for x in r] for r in v[-len(times) :]]
            crecord(res, spec, times, cf)
        if ctl:
            from ..control import record

            if ctl["kind"] == "average":
                cf, cq = _ref_forces(dat)
                if len(cf) >= len(times) and len(cq) >= len(times):
                    record(res, ctl, None, cf[-len(times) :], cq[-len(times) :])
            elif len(cforce) == len(times):
                record(res, ctl, times, cforce)
    elif kind == "buckle":
        res.history["mode"] = list(range(1, len(factors) + 1))
        res.history["buckling.factor"] = factors
        res.summary = {
            "좌굴 계수": [round(f, 5) for f in factors],
            "가한 기준 하중 (N)": round(ref_load, 4),
            "임계 하중 (N) = 계수 × 기준 하중": [round(f * ref_load, 2) for f in factors],
        }
        if any(f < 0 for f in factors):
            notes.append("음수 계수는 하중 방향이 반대(인장)일 때 좌굴한다는 뜻입니다. 압축 방향 하중인지 확인하세요")
        if factors and factors[0] < 1:
            notes.append("1차 계수가 1보다 작으면 가한 하중이 이미 좌굴 하중보다 큽니다")
    else:
        res.history["mode"] = list(range(1, len(freqs) + 1))
        res.history["frequency"] = freqs
        res.summary = {"고유진동수 (Hz)": [round(f, 5) for f in freqs]}
    if notes:
        res.summary["참고"] = notes
    res.log = out
    return res
