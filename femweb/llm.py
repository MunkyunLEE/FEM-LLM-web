import http.client
import json
import os
import re
import socket
import threading
import urllib.parse


class Cancelled(Exception):
    pass


ACTIVE = {"conn": None, "cancelled": False}
_alock = threading.Lock()


def cancel():
    with _alock:
        ACTIVE["cancelled"] = True
        c = ACTIVE["conn"]
    if c is not None:
        try:
            c.sock.shutdown(socket.SHUT_RDWR)
        except Exception:
            pass
        try:
            c.close()
        except Exception:
            pass
    return c is not None


def _post(url, body, headers, timeout):
    u = urllib.parse.urlsplit(url)
    cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
    c = cls(u.hostname, u.port, timeout=timeout)
    with _alock:
        if ACTIVE["cancelled"]:
            raise Cancelled()
        ACTIVE["conn"] = c
    try:
        path = u.path + ("?" + u.query if u.query else "")
        c.request("POST", path, body=json.dumps(body).encode("utf-8"), headers=headers)
        r = c.getresponse()
        data = r.read()
        if r.status >= 400:
            raise RuntimeError(f"HTTP {r.status}: {data[:200].decode('utf-8', 'replace')}")
        return json.loads(data)
    except Exception as e:
        if ACTIVE["cancelled"]:
            raise Cancelled() from e
        raise
    finally:
        with _alock:
            ACTIVE["conn"] = None
        try:
            c.close()
        except Exception:
            pass

SYSTEM = """You convert a structural engineer's request (Korean or English) into JSON edit operations for an FEM model spec.
Reply with ONE JSON object only: {"ops": [...], "reply": "<short Korean sentence describing what you did or what is unclear>"}.
Never invent node coordinates or group names. Use only groups listed in MODEL. Units are mm, N, tonne, s, MPa. Convert: kN->N x1000, m->mm x1000, kPa->MPa x0.001, GPa->MPa x1000, N*m->N*mm x1000.

SELECT grammar (choose target nodes):
 {"group": "<name from MODEL.groups>"}
 {"plane": {"axis": "x"|"y"|"z", "at": "min"|"max"|"mid"|<number>}}
 {"box": {"min": [x,y,z], "max": [x,y,z]}}
 {"nearest": [x,y,z]}   {"nodes": [ids]}   {"all": true}
 {"boundary": true} = free edges (perimeter) of shells   {"corners": true} = corner points
 {"and": [SELECT, ...]}  {"or": [SELECT, ...]}
 Words: bottom/바닥/하단 = z min, top/상단 = z max, unless the user names an axis.

OPS:
 {"op":"add_bc","name":str,"select":SELECT,"type":"fixed"|"pinned"|"xsymm"|"ysymm"|"zsymm"}  or with "dofs":[1..6] (1-3 translations x,y,z; 4-6 rotations) and optional "value" (prescribed displacement in mm, applied to each listed dof)
 {"op":"add_load","type":"force","name":str,"select":SELECT,"vector":[Fx,Fy,Fz]}   total force in N distributed over the selection
 {"op":"add_load","type":"moment","select":SELECT,"vector":[Mx,My,Mz]}   N*mm
 {"op":"add_load","type":"pressure","select":SELECT,"value":p,"direction":[dx,dy,dz]}   MPa, direction optional
 {"op":"add_load","type":"gravity","vector":[0,0,-9810]}
 {"op":"update","id":"bc1"|"ld1"|"pl1", ...fields to change}
 {"op":"add_contact","type":"rigid","shape":"sphere","radius":mm,"motion":[dx,dy,dz],"friction":mu,"center":[x,y,z]?}   rigid sphere pushed by motion (mm); omit center to place it touching the model on the side it comes from
 {"op":"add_contact","type":"rigid","shape":"plane","normal":[nx,ny,nz],"motion":[dx,dy,dz],"friction":mu}   rigid plane; normal points from the plane toward the body ([0,0,1] = floor under the model)
 {"op":"add_contact","type":"pair","a":<select>,"b":<select>,"friction":mu}   contact between two deformable regions (e.g. {"group":"body_1"})
 {"op":"remove","id":"..."}    {"op":"clear","what":"bcs"|"loads"|"plots"|"contacts"|"all"}
 {"op":"set_material","name":"default","E":MPa,"nu":..,"rho":tonne/mm3,"yield":MPa|null,"hardening":MPa,"curve":[[plastic_strain,true_stress_MPa],...]|null}   curve = multilinear hardening, first point at plastic strain 0; or "power":[k,m] for sigma = yield + k*p^m
 {"op":"set_section","group":"*"|"<group>","thickness":mm,"beam":{"shape":"rect","b":mm,"h":mm}|{"shape":"circ","r":mm}}
 {"op":"set_hinge","group":"*"|"<hinge group>","kappa":N*mm/rad/mm|null,"ratio":number,"pin":bool}   pin=true is a pin joint (free rotation)
 {"op":"auto_hinge","mode":"auto"|"groups"|"crease"|"all"|"off","angle":deg}   turn shell-to-shell shared edges into hinges (auto = group boundaries + creases)
 {"op":"set_step","type":"static"|"buckle"|"modal"|"arclength","nlgeom":bool,"increments":int,"modes":int,"imperfection":mm,"control":"load"|"displacement","control_disp":mm,"control_load":"ld1","control_kind":"average"|"rigid","branch":"auto"|"primary"}   control=displacement follows post-buckling/snap-through by controlling the displacement under the chosen load (average keeps the load distribution, rigid pushes the loaded nodes equally); imperfection adds the first buckling mode as initial shape error
 {"op":"set_solver","solver":"calculix"|"barhinge"|"tensormesh"|"shell"}   shell = own large-deformation shell solver (OBJ panels, static only); the only one with self-contact and damping
 {"op":"add_contact","type":"self","friction":mu}   self-contact between all shell faces (solver shell only)
 {"op":"set_step","damping":{"kind":"none"|"mass"|"stiffness"|"rayleigh","zeta":percent,"time":seconds},"amplitude":"linear"|"smooth"}   damping ratio at modes 1 and 2 (solver shell only); time 0 = automatic
 {"op":"set_divisions","divisions":int}
 {"op":"add_plot","title":str,"x":QUANTITY,"series":[QUANTITY,...]}
QUANTITY: {"quantity": "time"|"frame"|"energy.total"|"energy.bar"|"energy.bend"|"energy.fold"|"reaction.x"|"reaction.y"|"reaction.z"|"u.max"|"vonMises.max"|"fold.mean"|"buckling.factor"|"frequency"|"control.disp"|"control.force"|"contact.disp"|"contact.force"|"contact.normal"|"contact.friction"|"reaction.<bc id>.x|y|z"}   (the last three: solver shell only; nodal fields RF, CFORCE, CPRESS also exist there)
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
    "model": os.environ.get("FEMWEB_LLM_MODEL", "qwen3:8b"),
    "key": os.environ.get("FEMWEB_LLM_KEY", ""),
    "mode": os.environ.get("FEMWEB_LLM_MODE", "rule_first"),
}


def context(model, spec, result_meta=None):
    s = {k: spec[k] for k in ("solver", "divisions", "materials", "sections", "hinges", "bcs", "loads", "contacts", "step", "plots") if k in spec}
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
    with _alock:
        ACTIVE["cancelled"] = False
    msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": ctx}]
    for h in (history or [])[-6:]:
        msgs.append({"role": h["role"], "content": h["content"]})
    nothink = "qwen3" in str(cfg.get("model", "")).lower() and os.environ.get("FEMWEB_LLM_THINK", "0") != "1"
    msgs.append({"role": "user", "content": text + (" /no_think" if nothink else "")})
    if error:
        msgs.append({"role": "user", "content": f"The previous ops failed validation: {error}\nReturn corrected JSON."})
    headers = {"Content-Type": "application/json"}
    if cfg.get("key"):
        headers["Authorization"] = "Bearer " + cfg["key"]
    body = {"model": cfg["model"], "messages": msgs, "temperature": 0, "stream": False, "response_format": {"type": "json_object"}}
    resp = _post(cfg["url"].rstrip("/") + "/chat/completions", body, headers, 900)
    data = _extract(resp["choices"][0]["message"]["content"])
    ops = data.get("ops", [])
    if not isinstance(ops, list):
        raise ValueError("ops 가 배열이 아닙니다")
    return ops, str(data.get("reply", ""))


def provider(url):
    u = urllib.parse.urlsplit(url or "")
    h = (u.hostname or "").lower()
    if h in ("localhost", "127.0.0.1", "::1", "0.0.0.0"):
        if u.port == 11434:
            return "Ollama (이 컴퓨터)"
        if u.port == 1234:
            return "LM Studio (이 컴퓨터)"
        return "로컬 서버 (이 컴퓨터)"
    names = {
        "generativelanguage.googleapis.com": "Google Gemini API (클라우드)",
        "api.openai.com": "OpenAI API (클라우드)",
        "api.anthropic.com": "Anthropic API (클라우드)",
        "openrouter.ai": "OpenRouter (클라우드)",
    }
    return names.get(h, f"외부 서버 {h}")


def _norm(name):
    return str(name).split("/", 1)[1] if str(name).startswith("models/") else str(name)


def llm_status(cfg):
    import time

    import httpx

    u = urllib.parse.urlsplit(cfg["url"])
    base = {"provider": provider(cfg["url"]), "host": (u.hostname or "") + (f":{u.port}" if u.port else ""), "key_set": bool(cfg.get("key"))}
    try:
        t = time.time()
        r = httpx.get(cfg["url"].rstrip("/") + "/models", timeout=4, headers={"Authorization": "Bearer " + cfg["key"]} if cfg.get("key") else {})
        ms = round((time.time() - t) * 1000)
        r.raise_for_status()
        names = [_norm(m.get("id")) for m in r.json().get("data", [])]
        return {**base, "ok": True, "ms": ms, "models": names, "has_model": _norm(cfg["model"]) in names}
    except Exception as e:
        return {**base, "ok": False, "error": str(e)[:200], "models": []}


def test_llm(cfg):
    import time

    nothink = "qwen3" in str(cfg.get("model", "")).lower() and os.environ.get("FEMWEB_LLM_THINK", "0") != "1"
    msgs = [
        {"role": "system", "content": 'Reply with ONE JSON object only: {"ops": [], "reply": "<short Korean greeting>"}'},
        {"role": "user", "content": "안녕" + (" /no_think" if nothink else "")},
    ]
    headers = {"Content-Type": "application/json"}
    if cfg.get("key"):
        headers["Authorization"] = "Bearer " + cfg["key"]
    body = {"model": cfg["model"], "messages": msgs, "temperature": 0, "stream": False, "response_format": {"type": "json_object"}}
    with _alock:
        ACTIVE["cancelled"] = False
    t = time.time()
    try:
        resp = _post(cfg["url"].rstrip("/") + "/chat/completions", body, headers, 300)
        text = resp["choices"][0]["message"]["content"]
        try:
            data = _extract(text)
            json_ok = True
            reply = str(data.get("reply", ""))
        except Exception:
            json_ok = False
            reply = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
        return {"ok": True, "ms": round((time.time() - t) * 1000), "model": resp.get("model"), "json_ok": json_ok, "reply": reply[:160]}
    except Cancelled:
        return {"ok": False, "error": "중단했습니다"}
    except Exception as e:
        return {"ok": False, "error": str(e)[:300]}


def rule_parse(text, model, spec):
    from . import rules

    ops, reply, _ = rules.parse(text, model, spec)
    return ops, reply
