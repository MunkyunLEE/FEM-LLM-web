import json
import multiprocessing
import os
import signal
import threading
import time
import traceback

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from . import llm, plots, rules
from .hinges import auto_key, with_auto
from .io_obj import parse_obj, refine
from .io_stl import check_stl, stl_to_solid
from .loads import preview
from .solvers import calculix
from .spec import apply_ops, default_spec, warnings_for

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WEB = os.path.join(ROOT, "web")
EXAMPLES = os.path.join(ROOT, "examples")

app = FastAPI(title="FEM-LLM-web")
lock = threading.Lock()
S = {"base": None, "spec": None, "result": None, "stale": False, "chat": [], "llm": dict(llm.DEFAULTS), "cache": {}, "solve_time": None, "undo": [], "eff": (None, None), "refs": []}


def fail(msg, code=400):
    raise HTTPException(status_code=code, detail=str(msg))


def need_model():
    if S["base"] is None:
        fail("먼저 지오메트리를 불러오세요")


def eff_base():
    key = auto_key(S["spec"])
    if S["eff"][0] != key:
        S["eff"] = (key, with_auto(S["base"], S["spec"]))
        S["cache"] = {}
    return S["eff"][1]


def active():
    base, spec = eff_base(), S["spec"]
    if base.kind != "obj" or spec["solver"] == "barhinge":
        return base
    k = int(spec["divisions"])
    if k not in S["cache"]:
        S["cache"] = {k: refine(base, k)}
    return S["cache"][k]


def has_tm():
    import importlib.util

    return importlib.util.find_spec("tensormesh") is not None and importlib.util.find_spec("torch") is not None


def state(view=True):
    if S["base"] is None:
        return {"model": None, "tools": tools(), "llm": llm_public()}
    m = active()
    out = {
        "model": m.summary(),
        "base_counts": S["base"].summary()["counts"],
        "spec": S["spec"],
        "preview": preview(m, S["spec"]),
        "warnings": warnings_for(S["spec"], eff_base()),
        "hinge_info": hinge_info(),
        "can_undo": bool(S["undo"]),
        "result": S["result"].meta() if S["result"] else None,
        "stale": S["stale"],
        "solve_time": S["solve_time"],
        "tools": tools(),
        "llm": llm_public(),
    }
    if view:
        out["view"] = m.to_view()
    return out


def hinge_info():
    from .hinges import auto_cfg, is_pin

    b = eff_base()
    out = {"auto": auto_cfg(S["spec"]), "groups": []}
    for name, es in b.hinges.items():
        L = float(sum(((b.nodes[a] - b.nodes[c]) ** 2).sum() ** 0.5 for a, c in es))
        out["groups"].append({"name": name, "edges": len(es), "length": round(L, 4), "pin": is_pin(S["spec"], name)})
    return out


GPU = {}


def gpu():
    if "name" not in GPU:
        try:
            import torch

            GPU["name"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else ("Apple GPU" if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available() else None)
        except Exception:
            GPU["name"] = None
    return GPU["name"]


def tools():
    return {"ccx": calculix.find_ccx(), "tensormesh": has_tm(), "gpu": gpu()}


def llm_public():
    c = S["llm"]
    return {"url": c["url"], "model": c["model"], "mode": c["mode"], "has_key": bool(c.get("key"))}


def set_model(base, spec=None):
    S["base"] = base
    S["spec"] = spec or default_spec(base)
    S["undo"] = []
    S["eff"] = (None, None)
    S["result"] = None
    S["stale"] = False
    S["cache"] = {}
    S["chat"] = []
    S["solve_time"] = None
    S["refs"] = []


def load_file(name, data, size=None):
    low = name.lower()
    if low.endswith(".obj"):
        return parse_obj(data.decode("utf-8", "ignore"), name)
    if low.endswith(".stl"):
        chk = check_stl(data)
        if not chk["watertight"]:
            fail(f"STL이 닫힌 표면이 아닙니다 (열린 엣지 {chk['open_edges']}개, 비매니폴드 엣지 {chk['nonmanifold_edges']}개). Rhino에서 닫힌 폴리서프/메쉬로 내보내세요")
        return stl_to_solid(data, name, size=size)
    fail("지원 형식: .obj (보·쉘), .stl (솔리드)")


@app.get("/api/state")
def get_state():
    with lock:
        return state()


@app.get("/api/examples")
def examples():
    p = json.load(open(os.path.join(EXAMPLES, "presets.json"), encoding="utf-8"))
    return [{"name": k, "title": v["title"], "note": v["note"], "prompts": v["prompts"], "file": v["file"]} for k, v in p.items()]


@app.post("/api/examples/{name}")
def load_example(name: str, preset: int = 1):
    p = json.load(open(os.path.join(EXAMPLES, "presets.json"), encoding="utf-8"))
    if name not in p:
        fail("예제가 없습니다", 404)
    ex = p[name]
    with lock:
        try:
            data = open(os.path.join(EXAMPLES, ex["file"]), "rb").read()
            base = load_file(ex["file"], data, ex.get("size"))
            ops = ex["ops"] if preset else [o for o in ex["ops"] if o["op"].startswith("set_")]
            set_model(base)
            if ops:
                _apply(ops)
            S["undo"] = []
            return state()
        except HTTPException:
            raise
        except Exception as e:
            fail(e)


@app.post("/api/upload")
def upload(file: UploadFile = File(...), size: float = Form(0.0)):
    data = file.file.read()
    with lock:
        try:
            set_model(load_file(file.filename or "model", data, size or None))
            return state()
        except HTTPException:
            raise
        except Exception as e:
            fail(e)


class Ops(BaseModel):
    ops: list


GEO_OPS = ("set_solver", "set_divisions", "auto_hinge")


def _apply(ops):
    saved = S["spec"]
    before = (S["spec"]["solver"], S["spec"]["divisions"], auto_key(S["spec"]))
    geo = [o for o in ops if isinstance(o, dict) and o.get("op") in GEO_OPS]
    rest = [o for o in ops if o not in geo]
    spec, log = apply_ops(S["spec"], geo, None)
    S["spec"] = spec
    try:
        m = active()
        spec, log2 = apply_ops(S["spec"], rest, m)
    except Exception:
        S["spec"] = saved
        raise
    if (spec["solver"], spec["divisions"], auto_key(spec)) != before:
        S["result"] = None
    S["spec"] = spec
    S["undo"].append(saved)
    del S["undo"][:-30]
    if any(o.get("op") != "add_plot" and not (o.get("op") == "remove" and str(o.get("id", "")).startswith("pl")) for o in ops if isinstance(o, dict)):
        S["stale"] = S["result"] is not None
    return log + log2


@app.post("/api/ops")
def post_ops(body: Ops):
    with lock:
        need_model()
        try:
            log = _apply(body.ops)
        except Exception as e:
            fail(e)
        out = state(view=True)
        out["log"] = log
        return out


class Chat(BaseModel):
    text: str


def _try(ops, m):
    apply_ops(S["spec"], [o for o in ops if isinstance(o, dict) and o.get("op") in GEO_OPS], None)
    probe = m if not any(isinstance(o, dict) and o.get("op") in GEO_OPS for o in ops) else None
    apply_ops(S["spec"], ops, probe)


def _ask_llm(cfg, text, m):
    ctx = llm.context(m, S["spec"], S["result"].meta() if S["result"] else None)
    ops, reply = llm.ask_llm(cfg, text, ctx, S["chat"])
    try:
        _try(ops, m)
    except Exception as e:
        ops, reply = llm.ask_llm(cfg, text, ctx, S["chat"], error=str(e))
    return ops, reply


@app.post("/api/chat")
def chat(body: Chat):
    with lock:
        need_model()
        cfg = S["llm"]
        m = active()
        text = body.text.strip()
        if not text:
            fail("내용이 비어 있습니다")
        mode = cfg["mode"]
        engine = "rule"
        note = ""
        action = None
        rops, rreply, info = rules.parse(text, m, S["spec"])
        rule_ok = info["complete"]
        if rule_ok and rops:
            try:
                _try(rops, m)
            except Exception as e:
                rule_ok = False
                rreply += f" (적용 검사 실패: {e})"
        direct = rule_ok or (info["ask"] and not info["unknown"]) or info["answered"]
        use_llm = (mode in ("auto", "llm") or (mode == "rule_first" and not direct)) and llm.configured(cfg)
        if mode == "llm" and not llm.configured(cfg):
            fail("LLM 이 설정되지 않았습니다. LLM 칸에 엔드포인트와 모델 이름을 넣어 주세요")
        ops, reply = rops, rreply
        if use_llm:
            try:
                ops, reply = _ask_llm(cfg, text, m)
                engine = "llm"
            except llm.Cancelled:
                fail("LLM 요청을 중단했습니다", 409)
            except Exception as e:
                if mode == "llm":
                    fail(f"LLM 호출 실패: {e}")
                note = f"LLM에 연결하지 못해 규칙 기반 해석 결과를 사용했습니다 ({str(e)[:120]})"
                ops, reply = rops, rreply
        elif mode == "rule_first":
            note = "규칙으로 바로 처리했습니다" if rule_ok else ""
        error = None
        log = []
        if engine == "rule" and info["undo"]:
            if S["undo"]:
                prev = S["undo"].pop()
                changed = (prev["solver"], prev["divisions"], auto_key(prev)) != (S["spec"]["solver"], S["spec"]["divisions"], auto_key(S["spec"]))
                S["spec"] = prev
                if changed:
                    S["result"] = None
                else:
                    S["stale"] = S["result"] is not None
                reply = "직전 상태로 되돌렸습니다."
            else:
                reply = "되돌릴 내용이 없습니다."
            ops = []
        elif ops:
            try:
                log = _apply(ops)
            except Exception as e:
                error = str(e)
        if engine == "rule" and info["run"] and not error and not info["ask"] and not info["unknown"]:
            action = "solve"
        S["chat"].append({"role": "user", "content": text})
        S["chat"].append({"role": "assistant", "content": json.dumps({"ops": ops, "reply": reply}, ensure_ascii=False)})
        del S["chat"][:-12]
        out = state(view=True)
        out["chat"] = {"reply": reply, "ops": ops, "engine": engine, "model": cfg["model"] if engine == "llm" else None, "provider": llm.provider(cfg["url"]) if engine == "llm" else None, "note": note, "error": error, "log": log, "action": action}
        return out


@app.post("/api/undo")
def undo():
    with lock:
        need_model()
        if not S["undo"]:
            fail("되돌릴 내용이 없습니다")
        prev = S["undo"].pop()
        changed = (prev["solver"], prev["divisions"], auto_key(prev)) != (S["spec"]["solver"], S["spec"]["divisions"], auto_key(S["spec"]))
        S["spec"] = prev
        if changed:
            S["result"] = None
        else:
            S["stale"] = S["result"] is not None
        return state(view=True)


def _solve_with(m, base, spec, progress=None):
    if spec["solver"] == "calculix":
        return calculix.solve(m, base, spec)
    if spec["solver"] == "barhinge":
        if base.kind != "obj":
            raise ValueError("바-힌지 솔버는 OBJ 쉘 패널 모델에만 쓸 수 있습니다")
        from .solvers import barhinge

        return barhinge.solve(base, spec)
    if spec["solver"] == "shell":
        from .solvers import shell

        return shell.solve(m, spec, progress=progress)
    if spec["solver"] == "tensormesh":
        from .solvers import solid_tm

        return solid_tm.solve(m, spec)
    raise ValueError("알 수 없는 솔버")


def _worker(conn, m, base, spec):
    try:
        os.setpgrp()
    except Exception:
        pass
    t0 = time.time()
    n = [0]

    def prog(lam):
        n[0] += 1
        try:
            conn.send(("prog", {"lam": float(lam), "inc": n[0], "t": time.time() - t0}))
        except Exception:
            pass

    from . import progress as _pg

    _pg.set_hook(lambda txt: conn.send(("stage", txt)))

    try:
        conn.send(("ok", _solve_with(m, base, spec, prog)))
    except BaseException as e:
        conn.send(("err", str(e), traceback.format_exc()))
    finally:
        conn.close()


JOB = {"proc": None, "cancelled": False, "t0": None, "prog": None, "info": None}


def _kill(proc):
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except Exception:
        pass
    try:
        proc.kill()
    except Exception:
        pass


def run_solver():
    ctx = multiprocessing.get_context("spawn")
    parent, child = ctx.Pipe(duplex=False)
    proc = ctx.Process(target=_worker, args=(child, active(), eff_base(), S["spec"]), daemon=True)
    JOB["cancelled"] = False
    JOB["t0"] = time.time()
    JOB["prog"] = None
    JOB["stage"] = None
    sp = S["spec"]
    mv = [abs(float(v)) for c in sp.get("contacts") or [] if c.get("type") == "rigid" for v in c.get("motion", [])]
    JOB["info"] = {"solver": sp["solver"], "increments": int(sp["step"].get("increments") or 0), "travel": max(mv) if mv else (float(sp["step"].get("control_disp") or 0.0) if sp["step"].get("control") == "displacement" else 0.0)}
    proc.start()
    JOB["proc"] = proc
    child.close()
    try:
        while True:
            if parent.poll(0.2):
                try:
                    msg = parent.recv()
                except EOFError:
                    msg = None
                if msg is not None and msg[0] == "prog":
                    JOB["prog"] = msg[1]
                    continue
                if msg is not None and msg[0] == "stage":
                    JOB["stage"] = msg[1]
                    continue
                break
            if not proc.is_alive():
                msg = parent.recv() if parent.poll(0.5) else None
                break
        if JOB["cancelled"]:
            raise RuntimeError("해석을 중단했습니다")
        if msg is None:
            raise RuntimeError("해석 프로세스가 결과 없이 종료되었습니다")
        if msg[0] == "err":
            print(msg[2])
            raise RuntimeError(msg[1])
        return msg[1]
    finally:
        _kill(proc)
        proc.join(timeout=5)
        JOB["proc"] = None
        JOB["t0"] = None
        parent.close()


@app.post("/api/solve")
def solve():
    with lock:
        need_model()
        t = time.time()
        try:
            S["result"] = run_solver()
        except Exception as e:
            if JOB["cancelled"]:
                fail("해석을 중단했습니다", 409)
            fail(e, 422)
        S["stale"] = False
        S["solve_time"] = round(time.time() - t, 2)
        return state(view=False)


@app.get("/api/progress")
def progress():
    t0 = JOB["t0"]
    if t0 is None:
        return {"running": False}
    el = time.time() - t0
    p = JOB["prog"] or {}
    info = JOB["info"] or {}
    lam = float(p.get("lam", 0.0))
    out = {"running": True, "elapsed": round(el, 1), "solver": info.get("solver"), "lam": lam if p else None, "inc": p.get("inc"), "increments": info.get("increments"), "stage": JOB.get("stage")}
    if p and lam > 0.01:
        rate = lam / max(p["t"], 1e-9)
        out["eta"] = round(max(0.0, (1.0 - lam) / rate - (el - p["t"])), 0)
    if p and info.get("travel"):
        out["disp"] = round(lam * info["travel"], 3)
    return out


@app.post("/api/cancel/solve")
def cancel_solve():
    p = JOB["proc"]
    if p is None:
        return {"cancelled": False}
    JOB["cancelled"] = True
    _kill(p)
    return {"cancelled": True}


@app.post("/api/cancel/llm")
def cancel_llm():
    return {"cancelled": llm.cancel()}


@app.get("/api/result/mesh")
def result_mesh():
    with lock:
        if not S["result"]:
            fail("결과가 없습니다", 404)
        return S["result"].mesh()


@app.get("/api/result/frame")
def result_frame(i: int = -1, field: str = "U"):
    with lock:
        r = S["result"]
        if not r:
            fail("결과가 없습니다", 404)
        i = i if i >= 0 else len(r.frames) - 1
        if i >= len(r.frames):
            fail("프레임 범위 밖", 404)
        return r.frame(i, field)


class GifReq(BaseModel):
    frames: list[str]
    fps: float = 8.0


@app.post("/api/export/gif")
def export_gif(req: GifReq):
    import base64
    import io

    from fastapi import Response

    try:
        from PIL import Image
    except Exception:
        fail("GIF 저장에는 Pillow 가 필요합니다 (pip install pillow). 영상 저장은 그대로 쓸 수 있습니다", 422)
    if not req.frames:
        fail("프레임이 없습니다", 422)
    ims = []
    for d in req.frames[:400]:
        raw = base64.b64decode(d.split(",", 1)[-1])
        ims.append(Image.open(io.BytesIO(raw)).convert("RGB"))
    pick = ims[:: max(1, len(ims) // 6)] + [ims[-1]]
    w, h = ims[0].size
    sheet = Image.new("RGB", (w * len(pick), h))
    for k, im in enumerate(pick):
        sheet.paste(im.resize((w, h)), (k * w, 0))
    pal = sheet.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
    out = [im.quantize(palette=pal, dither=Image.Dither.FLOYDSTEINBERG) for im in ims]
    buf = io.BytesIO()
    dur = int(1000 / max(1.0, min(30.0, req.fps)))
    out[0].save(buf, format="GIF", save_all=True, append_images=out[1:], duration=[dur] * (len(out) - 1) + [dur * 6], loop=0, optimize=False)
    return Response(content=buf.getvalue(), media_type="image/gif")


@app.get("/api/plots")
def get_plots():
    with lock:
        need_model()
        out = []
        if not S["result"]:
            return out
        from . import compare

        for p in S["spec"]["plots"]:
            try:
                q = plots.evaluate(p, S["result"], active())
                q["refs"] = []
                for r in S["refs"]:
                    if r["plot"] != p["id"]:
                        continue
                    k = min(int(r.get("series", 0)), len(q["series"]) - 1)
                    xr = [v * r["sx"] for v in r["x"]]
                    yr = [v * r["sy"] for v in r["y"]]
                    met = compare.metrics(q["series"][k]["x"], q["series"][k]["y"], xr, yr) if k >= 0 else {"overlap": None}
                    q["refs"].append({"id": r["id"], "name": r["name"], "x": xr, "y": yr, "series": k, "against": q["series"][k]["name"] if k >= 0 else "", "sx": r["sx"], "sy": r["sy"], "flipped": r.get("flipped", False), "metrics": met})
                out.append(q)
            except Exception as e:
                out.append({"id": p["id"], "title": p.get("title", "그래프"), "error": str(e), "series": [], "refs": []})
        return out


class RefIn(BaseModel):
    plot: str
    text: str
    name: str = "Abaqus"
    series: int = 0


class RefUpd(BaseModel):
    id: str
    sx: float | None = None
    sy: float | None = None
    series: int | None = None
    remove: bool = False


@app.post("/api/refs")
def add_ref(body: RefIn):
    with lock:
        need_model()
        from . import compare

        if body.plot not in [p["id"] for p in S["spec"]["plots"]]:
            fail("그래프가 없습니다", 404)
        try:
            curves = compare.parse(body.text, body.name)
        except Exception as e:
            fail(e)
        own = None
        if S["result"]:
            try:
                q = plots.evaluate(next(p for p in S["spec"]["plots"] if p["id"] == body.plot), S["result"], active())
                own = q["series"][min(body.series, len(q["series"]) - 1)]["y"] if q["series"] else None
            except Exception:
                own = None
        made = []
        for c in curves:
            sy = compare.auto_sign(own, c["y"]) if own else 1.0
            n = 1 + max([int(r["id"][3:]) for r in S["refs"]] or [0])
            r = {"id": f"ref{n}", "plot": body.plot, "name": c["name"], "x": c["x"], "y": c["y"], "sx": 1.0, "sy": sy, "series": body.series, "flipped": sy < 0}
            S["refs"].append(r)
            made.append({"id": r["id"], "name": r["name"], "points": len(r["x"]), "flipped": r["flipped"]})
        return made


@app.post("/api/refs/update")
def update_ref(body: RefUpd):
    with lock:
        r = next((x for x in S["refs"] if x["id"] == body.id), None)
        if r is None:
            fail("비교 곡선이 없습니다", 404)
        if body.remove:
            S["refs"].remove(r)
            return {"ok": True}
        for k in ("sx", "sy", "series"):
            v = getattr(body, k)
            if v is not None:
                r[k] = v
        return {"ok": True}


class LLMCfg(BaseModel):
    url: str | None = None
    model: str | None = None
    key: str | None = None
    mode: str | None = None


@app.post("/api/llm")
def set_llm(body: LLMCfg):
    with lock:
        for k in ("url", "model", "key", "mode"):
            v = getattr(body, k)
            if v is not None:
                S["llm"][k] = v.strip()
        if S["llm"]["mode"] not in ("rule_first", "auto", "llm", "rule"):
            S["llm"]["mode"] = "rule_first"
        return {"llm": llm_public(), "status": llm.llm_status(S["llm"])}


@app.post("/api/llm/test")
def test_llm():
    return {"llm": llm_public(), "provider": llm.provider(S["llm"]["url"]), "test": llm.test_llm(dict(S["llm"]))}


@app.get("/api/llm")
def get_llm():
    return {"llm": llm_public(), "status": llm.llm_status(S["llm"])}


@app.get("/api/export/inp")
def export_inp():
    with lock:
        need_model()
        try:
            from .control import disp_control

            sp, ctl = disp_control(active(), S["spec"])
            st = {"type": "static" if sp["step"]["type"] == "arclength" else sp["step"]["type"]}
            if sp["step"]["type"] == "arclength":
                st["nlgeom"] = True
            if ctl and ctl["kind"] == "average":
                ctl["k"] = calculix.machine_stiffness(active(), eff_base(), sp, ctl)
            text, notes, _ = calculix.build_inp(active(), eff_base(), sp, step=st, ctl=ctl)
        except Exception as e:
            fail(e)
        return PlainTextResponse(text, headers={"Content-Disposition": "attachment; filename=job.inp"})


@app.get("/api/export/abaqus")
def export_abaqus(element: str = "S4R"):
    with lock:
        need_model()
        from . import abaqus

        res = S.get("result")
        info = getattr(res, "abaqus", None) if res is not None and not S.get("stale") else None
        try:
            text, notes = abaqus.export(active(), S["spec"], info=info, element=element if element in ("S4R", "S4") else "S4R")
        except Exception as e:
            fail(e)
        return PlainTextResponse(text, headers={"Content-Disposition": "attachment; filename=job_abaqus.inp"})


@app.get("/api/export/spec")
def export_spec():
    with lock:
        need_model()
        return JSONResponse(S["spec"], headers={"Content-Disposition": "attachment; filename=spec.json"})


class SpecIn(BaseModel):
    spec: dict


@app.post("/api/import/spec")
def import_spec(body: SpecIn):
    with lock:
        need_model()
        base = default_spec(S["base"])
        for k, v in body.spec.items():
            if k in base:
                base[k] = v
        S["spec"] = base
        S["result"] = None
        S["cache"] = {}
        try:
            return state()
        except Exception as e:
            fail(e)


@app.get("/")
def index():
    return FileResponse(os.path.join(WEB, "index.html"), headers={"Cache-Control": "no-store"})


app.mount("/examples", StaticFiles(directory=EXAMPLES), name="examples")
app.mount("/", StaticFiles(directory=WEB), name="web")
