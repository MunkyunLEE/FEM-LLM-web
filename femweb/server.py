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

from . import llm, plots
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
S = {"base": None, "spec": None, "result": None, "stale": False, "chat": [], "llm": dict(llm.DEFAULTS), "cache": {}, "solve_time": None}


def fail(msg, code=400):
    raise HTTPException(status_code=code, detail=str(msg))


def need_model():
    if S["base"] is None:
        fail("먼저 지오메트리를 불러오세요")


def active():
    base, spec = S["base"], S["spec"]
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
        "warnings": warnings_for(S["spec"], S["base"]),
        "result": S["result"].meta() if S["result"] else None,
        "stale": S["stale"],
        "solve_time": S["solve_time"],
        "tools": tools(),
        "llm": llm_public(),
    }
    if view:
        out["view"] = m.to_view()
    return out


def tools():
    return {"ccx": calculix.find_ccx(), "tensormesh": has_tm()}


def llm_public():
    c = S["llm"]
    return {"url": c["url"], "model": c["model"], "mode": c["mode"], "has_key": bool(c.get("key"))}


def set_model(base, spec=None):
    S["base"] = base
    S["spec"] = spec or default_spec(base)
    S["result"] = None
    S["stale"] = False
    S["cache"] = {}
    S["chat"] = []
    S["solve_time"] = None


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
            spec, _ = apply_ops(default_spec(base), ops, base)
            set_model(base, spec)
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


def _apply(ops):
    before = (S["spec"]["solver"], S["spec"]["divisions"])
    m = active()
    geo = [o for o in ops if isinstance(o, dict) and o.get("op") in ("set_solver", "set_divisions")]
    rest = [o for o in ops if o not in geo]
    spec, log = apply_ops(S["spec"], geo, None)
    S["spec"] = spec
    if (spec["solver"], spec["divisions"]) != before:
        S["result"] = None
        m = active()
    try:
        spec, log2 = apply_ops(S["spec"], rest, m)
    except Exception:
        raise
    S["spec"] = spec
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


@app.post("/api/chat")
def chat(body: Chat):
    with lock:
        need_model()
        cfg = S["llm"]
        m = active()
        text = body.text.strip()
        if not text:
            fail("내용이 비어 있습니다")
        engine = "rule"
        note = ""
        ops, reply = [], ""
        used_llm = False
        if cfg["mode"] != "rule":
            ctx = llm.context(m, S["spec"], S["result"].meta() if S["result"] else None)
            try:
                ops, reply = llm.ask_llm(cfg, text, ctx, S["chat"])
                used_llm = True
                engine = "llm"
                try:
                    apply_ops(S["spec"], ops, m if not any(o.get("op") in ("set_solver", "set_divisions") for o in ops if isinstance(o, dict)) else None)
                except Exception as e:
                    ops, reply = llm.ask_llm(cfg, text, ctx, S["chat"], error=str(e))
            except llm.Cancelled:
                fail("LLM 요청을 중단했습니다", 409)
            except Exception as e:
                if cfg["mode"] == "llm":
                    fail(f"LLM 호출 실패: {e}")
                note = f"LLM에 연결하지 못해 규칙 기반 해석기를 사용했습니다 ({str(e)[:120]})"
                used_llm = False
        if not used_llm:
            ops, reply = llm.rule_parse(text, m, S["spec"])
            engine = "rule"
        error = None
        log = []
        if ops:
            try:
                log = _apply(ops)
            except Exception as e:
                error = str(e)
        S["chat"].append({"role": "user", "content": text})
        S["chat"].append({"role": "assistant", "content": json.dumps({"ops": ops, "reply": reply}, ensure_ascii=False)})
        out = state(view=True)
        out["chat"] = {"reply": reply, "ops": ops, "engine": engine, "note": note, "error": error, "log": log}
        return out


def _solve_with(m, base, spec):
    if spec["solver"] == "calculix":
        return calculix.solve(m, base, spec)
    if spec["solver"] == "barhinge":
        if base.kind != "obj":
            raise ValueError("바-힌지 솔버는 OBJ 쉘 패널 모델에만 쓸 수 있습니다")
        from .solvers import barhinge

        return barhinge.solve(base, spec)
    if spec["solver"] == "tensormesh":
        from .solvers import solid_tm

        return solid_tm.solve(m, spec)
    raise ValueError("알 수 없는 솔버")


def _worker(conn, m, base, spec):
    try:
        os.setpgrp()
    except Exception:
        pass
    try:
        conn.send(("ok", _solve_with(m, base, spec)))
    except BaseException as e:
        conn.send(("err", str(e), traceback.format_exc()))
    finally:
        conn.close()


JOB = {"proc": None, "cancelled": False}


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
    proc = ctx.Process(target=_worker, args=(child, active(), S["base"], S["spec"]), daemon=True)
    JOB["cancelled"] = False
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


@app.get("/api/plots")
def get_plots():
    with lock:
        need_model()
        out = []
        if not S["result"]:
            return out
        for p in S["spec"]["plots"]:
            try:
                out.append(plots.evaluate(p, S["result"], active()))
            except Exception as e:
                out.append({"id": p["id"], "title": p.get("title", "그래프"), "error": str(e), "series": []})
        return out


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
        if S["llm"]["mode"] not in ("auto", "llm", "rule"):
            S["llm"]["mode"] = "auto"
        return {"llm": llm_public(), "status": llm.llm_status(S["llm"])}


@app.get("/api/llm")
def get_llm():
    return {"llm": llm_public(), "status": llm.llm_status(S["llm"])}


@app.get("/api/export/inp")
def export_inp():
    with lock:
        need_model()
        try:
            text, notes, _ = calculix.build_inp(active(), S["base"], S["spec"], step={"type": "static" if S["spec"]["step"]["type"] == "arclength" else S["spec"]["step"]["type"]})
        except Exception as e:
            fail(e)
        return PlainTextResponse(text, headers={"Content-Disposition": "attachment; filename=job.inp"})


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
