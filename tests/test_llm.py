import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest
from helpers import ROOT
from fastapi.testclient import TestClient
from femweb import server

REPLIES = []
SEEN = []


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_GET(self):
        body = json.dumps({"data": [{"id": "qwen3:mock"}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        req = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        SEEN.append(req)
        content = REPLIES.pop(0)
        body = json.dumps({"choices": [{"message": {"role": "assistant", "content": content}}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(body)


@pytest.fixture(scope="module")
def client():
    srv = HTTPServer(("127.0.0.1", 0), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    old = dict(server.S["llm"])
    server.S["llm"].update({"url": f"http://127.0.0.1:{srv.server_port}/v1", "model": "qwen3:mock", "mode": "llm", "key": ""})
    yield TestClient(server.app)
    server.S["llm"].update(old)
    srv.shutdown()


def test_llm_ops_are_validated_and_applied(client):
    client.post("/api/examples/cantilever_beam?preset=0")
    REPLIES[:] = ['<think>plan</think>{"ops":[{"op":"add_bc","type":"fixed","select":{"plane":{"axis":"x","at":"min"}}},{"op":"add_load","type":"force","select":{"plane":{"axis":"x","at":"max"}},"vector":[0,0,-2000]}],"reply":"고정과 하중을 추가했습니다."}']
    SEEN.clear()
    r = client.post("/api/chat", json={"text": "왼쪽 끝 고정, 오른쪽 끝에 아래로 2 kN"}).json()
    assert r["chat"]["engine"] == "llm" and r["chat"]["error"] is None
    assert r["preview"]["loads"][0]["total"] == [0.0, 0.0, -2000.0]
    assert len(r["spec"]["bcs"]) == 1
    sent = SEEN[0]
    assert sent["model"] == "qwen3:mock" and sent["response_format"] == {"type": "json_object"}
    assert sent["messages"][-1]["content"].endswith("/no_think")
    assert "MODEL:" in sent["messages"][1]["content"] and '"beam"' in sent["messages"][1]["content"]


def test_llm_gets_one_retry_with_the_validation_error(client):
    client.post("/api/examples/cantilever_beam?preset=0")
    REPLIES[:] = [
        '{"ops":[{"op":"add_bc","type":"fixed","select":{"group":"left_end"}}],"reply":"x"}',
        '{"ops":[{"op":"add_bc","type":"fixed","select":{"plane":{"axis":"x","at":"min"}}}],"reply":"고정했습니다."}',
    ]
    SEEN.clear()
    r = client.post("/api/chat", json={"text": "왼쪽 끝 고정"}).json()
    assert len(SEEN) == 2 and "left_end" in SEEN[1]["messages"][-1]["content"]
    assert r["chat"]["error"] is None and len(r["spec"]["bcs"]) == 1


def test_auto_mode_falls_back_to_rules_when_llm_is_down(client):
    client.post("/api/examples/cantilever_beam?preset=0")
    url = server.S["llm"]["url"]
    server.S["llm"].update({"url": "http://127.0.0.1:9/v1", "mode": "auto"})
    try:
        r = client.post("/api/chat", json={"text": "x 최소인 끝을 완전 고정해줘"}).json()
    finally:
        server.S["llm"].update({"url": url, "mode": "llm"})
    assert r["chat"]["engine"] == "rule" and "규칙 기반" in r["chat"]["note"]
    assert len(r["spec"]["bcs"]) == 1


def test_llm_request_can_be_cancelled(client):
    import time

    from femweb import llm

    class Slow(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_POST(self):
            self.rfile.read(int(self.headers["Content-Length"]))
            time.sleep(8)

    slow = HTTPServer(("127.0.0.1", 0), Slow)
    threading.Thread(target=slow.serve_forever, daemon=True).start()
    cfg = {"url": f"http://127.0.0.1:{slow.server_port}/v1", "model": "qwen3:mock", "mode": "llm", "key": ""}
    res = {}

    def run():
        try:
            llm.ask_llm(cfg, "x", "ctx")
        except Exception as e:
            res["e"] = e

    th = threading.Thread(target=run)
    th.start()
    time.sleep(0.6)
    assert llm.cancel() is True
    th.join(timeout=3)
    slow.shutdown()
    assert isinstance(res.get("e"), llm.Cancelled)


def test_provider_names():
    from femweb import llm

    assert llm.provider("http://localhost:11434/v1") == "Ollama (이 컴퓨터)"
    assert llm.provider("https://generativelanguage.googleapis.com/v1beta/openai/").startswith("Google Gemini")
    assert llm.provider("https://example.org/v1").startswith("외부 서버")


def test_status_and_real_response_test(client):
    REPLIES[:] = ['{"ops": [], "reply": "안녕하세요"}']
    r = client.get("/api/llm").json()
    assert r["status"]["ok"] and r["status"]["has_model"] and r["status"]["provider"].startswith("로컬")
    t = client.post("/api/llm/test").json()["test"]
    assert t["ok"] and t["json_ok"] and t["reply"] == "안녕하세요"


def test_theory_page_is_served(client):
    r = client.get("/theory.html")
    assert r.status_code == 200 and "유한요소해석" in r.text
    assert 'href="/theory.html"' in client.get("/").text


def test_rule_first_skips_llm_for_clear_sentences(client):
    server.S["llm"]["mode"] = "rule_first"
    try:
        client.post("/api/examples/cantilever_beam?preset=0")
        SEEN.clear()
        r = client.post("/api/chat", json={"text": "왼쪽 끝을 고정하고 오른쪽 끝에 아래로 2 kN"}).json()
        assert SEEN == [] and r["chat"]["engine"] == "rule" and r["chat"]["error"] is None
        assert len(r["spec"]["bcs"]) == 1 and r["preview"]["loads"][0]["total"] == [0.0, 0.0, -2000.0]
        r = client.post("/api/chat", json={"text": "오른쪽 끝에 1 kN"}).json()
        assert SEEN == [] and "방향" in r["chat"]["reply"] and len(r["spec"]["loads"]) == 1
    finally:
        server.S["llm"]["mode"] = "llm"


def test_rule_first_sends_unclear_sentences_to_llm(client):
    server.S["llm"]["mode"] = "rule_first"
    try:
        client.post("/api/examples/cantilever_beam?preset=0")
        REPLIES[:] = ['{"ops":[{"op":"add_bc","type":"fixed","select":{"plane":{"axis":"x","at":"min"}}}],"reply":"고정했습니다."}']
        SEEN.clear()
        r = client.post("/api/chat", json={"text": "벽에 붙은 쪽을 단단히 잡아놔"}).json()
        assert len(SEEN) == 1 and r["chat"]["engine"] == "llm" and len(r["spec"]["bcs"]) == 1
    finally:
        server.S["llm"]["mode"] = "llm"


def test_rule_first_without_llm_applies_what_it_understood(client):
    client.post("/api/examples/cantilever_beam?preset=0")
    url = server.S["llm"]["url"]
    server.S["llm"].update({"url": "http://127.0.0.1:9/v1", "mode": "rule_first"})
    try:
        r = client.post("/api/chat", json={"text": "왼쪽 끝 고정, 그리고 뭔가 멋지게 해줘"}).json()
    finally:
        server.S["llm"].update({"url": url, "mode": "llm"})
    assert r["chat"]["engine"] == "rule" and len(r["spec"]["bcs"]) == 1
    assert "이해하지 못한" in r["chat"]["reply"] and "연결하지 못해" in r["chat"]["note"]


def test_undo_and_run_action(client):
    server.S["llm"]["mode"] = "rule_first"
    try:
        client.post("/api/examples/cantilever_beam?preset=1")
        n = len(client.get("/api/state").json()["spec"]["loads"])
        r = client.post("/api/chat", json={"text": "끝단에 y 방향으로 300 N"}).json()
        assert len(r["spec"]["loads"]) == n + 1 and r["can_undo"]
        r = client.post("/api/chat", json={"text": "되돌려"}).json()
        assert len(r["spec"]["loads"]) == n
        r = client.post("/api/chat", json={"text": "해석 실행해줘"}).json()
        assert r["chat"]["action"] == "solve"
        client.post("/api/chat", json={"text": "하중 2배"})
        assert client.post("/api/undo").json()["spec"]["loads"][0]["vector"] == [0.0, 0.0, -1000.0]
    finally:
        server.S["llm"]["mode"] = "llm"
