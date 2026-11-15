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
