import types

import pytest
from fastapi.testclient import TestClient

import roo_genai_openai_api as api


def test_gemini_routes_to_google_with_location(monkeypatch):
    called = {}

    def fake_get_client(location=None):
        called["location"] = location

        class DummyClient:
            def __init__(self):
                self.models = self

            def generate_content(self, model, contents, config):
                called["model"] = model
                return types.SimpleNamespace(
                    candidates=[
                        types.SimpleNamespace(
                            content=types.SimpleNamespace(parts=[types.SimpleNamespace(text="ok")])
                        )
                    ]
                )

        return DummyClient()

    monkeypatch.setattr(api, "_get_client", fake_get_client)
    client = TestClient(api.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "gemini-1.5-pro", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert called["location"] == api.DEFAULT_VERTEX_LOCATION
    assert called["model"] == "gemini-1.5-pro"


def test_claude_routes_to_anthropic_with_region(monkeypatch):
    called = {}

    class DummyAnthropic:
        def __init__(self):
            self.messages = self

        def create(self, **kwargs):
            called.update(kwargs)
            return types.SimpleNamespace(content=[types.SimpleNamespace(type="text", text="hello")])

    def fake_get_anthropic(region: str):
        called["region"] = region
        return DummyAnthropic()

    monkeypatch.setattr(api, "_get_anthropic_client", fake_get_anthropic)
    client = TestClient(api.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "claude-3-sonnet", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 200
    assert called["region"] == api.DEFAULT_VERTEX_REGION
    assert called["model"] == "claude-3-sonnet"


def test_unknown_model_errors():
    client = TestClient(api.app)
    resp = client.post(
        "/v1/chat/completions",
        json={"model": "unknown-model-foo", "messages": [{"role": "user", "content": "hi"}]},
    )
    assert resp.status_code == 400
    body = resp.json()
    assert "Unsupported model" in body["error"]["message"]
