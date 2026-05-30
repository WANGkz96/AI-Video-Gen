import importlib
import sys

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings


def test_auth_required_without_token_fails_fast(monkeypatch):
    monkeypatch.setenv("AI_VIDEO_GEN_AUTH_REQUIRED", "1")
    monkeypatch.setenv("AI_VIDEO_GEN_API_TOKEN", "")

    with pytest.raises(ValueError, match="AI_VIDEO_GEN_API_TOKEN"):
        Settings.from_env()


def load_main(monkeypatch, *, auth_required: bool, token: str = "test-token"):
    monkeypatch.setenv("AI_VIDEO_GEN_AUTH_REQUIRED", "1" if auth_required else "0")
    monkeypatch.setenv("AI_VIDEO_GEN_API_TOKEN", token)
    if "backend.app.main" in sys.modules:
        return importlib.reload(sys.modules["backend.app.main"])
    return importlib.import_module("backend.app.main")


def test_auth_disabled_allows_api_requests(monkeypatch):
    main = load_main(monkeypatch, auth_required=False, token="")
    client = TestClient(main.app)

    response = client.get("/api/health")

    assert response.status_code == 200


def test_auth_enabled_rejects_missing_or_invalid_token(monkeypatch):
    main = load_main(monkeypatch, auth_required=True, token="secret-token")
    client = TestClient(main.app)

    assert client.get("/api/health").status_code == 401
    assert client.get("/api/health", headers={"Authorization": "Bearer wrong"}).status_code == 401


def test_auth_enabled_accepts_valid_bearer_token(monkeypatch):
    main = load_main(monkeypatch, auth_required=True, token="secret-token")
    client = TestClient(main.app)

    response = client.get("/api/health", headers={"Authorization": "Bearer secret-token"})

    assert response.status_code == 200
