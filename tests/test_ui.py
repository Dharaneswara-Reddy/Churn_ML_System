"""
The browser console at ``/ui``.

It is the only part of the system a non-programmer touches, so the properties
worth pinning are about *exposure*, not appearance: the page must not leak a
credential, and serving it must not weaken the authentication in front of
``/predict``.
"""

from __future__ import annotations

import re

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    import churn_system.api.api as api_mod

    return TestClient(api_mod.app)


class TestConsoleIsServed:
    def test_ui_returns_html(self, client):
        response = client.get("/ui")

        assert response.status_code == 200
        assert response.headers["content-type"].startswith("text/html")
        assert "<title>Churn Prediction Console</title>" in response.text

    def test_root_redirects_to_the_console(self, client):
        """A bare host in a browser should land somewhere useful, not on JSON."""
        response = client.get("/", follow_redirects=False)

        assert response.status_code == 307
        assert response.headers["location"] == "/ui"

    def test_the_old_root_payload_is_still_reachable(self, client):
        """`/` used to return this. Moving it would break anything polling it."""
        assert client.get("/status").json() == {
            "status": "ok",
            "message": "Churn model is running",
        }

    def test_the_console_is_excluded_from_the_openapi_document(self, client):
        """
        The console is not part of the API contract. Listing it would put an HTML
        page in generated clients.
        """
        paths = client.get("/openapi.json").json()["paths"]

        assert "/ui" not in paths
        assert "/predict" in paths


class TestConsoleLeaksNothing:
    """
    The page is served unauthenticated, so anything baked into it is public.
    """

    def test_no_credential_is_embedded(self, client):
        body = client.get("/ui").text

        for marker in (
            "CHURN_API_KEY",
            "CHURN_ADMIN_API_KEY",
            "CHURN_ARTIFACT_SIGNING_KEY",
            "CHURN_SUBJECT_KEY_SALT",
            "POSTGRES_PASSWORD",
        ):
            assert marker not in body, f"{marker} appears in the served page"

    def test_no_long_hex_token_is_embedded(self, client):
        """
        Catches a key pasted in during debugging even if the variable name that
        would have flagged it is gone.
        """
        body = client.get("/ui").text

        assert not re.search(r"\b[0-9a-f]{32,}\b", body)

    def test_the_page_asks_the_visitor_for_the_key(self, client):
        """
        The credential must come from the person using the page, not from the
        server — that is what keeps the console safe to serve unauthenticated.
        """
        body = client.get("/ui").text

        assert 'id="apikey"' in body
        assert "X-API-Key" in body

    def test_serving_the_console_does_not_open_up_predict(self, client, monkeypatch):
        """
        The regression that would matter: adding an unauthenticated page must not
        accidentally place /predict outside the auth dependency.
        """
        import importlib

        import churn_system.api.api as api_mod

        monkeypatch.setenv("CHURN_API_KEY", "a-real-key")
        monkeypatch.delenv("CHURN_ALLOW_ANONYMOUS", raising=False)
        reloaded = importlib.reload(api_mod)
        try:
            secured = TestClient(reloaded.app)

            assert secured.get("/ui").status_code == 200
            assert secured.post("/predict", json={}).status_code == 401
        finally:
            monkeypatch.undo()
            importlib.reload(api_mod)


class TestConsoleStaysInSyncWithTheModel:
    def test_the_form_is_built_from_the_live_schema(self, client):
        """
        The form is generated from /openapi.json rather than hardcoded, so
        promoting a model with different features changes the page with no
        redeploy. A hardcoded field list would silently drift.
        """
        body = client.get("/ui").text

        assert "/openapi.json" in body
        assert "DynamicPredictionRequest" in body

    def test_the_threshold_is_read_from_the_response(self, client):
        """
        The page must render the threshold the API reports, not assume 0.5 — the
        deployed threshold is 0.14 and drawing the midpoint would misrepresent
        every result.
        """
        body = client.get("/ui").text

        assert "d.threshold" in body
        assert "0.5" not in body.split("<script>")[1].split("const HIGH_RISK")[0]


def test_the_console_is_declared_as_package_data():
    """
    ``/ui`` reads the file relative to the installed module. The container runs
    from the source tree so it works there either way, but a plain
    ``pip install`` would serve a 404 without this declaration — a failure that
    only appears in the one environment nobody tests.
    """
    from pathlib import Path

    source = Path("pyproject.toml").read_text(encoding="utf-8")

    # Parsed with tomllib where available; 3.10 has no tomllib in the stdlib and
    # this suite runs on 3.10 and 3.12, so it falls back to reading the section.
    try:
        import tomllib

        patterns = tomllib.loads(source)["tool"]["setuptools"]["package-data"][
            "churn_system"
        ]
        declared = " ".join(patterns)
    except ModuleNotFoundError:
        # Split on a newline-anchored header: "[" also opens the list value.
        section = source.split("[tool.setuptools.package-data]", 1)[1]
        declared = section.split("\n[", 1)[0]

    assert "api/static" in declared, (
        "api/static/*.html is not declared as package data; /ui would 404 in a "
        f"pip-installed environment. Declared: {declared.strip()}"
    )
