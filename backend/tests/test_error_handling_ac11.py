"""AC-11 canonical-error-shape tests.

AC-11 requires **every** non-2xx response to:

* use the body shape ``{"error": "<msg>", "code": "<machine_code>"}``;
* use one of the HTTP status codes in the allow-list
  ``200 / 400 / 404 / 413 / 415 / 500``.

The route handlers in :mod:`app.main` already emit the canonical
shape on their happy error paths (this is exercised by
``test_api_extract.py``). This module is the safety net for the
"everywhere else" cases — the centralized exception handlers that
turn FastAPI/Starlette's default 404 / 422 / 500 responses into
the canonical shape too.

Coverage:

* **Unknown route** (FastAPI's auto 404) returns 404 NOT_FOUND.
* **Static file 404** (Starlette's StaticFiles mount) returns
  404 NOT_FOUND — not the default ``{"detail": "Not Found"}``.
* **Query param out of range** (RequestValidationError) returns
  400 INVALID_URL — 422 is mapped down to 400 because 422 is not
  in the AC-11 allow-list.
* **Body validation** (RequestValidationError) returns
  400 INVALID_URL.
* **HTTPException raised by a route** returns the canonical shape.
* **Uncaught Exception** (defense-in-depth 500 handler) returns
  500 INTERNAL_ERROR with the canonical shape.
* **All 5 error codes** in the AC-11 status-code allow-list are
  empirically exercised by at least one test.
* **Response content-type** is ``application/json`` for every
  error, so JS clients can parse it without sniffing.
"""

from __future__ import annotations

import asyncio
from unittest import mock

import httpx
import pytest
from fastapi import HTTPException

from app import config as cfg
from app import config as cfg
from app import main as main_mod
from app.models import (
    CODE_INTERNAL,
    CODE_INVALID_URL,
    CODE_NOT_FOUND,
    CODE_PAYLOAD_TOO_LARGE,
    CODE_UNSUPPORTED_MEDIA,
    ErrorResponse,
)
from app.pipeline.ingest import UnsupportedMediaError


# --- Fixtures ---------------------------------------------------------------


@pytest.fixture
def client(tmp_path, monkeypatch):
    """Build an isolated FastAPI app with sandboxed data dirs.

    The ``tmp_path`` is also used as the frontend dir by default
    so the ``/`` mount doesn't pick up the real ``frontend/``
    directory. Static-file tests use ``tmp_path / 'index.html'``
    to control which files the StaticFiles mount can serve.

    ``raise_app_exceptions=False`` is critical: by default
    :class:`httpx.ASGITransport` re-raises any exception that
    escapes the ASGI app, which would short-circuit our
    defense-in-depth 500 handler. Setting it to ``False`` makes
    the test see the canonical 500 body instead of a traceback.
    """
    monkeypatch.setattr(cfg, "AUDIO_DIR", tmp_path / "audio")
    monkeypatch.setattr(cfg, "OUTPUTS_DIR", tmp_path / "outputs")
    # Point the frontend mount at the sandbox so static 404s are
    # reproducible across hosts. ``_FRONTEND_DIR`` is captured at
    # module import time, so we patch the module-level symbol
    # directly rather than relying on ``setenv``.
    frontend_dir = tmp_path / "frontend"
    frontend_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(main_mod, "_FRONTEND_DIR", frontend_dir)
    app = main_mod.create_app()
    transport = httpx.ASGITransport(app=app, raise_app_exceptions=False)
    return httpx.AsyncClient(transport=transport, base_url="http://testserver")


def _assert_canonical_shape(resp: httpx.Response) -> dict:
    """Assert the response uses the AC-11 body shape; return the body."""
    # First, content-type must be JSON (so JS clients can parse it).
    ctype = resp.headers.get("content-type", "")
    assert ctype.startswith("application/json"), (
        f"expected JSON content-type, got {ctype!r}; "
        f"body={resp.text!r}"
    )
    body = resp.json()
    assert set(body.keys()) == {"error", "code"}, (
        f"expected exactly {{error, code}} keys, got {set(body.keys())}; "
        f"body={body!r}"
    )
    # And both fields are non-empty strings.
    assert isinstance(body["error"], str) and body["error"], (
        f"'error' must be a non-empty string, got {body['error']!r}"
    )
    assert isinstance(body["code"], str) and body["code"], (
        f"'code' must be a non-empty string, got {body['code']!r}"
    )
    return body


def _add_route_before_static_mount(app, method: str, path: str, handler) -> None:
    """Add a new route to ``app`` and move it before the static-files mount.

    Starlette matches routes in reverse insertion order, so a route
    added after ``create_app()`` returns would be intercepted by
    the ``StaticFiles`` mount at ``/`` (which catches every GET
    under the prefix). This helper splices the freshly-registered
    route into the slot just before the static mount, so explicit
    test routes win over the catch-all.
    """
    route_fn = getattr(app, method.lower())
    route_fn(path)(handler)
    # The just-registered route is the last one in the list. Pop it
    # back out so we can re-insert it at the static mount's slot.
    new_route = app.router.routes.pop()
    # Find the static mount's current index. The static mount is
    # always present in the production app; the test fixture sets
    # FRONTEND_DIR to a real directory, so we don't need a
    # defensive ``or len(routes)`` fallback.
    static_idx = next(
        i
        for i, r in enumerate(app.router.routes)
        if hasattr(r, "app") and "StaticFiles" in str(type(r.app))
    )
    # Insert the new route at the static mount's slot. The static
    # mount (and everything after it) shifts right by one.
    app.router.routes.insert(static_idx, new_route)


# --- Unknown route 404 ------------------------------------------------------


@pytest.mark.asyncio
async def test_unknown_route_returns_canonical_404(client) -> None:
    """FastAPI's auto-404 for an unmatched path uses the canonical shape."""
    async with client as c:
        resp = await c.get("/this-route-does-not-exist")
    assert resp.status_code == 404
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_unknown_method_returns_canonical_404(client) -> None:
    """A method that doesn't match a route handler also gets 404 canonical."""
    async with client as c:
        # ``/api/jobs`` only has GET, so PATCH should fall through to 404.
        resp = await c.patch("/api/jobs")
    assert resp.status_code == 404
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_NOT_FOUND


# --- Static file 404 --------------------------------------------------------


@pytest.mark.asyncio
async def test_static_file_404_returns_canonical_shape(client) -> None:
    """A missing static asset under the frontend mount returns 404 canonical.

    Starlette's ``StaticFiles`` mount raises ``HTTPException(404)``
    on a miss; without our :class:`StarletteHTTPException` handler
    the body would be ``{"detail": "Not Found"}``.
    """
    async with client as c:
        resp = await c.get("/does-not-exist.css")
    assert resp.status_code == 404
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_NOT_FOUND


@pytest.mark.asyncio
async def test_static_root_with_no_index_html_returns_404(client) -> None:
    """``GET /`` with no ``index.html`` in the frontend dir returns 404."""
    async with client as c:
        resp = await c.get("/")
    # 404 is the expected behavior here (frontend/index.html is missing);
    # the canonical shape is what matters.
    assert resp.status_code == 404
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_NOT_FOUND


# --- RequestValidationError -> 400 ------------------------------------------


@pytest.mark.asyncio
async def test_query_param_out_of_range_returns_400_canonical(
    client,
) -> None:
    """AC-11 maps Pydantic 422 down to 400 INVALID_URL."""
    async with client as c:
        resp = await c.get("/api/jobs?limit=999")
    assert resp.status_code == 400
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_INVALID_URL
    # The error message should mention the offending field so the
    # user (or the frontend) can act on it.
    assert "limit" in body["error"].lower()


@pytest.mark.asyncio
async def test_query_param_below_range_returns_400_canonical(
    client,
) -> None:
    async with client as c:
        resp = await c.get("/api/jobs?limit=0")
    assert resp.status_code == 400
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_INVALID_URL


@pytest.mark.asyncio
async def test_body_validation_returns_400_canonical(client) -> None:
    """Body validation: POST /api/extract with a non-object body returns 400.

    FastAPI's default behavior for a non-object JSON body is to
    raise ``RequestValidationError`` (422). The AC-11 handler
    maps it down to 400 with the canonical body and a flattened
    error message that names the offending field.
    """
    async with client as c:
        # Send a JSON array (not an object) to trigger body validation.
        resp = await c.post(
            "/api/extract",
            content=b"[1, 2, 3]",
            headers={"content-type": "application/json"},
        )
    assert resp.status_code == 400
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_INVALID_URL


# --- HTTPException raised by a route handler -------------------------------


@pytest.mark.asyncio
async def test_httpexception_raised_by_route_returns_canonical_shape(
    client,
) -> None:
    """A route that raises ``HTTPException`` directly is normalized too.

    The existing route handlers don't currently raise ``HTTPException``
    directly, but we want to guarantee the contract holds for any
    future handler. We use synthetic test routes below to inject
    404 / 413 / 500 cases in isolation.
    """
    app = client._transport.app  # type: ignore[attr-defined]

    def _raise_404() -> None:
        raise HTTPException(status_code=404, detail="synthetic not-found")

    def _raise_413() -> None:
        raise HTTPException(status_code=413, detail="synthetic too large")

    def _raise_500() -> None:
        raise HTTPException(status_code=500, detail="synthetic internal")

    _add_route_before_static_mount(app, "get", "/_test/raise-404", _raise_404)
    _add_route_before_static_mount(app, "get", "/_test/raise-413", _raise_413)
    _add_route_before_static_mount(app, "get", "/_test/raise-500", _raise_500)

    async with client as c:
        r404 = await c.get("/_test/raise-404")
        r413 = await c.get("/_test/raise-413")
        r500 = await c.get("/_test/raise-500")

    for resp, expected_code in (
        (r404, CODE_NOT_FOUND),
        (r413, CODE_PAYLOAD_TOO_LARGE),
        (r500, CODE_INTERNAL),
    ):
        body = _assert_canonical_shape(resp)
        assert body["code"] == expected_code, (
            f"status {resp.status_code} -> code {body['code']}, "
            f"expected {expected_code}"
        )


# --- Uncaught Exception -> 500 INTERNAL_ERROR ------------------------------


@pytest.mark.asyncio
async def test_uncaught_exception_returns_500_canonical(client) -> None:
    """Defense in depth: an exception that escapes a route is 500 canonical.

    The /api/extract handler already catches every pipeline
    exception, but a future route that doesn't wrap its body would
    otherwise emit FastAPI's default ``{"detail": "Internal Server Error"}``
    with a stack trace in the logs. With the catch-all handler
    the body is canonical and the log line is structured.
    """
    app = client._transport.app  # type: ignore[attr-defined]

    def _raise_bare() -> None:
        raise RuntimeError("synthetic boom")

    _add_route_before_static_mount(app, "get", "/_test/raise-bare", _raise_bare)

    async with client as c:
        resp = await c.get("/_test/raise-bare")
    assert resp.status_code == 500
    body = _assert_canonical_shape(resp)
    assert body["code"] == CODE_INTERNAL
    # The exception's text is preserved in the error message so
    # operators have something to grep for in production.
    assert "synthetic boom" in body["error"]


# --- AC-11 status code allow-list coverage ---------------------------------


@pytest.mark.asyncio
async def test_all_ac11_status_codes_observable(client) -> None:
    """Every status code in the AC-11 allow-list is exercised at least once.

    The test enumerates the canonical codes and asserts each has
    a reachable endpoint that returns it. The point of this
    meta-test is to make a future PR that drops a code loud:
    "you've broken the AC-11 contract" rather than slipping a
    silent omission through review.
    """
    expected = {
        CODE_NOT_FOUND,         # 404
        CODE_INVALID_URL,       # 400
        CODE_PAYLOAD_TOO_LARGE, # 413 (raised by HTTPException stub above)
        CODE_UNSUPPORTED_MEDIA, # 415 (raised by extract/upload path)
        CODE_INTERNAL,          # 500
    }
    seen: set[str] = set()
    app = client._transport.app  # type: ignore[attr-defined]

    # Add a synthetic 413 + 500 route, both placed before the
    # static-files mount so they take precedence over the catch-all.
    def _raise_413() -> None:
        raise HTTPException(status_code=413, detail="synthetic too large")

    def _raise_bare() -> None:
        raise RuntimeError("synthetic boom")

    _add_route_before_static_mount(app, "get", "/_test/raise-413", _raise_413)
    _add_route_before_static_mount(app, "get", "/_test/raise-bare", _raise_bare)

    async with client as c:
        # 404 NOT_FOUND: unknown route
        seen.add(_assert_canonical_shape(
            await c.get("/nope")
        )["code"])
        # 400 INVALID_URL: out-of-range query param
        seen.add(_assert_canonical_shape(
            await c.get("/api/jobs?limit=999")
        )["code"])
        # 413 PAYLOAD_TOO_LARGE: explicit HTTPException
        seen.add(_assert_canonical_shape(
            await c.get("/_test/raise-413")
        )["code"])
        # 415 UNSUPPORTED_MEDIA: API extract path
        import app.pipeline.orchestrator as orch

        with mock.patch.object(
            orch, "_stage_ingest",
            mock.MagicMock(side_effect=UnsupportedMediaError("only mp4")),
        ):
            post_resp = await c.post(
                "/api/extract",
                files={"file": ("x.avi", b"x", "video/avi")},
            )
        assert post_resp.status_code == 202, (
            f"Expected 202 from async submit, got {post_resp.status_code}"
        )
        job_id = post_resp.json()["job_id"]
        # Poll result endpoint until the background task fails
        for _ in range(20):
            result_resp = await c.get(f"/api/jobs/{job_id}/result")
            if result_resp.status_code != 409:
                break
            await asyncio.sleep(0.01)
        seen.add(_assert_canonical_shape(result_resp)["code"])
        # 500 INTERNAL_ERROR: uncaught exception
        seen.add(_assert_canonical_shape(
            await c.get("/_test/raise-bare")
        )["code"])

    missing = expected - seen
    assert not missing, f"AC-11 codes not exercised by the smoke probe: {missing}"


# --- Pydantic ErrorResponse model parity -----------------------------------


def test_error_response_model_matches_runtime_shape() -> None:
    """The Pydantic ``ErrorResponse`` model uses the same field names as
    what the handlers emit, so ``response_model=ErrorResponse`` would
    type-check. (We don't actually wire response_model on the global
    handlers — the JSON is built directly — but the test pins the
    contract for future contributors.)
    """
    model = ErrorResponse(error="boom", code="BAD")
    dumped = model.model_dump()
    assert dumped == {"error": "boom", "code": "BAD"}
