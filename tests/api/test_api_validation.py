"""API validation and error paths. These need no trained model."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from api.main import create_app
from fastapi.testclient import TestClient

from inspection.config import Settings, load_settings


def make_settings(tmp_path: Path, **api: object) -> Settings:
    s = load_settings()
    s.paths.artifacts_dir = str(tmp_path / "artifacts")
    for k, v in api.items():
        setattr(s.api, k, v)
    return s


@pytest.fixture()
def client(tmp_path: Path) -> Iterator[TestClient]:
    with TestClient(create_app(make_settings(tmp_path), warm=False)) as c:
        yield c


def post(client: TestClient, files: object, params: str = "category=bottle", **kw: object):  # type: ignore[no-untyped-def]
    return client.post(f"/v1/inspect?{params}", files=files, **kw)


def assert_error(resp, status: int, code: str) -> None:  # type: ignore[no-untyped-def]
    assert resp.status_code == status, resp.text
    body = resp.json()
    assert body["error"]["code"] == code
    assert body["error"]["request_id"] == resp.headers["x-request-id"]
    assert "Traceback" not in resp.text


def test_health_ready_categories_empty(client: TestClient) -> None:
    assert client.get("/v1/health").json() == {"status": "ok"}
    r = client.get("/v1/ready")
    assert r.status_code == 503 and r.json()["ready"] is False
    assert client.get("/v1/categories").json() == []


def test_unknown_category_rejected_before_any_path_use(client: TestClient) -> None:
    r = post(client, {"file": ("a.png", b"x", "image/png")}, "category=../../etc/passwd")
    assert_error(r, 404, "unknown_category")


def test_known_category_without_artifact(client: TestClient) -> None:
    r = post(client, {"file": ("a.png", b"x", "image/png")})
    assert_error(r, 404, "category_not_trained")


def test_missing_parameters_use_stable_schema(client: TestClient) -> None:
    r = client.post("/v1/inspect")
    assert_error(r, 422, "invalid_request")


def test_404_route_uses_error_schema(client: TestClient) -> None:
    assert_error(client.get("/nope"), 404, "http_error")


def test_metrics_endpoint(client: TestClient) -> None:
    client.get("/v1/health")
    text = client.get("/metrics").text
    assert "inspection_requests_total" in text


def test_api_key_required_when_configured(tmp_path: Path) -> None:
    app = create_app(make_settings(tmp_path, api_key="secret"), warm=False)
    with TestClient(app) as c:
        assert_error(c.get("/v1/categories"), 401, "unauthorized")
        assert c.get("/v1/categories", headers={"X-API-Key": "secret"}).status_code == 200
        assert c.get("/v1/health").status_code == 200  # liveness stays open


def test_cors_disabled_by_default(client: TestClient) -> None:
    r = client.get("/v1/health", headers={"Origin": "http://evil.example"})
    assert "access-control-allow-origin" not in r.headers


# The following tests need a trained artifact entry only for category lookup; the failure
# happens in decoding, so a manifest stub is enough to get past the whitelist check.


@pytest.fixture()
def stub_client(tmp_path: Path) -> Iterator[TestClient]:
    import json

    d = tmp_path / "artifacts" / "bottle"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "thresholds": {"image": 1.0, "pixel": 1.0},
                "created_at": "x",
                "bank_shape": [1, 1],
            }
        )
    )
    s = make_settings(tmp_path, max_upload_bytes=1000)
    with TestClient(create_app(s, warm=False)) as c:
        yield c


def test_not_an_image(stub_client: TestClient) -> None:
    r = post(stub_client, {"file": ("a.png", b"this is not an image", "image/png")})
    assert_error(r, 400, "invalid_image")


def test_empty_upload(stub_client: TestClient) -> None:
    assert_error(post(stub_client, {"file": ("a.png", b"", "image/png")}), 400, "empty_upload")


def test_upload_too_large(stub_client: TestClient) -> None:
    r = post(stub_client, {"file": ("a.png", b"0" * 2000, "image/png")})
    assert_error(r, 413, "upload_too_large")


def test_content_type_is_not_trusted(stub_client: TestClient) -> None:
    r = post(stub_client, {"file": ("a.png", b"GIF89a....", "image/png")})
    assert_error(r, 400, "invalid_image")


def _real_image(data_root: Path) -> bytes:
    return sorted((data_root / "bottle" / "test" / "good").glob("*.png"))[0].read_bytes()


def test_dimension_and_pixel_limits_with_real_image(tmp_path: Path, data_root: Path) -> None:
    import json

    d = tmp_path / "artifacts" / "bottle"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {"thresholds": {"image": 1.0, "pixel": 1.0}, "created_at": "x", "bank_shape": [1, 1]}
        )
    )
    raw = _real_image(data_root)
    s = make_settings(tmp_path, max_side=100)
    with TestClient(create_app(s, warm=False)) as c:
        assert_error(post(c, {"file": ("a.png", raw, "image/png")}), 422, "bad_dimensions")
    s = make_settings(tmp_path, max_pixels=1000)
    with TestClient(create_app(s, warm=False)) as c:
        assert_error(post(c, {"file": ("a.png", raw, "image/png")}), 413, "image_too_large")


def test_timeout_returns_504_and_keeps_the_slot_busy(
    tmp_path: Path, data_root: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A timed-out inference keeps running in its thread; the next request must wait for it
    instead of starting a second inference next to it (concurrency limit of one)."""
    import json
    import time

    from api.service import ModelService

    d = tmp_path / "artifacts" / "bottle"
    d.mkdir(parents=True)
    (d / "manifest.json").write_text(
        json.dumps(
            {"thresholds": {"image": 1.0, "pixel": 1.0}, "created_at": "x", "bank_shape": [1, 1]}
        )
    )

    def slow_inspect(self, category, img, use_cascade):  # type: ignore[no-untyped-def]
        time.sleep(0.6)
        raise RuntimeError("never reached by the client")

    monkeypatch.setattr(ModelService, "inspect", slow_inspect)
    raw = _real_image(data_root)
    s = make_settings(tmp_path, request_timeout_s=0.1, concurrency=1)
    with TestClient(create_app(s, warm=False)) as c:
        t0 = time.perf_counter()
        assert_error(post(c, {"file": ("a.png", raw, "image/png")}), 504, "timeout")
        first = time.perf_counter() - t0
        t1 = time.perf_counter()
        assert_error(post(c, {"file": ("a.png", raw, "image/png")}), 504, "timeout")
        second = time.perf_counter() - t1
    assert first < 0.5  # the client is answered at the timeout
    assert second > 0.4  # it waited for the first inference to finish before starting
