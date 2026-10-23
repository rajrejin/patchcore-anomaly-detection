"""API with a real trained artifact and real MVTec images (skips if either is absent)."""

from __future__ import annotations

import base64
import io
from collections.abc import Iterator
from pathlib import Path

import pytest
from api.main import create_app
from fastapi.testclient import TestClient
from PIL import Image

from inspection.config import load_settings


@pytest.fixture(scope="module")
def client(bottle_artifact: Path, data_root: Path) -> Iterator[TestClient]:
    with TestClient(create_app(load_settings(), warm=False)) as c:
        yield c


def _first(data_root: Path, sub: str) -> Path:
    return sorted((data_root / "bottle" / "test" / sub).glob("*.png"))[0]


def test_inspect_good_and_defective(client: TestClient, data_root: Path) -> None:
    good = client.post(
        "/v1/inspect?category=bottle",
        files={"file": ("g.png", _first(data_root, "good").read_bytes(), "image/png")},
    )
    bad = client.post(
        "/v1/inspect?category=bottle",
        files={"file": ("b.png", _first(data_root, "broken_large").read_bytes(), "image/png")},
    )
    assert good.status_code == 200 and bad.status_code == 200, (good.text, bad.text)
    g, b = good.json(), bad.json()
    assert not g["is_anomalous"] and b["is_anomalous"]
    assert b["anomaly_score"] > g["anomaly_score"]
    assert b["boxes"] and b["stage_reached"] == 2
    assert b["latency"]["end_to_end_ms"] >= b["latency"]["model_ms"] > 0
    assert len(b["model_version"]) == 12 and b["request_id"]
    img = Image.open(io.BytesIO(base64.b64decode(b["annotated_png_base64"])))
    assert img.size == (224, 224)
    assert b["heatmap_png_base64"]


def test_categories_lists_bottle(client: TestClient) -> None:
    cats = {c["category"]: c for c in client.get("/v1/categories").json()}
    assert "bottle" in cats and cats["bottle"]["threshold"] > 0


def test_cascade_path_when_stage1_available(client: TestClient, data_root: Path) -> None:
    cats = {c["category"]: c for c in client.get("/v1/categories").json()}
    if not cats["bottle"]["cascade_available"]:
        pytest.skip("stage1.npz not built (run: mvinspect cascade bottle)")
    r = client.post(
        "/v1/inspect?category=bottle&use_cascade=true",
        files={"file": ("g.png", _first(data_root, "broken_large").read_bytes(), "image/png")},
    )
    assert r.status_code == 200 and r.json()["cascade_used"] is True
    assert r.json()["stage1_score"] is not None
