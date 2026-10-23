"""HTTP client for the inspection API. The app talks to the service only through this module."""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import Any

import httpx
from PIL import Image


class ApiProblem(Exception):
    """A readable problem to show in the UI."""


@dataclass
class Verdict:
    raw: dict[str, Any]

    @property
    def annotated(self) -> Image.Image:
        return decode_png(self.raw["annotated_png_base64"])

    @property
    def heatmap(self) -> Image.Image | None:
        b64 = self.raw.get("heatmap_png_base64")
        return decode_png(b64) if b64 else None


def decode_png(b64: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")


def _headers(api_key: str | None) -> dict[str, str]:
    return {"X-API-Key": api_key} if api_key else {}


def _check(resp: httpx.Response) -> Any:
    if resp.status_code >= 400:
        try:
            err = resp.json()["error"]
            raise ApiProblem(f"{err['code']}: {err['message']}")
        except (ValueError, KeyError, TypeError):
            raise ApiProblem(f"HTTP {resp.status_code} from the API") from None
    return resp.json()


def list_categories(
    base_url: str, api_key: str | None = None, http: httpx.Client | None = None
) -> list[dict[str, Any]]:
    try:
        c = http or httpx.Client(base_url=base_url, timeout=10)
        return list(_check(c.get("/v1/categories", headers=_headers(api_key))))
    except httpx.HTTPError as exc:
        raise ApiProblem(f"cannot reach the API at {base_url}: {type(exc).__name__}") from exc


def inspect_image(
    base_url: str,
    category: str,
    image_bytes: bytes,
    filename: str = "image.png",
    use_cascade: bool = False,
    api_key: str | None = None,
    http: httpx.Client | None = None,
) -> Verdict:
    try:
        c = http or httpx.Client(base_url=base_url, timeout=120)
        resp = c.post(
            "/v1/inspect",
            params={"category": category, "use_cascade": str(use_cascade).lower()},
            files={"file": (filename, image_bytes, "application/octet-stream")},
            headers=_headers(api_key),
        )
        return Verdict(_check(resp))
    except httpx.HTTPError as exc:
        raise ApiProblem(f"cannot reach the API at {base_url}: {type(exc).__name__}") from exc


def history_row(category: str, filename: str, v: Verdict) -> dict[str, Any]:
    r = v.raw
    return {
        "file": filename,
        "category": category,
        "verdict": "ANOMALOUS" if r["is_anomalous"] else "OK",
        "score": r["anomaly_score"],
        "threshold": r["threshold"],
        "stage": r["stage_reached"],
        "model_ms": round(r["latency"]["model_ms"], 1),
        "end_to_end_ms": round(r["latency"]["end_to_end_ms"], 1),
    }
