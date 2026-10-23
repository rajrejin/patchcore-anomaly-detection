from __future__ import annotations

from pydantic import BaseModel


class Box(BaseModel):
    x: int
    y: int
    w: int
    h: int
    score: float


class Latency(BaseModel):
    model_ms: float
    end_to_end_ms: float


class InspectResponse(BaseModel):
    request_id: str
    category: str
    anomaly_score: float | None
    threshold: float | None
    is_anomalous: bool
    stage_reached: int
    cascade_used: bool
    stage1_score: float | None
    latency: Latency
    boxes: list[Box]
    box_frame_px: int
    heatmap_png_base64: str | None
    annotated_png_base64: str
    model_version: str


class CategoryInfo(BaseModel):
    category: str
    threshold: float
    model_version: str
    created_at: str
    bank_shape: list[int]
    cascade_available: bool


class ErrorBody(BaseModel):
    code: str
    message: str
    request_id: str | None = None


class ErrorResponse(BaseModel):
    error: ErrorBody
