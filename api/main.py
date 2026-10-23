"""FastAPI application: versioned inspection routes, validation, observability."""

from __future__ import annotations

import asyncio
import hmac
import logging
import time
import uuid
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, Response
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
from starlette.exceptions import HTTPException as StarletteHTTPException

from api import metrics as pm
from api.schemas import Box, CategoryInfo, InspectResponse, Latency
from api.service import ApiError, ModelService, decode_upload, png_b64
from api.settings import get_settings
from inspection.config import Settings
from inspection.logging_utils import setup_logging

log = logging.getLogger("api")


def _log(msg: str, **fields: object) -> None:
    log.info(msg, extra={"extra_fields": fields})


def create_app(settings: Settings | None = None, warm: bool = True) -> FastAPI:
    cfg = settings or get_settings()
    state: dict[str, object] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        setup_logging(json_logs=True)
        service = ModelService(cfg)
        executor = ThreadPoolExecutor(max_workers=max(1, cfg.api.concurrency))
        state.update(
            service=service, executor=executor, sem=asyncio.Semaphore(max(1, cfg.api.concurrency))
        )
        service.discover()
        if warm:
            await asyncio.get_running_loop().run_in_executor(executor, service.warmup)
        else:
            service.ready = True
        yield
        executor.shutdown(wait=True)  # graceful shutdown: finish in-flight inference

    app = FastAPI(title="MVTec PatchCore inspection", version="1", lifespan=lifespan)
    if cfg.api.cors_origins:
        app.add_middleware(CORSMiddleware, allow_origins=cfg.api.cors_origins, allow_methods=["*"])

    def service() -> ModelService:
        return state["service"]  # type: ignore[return-value]

    def require_key(request: Request) -> None:
        key = cfg.api.api_key
        if key and not hmac.compare_digest(request.headers.get("x-api-key", ""), key):
            raise ApiError(401, "unauthorized", "missing or invalid API key")

    # ---------------------------------------------------------------- middleware
    @app.middleware("http")
    async def request_context(request: Request, call_next):  # type: ignore[no-untyped-def]
        rid = uuid.uuid4().hex
        request.state.request_id = rid
        t0 = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Request-ID"] = rid
        route = request.scope.get("route")
        path = getattr(route, "path", "unmatched")
        pm.REQUESTS.labels(path, str(response.status_code)).inc()
        _log(
            "request",
            request_id=rid,
            method=request.method,
            path=path,
            status=response.status_code,
            ms=round((time.perf_counter() - t0) * 1000, 1),
        )
        return response

    # ---------------------------------------------------------------- errors
    def err(status: int, code: str, message: str, request: Request) -> JSONResponse:
        pm.ERRORS.labels(code).inc()
        rid = getattr(request.state, "request_id", None)
        return JSONResponse(
            status_code=status,
            content={"error": {"code": code, "message": message, "request_id": rid}},
        )

    @app.exception_handler(ApiError)
    async def _api_error(request: Request, exc: ApiError) -> JSONResponse:
        return err(exc.status, exc.code, exc.message, request)

    @app.exception_handler(RequestValidationError)
    async def _validation(request: Request, exc: RequestValidationError) -> JSONResponse:
        fields = sorted({str(e["loc"][-1]) for e in exc.errors()})
        return err(
            422, "invalid_request", f"invalid or missing parameters: {', '.join(fields)}", request
        )

    @app.exception_handler(StarletteHTTPException)
    async def _http(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        return err(exc.status_code, "http_error", str(exc.detail), request)

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception(
            "unhandled error",
            extra={"extra_fields": {"request_id": getattr(request.state, "request_id", None)}},
        )
        return err(500, "internal_error", "an internal error occurred", request)

    # ---------------------------------------------------------------- routes
    @app.get("/v1/health")
    async def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/v1/ready")
    async def ready() -> JSONResponse:
        svc = service()
        ok = svc.ready and bool(svc.infos)
        return JSONResponse(
            status_code=200 if ok else 503,
            content={
                "ready": ok,
                "categories": len(svc.infos),
                "loaded": svc.loaded(),
                "backends": svc.backends,
            },
        )

    @app.get(
        "/v1/categories", response_model=list[CategoryInfo], dependencies=[Depends(require_key)]
    )
    async def categories() -> list[CategoryInfo]:
        svc = service()
        svc.discover()
        return [CategoryInfo(**v) for v in svc.infos.values()]

    @app.post("/v1/inspect", response_model=InspectResponse, dependencies=[Depends(require_key)])
    async def inspect(
        request: Request,
        category: str = Query(..., max_length=32),
        use_cascade: bool = Query(cfg.api.use_cascade_default),
        file: UploadFile = File(...),
    ) -> InspectResponse:
        t_req = time.perf_counter()
        svc = service()
        if category not in svc.infos:  # whitelist: never used in a path before this check
            from inspection.data import CATEGORIES

            if category not in CATEGORIES:
                raise ApiError(404, "unknown_category", "unknown category")
            raise ApiError(404, "category_not_trained", "no trained artifact for this category")
        data = await file.read(cfg.api.max_upload_bytes + 1)
        img = decode_upload(data, cfg)
        sem: asyncio.Semaphore = state["sem"]  # type: ignore[assignment]
        executor: ThreadPoolExecutor = state["executor"]  # type: ignore[assignment]
        t_wait = time.perf_counter()
        await sem.acquire()
        pm.QUEUE_WAIT.observe(time.perf_counter() - t_wait)
        fut = asyncio.get_running_loop().run_in_executor(
            executor, svc.inspect, category, img, use_cascade
        )

        # the slot is released when the worker thread finishes, not when the request gives up,
        # so a timed-out inference still counts against the concurrency limit while it runs
        def _release(f: asyncio.Future) -> None:  # type: ignore[type-arg]
            sem.release()
            if not f.cancelled():
                f.exception()  # mark a late failure as retrieved after a timeout

        fut.add_done_callback(_release)
        try:
            out = await asyncio.wait_for(asyncio.shield(fut), timeout=cfg.api.request_timeout_s)
        except asyncio.TimeoutError as exc:
            raise ApiError(504, "timeout", "inspection timed out") from exc
        total_ms = (time.perf_counter() - t_req) * 1000
        stage = str(out["stage_reached"])
        pm.LATENCY.labels(category, stage).observe(total_ms / 1000)
        pm.INSPECTIONS.labels(category, stage, "anomalous" if out["is_anomalous"] else "ok").inc()
        return InspectResponse(
            request_id=request.state.request_id,
            category=category,
            anomaly_score=out["anomaly_score"],
            threshold=out["threshold"],
            is_anomalous=out["is_anomalous"],
            stage_reached=out["stage_reached"],
            cascade_used=use_cascade,
            stage1_score=out["stage1_score"],
            latency=Latency(model_ms=out["model_ms"], end_to_end_ms=total_ms),
            boxes=[Box(x=b[0], y=b[1], w=b[2], h=b[3], score=b[4]) for b in out["boxes"]],
            box_frame_px=cfg.data.crop,
            heatmap_png_base64=png_b64(out["heatmap"]) if out["heatmap"] is not None else None,
            annotated_png_base64=png_b64(out["annotated"]),
            model_version=out["model_version"],
        )

    @app.get("/metrics")
    async def metrics() -> Response:
        return Response(generate_latest(pm.REGISTRY), media_type=CONTENT_TYPE_LATEST)

    return app


app = create_app()
