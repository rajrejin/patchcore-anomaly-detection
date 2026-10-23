"""Typed configuration loaded from YAML with environment variable overrides."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "configs" / "default.yaml"
_ACTIVE_CONFIG: list[Path] = [Path(os.environ.get("MVI_CONFIG", DEFAULT_CONFIG))]


class PathsCfg(BaseModel):
    data_dir: str = "data/mvtec_ad"
    raw_dir: str = "data/raw"
    cache_dir: str = "cache"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    logs_dir: str = "logs"


class DataCfg(BaseModel):
    calib_fraction: float = 0.1
    min_calib: int = 10
    resize: int = 256
    crop: int = 224


class BackboneCfg(BaseModel):
    name: str = "resnet50"
    weights: str = "imagenet"
    layers: list[str] = ["layer2", "layer3"]


class PatchCoreCfg(BaseModel):
    pool_kernel: int = 3
    reduce_dim: int | None = None
    coreset_ratio: float = 0.1
    coreset_proj_dim: int = 128
    min_bank: int = 1000
    knn_backend: str = "torch"
    map_sigma: float = 4.0
    batch_size: int = 16
    extractor_backend: str = "torch"


class PostCfg(BaseModel):
    morph_kernel: int = 5
    min_area: int = 30
    box_iou_hit: float = 0.1


class CascadeCfg(BaseModel):
    backbone: str = "resnet18"
    input_size: int = 128
    quantiles: list[float] = [0, 25, 50, 75]
    defect_rates: list[float] = [0.01, 0.05, 0.20]
    serve_q: float = 50  # chosen a priori for the API, never tuned on test labels


class LatencyCfg(BaseModel):
    warmup: int = 5
    max_images: int = 60


class ApiCfg(BaseModel):
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pixels: int = 25_000_000
    min_side: int = 32
    max_side: int = 8192
    api_key: str | None = None
    cors_origins: list[str] = []
    concurrency: int = 1
    lru_categories: int = 2
    request_timeout_s: float = 60.0
    use_cascade_default: bool = False
    # chosen by measurement (results/inference_benchmark*.json); fall back to torch if unavailable
    knn_backend: str = "faiss"
    extractor_backend: str = "onnx"


class Settings(BaseSettings):
    """All hyperparameters. Environment overrides use the MVI_ prefix and __ nesting."""

    model_config = SettingsConfigDict(env_prefix="MVI_", env_nested_delimiter="__", extra="ignore")

    seed: int = 0
    threads: int = 4
    device: str = "cpu"  # cpu | cuda | auto. cpu is the reference; cuda only speeds up studies
    paths: PathsCfg = PathsCfg()
    data: DataCfg = DataCfg()
    backbone: BackboneCfg = BackboneCfg()
    patchcore: PatchCoreCfg = PatchCoreCfg()
    postprocess: PostCfg = PostCfg()
    cascade: CascadeCfg = CascadeCfg()
    latency: LatencyCfg = LatencyCfg()
    api: ApiCfg = ApiCfg()

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        from pydantic_settings import YamlConfigSettingsSource

        path = _ACTIVE_CONFIG[0]
        sources: list[PydanticBaseSettingsSource] = [init_settings, env_settings]
        if path.exists():
            sources.append(YamlConfigSettingsSource(settings_cls, yaml_file=path))
        return tuple(sources)

    def resolve_device(self) -> str:
        """'auto' picks cuda when available; anything else is used as given."""
        if self.device == "auto":
            import torch

            return "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def path(self, name: str) -> Path:
        """Resolve a configured directory against the project root."""
        p = Path(getattr(self.paths, name))
        return p if p.is_absolute() else PROJECT_ROOT / p

    def model_config_dict(self) -> dict[str, Any]:
        """The subset of settings that defines a trained model (stored in manifests)."""
        return {
            "backbone": self.backbone.model_dump(),
            "patchcore": self.patchcore.model_dump(),
            "data": {"resize": self.data.resize, "crop": self.data.crop},
        }

    def config_hash(self) -> str:
        # serve_q only affects the API cascade, not a trained model; keeping it out of the hash
        # leaves artifacts trained before it was added valid
        dump = self.model_dump(
            exclude={
                "device": True,
                "cascade": {"serve_q"},
                "api": {"knn_backend", "extractor_backend"},
            }
        )
        blob = json.dumps(dump, sort_keys=True).encode()
        return hashlib.sha256(blob).hexdigest()[:16]


def load_settings(path: str | Path | None = None) -> Settings:
    if path is not None:
        _ACTIVE_CONFIG[0] = Path(path)
    return Settings()


def load_yaml(path: Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    return data if isinstance(data, dict) else {}
