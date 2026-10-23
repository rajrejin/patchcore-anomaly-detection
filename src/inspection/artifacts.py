"""Per-category model artifacts: memory bank, thresholds and a verified manifest."""

from __future__ import annotations

import hashlib
import json
import platform
import time
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
import torch

from inspection.config import PROJECT_ROOT

REQUIRED_KEYS = (
    "category", "config_hash", "seed", "backbone", "model_config", "dataset_fingerprint",
    "git_commit", "library_versions", "created_at", "bank_shape", "bank_sha256", "thresholds",
)  # fmt: skip


class ArtifactError(RuntimeError):
    """Raised when an artifact is missing, corrupt or inconsistent."""


def git_commit(root: Path = PROJECT_ROOT) -> str:
    """Read the current commit hash from .git without needing the git executable."""
    head = root / ".git" / "HEAD"
    try:
        text = head.read_text().strip()
        if text.startswith("ref:"):
            ref = root / ".git" / text.split(" ", 1)[1]
            if ref.exists():
                return ref.read_text().strip()
            packed = root / ".git" / "packed-refs"
            if packed.exists():
                for line in packed.read_text().splitlines():
                    if line.endswith(text.split(" ", 1)[1]):
                        return line.split()[0]
            return "unknown"
        return text
    except OSError:
        return "unknown"


def library_versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for lib in ("torch", "torchvision", "numpy", "scipy", "scikit-learn", "opencv-python-headless"):
        try:
            out[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            out[lib] = "missing"
    return out


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while block := fh.read(1 << 22):
            h.update(block)
    return h.hexdigest()


def manifest_version(manifest: dict[str, Any]) -> str:
    """Model version: short hash over the immutable manifest content."""
    blob = json.dumps(manifest, sort_keys=True).encode()
    return hashlib.sha256(blob).hexdigest()[:12]


def save_artifact(
    out_dir: Path,
    bank: torch.Tensor,
    fields: dict[str, Any],
) -> dict[str, Any]:
    out_dir.mkdir(parents=True, exist_ok=True)
    bank_path = out_dir / "bank.npy"
    np.save(bank_path, bank.float().numpy())
    manifest = dict(fields)
    manifest.update(
        {
            "git_commit": git_commit(),
            "library_versions": library_versions(),
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "bank_shape": list(bank.shape),
            "bank_sha256": sha256_file(bank_path),
        }
    )
    (out_dir / "thresholds.json").write_text(json.dumps(manifest["thresholds"], indent=2))
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def load_artifact(art_dir: Path, verify_hash: bool = True) -> tuple[torch.Tensor, dict[str, Any]]:
    """Load and validate. Refuses a missing or inconsistent artifact."""
    mpath, bpath, tpath = (
        art_dir / "manifest.json",
        art_dir / "bank.npy",
        art_dir / "thresholds.json",
    )
    for p in (mpath, bpath, tpath):
        if not p.exists():
            raise ArtifactError(f"artifact file missing: {p}")
    try:
        manifest = json.loads(mpath.read_text())
        thresholds = json.loads(tpath.read_text())
    except json.JSONDecodeError as exc:
        raise ArtifactError(f"unreadable manifest or thresholds in {art_dir}") from exc
    absent = [k for k in REQUIRED_KEYS if k not in manifest]
    if absent:
        raise ArtifactError(f"manifest missing keys {absent} in {art_dir}")
    if manifest["thresholds"] != thresholds:
        raise ArtifactError(f"thresholds.json disagrees with manifest in {art_dir}")
    if manifest["category"] != art_dir.name:
        raise ArtifactError(f"manifest category {manifest['category']} != directory {art_dir.name}")
    if verify_hash and sha256_file(bpath) != manifest["bank_sha256"]:
        raise ArtifactError(f"bank checksum mismatch in {art_dir}")
    bank = torch.from_numpy(np.load(bpath))
    if list(bank.shape) != list(manifest["bank_shape"]):
        raise ArtifactError(f"bank shape {list(bank.shape)} != manifest {manifest['bank_shape']}")
    return bank, manifest
