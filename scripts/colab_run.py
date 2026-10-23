"""Run the GPU studies end to end on Colab (same steps as notebooks/colab_studies.ipynb).

No Google Drive is needed: results stay in /content and are zipped at the end.
Resumable: every step skips finished work, so re-running after a disconnect continues.
Usage (from the unzipped project root): python scripts/colab_run.py
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
ENV = {**os.environ, "MVI_DEVICE": "cuda", "MVI_THREADS": "2", "PYTHONUNBUFFERED": "1"}
CATS = ["leather", "hazelnut", "screw"]


def run(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess[str]:
    print("\n$", " ".join(cmd), flush=True)
    t = time.time()
    p = subprocess.run(cmd, env=ENV, text=True)
    print(f"[{time.time() - t:.0f}s, exit {p.returncode}]", flush=True)
    if check and p.returncode != 0:
        sys.exit(f"FAILED: {' '.join(cmd)}")
    return p


def main() -> None:
    import torch

    assert torch.cuda.is_available(), "No GPU: Runtime > Change runtime type > T4 GPU"
    print("GPU:", torch.cuda.get_device_name(0), "torch", torch.__version__, flush=True)

    run([sys.executable, "-m", "pip", "-q", "install", "pydantic-settings", "typer", "tqdm"])
    run([sys.executable, "-m", "pip", "-q", "install", "psutil", "pyyaml", "pytest", "httpx"])
    run([sys.executable, "-m", "pip", "-q", "install", "opencv-python-headless", "scikit-learn"])
    run([sys.executable, "-m", "pip", "-q", "install", "-e", ".", "--no-deps"])

    print("\n== self-check: unit tests including the CUDA tests ==", flush=True)
    pytest_cmd = [sys.executable, "-m", "pytest", "tests/unit", "-q", "-rs"]
    r = subprocess.run(pytest_cmd, env=ENV, text=True, capture_output=True)
    print(r.stdout[-3000:], r.stderr[-1500:])
    if r.returncode != 0:
        sys.exit("unit tests failed on this machine: not running the studies")
    if "no CUDA device" in r.stdout:
        sys.exit("CUDA tests were skipped: the GPU is not visible to pytest")

    laptop = ROOT / "results_laptop_categories"
    if not laptop.exists():
        shutil.copytree(ROOT / "results" / "categories", laptop)
    run(["mvinspect", "fetch-data", *CATS])
    run(["mvinspect", "train", *CATS])
    run(["mvinspect", "evaluate", *CATS, "--force"])

    print("\n== parity with the laptop (CPU) results ==", flush=True)
    rows = []
    for c in CATS:
        cpu = json.loads((laptop / f"{c}.json").read_text())
        gpu = json.loads((ROOT / "results" / "categories" / f"{c}.json").read_text())
        rows.append(
            {
                "category": c,
                "image_auroc_cpu": cpu["image_auroc"],
                "image_auroc_gpu": gpu["image_auroc"],
                "pixel_auroc_cpu": cpu["pixel_auroc"],
                "pixel_auroc_gpu": gpu["pixel_auroc"],
            }
        )
        print(rows[-1], flush=True)
        if abs(cpu["image_auroc"] - gpu["image_auroc"]) >= 0.02:
            sys.exit(f"image AUROC parity failed for {c}")
        if abs(cpu["pixel_auroc"] - gpu["pixel_auroc"]) >= 0.01:
            sys.exit(f"pixel AUROC parity failed for {c}")
    (ROOT / "results" / "parity_cpu_vs_gpu.json").write_text(json.dumps(rows, indent=2))

    run(["mvinspect", "studies", "robustness", *CATS])
    run(["mvinspect", "studies", "fewshot", *CATS])
    run(["mvinspect", "ablate", *CATS])

    out = shutil.make_archive("/content/studies_results", "zip", ROOT / "results", "studies")
    shutil.copy(ROOT / "results" / "parity_cpu_vs_gpu.json", "/content/parity_cpu_vs_gpu.json")
    print("\nALL DONE:", out, Path(out).stat().st_size, "bytes", flush=True)


if __name__ == "__main__":
    main()
