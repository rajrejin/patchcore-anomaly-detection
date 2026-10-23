"""Build the Colab bundle: colab/mvtec-inspection.zip and notebooks/colab_studies.ipynb.

The zip holds code, configs, tests and the small result files, never data, caches or artifacts.
The notebook runs scripts/colab_run.py, the same script that produced the reported study results.
Usage: python scripts/make_colab_bundle.py
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE_DIRS = ["src", "configs", "tests", "scripts", "api", "app"]
INCLUDE_FILES = ["pyproject.toml", "requirements.txt", "README.md"]
SKIP_PARTS = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache"}


def build_zip() -> Path:
    out = ROOT / "colab" / "mvtec-inspection.zip"
    out.parent.mkdir(exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for d in INCLUDE_DIRS:
            for p in sorted((ROOT / d).rglob("*")):
                if p.is_file() and not SKIP_PARTS & set(p.parts) and p.suffix != ".pyc":
                    z.write(p, p.relative_to(ROOT).as_posix())
        for f in INCLUDE_FILES:
            z.write(ROOT / f, f)
        for p in sorted((ROOT / "results" / "categories").glob("*.json")):
            z.write(p, p.relative_to(ROOT).as_posix())
    return out


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.strip().splitlines(True)}


def code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "execution_count": None,
        "outputs": [],
        "source": text.strip().splitlines(True),
    }


CELLS = [
    md(
        """
# MVTec PatchCore studies on a Colab GPU

Runs the ablation, few-shot and robustness studies for leather, hazelnut and screw on a GPU with
`scripts/colab_run.py`, the script that produced the reported study results. The code is the same
as on the laptop; only `MVI_DEVICE=cuda` differs. Latency is not measured here; the laptop measures
it afterwards.

1. Runtime > Change runtime type > T4 GPU.
2. Upload `mvtec-inspection.zip` (built by `python scripts/make_colab_bundle.py`) with the Files
   panel on the left, into `/content`.
3. Run the cells. The script stops before any study if the unit tests (including the CUDA tests)
   or the CPU versus GPU parity check fail. It is resumable: after a disconnect, run it again.
"""
    ),
    code(
        """
import zipfile
zipfile.ZipFile("/content/mvtec-inspection.zip").extractall("/content/work/mvtec-inspection")
%cd /content/work/mvtec-inspection
!python scripts/colab_run.py
"""
    ),
    md("Download the study results, then put `studies/` into `results/` on the laptop."),
    code(
        """
from google.colab import files
files.download("/content/studies_results.zip")
"""
    ),
]


def build_notebook() -> Path:
    nb = {
        "cells": CELLS,
        "metadata": {
            "accelerator": "GPU",
            "colab": {"provenance": []},
            "kernelspec": {"display_name": "Python 3", "name": "python3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = ROOT / "notebooks" / "colab_studies.ipynb"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(nb, indent=1), encoding="utf-8")
    return out


if __name__ == "__main__":
    print(build_zip())
    print(build_notebook())
