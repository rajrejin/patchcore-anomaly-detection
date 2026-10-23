"""Check that the GPU study runs agree with the CPU runs.

1. The GPU baseline (study variant coreset@0.1, seed 0) against the laptop CPU main run; both use
   the same config, split and seed.
2. Robustness settings on leather that were computed on both devices (results/studies_cpu_partial
   holds a CPU run that was stopped after six settings when the studies moved to the GPU).

Writes results/parity_cpu_vs_gpu.json from files in results/ only.
"""

from __future__ import annotations

import json
from pathlib import Path

R = Path(__file__).resolve().parents[1] / "results"


def main() -> None:
    rows = []
    for cat in ("leather", "hazelnut", "screw"):
        cpu = json.loads((R / "categories" / f"{cat}.json").read_text())
        gpu = json.loads((R / "studies" / "ablations" / f"{cat}.json").read_text())["coreset@0.1"]
        rows.append(
            {
                "category": cat,
                "gpu_device": gpu.get("device"),
                "image_auroc_cpu": cpu["image_auroc"],
                "image_auroc_gpu": gpu["image_auroc"],
                "pixel_auroc_cpu": cpu["pixel_auroc"],
                "pixel_auroc_gpu": gpu["pixel_auroc"],
                "image_auroc_abs_diff": abs(cpu["image_auroc"] - gpu["image_auroc"]),
                "pixel_auroc_abs_diff": abs(cpu["pixel_auroc"] - gpu["pixel_auroc"]),
                "bank_rows_cpu": cpu["bank_shape"][0],
                "bank_rows_gpu": gpu["bank_shape"][0],
                "threshold_cpu": cpu["threshold_image"],
                "threshold_gpu": gpu["threshold"],
            }
        )
    overlap: dict[str, object] = {"settings": []}
    cpu_rob = R / "studies_cpu_partial" / "robustness" / "leather.json"
    if cpu_rob.exists():
        c = json.loads(cpu_rob.read_text())
        g = json.loads((R / "studies" / "robustness" / "leather.json").read_text())
        settings = []
        for key in sorted(set(c) & set(g)):
            settings.append(
                {
                    "setting": key,
                    "image_auroc_cpu": c[key]["image_auroc"],
                    "image_auroc_gpu": g[key]["image_auroc"],
                    "fpr_normals_cpu": c[key]["fpr_normals"],
                    "fpr_normals_gpu": g[key]["fpr_normals"],
                }
            )
        overlap = {
            "category": "leather",
            "settings": settings,
            "max_abs_auroc_diff": max(
                abs(x["image_auroc_cpu"] - x["image_auroc_gpu"]) for x in settings
            ),
            "max_abs_fpr_diff": max(
                abs(x["fpr_normals_cpu"] - x["fpr_normals_gpu"]) for x in settings
            ),
        }
    out = {"main_runs": rows, "robustness_overlap": overlap}
    (R / "parity_cpu_vs_gpu.json").write_text(json.dumps(out, indent=2))
    for r in rows:
        print(
            r["category"],
            f"image diff {r['image_auroc_abs_diff']:.2e}",
            f"pixel diff {r['pixel_auroc_abs_diff']:.2e}",
            f"bank {r['bank_rows_cpu']} vs {r['bank_rows_gpu']}",
            f"thr {r['threshold_cpu']:.4f} vs {r['threshold_gpu']:.4f}",
        )
    if overlap["settings"]:
        print(
            f"robustness overlap: {len(overlap['settings'])} settings, "
            f"max AUROC diff {overlap['max_abs_auroc_diff']:.1e}, "
            f"max FPR diff {overlap['max_abs_fpr_diff']:.1e}"
        )


if __name__ == "__main__":
    main()
