"""Turn raw study JSON into CSV tables and figures under results/."""

from __future__ import annotations

import json
import logging
from typing import Any

import numpy as np
import pandas as pd

from inspection.config import Settings

log = logging.getLogger(__name__)


def _load(settings: Settings, study: str) -> dict[str, dict[str, Any]]:
    d = settings.path("results_dir") / "studies" / study
    return (
        {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))} if d.exists() else {}
    )


def _main_results(settings: Settings, cat: str) -> dict[str, Any] | None:
    f = settings.path("results_dir") / "categories" / f"{cat}.json"
    return json.loads(f.read_text()) if f.exists() else None


def ablation_table(settings: Settings) -> None:
    data = _load(settings, "ablations")
    if not data:
        return
    rows = []
    for cat, entries in data.items():
        base = _main_results(settings, cat)
        if base:
            rows.append(
                {
                    "category": cat,
                    "study": "baseline",
                    "variant": "baseline_seed0 (main run)",
                    "image_auroc": base["image_auroc"],
                    "pixel_auroc": base["pixel_auroc"],
                    "p50_ms": base["latency"]["p50_ms"],
                    "bank_mb": base["bank_mb"],
                    "fpr_normals": base["calibrated"]["fpr"],
                }
            )
        for name, r in entries.items():
            if name == "_meta":
                continue
            study = (
                "baseline_seed"
                if name.startswith("baseline_seed")
                else "coreset_ratio"
                if name.startswith("coreset")
                else "preprocessing"
                if name.startswith("pre_")
                else "backbone"
            )
            rows.append(
                {
                    "category": cat,
                    "study": study,
                    "variant": name,
                    "image_auroc": r["image_auroc"],
                    "pixel_auroc": r.get("pixel_auroc"),
                    "p50_ms": r.get("latency", {}).get("p50_ms"),
                    "bank_mb": r["bank_mb"],
                    "fpr_normals": r["calibrated"]["fpr"],
                }
            )
    df = pd.DataFrame(rows)
    out = settings.path("results_dir")
    df.to_csv(out / "ablations.csv", index=False)
    seeds = df[df.study.isin(["baseline", "baseline_seed"])]
    spread = (
        seeds.groupby("category")
        .agg(
            image_auroc_min=("image_auroc", "min"),
            image_auroc_max=("image_auroc", "max"),
            pixel_auroc_min=("pixel_auroc", "min"),
            pixel_auroc_max=("pixel_auroc", "max"),
            n_seeds=("image_auroc", "size"),
        )
        .reset_index()
    )
    spread["image_auroc_range"] = spread.image_auroc_max - spread.image_auroc_min
    spread["pixel_auroc_range"] = spread.pixel_auroc_max - spread.pixel_auroc_min
    spread.to_csv(out / "ablation_seed_spread.csv", index=False)


def fewshot_table(settings: Settings) -> None:
    data = _load(settings, "fewshot")
    if not data:
        return
    rows = []
    for cat, entries in data.items():
        full = _main_results(settings, cat)
        for name, r in entries.items():
            if name == "_meta":
                continue
            rows.append(
                {
                    "category": cat,
                    "n": str(r["n_label"]),
                    "n_numeric": r["n_label"],
                    "seed": r["seed"],
                    "image_auroc": r["image_auroc"],
                    "fpr_normals": r["calibrated"]["fpr"],
                    "bank_size": r["bank_shape"][0],
                }
            )
        abl: dict[str, Any] = _load(settings, "ablations").get(cat) or {}
        full_rows = [(0, full)] + [(s, abl.get(f"baseline_seed{s}")) for s in (1, 2)]
        for seed, fr in full_rows:
            if fr:
                rows.append(
                    {
                        "category": cat,
                        "n": "all",
                        "n_numeric": 10**6,
                        "seed": seed,
                        "image_auroc": fr["image_auroc"],
                        "fpr_normals": fr["calibrated"]["fpr"],
                        "bank_size": fr["bank_shape"][0],
                    }
                )
    df = pd.DataFrame(rows)
    out = settings.path("results_dir")
    df.to_csv(out / "fewshot.csv", index=False)
    agg = (
        df.groupby(["category", "n", "n_numeric"])
        .agg(
            auroc_mean=("image_auroc", "mean"),
            auroc_min=("image_auroc", "min"),
            auroc_max=("image_auroc", "max"),
            fpr_mean=("fpr_normals", "mean"),
        )
        .reset_index()
        .sort_values(["category", "n_numeric"])
    )
    agg.to_csv(out / "fewshot_summary.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for cat, g in agg.groupby("category"):
        x = np.arange(len(g))
        ax.plot(x, g.auroc_mean, marker="o", label=cat)
        ax.fill_between(x, g.auroc_min, g.auroc_max, alpha=0.2)
        ax.set_xticks(x, [str(n) for n in g.n])
    ax.set_xlabel("normal images in the memory bank (N)")
    ax.set_ylabel("image AUROC (mean, min to max over 3 seeds)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "figures" / "fewshot_auroc.png", dpi=110)
    plt.close(fig)


def robustness_table(settings: Settings) -> None:
    data = _load(settings, "robustness")
    if not data:
        return
    rows = []
    for cat, entries in data.items():
        base = entries.get("none:-")
        for key, r in entries.items():
            rows.append(
                {
                    "category": cat,
                    "perturbation": r["perturbation"],
                    "severity": r["severity"],
                    "image_auroc": r["image_auroc"],
                    "auroc_change": r["image_auroc"] - base["image_auroc"] if base else None,
                    "fpr_normals": r["fpr_normals"],
                    "recall": r["recall"],
                }
            )
            _ = key
    df = pd.DataFrame(rows)
    out = settings.path("results_dir")
    df.to_csv(out / "robustness.csv", index=False)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df["setting"] = df.perturbation + " " + df.severity
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    for ax, metric, title in (
        (axes[0], "image_auroc", "image AUROC"),
        (axes[1], "fpr_normals", "false positive rate on normals (fixed threshold)"),
    ):
        p = df.pivot_table(index="setting", columns="category", values=metric, sort=False)
        p.plot.barh(ax=ax, width=0.8, legend=False)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel("")
        ax.invert_yaxis()
        ax.tick_params(labelsize=7)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(out / "figures" / "robustness.png", dpi=110)
    plt.close(fig)


def summarise_studies(settings: Settings) -> None:
    (settings.path("results_dir") / "figures").mkdir(parents=True, exist_ok=True)
    ablation_table(settings)
    fewshot_table(settings)
    robustness_table(settings)
