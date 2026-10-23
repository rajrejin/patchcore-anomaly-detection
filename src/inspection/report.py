"""Build tables and figures from results/ (every number comes from a results file)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from inspection.config import Settings
from inspection.data import CATEGORIES, OBJECTS, test_samples

log = logging.getLogger(__name__)

# Roth et al., CVPR 2022, arXiv 2106.08265, supplementary Tables S1 and S2, coreset 10 percent,
# WideResNet50, 224 x 224. Read via an HTML rendering (ar5iv) through a summarising fetch; the
# column means reproduce the paper's reported averages (pixel 98.15 vs 98.1, image 98.93 vs 99.0).
# Check against the PDF before quoting precisely.
PUBLISHED = {
    # category: (image AUROC, pixel AUROC) in percent
    "bottle": (100.0, 98.6), "cable": (99.4, 98.5), "capsule": (97.8, 98.9),
    "carpet": (98.7, 99.1), "grid": (97.9, 98.7), "hazelnut": (100.0, 98.7),
    "leather": (100.0, 99.3), "metal_nut": (100.0, 98.4), "pill": (96.0, 97.6),
    "screw": (97.0, 99.4), "tile": (98.9, 95.9), "toothbrush": (99.7, 98.7),
    "transistor": (100.0, 96.4), "wood": (99.0, 95.1), "zipper": (99.5, 98.9),
}  # fmt: skip
PUBLISHED_SOURCE = (
    "Roth et al. 2022 arXiv:2106.08265 supplementary S1/S2, PatchCore 10% coreset, "
    "WideResNet50, 224px; different backbone and settings, indicative only"
)


def load_category_results(settings: Settings) -> dict[str, dict[str, Any]]:
    d = settings.path("results_dir") / "categories"
    return {p.stem: json.loads(p.read_text()) for p in sorted(d.glob("*.json"))}


def _md(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines) + "\n"


def per_category_table(settings: Settings, res: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for cat in CATEGORIES:
        r = res.get(cat)
        if r is None:
            continue
        c = r["calibrated"]
        rows.append(
            {
                "category": cat,
                "kind": "object" if cat in OBJECTS else "texture",
                "image_auroc": r["image_auroc"],
                "auroc_ci_lo": r["image_auroc_ci95"][0],
                "auroc_ci_hi": r["image_auroc_ci95"][1],
                "pixel_auroc": r["pixel_auroc"],
                "au_pro": r["au_pro"],
                "precision": c["precision"],
                "recall": c["recall"],
                "f1": c["f1"],
                "fpr_normals": c["fpr"],
                "oracle_f1": r["oracle_best_f1"]["f1"],
                "p50_ms": r["latency"]["p50_ms"],
                "p95_ms": r["latency"]["p95_ms"],
                "fps": r["latency"]["fps"],
                "peak_rss_mb": r["latency"]["peak_rss_mb"],
                "rss_after_load_mb": r["rss_after_load_mb"],
                "bank_mb": r["bank_mb"],
            }
        )
    df = pd.DataFrame(rows)
    if len(df):
        mean = df.drop(columns=["category", "kind"]).mean()
        df = pd.concat(
            [df, pd.DataFrame([{"category": "MEAN", "kind": "", **mean.to_dict()}])],
            ignore_index=True,
        )
    return df


def write_baselines(settings: Settings, df: pd.DataFrame) -> pd.DataFrame:
    out = settings.path("results_dir")
    pub = pd.DataFrame(
        [
            {
                "category": c,
                "published_image_auroc": v[0],
                "published_pixel_auroc": v[1],
                "source": PUBLISHED_SOURCE,
            }
            for c, v in PUBLISHED.items()
        ]
    )
    pub.to_csv(out / "published_baselines.csv", index=False)
    ours = df[df.category != "MEAN"][["category", "image_auroc", "pixel_auroc"]].copy()
    ours["image_auroc"] *= 100
    ours["pixel_auroc"] *= 100
    cmp = ours.merge(
        pub[["category", "published_image_auroc", "published_pixel_auroc"]], on="category"
    )
    cmp["image_delta"] = cmp.image_auroc - cmp.published_image_auroc
    cmp["pixel_delta"] = cmp.pixel_auroc - cmp.published_pixel_auroc
    cmp.loc[len(cmp)] = ["MEAN", *cmp.drop(columns="category").mean().to_list()]
    cmp.round(2).to_csv(out / "comparison_published.csv", index=False)
    return cmp.round(2)


def failure_analysis(settings: Settings, res: dict[str, dict[str, Any]]) -> dict[str, Any]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from inspection.data import geometry_image, load_rgb
    from inspection.postprocess import heatmap_overlay

    weakest = sorted(res, key=lambda c: res[c]["image_auroc"])[:3]
    out: dict[str, Any] = {"weakest_by_image_auroc": weakest, "categories": {}}
    fdir = settings.path("results_dir") / "figures" / "failures"
    fdir.mkdir(parents=True, exist_ok=True)
    for cat in weakest:
        cd = settings.path("cache_dir") / cat
        maps = np.load(cd / "test_maps.npy", mmap_mode="r")
        scores, labels = np.load(cd / "test_scores.npy"), np.load(cd / "test_labels.npy")
        thr = res[cat]["threshold_image"]
        samples = test_samples(settings.path("data_dir"), cat)
        fn = [i for i in np.where(labels == 1)[0] if scores[i] <= thr]
        fp = [i for i in np.where(labels == 0)[0] if scores[i] > thr]
        fn = sorted(fn, key=lambda i: scores[i])[:4]  # lowest-scoring defective = worst misses
        fp = sorted(fp, key=lambda i: -scores[i])[:4]  # highest-scoring normal = worst false alarms
        from collections import Counter

        missed_types = Counter(
            samples[i].defect_type
            for i in range(len(samples))
            if labels[i] == 1 and scores[i] <= thr
        )
        out["categories"][cat] = {
            "image_auroc": res[cat]["image_auroc"],
            "n_false_negatives": int(((labels == 1) & (scores <= thr)).sum()),
            "n_false_positives": int(((labels == 0) & (scores > thr)).sum()),
            "missed_defect_types": dict(missed_types),
            "worst_fn": [samples[i].path.name + "|" + samples[i].defect_type for i in fn],
            "worst_fp": [samples[i].path.name for i in fp],
        }
        cols = max(len(fn), len(fp), 1)
        fig, axes = plt.subplots(2, cols, figsize=(2.6 * cols, 5.4), squeeze=False)
        for row, (idxs, name) in enumerate(((fn, "false negative"), (fp, "false positive"))):
            for c in range(cols):
                ax = axes[row][c]
                ax.axis("off")
                if c < len(idxs):
                    i = idxs[c]
                    rgb = geometry_image(
                        load_rgb(samples[i].path), settings.data.resize, settings.data.crop
                    )
                    ax.imshow(heatmap_overlay(rgb, np.asarray(maps[i]), thr))
                    ax.set_title(
                        f"{name}\n{samples[i].defect_type} s={scores[i]:.2f} t={thr:.2f}",
                        fontsize=7,
                    )
        fig.suptitle(f"{cat}: worst errors at the calibrated threshold", fontsize=9)
        fig.tight_layout()
        fig.savefig(fdir / f"{cat}.png", dpi=100)
        plt.close(fig)
    (settings.path("results_dir") / "failure_analysis.json").write_text(json.dumps(out, indent=2))
    return out


def cascade_tables(settings: Settings) -> None:
    cdir = settings.path("results_dir") / "cascade"
    files = sorted(cdir.glob("*.json"))
    if not files:
        return
    rows: list[dict[str, Any]] = []
    mix: list[dict[str, Any]] = []
    for f in files:
        d = json.loads(f.read_text())
        for r in d["rows"]:
            rows.append(
                {
                    "category": d["category"],
                    "q": r["q"],
                    "normal_skipped": r["normal_skipped"],
                    "defective_wrongly_skipped": r["defective_wrongly_skipped"],
                    "image_auroc": r["image_auroc"],
                    "precision": r["precision"],
                    "recall": r["recall"],
                    "f1": r["f1"],
                    "patchcore_f1": d["patchcore_alone"]["f1"],
                    "mean_latency_ms": r["mean_latency_ms"],
                    "speedup": r["speedup"],
                }
            )
            for rate, m in r["production_mix_calculation"].items():
                mix.append(
                    {"category": d["category"], "q": r["q"], "defect_rate": float(rate), **m}
                )
    df = pd.DataFrame(rows)
    avg = df.drop(columns="category").groupby("q").mean().reset_index()
    avg.insert(0, "category", "MEAN")
    table = pd.DataFrame([*df.to_dict("records"), *avg.to_dict("records")])
    table.to_csv(settings.path("results_dir") / "cascade_tradeoff.csv", index=False)
    mdf = pd.DataFrame(mix)
    mavg = mdf.drop(columns="category").groupby(["q", "defect_rate"]).mean().reset_index()
    mavg.to_csv(
        settings.path("results_dir") / "cascade_production_mix_calculation.csv", index=False
    )

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    for cat, g in df.groupby("category"):
        g = g.sort_values("q")
        ax.plot(g.speedup, g.defective_wrongly_skipped, marker="o", label=cat, alpha=0.7)
    ax.set_xlabel("speed-up vs PatchCore alone (computed from measured stage means)")
    ax.set_ylabel("fraction of defective images wrongly skipped")
    ax.legend(fontsize=6, ncol=2)
    fig.tight_layout()
    fdir = settings.path("results_dir") / "figures"
    fig.savefig(fdir / "cascade_missed_vs_speedup.png", dpi=110)
    plt.close(fig)


def build_report(settings: Settings) -> Path:
    rdir = settings.path("results_dir")
    res = load_category_results(settings)
    df = per_category_table(settings, res)
    df.to_csv(rdir / "metrics_per_category.csv", index=False)
    shown = df.copy()
    for c in shown.columns:
        if c not in ("category", "kind"):
            shown[c] = shown[c].map(lambda v: f"{v:.3f}" if abs(v) < 10 else f"{v:.1f}")
    (rdir / "metrics_per_category.md").write_text(_md(shown))
    if len(df) > 1:
        cmp = write_baselines(settings, df)
        (rdir / "comparison_published.md").write_text(_md(cmp.astype(str)))
        failure_analysis(settings, res)
    cascade_tables(settings)
    from inspection.studies.summaries import summarise_studies

    summarise_studies(settings)
    log.info("report written to %s", rdir)
    return rdir
