"""Render README.md and docs/MODEL_CARD.md from their templates and results/ files.

Markers in templates:
  <!--TABLE:name-->  replaced by a markdown table built from results/
  {{key}}            replaced by a value computed from results/
Every number that appears through this script therefore traces to a results file.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"


INT_COLUMNS = {"flagged defective images", "q", "requests", "errors", "concurrency"}
HEADERS = {
    "pixel_auroc": "pixel AUROC",
    "au_pro": "AU-PRO",
    "image_auroc": "image AUROC",
    "auroc_ci_lo": "AUROC CI low",
    "auroc_ci_hi": "AUROC CI high",
}


def md(df: pd.DataFrame, floatfmt: str = "{:.3f}") -> str:
    cols = list(df.columns)
    heads = [HEADERS.get(c, str(c).replace("_", " ")) for c in cols]
    out = ["| " + " | ".join(heads) + " |", "|" + "|".join("---" for _ in cols) + "|"]
    for _, row in df.iterrows():
        cells = []
        for c, v in zip(cols, row.values, strict=True):
            if isinstance(v, float) and pd.isna(v):
                cells.append("")
            elif c in INT_COLUMNS and isinstance(v, (int, float)) and float(v).is_integer():
                cells.append(str(int(v)))
            elif isinstance(v, float):
                cells.append(floatfmt.format(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def csv(name: str) -> pd.DataFrame | None:
    p = R / name
    return pd.read_csv(p) if p.exists() else None


def t_per_category() -> str:
    d = csv("metrics_per_category.csv")
    if d is None:
        return "_not available_"
    d = d.copy()
    d["image AUROC (95% CI)"] = d.apply(
        lambda r: f"{r.image_auroc:.3f} ({r.auroc_ci_lo:.3f} to {r.auroc_ci_hi:.3f})", axis=1
    )
    cols = ["category", "image AUROC (95% CI)", "pixel_auroc", "au_pro", "precision", "recall"]
    cols += ["f1", "fpr_normals", "oracle_f1"]
    return md(d[cols].rename(columns={"fpr_normals": "FPR on normals", "oracle_f1": "oracle F1"}))


def t_latency() -> str:
    d = csv("metrics_per_category.csv")
    if d is None:
        return "_not available_"
    cols = ["category", "p50_ms", "p95_ms", "fps", "rss_after_load_mb", "peak_rss_mb", "bank_mb"]
    return md(d[cols], "{:.1f}")


def t_baselines() -> str:
    d = csv("comparison_published.csv")
    return md(d, "{:.2f}") if d is not None else "_not available_"


def t_postprocess() -> str:
    rows = []
    for f in sorted((R / "postprocess").glob("*.json")):
        d = json.loads(f.read_text())
        a = d["flagged_defective"]
        rows.append(
            {
                "category": d["category"],
                "flagged defective images": a["n_images"],
                "IoU raw": a["raw_mean_iou"],
                "IoU cleaned": a["clean_mean_iou"],
                "Dice raw": a["raw_mean_dice"],
                "Dice cleaned": a["clean_mean_dice"],
                "box hit rate": a["box_hit_rate_images"],
                "box precision": a["box_precision"],
                "boxes per image": a["boxes_per_image"],
            }
        )
    if not rows:
        return "_not available_"
    df = pd.DataFrame(rows)
    mean = df.drop(columns="category").mean()
    df = pd.DataFrame([*df.to_dict("records"), {"category": "MEAN", **mean.to_dict()}])
    return md(df)


def t_alpha() -> str:
    rows = []
    for f in sorted((R / "postprocess").glob("*.json")):
        d = json.loads(f.read_text())
        for s in d["alpha_sensitivity"]:
            rows.append({"category": d["category"], **s})
    if not rows:
        return "_not available_"
    df = pd.DataFrame(rows).drop(columns="category").groupby("alpha").mean().reset_index()
    return md(
        df.rename(
            columns={
                "mean_iou_defective": "mean IoU (defective)",
                "mean_dice_defective": "mean Dice (defective)",
                "normal_pixel_fp_fraction": "pixels flagged on normals",
                "normal_images_with_any_pixel": "normal images with any flagged pixel",
            }
        )
    )


def t_cascade() -> str:
    d = csv("cascade_tradeoff.csv")
    if d is None:
        return "_not available_"
    m = d[d.category == "MEAN"].drop(columns="category")
    return md(m)


def t_cascade_cat() -> str:
    d = csv("cascade_tradeoff.csv")
    if d is None:
        return "_not available_"
    d = d[(d.category != "MEAN") & (d.q == 50)][
        ["category", "normal_skipped", "defective_wrongly_skipped", "f1", "patchcore_f1", "speedup"]
    ]
    return md(d)


def t_mix() -> str:
    d = csv("cascade_production_mix_calculation.csv")
    return md(d) if d is not None else "_not available_"


def t_ablation() -> str:
    d = csv("ablations.csv")
    if d is None:
        return "_not available_"
    d = d[d.study.isin(["baseline", "preprocessing", "backbone"])].copy()
    d["variant"] = d.variant.str.replace(
        "baseline_seed0 (main run)", "baseline (main run)", regex=False
    )
    cols = ["category", "variant", "image_auroc", "pixel_auroc", "fpr_normals"]
    return md(d[cols], "{:.4f}")


def t_seed_spread() -> str:
    d = csv("ablation_seed_spread.csv")
    return md(d, "{:.4f}") if d is not None else "_not available_"


def t_fewshot() -> str:
    d = csv("fewshot_summary.csv")
    return md(d.drop(columns=["n_numeric"]), "{:.4f}") if d is not None else "_not available_"


def t_robustness() -> str:
    d = csv("robustness.csv")
    if d is None:
        return "_not available_"
    d = d.copy()
    d["cell"] = d.apply(
        lambda r: f"{r.image_auroc:.3f} ({r.auroc_change:+.3f}) / FPR {r.fpr_normals:.2f}", axis=1
    )
    p = d.pivot_table(
        index=["perturbation", "severity"],
        columns="category",
        values="cell",
        aggfunc="first",
        sort=False,
    ).reset_index()
    return md(p)


def t_runtime() -> str:
    f = R / "runtime.json"
    if not f.exists():
        return "_not available_"
    d = json.loads(f.read_text())
    df = pd.DataFrame(d).fillna("")
    df.index.name = "category"
    return md(df.reset_index(), "{:.0f}")


def t_failures() -> str:
    f = R / "failure_analysis.json"
    if not f.exists():
        return "_not available_"
    d = json.loads(f.read_text())
    rows = [
        {
            "category": c,
            "image AUROC": v["image_auroc"],
            "false negatives": v["n_false_negatives"],
            "false positives": v["n_false_positives"],
            "missed defect types": ", ".join(
                f"{k} ({n})" for k, n in v["missed_defect_types"].items()
            ),
        }
        for c, v in d["categories"].items()
    ]
    return md(pd.DataFrame(rows))


def t_loadtest() -> str:
    rows = []
    for f in sorted(R.glob("load_test_*.json")):
        d = json.loads(f.read_text())
        for lv in d["levels"]:
            rows.append({"server": d["label"], **lv})
    return md(pd.DataFrame(rows), "{:.1f}") if rows else "_not available_"


def t_benchmark() -> str:
    rows = []
    for f in sorted(R.glob("inference_benchmark*.json")):
        d = json.loads(f.read_text())
        for k, v in d["end_to_end_model_only"].items():
            rows.append(
                {"run": f.stem, "combination": k, "p50 ms": v["p50_ms"], "p95 ms": v["p95_ms"]}
            )
    return md(pd.DataFrame(rows), "{:.0f}") if rows else "_not available_"


def t_parity() -> str:
    f = R / "parity_cpu_vs_gpu.json"
    if not f.exists():
        return "_not available_"
    df = pd.DataFrame(json.loads(f.read_text())["main_runs"])
    keep = ["category", "image_auroc_cpu", "image_auroc_gpu", "pixel_auroc_cpu", "pixel_auroc_gpu"]
    keep += ["image_auroc_abs_diff", "pixel_auroc_abs_diff", "threshold_cpu", "threshold_gpu"]
    out = df[keep].copy()
    for c in ("image_auroc_abs_diff", "pixel_auroc_abs_diff"):
        out[c] = out[c].map(lambda x: f"{x:.1e}")
    return md(out, "{:.4f}")


def t_ratio() -> str:
    d = csv("ablations.csv")
    if d is None:
        return "_not available_"
    r = d[d.study == "coreset_ratio"].copy()
    r["ratio"] = r.variant.str.split("@").str[1]
    cols = ["category", "ratio", "image_auroc", "pixel_auroc", "fpr_normals", "p50_ms", "bank_mb"]
    out = r[cols].rename(columns={"p50_ms": "CPU p50 ms", "bank_mb": "bank MB"}).copy()
    out["CPU p50 ms"] = out["CPU p50 ms"].map(lambda x: f"{x:.0f}")
    out["bank MB"] = out["bank MB"].map(lambda x: f"{x:.1f}")
    return md(out.sort_values(["category", "ratio"]), "{:.3f}")


def t_seeds() -> str:
    d = csv("ablations.csv")
    if d is None:
        return "_not available_"
    s = d[d.study.isin(["baseline", "baseline_seed"])].copy()
    s["seed"] = s.variant.map(lambda v: "0" if v.startswith("baseline_seed0") else v[-1])
    return md(
        s[["category", "seed", "image_auroc", "pixel_auroc", "fpr_normals"]].sort_values(
            ["category", "seed"]
        ),
        "{:.4f}",
    )


def lookup(kind: str, args: list[str]) -> str:
    """Resolve {{rob|category|perturbation|severity|metric}}, {{few|category|n|metric}},
    {{abl|category|variant|metric}} and {{fprrange|category}} from results/ files."""
    if kind == "rob":
        cat, pert, sev, metric = args
        d = json.loads((R / "studies" / "robustness" / f"{cat}.json").read_text())
        return f"{d[f'{pert}:{sev}'][metric]:.3f}"
    if kind == "few":
        cat, n, metric = args
        d = csv("fewshot_summary.csv")
        assert d is not None
        row = d[(d.category == cat) & (d.n.astype(str) == n)].iloc[0]
        return f"{row[metric]:.3f}"
    if kind == "abl":
        cat, variant, metric = args
        d = csv("ablations.csv")
        assert d is not None
        row = d[(d.category == cat) & (d.variant == variant)].iloc[0]
        return f"{row[metric]:.3f}"
    if kind == "cat":
        cat, metric = args
        d = csv("metrics_per_category.csv")
        assert d is not None
        return f"{d[d.category == cat].iloc[0][metric]:.3f}"
    if kind == "spread":
        cat, metric = args
        d = csv("ablation_seed_spread.csv")
        assert d is not None
        return f"{d[d.category == cat].iloc[0][metric]:.4f}"
    if kind == "fprrange":
        d = csv("ablations.csv")
        assert d is not None
        s = d[(d.category == args[0]) & d.study.isin(["baseline", "baseline_seed"])].fpr_normals
        return f"{s.min():.2f} to {s.max():.2f}"
    raise KeyError(kind)


TABLES = {k[2:]: v for k, v in globals().items() if k.startswith("t_") and callable(v)}


def values() -> dict[str, str]:
    v: dict[str, str] = {}
    d = csv("metrics_per_category.csv")
    if d is not None:
        mean = d[d.category == "MEAN"].iloc[0]
        cats = d[d.category != "MEAN"]
        v["mean_image_auroc"] = f"{mean.image_auroc:.3f}"
        v["mean_pixel_auroc"] = f"{mean.pixel_auroc:.3f}"
        v["mean_au_pro"] = f"{mean.au_pro:.3f}"
        v["mean_f1"] = f"{mean.f1:.3f}"
        v["mean_oracle_f1"] = f"{mean.oracle_f1:.3f}"
        v["mean_fpr"] = f"{mean.fpr_normals:.3f}"
        v["mean_p50_ms"] = f"{mean.p50_ms:.0f}"
        v["mean_fps"] = f"{mean.fps:.2f}"
        v["n_categories"] = str(len(cats))
        low = cats.nsmallest(3, "image_auroc")
        v["weakest"] = ", ".join(f"{r.category} ({r.image_auroc:.3f})" for r in low.itertuples())
        top = cats[cats.image_auroc >= 0.999].category.tolist()
        v["perfect"] = ", ".join(top)
        v["max_peak_rss_mb"] = f"{cats.peak_rss_mb.max():.0f}"
        v["max_bank_mb"] = f"{cats.bank_mb.max():.0f}"
        v["min_bank_mb"] = f"{cats.bank_mb.min():.0f}"
    env = R / "environment.json"
    if env.exists():
        e = json.loads(env.read_text())
        v["cpu"] = e["cpu"].strip()
        v["ram_gb"] = str(e["ram_gb"])
        v["cores"] = f"{e['physical_cores']} physical, {e['logical_cores']} logical"
        v["python"] = e["python"]
        v["torch"] = e["libraries"]["torch"]
    rt = R / "runtime.json"
    if rt.exists():
        j = json.loads(rt.read_text())
        tr = j.get("train_seconds", {})
        v["train_total_min"] = f"{sum(tr.values()) / 60:.0f}"
        ev = j.get("evaluate_seconds", {})
        v["eval_total_min"] = f"{sum(ev.values()) / 60:.0f}"
    c = csv("cascade_tradeoff.csv")
    if c is not None:
        m = c[c.category == "MEAN"].set_index("q")
        for q in (25, 50, 75):
            v[f"casc{q}_normal_skipped"] = f"{m.loc[q, 'normal_skipped']:.3f}"
            v[f"casc{q}_def_skipped"] = f"{m.loc[q, 'defective_wrongly_skipped']:.3f}"
            v[f"casc{q}_speedup"] = f"{m.loc[q, 'speedup']:.2f}"
            v[f"casc{q}_f1"] = f"{m.loc[q, 'f1']:.3f}"
    cmp_ = csv("comparison_published.csv")
    if cmp_ is not None:
        m = cmp_[cmp_.category == "MEAN"].iloc[0]
        v["cmp_image_delta"] = f"{m.image_delta:+.2f}"
        v["cmp_pixel_delta"] = f"{m.pixel_delta:+.2f}"
        rest = cmp_[cmp_.category != "MEAN"].nsmallest(3, "image_delta")
        v["cmp_worst"] = ", ".join(
            f"{r.category} ({r.image_delta:+.2f})" for r in rest.itertuples()
        )
        v["cmp_within1"] = str(int((cmp_[cmp_.category != "MEAN"].image_delta.abs() < 1.0).sum()))
    par = R / "parity_cpu_vs_gpu.json"
    if par.exists():
        ov = json.loads(par.read_text()).get("robustness_overlap", {})
        if ov.get("settings"):
            v["parity_rob_n"] = str(len(ov["settings"]))
            v["parity_rob_diff"] = f"{ov['max_abs_auroc_diff']:.1e}"
    dv = R / "docker_verification.json"
    if dv.exists():
        d = json.loads(dv.read_text())
        sizes = d["image_bytes"]
        v["docker_api_gb"] = f"{sizes['mvtec-inspection-api:latest'] / 1e9:.2f}"
        v["docker_app_mb"] = f"{sizes['mvtec-inspection-app:latest'] / 1e6:.0f}"
        v["docker_api_mem"] = [s.split(" ", 1)[1] for s in d["stats"] if "api" in s][0]
        v["docker_version"] = d["docker_version"].replace("Docker version ", "").split(",")[0]
        v["docker_score"] = f"{d['inspections']['defective']['anomaly_score']:.4f}"
    pp = [json.loads(f.read_text()) for f in sorted((R / "postprocess").glob("*.json"))]
    if pp:

        def avg(key: str) -> float:
            return sum(p["flagged_defective"][key] for p in pp) / len(pp)

        v["pp_raw_iou"] = f"{avg('raw_mean_iou'):.3f}"
        v["pp_clean_iou"] = f"{avg('clean_mean_iou'):.3f}"
        v["pp_raw_dice"] = f"{avg('raw_mean_dice'):.3f}"
        v["pp_clean_dice"] = f"{avg('clean_mean_dice'):.3f}"
        v["pp_hit"] = f"{avg('box_hit_rate_images'):.3f}"
        v["pp_boxprec"] = f"{avg('box_precision'):.3f}"
    return v


def render(template: Path, out: Path) -> None:
    text = template.read_text(encoding="utf-8")
    vals = values()
    text = re.sub(
        r"<!--TABLE:(\w+)-->", lambda m: TABLES[m.group(1)](), text
    )  # unknown table names raise KeyError
    text = re.sub(
        r"\{\{(rob|few|abl|cat|spread|fprrange)\|([^}]+)\}\}",
        lambda m: lookup(m.group(1), m.group(2).split("|")),
        text,
    )
    missing = set(re.findall(r"\{\{(\w+)\}\}", text)) - set(vals)
    if missing:
        raise SystemExit(f"template uses undefined values: {sorted(missing)}")
    text = re.sub(r"\{\{(\w+)\}\}", lambda m: vals[m.group(1)], text)
    out.write_text(text, encoding="utf-8")
    print("wrote", out)


if __name__ == "__main__":
    render(ROOT / "docs" / "README.template.md", ROOT / "README.md")
    render(ROOT / "docs" / "MODEL_CARD.template.md", ROOT / "docs" / "MODEL_CARD.md")
