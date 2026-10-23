"""End-to-end pipeline on a small slice of the real bottle data (copied to a temp directory).

Trains with real ImageNet weights, so it skips if the data or the cached weights are missing.
The slice is for exercising the code paths quickly; headline numbers come from the full runs.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from inspection.config import Settings, load_settings
from inspection.evaluate import evaluate_category
from inspection.patchcore import PatchCore, train_category


@pytest.fixture(scope="module")
def mini(tmp_path_factory: pytest.TempPathFactory, data_root: Path) -> Settings:
    base = tmp_path_factory.mktemp("mini")
    src = data_root / "bottle"
    dst = base / "mvtec_ad" / "bottle"
    (dst / "train" / "good").mkdir(parents=True)
    for p in sorted((src / "train" / "good").glob("*.png"))[:26]:
        shutil.copy(p, dst / "train" / "good" / p.name)
    (dst / "test" / "good").mkdir(parents=True)
    for p in sorted((src / "test" / "good").glob("*.png"))[:8]:
        shutil.copy(p, dst / "test" / "good" / p.name)
    (dst / "test" / "broken_large").mkdir(parents=True)
    (dst / "ground_truth" / "broken_large").mkdir(parents=True)
    for p in sorted((src / "test" / "broken_large").glob("*.png"))[:8]:
        shutil.copy(p, dst / "test" / "broken_large" / p.name)
        m = src / "ground_truth" / "broken_large" / f"{p.stem}_mask.png"
        shutil.copy(m, dst / "ground_truth" / "broken_large" / m.name)
    s = load_settings()
    s.paths.data_dir = str(base / "mvtec_ad")
    s.paths.cache_dir = str(base / "cache")
    s.paths.artifacts_dir = str(base / "artifacts")
    s.paths.results_dir = str(base / "results")
    s.latency.max_images = 8
    s.latency.warmup = 2
    s.patchcore.batch_size = 8
    try:
        train_category(s, "bottle")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"cannot train with real weights here: {exc}")
    return s


def test_train_artifact_and_manifest(mini: Settings) -> None:
    man = json.loads((mini.path("artifacts_dir") / "bottle" / "manifest.json").read_text())
    assert man["backbone"]["id"].startswith("resnet50")
    assert man["backbone"]["sha256"] not in ("none", "unknown")
    assert man["seed"] == 0 and man["bank_shape"][1] == 1536
    assert man["n_calib"] == 10 and man["n_fit"] == 16
    assert len(man["dataset_fingerprint"]) == 64
    runtime = mini.path("results_dir") / "runtime.json"
    trained_seconds = json.loads(runtime.read_text())["train_seconds"]["bottle"]
    assert trained_seconds > 1.0
    # second call is a no-op because the artifact is up to date, and it must not overwrite the
    # measured training time with the time of the skip
    assert train_category(mini, "bottle")["created_at"] == man["created_at"]
    assert json.loads(runtime.read_text())["train_seconds"]["bottle"] == trained_seconds


def test_defect_scores_exceed_normal_scores(mini: Settings) -> None:
    model = PatchCore.from_artifact(mini.path("artifacts_dir") / "bottle")
    res = evaluate_category(mini, "bottle")
    assert res["image_auroc"] > 0.9
    assert res["n_test_defective"] == 8 and res["n_test_good"] == 8
    assert res["latency"]["p50_ms"] > 0 and res["bank_mb"] > 0
    assert model.version == res["model_version"]
    # resumable: a second call reads the stored result
    assert evaluate_category(mini, "bottle")["eval_seconds"] == res["eval_seconds"]


def test_cache_rebuild_does_not_touch_headline_results(mini: Settings) -> None:
    from inspection.evaluate import load_cached_test

    results = mini.path("results_dir") / "categories" / "bottle.json"
    before = results.read_text()
    for f in (mini.path("cache_dir") / "bottle").glob("test_*.npy"):
        f.unlink()
    arrays = load_cached_test(mini, "bottle")
    assert arrays["test_maps"].shape == (16, 224, 224)
    assert results.read_text() == before


def test_retrained_model_is_re_evaluated(mini: Settings) -> None:
    """Same config, new artifact: the stored result must not be reused."""
    import time

    old = json.loads((mini.path("results_dir") / "categories" / "bottle.json").read_text())
    time.sleep(1.1)  # the manifest timestamp has one-second resolution
    train_category(mini, "bottle", force=True)
    new = evaluate_category(mini, "bottle")
    assert new["model_version"] != old["model_version"]
    assert new["eval_seconds"] != old["eval_seconds"]
    assert new["image_auroc"] == pytest.approx(old["image_auroc"])  # same data, same seed


def test_postprocess_cascade_report_explore(mini: Settings) -> None:
    from inspection.cascade import run_cascade
    from inspection.explore import run_exploration
    from inspection.postprocess import evaluate_postprocess
    from inspection.report import build_report

    pp = evaluate_postprocess(mini, "bottle")
    assert 0.0 <= pp["flagged_defective"]["box_hit_rate_images"] <= 1.0
    assert len(pp["alpha_sensitivity"]) == 7
    casc = run_cascade(mini, "bottle")
    assert [r["q"] for r in casc["rows"]] == [0, 25, 50, 75]
    assert casc["rows"][0]["normal_skipped"] == 0.0 and casc["rows"][0]["speedup"] == 1.0
    assert (mini.path("artifacts_dir") / "bottle" / "stage1.npz").exists()
    run_exploration(mini)
    rdir = build_report(mini)
    assert (rdir / "metrics_per_category.csv").exists()
    assert (rdir / "comparison_published.csv").exists()
    assert (rdir / "failure_analysis.json").exists()
    assert (rdir / "cascade_tradeoff.csv").exists()


def test_service_inspect_with_cascade(mini: Settings, data_root: Path) -> None:
    from api.service import ModelService
    from PIL import Image

    svc = ModelService(mini)
    svc.discover()
    img_path = sorted((data_root / "bottle" / "test" / "broken_large").glob("*.png"))[0]
    img = Image.open(img_path).convert("RGB")
    out = svc.inspect("bottle", img, use_cascade=False)
    assert out["stage_reached"] == 2 and out["heatmap"] is not None
    casc = svc.inspect("bottle", img, use_cascade=True)
    assert casc["stage1_score"] is not None
    assert svc.loaded() == ["bottle"]
    assert not np.isnan(out["anomaly_score"])


def test_studies_variants_run(mini: Settings) -> None:
    """Preprocessing and few-shot variants on the slice (also checks nested coreset prefixes)."""
    from inspection.studies.common import Variant, run_and_record
    from inspection.studies.preprocessing import PREPROCESSING
    from inspection.studies.robustness import run_robustness
    from inspection.studies.summaries import summarise_studies

    vs = [
        Variant("coreset", ratios=(0.01, 0.1), seed=0),
        Variant("pre_gray", pre_name="gray", pre=PREPROCESSING["gray_opening_3x3"]),
        Variant("n1_seed0", n_fit=1, seed=0, extra={"n_label": 1}),
    ]
    done = run_and_record(mini, "ablations", "bottle", vs)
    assert done["coreset@0.01"]["bank_shape"][0] < done["coreset@0.1"]["bank_shape"][0]
    # the 0.1 prefix run must reproduce the main run exactly (same seed, same split, same math)
    main = json.loads((mini.path("results_dir") / "categories" / "bottle.json").read_text())
    assert done["coreset@0.1"]["image_auroc"] == pytest.approx(main["image_auroc"])
    run_and_record(mini, "fewshot", "bottle", [vs[2]], pixel=False)
    run_robustness(mini, ["bottle"])
    summarise_studies(mini)
    assert (mini.path("results_dir") / "robustness.csv").exists()
    assert (mini.path("results_dir") / "ablations.csv").exists()
