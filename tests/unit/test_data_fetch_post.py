"""Config, split logic, safe extraction and OpenCV post-processing on toy arrays."""

from __future__ import annotations

import io
import tarfile
from pathlib import Path

import numpy as np
import pytest

from inspection import postprocess as P
from inspection.config import Settings, load_settings
from inspection.data import (
    CATEGORIES,
    OBJECTS,
    TEXTURES,
    available_categories,
    geometry_mask,
    split_fit_calib,
    train_paths,
    verify_dataset,
)
from inspection.data import test_samples as list_test_samples
from inspection.fetch import safe_extract


def test_config_defaults_and_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    s = load_settings()
    assert s.seed == 0 and s.backbone.weights == "imagenet" and s.patchcore.coreset_ratio == 0.1
    monkeypatch.setenv("MVI_PATCHCORE__CORESET_RATIO", "0.25")
    assert Settings().patchcore.coreset_ratio == 0.25
    assert len(s.config_hash()) == 16


def test_category_lists() -> None:
    assert len(OBJECTS) == 10 and len(TEXTURES) == 5 and len(CATEGORIES) == 15


def test_split_fit_calib_rules() -> None:
    paths = [Path(f"{i}.png") for i in range(100)]
    fit, cal = split_fit_calib(paths, seed=0)
    assert len(cal) == 10 and len(fit) == 90 and not set(fit) & set(cal)
    assert (fit, cal) == split_fit_calib(paths, seed=0)
    assert cal != split_fit_calib(paths, seed=1)[1]
    fit, cal = split_fit_calib(paths[:40], seed=0)
    assert len(cal) == 10  # minimum calibration size
    fit, cal = split_fit_calib(paths[:8], seed=0)
    assert cal == paths[:8]  # uses all if fewer than the minimum


def test_mask_geometry_nearest_and_crop() -> None:
    from PIL import Image

    m = np.zeros((16, 16), dtype=np.uint8)
    m[:, 8:] = 255
    out = geometry_mask(Image.fromarray(m), resize=16, crop=8)
    assert out.shape == (8, 8) and set(np.unique(out)) <= {0, 1}
    assert out[:, :4].sum() == 0 and out[:, 4:].all()
    assert geometry_mask(None, 16, 8).sum() == 0


def test_verify_dataset_reports_missing(tmp_path: Path) -> None:
    rep = verify_dataset(tmp_path)
    assert not rep["ok"] and len(rep["problems"]) == 15  # type: ignore[arg-type]
    assert available_categories(tmp_path) == []


def test_listing_layout(tmp_path: Path) -> None:
    from PIL import Image

    (tmp_path / "cat" / "train" / "good").mkdir(parents=True)
    (tmp_path / "cat" / "test" / "good").mkdir(parents=True)
    (tmp_path / "cat" / "test" / "crack").mkdir(parents=True)
    (tmp_path / "cat" / "ground_truth" / "crack").mkdir(parents=True)
    for p in ("train/good/000.png", "test/good/000.png", "test/crack/000.png"):
        Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(tmp_path / "cat" / p)
    assert len(train_paths(tmp_path, "cat")) == 1
    s = list_test_samples(tmp_path, "cat")
    assert [x.label for x in s] == [1, 0]  # crack sorts before good
    assert s[0].mask_path == tmp_path / "cat" / "ground_truth" / "crack" / "000_mask.png"


def _tar(path: Path, members: list[tuple[str, bytes]], link: str | None = None) -> None:
    with tarfile.open(path, "w:xz") as tf:
        for name, data in members:
            ti = tarfile.TarInfo(name)
            ti.size = len(data)
            tf.addfile(ti, io.BytesIO(data))
        if link:
            ti = tarfile.TarInfo(link)
            ti.type = tarfile.SYMTYPE
            ti.linkname = "/etc/passwd"
            tf.addfile(ti)


def test_safe_extract_ok_and_rejections(tmp_path: Path) -> None:
    ok = tmp_path / "ok.tar.xz"
    _tar(ok, [("bottle/a.txt", b"x")])
    assert safe_extract(ok, tmp_path / "out") == 1
    assert (tmp_path / "out" / "bottle" / "a.txt").read_bytes() == b"x"
    bad = tmp_path / "bad.tar.xz"
    _tar(bad, [("../evil.txt", b"x")])
    with pytest.raises(ValueError):
        safe_extract(bad, tmp_path / "o2")
    absolute = tmp_path / "abs.tar.xz"
    _tar(absolute, [("/tmp/evil", b"x")])
    with pytest.raises(ValueError):
        safe_extract(absolute, tmp_path / "o3")
    sym = tmp_path / "sym.tar.xz"
    _tar(sym, [("a.txt", b"x")], link="l")
    with pytest.raises(ValueError):
        safe_extract(sym, tmp_path / "o4")


def test_fetch_skips_finished_categories_without_network(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import json

    from inspection import fetch

    data = tmp_path / "mvtec_ad"
    (data / "bottle" / "test").mkdir(parents=True)
    rec = tmp_path / "downloads.json"
    rec.write_text(json.dumps({"bottle": "abc"}))

    def no_network() -> dict[str, str]:
        raise AssertionError("the downloads page must not be read when nothing is missing")

    monkeypatch.setattr(fetch, "discover_links", no_network)
    assert fetch.fetch_all(data, tmp_path / "raw", ["bottle"], rec) == {"bottle": "abc"}


def test_fetch_uses_local_archive_first_and_keeps_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from inspection import fetch

    raw = tmp_path / "raw"
    raw.mkdir()
    _tar(raw / "bottle.tar.xz", [("bottle/test/good/a.txt", b"x")])

    def no_network() -> dict[str, str]:
        raise AssertionError("a local archive must be used before any download")

    monkeypatch.setattr(fetch, "discover_links", no_network)
    rec = fetch.fetch_all(tmp_path / "mvtec_ad", raw, ["bottle"], tmp_path / "downloads.json")
    assert (tmp_path / "mvtec_ad" / "bottle" / "test" / "good" / "a.txt").read_bytes() == b"x"
    assert len(rec["bottle"]) == 64  # SHA-256 recorded
    assert (raw / "bottle.tar.xz").exists()  # the user's archive is not deleted


def test_download_fails_at_once_on_404(tmp_path: Path) -> None:
    import httpx

    from inspection import fetch

    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(404, text="not found")

    with pytest.raises(fetch.DownloadUnavailable, match="data/raw"):
        fetch.download("https://example.invalid/x.tar.xz", tmp_path / "x.tar.xz",
                       transport=httpx.MockTransport(handler))  # fmt: skip
    assert len(calls) == 1  # a permanent error is not retried


def test_download_resumes_and_completes(tmp_path: Path) -> None:
    import httpx

    from inspection import fetch

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers.get("Range") == "bytes=3-":
            return httpx.Response(206, content=b"def")
        return httpx.Response(200, content=b"abcdef")

    (tmp_path / "x.tar.xz.part").write_bytes(b"abc")
    out = fetch.download("https://example.invalid/x.tar.xz", tmp_path / "x.tar.xz",
                         transport=httpx.MockTransport(handler))  # fmt: skip
    assert out.read_bytes() == b"abcdef"


def test_clean_mask_removes_speck_and_keeps_blob() -> None:
    m = np.zeros((40, 40), dtype=np.uint8)
    m[10:25, 10:25] = 1
    m[2, 2] = 1
    out = P.clean_mask(m, 5)
    assert out[2, 2] == 0 and out[15, 15] == 1


def test_boxes_and_min_area() -> None:
    m = np.zeros((40, 40), dtype=np.uint8)
    m[10:20, 5:15] = 1
    m[30:32, 30:32] = 1
    amap = m.astype(np.float32) * 3
    boxes, _ = P.find_boxes(m, amap, min_area=20)
    assert len(boxes) == 1
    x, y, w, h, s = boxes[0]
    assert (x, y) == (5, 10) and w >= 9 and h >= 9 and s == 3.0


def test_box_iou_and_gt_boxes() -> None:
    assert P.box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert P.box_iou((0, 0, 10, 10), (20, 20, 5, 5)) == 0.0
    assert P.box_iou((0, 0, 10, 10), (5, 0, 10, 10)) == pytest.approx(50 / 150)
    gt = np.zeros((20, 20), dtype=np.uint8)
    gt[2:6, 3:9] = 1
    assert P.gt_region_boxes(gt) == [(3, 2, 6, 4)]


def test_annotate_keeps_rgb_order() -> None:
    rgb = np.zeros((60, 60, 3), dtype=np.uint8)
    out = P.annotate(rgb, [(10, 30, 20, 20, 1.0)], 1.0)
    # the box is drawn red: in RGB output the red channel is 255 and blue is 0
    assert tuple(out[30, 15]) == (255, 0, 0)


def test_heatmap_overlay_shape_dtype() -> None:
    rgb = np.full((8, 8, 3), 100, dtype=np.uint8)
    amap = np.linspace(0, 4, 64, dtype=np.float32).reshape(8, 8)
    out = P.heatmap_overlay(rgb, amap, threshold=1.0)
    assert out.shape == (8, 8, 3) and out.dtype == np.uint8
