"""Coreset, kNN, features, threshold, artifacts: math and shape checks on toy inputs."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from inspection.artifacts import ArtifactError, load_artifact, save_artifact
from inspection.coreset import (
    coreset_indices,
    greedy_kcenter,
    max_nearest_selected_distance,
    project_chunked,
    random_projection,
)
from inspection.features import PatchFeatureExtractor, RandomWeightsError, require_real_weights
from inspection.knn import TorchKNN, make_knn
from inspection.patchcore import PatchCore, gaussian_blur, to_anomaly_map
from inspection.threshold import apply_alpha, calibrate


def _lattice() -> torch.Tensor:
    """A 30 x 30 lattice with a dense cluster in one corner (hand-built, no randomness)."""
    g = torch.stack(torch.meshgrid(torch.arange(30.0), torch.arange(30.0), indexing="ij"), -1)
    pts = g.reshape(-1, 2)
    cluster = pts[:100] * 0.05
    return torch.cat([pts, cluster])


def test_coreset_count_and_determinism() -> None:
    z = _lattice()
    a = greedy_kcenter(z, 25, seed=0)
    b = greedy_kcenter(z, 25, seed=0)
    assert len(a) == 25 and len(set(a.tolist())) == 25
    assert (a == b).all()


def test_coreset_covers_better_than_strided_selection() -> None:
    z = _lattice()
    greedy = greedy_kcenter(z, 25, seed=0)
    strided = np.arange(0, len(z), len(z) // 25)[:25]  # deterministic non-greedy baseline
    assert max_nearest_selected_distance(z, greedy) < max_nearest_selected_distance(z, strided)


def test_coreset_indices_uses_projection_and_returns_requested() -> None:
    feats = _lattice().numpy().astype(np.float16)
    feats = np.concatenate([feats, feats], axis=1)  # D = 4
    idx = coreset_indices(feats, 10, proj_dim=3, seed=0)
    assert idx.shape == (10,)
    proj = random_projection(4, 3, 0)
    assert project_chunked(feats, proj).shape == (len(feats), 3)


def test_knn_vector_in_bank_scores_zero_and_matches_manual() -> None:
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
    knn = TorchKNN(bank)
    q = torch.tensor([[3.0, 4.0], [0.0, 1.0], [6.0, 8.0]])
    d = knn.search(q)
    assert d[0] == 0.0
    assert d[1].item() == pytest.approx(1.0)
    assert d[2].item() == pytest.approx(5.0)


def test_knn_backends_agree() -> None:
    pytest.importorskip("faiss")
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
    q = torch.tensor([[0.0, 1.0], [6.0, 8.0], [9.0, 1.0]])
    assert torch.allclose(
        make_knn("torch", bank).search(q), make_knn("faiss", bank).search(q), atol=1e-5
    )
    with pytest.raises(ValueError):
        make_knn("nope", bank)


def test_feature_shapes_and_frozen() -> None:
    ext = PatchFeatureExtractor("resnet50", "random")
    assert all(not p.requires_grad for p in ext.parameters())
    assert not ext.training
    ext.train()
    assert not ext.training
    x = torch.zeros(2, 3, 224, 224)
    f2, f3 = ext.layer_maps(x)
    assert tuple(f2.shape) == (2, 512, 28, 28)
    assert tuple(f3.shape) == (2, 1024, 14, 14)
    assert tuple(ext.descriptors(x).shape) == (2, 784, 1536)
    assert not ext.descriptors(x).requires_grad


def test_channel_reduction_shape() -> None:
    ext = PatchFeatureExtractor("resnet18", "random", reduce_dim=128)
    assert tuple(ext.descriptors(torch.zeros(1, 3, 224, 224)).shape) == (1, 784, 128)


def test_random_weights_refused_for_real_runs() -> None:
    with pytest.raises(RandomWeightsError):
        require_real_weights("random")
    require_real_weights("imagenet")


def test_image_score_is_max_patch_score_and_bank_member_scores_zero() -> None:
    ext = PatchFeatureExtractor("resnet18", "random")
    x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    desc = ext.descriptors(x)[0]
    model = PatchCore(ext, desc.clone())
    ps = model.patch_scores(x)
    # near-duplicate patches can tie in the matrix-product argmin; stay under 1 percent of the norm
    assert float(ps.abs().max()) < 1e-2 * float(desc.norm(dim=1).mean())
    shifted = PatchCore(ext, desc[:100].clone())
    ps2 = shifted.patch_scores(x)
    score, amap = shifted.score(x)
    assert score.item() == pytest.approx(float(ps2.max()))
    assert tuple(amap.shape) == (1, 224, 224)


def test_anomaly_map_upsample_and_blur_preserve_constant() -> None:
    ps = torch.full((1, 28, 28), 2.0)
    amap = to_anomaly_map(ps, 224, 4.0)
    assert amap.shape == (1, 224, 224)
    assert torch.allclose(amap, torch.full_like(amap, 2.0), atol=1e-4)
    assert gaussian_blur(amap, 0.0) is amap


def test_calibrate_is_max_and_label_free() -> None:
    t = calibrate(np.array([1.0, 1.5, 1.2]))
    assert t["image"] == 1.5 and t["pixel"] == 1.5 and t["n_calib"] == 3
    assert apply_alpha(2.0, 0.6) == pytest.approx(1.2)
    with pytest.raises(ValueError):
        calibrate(np.array([]))


def _save(tmp_path, cat="bottle"):  # type: ignore[no-untyped-def]
    d = tmp_path / cat
    bank = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    thr = {"image": 1.0, "pixel": 1.0}
    fields = {
        "category": cat, "config_hash": "abc", "seed": 0,
        "backbone": {"name": "resnet50", "id": "x", "sha256": "y"},
        "model_config": {}, "dataset_fingerprint": "f", "thresholds": thr,
    }  # fmt: skip
    save_artifact(d, bank, fields)
    return d, bank


def test_artifact_round_trip(tmp_path) -> None:  # type: ignore[no-untyped-def]
    d, bank = _save(tmp_path)
    loaded, man = load_artifact(d)
    assert torch.equal(loaded, bank)
    assert man["bank_shape"] == [2, 2] and man["thresholds"]["image"] == 1.0


def test_artifact_rejects_tampering(tmp_path) -> None:  # type: ignore[no-untyped-def]
    d, _ = _save(tmp_path)
    np.save(d / "bank.npy", np.zeros((2, 2), dtype=np.float32))
    with pytest.raises(ArtifactError, match="checksum"):
        load_artifact(d)
    d2, _ = _save(tmp_path, "cable")
    (d2 / "thresholds.json").write_text(json.dumps({"image": 9.0, "pixel": 9.0}))
    with pytest.raises(ArtifactError, match="disagrees"):
        load_artifact(d2)
    d3, _ = _save(tmp_path, "grid")
    (d3 / "manifest.json").unlink()
    with pytest.raises(ArtifactError, match="missing"):
        load_artifact(d3)
