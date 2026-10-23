"""Device handling. The CPU path is the reference; CUDA tests run only where a GPU exists
(for example in the Colab notebook, which runs this file before any study)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from inspection.config import Settings
from inspection.coreset import coreset_indices, greedy_kcenter, project_chunked, random_projection
from inspection.features import PatchFeatureExtractor
from inspection.knn import TorchKNN
from inspection.patchcore import PatchCore

cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")


def _feats() -> np.ndarray:
    g = np.stack(np.meshgrid(np.arange(40.0), np.arange(40.0), indexing="ij"), -1).reshape(-1, 2)
    return np.concatenate([g, g[::-1] * 0.5, g * 0.1], axis=1).astype(np.float16)  # (1600, 6)


def test_resolve_device(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MVI_DEVICE", raising=False)  # Colab sets it to cuda
    s = Settings()
    assert s.resolve_device() == "cpu"
    s.device = "auto"
    assert s.resolve_device() == ("cuda" if torch.cuda.is_available() else "cpu")
    s.device = "cuda"
    assert s.resolve_device() == "cuda"


def test_device_is_not_part_of_the_config_hash() -> None:
    a, b = Settings(), Settings()
    b.device = "cuda"
    assert a.config_hash() == b.config_hash()


def test_cpu_device_paths_match_default() -> None:
    f = _feats()
    assert (coreset_indices(f, 30, 4, 0) == coreset_indices(f, 30, 4, 0, device="cpu")).all()
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0]])
    assert TorchKNN(bank, device="cpu").search(torch.tensor([[3.0, 4.0]]))[0] == 0.0


@cuda
def test_cuda_coreset_matches_cpu() -> None:
    f = _feats()
    a = coreset_indices(f, 60, 4, 0, device="cpu")
    b = coreset_indices(f, 60, 4, 0, device="cuda")
    assert a.shape == b.shape and len(set(b.tolist())) == 60
    assert len(set(a.tolist()) & set(b.tolist())) >= 57  # ties may differ by rounding


@cuda
def test_cuda_projection_and_greedy_run_on_device() -> None:
    f = _feats()
    z = project_chunked(f, random_projection(f.shape[1], 4, 0), device="cuda")
    assert z.device.type == "cuda"
    assert len(greedy_kcenter(z, 20, 0)) == 20


@cuda
def test_cuda_knn_matches_cpu() -> None:
    bank = torch.tensor([[0.0, 0.0], [3.0, 4.0], [10.0, 0.0]])
    q = torch.tensor([[0.0, 1.0], [6.0, 8.0], [9.0, 1.0]])
    cpu = TorchKNN(bank, device="cpu").search(q)
    gpu = TorchKNN(bank, device="cuda").search(q).cpu()
    assert torch.allclose(cpu, gpu, atol=1e-5)


@cuda
def test_cuda_features_and_scores_match_cpu() -> None:
    x = torch.linspace(-1, 1, 3 * 224 * 224).reshape(1, 3, 224, 224)
    ext_cpu = PatchFeatureExtractor("resnet18", "random")
    ext_gpu = PatchFeatureExtractor("resnet18", "random")
    ext_gpu.load_state_dict(ext_cpu.state_dict())
    ext_gpu = ext_gpu.to("cuda")
    a, b = ext_cpu.descriptors(x), ext_gpu.descriptors(x).cpu()
    assert (a - b).abs().max() < 1e-3
    bank = a[0, :200].clone()
    s_cpu = PatchCore(ext_cpu, bank).score(x)[0]
    s_gpu = PatchCore(ext_gpu, bank, device="cuda").score(x)[0]
    assert torch.allclose(s_cpu, s_gpu, rtol=1e-3, atol=1e-3)
