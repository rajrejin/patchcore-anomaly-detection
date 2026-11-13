"""Approximate greedy k-center (coreset) selection with random projection."""

from __future__ import annotations

import numpy as np
import torch


def random_projection(dim: int, proj_dim: int, seed: int) -> torch.Tensor:
    """Fixed Gaussian projection, always drawn on the CPU so the matrix is device independent."""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(dim, proj_dim, generator=g) / proj_dim**0.5


def project_chunked(
    feats: np.ndarray,
    proj: torch.Tensor,
    chunk: int = 20000,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Project (N, D) features (possibly float16 or memmap) to (N, proj_dim) float32 in chunks.
    The result lives on `device`."""
    proj = proj.to(device)
    out = torch.empty(feats.shape[0], proj.shape[1], dtype=torch.float32, device=device)
    for i in range(0, feats.shape[0], chunk):
        block = torch.from_numpy(np.asarray(feats[i : i + chunk], dtype=np.float32)).to(device)
        out[i : i + chunk] = block @ proj
    return out


def greedy_kcenter(z: torch.Tensor, n_select: int, seed: int) -> np.ndarray:
    """Greedy k-center on already-projected points z (N, d), on whatever device z lives.

    Start from a random point, then repeatedly pick the point that is farthest from everything
    picked so far. The result covers the whole point cloud evenly instead of oversampling
    dense regions. Deterministic given the seed. Returns indices in selection order.
    """
    n = z.shape[0]
    n_select = min(n_select, n)
    rng = np.random.default_rng(seed)
    newest = int(rng.integers(n))
    selected = np.empty(n_select, dtype=np.int64)
    selected[0] = newest

    squared_norms = (z * z).sum(1)
    # For every point: squared distance to its nearest already-selected point.
    dist_to_selected = torch.full((n,), float("inf"), device=z.device)
    dist_to_newest = torch.empty(n, device=z.device)
    for i in range(1, n_select):
        # ||a - b||^2 = ||a||^2 - 2 a.b + ||b||^2, computed in place to avoid new allocations.
        torch.mv(z, z[newest], out=dist_to_newest)
        dist_to_newest.mul_(-2.0).add_(squared_norms).add_(squared_norms[newest])
        torch.minimum(dist_to_selected, dist_to_newest, out=dist_to_selected)
        newest = int(torch.argmax(dist_to_selected))
        selected[i] = newest
    return selected


def coreset_indices(
    feats: np.ndarray,
    n_select: int,
    proj_dim: int,
    seed: int,
    device: str | torch.device = "cpu",
) -> np.ndarray:
    """Select `n_select` of (N, D) descriptors; distances use a fixed Gaussian projection."""
    proj = random_projection(feats.shape[1], proj_dim, seed)
    z = project_chunked(feats, proj, device=device)
    return greedy_kcenter(z, n_select, seed)


def max_nearest_selected_distance(z: torch.Tensor, idx: np.ndarray) -> float:
    """Covering radius: the largest distance from any point to its nearest selected point."""
    sel = z[torch.from_numpy(idx).to(z.device)]
    best = torch.full((z.shape[0],), float("inf"), device=z.device)
    for i in range(0, z.shape[0], 4096):
        best[i : i + 4096] = torch.cdist(z[i : i + 4096], sel).min(1).values
    return float(best.max())
