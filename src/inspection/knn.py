"""Nearest neighbour search over the memory bank (k = 1, Euclidean)."""

from __future__ import annotations

import numpy as np
import torch


class TorchKNN:
    """Chunked exact search. The argmin comes from the matrix-product form; the returned
    distance is recomputed exactly for the winning entry, so a vector in the bank scores 0.
    The bank lives on `device`; queries are moved there and results are returned on it."""

    name = "torch"

    def __init__(
        self, bank: torch.Tensor, chunk: int = 2048, device: str | torch.device = "cpu"
    ) -> None:
        self.bank = bank.float().contiguous().to(device)
        self.bank_sq = (self.bank * self.bank).sum(1)
        self.chunk = chunk

    def search(self, q: torch.Tensor) -> torch.Tensor:
        q = q.float().to(self.bank.device)
        out = torch.empty(q.shape[0], device=q.device)
        for i in range(0, q.shape[0], self.chunk):
            block = q[i : i + self.chunk]
            d2 = (
                (block * block).sum(1, keepdim=True)
                + self.bank_sq[None, :]
                - 2.0 * block @ self.bank.T
            )
            idx = d2.argmin(1)
            out[i : i + self.chunk] = (block - self.bank[idx]).norm(dim=1)
        return out


class FaissKNN:
    """Exact flat L2 index from faiss-cpu (optional dependency). Always runs on the CPU."""

    name = "faiss"

    def __init__(self, bank: torch.Tensor) -> None:
        import faiss

        self.bank = np.ascontiguousarray(bank.float().cpu().numpy())
        self.index = faiss.IndexFlatL2(self.bank.shape[1])
        self.index.add(self.bank)

    def search(self, q: torch.Tensor) -> torch.Tensor:
        qn = np.ascontiguousarray(q.float().cpu().numpy())
        _, idx = self.index.search(qn, 1)
        diff = qn - self.bank[idx[:, 0]]
        return torch.from_numpy(np.linalg.norm(diff, axis=1).astype(np.float32))


def make_knn(
    backend: str, bank: torch.Tensor, device: str | torch.device = "cpu"
) -> TorchKNN | FaissKNN:
    if backend == "faiss":
        return FaissKNN(bank)
    if backend == "torch":
        return TorchKNN(bank, device=device)
    raise ValueError(f"unknown knn backend {backend}")
