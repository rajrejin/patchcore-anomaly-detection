"""Load test: asyncio + httpx against a running API at client concurrency 1, 2 and 4.

Usage: python scripts/load_test.py [--url URL] [--category bottle] [--n 24] [--label NAME]
Sends real MVTec test images. Writes results/load_test_<label>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import httpx
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


async def run_level(url: str, cat: str, images: list[bytes], conc: int, n: int) -> dict[str, float]:
    sem = asyncio.Semaphore(conc)
    lat: list[float] = []
    errors = 0

    async with httpx.AsyncClient(base_url=url, timeout=300) as client:

        async def one(i: int) -> None:
            nonlocal errors
            async with sem:
                t = time.perf_counter()
                r = await client.post(
                    "/v1/inspect",
                    params={"category": cat},
                    files={"file": ("x.png", images[i % len(images)], "image/png")},
                )
                if r.status_code != 200:
                    errors += 1
                lat.append((time.perf_counter() - t) * 1000)

        t0 = time.perf_counter()
        await asyncio.gather(*(one(i) for i in range(n)))
        wall = time.perf_counter() - t0
    a = np.asarray(lat)
    return {
        "concurrency": conc,
        "requests": n,
        "errors": errors,
        "p50_ms": float(np.percentile(a, 50)),
        "p95_ms": float(np.percentile(a, 95)),
        "throughput_rps": n / wall,
    }


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--category", default="bottle")
    ap.add_argument("--n", type=int, default=24)
    ap.add_argument("--label", default="default")
    ap.add_argument("--data", default=str(ROOT / "data" / "mvtec_ad"))
    args = ap.parse_args()
    files = sorted(Path(args.data, args.category, "test").glob("*/*.png"))
    images = [files[i].read_bytes() for i in np.linspace(0, len(files) - 1, 12).astype(int)]
    out = {"url": args.url, "category": args.category, "label": args.label, "levels": []}
    async with httpx.AsyncClient(base_url=args.url, timeout=60) as c:
        r = await c.get("/v1/ready")
        out["ready"] = r.json()
        await c.post(
            "/v1/inspect", params={"category": args.category}, files={"file": ("w.png", images[0])}
        )  # warm the category before timing
    for conc in (1, 2, 4):
        res = await run_level(args.url, args.category, images, conc, args.n)
        out["levels"].append(res)
        print(res)
    dest = ROOT / "results" / f"load_test_{args.label}.json"
    dest.write_text(json.dumps(out, indent=2))


if __name__ == "__main__":
    asyncio.run(main())
