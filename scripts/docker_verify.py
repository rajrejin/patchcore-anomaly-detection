"""Build and start the compose stack, hit the API, record facts in results/docker_verification.json.

Usage: python scripts/docker_verify.py
Needs Docker, trained artifacts in ./artifacts and the dataset in ./data/mvtec_ad.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
API = "http://127.0.0.1:8000"


def sh(*cmd: str, check: bool = True) -> str:
    p = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    if check and p.returncode != 0:
        sys.exit(f"command failed: {' '.join(cmd)}\n{p.stdout[-1500:]}\n{p.stderr[-1500:]}")
    return p.stdout.strip()


def main() -> None:
    out: dict[str, object] = {"docker_version": sh("docker", "--version")}
    sh("docker", "compose", "config", "--quiet")
    out["compose_config_ok"] = True
    t0 = time.time()
    sh("docker", "compose", "up", "-d", "--build")
    try:
        deadline = time.time() + 180
        while time.time() < deadline:
            try:
                if httpx.get(f"{API}/v1/ready", timeout=3).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(3)
        else:
            sys.exit("API never became ready")
        out["build_and_start_seconds"] = round(time.time() - t0, 1)
        out["ready"] = httpx.get(f"{API}/v1/ready").json()
        out["health"] = httpx.get(f"{API}/v1/health").json()
        test = ROOT / "data" / "mvtec_ad" / "bottle" / "test"
        bad = next((test / "broken_large").glob("*.png"))
        good = next((test / "good").glob("*.png"))
        results = {}
        for name, path, cascade in (("defective", bad, False), ("good_cascade", good, True)):
            r = httpx.post(
                f"{API}/v1/inspect",
                params={"category": "bottle", "use_cascade": str(cascade).lower()},
                files={"file": (path.name, path.read_bytes(), "image/png")},
                timeout=120,
            )
            j = r.json()
            results[name] = {
                "status": r.status_code,
                "is_anomalous": j.get("is_anomalous"),
                "anomaly_score": j.get("anomaly_score"),
                "stage_reached": j.get("stage_reached"),
                "model_ms": j.get("latency", {}).get("model_ms"),
                "n_boxes": len(j.get("boxes", [])),
            }
        out["inspections"] = results
        bad_cat = httpx.post(
            f"{API}/v1/inspect",
            params={"category": "../etc"},
            files={"file": ("x.png", good.read_bytes(), "image/png")},
        )
        out["path_traversal_category"] = {"status": bad_cat.status_code, "body": bad_cat.json()}
        out["app_health_http"] = httpx.get("http://127.0.0.1:8501/_stcore/health").status_code
        sizes = sh(
            "docker",
            "image",
            "inspect",
            "mvtec-inspection-api",
            "mvtec-inspection-app",
            "--format",
            "{{index .RepoTags 0}} {{.Size}}",
        ).splitlines()
        out["image_bytes"] = {s.split()[0]: int(s.split()[1]) for s in sizes}
        out["container_user"] = sh("docker", "exec", "mvtec-inspection-api-1", "id")
        ro = sh(
            "docker", "exec", "mvtec-inspection-api-1", "sh", "-c", "touch /app/x 2>&1", check=False
        )
        out["read_only_root_fs"] = "Read-only file system" in ro
        out["stats"] = sh(
            "docker", "stats", "--no-stream", "--format", "{{.Name}} {{.MemUsage}}"
        ).splitlines()
    finally:
        sh("docker", "compose", "down", check=False)
    (ROOT / "results" / "docker_verification.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
