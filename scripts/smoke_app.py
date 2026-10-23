"""Headless smoke test: real API process + Streamlit AppTest over several real categories.

Usage: python scripts/smoke_app.py [category ...]   (default: bottle cable capsule)
Writes results/app_smoke.json. Requires trained artifacts and the dataset.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
PORT = 8765


def main(cats: list[str]) -> int:
    env = {**os.environ, "MVI_API_URL": f"http://127.0.0.1:{PORT}"}
    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "api.main:app",
            "--port",
            str(PORT),
            "--log-level",
            "warning",
        ],
        cwd=ROOT,
        env=env,
    )
    try:
        for _ in range(180):
            try:
                if httpx.get(f"http://127.0.0.1:{PORT}/v1/ready", timeout=2).status_code == 200:
                    break
            except httpx.HTTPError:
                pass
            time.sleep(2)
        else:
            print("API never became ready")
            return 1
        os.environ.update(env)
        from streamlit.testing.v1 import AppTest

        results = []
        for cat in cats:
            at = AppTest.from_file(str(ROOT / "app" / "streamlit_app.py"), default_timeout=180)
            at.run()
            assert not at.exception, at.exception
            at.sidebar.selectbox[0].set_value(cat)
            at.run()
            sample = next(s for s in at.selectbox if s.label.startswith("...or pick"))
            defect = next(o for o in sample.options if "/" in o and not o.startswith("good/"))
            sample.set_value(defect)
            at.run()
            next(b for b in at.button if b.label == "Inspect").click()
            at.run()
            assert not at.exception, at.exception
            assert not at.error, [e.value for e in at.error]
            verdicts = [m.value for m in at.metric if m.label == "Verdict"]
            results.append({"category": cat, "sample": defect, "verdict": verdicts})
            print(cat, defect, verdicts)
        out = ROOT / "results" / "app_smoke.json"
        out.write_text(json.dumps(results, indent=2))
        return 0 if all(r["verdict"] for r in results) else 1
    finally:
        proc.terminate()
        proc.wait(timeout=30)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:] or ["bottle", "cable", "capsule"]))
