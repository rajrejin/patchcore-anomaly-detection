"""Streamlit front end. Talks to the API over HTTP only and never imports model code."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))
import client  # noqa: E402

DATA_DIR = Path(os.environ.get("MVI_DATA_DIR", "data/mvtec_ad"))
RESULTS_DIR = Path(os.environ.get("MVI_RESULTS_DIR", "results"))

st.set_page_config(page_title="PatchCore inspection", layout="wide")
st.title("Industrial visual inspection (PatchCore, MVTec AD)")

if "history" not in st.session_state:
    st.session_state["history"] = []

with st.sidebar:
    api_url = st.text_input("API URL", os.environ.get("MVI_API_URL", "http://localhost:8000"))
    api_key = st.text_input("API key (optional)", type="password") or None
    cats: list[dict] = []
    try:
        cats = client.list_categories(api_url, api_key)
    except client.ApiProblem as exc:
        st.error(str(exc))
    names = [c["category"] for c in cats]
    category = st.selectbox("Category", names) if names else None
    use_cascade = st.toggle("Use two-stage cascade", value=False)
    if category:
        info = next(c for c in cats if c["category"] == category)
        st.caption(f"model {info['model_version']}, threshold {info['threshold']:.3f}")
        if use_cascade and not info["cascade_available"]:
            st.warning("No stage-1 data for this category; the API will refuse the cascade.")

tab_inspect, tab_history, tab_results = st.tabs(["Inspect", "History", "Results"])

with tab_inspect:
    upload = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp"])
    sample_path: Path | None = None
    if category and (DATA_DIR / category / "test").is_dir():
        samples = sorted((DATA_DIR / category / "test").glob("*/*.png"))
        labels = [f"{p.parent.name}/{p.name}" for p in samples]
        choice = st.selectbox("...or pick a test sample (dataset is mounted)", ["-"] + labels)
        if choice != "-":
            sample_path = samples[labels.index(choice)]
    if st.button("Inspect", type="primary", disabled=not category):
        if upload is not None:
            data, fname = upload.getvalue(), upload.name
        elif sample_path is not None:
            data, fname = sample_path.read_bytes(), f"{sample_path.parent.name}/{sample_path.name}"
        else:
            data, fname = b"", ""
            st.info("Upload an image or choose a sample first.")
        if data and category:
            try:
                with st.spinner("Inspecting..."):
                    verdict = client.inspect_image(
                        api_url, category, data, fname.split("/")[-1], use_cascade, api_key
                    )
            except client.ApiProblem as exc:
                st.error(f"Inspection failed. {exc}")
            else:
                r = verdict.raw
                st.session_state["history"].append(client.history_row(category, fname, verdict))
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Verdict", "ANOMALOUS" if r["is_anomalous"] else "OK")
                score = (
                    "skipped (stage 1)"
                    if r["anomaly_score"] is None
                    else f"{r['anomaly_score']:.3f}"
                )
                c2.metric("Score", score)
                c3.metric("Threshold", "-" if r["threshold"] is None else f"{r['threshold']:.3f}")
                c4.metric("Stage reached", r["stage_reached"])
                st.caption(
                    f"model {r['latency']['model_ms']:.0f} ms, end to end "
                    f"{r['latency']['end_to_end_ms']:.0f} ms, model version {r['model_version']}"
                )
                a, b, c = st.columns(3)
                a.image(data, caption="original upload")
                hm = verdict.heatmap
                if hm is not None:
                    b.image(hm, caption="heatmap overlay (224 px model frame)")
                else:
                    b.info("No heatmap: the image was cleared by stage 1.")
                c.image(verdict.annotated, caption="annotated")

with tab_history:
    if st.session_state["history"]:
        st.dataframe(pd.DataFrame(st.session_state["history"]), width="stretch")
    else:
        st.write("No inspections yet in this session.")

with tab_results:
    csv = RESULTS_DIR / "metrics_per_category.csv"
    if csv.exists():
        st.subheader("Per-category metrics (from results/metrics_per_category.csv)")
        st.dataframe(pd.read_csv(csv), width="stretch")
    else:
        st.write("results/metrics_per_category.csv not found (run: mvinspect report).")
    for extra in ("cascade_tradeoff.csv", "ablations.csv", "robustness.csv", "fewshot_summary.csv"):
        f = RESULTS_DIR / extra
        if f.exists():
            with st.expander(extra):
                st.dataframe(pd.read_csv(f), width="stretch")
    for fig in (
        sorted((RESULTS_DIR / "figures").glob("*.png"))
        if (RESULTS_DIR / "figures").exists()
        else []
    ):
        st.image(str(fig), caption=fig.name)
