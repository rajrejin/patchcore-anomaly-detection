"""The `mvinspect` command line interface."""

from __future__ import annotations

import json
import logging

import typer

from inspection.config import Settings, load_settings
from inspection.logging_utils import setup_logging

app = typer.Typer(add_completion=False, help="PatchCore inspection on MVTec AD")
log = logging.getLogger("mvinspect")


@app.callback()
def _main(
    config: str = typer.Option(None, "--config", help="Path to a YAML config"),
    verbose: bool = typer.Option(False, "--verbose"),
) -> None:
    setup_logging(logging.DEBUG if verbose else logging.INFO)
    if config:
        load_settings(config)


def _cats(settings: Settings, categories: list[str] | None, all_: bool) -> list[str]:
    from inspection.data import CATEGORIES, available_categories

    if categories:
        bad = [c for c in categories if c not in CATEGORIES]
        if bad:
            raise typer.BadParameter(f"unknown categories: {bad}")
        return categories
    if all_:
        return available_categories(settings.path("data_dir"))
    raise typer.BadParameter("give category names or --all")


@app.command("fetch-data")
def fetch_data(
    categories: list[str] = typer.Argument(None, help="Subset of categories (default: all)"),
) -> None:
    """Download MVTec AD per-category archives from the official links."""
    from inspection.fetch import fetch_all

    s = load_settings()
    fetch_all(s.path("data_dir"), s.path("raw_dir"), categories or None)


@app.command("verify-data")
def verify_data() -> None:
    """Check the dataset against the documented counts and layout."""
    from inspection.data import verify_dataset

    s = load_settings()
    rep = verify_dataset(s.path("data_dir"))
    typer.echo(json.dumps({"totals": rep["totals"], "problems": rep["problems"]}, indent=2))
    raise typer.Exit(0 if rep["ok"] else 1)


@app.command()
def env() -> None:
    """Write results/environment.json (hardware, OS, library versions, seed)."""
    from inspection.envinfo import write

    typer.echo(str(write(load_settings())))


@app.command()
def explore() -> None:
    """Write exploration figures to results/figures/exploration."""
    from inspection.explore import run_exploration

    run_exploration(load_settings())


@app.command()
def train(
    categories: list[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Fit memory banks and calibrate thresholds (resumable per category)."""
    from inspection.patchcore import train_category

    s = load_settings()
    for c in _cats(s, categories, all_):
        train_category(s, c, force)  # records its run time only when it actually trains


@app.command()
def evaluate(
    categories: list[str] = typer.Argument(None),
    all_: bool = typer.Option(False, "--all"),
    force: bool = typer.Option(False, "--force"),
) -> None:
    """Evaluate trained categories on the test split (resumable per category)."""
    from inspection.evaluate import evaluate_category

    s = load_settings()
    for c in _cats(s, categories, all_):
        r = evaluate_category(s, c, force)  # records its run time only when it actually runs
        log.info("%s image AUROC %.4f pixel AUROC %.4f", c, r["image_auroc"], r["pixel_auroc"])


@app.command()
def postprocess(
    categories: list[str] = typer.Argument(None), all_: bool = typer.Option(False, "--all")
) -> None:
    """OpenCV post-processing quantification, figures and alpha sensitivity."""
    from inspection.postprocess import evaluate_postprocess

    s = load_settings()
    for c in _cats(s, categories, all_):
        evaluate_postprocess(s, c)


@app.command()
def cascade(
    categories: list[str] = typer.Argument(None), all_: bool = typer.Option(False, "--all")
) -> None:
    """Two-stage cascade study."""
    from inspection.cascade import run_cascade

    s = load_settings()
    for c in _cats(s, categories, all_):
        run_cascade(s, c)


@app.command()
def ablate(
    categories: list[str] = typer.Argument(None),
    cpu_latency: bool = typer.Option(
        False, "--cpu-latency", help="Only measure CPU latency of the coreset-ratio banks"
    ),
) -> None:
    """Preprocessing, coreset ratio, backbone ablations (default: leather hazelnut screw)."""
    from inspection.studies.ablations import measure_ratio_latency, run_ablations

    s = load_settings()
    cats = categories or ["leather", "hazelnut", "screw"]
    if cpu_latency:
        measure_ratio_latency(s, cats)
    else:
        run_ablations(s, cats)


@app.command()
def studies(
    which: str = typer.Argument("all", help="fewshot | robustness | all"),
    categories: list[str] = typer.Argument(None),
) -> None:
    """Few-shot and robustness studies."""
    s = load_settings()
    cats = categories or ["leather", "hazelnut", "screw"]
    if which in ("fewshot", "all"):
        from inspection.studies.fewshot import run_fewshot

        run_fewshot(s, cats)
    if which in ("robustness", "all"):
        from inspection.studies.robustness import run_robustness

        run_robustness(s, cats)


@app.command("export-onnx")
def export_onnx(category: str = typer.Argument("bottle")) -> None:
    """Export the feature extractor to ONNX, check parity and benchmark backends."""
    from inspection.onnx_backend import export_and_benchmark

    export_and_benchmark(load_settings(), category)


@app.command()
def serve(host: str = "127.0.0.1", port: int = 8000) -> None:
    """Run the FastAPI service."""
    import uvicorn

    uvicorn.run("api.main:app", host=host, port=port)


@app.command()
def report() -> None:
    """Build result tables, baseline comparison and failure analysis from results/."""
    from inspection.report import build_report

    build_report(load_settings())


if __name__ == "__main__":
    app()
