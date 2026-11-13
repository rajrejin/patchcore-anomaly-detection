# Decisions

Each entry: decision, alternatives considered, reason. Numbers quoted here come from files in `results/`.

## D1. Version pins forced by a Windows App Control policy

- Decision: pin `scipy==1.15.3`, `pandas==2.2.3`, `scikit-learn==1.6.1`, and run `mypy==1.14.1` as a pure Python build (`pip install --no-binary mypy`).
- Alternatives: newest releases of each; disabling the policy.
- Reason: on my laptop the newest `scipy`, `pandas`, `faiss` and compiled `mypy` binaries were blocked with "An Application Control policy has blocked this file". Changing security settings is out of scope, so older releases (which load) were chosen. `requirements.txt` records the pins. On other machines newer versions should work but were not tested.

## D2. Python 3.13 and torch 2.14

- Decision: use the Python 3.13 that is installed, CPU-only torch from the PyTorch CPU index, pinned in `requirements.txt`.
- Alternatives: install 3.12. Not needed: all dependencies had 3.13 wheels. The Docker images use 3.12.8 (built and tested, see `results/docker_verification.json`).

## D4. Pixel threshold equals the image threshold

- Decision: as specified, both thresholds are the maximum image score over calibration normals.
- Consequence: the anomaly map is a blurred, upsampled version of the patch scores, so its maximum is a little lower than the image score (the image score is the maximum raw patch score). An image that barely exceeds the image threshold can therefore produce an empty thresholded mask and no boxes. This is visible in the alpha sensitivity tables (`results/postprocess/*.json`). It was not "fixed" by tuning, because that would use test information.

## D5. Coreset ratio and bank floor

- Decision: `coreset_ratio` 0.1 with a bank floor `min_bank` of 1000 patches (never binding on full-data runs, where the pool is above 100000 patches).
- Reason: the few-shot study uses 1 to 50 images. 10 percent of one image is 78 patches, which would measure a degenerate bank rather than the effect of fewer images. With the floor, N = 1 uses all 784 patches of that image. The "all" points use the ordinary 10 percent coreset.

## D6. Greedy coreset is nested, so ratio studies reuse one run

- Decision: for the coreset ratio ablation one greedy k-center run at 25 percent is taken and its prefixes give the 1 and 10 percent banks.
- Reason: greedy selection with the same seed and projection is a prefix property, so this equals separate runs. The integration test `test_studies_variants_run` checks that the 10 percent prefix reproduces the main run exactly.

## D7. Tiny differences between runs

- The anomaly map blur uses replicate padding and a kernel radius of 4 sigma (33 taps at sigma 4). The reference PatchCore code uses a similar Gaussian filter; exact padding differs, which can move pixel metrics slightly.
- Image resize uses PIL bilinear resize to 256 x 256 (aspect ratio is already 1:1 in MVTec AD) then a center crop of 224.

## D8. Cascade design choices

- Stage 1 embeds the 224 crop downscaled to 128 x 128 (bilinear with antialiasing) with ResNet-18 ImageNet weights and the global average pool of the last block.
- `q = 0` is implemented as "skip nothing" (threshold of minus infinity), not as the minimum of the calibration scores.
- Cascade latency is computed from measured per-stage mean latencies and the measured skip fractions (the combined pipeline was not timed as one process). The API cascade quantile `serve_q = 50` was fixed before any cascade result was looked at, so it is not tuned on test labels.

## D9. Study seeds

- "Seed" changes the calibration split and the coreset starting point (and the few-shot subset). Seeds 0, 1, 2. Differences smaller than the spread over these seeds are reported as not meaningful.

## D10. Published baselines

- Source values were read from an HTML rendering of the paper's supplementary tables through a summarising fetch tool, not from the PDF. The column means reproduce the reported averages for pixel AUROC (98.15 against the reported 98.1) and approximately for image AUROC (98.93 against the reported 99.0). They should be checked against the PDF before being quoted precisely. See `results/published_baselines.csv`.

## D11. Warm-up uses a zero tensor

- The API warm-up runs one forward pass on a zero tensor so the first real request is fast. It produces no stored result and no number in any document.

## D12. Test images

- Tests that need images use real MVTec AD images copied to a temporary directory at test time, never into git. Tests of pure arithmetic use small hand-computed arrays. Unit tests of tensor shapes use random-weight backbones and constant inputs; no results come from them, and `train` and `evaluate` refuse random weights.

## D13. Config hash excludes the API cascade quantile

- `Settings.config_hash` ignores `cascade.serve_q` so artifacts trained before that field existed stay valid.

## D14. Serving backends chosen by measurement

- Decision: the API serves with the ONNX Runtime feature extractor and the faiss nearest-neighbour index (`api.extractor_backend: onnx`, `api.knn_backend: faiss`), falling back to PyTorch automatically if the ONNX file, onnxruntime or faiss is unavailable. Training and the reported evaluation use PyTorch.
- Evidence: two benchmark runs on bottle (`results/inference_benchmark_run1.json`, `results/inference_benchmark.json`). ONNX plus faiss had the lowest p50 in both runs, by about 5 to 11 percent over PyTorch plus PyTorch, with feature parity far below the 1e-3 target. The margin is modest, and run to run variation on this laptop is of similar size, so this is a mild preference, not a strong result.
- Alternative: keep PyTorch everywhere (fewer dependencies). Rejected because the measured direction was the same in both runs.

## D15. Studies run on a Colab GPU

- Decision: the ablation, few-shot and robustness studies were run on a free Colab T4 through an optional `device` setting (default `cpu`), then checked against the CPU main run.
- Reason: the coreset step made the full study set several hours on this CPU. On the GPU the whole set finished in about 19 minutes (from the Colab console, not stored in `results/`).
- Safeguards: the `device` setting is excluded from the config hash; unit tests check CPU and CUDA paths against each other; `scripts/colab_run.py` stops before any study if those tests or a per-category AUROC parity check fail; `results/parity_cpu_vs_gpu.json` compares the GPU baseline with the CPU main run (image AUROC and thresholds identical, pixel AUROC within 2e-6).
- Consequence: latency is not measured in the studies. The coreset ratio banks were rebuilt on the CPU from the saved selection order and timed there (`mvinspect ablate --cpu-latency`). The main results (training, evaluation, latency, cascade) were all produced on the CPU.
- The project owner chose this. It used a free Colab tier with no paid service, and only the study results (not the headline results or latency) came from it.

## D16. Latency of the ratio banks is a separate session

- The latency column in the coreset ratio table was measured in its own session (20 images, rebuilt banks) and runs 20 to 30 percent above the main latency table for the same bank size. The two tables are not comparable with each other; compare rows within one table only.

## D17. Colab bundle without Google Drive

- The Colab run does not use Google Drive. The project bundle is embedded in a notebook cell as base64 and `scripts/colab_run.py` does the rest, to avoid a Drive permission grant. `notebooks/colab_studies.ipynb` is a Drive based variant of the same steps.

## D18. CI downloads one category

- Decision: the CI workflow downloads bottle (about 150 MB) from the official MVTec link, caches it with the ImageNet weights, and trains it before running the tests.
- Reason: the integration tests need real images and a trained model. Without them they skip and coverage falls below the 80 percent limit. Committing data is not allowed by the license, and synthetic data is not used anywhere in this project.
- Alternative: lower the coverage limit in CI. Rejected because it would hide untested code.
- Status: on 22 October 2026 the official links returned HTTP 404 (KNOWN_ISSUES K10). While that lasts the CI job fails at its download step. Making the coverage limit optional when data is missing was considered and rejected, because it would let CI pass while the integration tests do not run.
