# Industrial visual inspection with PatchCore on MVTec AD

A from-scratch implementation of PatchCore (pretrained CNN feature extraction, a coreset memory bank of normal patches, nearest-neighbour anomaly scoring) that detects defective parts and localises defects at pixel level using only defect-free training images. It covers all {{n_categories}} MVTec AD categories, is measured on CPU (accuracy, localisation, latency, throughput, memory), and is served through a FastAPI service with a Streamlit client, plus Docker files, tests and CI configuration.

**Headline (all from `results/`, test split of MVTec AD, seed 0):** mean image AUROC {{mean_image_auroc}}, mean pixel AUROC {{mean_pixel_auroc}}, mean AU-PRO {{mean_au_pro}} over {{n_categories}} categories. Weakest categories by image AUROC: {{weakest}}. Median model-only latency {{mean_p50_ms}} ms per image on this CPU-only laptop ({{cpu}}; about {{mean_fps}} images per second, one image at a time). At the label-free calibrated threshold the mean F1 is {{mean_f1}} (oracle F1, chosen with test labels: {{mean_oracle_f1}}) and the mean false positive rate on test normals is {{mean_fpr}}.

> Screenshots / demo GIF: record your own and drop it here (the Streamlit app runs with `make app`).

## Architecture

```mermaid
flowchart LR
  D[("MVTec AD")] --> S["split: fit 90% / calibration 10%"]
  S --> F["frozen ResNet layer2 + layer3<br/>pool, upsample, concat"]
  F --> C["greedy k-center coreset"]
  C --> B[("memory bank + manifest")]
  S -->|calibration normals| T["threshold"]
  T --> B
  B --> K["nearest-neighbour score<br/>image score = max patch score"]
  K --> P["OpenCV: threshold, open/close,<br/>contours, boxes, overlay"]
  P --> API["FastAPI /v1/inspect"]
  API --> UI["Streamlit app"]
```

The full diagram with module names is in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). A study guide that explains the whole project for learning is in [docs/STUDY_GUIDE.pdf](docs/STUDY_GUIDE.pdf).

## Quick start

```bash
pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements-dev.txt && pip install -e . --no-deps
mvinspect fetch-data && mvinspect train --all && mvinspect evaluate --all
mvinspect serve   # then: make app
```

Trained artifacts are not committed (`artifacts/` is git-ignored), so train first, or see [HANDOFF.md](HANDOFF.md) for exact steps and measured run times. The dataset is not committed either.

## Method summary

- **Backbone.** Torchvision ResNet-50 with ImageNet weights, eval mode, frozen. This is pretrained CNN feature extraction; nothing is fine-tuned. Features come from `layer2` (512 x 28 x 28) and `layer3` (1024 x 14 x 14) at 224 input.
- **Patch descriptors.** 3 x 3 average pooling for local aggregation, bilinear upsampling of `layer3`, concatenation: 784 descriptors of 1536 dimensions per image.
- **Memory bank.** Approximate greedy k-center selection (distances computed in a fixed 128-dimensional Gaussian projection) keeps 10 percent of the fit-split patches. Original descriptors are stored.
- **Scoring.** Patch score = Euclidean distance to the nearest bank entry; image score = maximum patch score; anomaly map = patch scores upsampled bilinearly to 224 x 224 and Gaussian blurred (sigma 4).
- **Threshold.** The maximum image score over calibration normals (10 percent of `train/good`, at least 10 images). No test labels are used. The pixel threshold is the same value. With about 20 to 30 calibration images this is noisy, so the calibrated operating point is always shown next to threshold-free AUROC and the oracle point.
- **One model per category.** Matches a station that inspects one product type; the API takes the category as input.
- **Input.** Resize to 256, center crop 224, ImageNet normalisation; masks use the same geometry with nearest-neighbour resizing. Center cropping can remove defects at the image border.
- **No PatchCore library.** Everything above is implemented here (`src/inspection`); `anomalib` and similar libraries are not used. ImageNet weights are downloaded from the PyTorch model zoo on first use and their checksum is stored in every artifact manifest.
- **Why these choices:** see [docs/DECISIONS.md](docs/DECISIONS.md).

## Results

Dataset facts verified by `mvinspect verify-data`: 15 categories, 5354 images (3629 train, 1725 test), the test split holds 467 defect-free and 1258 defective images. **About 73 percent of the test set is defective**, unlike a real line, which matters for precision and for the cascade analysis. All numbers below are generated from `results/` by `scripts/render_docs.py`.

### Per-category accuracy and localisation

Image AUROC has a 95 percent bootstrap confidence interval (1000 resamples, seed 0). Pixel AUROC is over all pixels at 224 x 224. AU-PRO is a re-implementation at 224 x 224 (up to FPR 0.3, 200 thresholds), so it may differ slightly from the official evaluation code. Precision, recall, F1 and the FPR on test normals are at the calibrated threshold; "oracle F1" is the best F1 over thresholds chosen with test labels (analysis only).

<!--TABLE:per_category-->

Categories at image AUROC of at least 0.999: {{perfect}}. The sanity expectation (bottle, leather and tile near perfect; screw and capsule among the lowest) holds for bottle and leather and tile (0.988); screw is the lowest, capsule is in the lower group together with pill and grid.

The calibrated operating point is much less reliable than the AUROC: on carpet and leather it flags a large share of normal test images (FPR column) because the threshold is the maximum over only a few calibration images, while on screw it is so conservative that recall is low although AUROC is high. This is the cost of a label-free threshold from a small calibration set, and the reason the oracle column is shown.

### Comparison with published PatchCore numbers

Published values are from Roth et al., "Towards Total Recall in Industrial Anomaly Detection", CVPR 2022 (arXiv 2106.08265), supplementary tables, 10 percent coreset. That paper uses a WideResNet-50 backbone and different settings, so the comparison is indicative only. The values were read from an HTML rendering through a summarising tool and should be checked against the PDF before being quoted precisely (see DECISIONS D10). Values are in percent.

<!--TABLE:baselines-->

Mean difference to the published numbers: {{cmp_image_delta}} points image AUROC and {{cmp_pixel_delta}} points pixel AUROC. {{cmp_within1}} of 15 categories are within one point on image AUROC; the largest gaps are {{cmp_worst}}. The differences are small and of mixed sign on pixel AUROC for a few categories, which is what one would expect from a different backbone and implementation details (see DECISIONS D7).

### Latency, throughput and memory

One image at a time, model only (decoded tensor to score and map), after 5 warm-up images, 60 images spread over the test split, PyTorch with exact nearest-neighbour search (a chunked matrix product, no approximation), 4 threads, on a laptop with an {{cpu}} ({{cores}}, {{ram_gb}} GB RAM; no discrete GPU, and the integrated graphics were not used). Memory is process resident memory after model load and the peak during the latency run. The largest peak over categories is {{max_peak_rss_mb}} MB, well below the 5 GB budget. Memory bank sizes range from {{min_bank_mb}} to {{max_bank_mb}} MB.

<!--TABLE:latency-->

Total measured runtimes in seconds (training ran in the background alongside other work, so training times are upper bounds):

<!--TABLE:runtime-->

Sum of training times: {{train_total_min}} minutes; sum of evaluation times: {{eval_total_min}} minutes. Bottle was retrained and re-evaluated in a later session after a timing bug was fixed (see docs/KNOWN_ISSUES.md), so its row comes from that session; its accuracy is unchanged.

#### Inference backends

The feature extractor (including pooling, upsampling and concatenation) exports to ONNX with a maximum absolute feature difference far below the 1e-3 target (see `results/inference_benchmark.json`). Two benchmark runs on bottle, model only, p50 and p95 in milliseconds:

<!--TABLE:benchmark-->

ONNX Runtime plus faiss was fastest in both runs, by a modest margin, so the API serves with those backends by default (config `api.extractor_backend` and `api.knn_backend`, with automatic fallback to PyTorch). Training and the reported evaluation use PyTorch for reproducibility.

#### API load test

Real bottle images, client concurrency 1, 2 and 4, against servers with 1 and 2 inference slots (`scripts/load_test.py`, raw files `results/load_test_*.json`). Latency is end-to-end including upload and decoding:

<!--TABLE:loadtest-->

### OpenCV post-processing

Flagged images are binarised at the pixel threshold, cleaned with a morphological opening then closing (5 x 5 ellipse), contours below 30 pixels are dropped and `boundingRect` gives the boxes. Quantified against ground truth masks on defective test images that the model flagged. A box counts as a hit when its IoU with the bounding rectangle of a ground truth region is at least 0.1. Mean over categories: IoU {{pp_raw_iou}} raw against {{pp_clean_iou}} cleaned, Dice {{pp_raw_dice}} raw against {{pp_clean_dice}} cleaned, box hit rate {{pp_hit}}, box precision {{pp_boxprec}}.

**The morphological cleanup does not help measurably**: the raw thresholded map is already smooth because of the Gaussian blur, so opening and closing change almost nothing. Its practical value is limited to dropping tiny contours when boxes are drawn. Figures: `results/figures/postprocess/`.

<!--TABLE:postprocess-->

Alpha sensitivity (pixel threshold multiplied by alpha, averaged over categories; analysis only, never used to set the deployed value):

<!--TABLE:alpha-->

### Two-stage cascade

Stage 1 is ResNet-18 at 128 x 128 with global-average-pooled embeddings and nearest fit-split embedding distance. Images scoring at or below the q-th percentile of calibration normals skip PatchCore. The quantile was never tuned on test data. Skipped images get the lowest possible score for AUROC. Mean latency is computed from the measured per-stage mean latencies and the measured skip fractions (not timing the combined process). This is an engineering trade-off built and measured here on a benchmark whose test set is 73 percent defective.

Mean over categories (q = 0 is PatchCore alone):

<!--TABLE:cascade-->

At q = 50 the cascade skips {{casc50_normal_skipped}} of normal test images and wrongly skips {{casc50_def_skipped}} of defective ones for a speed-up of only {{casc50_speedup}} on this defect-heavy benchmark mix; at q = 75 it wrongly skips {{casc75_def_skipped}} of defects. Where stage 1 fails is visible in the per-category table below (`defective_wrongly_skipped`).

Per category at q = 50:

<!--TABLE:cascade_cat-->

Production-mix calculation (a calculation from the measured per-class skip rates and latencies, reweighted to defect rates of 1, 5 and 20 percent, not a measurement):

<!--TABLE:mix-->

### Failure analysis

The three weakest categories by image AUROC. Figures with the worst false negatives (lowest-scoring defective images) and worst false positives are in `results/figures/failures/`.

<!--TABLE:failures-->

What the three failure cases show (numbers from the tables above, figures in `results/figures/failures/`):

- **screw.** Image AUROC is {{cat|screw|image_auroc}} but recall at the calibrated threshold is only {{cat|screw|recall}} with a false positive rate of {{cat|screw|fpr_normals}}: F1 {{cat|screw|f1}} against an oracle F1 of {{cat|screw|oracle_f1}}. The ranking of images is good; the label-free threshold sits above about half of the defective scores. Interpretation (not tested directly): screws lie in arbitrary rotations, so normal patches vary more than on aligned products, which pushes the calibration maximum up. The few-shot study below is consistent with this: screw is the one category whose AUROC keeps rising until nearly all training images are used.
- **pill.** The most frequently missed defect type is crack, and the anomaly maps of the missed cracks and scratches are only weakly lit. The single false positive is a normal pill with dense dark speckle that lands just above the threshold.
- **grid.** Only three defective images are missed (all glue) and three normal images are flagged, all close to the threshold. The grids are photographed at visibly different tilts, a source of normal variation that the memory bank has to cover.

In all three categories the misses and false alarms sit near the threshold rather than far from it, so the oracle column quantifies how much of the loss comes from the threshold and how much from the scores.

## Studies (leather, hazelnut, screw)

The studies are not needed to use the system. They ask narrow questions about the trained models on three categories of different character (a uniform texture, a natural object, and a part photographed in arbitrary orientation).

**How they were run.** The studies ran on a free Colab T4 GPU (`device: cuda`, driven by `scripts/colab_run.py` and `notebooks/colab_studies.ipynb`) instead of this CPU laptop, because the coreset step makes them slow on a CPU. Latency is therefore not measured in the studies, except for the coreset ratio banks, which were rebuilt from the saved selection order and timed on the CPU. To check that the GPU path does not change results, the GPU baseline (same config, split and seed as the laptop run) is compared with the CPU main run:

<!--TABLE:parity-->

Image AUROC, the number of bank rows and the calibrated threshold are identical, and pixel AUROC differs only in the sixth decimal. In addition, {{parity_rob_n}} leather robustness settings were computed on both devices (a CPU run stopped when the studies moved to the GPU, kept in `results/studies_cpu_partial/`); their largest AUROC difference is {{parity_rob_diff}}. The studies took about 19 minutes in total on the GPU (read from the Colab console, not stored in `results/`). Earlier measurements of the same coreset step on this CPU suggested several hours, which is an estimate and not a measurement.

### Seed spread (the yardstick for every ablation)

Seed changes the calibration split and the coreset starting point. Differences smaller than this spread are not meaningful. Image AUROC range over 3 seeds: leather {{spread|leather|image_auroc_range}}, hazelnut {{spread|hazelnut|image_auroc_range}}, screw {{spread|screw|image_auroc_range}}.

<!--TABLE:seeds-->

The calibrated false positive rate moves a lot with the seed even when AUROC does not: leather ranges over {{fprrange|leather}} across the three calibration splits, hazelnut over {{fprrange|hazelnut}} and screw over {{fprrange|screw}}, which means their thresholds sat above every normal test image on every seed. A threshold from about 20 to 30 calibration images is the weakest part of this pipeline.

### Coreset ratio

<!--TABLE:ratio-->

Bank size grows in proportion to the ratio. CPU latency grows with it on top of a fixed cost that the 1 percent row shows (feature extraction). The latency column was measured in one session and is not comparable with the main latency table, which was measured in a different session. Accuracy does not need a large bank: the 25 percent bank gives no gain over 10 percent on any of the three categories, and the 1 percent bank is equal on leather and hazelnut. Screw is the exception: its image AUROC falls from {{abl|screw|coreset@0.1|image_auroc}} at 10 percent to {{abl|screw|coreset@0.01|image_auroc}} at 1 percent, slightly more than the seed spread, so a small bank costs accuracy where the normal data is diverse.

### Preprocessing and backbone

Preprocessing and backbone variants against the baseline (the main run). Coreset ratios and seeds are in the tables above.

<!--TABLE:ablation-->

- Gaussian blur (3 x 3, sigma 1) and the grayscale opening change nothing beyond the seed spread, with one small exception: hazelnut with the opening, {{abl|hazelnut|pre_gray_opening_3x3|image_auroc}} against 1.000.
- The CIELAB transform hurts: leather image AUROC drops to {{abl|leather|pre_cielab|image_auroc}} and screw to {{abl|screw|pre_cielab|image_auroc}}, both beyond the seed spread. A plausible reason is that the pretrained backbone expects RGB statistics and receives LAB channels as if they were RGB. That explanation was not tested.
- Wide ResNet-50-2 against ResNet-50: no meaningful difference (screw {{abl|screw|backbone_wide_resnet50_2|image_auroc}} against {{abl|screw|coreset@0.1|image_auroc}}, within the seed spread).

### Few-shot: how many normal images are needed

Memory bank built from N normal images (three seeds, which also change the calibration split; the "all" row is the main run plus the two seed baselines). Spread is min and max over seeds.

<!--TABLE:fewshot-->

![Few-shot AUROC against N](results/figures/fewshot_auroc.png)

Leather is solved by a single image (AUROC {{few|leather|1|auroc_mean}}), as expected for a uniform texture, although its calibrated false positive rate stays high. Hazelnut needs about ten images: {{few|hazelnut|1|auroc_mean}} at N = 1, {{few|hazelnut|10|auroc_mean}} at N = 10, {{few|hazelnut|25|auroc_mean}} at N = 25. Screw is far behind even at 50 images ({{few|screw|50|auroc_mean}}, against {{few|screw|all|auroc_mean}} with the full fit split), and at N = 1 it is below chance ({{few|screw|1|auroc_mean}}), meaning normal test images score higher on average than defective ones. One reference image of a part that appears in many poses is a poor reference (interpretation, not tested directly). For bank sizes below 1000 rows the bank holds all patches of the chosen images (see DECISIONS D5).

### Robustness to image perturbations

The model and the calibrated threshold stay fixed; test images are perturbed after cropping. Each cell is image AUROC (change from the unperturbed run) and the false positive rate on normals at the fixed threshold.

![Robustness of AUROC and false positive rate](results/figures/robustness.png)

<!--TABLE:robustness-->

- **AUROC can survive while the threshold does not.** On hazelnut and leather, blur, noise and JPEG leave AUROC high (hazelnut with blur sigma 2: {{rob|hazelnut|gaussian_blur|sigma2|image_auroc}}) but push the false positive rate to {{rob|hazelnut|gaussian_blur|sigma2|fpr_normals}}: every normal image is rejected. A fixed threshold is useless once focus, noise or compression change; the ranking would need a recalibration.
- **Screw loses the ranking as well.** AUROC under JPEG quality 30 is {{rob|screw|jpeg|q30|image_auroc}} and under blur sigma 1 {{rob|screw|gaussian_blur|sigma1|image_auroc}}, which is below chance for JPEG and about chance for blur. A likely reason is that the defects are fine detail that compression and blur remove (not tested directly).
- **Geometry.** Hazelnut is unaffected by rotation and translation, screw degrades gradually (rotation of 10 degrees: {{rob|screw|rotation|10deg|image_auroc}}), and leather keeps a high AUROC but its false positive rate rises quickly because the unperturbed false positive rate was already {{rob|leather|none|-|fpr_normals}}.
- **Brightness.** Darkening by 20 percent (x0.8) is tolerated by all three. Brightening by 20 percent (x1.2) already breaks leather's threshold (false positive rate {{rob|leather|brightness|x1.2|fpr_normals}}) while hazelnut and screw tolerate it.
- **Practical reading:** exposure, focus and lossy compression in the image path are the real risks. A deployment would need to hold them constant or recalibrate the threshold when they change.

## Limitations

- Benchmark images are aligned and well lit, so results do not transfer automatically to a real line.
- Per-category models require knowing the product type.
- Center cropping (resize 256, crop 224) can cut off border defects.
- The test set is about 73 percent defective, unlike a real line; precision and the cascade speed-up depend on that mix.
- CPU measurements come from one laptop under the load described in `docs/KNOWN_ISSUES.md`.
- The ablation, few-shot and robustness studies cover three categories and ran on a GPU (results checked against the CPU run).
- The cascade is evaluated on a benchmark and may not generalise.
- No testing on non-MVTec images.
- AU-PRO is a re-implementation, not the official code.
- The calibrated threshold comes from about 20 to 30 images and is noisy.
- The dataset license is non-commercial.
- The CI workflow was written and parses as YAML but has not run (no GitHub runner here). The Docker images were built and tested locally on one machine only.

## API usage

Start the service (artifacts must exist; the first request for a category loads it):

```bash
mvinspect serve --port 8000
curl -s localhost:8000/v1/health
curl -s localhost:8000/v1/ready
curl -s localhost:8000/v1/categories
curl -s -X POST "localhost:8000/v1/inspect?category=bottle" \
  -F "file=@data/mvtec_ad/bottle/test/broken_large/000.png" | python -m json.tool | head -40
curl -s -X POST "localhost:8000/v1/inspect?category=bottle&use_cascade=true" \
  -F "file=@data/mvtec_ad/bottle/test/good/000.png"
curl -s localhost:8000/metrics | head
```

Response fields: `category`, `anomaly_score`, `threshold`, `is_anomalous`, `stage_reached`, `latency` (`model_ms`, `end_to_end_ms`), `boxes` (x, y, w, h, score in the 224 x 224 model frame), `heatmap_png_base64`, `annotated_png_base64`, `model_version` (manifest hash), `request_id`. Errors use `{"error": {"code", "message", "request_id"}}` with no stack traces. Set `MVI_API__API_KEY` to require an `X-API-Key` header; CORS is disabled unless `MVI_API__CORS_ORIGINS` is set. Uploads are decoded with Pillow, limited by size (default 10 MB), pixel count and side length, and limited to PNG, JPEG and BMP.

## Docker

```bash
docker compose up --build   # API on :8000, Streamlit on :8501
```

Images use multi-stage builds, CPU-only torch, a non-root user, a pinned base tag and a health check. Data and artifacts are mounted read-only and are not baked in (`./artifacts`, `./results`, `./data/mvtec_ad`). The ImageNet weights are baked in, since they are not dataset content; the manifest checksum is verified when a model loads.

Built and tested with Docker {{docker_version}} on Docker Desktop (WSL 2 backend) by `scripts/docker_verify.py`, which writes `results/docker_verification.json`: both containers start and report healthy, `/v1/ready` lists all 15 categories, a real defective bottle image gets the same anomaly score inside the container as on the host ({{docker_score}}), a path-traversal category is rejected with a 404 error body, the API runs as a non-root user on a read-only root filesystem, and the app container reaches the API over the compose network. Image sizes: API {{docker_api_gb}} GB, app {{docker_app_mb}} MB (uncompressed, as reported by Docker). The API container used {{docker_api_mem}} after serving two categories. Not tested: a Linux build on a clean machine other than this one, and the CI workflow that builds the same files.

## Reproduction

1. Python 3.10 or newer; install pinned dependencies (see the first quick-start line).
2. `mvinspect fetch-data` then `mvinspect verify-data`. Archives already in `data/raw/` are used first, without network access; only missing categories are downloaded from the official page. As of 22 October 2026 the links on that page returned HTTP 404, so the archives may have to be placed in `data/raw/` by hand (see HANDOFF.md).
3. `mvinspect train --all`, `mvinspect evaluate --all`, `mvinspect postprocess --all`, `mvinspect cascade --all`.
4. Studies: `mvinspect studies robustness leather hazelnut screw`, `mvinspect studies fewshot leather hazelnut screw`, `mvinspect ablate leather hazelnut screw`. These take hours on a CPU. On a GPU (for example a free Colab T4) `python scripts/colab_run.py` does steps 2 to 4 for the three study categories in about 20 minutes; copy `results/studies/` back and run `python scripts/parity_from_studies.py` and `mvinspect ablate --cpu-latency`.
5. `mvinspect report`, then `python scripts/render_docs.py` to regenerate this README from `results/`.
6. Measured on the build machine: training {{train_total_min}} minutes in total, evaluation {{eval_total_min}} minutes in total (see the runtime table above; studies timings are in HANDOFF.md).

Everything is seeded (seed 0 unless a study states otherwise) and each category result is resumable.

## Dataset license, citation, references

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial, attribution, share-alike). This is a non-commercial portfolio project; the dataset itself is not redistributed and is not in this repository. The figures under `results/figures/` contain small excerpts of MVTec AD images (sample grids, overlays, failure cases) and are shared under the same CC BY-NC-SA 4.0 terms, with attribution to MVTec Software GmbH.

- Bergmann, Fauser, Sattlegger, Steger. "MVTec AD: A Comprehensive Real-World Dataset for Unsupervised Anomaly Detection." CVPR 2019.
- Roth, Pemula, Zepeda, Scholkopf, Brox, Gehler. "Towards Total Recall in Industrial Anomaly Detection." CVPR 2022, arXiv:2106.08265.
