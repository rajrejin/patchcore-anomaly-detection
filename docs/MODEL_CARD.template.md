# Model card: PatchCore per-category inspection models (MVTec AD)

## Intended use

Research, study and portfolio demonstration of unsupervised visual anomaly detection and localisation on the MVTec AD benchmark. Not validated for any real production line and not suitable for safety-relevant decisions.

## Model

One PatchCore model per MVTec AD category. A frozen ImageNet-pretrained ResNet-50 (pretrained CNN feature extraction, no fine-tuning) provides layer2 and layer3 patch descriptors (1536 dimensions, 784 patches per 224 x 224 input). A greedy k-center coreset (10 percent of fit-split patches) forms the memory bank. The anomaly score of a patch is its Euclidean distance to the nearest bank entry; the image score is the maximum patch score. Implemented from scratch in this repository.

## Data

MVTec AD (Bergmann et al., CVPR 2019), license CC BY-NC-SA 4.0 (non-commercial). 15 categories, 5354 images: 3629 defect-free training images, 1725 test images (467 defect-free, 1258 defective, so about 73 percent of the test set is defective). Training uses `train/good` only: 90 percent to fit the memory bank, 10 percent (at least 10 images) to calibrate the threshold. The test split is used only for evaluation. Dataset fingerprints are stored in each artifact manifest.

## Training procedure

`mvinspect train --all`: split with seed 0, extract float16 descriptors, greedy coreset with a 128-dimensional Gaussian projection, threshold = maximum image score over calibration normals. Total training time on an {{cpu}} ({{cores}}, {{ram_gb}} GB RAM, CPU only): {{train_total_min}} minutes (background, shared CPU, upper bound). Python {{python}}, torch {{torch}}.

## Metrics per category

Generated from `results/metrics_per_category.csv`. Mean image AUROC {{mean_image_auroc}}, mean pixel AUROC {{mean_pixel_auroc}}, mean AU-PRO {{mean_au_pro}} (re-implementation at 224 x 224).

<!--TABLE:per_category-->

Latency, memory:

<!--TABLE:latency-->

## Limitations and failure modes

- Benchmark images are aligned and well lit; real lines have drift, glare, dirt and misalignment.
- One model per product type; a wrong category input gives meaningless scores.
- Center crop (256 to 224) can remove border defects.
- The calibrated threshold comes from a handful of images. On several categories it either flags many normal images or misses many defects (compare the FPR and recall columns with the oracle F1 column).
- Weakest categories: {{weakest}}. See the failure analysis in the README and `results/figures/failures/`.
- Perturbation robustness and few-shot behaviour are reported in the README for three categories only.
- The cascade trade-off was measured on a defect-heavy benchmark mix.
- Defect localisation: mean box hit rate {{pp_hit}}, mean IoU of the thresholded map {{pp_raw_iou}}.

## What would be needed to deploy on real production data

- Images from the actual line and camera, including defect-free samples across shifts, lots and lighting, and a labelled validation set to choose the operating point (instead of the label-free calibration used here).
- Alignment or augmentation strategy for parts that are not fixtured (screw-like categories suffer from pose variation).
- Threshold review per category against the real cost of missed defects against false rejects, and a monitoring plan for drift (score distribution of accepted parts).
- Latency and throughput verified on the target hardware at the line rate; the measured throughput here is about {{mean_fps}} images per second on one CPU core set.
- A defect-rate-aware evaluation (real lines are mostly defect-free) and an operator review loop.
- Licensing: MVTec AD is non-commercial, so the benchmark data cannot be reused commercially.
