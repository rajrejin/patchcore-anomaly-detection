# Model card: PatchCore per-category inspection models (MVTec AD)

## Intended use

Research, study and portfolio demonstration of unsupervised visual anomaly detection and localisation on the MVTec AD benchmark. Not validated for any real production line and not suitable for safety-relevant decisions.

## Model

One PatchCore model per MVTec AD category. A frozen ImageNet-pretrained ResNet-50 (pretrained CNN feature extraction, no fine-tuning) provides layer2 and layer3 patch descriptors (1536 dimensions, 784 patches per 224 x 224 input). A greedy k-center coreset (10 percent of fit-split patches) forms the memory bank. The anomaly score of a patch is its Euclidean distance to the nearest bank entry; the image score is the maximum patch score. Implemented from scratch in this repository.

## Data

MVTec AD (Bergmann et al., CVPR 2019), license CC BY-NC-SA 4.0 (non-commercial). 15 categories, 5354 images: 3629 defect-free training images, 1725 test images (467 defect-free, 1258 defective, so about 73 percent of the test set is defective). Training uses `train/good` only: 90 percent to fit the memory bank, 10 percent (at least 10 images) to calibrate the threshold. The test split is used only for evaluation. Dataset fingerprints are stored in each artifact manifest.

## Training procedure

`mvinspect train --all`: split with seed 0, extract float16 descriptors, greedy coreset with a 128-dimensional Gaussian projection, threshold = maximum image score over calibration normals. Total training time on an AMD Ryzen 5 3500U with Radeon Vega Mobile Gfx (4 physical, 8 logical, 6.94 GB RAM, CPU only): 92 minutes (background, shared CPU, upper bound). Python 3.13.14, torch 2.14.1+cpu.

## Metrics per category

Generated from `results/metrics_per_category.csv`. Mean image AUROC 0.985, mean pixel AUROC 0.977, mean AU-PRO 0.906 (re-implementation at 224 x 224).

| category | image AUROC (95% CI) | pixel AUROC | AU-PRO | precision | recall | f1 | FPR on normals | oracle F1 |
|---|---|---|---|---|---|---|---|---|
| bottle | 1.000 (1.000 to 1.000) | 0.983 | 0.929 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| cable | 0.994 (0.984 to 0.999) | 0.985 | 0.922 | 0.967 | 0.967 | 0.967 | 0.052 | 0.979 |
| capsule | 0.968 (0.929 to 0.994) | 0.987 | 0.916 | 0.978 | 0.835 | 0.901 | 0.087 | 0.977 |
| carpet | 0.969 (0.935 to 0.992) | 0.985 | 0.912 | 0.888 | 0.978 | 0.930 | 0.393 | 0.948 |
| grid | 0.964 (0.919 to 0.996) | 0.969 | 0.874 | 0.947 | 0.947 | 0.947 | 0.143 | 0.957 |
| hazelnut | 1.000 (1.000 to 1.000) | 0.984 | 0.928 | 1.000 | 1.000 | 1.000 | 0.000 | 1.000 |
| leather | 1.000 (1.000 to 1.000) | 0.989 | 0.961 | 0.893 | 1.000 | 0.944 | 0.344 | 1.000 |
| metal_nut | 0.997 (0.988 to 1.000) | 0.986 | 0.921 | 0.989 | 0.989 | 0.989 | 0.045 | 0.989 |
| pill | 0.956 (0.923 to 0.984) | 0.973 | 0.922 | 0.991 | 0.787 | 0.877 | 0.038 | 0.952 |
| screw | 0.950 (0.915 to 0.977) | 0.990 | 0.944 | 1.000 | 0.487 | 0.655 | 0.000 | 0.934 |
| tile | 0.988 (0.964 to 1.000) | 0.951 | 0.790 | 1.000 | 0.905 | 0.950 | 0.000 | 0.988 |
| toothbrush | 1.000 (1.000 to 1.000) | 0.987 | 0.860 | 0.968 | 1.000 | 0.984 | 0.083 | 1.000 |
| transistor | 1.000 (0.997 to 1.000) | 0.967 | 0.940 | 0.889 | 1.000 | 0.941 | 0.083 | 0.988 |
| wood | 0.989 (0.970 to 1.000) | 0.941 | 0.839 | 0.983 | 0.967 | 0.975 | 0.053 | 0.975 |
| zipper | 0.997 (0.990 to 1.000) | 0.983 | 0.928 | 0.975 | 0.992 | 0.983 | 0.094 | 0.988 |
| MEAN | 0.985 (0.968 to 0.996) | 0.977 | 0.906 | 0.965 | 0.924 | 0.936 | 0.094 | 0.978 |

Latency, memory:

| category | p50 ms | p95 ms | fps | rss after load mb | peak rss mb | bank mb |
|---|---|---|---|---|---|---|
| bottle | 667.3 | 750.0 | 1.5 | 651.4 | 752.6 | 86.4 |
| cable | 685.9 | 718.4 | 1.5 | 727.4 | 833.1 | 92.8 |
| capsule | 600.1 | 748.6 | 1.6 | 747.5 | 796.0 | 90.5 |
| carpet | 635.4 | 709.5 | 1.5 | 755.6 | 883.4 | 115.8 |
| grid | 622.1 | 659.4 | 1.6 | 803.5 | 843.9 | 109.3 |
| hazelnut | 870.8 | 907.8 | 1.1 | 869.9 | 982.0 | 161.7 |
| leather | 590.4 | 657.7 | 1.7 | 741.5 | 835.8 | 101.5 |
| metal_nut | 544.6 | 567.8 | 1.8 | 683.9 | 817.5 | 91.0 |
| pill | 626.1 | 657.9 | 1.6 | 801.5 | 890.6 | 110.2 |
| screw | 735.4 | 772.8 | 1.4 | 834.3 | 906.2 | 132.3 |
| tile | 557.7 | 590.7 | 1.8 | 772.5 | 834.5 | 95.1 |
| toothbrush | 225.9 | 251.3 | 4.3 | 593.1 | 607.9 | 23.0 |
| transistor | 527.2 | 552.9 | 1.9 | 671.5 | 813.7 | 88.2 |
| wood | 590.4 | 624.9 | 1.7 | 795.5 | 832.8 | 102.0 |
| zipper | 729.0 | 788.2 | 1.4 | 748.1 | 856.5 | 99.2 |
| MEAN | 613.9 | 663.9 | 1.8 | 746.5 | 832.4 | 99.9 |

## Limitations and failure modes

- Benchmark images are aligned and well lit; real lines have drift, glare, dirt and misalignment.
- One model per product type; a wrong category input gives meaningless scores.
- Center crop (256 to 224) can remove border defects.
- The calibrated threshold comes from a handful of images. On several categories it either flags many normal images or misses many defects (compare the FPR and recall columns with the oracle F1 column).
- Weakest categories: screw (0.950), pill (0.956), grid (0.964). See the failure analysis in the README and `results/figures/failures/`.
- Perturbation robustness and few-shot behaviour are reported in the README for three categories only.
- The cascade trade-off was measured on a defect-heavy benchmark mix.
- Defect localisation: mean box hit rate 0.843, mean IoU of the thresholded map 0.330.

## What would be needed to deploy on real production data

- Images from the actual line and camera, including defect-free samples across shifts, lots and lighting, and a labelled validation set to choose the operating point (instead of the label-free calibration used here).
- Alignment or augmentation strategy for parts that are not fixtured (screw-like categories suffer from pose variation).
- Threshold review per category against the real cost of missed defects against false rejects, and a monitoring plan for drift (score distribution of accepted parts).
- Latency and throughput verified on the target hardware at the line rate; the measured throughput here is about 1.76 images per second on one CPU core set.
- A defect-rate-aware evaluation (real lines are mostly defect-free) and an operator review loop.
- Licensing: MVTec AD is non-commercial, so the benchmark data cannot be reused commercially.
