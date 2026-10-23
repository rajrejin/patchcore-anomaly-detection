# Known issues

Limitations of this environment or of the method, with what was tried.

## K1. Docker is verified on one machine only

The images build and the stack runs on the build machine (Docker Desktop 4.94, WSL 2 backend), see `results/docker_verification.json`. Not tested: a build on a clean Linux host, and the CI job that builds the same Dockerfiles. The ONNX file and faiss are used inside the container only because `artifacts/_onnx` is mounted with the artifacts; without it the service falls back to PyTorch (slower start, same results).

## K2. CI workflow has not run

`.github/workflows/ci.yml` parses as YAML. It cannot execute here. On a clean runner the pinned `scipy`, `pandas` and `scikit-learn` (DECISIONS D1) should install from wheels, but this was not tested.

## K3. Pixel threshold can leave flagged images without a box

See DECISIONS D4. Images whose score only just exceeds the threshold may have an anomaly map that never exceeds the same threshold after blurring, so no box is produced. The alpha sensitivity tables quantify what other pixel thresholds would do; they were not used to change the deployed value.

## K4. Baseline numbers were read through a summarising fetch

See DECISIONS D10. The published per-class values should be checked against the PDF before being quoted precisely.

## K5. Timing caveats

Timings are from one Ryzen 5 3500U laptop. Training of the 15 categories ran in the background while other work shared the CPU, so the training times in `results/runtime.json` are upper bounds. Bottle was retrained and re-evaluated in a later session (after the timing fix in K9), so its training time, evaluation time and latency row come from that session. The latency numbers in `results/categories/*.json` come from a separate evaluation pass on an otherwise idle machine. The latency column of the coreset ratio table was measured in a different session and is 20 to 30 percent higher than the main table for the same bank size (DECISIONS D16). The Colab study time (about 19 minutes) was read from the console and is not stored in `results/`.

## K6. Studies cover three categories and ran on a GPU

The ablation, few-shot and robustness studies cover leather, hazelnut and screw only, and ran on a Colab T4. The GPU path matched the CPU main run to the precision reported in `results/parity_cpu_vs_gpu.json`, but that comparison covers only those three categories and one configuration. The CUDA unit tests are skipped on machines without a GPU, so CI would not exercise them.

## K7. Calibrated thresholds are noisy

The threshold is the maximum over 10 to about 30 calibration images. In the seed study the false positive rate on leather moved between 0.25 and 0.81 across three calibration splits with unchanged AUROC. This is a property of the specified method, not a bug. A deployment should calibrate on far more normal images and validate the operating point against labelled data.

## K8. A scripted file-drop into Google Drive does not work

During the Colab setup, a synthetic drop event on the Drive page was ignored by the browser (untrusted events), so the bundle went into the notebook as text instead. Nothing was uploaded to Drive by that attempt.

## K9. Fixed in the final audit

These were found by reviewing the code and are fixed, each with a test:

- Re-running a resumable command (`mvinspect train --all`, `mvinspect evaluate --all`) wrote the time of the skip into `results/runtime.json`, replacing the real measurement. Times are now recorded only when work is actually done. This is why the bottle training time was missing before the audit.
- `evaluate` reused a stored result whenever the config hash matched, even after the model had been retrained. It now also compares the model version (manifest hash).
- Post-processing, the cascade and the studies rebuilt a missing test-score cache by running the full evaluation, which silently overwrote the headline result file and its latency numbers. The cache is now rebuilt without touching that file.
- `fetch-data` demanded 12 GB of free disk even for one category and read the downloads page even when nothing was missing. It now asks for about 2 GB per missing category and does nothing when all are present.
- The feature cache of the studies did not include the model settings in its signature, so a config change could reuse stale features.
- After a request timeout the API released its concurrency slot while the inference thread was still running, which made the queue-wait metric undercount. The slot is now released when the thread finishes.
- The bootstrap confidence interval crashed on one-class input instead of returning no value.
- The CI workflow would have failed its coverage limit on GitHub because no data is available there; it now downloads and trains bottle first (DECISIONS D18).
- Coverage no longer excludes `cli.py` and `studies/`; both are now measured.
- `fetch-data` required the online link list before looking at archives in `data/raw/`, so archives placed by hand could not be used; local archives are now used first, without network access. It also deleted archives placed by the user after extracting them; only archives it downloaded itself are deleted now.
- A permanent download error such as HTTP 404 was retried six times with back-off and ended with no guidance; it now fails at once and says how to place the archives manually.
- Git for Windows treated the PDF as text and would have rewritten its line endings on checkout; `.gitattributes` now marks binary formats.

## K10. The official download links returned HTTP 404 on 22 October 2026

All 16 archive links on the official MVTec AD downloads page (15 categories and the full archive, hosted on mydrive.ch) answered HTTP 404 to any client during the final fresh-clone check; on 10 October the same links served all 15 categories. The page itself shows no notice. `mvinspect fetch-data` now fails at once with instructions, and archives placed in `data/raw/` are used without the network. The fresh-clone check was completed by placing a bottle archive built from the local copy of the dataset in `data/raw/`. Consequence for CI: its download step (DECISIONS D18) will fail while the links are broken, so the CI run would fail rather than pass with the coverage limit silently lowered.
