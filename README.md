# DiskGenerator

DiskGenerator creates synthetic protoplanetary disk images and their simulated radio-interferometric measurements. It generates ground-truth images, ALMA-like UV coverages, noisy visibilities, CASA Measurement Sets, and dirty images used by R2D2 and the companion [RI-Benchmark](https://github.com/MauroRitacco/RI-Benchmark).

## Overview

Radio interferometers sample Fourier components of sky brightness rather than directly measuring images. Reconstructing an image from these measurements is therefore an inverse problem.

DiskGenerator implements the data-generation stage of a thesis research project on interferometric image reconstruction. Given controlled parametric models, it produces:

- ground-truth disk images in FITS format, with model parameters recorded in the headers;
- simulated UV coverages and CASA Measurement Sets (MS);
- complex visibilities with additive Gaussian noise, stored in MAT files and copied into the MS `DATA` column for benchmark and sweep workflows;
- PSF-normalized dirty images, used as initial residuals by R2D2.

The generated data support two downstream workflows:

1. training and validation of the R2D2 reconstruction network;
2. controlled sweep and held-out test datasets for [RI-Benchmark](https://github.com/MauroRitacco/RI-Benchmark), which compares CLEAN (`tclean`), WSClean, FISTA-wavelet reconstruction through DeepInverse, and R2D2.

## Repository structure

```text
DiskGenerator/
├── README.md, LICENSE, .gitignore, requirements.txt
├── config.yaml                     # NUFFT and Briggs-weighting settings
├── run_pipeline.sh                 # training-set pipeline
├── run_pipeline_validation.sh      # validation-set pipeline
├── run_pipeline_benchmark.sh       # per-disk layout for RI-Benchmark
├── src/
│   ├── disk_generator.py           # parametric models -> ground-truth FITS
│   ├── simulator.py                # ground truth + UV pattern -> noisy visibilities
│   ├── dirty_image.py              # visibilities -> dirty image and initial residual
│   ├── ms_to_mat.py                # synthetic visibilities -> MS DATA column
│   └── transforms.py               # FITS/WCS utilities
├── utils/
│   └── ri_measurement_operator/    # vendored BASP measurement-operator package
└── studies/sweep_experiment/
    ├── run_pipeline_sweep.sh          # 36 cells × 10 realizations
    ├── run_pipeline_sweep_testcell.sh # single-cell test pipeline
    ├── collect_metadata.py            # per-disk metadata aggregation
    └── validate_sweep.py              # dataset-structure checks
```

## Workflow

The pipelines share four main stages. Benchmark and sweep workflows additionally copy the simulated visibilities into the Measurement Set.

```text
UV pattern + MS -> ground-truth FITS -> noisy visibilities -> dirty image
  simms + CASA      parametric model       NUFFT + noise      back-projection
```

1. **UV pattern and Measurement Set**  
   `utils/ri_measurement_operator/pyutils/sim_vla_ms.py` uses `simms` and CASA to simulate an ALMA C40-5 or C40-8 observation with single-channel, 243 GHz continuum settings. It exports the UV coordinates, reference frequency, and nominal pixel size to `uv_XXXX.mat`, and creates an empty `disk_XXXX.MS`.

2. **Ground-truth generation**  
   `src/disk_generator.py` renders a 64 × 64 disk with 1–4 Gaussian rings, power-law intensity decay, and per-ring positional and orientation jitter. A central star and a compact extra source are optional. Images are normalized to `[0,1]`, use `BUNIT='Jy/pixel'`, and record the sampled parameters in the FITS header. Sweep workflows also write a JSON sidecar. The pixel scale is read from the corresponding UV MAT file, so UV generation must run first.

3. **Visibility simulation**  
   `src/simulator.py` applies a NUFFT measurement operator to the ground truth and adds complex Gaussian noise. The resulting MAT file contains the visibilities `y`, noise weights `nW = 1/σ`, UV coordinates, frequency, and pixel size. Here, `σ` is the standard deviation of the visibility noise.

4. **Dirty-image generation**  
   `src/dirty_image.py` back-projects the weighted visibilities using Briggs weighting with `robust=2`, normalizes by the PSF peak, and writes `disk_XXXX_res.fits`. It also creates an all-zeros `disk_XXXX_rec.fits`, which serves as the initial reconstruction expected by R2D2.

5. **Measurement Set population, for benchmark and sweep workflows**  
   `src/ms_to_mat.py` copies the simulated visibilities into the MS `DATA` column so that CASA and WSClean can image the synthetic measurements.

### Pipeline variants

- **Training:** `run_pipeline.sh` creates the dataset tree consumed by R2D2.
- `run_pipeline_validation.sh` generates the R2D2 validation split and merges it into the training tree.
- **Benchmark:** `run_pipeline_benchmark.sh` creates one directory per disk in the layout expected by RI-Benchmark.
- **Sweep and held-out test:** `run_pipeline_sweep.sh` evaluates a grid over field of view (0.5, 2, or 4 arcsec), number of rings (1, 2, or 4), central star, and extra source. The 36 parameter cells × 10 realizations produce 360 disks per dataset, with per-disk JSON metadata and dataset-level `metadata.json`.

## Requirements

The original workflows used Python 3.10 and the packages listed in `requirements.txt`, including the CASA Python bindings. UV generation also requires `simms`, and merging the validation split uses `rsync`.

## Data

No datasets or generated outputs are included in the repository. All data are generated locally and excluded from version control.

Sweep and test targets follow this naming convention:

```text
disk_s<FOV>_r<rings>_c<star>_x<extra>_<rep>
```

Each target directory contains the ground-truth FITS, Measurement Set, visibility MAT file, dirty-image FITS, and JSON metadata.

## Usage

Run the commands from the repository root.

```text
# Training set: set n, START_ID, and img_size at the top of the script
./run_pipeline.sh

# Validation set, merged into the training tree
./run_pipeline_validation.sh

# Per-disk folders in the layout expected by RI-Benchmark
./run_pipeline_benchmark.sh

# 360-disk sweep dataset and held-out test dataset
./studies/sweep_experiment/run_pipeline_sweep.sh sweep_val
./studies/sweep_experiment/run_pipeline_sweep.sh test

# Check the structure of a generated sweep dataset
python studies/sweep_experiment/validate_sweep.py dataset/sweep_val
```

The main pipeline scripts define `n`, `START_ID`, and `img_size=64` near the top of each file. The sweep script accepts a dataset name, an optional base-seed override, and an optional output root. Modules under `src/` also provide individual command-line interfaces; use `--help` for their options.

## Outputs

| File | Content | Produced by |
| --- | --- | --- |
| `disk_XXXX.fits` | Ground-truth image with model parameters in the FITS header | `disk_generator.py` |
| `disk_XXXX.json` | Sampled model parameters and grid-cell information, in sweep workflows | `disk_generator.py --metadata` |
| `uv_XXXX.mat` | UV sampling and pixel scale | `sim_vla_ms.py` |
| `disk_XXXX.MS` | Measurement Set populated with synthetic visibilities in benchmark and sweep workflows | `sim_vla_ms.py` and `ms_to_mat.py` |
| `disk_XXXX.mat` | Noisy visibilities and their weights | `simulator.py` |
| `disk_XXXX_res.fits` | Dirty image | `dirty_image.py` |
| `disk_XXXX_rec.fits` | All-zeros initial reconstruction for R2D2 | `dirty_image.py` |
| `metadata.json` | Aggregated per-disk metadata | `collect_metadata.py` |

## Results and reproducibility

DiskGenerator produced the datasets used in the thesis: 4,000 training and 600 validation disks for R2D2, plus 360 sweep and 360 held-out test disks for RI-Benchmark. Reconstruction metrics and analyses are available in the companion repository.

Given the same UV metadata, ground-truth generation is reproducible from the recorded seeds. The base seed is `2e9` for `sweep_val` and `3e9` for `test`; each image uses `base + index * 10^6`.

Reproducibility limitations:

- UV pointing and flagging use unseeded random generators.
- Visibility noise uses unseeded `torch.randn`.
- Training and validation pipelines do not fix a seed.
- Consequently, complete generated datasets are not bit-reproducible.
- The repository has no automated test suite or continuous integration.

## Third-party code and local modifications

- `utils/ri_measurement_operator/` is a vendored copy of measurement-operator code from the BASP R2D2 ecosystem; `sim_vla_ms.py` credits A. Dabbech. The corresponding license is included in that directory.
- The local copy adds a `--pixelsize` argument to `sim_vla_ms.py`, allowing an explicit field of view. Sweep pipelines pass `FOV/64`.
- This repository is distributed under GPL-3.0; see `LICENSE`.

## Limitations

- Simulations use single-channel continuum data at a fixed reference frequency of 243 GHz.
- Simulated arrays use ALMA C40-5/C40-8 configurations despite the historical script name `sim_vla_ms.py`.

## Citation and contact

- Companion benchmark: [RI-Benchmark](https://github.com/MauroRitacco/RI-Benchmark)
- Author: [Mauro Ritacco](https://github.com/MauroRitacco) (rmauro)
