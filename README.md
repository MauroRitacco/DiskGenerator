# DiskGenerator

Synthetic dataset generator for radio-interferometric (RI) image reconstruction.

DiskGenerator simulates the full observation pipeline of a radio interferometer on simple
protoplanetary disk models: it samples realistic UV coverages, computes the corresponding
complex visibilities with a NUFFT-based measurement operator, and produces the "dirty"
images that a real instrument would return. The result is paired ground-truth / measurement
data designed to train and benchmark neural-network reconstructions (e.g. R2D2) and
classical imaging algorithms (CASA `tclean`, WSClean, etc.).

Generated samples are square FITS images (default 64x64, normalized to [0, 1]) of
disks with 1-4 Gaussian rings, optional central star and optional faint compact source,
each with full parameter metadata stored in the FITS header and an optional JSON sidecar.

## Installation

Requirements: **Python 3.10** and pip.

```bash
git clone https://github.com/MauroRitacco/DiskGenerator.git
cd DiskGenerator

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

Notes:

- `casatools`/`casatasks` 6.6 are required for Measurement Set (MS) handling; `simms`
  generates the simulated MS.
- Three NUFFT backends are supported (`finufft`, `torchkbnufft`, `pynufft`); `torchkbnufft`
  is the default and is installed out of the box.

All pipeline scripts invoke the interpreter at `.venv/bin/python`, so the virtual
environment must live at the repository root with the name `.venv`.

## Usage

### Training set

Edit `n`, `START_ID` and `img_size` at the top of `run_pipeline.sh`, then:

```bash
./run_pipeline.sh
```

Steps:

1. **UV patterns** (`sim_vla_ms.py`): VLA UV coverages as `.mat` (`pipeline/uvw/`) plus
   simulated Measurement Sets (`pipeline/ms/`).
2. **Ground truth** (`disk_generator.py`): `n` disk images in `pipeline/dataset/trainingset/`
   (`disk_0000.fits`, ...). The pixel size is taken from each sample's UV pattern.
3. **Visibilities** (`simulator.py`): complex visibilities with Gaussian noise
   in `pipeline/dataset/measurements/` (`disk_0000.mat`, ...).
4. **Dirty images** (`dirty_image.py`): initial residuals `_res.fits` and zero initial
   reconstructions `_rec.fits` in `pipeline/dataset/iteration_1/`.
5. A symlink `pipeline/dataset/iteration_1/trainingset -> ../trainingset` is created so
   the folder is directly consumable by R2D2-style training loops.

The script is safe for incremental runs: existing UV patterns/MS are kept and merged.

Once the training set is done, generate the validation set (next section) — it merges
into the same `pipeline/dataset/` tree.

### Validation set

```bash
./run_pipeline_validation.sh
```

Same pipeline with `--generate_validation True` (600 samples by default), producing
`val_disk_XXXX.fits` / `val_disk_XXXX.mat` under `pipeline/dataset/validationset/`,
`val_measurements/` and `pipeline/val_uvw/`. Everything is merged into the same
`pipeline/dataset/` tree used for training.

### Benchmark set

```bash
./run_pipeline_benchmark.sh
```

Generates 2 samples (default) restructured one folder per disk:

```
pipeline/benchmark/
└── disk_0000/
    ├── disk_0000.fits       # ground truth
    ├── disk_0000.MS/        # CASA Measurement Set
    ├── disk_0000.mat        # visibilities
    └── disk_0000_res.fits   # dirty image
```

This is the layout expected by the companion benchmarking harness
([RI-Benchmark](https://github.com/MauroRitacco/RI-Benchmark)). The visibilities are
copied into the MS (`ms_to_mat.py`) so both representations match.

### Sweep dataset

`studies/sweep_experiment/` generates a grid of disks over the physical parameters,
used to compare reconstruction methods across the parameter space:

- field of view: 0.5, 2, 4 arcsec
- number of rings: 1, 2, 4
- central star: on/off
- extra source: on/off

36 cells x 10 realizations = 360 samples, written as
`dataset/<name>/disk_s{size}_r{rings}_c{center}_x{extra}_{rep}/`, each folder holding
`.fits`, `.MS`, `.mat`, `_res.fits` and a `.json` sidecar.

```bash
# Full grid (360 images). dataset_name: sweep_val or test (fixed base seeds).
./studies/sweep_experiment/run_pipeline_sweep.sh <dataset_name> [base_seed] [output_root]

# Single cell (10 images), for quick tests.
./studies/sweep_experiment/run_pipeline_sweep_testcell.sh [output_root]

# Aggregate per-image JSON sidecars into dataset/<name>/metadata.json
./studies/sweep_experiment/collect_metadata.py dataset/<name>

# Sanity-check a completed sweep dataset
./studies/sweep_experiment/validate_sweep.py dataset/<name>
```

Both generator scripts are resumable: completed folders are skipped, partial ones are
regenerated.

## Repository layout

| Path | Description |
|---|---|
| `src/disk_generator.py` | Synthetic disk model generator. Samples ring geometry, inclination, PA, star and extra source; writes ground-truth FITS with parameter headers. |
| `src/simulator.py` | Visibility simulator. Applies the NUFFT measurement operator to each ground truth and adds complex Gaussian noise; writes `.mat` measurement files (`u, v, w, y, nW, frequency, nominal_pixelsize`). |
| `src/dirty_image.py` | Dirty image generator. Builds the weighted adjoint operator (Briggs/natural weighting), computes a PSF normalization and back-projects the visibilities into the initial residual image; also writes the zero "initial reconstruction". |
| `src/ms_to_mat.py` | Copies the simulated visibilities into the `DATA` column of the corresponding CASA Measurement Set, so the `.MS` and `.mat` of a sample stay consistent. |
| `src/transforms.py` | FITS utilities: cell size from the longest baseline, Jy/beam to Jy/pixel conversion, Gaussian smoothing to a common beam, WCS header formatting. |
| `config.yaml` | Central configuration for the measurement operator (NUFFT backend, weighting). |
| `run_pipeline.sh` | Generates the training set end to end. |
| `run_pipeline_validation.sh` | Generates the validation set and merges it into the training output tree. |
| `run_pipeline_benchmark.sh` | Generates a small per-disk benchmark set (one folder per disk, as consumed by RI-Benchmark). |
| `utils/ri_measurement_operator/` | Copy of the BASP R2D2 measurement-operator package (see [External code](#external-code)). |
| `studies/sweep_experiment/` | Parameter-sweep dataset generator used for the final method comparison (see [Sweep dataset](#sweep-dataset)). |

Generated data is written to `pipeline/` (training/validation/benchmark) or `dataset/`
(sweep) and is not tracked by git.

## Configuration

`config.yaml` controls the measurement operator used by `dirty_image.py`:

| Key | Default | Description |
|---|---|---|
| `nufft_pkg` | `tkbn` | NUFFT backend: `tkbn`, `finufft` or `pynufft`. |
| `data_weighting` | `True` | Apply visibility weights (nW) on load. |
| `load_weight` | `False` | Load precomputed imaging weights. |
| `weight_type` | `briggs` | Gridding weighting scheme. |
| `weight_robustness` | `2.0` | Briggs robustness parameter. |
| `weight_gridsize` | `2` | Weight-grid support size. |
| `super_resolution` | `1` | Super-resolution factor of the operator. |
| `target_dynamic_range_heuristic` | `1e4` | Dynamic range used in weight computations. |
| `nufft_oversampling_factor` / `nufft_kernel_dim` / `nufft_mode` | `2.0` / `7` / `table` | NUFFT gridding accuracy settings. |

The number of samples, image size and output directory are set as variables at the top
of each `run_pipeline*.sh` script.

## Standalone tools

Each `src/` module can also be run directly:

```bash
# Generate disks only (e.g. force 2 rings, no star, with metadata sidecar)
python src/disk_generator.py --output_dir out --num_samples 5 --img_size 64 \
    --num_rings 2 --has_center false --metadata out/disk_0000.json

# Simulate visibilities only
python src/simulator.py --uv_dir pipeline/uvw --gt_dir pipeline/dataset/trainingset \
    --out_dir pipeline/dataset/measurements --isnr 40

# Dirty images only
python src/dirty_image.py --config config.yaml \
    --data_file pipeline/dataset/measurements/disk_0000.mat \
    --output_path pipeline/dataset/iteration_1 --img_size 64
```

## External code

`utils/ri_measurement_operator/` is a copy of the
[BASP R2D2](https://github.com/basp-group/R2D2-RI) measurement-operator package
(GPL-3.0), kept as a plain copy so the pipeline is self-contained.

One modification is applied on top of the original copy:
`pyutils/sim_vla_ms.py` accepts a `--pixelsize` argument so the pixel size (and hence
the field of view) of generated UV patterns is controlled explicitly instead of being
drawn from the original random range. When the argument is omitted the original
behaviour is preserved.

## License

GPL-3.0. See [LICENSE](LICENSE).
