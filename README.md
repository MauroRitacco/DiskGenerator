# DiskGenerator

Synthetic data generator for radio-interferometric image reconstruction of
protoplanetary disks. It simulates the full observation pipeline of a ALMA-like
interferometer on parametric disk models: UV coverages as CASA Measurement Sets,
complex visibilities computed with a NUFFT measurement operator plus Gaussian
noise, and the dirty images a real instrument would return. Output is paired
ground truth / measurements, ready to train reconstruction networks (R2D2) and
to feed the companion benchmark ([RI-Benchmark](https://github.com/MauroRitacco/RI-Benchmark)).

## Datasets

- **Training / validation**: `run_pipeline.sh` + `run_pipeline_validation.sh`.
  Ground-truth FITS (1-4 Gaussian rings, optional star and compact source,
  parameters in the FITS header), `.mat` visibilities, dirty residuals — merged
  into one tree directly consumable by R2D2 (`trainingset/`, `measurements/`,
  `iteration_1/`).
- **Sweep / test grids**: `studies/sweep_experiment/` generates a fixed grid
  over field of view (0.5/2/4 arcsec), rings (1/2/4), star on/off and extra
  source on/off — 36 cells x 10 realizations = 360 disks per dataset, one folder
  per disk, with a JSON metadata sidecar.

## Repository structure

- `src/` — pipeline stages: `disk_generator.py` (disk models), `simulator.py`
  (visibilities + noise), `dirty_image.py` (initial residuals), `ms_to_mat.py`
  (MS/.mat consistency), `transforms.py` (FITS/WCS utilities).
- `utils/` — copy of the BASP R2D2 measurement-operator package; one local
  modification (`--pixelsize` in `sim_vla_ms.py`) makes the field of view
  explicit.
- `studies/sweep_experiment/` — sweep/test dataset generators and validators.
- `run_pipeline*.sh` — end-to-end pipelines (training, validation, benchmark layout).
- `config.yaml` — measurement-operator settings (NUFFT backend, Briggs weighting).

## Usage

Requirements: Python 3.10, pip. CASA tasks (`casatools`/`casatasks`) and `simms`
handle Measurement Sets.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

./run_pipeline.sh                 # training set (set n / START_ID / img_size at the top)
./run_pipeline_validation.sh      # validation set, merged into the same tree
./run_pipeline_benchmark.sh       # per-disk folders in the layout RI-Benchmark expects

./studies/sweep_experiment/run_pipeline_sweep.sh sweep_val   # 360-disk sweep dataset
./studies/sweep_experiment/run_pipeline_sweep.sh test        # held-out test set
```

## Data

Generated data goes to `pipeline/` or `dataset/`.

## License

GPL-3.0. See [LICENSE](LICENSE).
