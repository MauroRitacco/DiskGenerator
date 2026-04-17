# DiskGenerator

DiskGenerator is an automated pipeline for generating synthetic radio interferometry datasets consisting of simple disk models. It simulates realistic VLA (Very Large Array) UV coverage and mathematically models the measurement visibilities to produce ground-truth sources and their corresponding "dirty" (raw, point spread function-corrupted) images. These datasets are specifically designed to train neural networks (like R2D2) for image reconstruction and iterative denoising.

## Installation

The pipeline scripts default to using a local virtual environment located at `.venv/`.

1. **Create and activate the virtual environment:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```

2. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure your system can map `finufft` / `torchkbnufft` and correct PyTorch CUDA builds depending on your hardware.*

## Usage

Dataset generation is handled by two main bash scripts. You can adjust the number of samples (`n`) and the image dimensions (`img_size`) directly at the top of these bash files.

### 1. Generating the Training Set
Run the main pipeline to generate a training dataset:
```bash
./run_pipeline.sh
```
This executes a 5-step process:
1. **UV Pattern Generation (`sim_vla_ms.py`)**: Computes visibility coverage in the spatial-frequency (`uvw`) domain.
2. **Disk Generator (`disk_generator.py`)**: Produces the model ground-truth images (saved to `dataset/trainingset`).
3. **Visibility Simulation (`simulator.py`)**: Evaluates the measurement operator on the ground truth to compute complex visibilities (`dataset/measurements`).
4. **Dirty Image Generation (`dirty_image.py`)**: Inverts the ungridded visibilities via NUFFT back into structural dirty images and dummy zero configurations inside `dataset/iteration_1`.
5. **Data Linking**: Symlinks the raw dataset mappings so the resulting iteration folder is uniformly ready for training loops.

Outputs are written directly to the `pipeline/` directory.

### 2. Generating the Validation Set
Run the validation script to generate an independent distribution dataset:
```bash
./run_pipeline_validation.sh
```
This functions similarly to the train script but flags the underlying Python programs with `--generate_validation True`. It isolates the measurement processes and auto-merges the validated baseline and structures alongside your generated training sets to streamline subsequent evaluations.

## Configuration

Physical interferometry parameters and grid processing specifications are centrally defined in `config.yaml`.
Key settings include:
- `nufft_pkg`: Sets the Non-Uniform Fast Fourier Transform backend (e.g., `finufft`).
- `weight_type`: Defines grid weighting algorithms (e.g., `briggs`, `natural`) for PSF correction limits.
- `weight_robustness` & `nufft_oversampling_factor`: Regulates spatial mappings.
- `meas_op_on_gpu`: Accelerates measurement calculations via CUDA.

## Acknowledgments

Everything in `utils/` is taken from BASP's R2D2 GitHub repository.