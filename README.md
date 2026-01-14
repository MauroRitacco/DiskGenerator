# Synthetic Disks & Interferometry Simulation

This repository contains the simulation pipeline I developed for my thesis on Radio Interferometric Imaging. The goal of this project is to generate synthetic ground-truth images of protoplanetary disks and simulate their observation using realistic Very Large Array (VLA) baseline distributions.

The output data is intended to be used for training Deep Learning models, specifically the R2D2 network.

## Context

The pipeline handles the full data generation process: from creating a synthetic "sky" (the disk) to simulating how an interferometer like the VLA would see it (visibilities in the Fourier domain).

It relies on a combination of custom Python scripts for the physics of the disks and external tools (CASA, BASPLib) for the interferometry simulation.

## Repository Structure

* **src/disk_generator.py**: The core logic that generates the FITS images. It simulates stars, gaussian rings, and planetary gaps.
* **src/sim_vla_ms.py**: A utility script that generates empty Measurement Sets (MS) reflecting specific VLA antenna configurations (A and C).
* **src/disk_and_baseline_dist_generator.py**: The orchestrator for the first stage. It calls the two scripts above to create the ground truth and the corresponding UV coverage.
* **src/simulator.py**: The second stage. It takes the images and the empty MS files and computes the complex visibilities using a simulation operator.
* **src/dirty_image.py**: The final stage. It reads the visibilities and generates the "dirty images" (inverse Fourier transform with zeroes filled for missing data) and prepares the folder structure for training the R2D2 network.

## Installation

### Cloning

I recommend setting up a fresh virtual environment with Python 3.10.

Since all dependencies are included directly in the repository, you can simply clone it:

```bash
git clone https://github.com/MauroRitacco/DiskGenerator.git
```

**Install the dependencies:**
```bash
cd DiskGenerator
pip install -r requirements.txt
```

## Workflow

The simulation is designed to run in two sequential steps.

**Step 1: Generate Ground Truth & UV Coverage**
Run the orchestrator script. This will generate the clean FITS images of the disks and create the empty Measurement Sets with the randomized VLA baselines. You may change the amount of files generated in the config section of the script.

```bash
python src/disk_and_baseline_dist_generator.py
```

**Step 2: Simulate Observations**
Once the files from Step 1 are ready, run the simulator. This script applies the measurement operator (located in `utils/`) to transform the image data into visibility data, populating the MS files.

```bash
python src/simulator.py
```

**Step 3: Generate Dirty Images & Training Data**
Finally, run the dirty image generator. This script reconstructs the "dirty images" from the simulated visibilities. These images serve as the input for the R2D2 network training, while the ground truth from Step 1 serves as the target.

```bash
python src/dirty_image.py --config config.yaml
```

## Acknowledgements

The code located in `utils/ri_measurement_operator` and parts of the visibility simulation logic are based on the R2D2/BASPLib repository. The disk generation logic is a custom implementation for this thesis.