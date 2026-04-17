#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Path to the virtual environment python
PYTHON=".venv/bin/python"
#Number of samples
n=5
#Image size
img_size=64
#Path to the output directory
OUTDIR="pipeline"

echo "--------------------------------------------------"
echo "Starting DiskGenerator Pipeline..."
echo "--------------------------------------------------"

mkdir -p ${OUTDIR}

# Step 1: UV Pattern Generation
echo "[1/4] Generating UV patterns..."
$PYTHON utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
    --outdir ${OUTDIR} \
    --n $n

rm -r ${OUTDIR}/vla_sims/png
rm -r ${OUTDIR}/vla_sims/ms
mv ${OUTDIR}/vla_sims/uvw ${OUTDIR}/ && rmdir ${OUTDIR}/vla_sims

# Step 2: Disk Generator
echo "[2/4] Generating disks..."
$PYTHON src/disk_generator.py \
    --output_dir ${OUTDIR}/dataset/trainingset \
    --num_samples $n \
    --img_size $img_size

# Step 3: Visibility Simulation
echo "[3/4] Simulating visibilities..."
$PYTHON src/simulator.py \
    --uv_dir ${OUTDIR}/uvw \
    --gt_dir ${OUTDIR}/dataset/trainingset \
    --out_dir ${OUTDIR}/dataset/measurements \
    --img_size $img_size

# Step 4: Dirty image generation (initial residual) and save initial reconstruction (zeros)
echo "[4/4] Generating dirty image (initial residual) and save initial reconstruction (zeros)..."
$PYTHON src/dirty_image.py \
    --config config.yaml \
    --data_file ${OUTDIR}/dataset/measurements/dummy.mat \
    --output_path ${OUTDIR}/dataset/iteration_1 \
    --img_size $img_size

# Step 5: Link core datasets into the iteration directory for easy access
echo "[5/5] Creating symbolic links to measurements and trainingset..."
ln -sfn $(pwd)/${OUTDIR}/dataset/trainingset ${OUTDIR}/dataset/iteration_1/trainingset

echo "--------------------------------------------------"
echo "Pipeline completed successfully!"
echo "--------------------------------------------------"
