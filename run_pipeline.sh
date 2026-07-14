#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Path to the virtual environment python
PYTHON=".venv/bin/python"
#Number of samples — only 1495 remaining UV patterns needed to reach 3000 total (0000-2999)
n=2
#Starting index
START_ID=0000
#Image size
img_size=64
#Path to the output directory
OUTDIR="pipeline"

echo "--------------------------------------------------"
echo "Starting DiskGenerator Pipeline..."
echo "--------------------------------------------------"

mkdir -p ${OUTDIR}

# Step 1: UV Pattern Generation
echo "[1/5] Generating UV patterns..."
$PYTHON utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
    --outdir ${OUTDIR} \
    --n $n \
    --start_id $START_ID
# 
rm -rf ${OUTDIR}/alma_sims/png
# Merge new uvw/ms files into existing directories (safe for incremental runs)
# Note: .MS files are directories in CASA; -f ensures fresh copies always win
mkdir -p ${OUTDIR}/uvw ${OUTDIR}/ms
[ -d "${OUTDIR}/alma_sims/uvw" ] && mv -f ${OUTDIR}/alma_sims/uvw/* ${OUTDIR}/uvw/ && rm -rf ${OUTDIR}/alma_sims/uvw
[ -d "${OUTDIR}/alma_sims/ms" ]  && mv -f ${OUTDIR}/alma_sims/ms/* ${OUTDIR}/ms/  && rm -rf ${OUTDIR}/alma_sims/ms
rmdir ${OUTDIR}/alma_sims 2>/dev/null || true

# Step 2: Disk Generator
echo "[2/5] Generating disks..."
$PYTHON src/disk_generator.py \
    --output_dir ${OUTDIR}/dataset/trainingset \
    --num_samples $n \
    --img_size $img_size \
    --start_id $START_ID

# Step 3: Visibility Simulation
echo "[3/5] Simulating visibilities..."
$PYTHON src/simulator.py \
    --uv_dir ${OUTDIR}/uvw \
    --gt_dir ${OUTDIR}/dataset/trainingset \
    --out_dir ${OUTDIR}/dataset/measurements \
    --img_size $img_size

# Step 4: Dirty image generation (initial residual) and save initial reconstruction (zeros)
echo "[4/5] Generating dirty image (initial residual) and save initial reconstruction (zeros)..."
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
