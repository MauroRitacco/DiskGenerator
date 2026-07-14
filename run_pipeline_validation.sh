#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Path to the virtual environment python
PYTHON=".venv/bin/python"
#Number of samples
n=200
#Starting index
START_ID=200
#Image size
img_size=64
#Path to the output directory
OUTDIR="pipeline/val_set"

echo "--------------------------------------------------"
echo "Starting DiskGenerator Pipeline..."
echo "--------------------------------------------------"

mkdir -p ${OUTDIR}

# Step 1: UV Pattern Generation
echo "[1/7] Generating UV patterns..."
$PYTHON utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
    --outdir ${OUTDIR} \
    --n $n \
    --start_id $START_ID

rm -r ${OUTDIR}/alma_sims/png
mv ${OUTDIR}/alma_sims/ms ${OUTDIR}/
mv ${OUTDIR}/alma_sims/uvw ${OUTDIR}/ && rmdir ${OUTDIR}/alma_sims


# Step 2: Disk Generator
echo "[2/7] Generating disks..."
$PYTHON src/disk_generator.py \
    --output_dir ${OUTDIR}/dataset/validationset \
    --num_samples $n \
    --img_size $img_size \
    --generate_validation True \
    --start_id $START_ID


# Step 3: Visibility Simulation
echo "[3/7] Simulating visibilities..."
$PYTHON src/simulator.py \
    --uv_dir ${OUTDIR}/uvw \
    --gt_dir ${OUTDIR}/dataset/validationset \
    --out_dir ${OUTDIR}/dataset/val_measurements \
    --img_size $img_size \
    --generate_validation True


# Step 4: Dirty image generation (initial residual) and save initial reconstruction (zeros)
echo "[4/7] Generating dirty image (initial residual) and save initial reconstruction (zeros)..."
$PYTHON src/dirty_image.py \
    --config config.yaml \
    --data_file ${OUTDIR}/dataset/val_measurements/dummy.mat \
    --output_path ${OUTDIR}/dataset/iteration_1 \
    --img_size $img_size \
    --generate_validation True

# Step 5: Link core datasets into the iteration directory for easy access
echo "[5/7] Creating symbolic links to measurements and trainingset..."
ln -sfn ../validationset ${OUTDIR}/dataset/iteration_1/validationset

# Step 6: Merge validation and training dataset directories
echo "[6/7] Merging validation and training dataset directories..."
rsync -a ${OUTDIR}/dataset ${OUTDIR}/..

# Step 7: Rename baseline distribution files and remove val_set/dataset
echo "[7/7] Renaming baseline distribution files and removing cleaning directories"
mkdir -p ${OUTDIR}/../val_uvw
for f in ${OUTDIR}/uvw/uv_*.mat; do mv "$f" "${OUTDIR}/../val_uvw/val_$(basename "$f")"; done
rm -r ${OUTDIR}

echo "--------------------------------------------------"
echo "Pipeline completed successfully!"
echo "--------------------------------------------------"
