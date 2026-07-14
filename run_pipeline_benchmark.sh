#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Path to the virtual environment python
PYTHON=".venv/bin/python"
#Number of samples
n=2
#Image size
img_size=64
#Path to the output directory
OUTDIR="pipeline/benchmark"

echo "--------------------------------------------------"
echo "Starting DiskGenerator Pipeline..."
echo "--------------------------------------------------"

mkdir -p ${OUTDIR}

# Step 1: UV Pattern Generation
echo "[1/4] Generating UV patterns..."
$PYTHON utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
    --outdir ${OUTDIR} \
    --n $n

rm -r ${OUTDIR}/alma_sims/png
mv ${OUTDIR}/alma_sims/uvw ${OUTDIR}/ && mv ${OUTDIR}/alma_sims/ms ${OUTDIR}/ && rmdir ${OUTDIR}/alma_sims

# Step 2: Disk Generator
echo "[2/4] Generating disks..."
$PYTHON src/disk_generator.py \
    --output_dir ${OUTDIR}/groundtruth \
    --num_samples $n \
    --img_size $img_size \
    
# Step 3: Visibility Simulation
echo "[3/4] Simulating visibilities..."
$PYTHON src/simulator.py \
    --uv_dir ${OUTDIR}/uvw \
    --gt_dir ${OUTDIR}/groundtruth \
    --out_dir ${OUTDIR}/measurements_mat \
    --img_size $img_size

$PYTHON -c "import src.ms_to_mat; [src.ms_to_mat.copyVisibilitiesToData(f'${OUTDIR}/ms/disk_{i:04d}.MS', f'${OUTDIR}/measurements_mat/disk_{i:04d}.mat') for i in range($n)]"

mv ${OUTDIR}/ms ${OUTDIR}/measurements_ms

# Step 3.5: Dirty Image Generation
echo "[3.5/4] Generating dirty images..."
$PYTHON src/dirty_image.py \
    --config config.yaml \
    --data_file ${OUTDIR}/measurements_mat/dummy.mat \
    --output_path ${OUTDIR}/dirty \
    --img_size $img_size

# Step 4: Restructure Directories
echo "[4/4] Restructuring output directory..."
for i in $(seq -f "%04g" 0 $((n-1))); do
    TARGET_DIR="${OUTDIR}/disk_${i}"
    mkdir -p "${TARGET_DIR}"
    
    # Move the respective files into the target directory
    mv "${OUTDIR}/groundtruth/disk_${i}.fits" "${TARGET_DIR}/"
    mv "${OUTDIR}/measurements_ms/disk_${i}.MS" "${TARGET_DIR}/"
    mv "${OUTDIR}/measurements_mat/disk_${i}.mat" "${TARGET_DIR}/"
    
    # Move the generated dirty images
    mv "${OUTDIR}/dirty/trainingset_resN1/disk_${i}_res.fits" "${TARGET_DIR}/"
done

# Clean up empty leftover directories
rm -rf "${OUTDIR}/groundtruth" "${OUTDIR}/measurements_ms" "${OUTDIR}/measurements_mat" "${OUTDIR}/uvw" "${OUTDIR}/alma_sims" "${OUTDIR}/dirty"

echo "Pipeline finished successfully!"

