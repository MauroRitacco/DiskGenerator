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
OUTDIR="pipeline/benchmark"

echo "--------------------------------------------------"
echo "Starting DiskGenerator Pipeline..."
echo "--------------------------------------------------"

mkdir -p ${OUTDIR}

# Step 1: UV Pattern Generation
echo "[1/5] Generating UV patterns..."
$PYTHON utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
    --outdir ${OUTDIR} \
    --n $n

rm -r ${OUTDIR}/vla_sims/png
mv ${OUTDIR}/vla_sims/uvw ${OUTDIR}/ && mv ${OUTDIR}/vla_sims/ms ${OUTDIR}/ && rmdir ${OUTDIR}/vla_sims

# Step 2: Disk Generator
echo "[2/5] Generating disks..."
$PYTHON src/disk_generator.py \
    --output_dir ${OUTDIR}/groundtruth \
    --num_samples $n \
    --img_size $img_size

# Step 3: Visibility Simulation
echo "[3/5] Simulating visibilities..."
$PYTHON src/simulator.py \
    --uv_dir ${OUTDIR}/uvw \
    --gt_dir ${OUTDIR}/groundtruth \
    --out_dir ${OUTDIR}/measurements_mat \
    --img_size $img_size

$PYTHON -c "import src.ms_to_mat; [src.ms_to_mat.copyVisibilitiesToData(f'${OUTDIR}/ms/vla_{i:04d}.MS', f'${OUTDIR}/measurements_mat/disk_{i:04d}.mat') for i in range($n)]"

mv ${OUTDIR}/ms ${OUTDIR}/measurements_ms
for i in $(seq -f "%04g" 0 $((n-1))); do mv ${OUTDIR}/measurements_ms/vla_${i}.MS ${OUTDIR}/measurements_ms/disk_${i}.MS; done

