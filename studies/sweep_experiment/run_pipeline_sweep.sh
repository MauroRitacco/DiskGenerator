#!/bin/bash
# Generate a sweep dataset for the final method-comparison experiment.
# Usage: ./run_pipeline_sweep.sh <dataset_name> <base_seed> [output_root]
#   dataset_name: e.g. sweep_val or test
#   base_seed:    dataset-specific integer base seed (must differ across datasets)
#   output_root:  defaults to dataset/<dataset_name>

set -e

# Run from repo root (paths are relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

PYTHON=".venv/bin/python"
IMG_SIZE=64

DATASET=${1:?"Usage: $0 <dataset_name> [base_seed] [output_root]"}
# Fixed base seed per official dataset (CLI arg overrides)
case "${DATASET}" in
  sweep_val) DEFAULT_BASE_SEED=2000000000 ;;
  test)      DEFAULT_BASE_SEED=3000000000 ;;
  trainingset|validationset)
    echo "ERROR: '${DATASET}' is a reserved name. Use sweep_val or test."
    exit 1
    ;;
  *)         DEFAULT_BASE_SEED=0 ;;
esac
BASE_SEED=${2:-${DEFAULT_BASE_SEED}}
if [ "${BASE_SEED}" -eq 0 ]; then
  echo "ERROR: base_seed required for dataset '${DATASET}'"
  exit 1
fi
OUTDIR=${3:-"dataset/${DATASET}"}

# Grid definition
IMAGE_SIZES=(0.5 2.0 4.0)
NUM_RINGS_LIST=(1 2 4)
HAS_CENTER_LIST=(true false)
HAS_EXTRA_LIST=(true false)
REPS=10

mkdir -p "${OUTDIR}"

# Last folder in generation order may be partial from an interrupted run; delete it.
# Complete folders are skipped later.
last_disk=$(${PYTHON} -c "
import re
from pathlib import Path
out = Path('${OUTDIR}')
dirs = [p for p in out.glob('disk_*') if p.is_dir()]
if not dirs:
    print('')
else:
    def order_key(p):
        m = re.match(r'disk_s(0p5|2|4)_r([124])_c([01])_x([01])_(\d{3})$', p.name)
        if not m:
            return (-1, -1, -1, -1, -1)
        size_idx = {'0p5': 0, '2': 1, '4': 2}[m.group(1)]
        rings_idx = {1: 0, 2: 1, 4: 2}[int(m.group(2))]
        # true (1) is generated before false (0)
        center_inv = 1 - int(m.group(3))
        extra_inv = 1 - int(m.group(4))
        rep = int(m.group(5))
        return (size_idx, rings_idx, center_inv, extra_inv, rep)
    print(str(max(dirs, key=order_key)))
")
if [ -n "${last_disk}" ]; then
  echo "WARNING: removing potentially incomplete last folder: ${last_disk}"
  rm -rf "${last_disk}"
fi

echo "--------------------------------------------------"
echo "Generating sweep dataset: ${DATASET}"
echo "Base seed: ${BASE_SEED}"
echo "Output: ${OUTDIR}"
echo "--------------------------------------------------"

global_idx=0
cell_idx=0

for size_arcsec in "${IMAGE_SIZES[@]}"; do
  for nrings in "${NUM_RINGS_LIST[@]}"; do
    for center in "${HAS_CENTER_LIST[@]}"; do
      for extra in "${HAS_EXTRA_LIST[@]}"; do
        nominal_pixelsize=$(${PYTHON} -c "print(${size_arcsec} / ${IMG_SIZE})")
        echo "--- Cell ${cell_idx}: FOV=${size_arcsec}arcsec rings=${nrings} center=${center} extra=${extra} pixel=${nominal_pixelsize}arcsec ---"

        for rep in $(seq 0 $((REPS - 1))); do
          idx=$(printf "%04d" ${global_idx})
          derived_seed=$((BASE_SEED + global_idx * 1000000))

          # Skip if the complete folder already exists
          size_str=$(${PYTHON} -c "s=${size_arcsec}; print(str(int(s)) if s.is_integer() else f'{s:.1f}'.replace('.', 'p'))")
          center_code=$([ "${center}" = "true" ] && echo 1 || echo 0)
          extra_code=$([ "${extra}" = "true" ] && echo 1 || echo 0)
          expected_name="disk_s${size_str}_r${nrings}_c${center_code}_x${extra_code}_$(printf '%03d' ${rep})"
          target="${OUTDIR}/${expected_name}"
          if [ -d "${target}" ] \
             && [ -f "${target}/${expected_name}.fits" ] \
             && [ -d "${target}/${expected_name}.MS" ] \
             && [ -f "${target}/${expected_name}.mat" ] \
             && [ -f "${target}/${expected_name}_res.fits" ] \
             && [ -f "${target}/${expected_name}.json" ]; then
            echo "[${idx}] ${expected_name} already exists, skipping"
            global_idx=$((global_idx + 1))
            continue
          fi
          # Remove partial folder if present
          if [ -d "${target}" ]; then
            echo "[${idx}] ${expected_name} incomplete, removing and regenerating"
            rm -rf "${target}"
          fi

          echo "[${idx}] rep=${rep} seed=${derived_seed}"

          tmp=$(mktemp -d)

          # 1. UV pattern + MS
          ${PYTHON} utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
              --outdir "${tmp}" \
              --npatterns 1 \
              --start_id ${global_idx} \
              --pixelsize ${nominal_pixelsize}

          rm -rf "${tmp}/alma_sims/png"
          mkdir -p "${tmp}/uvw" "${tmp}/ms"
          mv "${tmp}/alma_sims/uvw/"* "${tmp}/uvw/"
          mv "${tmp}/alma_sims/ms/"* "${tmp}/ms/"
          rmdir "${tmp}/alma_sims/uvw" "${tmp}/alma_sims/ms" "${tmp}/alma_sims" 2>/dev/null || true

          # 2. Ground-truth disk
          mkdir -p "${tmp}/groundtruth" "${tmp}/metadata"
          ${PYTHON} src/disk_generator.py \
              --output_dir "${tmp}/groundtruth" \
              --num_samples 1 \
              --img_size ${IMG_SIZE} \
              --start_id ${global_idx} \
              --seed ${derived_seed} \
              --num_rings ${nrings} \
              --has_center ${center} \
              --has_extra ${extra} \
              --metadata "${tmp}/metadata/disk_${idx}.json"

          # 3. Visibility simulation
          mkdir -p "${tmp}/measurements_mat"
          ${PYTHON} src/simulator.py \
              --uv_dir "${tmp}/uvw" \
              --gt_dir "${tmp}/groundtruth" \
              --out_dir "${tmp}/measurements_mat" \
              --img_size ${IMG_SIZE}

          # 4. Copy visibilities from MS into the measurement .mat
          ${PYTHON} -c "
import src.ms_to_mat
src.ms_to_mat.copyVisibilitiesToData('${tmp}/ms/disk_${idx}.MS', '${tmp}/measurements_mat/disk_${idx}.mat')
"

          # 5. Dirty image
          mkdir -p "${tmp}/dirty"
          ${PYTHON} src/dirty_image.py \
              --config config.yaml \
              --data_file "${tmp}/measurements_mat/dummy.mat" \
              --output_path "${tmp}/dirty" \
              --img_size ${IMG_SIZE}

          # 6. Rename folder to disk_s{size}_r{rings}_c{center}_x{extra}_{rep}
          export _SWEEP_TMP="${tmp}"
          export _SWEEP_IDX="${idx}"
          export _SWEEP_REP="${rep}"
          export _SWEEP_SIZE="${size_arcsec}"
          export _SWEEP_OUTDIR="${OUTDIR}"
          export _SWEEP_CELL="${cell_idx}"
          ${PYTHON} - <<'PY'
import os
import json
import shutil
from pathlib import Path
from astropy.io import fits

tmp = os.environ.get('_SWEEP_TMP', '')
idx = os.environ.get('_SWEEP_IDX', '')
rep = int(os.environ.get('_SWEEP_REP', '0'))
size_arcsec = float(os.environ.get('_SWEEP_SIZE', '2.0'))
outdir = os.environ.get('_SWEEP_OUTDIR', '')
cell_idx = int(os.environ.get('_SWEEP_CELL', '0'))

gt_path = Path(tmp) / 'groundtruth' / f'disk_{idx}.fits'
h = fits.getheader(str(gt_path))
n_rings = int(h['N_RINGS'])
center_code = 1 if str(h['HAS_STAR']).lower() == 'true' else 0
extra_code = 1 if str(h['HAS_EXT']).lower() == 'true' else 0

size_str = str(int(size_arcsec)) if size_arcsec.is_integer() else f'{size_arcsec:.1f}'.replace('.', 'p')
newname = f'disk_s{size_str}_r{n_rings}_c{center_code}_x{extra_code}_{rep:03d}'

target = Path(outdir) / newname
target.mkdir(parents=True, exist_ok=True)

moves = [
    (Path(tmp) / 'groundtruth' / f'disk_{idx}.fits', target / f'{newname}.fits'),
    (Path(tmp) / 'ms' / f'disk_{idx}.MS', target / f'{newname}.MS'),
    (Path(tmp) / 'measurements_mat' / f'disk_{idx}.mat', target / f'{newname}.mat'),
    (Path(tmp) / 'dirty' / 'trainingset_resN1' / f'disk_{idx}_res.fits', target / f'{newname}_res.fits'),
    (Path(tmp) / 'metadata' / f'disk_{idx}.json', target / f'{newname}.json'),
]
for src, dst in moves:
    if dst.exists():
        if dst.is_dir():
            shutil.rmtree(str(dst))
        else:
            dst.unlink()
    os.rename(str(src), str(dst))

meta_path = target / f'{newname}.json'
with open(meta_path, 'r') as f:
    meta = json.load(f)
meta['name'] = f'{newname}.fits'
meta['cell']['cell_index'] = cell_idx
meta['cell']['repetition'] = rep
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f'Saved {newname}')
PY

          rm -rf "${tmp}"
          # Drop CASA logs
          rm -f casa-*.log

          global_idx=$((global_idx + 1))
        done

        cell_idx=$((cell_idx + 1))
      done
    done
  done
done

# 7. Aggregate metadata
${PYTHON} studies/sweep_experiment/collect_metadata.py "${OUTDIR}"

echo "--------------------------------------------------"
echo "Dataset ${DATASET} completed: ${global_idx} images"
echo "Metadata: ${OUTDIR}/metadata.json"
echo "--------------------------------------------------"
