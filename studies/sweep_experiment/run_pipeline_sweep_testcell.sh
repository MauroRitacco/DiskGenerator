#!/bin/bash
# Generate a single test cell (10 realizations) for the sweep dataset.
# Usage: ./run_pipeline_sweep_testcell.sh [outdir]

set -e

# Run from repo root (paths are relative)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}/../.."

PYTHON=".venv/bin/python"
IMG_SIZE=64
OUTDIR=${1:-"pipeline/sweep_testcell"}

# Fixed cell parameters
SIZE_ARCSEC=2.0
NUM_RINGS=4
HAS_CENTER=true
HAS_EXTRA=false
REPS=10
BASE_SEED=1000000000

mkdir -p "${OUTDIR}"

# Last repetition folder may be partial from an interrupted run; delete it.
last_disk=$(${PYTHON} -c "
import re
from pathlib import Path
out = Path('${OUTDIR}')
dirs = [p for p in out.glob('disk_*') if p.is_dir()]
if not dirs:
    print('')
else:
    def rep(p):
        m = re.search(r'_(\d{3})$', p.name)
        return int(m.group(1)) if m else -1
    print(str(max(dirs, key=rep)))
")
if [ -n "${last_disk}" ]; then
  echo "WARNING: removing potentially incomplete last folder: ${last_disk}"
  rm -rf "${last_disk}"
fi

echo "--------------------------------------------------"
echo "Generating single test cell"
echo "FOV=${SIZE_ARCSEC}arcsec rings=${NUM_RINGS} center=${HAS_CENTER} extra=${HAS_EXTRA}"
echo "Output: ${OUTDIR}"
echo "--------------------------------------------------"

nominal_pixelsize=$(${PYTHON} -c "print(${SIZE_ARCSEC} / ${IMG_SIZE})")

size_str=$(${PYTHON} -c "s=${SIZE_ARCSEC}; print(str(int(s)) if s.is_integer() else f'{s:.1f}'.replace('.', 'p'))")
center_code=$([ "${HAS_CENTER}" = "true" ] && echo 1 || echo 0)
extra_code=$([ "${HAS_EXTRA}" = "true" ] && echo 1 || echo 0)

for rep in $(seq 0 $((REPS - 1))); do
  idx=$(printf "%04d" ${rep})
  derived_seed=$((BASE_SEED + rep * 1000000))

  expected_name="disk_s${size_str}_r${NUM_RINGS}_c${center_code}_x${extra_code}_$(printf '%03d' ${rep})"
  target="${OUTDIR}/${expected_name}"
  if [ -d "${target}" ] \
     && [ -f "${target}/${expected_name}.fits" ] \
     && [ -d "${target}/${expected_name}.MS" ] \
     && [ -f "${target}/${expected_name}.mat" ] \
     && [ -f "${target}/${expected_name}_res.fits" ] \
     && [ -f "${target}/${expected_name}.json" ]; then
    echo "[${idx}] ${expected_name} already exists, skipping"
    continue
  fi
  # Remove partial folder if present
  if [ -d "${target}" ]; then
    echo "[${idx}] ${expected_name} incomplete, removing and regenerating"
    rm -rf "${target}"
  fi

  echo "[${idx}] seed=${derived_seed}"

  tmp=$(mktemp -d)

  # 1. UV pattern + MS
  ${PYTHON} utils/ri_measurement_operator/pyutils/sim_vla_ms.py \
      --outdir "${tmp}" \
      --npatterns 1 \
      --start_id ${rep} \
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
      --start_id ${rep} \
      --seed ${derived_seed} \
      --num_rings ${NUM_RINGS} \
      --has_center ${HAS_CENTER} \
      --has_extra ${HAS_EXTRA} \
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
  export _SWEEP_SIZE="${SIZE_ARCSEC}"
  export _SWEEP_OUTDIR="${OUTDIR}"
  ${PYTHON} - <<'PY'
import os
import json
import shutil
from pathlib import Path
from astropy.io import fits

tmp = os.environ.get('_SWEEP_TMP', '')
if not tmp:
    raise RuntimeError('temporary directory not set')
idx = os.environ.get('_SWEEP_IDX', '')
rep = int(os.environ.get('_SWEEP_REP', '0'))
size_arcsec = float(os.environ.get('_SWEEP_SIZE', '2.0'))
outdir = os.environ.get('_SWEEP_OUTDIR', '')

gt_path = Path(tmp) / 'groundtruth' / f'disk_{idx}.fits'
h = fits.getheader(str(gt_path))
n_rings = int(h['N_RINGS'])
center_code = 1 if str(h['HAS_STAR']).lower() == 'true' else 0
extra_code = 1 if str(h['HAS_EXT']).lower() == 'true' else 0

size_str = str(int(size_arcsec)) if size_arcsec.is_integer() else f'{size_arcsec:.1f}'.replace('.', 'p')
newname = f'disk_s{size_str}_r{n_rings}_c{center_code}_x{extra_code}_{rep:03d}'

target = Path(outdir) / newname
if target.exists():
    shutil.rmtree(str(target))
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

# Update JSON sidecar with final name
meta_path = target / f'{newname}.json'
with open(meta_path, 'r') as f:
    meta = json.load(f)
meta['name'] = f'{newname}.fits'
meta['cell']['repetition'] = rep
with open(meta_path, 'w') as f:
    json.dump(meta, f, indent=2)

print(f'Saved {newname}')
PY

  rm -rf "${tmp}"
  # Drop CASA logs
  rm -f casa-*.log
done

# 7. Aggregate metadata
${PYTHON} studies/sweep_experiment/collect_metadata.py "${OUTDIR}"

echo "--------------------------------------------------"
echo "Test cell completed: ${REPS} images"
echo "Metadata: ${OUTDIR}/metadata.json"
echo "--------------------------------------------------"
