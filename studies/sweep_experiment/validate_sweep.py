#!/usr/bin/env python3
"""Validate a generated sweep dataset.

Checks:
- expected number of images (360)
- every image folder contains the 5 expected files
- filenames follow the pattern disk_s{tamaño}_r{rings}_c{central}_x{extra}_{rep}
- metadata matches FITS header (N_RINGS, HAS_STAR, HAS_EXT, IMSIZE_A)
- each cell has exactly 10 repetitions
"""

import json
import re
import sys
from pathlib import Path

from astropy.io import fits

EXPECTED_CELLS = 36
EXPECTED_REPS = 10


def main():
    if len(sys.argv) < 2:
        print("Usage: validate_sweep.py <dataset_dir>")
        sys.exit(1)

    dataset_dir = Path(sys.argv[1]).resolve()
    metadata_path = dataset_dir / "metadata.json"
    if not metadata_path.exists():
        print(f"ERROR: metadata.json not found in {dataset_dir}")
        sys.exit(1)

    with open(metadata_path) as f:
        metadata = json.load(f)

    errors = []
    images = metadata.get("images", [])
    expected_images = EXPECTED_CELLS * EXPECTED_REPS
    if len(images) != expected_images:
        errors.append(f"image count: expected {expected_images}, got {len(images)}")

    pattern = re.compile(
        r"^disk_s(?P<size>0p5|2|4)_r(?P<rings>[124])_c(?P<center>[01])_x(?P<extra>[01])_(?P<rep>\d{3})\.fits$"
    )

    cell_counts = {}

    for record in images:
        name = record["name"]
        m = pattern.match(name)
        if not m:
            errors.append(f"filename does not match expected pattern: {name}")
            continue

        cell_key = (
            record["cell"]["image_size_arcsec"],
            record["cell"]["num_rings"],
            int(m.group("center")),
            int(m.group("extra")),
        )
        cell_counts[cell_key] = cell_counts.get(cell_key, 0) + 1

        base = name.replace(".fits", "")
        folder = dataset_dir / base

        expected_files = {
            f"{base}.fits",
            f"{base}.MS",
            f"{base}.mat",
            f"{base}_res.fits",
            f"{base}.json",
        }
        missing = expected_files - {p.name for p in folder.glob("*") if p.is_file() or p.is_dir()}
        if missing:
            errors.append(f"{base}: missing files {sorted(missing)}")
            continue

        # Check FITS header consistency
        fits_path = folder / f"{base}.fits"
        hdr = fits.getheader(str(fits_path))
        meta_cell = record["cell"]
        meta_params = record["sampled_params"]

        if int(hdr["N_RINGS"]) != meta_cell["num_rings"]:
            errors.append(
                f"{base}: header N_RINGS={hdr['N_RINGS']} != metadata num_rings={meta_cell['num_rings']}"
            )

        hdr_center = 1 if str(hdr.get("HAS_STAR", "")).lower() == "true" else 0
        if hdr_center != int(m.group("center")):
            errors.append(
                f"{base}: header HAS_STAR={hdr['HAS_STAR']} != filename center={m.group('center')}"
            )

        hdr_extra = 1 if str(hdr.get("HAS_EXT", "")).lower() == "true" else 0
        if hdr_extra != int(m.group("extra")):
            errors.append(
                f"{base}: header HAS_EXT={hdr['HAS_EXT']} != filename extra={m.group('extra')}"
            )

        if not abs(float(hdr["IMSIZE_A"]) - meta_cell["image_size_arcsec"]) < 1e-9:
            errors.append(
                f"{base}: header IMSIZE_A={hdr['IMSIZE_A']} != metadata image_size_arcsec={meta_cell['image_size_arcsec']}"
            )

        if meta_params.get("seed") is None:
            errors.append(f"{base}: missing seed in metadata")

    for cell_key, count in cell_counts.items():
        if count != EXPECTED_REPS:
            errors.append(f"cell {cell_key}: expected {EXPECTED_REPS} reps, got {count}")

    if len(cell_counts) != EXPECTED_CELLS:
        errors.append(f"cell count: expected {EXPECTED_CELLS}, got {len(cell_counts)}")

    if errors:
        print(f"VALIDATION FAILED ({len(errors)} errors):")
        for e in errors:
            print(f"  - {e}")
        sys.exit(1)

    print(f"VALIDATION PASSED: {len(images)} images, {len(cell_counts)} cells, {EXPECTED_REPS} reps each.")


if __name__ == "__main__":
    main()
