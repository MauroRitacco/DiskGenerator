"""
Synthetic Protoplanetary Disk Generator
---------------------------------------
Characteristics:
1. Outputs FITS files (default 512x512) normalized [0,1] with header metadata.
2. Generates 1-5 gaussian rings with power-law decay and random spatial offsets.
3. Simulates a faint point source (planets/background).
4. Normalizes the disk structure first.
5. Adds a central star (50% probability) after normalization to avoid disks overlapping being brighter.
"""

import numpy as np
from astropy.io import fits
import os
import random
import argparse
from scipy.io import loadmat


def generate_disk(output_dir, num_samples=50, img_size=512, start_id=0, generate_validation=False, seed=None):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # ============================================================
    # CONFIGURATION — All tunable parameters in one place
    # ============================================================

    # --- Disk geometry ---
    AXIS_RATIO_RANGE = (0.15, 0.95)        # min/max axis ratio (cos of inclination)
    PA_RANGE = (0, 180)                    # position angle range (degrees)
    Q_INDEX_RANGE = (0.5, 1.5)            # power-law decay index range
    R_REF_FRAC = 0.06                      # reference radius as fraction of img_size

    # --- Ring structure ---
    NUM_RINGS_RANGE = (1, 5)               # min/max number of rings
    RING_SIGMA_MIN_FRAC = 12.0 / 512.0      # ring width (sigma) lower bound as fraction of img_size
    RING_SIGMA_MAX_FRAC = 21.0 / 512.0     # ring width (sigma) upper bound as fraction of img_size
    RING_R_MIN_FRAC = 0.05                 # innermost ring radius as fraction of img_size
    RING_R_MAX_MARGIN_FRAC = 0.06          # margin from edge for outermost ring (fraction of img_size)
    RING_MIN_SEP_FRAC = 0.09              # minimum separation between rings (fraction of img_size)
    RING_MIN_SEP_ABS = 4.0                 # absolute minimum separation (pixels)
    RING_AMP_RANGE = (0.8, 1.2)            # ring amplitude jitter range
    RING_PLACEMENT_ATTEMPTS = 50           # max attempts to place a ring without overlap

    # --- Ring wobble (per-ring center offset) ---
    WOBBLE_MAX_FRAC = 4.0 / 512.0          # max wobble offset as fraction of img_size

    # --- Per-ring inclination/PA variation ---
    DELTA_INC_MANY = (-2, 2)               # inclination wobble (degrees) when >3 rings
    DELTA_PA_MANY = (-1, 1)                # PA wobble (degrees) when >3 rings
    DELTA_INC_FEW = (-5, 5)                # inclination wobble (degrees) when <=3 rings
    DELTA_PA_FEW = (-2.5, 2.5)             # PA wobble (degrees) when <=3 rings

    # --- Extra faint source (planet/background) ---
    EXTRA_SOURCE_PROB = 0.5                # probability of adding an extra source
    EXTRA_MARGIN_FRAC = 0.1                # edge margin as fraction of img_size
    EXTRA_AMP_RANGE = (0.2, 0.5)           # brightness range
    EXTRA_SIGMA_MIN_FRAC = 8.0 / 512.0     # sigma lower bound as fraction of img_size
    EXTRA_SIGMA_MAX_FRAC = 16.0 / 512.0    # sigma upper bound as fraction of img_size

    # --- Central star ---
    STAR_PROB = 0.5                        # probability of adding a central star
    STAR_DISK_SCALE_RANGE = (0.4, 0.8)     # disk intensity scale-down when star is present
    STAR_PEAK = 1.0                        # star peak intensity
    STAR_SIGMA_MIN_FRAC = 8.0 / 512.0      # star sigma lower bound as fraction of img_size
    STAR_SIGMA_MAX_FRAC = 16.0 / 512.0     # star sigma upper bound as fraction of img_size

    # ============================================================

    print(f"Generating {num_samples} disks in '{output_dir}'...")
    # Grid setup
    x = np.linspace(-img_size / 2, img_size / 2, img_size)
    y = np.linspace(-img_size / 2, img_size / 2, img_size)
    xx, yy = np.meshgrid(x, y)

    for i in range(num_samples):
        # Empty image
        disk_image = np.zeros((img_size, img_size), dtype=np.float32)

        # Randomize disk params (Base values)
        base_axis_ratio = random.uniform(*AXIS_RATIO_RANGE)
        base_inc_rad = np.arccos(base_axis_ratio)

        base_angle_deg = random.uniform(*PA_RANGE)

        q_index = random.uniform(*Q_INDEX_RANGE)
        r_ref = max(1.0, img_size * R_REF_FRAC)

        # 1. Rings (with wobble)
        num_rings = random.randint(*NUM_RINGS_RANGE)

        if num_rings > 0:
            # Generate radii with minimum separation to avoid overlap
            radii = []
            min_sep = max(RING_MIN_SEP_ABS, img_size * RING_MIN_SEP_FRAC)

            # Available range
            r_min = max(3, int(img_size * RING_R_MIN_FRAC))
            r_max = max(r_min + 5, img_size // 2 - int(img_size * RING_R_MAX_MARGIN_FRAC))

            for _ in range(num_rings):
                for attempt in range(RING_PLACEMENT_ATTEMPTS):
                    r_new = random.uniform(r_min, r_max)
                    if all(abs(r_new - r) > min_sep for r in radii):
                        radii.append(r_new)
                        break

            radii.sort()

            for mu in radii:
                sigma = random.uniform(
                    max(1.0, img_size * RING_SIGMA_MIN_FRAC),
                    max(2.0, img_size * RING_SIGMA_MAX_FRAC),
                )

                # Random center offset (wobble proportional to size)
                off_r = random.uniform(0, max(0.5, img_size * WOBBLE_MAX_FRAC))
                off_phi = random.uniform(0, 2 * np.pi)
                dx = off_r * np.cos(off_phi)
                dy = off_r * np.sin(off_phi)

                # Per-ring inclination variation
                if num_rings > 3:
                    delta_inc_deg = random.uniform(*DELTA_INC_MANY)
                    delta_pa_deg = random.uniform(*DELTA_PA_MANY)
                else:
                    delta_inc_deg = random.uniform(*DELTA_INC_FEW)
                    delta_pa_deg = random.uniform(*DELTA_PA_FEW)

                # Apply to base
                ring_inc_rad = base_inc_rad + np.radians(delta_inc_deg)
                ring_pa_deg = base_angle_deg + delta_pa_deg

                # Convert back to axis_ratio/theta
                ring_axis_ratio = max(0.05, np.abs(np.cos(ring_inc_rad)))
                ring_theta = np.radians(ring_pa_deg)

                # Shift grid per ring
                xx_s = xx - dx
                yy_s = yy - dy

                # Rotate using ring-specific theta (Standard CCW rotation)
                xx_rot = xx_s * np.cos(ring_theta) - yy_s * np.sin(ring_theta)
                yy_rot = xx_s * np.sin(ring_theta) + yy_s * np.cos(ring_theta)

                # Elliptical distance
                r_ell = np.hypot(xx_rot, yy_rot / ring_axis_ratio)

                # Intensity decay law
                decay = (mu / r_ref) ** (-q_index)
                amp = decay * random.uniform(*RING_AMP_RANGE)

                # Add ring to disk buffer
                disk_image += amp * np.exp(-((r_ell - mu) ** 2) / (2 * sigma ** 2))

        # 2. Extra faint source
        has_extra = random.random() < EXTRA_SOURCE_PROB

        if has_extra:
            margin = max(2, int(img_size * EXTRA_MARGIN_FRAC))
            ex_x = random.uniform(-(img_size / 2) + margin, (img_size / 2) - margin)
            ex_y = random.uniform(-(img_size / 2) + margin, (img_size / 2) - margin)

            r_ex = np.hypot(xx - ex_x, yy - ex_y)

            amp_ex = random.uniform(*EXTRA_AMP_RANGE)
            sig_ex = random.uniform(
                max(1.0, img_size * EXTRA_SIGMA_MIN_FRAC),
                max(2.0, img_size * EXTRA_SIGMA_MAX_FRAC),
            )

            disk_image += amp_ex * np.exp(-(r_ex ** 2) / (2 * sig_ex ** 2))

        # 3. Intermediate Normalization (The Disk)
        vmax_d, vmin_d = np.max(disk_image), np.min(disk_image)
        if vmax_d > vmin_d + 1e-9:
            disk_image = (disk_image - vmin_d) / (vmax_d - vmin_d)

        # 4. Central Star
        has_center = random.random() < STAR_PROB

        final_image = disk_image

        if has_center:
            disk_scale_factor = random.uniform(*STAR_DISK_SCALE_RANGE)
            final_image = final_image * disk_scale_factor

            sigma_star = random.uniform(
                max(1.0, img_size * STAR_SIGMA_MIN_FRAC),
                max(2.0, img_size * STAR_SIGMA_MAX_FRAC),
            )

            r_star = np.hypot(xx, yy)
            star_blob = STAR_PEAK * np.exp(-(r_star ** 2) / (2 * sigma_star ** 2))

            final_image += star_blob

        # 5. Final Normalize & Save
        # Re-normalize to ensure [0,1] range
        vmax, vmin = np.max(final_image), np.min(final_image)
        if vmax > vmin + 1e-9:
            final_image = (final_image - vmin) / (vmax - vmin)
            
        if generate_validation:
            fname = f"val_disk_{start_id + i:04d}.fits"
        else:
            fname = f"disk_{start_id + i:04d}.fits"
        fpath = os.path.join(output_dir, fname)
        if generate_validation:
            ms_path=output_dir+"/../../ms/"+f"disk_{start_id + i:04d}.MS"
        elif "groundtruth" in output_dir:
            ms_path=output_dir+"/../ms/"+f"disk_{start_id + i:04d}.MS"
        else:
            ms_path=output_dir+"/../../ms/"+f"disk_{start_id + i:04d}.MS"

        hdr = fits.Header()
        hdr['SIMPLE'] = True
        hdr['BITPIX'] = -32
        hdr['NAXIS'] = 2
        hdr['NAXIS1'] = img_size
        hdr['NAXIS2'] = img_size
        hdr['OBJECT'] = 'Synthetic Disk'
        hdr['HAS_STAR'] = str(has_center)
        hdr['HAS_EXT'] = str(has_extra)
        hdr['BUNIT']='Jy/pixel'

        # Coordinate System (WCS) mapping to degrees
        # Load nominal_pixelsize directly from the corresponding .mat file (sibling uvw/ directory)
        uv_mat_path = os.path.join(os.path.dirname(ms_path), '..', 'uvw', f'uv_{start_id + i:04d}.mat')
        cell_deg = loadmat(uv_mat_path)['nominal_pixelsize'].item() / 3600.0
        hdr['CDELT1'] = -cell_deg
        hdr['CDELT2'] = cell_deg
        hdr['CUNIT1'] = 'deg'
        hdr['CUNIT2'] = 'deg'
        hdr['CTYPE1'] = 'RA---SIN'
        hdr['CTYPE2'] = 'DEC--SIN'
        hdr['CRPIX1'] = (img_size / 2) + 0.5
        hdr['CRPIX2'] = (img_size / 2) + 0.5

        fits.writeto(fpath, final_image, header=hdr, overwrite=True)

        if (i + 1) % 10 == 0:
            print(f"  Saved {i + 1}/{num_samples} ")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Protoplanetary Disk Generator")
    parser.add_argument("--output_dir", type=str, default="../pipeline/ground_truth", help="Directory to save generated fits files")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of disks to generate")
    parser.add_argument("--img_size", type=int, default=64, help="Image size (assumed square)")
    parser.add_argument("--start_id", type=int, default=0, help="Starting ID for the generated files")
    parser.add_argument("--generate_validation", type=bool, default=False, help="Generate validation set")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_disk(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        start_id=args.start_id,
        generate_validation=args.generate_validation,
        seed=args.seed
    )