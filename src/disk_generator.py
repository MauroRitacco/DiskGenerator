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


def generate_disk(output_dir, num_samples=50, img_size=512, start_id=0, generate_validation=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    print(f"Generating {num_samples} disks in '{output_dir}'...")
    # Grid setup
    x = np.linspace(-img_size / 2, img_size / 2, img_size)
    y = np.linspace(-img_size / 2, img_size / 2, img_size)
    xx, yy = np.meshgrid(x, y)

    for i in range(num_samples):
        # Empty image
        disk_image = np.zeros((img_size, img_size), dtype=np.float32)

        # Randomize disk params (Base values)
        base_axis_ratio = random.uniform(0.3, 0.95)
        base_inc_rad = np.arccos(base_axis_ratio) # Base inclination in radians

        base_angle_deg = random.uniform(0, 180)
        
        q_index = random.uniform(0.4, 0.8)
        r_ref = max(1.0, img_size * 0.06)

        # 1. Rings (with wobble)
        num_rings = random.randint(1, 5)

        if num_rings > 0:
            # Generate radii with minimum separation to avoid overlap
            radii = []
            min_sep = max(7.0, img_size * 0.07)
            max_attempts = 50
            
            # Available range
            r_min = max(5, int(img_size * 0.05))
            r_max = max(r_min + 5, img_size // 2 - int(img_size * 0.1))
            
            for _ in range(num_rings):
                for attempt in range(max_attempts):
                    r_new = random.uniform(r_min, r_max)
                    # Check distance to existing rings
                    if all(abs(r_new - r) > min_sep for r in radii):
                        radii.append(r_new)
                        break
            
            radii.sort()

            for mu in radii:
                sigma = random.uniform(max(0.8, img_size * (4.0 / 512.0)), max(1.5, img_size * (11.0 / 512.0)))

                # Random center offset (wobble proportional to size)
                off_r = random.uniform(0, max(0.5, img_size * (4.0 / 512.0)))
                off_phi = random.uniform(0, 2 * np.pi)
                dx = off_r * np.cos(off_phi)
                dy = off_r * np.sin(off_phi)

                # --- 2) Inclination variation ---
                if num_rings > 3:
                    # Tighter constraints for many rings to avoid overlap
                    delta_inc_deg = random.uniform(-2, 2)
                    delta_pa_deg = random.uniform(-1, 1)
                else:
                    # Standard constraints (< 15 deg spread -> +/- 7.5 deg, but user set to +/- 5)
                    delta_inc_deg = random.uniform(-5, 5)
                    delta_pa_deg = random.uniform(-2.5, 2.5)

                # Apply to base
                ring_inc_rad = base_inc_rad + np.radians(delta_inc_deg)
                ring_pa_deg = base_angle_deg + delta_pa_deg
                
                # Convert back to axis_ratio/theta
                # Clamp inclination
                ring_axis_ratio = max(0.05, np.abs(np.cos(ring_inc_rad)))
                
                ring_theta = np.radians(ring_pa_deg)

                # Shift grid per ring
                xx_s = xx - dx
                yy_s = yy - dy

                # Rotate using ring-specific theta (Standard CCW rotation)
                # x' = x cos - y sin
                # y' = x sin + y cos
                xx_rot = xx_s * np.cos(ring_theta) - yy_s * np.sin(ring_theta)
                yy_rot = xx_s * np.sin(ring_theta) + yy_s * np.cos(ring_theta)

                # Elliptical distance
                r_ell = np.hypot(xx_rot, yy_rot / ring_axis_ratio)

                # Intensity decay law
                decay = (mu / r_ref) ** (-q_index)
                amp = decay * random.uniform(0.8, 1.2)

                # Add ring to disk buffer
                disk_image += amp * np.exp(-((r_ell - mu) ** 2) / (2 * sigma ** 2))

        # 2. Extra faint source
        has_extra = random.choice([True, False])

        if has_extra:
            # Random position (keep away from edges)
            margin = max(2, int(img_size * 0.1))
            ex_x = random.uniform(-(img_size / 2) + margin, (img_size / 2) - margin)
            ex_y = random.uniform(-(img_size / 2) + margin, (img_size / 2) - margin)

            # Distance calculation
            r_ex = np.hypot(xx - ex_x, yy - ex_y)

            # Brightness relative to local disk structures
            amp_ex = random.uniform(0.2, 0.5)
            sig_ex = random.uniform(max(0.5, img_size * (1.5 / 512.0)), max(1.0, img_size * (3.0 / 512.0)))

            disk_image += amp_ex * np.exp(-(r_ex ** 2) / (2 * sig_ex ** 2))

        # 3. Intermediate Normalization (The Disk)
        # We normalize the disk structure now so when disks overlap they do not result brighter than the star
        vmax_d, vmin_d = np.max(disk_image), np.min(disk_image)
        if vmax_d > vmin_d + 1e-9:
            disk_image = (disk_image - vmin_d) / (vmax_d - vmin_d)

        # 4. Central Star
        has_center = random.choice([True, False])

        # Initialize final image container
        final_image = disk_image

        if has_center:
            # Scale down the disk intensity so the star pops out
            # Disk becomes 40-80% intensity, Star will be 100%
            disk_scale_factor = random.uniform(0.4, 0.8)
            final_image = final_image * disk_scale_factor

            peak = 1.0 # Star is the new maximum
            sigma_star = random.uniform(max(0.5, img_size * (1.5 / 512.0)), max(1.0, img_size * (3.0 / 512.0)))

            # Distance from center
            r_star = np.hypot(xx, yy)
            star_blob = peak * np.exp(-(r_star ** 2) / (2 * sigma_star ** 2))

            # Add star to the scaled disk
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

        hdr = fits.Header()
        hdr['OBJECT'] = 'Synthetic Disk'
        hdr['HAS_STAR'] = str(has_center)
        hdr['HAS_EXT'] = str(has_extra)

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
    args = parser.parse_args()

    generate_disk(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        img_size=args.img_size,
        start_id=args.start_id,
        generate_validation=args.generate_validation
    )