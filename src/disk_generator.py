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


def generate_disk(output_dir, num_samples=50, img_size=512):
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

        # Randomize disk params
        axis_ratio = random.uniform(0.3, 0.95)
        angle_deg = random.uniform(0, 180)
        theta = np.radians(angle_deg)
        q_index = random.uniform(0.4, 0.8)
        r_ref = 30.0

        # 1. Rings (with wobble)
        num_rings = random.randint(1, 5)

        if num_rings > 0:
            radii = sorted([random.uniform(20, img_size // 2 - 20) for _ in range(num_rings)])

            for mu in radii:
                sigma = random.uniform(3.0, 9.0)

                # Random center offset (wobble < 10px)
                off_r = random.uniform(0, 10.0)
                off_phi = random.uniform(0, 2 * np.pi)
                dx = off_r * np.cos(off_phi)
                dy = off_r * np.sin(off_phi)

                # Shift grid per ring
                xx_s = xx - dx
                yy_s = yy - dy

                # Rotate
                xx_rot = xx_s * np.cos(theta) + yy_s * np.sin(theta)
                yy_rot = -xx_s * np.sin(theta) + yy_s * np.cos(theta)

                # Elliptical distance
                r_ell = np.hypot(xx_rot, yy_rot / axis_ratio)

                # Intensity decay law
                decay = (mu / r_ref) ** (-q_index)
                amp = decay * random.uniform(0.8, 1.2)

                # Add ring to disk buffer
                disk_image += amp * np.exp(-((r_ell - mu) ** 2) / (2 * sigma ** 2))

        # 2. Extra faint source
        has_extra = random.choice([True, False])

        if has_extra:
            # Random position (keep away from edges)
            ex_x = random.uniform(-(img_size / 2) + 30, (img_size / 2) - 30)
            ex_y = random.uniform(-(img_size / 2) + 30, (img_size / 2) - 30)

            # Distance calculation
            r_ex = np.hypot(xx - ex_x, yy - ex_y)

            # Brightness relative to local disk structures
            amp_ex = random.uniform(0.2, 0.5)
            sig_ex = random.uniform(1.5, 3.0)

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
            sigma_star = random.uniform(1.5, 3.0)

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

        fname = f"disk_{i:04d}.fits"
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
    generate_disk(
        output_dir="../pipeline/ground_truth",
        num_samples=50,
        img_size=512
    )