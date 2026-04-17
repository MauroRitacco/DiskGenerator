import os
import sys
import argparse
import gc
import resource
import torch
import numpy as np
from pathlib import Path
from math import *
from scipy.io import loadmat, savemat
from astropy.io import fits

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
RI_OP_DIR = BASE_DIR / "utils" / "ri_measurement_operator"
sys.path.insert(0, str(RI_OP_DIR))



from pysrc.utils.io import load_data_to_tensor
from pysrc.utils.gen_imaging_weights import gen_imaging_weights

def log_memory():
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux returns kb, macOS returns bytes. Assuming Linux based on environment.
    print(f"Memory usage: {usage / 1024:.2f} MB")


def simulator(uv_path,gdth_path,measurement_path,super_resolution=1,img_size=(128,128),nufft_pkg='tkbn', iSNR=40):
    # Define u, v, w, y uv parameters from uv path
    uv = loadmat(uv_path, variable_names=["u", "v", "w", "frequency","nominal_pixelsize"])

    u = uv["u"]
    v = uv["v"]
    w = uv["w"]
    frequency = uv["frequency"]
    image_pixel_size= uv["nominal_pixelsize"].item()

    speed_of_light = 299792458
    wavelength = speed_of_light / frequency  # Observation wavelength in meters

    # Convert the uvw coordinates from meters to units of wavelength
    u /= wavelength
    v /= wavelength
    w /= wavelength

    # Transform data into tensor
    data= load_data_to_tensor(uv_file_path=uv_path,
                              super_resolution=super_resolution,
                              img_size=img_size,
                              image_pixel_size=image_pixel_size)

    data["u"] /= wavelength
    data["v"] /= wavelength
    data["w"] /= wavelength

    # Choose nufft package from nufft_pkg variable
    match nufft_pkg:
        case "finufft":
            from pysrc.measOperator.meas_op_nufft_pytorch_finufft import MeasOpPytorchFinufft

            Operator = MeasOpPytorchFinufft
        case "tkbn":
            from pysrc.measOperator.meas_op_nufft_tkbn import MeasOpTkbNUFFT

            Operator = MeasOpTkbNUFFT
        case "pynufft":
            from pysrc.measOperator.meas_op_nufft_pynufft import MeasOpPynufft

            Operator = MeasOpPynufft

    # Define Operator (has more parameters)
    meas_op = Operator(
        u=data["u"],
        v=data["v"],
        img_size=img_size,
    )

    # Define gdth from ground truth images and convert it into tensor
    gdth=fits.getdata(gdth_path)
    gdth = torch.tensor(gdth.astype(float), dtype=torch.float64).view(1, 1, *gdth.shape)
    
    with torch.no_grad():
        y_clean = meas_op.forward_op(gdth)

        # Input random Gaussian noise
        M = y_clean.numel()
        tau = 10 ** (-iSNR / 20) * torch.linalg.norm(y_clean, dtype=torch.complex128) / sqrt(M)  # Calculate tau
        noise = (torch.randn(M) + 1j * torch.randn(M)) / sqrt(2)  # Random Gaussian noise with std tau and mean 0
        # Define y as y clean plus noise
        y = y_clean + noise

    # Save visibilities
    nW = torch.ones(M) / tau  # The inverse of the noise std
    max_proj_baseline = np.sqrt(np.max(u ** 2 + v ** 2))
    savemat(measurement_path, {
        "y": y.numpy().reshape(-1, 1),
        "nW": nW.numpy().reshape(-1, 1),
        "u": u.reshape(-1, 1),
        "v": v.reshape(-1, 1),
        "w": w.reshape(-1, 1),
        "frequency": frequency.item(),
        "nominal_pixelsize": image_pixel_size,
    })
    
    # Explicit garbage flushing inside the function scope
    del meas_op, data, gdth, y_clean, y, noise, u, v, w, nW
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simulator for DiskGenerator")
    parser.add_argument("--uv_dir", type=str, required=True, help="Directory containing UV patterns")
    parser.add_argument("--gt_dir", type=str, required=True, help="Directory containing ground truth fits files")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save measurement files")
    parser.add_argument("--super_resolution", type=int, default=1, help="Super resolution factor")
    parser.add_argument("--img_size", type=int, default=128, help="Image size (assumed square)")
    parser.add_argument("--nufft_pkg", type=str, default="tkbn", choices=["finufft", "tkbn", "pynufft"], help="NUFFT package to use")
    parser.add_argument("--isnr", type=float, default=40.0, help="Input Signal-to-Noise Ratio")
    parser.add_argument("--generate_validation", type=bool, default=False, help="Generate validation data")
    args = parser.parse_args()

    uv_dir = Path(args.uv_dir)
    gt_dir = Path(args.gt_dir)
    out_dir = Path(args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    # Run the simulator for all the files in the respective folders
    uv_files = sorted(uv_dir.glob("uv_*.mat"))
    for uv_file in uv_files:
        # Extract index from filename, e.g., "uv_1234.mat" -> 1234
        try:
            i = int(uv_file.stem.split('_')[1])
        except (IndexError, ValueError):
            print(f"Skipping file with unexpected name format: {uv_file.name}")
            continue

        if args.generate_validation:
            gdth_file = gt_dir / f"val_disk_{i:04d}.fits"
        else:
            gdth_file = gt_dir / f"disk_{i:04d}.fits"

        if not gdth_file.exists():
            print(f"Warning: Ground truth file not found for index {i}: {gdth_file}")
            continue

        if args.generate_validation:
            simulator(uv_file, gdth_file, out_dir / f"val_disk_{i:04d}.mat",
                      super_resolution=args.super_resolution,
                      img_size=(args.img_size, args.img_size),
                      nufft_pkg=args.nufft_pkg,
                      iSNR=args.isnr)
        else:
            simulator(uv_file, gdth_file, out_dir / f"disk_{i:04d}.mat",
                      super_resolution=args.super_resolution,
                      img_size=(args.img_size, args.img_size),
                      nufft_pkg=args.nufft_pkg,
                      iSNR=args.isnr)
        
        # Explicit garbage collection to prevent memory leaks
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        log_memory()
