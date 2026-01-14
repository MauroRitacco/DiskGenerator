import os
import sys
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

DIR_GT = BASE_DIR / "pipeline" / "ground_truth"
DIR_UV = BASE_DIR / "pipeline" / "uv_patterns"
DIR_OUT = BASE_DIR / "pipeline" / "measurements"
os.makedirs(DIR_OUT, exist_ok=True)


from pysrc.utils.io import load_data_to_tensor
from pysrc.utils.gen_imaging_weights import gen_imaging_weights


def simulator(uv_path,gdth_path,measurement_path,super_resolution=1,img_size=(512,512),nufft_pkg='tkbn'):
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
    y_clean = meas_op.forward_op(gdth)

    # Input random Gaussian noise
    iSNR = 40  # input SNR defined by user
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

# Run the simulator for all the files in the respective folders
for i in range(len([f for f in DIR_GT.glob("*") if f.is_file()])):
    simulator(DIR_UV / f"uv_{i:04d}.mat",DIR_GT /f"disk_{i:04d}.fits",
              DIR_OUT / f"measurement_{i:04d}.mat")
