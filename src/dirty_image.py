import os
import gc
import yaml
import argparse
from math import *
from pathlib import Path

import torch
import numpy as np
from scipy.io import loadmat
from astropy.io import fits

import matplotlib.pyplot as plt
import matplotlib
from matplotlib.ticker import FormatStrFormatter

# Styling defaults from your script
plt.rcParams.update({'font.size': 15})
matplotlib.rcParams['mathtext.fontset'] = 'stix'
matplotlib.rcParams['font.family'] = 'STIXGeneral'

# Path setup to include local utils
BASE_DIR = Path(__file__).resolve().parent.parent
import sys
sys.path.insert(0, str(BASE_DIR / "utils" / "ri_measurement_operator"))

from pysrc.utils.io import load_data_to_tensor

def generate_dirty_images():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--data_file", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--generate_validation",type=bool,default=False)
    args = parser.parse_args()

    # Read yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Resolution settings
    super_resolution = config.get("super_resolution", 1)
    img_size = (args.img_size, args.img_size)
    real_flag = True

    # Get execution directory fallback mechanisms
    out_dir_path = args.output_path if args.output_path else config.get("output_path", BASE_DIR / "pipeline" / "dirty")
    out_dir = Path(out_dir_path).resolve()
    
    if args.generate_validation:
        res_dir = out_dir / "validationset_resN1"
        rec_dir = out_dir / "validationset_recN1"
    else:
        res_dir = out_dir / "trainingset_resN1"
        rec_dir = out_dir / "trainingset_recN1"
    os.makedirs(res_dir, exist_ok=True)
    os.makedirs(rec_dir, exist_ok=True)
    
    # Locate dataset
    meas_dir = Path(args.data_file).parent if args.data_file else BASE_DIR / "pipeline" / "dataset" / "measurements"
    mat_files = sorted(list(meas_dir.glob("*.mat")))

    if not mat_files:
        print(f"ERROR: No .mat files found in {meas_dir}")
        return
        
    print(f"INFO: Generating dirty images for {len(mat_files)} measurement files...")
    
    # Import the NUFFT package
    nufft_pkg = config.get("nufft_pkg", "finufft")
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
            
    nufft_kernel_dim = config.get("nufft_kernel_dim", 7)
    nufft_oversampling_factor = config.get("nufft_oversampling_factor", 2.0)
    nufft_grid_size = tuple([int(nufft_oversampling_factor * i) for i in img_size])

    for mat_path in mat_files:
        print(f"Processing: {mat_path.name}...")
        
        # 1. Load measurement dataset explicitly generated from simulator.py
        # This parses u, v, w, y (which contains simulator noise), and nW (noise weight parameter)
        # Note: If data_weighting is True, load_data_to_tensor intrinsically applies nW and calculates nWimag into y.
        data = load_data_to_tensor(
            uv_file_path=str(mat_path),
            super_resolution=super_resolution,
            img_size=img_size,
            data_weighting=config.get("data_weighting", True),
            load_weight=config.get("load_weight", False),
            weight_type=config.get("weight_type", "briggs"),
            weight_robustness=config.get("weight_robustness", 0.0),
        )
        
        # 2. Extract components
        nW = data["nW"]
        nWimag = data.get("nWimag", torch.ones_like(nW))
        
        # 3. Create weighted operator natively combining weights
        meas_op_weighted = Operator(
            u=data["u"],
            v=data["v"],
            img_size=img_size,
            real_flag=real_flag,
            natural_weight=nW,
            image_weight=nWimag,
            grid_size=nufft_grid_size,
            kernel_dim=nufft_kernel_dim,
            dtype=torch.float64
        )
        
        with torch.no_grad():
            # 4. Generate PSF and Extract Kappa factor 
            dirac = torch.zeros(1, 1, *img_size, dtype=torch.float64, device=data["u"].device)
            dirac[0, 0, img_size[0] // 2, img_size[1] // 2] = 1
            
            psf_0 = meas_op_weighted.adjoint_op(meas_op_weighted.forward_op(dirac))
            kappa = 1 / torch.max(psf_0)
            
            # 5. Calculate back-projected dirty image natively using pre-calculated Operator Weight Math
            dirty = kappa * meas_op_weighted.adjoint_op(data["y"])
            dirty_norm = dirty.squeeze().cpu().numpy()

        # 6. Save results locally in separated folders
        base_id = mat_path.stem.split("_")[-1]
        if args.generate_validation:
            res_name = f"val_disk_{base_id}_res.fits"
            fits.writeto(res_dir / res_name, dirty_norm, overwrite=True)

            rec_name = f"val_disk_{base_id}_rec.fits"
            fits.writeto(rec_dir / rec_name, np.zeros_like(dirty_norm), overwrite=True)
        else:
            res_name = f"disk_{base_id}_res.fits"
            fits.writeto(res_dir / res_name, dirty_norm, overwrite=True)

            rec_name = f"disk_{base_id}_rec.fits"
            fits.writeto(rec_dir / rec_name, np.zeros_like(dirty_norm), overwrite=True)
        
        del data, nW, nWimag, meas_op_weighted, dirac, psf_0, dirty, dirty_norm
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

if __name__ == "__main__":
    generate_dirty_images()