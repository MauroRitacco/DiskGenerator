import os
import sys
import torch
import numpy as np
from pathlib import Path
from astropy.io import fits

# 1. Path Configuration
BASE_DIR = Path(os.getcwd()).absolute()
sys.path.insert(0, str(BASE_DIR))
sys.path.insert(0, str(BASE_DIR / "utils"))

# Directories (Solo las de entrada son fijas, la de salida depende del config)
DIR_MEASUREMENTS = BASE_DIR / "pipeline" / "measurements"

# Import tools from your local files
from utils import load_data_to_tensor, create_meas_op, parse_args_imaging


def generate_training_data():
    """
    Generates normalized dirty images and initial reconstructions.
    Output location is defined by 'output_path' in config.yaml.
    """
    # Load arguments (incluyendo output_path del yaml)
    args = parse_args_imaging()

    # --- CONFIGURACIÓN DE RUTAS DE SALIDA ---
    # Usamos args.output_path definido en tu YAML
    # Convertimos a Path absoluto para evitar ambigüedades
    output_root = Path(args.output_path).resolve()

    print(f"INFO: Output Directory set to: {output_root}")

    # --- DEFINICIÓN DE EXTENSIONES ---
    RES_EXT = "_resN1"  # Sufijo carpeta residuos
    REC_EXT = "_recN1"  # Sufijo carpeta reconstrucciones
    DIRTY_FILE_EXT = "_dirty"  # Sufijo archivo dirty
    REC_FILE_EXT = "_rec"  # Sufijo archivo rec

    # Create output folders inside the output_path defined in yaml
    dir_res = output_root / f"ground_truth{RES_EXT}"
    dir_rec = output_root / f"ground_truth{REC_EXT}"

    os.makedirs(dir_res, exist_ok=True)
    os.makedirs(dir_rec, exist_ok=True)

    # Search for visibility files (.mat)
    mat_files = sorted(list(DIR_MEASUREMENTS.glob("*.mat")))

    if not mat_files:
        print(f"ERROR: No .mat files found in {DIR_MEASUREMENTS}")
        return

    print(f"INFO: Found {len(mat_files)} files. Device: {args.device}")

    for mat_path in mat_files:
        # Load visibility data (y) and weights (nW)
        try:
            data = load_data_to_tensor(
                uv_file_path=str(mat_path),
                super_resolution=args.super_resolution,
                image_pixel_size=args.image_pixel_size,
                device=args.device
            )
        except Exception as e:
            print(f"Skipping {mat_path.name}: {e}")
            continue

        # Instantiate the Measurement Operator (Phi)
        meas_op = create_meas_op(args=args, data=data, device=args.device)

        with torch.no_grad():
            # Initial residual r(0) is the normalized dirty image
            vis_weighted = data["y"] * data["nW"]
            dirty_image = meas_op.adjoint_op(vis_weighted)

            # Normalization by PSF peak
            psf = meas_op.get_psf()
            psf_peak = torch.amax(psf, dim=(-2, -1), keepdim=True)
            dirty_norm = (dirty_image / psf_peak).squeeze().cpu().numpy()

        # Filename handling
        base_id = mat_path.stem.split("_")[-1]
        clean_fname = f"disk_{base_id}"

        # 1. Save Residual (Dirty Image)
        res_name = f"{clean_fname}{DIRTY_FILE_EXT}.fits"
        fits.writeto(dir_res / res_name, dirty_norm, overwrite=True)

        # 2. Save Initial Reconstruction (Zeros)
        rec_zeros = np.zeros_like(dirty_norm)
        rec_name = f"{clean_fname}{REC_FILE_EXT}.fits"
        fits.writeto(dir_rec / rec_name, rec_zeros, overwrite=True)

        print(f"  Generated tuples for: {clean_fname} in {output_root.name}")


if __name__ == "__main__":
    generate_training_data()