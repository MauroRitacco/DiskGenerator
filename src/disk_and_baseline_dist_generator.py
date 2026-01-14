"""
Pipeline Step 1 & 2: Dataset Generation
--------------------------------------
1. Generates Ground Truth (GT) disk images using the DiskGenerator.
2. Runs the VLA simulation script to create UV coverage patterns (.mat files).
3. Organizes output into a clean 'pipeline' directory for training.
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

# Paths
BASE_DIR = Path(os.getcwd()).absolute()

# Source code folders
DISK_GEN_DIR = BASE_DIR / "src"
RI_SRC = BASE_DIR / "utils"
VLA_SCRIPT = RI_SRC / "ri_measurement_operator" / "pyutils" / "sim_vla_ms.py"

# Data output folders
PIPELINE_OUT = BASE_DIR / "pipeline"
GT_DIR = PIPELINE_OUT / "ground_truth"
UV_DIR = PIPELINE_OUT / "uv_patterns"
TEMP_VLA = BASE_DIR / "vla_sims"

# Environment Setup
sys.path.insert(0, str(DISK_GEN_DIR))
sys.path.insert(0, str(RI_SRC))

try:
    from disk_generator import generate_disk

    print("Disk generator loaded.")
except ImportError as e:
    print(f"Error loading disk_generator: {e}")
    sys.exit(1)

# Config
NUM_SAMPLES = 5
IMG_SIZE = 512


def step_1_generate_gt():
    #Create the synthetic disk images (Ground Truth)
    print(f"Step 1: Generating {NUM_SAMPLES} disks...")

    if not GT_DIR.exists():
        GT_DIR.mkdir(parents=True)

    generate_disk(
        output_dir=str(GT_DIR),
        num_samples=NUM_SAMPLES,
        img_size=IMG_SIZE
    )


def step_2_generate_uv():
    # Run VLA simulator to get UV coverage patterns
    print("Step 2: Simulating VLA UV coverage...")

    if not VLA_SCRIPT.exists():
        print(f"Script missing at: {VLA_SCRIPT}")
        sys.exit(1)

    # Set PYTHONPATH so the subprocess can find the RI modules
    env = os.environ.copy()
    env["PYTHONPATH"] = str(RI_SRC) + os.pathsep + env.get("PYTHONPATH", "")

    # Run the external VLA simulation script
    cmd = [sys.executable, str(VLA_SCRIPT), "--npatterns", str(NUM_SAMPLES)]

    try:
        subprocess.run(cmd, check=True, cwd=BASE_DIR, env=env)
    except subprocess.CalledProcessError as e:
        print(f"VLA simulation failed: {e}")
        sys.exit(1)

    # Create final directory for UV patterns
    if not UV_DIR.exists():
        UV_DIR.mkdir(parents=True)

    print(f"Moving files to: {UV_DIR}")

    # The VLA script outputs to a temp 'uvw' folder
    uvw_temp_path = TEMP_VLA / "uvw"
    if uvw_temp_path.exists():
        for mat_file in uvw_temp_path.glob("*.mat"):
            shutil.move(str(mat_file), str(UV_DIR / mat_file.name))

        # Cleanup temp VLA directory
        shutil.rmtree(TEMP_VLA, ignore_errors=True)
        print("UV patterns moved and temp files cleaned.")
    else:
        print("Error: No .mat files found in temp directory.")


if __name__ == "__main__":
    step_1_generate_gt()
    step_2_generate_uv()
    print("Pipeline data ready.")