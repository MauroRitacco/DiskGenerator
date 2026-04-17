# Pipeline Optimization & Neural Network Preparedness

## 1. Pipeline Automation
* **End-to-End Execution**: Integrated `dirty_image.py` into the main `run_pipeline.sh` script to fully automate the dataset generation process. 
* **Dynamic Sizing**: Added CLI arguments to `disk_generator.py`, `simulator.py`, and `dirty_image.py` to universally scale image resolution geometry.
* **Symlink Generation**: Added a 5th pipeline step using bulletproof absolute path shortcuts (`ln -sfn $(pwd)`) to neatly structure and link the core generated measurements and ground-truth datablocks directly inside your target iteration working folder.

## 2. Memory & Scaling Fixes
* **PyTorch Graph Builder**: De-activated the automatic underlying autograd logic (`with torch.no_grad():`) during Fourier simulation steps to prevent invisible RAM hoarding.
* **Aggressive Garbage Collection**: Added explicit variable destruction (`del`) at the end of function scopes in `simulator.py` and `dirty_image.py`, coupled with immediate `gc.collect()` calls. The pipeline can now safely scale to `n=2500` (or more) iterations without memory crashes.

## 3. Deep Learning Target Distribution
* **Artifact Domain Randomization**: Updated `sim_vla_ms.py` to randomly distribute Fourier $u,v$ geometry across a continuous spectrum to ensure the network properly models PSF structural laws rather than overfitting to static noise.
* **75 / 25 Network Skew**: Weighted the probability distribution.
  * **75% Severe Artifacts**: Short VLA tracking durations (0.5 - 2.0h) with intense snapshot gaps to aggressively train the network to resolve massive sidelobe streaks.
  * **25% Identity Mapping**: Long VLA tracking durations (2.0 - 8.0h) with tight gaps to act as a mathematical "anchor", preventing the network from hallucinating or behaving overly aggressively on clean data.
