# SETUP — CellFlow environment (2026-11-01)

Installed (pip): jax 0.10.2 + `jax[cuda12]` (jax-cuda12-plugin 0.10.2, CUDA 12.9 wheels), cellflow-tools 0.0.9,
omnipath 1.0.12, scanpy 1.11.5, dcor 0.7, papermill 2.7.0, ipykernel 7.4.0, anndata 0.12.19. rapids-singlecell NOT installed.
Caches/tmp for conda pkgs and pip live under `data_pavel/cellflow/` to spare home quota.

## Verification
* `import jax; jax.devices()` on the login node: `[CpuDevice(id=0)]` (cuInit fails — no GPU/driver there, expected).
* **CudaDevice NOT yet verified.** `srun -p gpu-a100-80gb --gres=gpu:1` queued (cluster full) and was cancelled.
  To verify: `srun -p gpu-a100-80gb --gres=gpu:1 -t 5 --mem=8G -c 2 python -I -c "import jax; print(jax.devices())"`.
* HF cache redirect: default would be `~/.cache/huggingface` (under `$HOME`), so `HF_HOME` was set before download.
