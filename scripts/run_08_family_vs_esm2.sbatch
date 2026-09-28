#!/bin/bash
# Execute notebooks/08_family_vs_esm2.ipynb on an A40 GPU node (arms B and C trained back to back, ~8 h; arm A reuses the 06 model).
# The OOD VS Code / Jupyter session node (max031, `normal` partition) has no GPU, so JAX falls back to CPU there
# (cuInit error 303). Submit from max-login:
#   ssh max-login sbatch /fast/AG_Bunina/Pavel_1/cellflow-regen/scripts/run_08_family_vs_esm2.sbatch
#SBATCH --job-name=cf_08_family
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=160G
#SBATCH --time=36:00:00
#SBATCH --output=/fast/AG_Bunina/Pavel_1/cellflow-regen/data/08_run.%j.log

set -euo pipefail
REPO=/fast/AG_Bunina/Pavel_1/cellflow-regen
source /fast/AG_Bunina/Pavel_1/miniforge3/bin/activate /fast/AG_Bunina/Pavel_1/miniforge3/envs/cellflow
export TMPDIR=$REPO/data/tmp   # node /tmp is 5 GB
mkdir -p "$TMPDIR"

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
python -c "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"

cd "$REPO/notebooks"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=cellflow \
    --ExecutePreprocessor.timeout=-1 \
    08_family_vs_esm2.ipynb
