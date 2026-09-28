#!/bin/bash
# Execute notebooks/09_receptor_competence.ipynb on an A40 GPU node (full-dataset DE + 06-model predictions).
# The OOD VS Code / Jupyter session node (max031, `normal` partition) has no GPU, so JAX falls back to CPU there
# (cuInit error 303). Submit from max-login:
#   ssh max-login sbatch /fast/AG_Bunina/Pavel_1/cellflow-regen/scripts/run_09_receptor_competence.sbatch
#SBATCH --job-name=cf_09_receptor
#SBATCH --partition=gpu
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=300G
#SBATCH --time=12:00:00
#SBATCH --output=/fast/AG_Bunina/Pavel_1/cellflow-regen/data/09_run.%j.log

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
    09_receptor_competence.ipynb
