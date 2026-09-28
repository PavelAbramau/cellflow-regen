#!/bin/bash
# Execute notebooks/07_condition_embedding.ipynb on a CPU node (Part II reads the full 9.7M-cell dataset once; result cached).
# Submit from max-login:
#   ssh max-login sbatch /fast/AG_Bunina/Pavel_1/cellflow-regen/scripts/run_07_condition_embedding.sbatch
#SBATCH --job-name=cf_07_embedding
#SBATCH --partition=normal
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
#SBATCH --output=/fast/AG_Bunina/Pavel_1/cellflow-regen/data/07_run.%j.log

set -euo pipefail
REPO=/fast/AG_Bunina/Pavel_1/cellflow-regen
source /fast/AG_Bunina/Pavel_1/miniforge3/bin/activate /fast/AG_Bunina/Pavel_1/miniforge3/envs/cellflow
export TMPDIR=$REPO/data/tmp JAX_PLATFORMS=cpu
hostname
cd "$REPO/notebooks"
jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=cellflow \
    --ExecutePreprocessor.timeout=-1 \
    07_condition_embedding.ipynb
