#!/bin/bash
# Execute notebooks/20_response_matrix.ipynb on a CPU node (reads data/pseudobulk_log1p_cp10k.npz from scripts/run_20_pseudobulk.sbatch).
#   sbatch scripts/run_20_response_matrix.sbatch   (from the repo root)
#SBATCH -p batch
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/20_response_matrix_%j.log
source /apl/trytonp/miniconda3/py313_25.9.1-3/etc/profile.d/conda.sh
conda activate /users/scratch1/pavelabramau/envs/cellflow
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES=
cd /users/kdm/pavelabramau/cellflow-regen
papermill -k cellflow --cwd . notebooks/20_response_matrix.ipynb notebooks/20_response_matrix.ipynb
