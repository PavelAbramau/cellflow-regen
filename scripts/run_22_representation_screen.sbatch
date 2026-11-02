#!/bin/bash
# CPU only. Run from repo root: sbatch scripts/run_22_representation_screen.sbatch
#SBATCH -p batch
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -o logs/22_representation_screen_%j.log
source /apl/trytonp/miniconda3/py313_25.9.1-3/etc/profile.d/conda.sh
conda activate /users/scratch1/pavelabramau/envs/cellflow
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
cd /users/kdm/pavelabramau/cellflow-regen
papermill -k cellflow --cwd notebooks notebooks/22_representation_screen.ipynb notebooks/22_representation_screen.ipynb
