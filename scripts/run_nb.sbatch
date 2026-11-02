#!/bin/bash
# CPU papermill runner: sbatch scripts/run_nb.sbatch <notebook-name-without-ext> [mem]
#SBATCH -p batch
#SBATCH -c 8
#SBATCH --mem=120G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x_%j.log
source /apl/trytonp/miniconda3/py313_25.9.1-3/etc/profile.d/conda.sh
conda activate /users/scratch1/pavelabramau/envs/cellflow
export JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8
cd /users/kdm/pavelabramau/cellflow-regen
papermill -k cellflow --cwd notebooks notebooks/$1.ipynb notebooks/$1.ipynb
