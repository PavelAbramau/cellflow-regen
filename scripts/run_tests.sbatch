#!/bin/bash
#SBATCH -J unittests
#SBATCH -p batch
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/unittests_%j.log
source /apl/trytonp/miniconda3/py313_25.9.1-3/etc/profile.d/conda.sh
conda activate /users/scratch1/pavelabramau/envs/cellflow
cd /users/kdm/pavelabramau/cellflow-regen
JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES= python -m pytest -v tests/test_arms.py
