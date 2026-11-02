#!/bin/bash
#SBATCH -p batch
#SBATCH -c 4
#SBATCH --mem=48G
#SBATCH -t 04:00:00
#SBATCH -o logs/20_pseudobulk_%j.log
/users/scratch1/pavelabramau/envs/cellflow/bin/python -I scripts/20_pseudobulk.py data/pseudobulk_log1p_cp10k.npz
