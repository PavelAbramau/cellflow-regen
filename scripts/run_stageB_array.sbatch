#!/bin/bash
# Pilot: 2 ligands x 4 Tier-1 arms x 1 seed = 8 tasks.  sbatch --array=0-7 scripts/run_stageB_array.sbatch
# Each task is a parameterised papermill notebook (notebooks/40_run_zeroshot.ipynb) -> notebooks/runs/pilot/<run>.ipynb
#SBATCH -J pilot
#SBATCH -p gpu-a100-80gb
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=60G
#SBATCH -t 12:00:00
#SBATCH -o logs/pilot_%A_%a.log
source /apl/trytonp/miniconda3/py313_25.9.1-3/etc/profile.d/conda.sh
conda activate /users/scratch1/pavelabramau/envs/cellflow
cd /users/kdm/pavelabramau/cellflow-regen
export PYTHONPATH=$PWD/cellflow_fork:${PYTHONPATH:-}
LIGS=(IL-2 IFN-gamma); ARMS=(constant:none esm2:none binding:none binding:receptor)
L=${LIGS[$((SLURM_ARRAY_TASK_ID / 4))]}; A=${ARMS[$((SLURM_ARRAY_TASK_ID % 4))]}; PR=${A%%:*}; GT=${A##*:}
ITERS=${ITERS:-200000}; SEED=${SEED:-0}; TAG=${TAG:-pilot}
mkdir -p notebooks/runs/$TAG
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
papermill -k cellflow --cwd notebooks notebooks/40_run_zeroshot.ipynb notebooks/runs/$TAG/${L}__${PR}__${GT}__s${SEED}.ipynb \
  -p held_out "$L" -p pert_rep $PR -p gate $GT -p seed $SEED -p num_iterations $ITERS -p valid_freq 5000 -p tag $TAG
