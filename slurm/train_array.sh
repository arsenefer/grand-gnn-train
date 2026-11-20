#!/bin/bash
# One array task per ensemble seed, on a GPU. The dataset must exist first.
#   sbatch --array=0-19 slurm/train_array.sh <gnn_model_card.json> <settings.json>
#SBATCH --job-name=gnn_train
#SBATCH --output=logs/gnn_train_%A_%a.log
#SBATCH --partition=gpu_v100
#SBATCH --gres=gpu:v100:1
#SBATCH --ntasks=1
#SBATCH --mem=10000
#SBATCH --time=0-08:00:00
#SBATCH --licenses=sps

source slurm/env.sh
srun python -m grand_gnn_train --gnn-model-card "$1" --settings "$2" --steps train --seeds "$SLURM_ARRAY_TASK_ID"
