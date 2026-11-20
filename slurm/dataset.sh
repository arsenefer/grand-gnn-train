#!/bin/bash
# Formats the simulations of a GNN model card into graphs (CPU; reused if they exist).
#   sbatch slurm/dataset.sh <gnn_model_card.json> <settings.json> [max_events]
#SBATCH --job-name=gnn_dataset
#SBATCH --output=logs/gnn_dataset_%j.log
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --time=2-00:00:00
#SBATCH --licenses=sps

source slurm/env.sh
MAX=${3:+--max-events $3}
srun python -m grand_gnn_train --gnn-model-card "$1" --settings "$2" --steps dataset $MAX
