#!/bin/bash
# Exports the trained seeds as one network, plots its performance on the validation graphs,
# and, with a test settings file, tests grand's inference on those other simulations.
#   sbatch slurm/evaluate.sh <gnn_model_card.json> <settings.json> [test_settings.json]
#SBATCH --job-name=gnn_evaluate
#SBATCH --output=logs/gnn_evaluate_%j.log
#SBATCH --partition=htc
#SBATCH --ntasks=1
#SBATCH --mem=16000
#SBATCH --time=0-12:00:00
#SBATCH --licenses=sps

source slurm/env.sh
TEST=${3:+--test-settings $3}
srun python -m grand_gnn_train --gnn-model-card "$1" --settings "$2" --steps export evaluate $TEST
