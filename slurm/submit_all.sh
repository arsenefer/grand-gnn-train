#!/bin/bash
# Submits everything for one or more GNN model cards, chained with SLURM dependencies:
#   dataset  ->  one GPU job per seed  ->  export + evaluation (+ inference test)
#
#   slurm/submit_all.sh <settings.json> [test_settings.json|-] <gnn_model_card.json> [more cards...]
#
# Run it from the repository root. Cards whose graphs differ (other inputs or neighbours) get
# their own dataset, formatted in parallel. Cards needing the same graphs share one: the second
# card's dataset job waits for the first, finds the dataset and stops at once.
set -e
SETTINGS=${1:?usage: slurm/submit_all.sh <settings.json> [test_settings.json|-] <card.json> [more cards...]}
TEST=${2:?give a test settings file, or - for none}
shift 2
[ "$TEST" = "-" ] && TEST=""
mkdir -p logs
source slurm/env.sh

declare -A dataset_job   # dataset directory -> the job formatting it
for CARD in "$@"; do
    read -r seeds dataset_dir < <(python -c "
from grand_gnn_train import config
run = config.load('$CARD', '$SETTINGS')
print(','.join(map(str, run['gnn_model_card']['training']['seeds'])), run['dataset_dir'])")
    previous=${dataset_job[$dataset_dir]}
    dataset=$(sbatch --parsable ${previous:+--dependency=afterok:$previous} slurm/dataset.sh "$CARD" "$SETTINGS")
    dataset_job[$dataset_dir]=$dataset
    train=$(sbatch --parsable --dependency=afterok:$dataset --array="$seeds" slurm/train_array.sh "$CARD" "$SETTINGS")
    evaluate=$(sbatch --parsable --dependency=afterok:$train slurm/evaluate.sh "$CARD" "$SETTINGS" $TEST)
    echo "$CARD: dataset job $dataset ($dataset_dir), training array $train (seeds $seeds), evaluation job $evaluate"
done
echo "Follow with: squeue --me ; logs in logs/"
