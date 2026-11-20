# Sourced by every job (and by you in an interactive shell): grand + torch environment.
# Set up once as in RUNBOOK_CC.md, step 1; edit the three paths if yours differ.
GRAND_DIR=${GRAND_DIR:-$HOME/WorkingDir/grand}
GNN_ENV=${GNN_ENV:-/sps/grand/aferrier/conda_envs/grand-gnn}
MICROMAMBA=${MICROMAMBA:-$HOME/bin/micromamba}

if [ -x "$MICROMAMBA" ]; then
    export MAMBA_ROOT_PREFIX=${MAMBA_ROOT_PREFIX:-/sps/grand/aferrier/micromamba}
    eval "$("$MICROMAMBA" shell hook -s bash)"
    micromamba activate "$GNN_ENV"
else
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$GNN_ENV"
fi
source "$GRAND_DIR/env/setup.sh" > /dev/null
export PYTHONUNBUFFERED=1   # log lines appear as the job runs
