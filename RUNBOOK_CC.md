# CC-IN2P3 runbook: dataset, training, evaluation

Goal of the day: on CC-IN2P3, with the simulations on `/sps`,

1. build the dataset from the **DC2 ADC L1 simulations with noise (the `*-AN` directories)**;
2. train the **N = 20 seeds** of the networks of the GNN paper (GNN_forPaper
   `after_writing_selection`: configuration 33 for the direction, 28 for the energy);
3. get the **performance plots** automatically at the end of training, then **test the inference**
   (grand's `recons_GNN` / `recons_energy_GNN`) on other simulations.

Everything is one command per step, or one `slurm/submit_all.sh` for all of them.

| step | command | where | writes |
|---|---|---|---|
| check the simulations | `python -m grand_gnn_train.simulations --settings settings/ccin2p3.json` | login node | nothing (prints) |
| trial on 50 events | `python -m grand_gnn_train ... --max-events 50 --seeds 0` | interactive job | `.../trials_50_events/` |
| everything | `slurm/submit_all.sh settings/ccin2p3.json - <cards>` | login node → SLURM | dataset, seeds, network, plots |
| inference test | `sbatch slurm/evaluate.sh <card> settings/ccin2p3.json settings/ccin2p3_test.json` | SLURM | `evaluation/test_ccin2p3_test/` |

Paths used (from `settings/ccin2p3.json`; change them there if needed):

| what | path |
|---|---|
| simulations | `/sps/grand/DC2.1rc4/*-AN/*/`, files `adc_*_L1_*.root` (tree `tadc`) |
| datasets | `/sps/grand/aferrier/DATA/GNN/grand_gnn_train/datasets/` |
| trained seeds + plots | `/sps/grand/aferrier/MODELS/GNN/grand_gnn_train/runs/<name>_v1/` |
| networks for grand | `/sps/grand/aferrier/MODELS/GNN/grand_gnn_train/models/<name>/v1/` |
| test simulations | **blank**: fill `directories` in `settings/ccin2p3_test.json` |

The simulation path is the one `GNN_forPaper/make_dataset_root.py` used on CC (its commented
"paths CC" line). Step 2 checks that it is still right.

---

## 0. Before leaving the laptop (5 min)

Both repositories have uncommitted changes (branch `gnn_reconstruction` of grand, and
grand-gnn-train), so copy the working trees. The data grand downloads are left out: `env/setup.sh`
downloads them again on CC.

```bash
rsync -av --exclude '__pycache__' --exclude 'build' --exclude '*.egg-info' \
    --exclude 'data/detector' --exclude 'data/geomagnet' --exclude 'data/topography' --exclude 'data/noise' \
    ~/Documents/grand/ aferrier@cca.in2p3.fr:WorkingDir/grand/
rsync -av --exclude '__pycache__' --exclude 'work' --exclude 'logs' \
    ~/Documents/WorkingDir/grand-gnn-train/ aferrier@cca.in2p3.fr:WorkingDir/grand-gnn-train/
```

## 1. Environment on CC (once, ~30 min)

```bash
ssh aferrier@cca.in2p3.fr

# micromamba, if you have no conda at hand (conda 22 ran out of memory on the laptop)
cd ~ && curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
export MAMBA_ROOT_PREFIX=/sps/grand/aferrier/micromamba
eval "$(~/bin/micromamba shell hook -s bash)"

# grand's environment (ROOT, uproot, scipy, matplotlib, compilers), on /sps where there is room
micromamba create -y -p /sps/grand/aferrier/conda_envs/grand-gnn -f ~/WorkingDir/grand/env/conda/grand-dev.yml
micromamba activate /sps/grand/aferrier/conda_envs/grand-gnn

# torch for the V100 GPUs, the same version as on the laptop, then PyTorch Geometric
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install torch_geometric pandas
pip install -e ~/WorkingDir/grand-gnn-train

# grand: compiles TURTLE/GULL, downloads the data model
cd ~/WorkingDir/grand && source env/setup.sh && cd ~/WorkingDir/grand-gnn-train

# check
python -c "import torch, grand.analysis.gnn as g; print(torch.__version__, 'GNN code OK')"
```

`slurm/env.sh` activates this environment in every job. Its three paths (`GRAND_DIR`, `GNN_ENV`,
`MICROMAMBA`) are those above; edit them there if you put things elsewhere. In a new shell:
`cd ~/WorkingDir/grand-gnn-train && source slurm/env.sh`.

If `import torch` then `import ROOT` fails with `GLIBCXX_3.4.31` (it did on the laptop), the
pipeline is not concerned: it reads the simulations with uproot, without ROOT.

## 2. Check the simulations (10 min)

```bash
ls -d /sps/grand/DC2.1rc4/*-AN/*/ | wc -l            # how many directories
ls "$(ls -d /sps/grand/DC2.1rc4/*-AN/*/ | head -1)"   # the file names of one of them
python -m grand_gnn_train.simulations --settings settings/ccin2p3.json --events 10
```

The last command prints the files each pattern finds, then, for 10 events, the number of antennas,
the trace shape, `du_seconds`, the median peak amplitude, how many antennas pass the 5σ trigger,
zenith and energy, and at the end an estimate of the noise. Check:

- [ ] **Directories and files.** Each pattern finds files. If `adc_*_L1_*.root` finds nothing,
      look at the names (`ls` above) and set `trace_files` / `time_files` in `settings/ccin2p3.json`.
- [ ] **Channels.** The trace shape is `(channels, samples)`. The pipeline takes x, y as channels 0, 1
      of simulated traces, as the GNN codes did on DC2 `tadc`. If the traces have 4 channels with x,
      y at 1, 2, change `CHANNEL_INDEX["sim"]` in `grand/analysis/constants.py` (the one place).
- [ ] **Start times.** `du_seconds` should be filled (not all 0). If not, take the times where the
      GNN codes did: `"time_files": "voltage_*_L0_*.root", "time_tree": "tvoltage"`.
- [ ] **Noise and trigger.** The cards trigger at 5σ with `noise_rms_per_channel` = 15 ADC
      (5·15·√2 = 106 ADC on x, y). Compare with the printed noise RMS. The latest GNN_forPaper
      dataset (`GNNdata_45fADC_dummy_smearing`) used 5σ of *its* noise, about 45 ADC on x, y. If the
      AN files' noise differs, set `inputs.trigger.noise_rms_per_channel` to the measured value in
      **both** cards (`gnn_model_cards/pgnn_*_xy.json`) before formatting.
- [ ] **Truth.** The printed zenith is a plausible angle in degrees (if it reads below 2°, the
      files store radians: set `"truth_angle_unit": "rad"`), and `E_em` is in GeV (1e7–1e10).
- [ ] **Layouts.** `max_du_id: 277` keeps GP300 antennas in *every* directory. The GNN codes
      applied it only to `GP300` directories: if the glob also finds other layouts, narrow
      `directories` or set `max_du_id` to `null`.

## 3. A trial on 50 events (15 min)

In an interactive job, so the login node is spared:

```bash
srun -p htc_interactive --mem=8G -t 0-02:00:00 --pty bash -i   # or any interactive session
cd ~/WorkingDir/grand-gnn-train && source slurm/env.sh
python -m grand_gnn_train --gnn-model-card gnn_model_cards/pgnn_direction_xy.json \
    --settings settings/ccin2p3.json --max-events 50 --seeds 0
```

It runs every step on 50 events and one seed (110 epochs on CPU take a few minutes on 40 graphs)
and writes the plots in `.../runs/trials_50_events/pgnn_direction_xy_v1/evaluation/validation/`.
Trials have their own dataset, seeds and network: they never mix with the real run. This is only to
see the whole chain work on the real files, not to judge the network.

## 4. Submit everything (2 min, then hours)

```bash
cd ~/WorkingDir/grand-gnn-train
slurm/submit_all.sh settings/ccin2p3.json - \
    gnn_model_cards/pgnn_direction_xy.json gnn_model_cards/pgnn_energy_xy.json
squeue --me
tail -f logs/gnn_dataset_*.log
```

To also train the alternative network (`*_cat5`: the same EdgeCNN for both tasks, 5 layers of 256
with batch norm, jk `cat`, 8 neighbours; inputs, normalisation and training as v1), add its cards:

```bash
slurm/submit_all.sh settings/ccin2p3.json - \
    gnn_model_cards/pgnn_direction_xy.json gnn_model_cards/pgnn_energy_xy.json \
    gnn_model_cards/pgnn_direction_xy_cat5.json gnn_model_cards/pgnn_energy_xy_cat5.json
```

They need the same graphs as the energy v1 card (same inputs, 8 neighbours): one dataset for the
three, plus the direction v1's (32 neighbours). That makes 80 GPU jobs (4 cards × 20 seeds).

For each card this submits, chained with `--dependency=afterok`:

1. `slurm/dataset.sh` (htc, 16 GB, up to 2 days): formats the simulations. The direction links each
   antenna to 32 neighbours and the energy to 8, so each card has its own dataset; the two are
   formatted in parallel. (Cards needing the same graphs would share one: the second card's dataset
   job waits for the first, finds the dataset and stops at once.)
2. `slurm/train_array.sh` (gpu_v100, one V100 per seed, 8 h max each): an array of 20 jobs, one
   per seed, as the cards say. The GNN codes' job asked 8 h for its 20 seeds in a row, so one seed
   per job should be well within the limit.
3. `slurm/evaluate.sh` (htc): exports the 20 seeds as one network for grand, then evaluates it.

Replace `-` by `settings/ccin2p3_test.json` once its directories are filled, and step 6 is done
too. If `sbatch` asks for an account, add `#SBATCH --account=grand` to the three job scripts.

Running anything again is safe: a finished dataset is reused, trained seeds are skipped, and the
export overwrites the network. A card changed under the same name and version is refused: give it a
new `version`.

## 5. Read the results

In `/sps/grand/aferrier/MODELS/GNN/grand_gnn_train/runs/<name>_v1/evaluation/validation/`:

| file | what to look at |
|---|---|
| `summary.json` | the numbers: 68 % angular error (GNN and plane wave), energy bias and resolution, pull mean and std, 1σ coverage |
| `training_losses.png` | training and validation loss of each seed: decreasing, then flat, without divergence; seeds alike |
| `direction_angular_error.png` | angular error of the GNN against the plane-wave fit on the same antennas: the GNN histogram should sit to the left |
| `direction_resolution_vs.png` | 68 % angular error vs true zenith, energy and number of antennas |
| `direction_reco_vs_true.png` | reconstructed vs true zenith; azimuth residual |
| `energy_reco_vs_true.png` | log10(E_reco/E_true) and reconstructed vs true E_em |
| `energy_resolution_vs.png` | bias and 68 % half-width vs energy and zenith |
| `uncertainty_pulls.png` | pulls against N(0, 1); coverage: the curves should follow the black one (above: σ overestimated, below: underestimated) |
| `predictions.csv` | one row per event, to make any other plot |

Bring them back to the laptop:

```bash
rsync -av --include '*/' --include '*.png' --include '*.json' --include '*.csv' --exclude '*' \
    aferrier@cca.in2p3.fr:/sps/grand/aferrier/MODELS/GNN/grand_gnn_train/runs/ ~/Documents/gnn_runs/
```

The numbers are comparable with the paper's, but not identical: these datasets come from the DC2
ADC L1 AN files with grand's peak functions (`sqrt(|H(x)|² + |H(y)|²)`), where the paper's came from
voltage simulations, `new_rfchain` noise and `|H(‖x, y‖)|`.

## 6. Inference test on other simulations

1. Fill `directories` in `settings/ccin2p3_test.json` with simulations **not** matched by the
   training glob (otherwise the test is not one). Check them as in step 2:
   `python -m grand_gnn_train.simulations --settings settings/ccin2p3_test.json`.
2. Then:

```bash
sbatch slurm/evaluate.sh gnn_model_cards/pgnn_direction_xy.json settings/ccin2p3.json settings/ccin2p3_test.json
sbatch slurm/evaluate.sh gnn_model_cards/pgnn_energy_xy.json settings/ccin2p3.json settings/ccin2p3_test.json
```

Each event of those simulations goes through what grand does on measured data: peaks with
`grand.analysis.signals` on the card's channels, then `gnn.recons_GNN` / `gnn.recons_energy_GNN`
with the exported network. The same plots and `summary.json` land in
`.../runs/<name>_v1/evaluation/test_ccin2p3_test/`. The test can also be run on its own:
`python -m grand_gnn_train.evaluate --gnn-model-card <card> --settings settings/ccin2p3.json
--test-settings settings/ccin2p3_test.json [--max-events 2000]`.

To use the network in grand afterwards (e.g. `examples/analysis/main_DOI.py`), name its directory in
`examples/analysis/config.py`:
`gnn_models = {"direction": "/sps/grand/aferrier/MODELS/GNN/grand_gnn_train/models/pgnn_direction_xy/v1", ...}`.

## If something goes wrong

| symptom | cause, fix |
|---|---|
| `no simulation directory matches` | the glob in `settings/ccin2p3.json`; check with `ls -d` |
| `<dir>: no file matches adc_*_L1_*.root` | other file names or level: set `trace_files` / `time_files` |
| `ModuleNotFoundError: grand._core` | `source env/setup.sh` failed (it prints which step): rerun it in grand's directory |
| `No module named grand_gnn_train` | `pip install -e ~/WorkingDir/grand-gnn-train`, or run from that directory |
| dataset job killed at 2 days | raise `--time` in `slurm/dataset.sh` and resubmit (it starts over: only a finished dataset is reused), or narrow `directories` |
| `holds seeds of another GNN model card` | the card changed: bump its `version` |
| a training job ran out of time | resubmit just that seed: `sbatch --array=<seed> slurm/train_array.sh <card> settings/ccin2p3.json` |
| very few events in the dataset (`too_few_antennas` large in `dataset_card.json`) | the trigger is too high for this noise: step 2, noise and trigger |

## What the cards are

`gnn_model_cards/pgnn_direction_xy.json` and `pgnn_energy_xy.json` (version `v1`) are the networks
of the GNN paper, selected in GNN_forPaper `reconstructions/*_ADC/after_writing_selection/` (the ones
`plot_paper/direction_recons_plots.ipynb` and `energy_recons_plots.ipynb` load):

- direction: `Direction_ADC/after_writing_selection/configs/config_33.json`,
  `model_k_h256_l4_d0_seqT_rmean-max-sum_nbatch_i6_pTrue_tr0_nn32_lfGaussianNLLLoss_33`
  (trained in `no_trace_after_writing_selection`);
- energy: `Energy_ADC/after_writing_selection/configs/config_28.json`,
  `model_E_em_h64_l7_d0_seqF_rmean-max-sum_nbatch_i6_pTrue_tr0_nn8_lfGaussianNLLLoss_28`
  (trained in `no_trace_after_writing_selection_EM`).

`gnn_utils/model.py` (unchanged since 2025-08-27, before these trainings) maps
`sequential: true/false` to jk `last`/`cat`, and its `GNNScale` uses the mean and max readouts
whatever `readout` says (`"sum"` was not even a name it knew; it expected `"add"`).

| GNN_forPaper | card |
|---|---|
| inputs `pos, pos, pos, t_sm, p2p_sm, delta_t`, `pwf_input: true` | `node_features` pos_x, pos_y, pos_z, time, amplitude, delta_t; `graph_features` k_pwf_x/y/z |
| `NormaliseLayer` (pos/1e4, z/1e2, t·2e4, amp^0.2/8) and `NormaliseLayerDeltaT` (`delta_t`/2.6e-8) | `"normalisation": "gnn_lib"` (those constants, in grand's `features.py`) |
| `make_delta_t`: speed c·1.00014 | `delta_t_wave_speed` 299834428.9 m/s |
| direction: hidden 256, 4 layers, norm `batch`, `sequential: true` → jk `last`, mean+max readout | `architecture` |
| energy: hidden 64, 7 layers, norm `batch`, `sequential: false` → jk `cat`, mean+max readout | `architecture` |
| `n_neighbours`: 32 (direction), 8 (energy) | `knn_k` |
| GaussianNLLLoss (eps 1e-6), 110 epochs as 110//7, 110//2.5, 110//2.5, 110//7 at 1.5e-4, 8e-5, 4e-5, 2e-5, a new Adam per phase; batch 4 | `training.phases`, `batch_size`; the same loss |
| `EarlyStopper(patience=50)` made anew each phase (phases are shorter), its "best" `state_dict` not copied: the saved network is the last state | `"patience": null`, `"keep": "last"` |
| seeds `range(0, 20)` | `training.seeds` 0–19 (N: edit this list) |
| `train_size` 5000, `val_size` 1200 (the first files) | `max_train_events` 5000, `max_val_events` 1200 (the first graphs); `null` uses the whole dataset |
| jitter 5 ns, gain smearing 7 % | `simulations.augment` (defaults) |
| noise from `new_rfchain` added to voltage | none added: the AN files already have their noise |
| validation: `event_number % 10` in {0, 1} | `validation_modulo` (default) |

What differs by design: the simulations (DC2 ADC L1 AN files instead of voltage simulations with
`new_rfchain` noise), hence the trigger threshold (5σ of the measured noise: step 2); the peak
amplitude, grand's `get_peak_amplitude` (`sqrt(|H(x)|² + |H(y)|²)`) instead of `|H(‖x, y‖)|`; the
plane-wave fit, grand's `PWF_semianalytical` with n = 1.000136 instead of `GNN_lib`'s with 1.00014.

To try another normalisation: `"normalisation": "fit"` fits, on the training graphs, the power `p`
of `sign(x)·|x|^p` of each feature (maximum likelihood, as a Box-Cox transform: about 0.2 for
amplitudes spread over decades, about 1 for positions), then centres and scales it to a standard
deviation of 1. The fitted constants are in `runs/<name>_<version>/normalisation.json` and in the
exported network's card. Give such a card another `name` or `version`.
