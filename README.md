# grand-gnn-train

Simulation formatting and training of the graph neural networks (GNN) that
`grand.analysis.gnn` uses to reconstruct the shower direction and
electromagnetic energy. Inference lives in grand; this repository only makes
the networks.

```
GNN model card ─────────▶ dataset (built, or reused) ──▶ seeds ─────▶ <models_dir>/<name>/<version>/ ──▶ grand
gnn_model_cards/*.json    grand.analysis.signals,           grand's     gnn_model_card.json + seed_XX.pt      (forge: manual upload)
                          grand.analysis.gnn.features       network
```

```bash
python -m grand_gnn_train --gnn-model-card gnn_model_cards/pgnn_direction_xy.json --settings settings/ccin2p3.json
```

## One definition of the inputs

The peaks are taken with `grand.analysis.signals` (the functions of the ADF
chain), and the graph -- trigger, plane-wave fit, the GNN model card's features in its
order, neighbours -- with `grand.analysis.gnn.features.make_graph`, the
same call `grand.analysis.gnn.recons_GNN` makes on a measured event. The
network is grand's `GNNScaleMDN`. Nothing about the
inputs is re-implemented here, so a network cannot be trained on something
different from what it sees in grand (`tests/test_pipeline.py` checks this).

Features are the functions of `NODE_FEATURES` and `GRAPH_FEATURES` in grand's
`grand/analysis/gnn/features.py`; a GNN model card names the ones it uses, in order. To
add one, add its function there, so inference computes it the same way, then
name it in a GNN model card (with `"normalisation": "fit"`, or give it its constants).

## The GNN model card

The input of a training is a GNN model card, `gnn_model_cards/<name>.json`: what the
network is and how it is trained. It is copied, completed, into the network
grand stores and reads.

| block | what |
|---|---|
| `name`, `version`, `task`, `description` | `task`: `direction` or `energy` |
| `inputs` | `channels` (`["x","y"]` by default), `node_features` and `graph_features` (in order), `trigger` (antennas above `n_sigma` × noise, default 5σ with 15 ADC per channel; `min_antennas`), `knn_k` (neighbours, default 8), `delta_t_wave_speed` |
| `normalisation` | `"gnn_lib"` (default: GNN_lib's `NormaliseLayer`, as GNN_forPaper), `"fit"` (power, offset and scale of each feature fitted on the training graphs), or per feature `{scale, power, offset}` of `scale * sign(x)·|x|^power + offset` |
| `architecture`, `output` | `GNNScaleMDN` sizes; energy: `log(E / scale_eV)` |
| `training` | hyperparameters: `seeds` (one member each), `batch_size`, learning-rate `phases` `[[epochs, lr], ...]`, `weight_decay`, `patience` (`null`: no early stop), `keep` (`best` or `last` state), `max_train_events` / `max_val_events` |

What a GNN model card does not state comes from grand's `grand/analysis/gnn/defaults.json`
(`training`: from `grand_gnn_train/defaults.json`). The stored GNN model card holds every
value, plus a `provenance` block (commits, GNN model card and settings files, dataset,
best loss of each member).
Shipped: `pgnn_direction_xy.json` and `pgnn_energy_xy.json` (v1): pGNN on x, y
with node features (pos_x, pos_y, pos_z, time, amplitude) and the plane-wave
direction as graph input. For x, y, z: copy a GNN model card, set `inputs.channels` to
`["x", "y", "z"]`, give it another name.

## Everything else: settings

Where the simulations are, the machine and where to write are not part of the
network. They come from `grand_gnn_train/defaults.json`, overridden by a site
settings file (`--settings`, e.g. `settings/ccin2p3.json`: DC2 on `/sps`). The
simulations used are recorded in the saved GNN model card's provenance.

| block | what |
|---|---|
| `simulations` | directories (glob), `trace_files` / `time_files` (globs naming the level, e.g. `adc_*_L1_*.root`) and their trees, `max_du_id`, validation split, augmentation |
| `compute` | `device` (`cpu`, `cuda`, `auto`), `num_threads` |
| `paths` | `datasets_dir`, `runs_dir`, `models_dir` (default `work/…` in the current directory) |

## Steps

Environment: grand's (conda env with ROOT, torch, torch_geometric; then
`source <grand>/env/setup.sh`), and `pip install -e .` here.

`python -m grand_gnn_train --gnn-model-card <gnn_model_card.json> [--settings <settings>]` runs:

1. **dataset**: `<datasets_dir>/<channels>_<hash>/`, the hash being that of
   the GNN model card's `inputs` and of the `simulations`: what changes the graphs.
   A dataset with its `dataset_card.json` (written last) is reused, so GNN model cards
   needing the same graphs (same inputs, including `knn_k`) share one.
   The graphs hold the true shower, and each GNN model card's target is made from
   it when training.
2. **train**: the seeds of the GNN model card's `training.seeds` not yet in
   `<runs_dir>/<name>_<version>/`. A changed GNN model card (inputs, architecture or
   hyperparameters) under the same name and version is refused: give it
   another version.
3. **export**: `<models_dir>/<name>/<version>/` with grand's `save_model`
   (and the forge archive with `--tarball <dir>`).
4. **evaluate**: the network, loaded by grand, on the validation graphs:
   `<run_dir>/evaluation/validation/` holds `predictions.csv`, `summary.json`
   and the plots (resolution, comparison with the plane-wave fit, pulls and
   coverage of the predicted uncertainty, loss curves of every seed).
5. **test** (with `--test-settings <settings>`): grand's inference
   (`recons_GNN` / `recons_energy_GNN` on the peaks of `grand.analysis.signals`)
   on other simulations, in `<run_dir>/evaluation/test_<settings name>/`.

Before formatting new simulations, check what the settings select:
`python -m grand_gnn_train.simulations --settings <settings>` (files found,
channels, start times, antennas above the trigger). At CC-IN2P3, follow
`RUNBOOK_CC.md`.

`--steps` runs some of them, `--seeds` some seeds, `--max-events N` a quick
trial on N events (with its own dataset). On SLURM, everything chained
(dataset, one GPU job per seed, export and evaluation): the job scripts of
`slurm/` and their list, `bash slurm/all_jobs.sh` (see RUNBOOK_CC.md). `slurm/` holds
your paths, environment and e-mail, so it is in `.gitignore`.

To use the network in grand before uploading it, name its directory:
`gnn_models = {"direction": "<models_dir>/pgnn_direction_xy/v1", ...}` in
grand's `grand/analysis/example/config.py`, or
`gnn.load_model("direction", "<models_dir>/pgnn_direction_xy/v1")`.

To publish: attach `gnn_<name>_<version>.tar.gz` to the GRAND forge, put
`<name> <version> <attachment id>` in grand's `data/gnn_model_version.flag`,
and, to make it the default, choose it in grand's `grand/analysis/gnn/settings.json`.
`env/setup.sh` then downloads it.

## The existing ensembles (v0)

`gnn_model_cards/legacy/` describes the pGNN ensembles of the GNN codes
(GNN_forICRC `GNNScaleMDN`, applied to data in `applied_on_data 2026`), with
what that code did implicitly: six node features including `delta_t` (not
normalised there), `delta_t` with the speed `c·1.00014`, the normalisation
constants of `GNN_lib`'s `NormaliseLayer`. Only their `model.dict` files are
used (`model.mdl` needs `GNN_lib` to unpickle):

```bash
python -m grand_gnn_train.convert_legacy --gnn-model-card gnn_model_cards/legacy/pgnn_direction_xy_v0.json \
    --dicts '<models>/Direction_ADC/adding_pwf/MDN/gnn_8_knn_AN106ADC/seed_[0-4][0-9]/model.dict' \
    --settings settings/ccin2p3.json --tarball <dir>
```

Their trigger is 5σ with 15 ADC per channel (5·15·√2 = 106.07 ADC on x, y,
the `GP300_106ADC` datasets) and `knn_k` is 8. They were trained on peaks
`|H(‖x, y‖)|` (the Hilbert envelope of the norm), while grand's
`get_peak_amplitude(trace, [x, y])` now gives `sqrt(|H(x)|² + |H(y)|²)`, as the
2026 notebook did on data: the two differ by a few percent, so the v0
networks see slightly different amplitudes in grand than in training. The v1
networks, trained here, do not have this mismatch.

## Not covered by the tests here

`simulations.py` (reading DC2 directories) was written from
`make_dataset_root.py` but not run on DC2 files, which are on the cluster;
check it on one directory first (`--steps dataset --max-events 20`). The noise of
`new_rfchain` is not included: plug it in with `simulations.augment.noise`
(see `augment.py`).
