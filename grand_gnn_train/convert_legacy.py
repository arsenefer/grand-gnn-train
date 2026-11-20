"""The ensembles of the GNN codes (GNN_forICRC / GNN_forPaper) as grand networks.

Those codes saved each seed as model.mdl (a pickle needing GNN_lib) and
model.dict (its state_dict); the model.dict files are used.  The legacy
GNN model cards (gnn_model_cards/legacy/*.json) state what that code did implicitly.

    python -m grand_gnn_train.convert_legacy --gnn-model-card gnn_model_cards/legacy/pgnn_direction_xy_v0.json \\
        --dicts '<models>/Direction_ADC/adding_pwf/MDN/gnn_8_knn_AN106ADC/seed_[0-4][0-9]/model.dict' \\
        [--settings settings/ccin2p3.json] [--tarball <dir>]
"""

import argparse
import glob
import os

import torch

from grand.analysis.gnn import save_model

from . import config, export


def convert(run, dict_pattern, tarball_dir=None, log=print):
    """Saves the model.dict files matching `dict_pattern` as a network; returns its directory."""
    paths = sorted(glob.glob(dict_pattern))
    states = []
    for path in paths:
        state = torch.load(path, map_location="cpu", weights_only=True)
        state.pop("dummy_param", None)  # registered by GNN_lib's base class only
        states.append(state)
    gnn_model_card = dict(run["gnn_model_card"], provenance={**config.provenance(run), "converted_from": paths})
    gnn_model_card.pop("training")  # trained by the GNN codes
    gnn_model_card["provenance"].pop("simulations")
    directory = os.path.join(run["paths"]["models_dir"], gnn_model_card["name"], gnn_model_card["version"])
    save_model(directory, gnn_model_card, states)
    log("Converted %d members to %s" % (len(states), directory))
    if tarball_dir:
        export.tarball(directory, gnn_model_card["name"], gnn_model_card["version"], tarball_dir, log)
    return directory


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Converts an ensemble of the GNN codes.")
    parser.add_argument("--gnn-model-card", required=True)
    parser.add_argument("--dicts", required=True, help="glob of the model.dict files")
    parser.add_argument("--settings", help="site settings file (paths.models_dir)")
    parser.add_argument("--tarball", help="also write the forge archive in this directory")
    args = parser.parse_args()
    convert(config.load(args.gnn_model_card, args.settings), args.dicts, args.tarball)
