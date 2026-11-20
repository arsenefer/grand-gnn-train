"""Trained seeds to a network grand loads, and its archive for the forge.

export() writes <models_dir>/<name>/<version>/ with grand's save_model.
To use it in grand before uploading it, name its directory:
gnn_models = {"direction": "<that directory>"} in examples/analysis/config.py.

To publish it: attach gnn_<name>_<version>.tar.gz to the forge, and add
"<name> <version> <attachment id>" to grand's data/gnn_model_version.flag.
"""

import glob
import json
import os
import tarfile

import torch

from grand.analysis.gnn import save_model

from . import config, training


def export(run, tarball_dir=None, log=print):
    """Saves the run's trained seeds as a network; returns its directory."""
    paths = sorted(glob.glob(os.path.join(run["run_dir"], "seed_*", "model.pt")))
    states = [torch.load(path, weights_only=True) for path in paths]
    gnn_model_card = dict(training.with_normalisation(run), provenance=config.provenance(run))
    gnn_model_card["provenance"]["members"] = []
    for path in paths:
        with open(os.path.join(os.path.dirname(path), "logs.json")) as f:
            logs = json.load(f)
        gnn_model_card["provenance"]["members"].append(
            {"seed": logs["seed"], "best_val_loss": logs["best_val_loss"], "last_val_loss": logs["last_val_loss"]})
    directory = os.path.join(run["paths"]["models_dir"], gnn_model_card["name"], gnn_model_card["version"])
    save_model(directory, gnn_model_card, states)
    log("Wrote %s (%d members)" % (directory, len(states)))
    if tarball_dir:
        tarball(directory, gnn_model_card["name"], gnn_model_card["version"], tarball_dir, log)
    return directory


def tarball(directory, name, version, out_dir, log=print):
    """Writes gnn_<name>_<version>.tar.gz holding <name>/<version>/, as data/download_gnn_models.py expects."""
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "gnn_%s_%s.tar.gz" % (name, version))
    with tarfile.open(path, "w:gz") as archive:
        archive.add(directory, arcname="%s/%s" % (name, version))
    log("Wrote %s; once on the forge, add to grand's data/gnn_model_version.flag: %s %s <attachment id>"
        % (path, name, version))
    return path
