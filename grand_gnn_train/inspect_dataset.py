"""Prints what a GNN model card's dataset holds: counts, and each feature's range before and after
the normalisation, to find what makes the loss NaN.

    python -m grand_gnn_train.inspect_dataset --gnn-model-card <card.json> --settings <settings.json>
"""

import argparse
import json
import os

import numpy as np
import torch

from . import config, training


def inspect(gnn_model_card_path, settings_path, max_events=2000):
    run = config.load(gnn_model_card_path, settings_path)
    gnn_model_card = run["gnn_model_card"]
    print("dataset", run["dataset_dir"])
    card_file = os.path.join(run["dataset_dir"], "dataset_card.json")
    if os.path.exists(card_file):
        with open(card_file) as f:
            print("counts", json.load(f)["counts"])
    graphs = training.load_split(run["dataset_dir"], "training", gnn_model_card, max_events)
    print("%d training graphs read" % len(graphs))
    if not graphs:
        return

    x = torch.cat([g.x for g in graphs]).double().numpy()
    x_graph = torch.cat([g.x_graph for g in graphs]).double().numpy()
    y = torch.cat([g.y for g in graphs]).double().numpy()
    names = gnn_model_card["inputs"]["node_features"]
    norm = training.with_normalisation(run, graphs)["normalisation"]
    print("%-12s %10s %12s %12s %12s   normalised: %10s %10s %10s" % ("feature", "non-finite", "min", "median", "max",
                                                                     "min", "median", "max"))
    for i, name in enumerate(names):
        v = x[:, i]
        n = norm[name]
        w = n["scale"] * np.sign(v) * np.abs(v) ** n["power"] + n["offset"]
        print("%-12s %10d %12.4g %12.4g %12.4g               %10.3g %10.3g %10.3g"
              % (name, (~np.isfinite(v)).sum(), np.nanmin(v), np.nanmedian(v), np.nanmax(v),
                 np.nanmin(w), np.nanmedian(w), np.nanmax(w)))
    for i, name in enumerate(gnn_model_card["inputs"]["graph_features"]):
        v = x_graph[:, i]
        print("%-12s %10d %12.4g %12.4g %12.4g" % (name, (~np.isfinite(v)).sum(), np.nanmin(v), np.nanmedian(v), np.nanmax(v)))
    print("%-12s %10d %12.4g %12.4g %12.4g" % ("target", (~np.isfinite(y)).sum(), np.nanmin(y), np.nanmedian(y), np.nanmax(y)))
    print("antennas per graph: min %d, median %d, max %d"
          % tuple(np.percentile([g.num_nodes for g in graphs], [0, 50, 100])))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--gnn-model-card", required=True)
    parser.add_argument("--settings")
    parser.add_argument("--max-events", type=int, default=2000)
    args = parser.parse_args()
    inspect(args.gnn_model_card, args.settings, args.max_events)
