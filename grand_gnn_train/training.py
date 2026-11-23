"""Training of the ensemble members, one seed at a time.

The network is grand's GNNScaleMDN, trained with a Gaussian negative
log-likelihood on (value, variance), with the hyperparameters of the GNN model card's
training block (learning-rate phases as in the GNN codes' train_model.py).
The best validation state of each seed is kept, in
<run_dir>/seed_XX/{model.pt, logs.json}.
"""

import glob
import json
import os

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from grand.analysis.gnn import features
from grand.analysis.gnn.architecture import GNNScaleMDN

from .formatting import target


def load_split(dataset_dir, split, gnn_model_card, max_events=None):
    """Graphs of one split ("training" or "validation"), with the GNN model card's target as y."""
    items = []
    for path in sorted(glob.glob(os.path.join(dataset_dir, split, "shard_*.pt"))):
        items.extend(torch.load(path, weights_only=False))
    items = items[:max_events]
    for data in items:
        data.y = torch.tensor(target(gnn_model_card, data), dtype=torch.float32)
    return items


def evaluate(model, loader, loss_fn, device):
    """Mean loss of `model` over `loader`."""
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            value, variance = model(batch.x, batch.edge_index, batch.batch, batch.x_graph)
            total += loss_fn(value, batch.y, variance).item() * batch.num_graphs
            count += batch.num_graphs
    return total / count


def train_member(gnn_model_card, train_set, val_set, seed, out_dir, device="cpu", log=print):
    """Trains one member; saves its state and its losses in `out_dir`.

    The training block chooses the state kept: "keep": "best" (lowest validation
    loss) or "last" (the end of the last phase, as GNN_forPaper did); and
    "patience": stop after that many epochs without improvement, or null to
    run every epoch of the phases.
    """
    training = gnn_model_card["training"]
    torch.manual_seed(seed)
    model = GNNScaleMDN(gnn_model_card).to(device)
    loss_fn = torch.nn.GaussianNLLLoss()   # eps 1e-6, as GNN_lib's GaussianNLLLoss
    train_loader = DataLoader(train_set, batch_size=training["batch_size"], shuffle=True,
                              generator=torch.Generator().manual_seed(seed))
    val_loader = DataLoader(val_set, batch_size=64)

    best_loss, best_state, stale = np.inf, None, 0
    logs = {"seed": seed, "train_loss": [], "val_loss": []}
    for n_epochs, lr in training["phases"]:
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=training["weight_decay"])
        for epoch in range(n_epochs):
            model.train()
            total, count = 0.0, 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                value, variance = model(batch.x, batch.edge_index, batch.batch, batch.x_graph)
                loss = loss_fn(value, batch.y, variance)
                loss.backward()
                optimizer.step()
                total += loss.item() * batch.num_graphs
                count += batch.num_graphs
            val_loss = evaluate(model, val_loader, loss_fn, device)
            logs["train_loss"].append(total / count)
            logs["val_loss"].append(val_loss)
            log("seed %d lr %.1e: train %.4f val %.4f" % (seed, lr, total / count, val_loss))
            if val_loss < best_loss:
                best_loss, stale = val_loss, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if training["patience"] is not None and stale >= training["patience"]:
                break
        if training["patience"] is not None and stale >= training["patience"]:
            break

    logs["best_val_loss"] = best_loss
    logs["last_val_loss"] = logs["val_loss"][-1]
    if training["keep"] == "last":
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    os.makedirs(out_dir, exist_ok=True)
    torch.save(best_state, os.path.join(out_dir, "model.pt"))
    with open(os.path.join(out_dir, "logs.json"), "w") as f:
        json.dump(logs, f)


def with_normalisation(run, train_set=None):
    """The run's GNN model card, with its normalization constants if it says "fit".

    The constants are fitted once on the training graphs (grand's
    features.fit_normalisation) and kept in <run_dir>/normalisation.json, so
    that every seed, and the export, use the same ones.
    """
    gnn_model_card = run["gnn_model_card"]
    if gnn_model_card["normalisation"] != "fit":
        return gnn_model_card
    path = os.path.join(run["run_dir"], "normalisation.json")
    if not os.path.exists(path):
        x = np.concatenate([data.x.numpy() for data in train_set])
        fitted = features.fit_normalisation(x, gnn_model_card["inputs"]["node_features"])
        temporary = "%s.%d" % (path, os.getpid())   # seeds may run at the same time: write, then rename
        with open(temporary, "w") as f:
            json.dump(fitted, f, indent=2)
        os.replace(temporary, path)
    with open(path) as f:
        return dict(gnn_model_card, normalisation=json.load(f))


def train(run, seeds=None, log=print):
    """Trains the seeds of the GNN model card (or `seeds`) that are not trained yet."""
    gnn_model_card, compute = run["gnn_model_card"], run["compute"]
    os.makedirs(run["run_dir"], exist_ok=True)
    # The seeds of one directory must come from the same GNN model card
    gnn_model_card_file = os.path.join(run["run_dir"], "gnn_model_card.json")
    if os.path.exists(gnn_model_card_file):
        with open(gnn_model_card_file) as f:
            if json.load(f) != json.loads(json.dumps(gnn_model_card)):
                raise ValueError("%s holds seeds of another GNN model card: change its version" % run["run_dir"])
    else:
        with open(gnn_model_card_file, "w") as f:
            json.dump(gnn_model_card, f, indent=2)

    if compute["num_threads"]:
        torch.set_num_threads(compute["num_threads"])
    device = compute["device"]
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    training = gnn_model_card["training"]
    train_set = load_split(run["dataset_dir"], "training", gnn_model_card, training["max_train_events"])
    val_set = load_split(run["dataset_dir"], "validation", gnn_model_card, training["max_val_events"])
    log("%d training and %d validation events" % (len(train_set), len(val_set)))
    if not train_set:
        raise FileNotFoundError("no graphs in %s: run --steps dataset first" % run["dataset_dir"])
    gnn_model_card = with_normalisation(run, train_set)
    for seed in seeds or gnn_model_card["training"]["seeds"]:
        out_dir = os.path.join(run["run_dir"], "seed_%02d" % seed)
        if os.path.exists(os.path.join(out_dir, "model.pt")):
            log("seed %d: already trained" % seed)
        else:
            train_member(gnn_model_card, train_set, val_set, seed, out_dir, device, log)
