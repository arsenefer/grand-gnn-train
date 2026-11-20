"""Simulations to graphs, with grand's own functions.

The peaks are those of grand.analysis.signals (get_peak_amplitude,
get_peak_time, compute_t0, convert_voltage_to_ADC) on the GNN model card's channels,
and the graph is grand.analysis.gnn.features.make_graph: what
grand.analysis.gnn.recons_GNN does on a measured event.

dataset() writes <dataset_dir>/{training,validation}/shard_XXXXX.pt (lists
of torch_geometric Data holding the graph and the true shower), then
dataset_card.json.  A dataset with its dataset_card.json is reused.  The
training target is computed from the true shower when training (target()),
so a direction and an energy GNN model card with the same inputs share a dataset.
"""

import datetime
import json
import os
import types

import numpy as np
import torch
from torch_geometric.data import Data

import grand.analysis.signals as sig
from grand.analysis.gnn import features

from . import augment
from . import config
from . import simulations


def target(gnn_model_card, truth):
    """Training target: the propagation vector, or log(E_em / scale_eV).

    `truth` has zenith, azimuth (rad) and energy_em (GeV).
    """
    if gnn_model_card["task"] == "direction":
        return features.propagation_vector(float(truth["zenith"]), float(truth["azimuth"]))[None, :]
    return np.array([[np.log(float(truth["energy_em"]) * 1e9 / gnn_model_card["output"]["scale_eV"])]])


def peaks(traces, du_seconds, du_nanoseconds, gnn_model_card, sim):
    """Peak amplitudes (ADC) and times (s) on the GNN model card's channels, as in grand."""
    if sim["trace_tree"] == "tvoltage":
        traces = np.array([sig.convert_voltage_to_ADC(tr, channels=range(tr.shape[0])) for tr in traces])
    t0 = sig.compute_t0(types.SimpleNamespace(du_seconds=du_seconds, du_nanoseconds=du_nanoseconds))
    channels = features.channel_indices(gnn_model_card["inputs"]["channels"], "sim")   # simulated traces
    amps = np.array([sig.get_peak_amplitude(tr, channels) for tr in traces])
    times = np.array([sig.get_peak_time(tr, t0[i], channels, dt_ns=sim["dt_ns"]) for i, tr in enumerate(traces)])
    return amps, times


def to_data(Xants, peak_time, peak_amps, gnn_model_card, **truth):
    """One event as a Data graph (make_graph, as in grand), or None if too few antennas pass the trigger."""
    graph = features.make_graph(Xants, peak_time, peak_amps, gnn_model_card["inputs"])
    if graph is None:
        return None
    data = Data(x=torch.tensor(graph["x"], dtype=torch.float32),
                edge_index=torch.tensor(graph["edge_index"], dtype=torch.long),
                x_graph=torch.tensor(graph["x_graph"], dtype=torch.float32))
    for key, value in truth.items():
        data[key] = torch.tensor([[value]], dtype=torch.float64)
    return data


def dataset(run, log=print):
    """Formats the run's simulations, unless its dataset exists; returns the dataset card."""
    out = run["dataset_dir"]
    dataset_card_file = os.path.join(out, "dataset_card.json")
    if os.path.exists(dataset_card_file):
        log("Reusing the dataset in %s" % out)
        with open(dataset_card_file) as f:
            return json.load(f)

    log("Formatting the dataset in %s" % out)
    gnn_model_card, sim = run["gnn_model_card"], run["simulations"]
    for split in ("training", "validation"):
        os.makedirs(os.path.join(out, split), exist_ok=True)
    rng = np.random.default_rng(sim["augment"]["seed"])
    noise = augment.noise_provider(sim["augment"]["noise"])
    modulo, val_rests = sim["validation_modulo"]
    counts = {"training": 0, "validation": 0, "too_few_antennas": 0}
    for d, directory in enumerate(simulations.directories(sim["directories"])):
        shards = {"training": [], "validation": []}
        for event in simulations.iter_events(directory, sim):
            keep = np.ones(len(event["du_id"]), dtype=bool)
            if sim["max_du_id"] is not None:
                keep = event["du_id"] < sim["max_du_id"]
            traces, nanoseconds = augment.apply(event["traces"][keep], event["du_nanoseconds"][keep],
                                                sim["augment"], rng, noise)
            amps, times = peaks(traces, event["du_seconds"][keep], nanoseconds, gnn_model_card, sim)
            truth = event["truth"]
            data = to_data(event["Xants"][keep], times, amps, gnn_model_card, event_number=event["event_number"],
                           zenith=truth["zenith"], azimuth=truth["azimuth"], energy_em=truth["energy_em"])
            if data is None:
                counts["too_few_antennas"] += 1
                continue
            split = "validation" if event["event_number"] % modulo in val_rests else "training"
            shards[split].append(data)
            counts[split] += 1
            if run["max_events"] and counts["training"] + counts["validation"] >= run["max_events"]:
                break
        for split, items in shards.items():
            if items:
                torch.save(items, os.path.join(out, split, "shard_%05d.pt" % d))
        log("%s: %d training, %d validation so far" % (directory, counts["training"], counts["validation"]))
        if run["max_events"] and counts["training"] + counts["validation"] >= run["max_events"]:
            break

    dataset_card = {"created": datetime.datetime.now().isoformat(timespec="seconds"), "counts": counts,
                    "inputs": gnn_model_card["inputs"], "simulations": sim, "max_events": run["max_events"],
                    "grand_commit": config.provenance(run)["grand_commit"]}
    with open(dataset_card_file, "w") as f:
        json.dump(dataset_card, f, indent=2)
    return dataset_card
