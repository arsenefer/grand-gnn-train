"""A GNN model card, and what is needed to train it.

The GNN model card (gnn_model_cards/*.json) is the network: grand's blocks (task, inputs,
normalisation, architecture, output) and its training hyperparameters
(training).  What it does not state comes from grand's defaults.json
(training: from defaults.json here).

Everything else comes from defaults.json here, then the site settings file
(settings/*.json): where the simulations are (simulations), the machine
(compute) and where to write (paths).

load() puts them together in a "run" dictionary.
"""

import hashlib
import json
import os
import subprocess

from grand.analysis.gnn.model_store import complete_gnn_model_card, read_json

DEFAULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "defaults.json")


def load(gnn_model_card_path, settings_path=None, max_events=None):
    """The run of a GNN model card: the complete GNN model card, the settings, and its directories.

    The dataset directory is named after what changes the graphs (the GNN model card's
    inputs and the simulations), so GNN model cards needing the same graphs share it.
    """
    defaults = read_json(DEFAULTS_FILE)
    site = read_json(settings_path) if settings_path else {}

    gnn_model_card = complete_gnn_model_card(read_json(gnn_model_card_path))
    gnn_model_card["training"] = {**defaults["training"], **gnn_model_card.get("training", {})}

    simulations = {**defaults["simulations"], **site.get("simulations", {})}
    simulations["augment"] = {**defaults["simulations"]["augment"], **site.get("simulations", {}).get("augment", {})}
    compute = {**defaults["compute"], **site.get("compute", {})}
    paths = {key: os.path.abspath(path) for key, path in {**defaults["paths"], **site.get("paths", {})}.items()}
    if max_events:  # a trial: its seeds and network are kept apart from the real ones
        paths["runs_dir"] = os.path.join(paths["runs_dir"], "trials_%d_events" % max_events)
        paths["models_dir"] = os.path.join(paths["models_dir"], "trials_%d_events" % max_events)

    key = json.dumps([gnn_model_card["inputs"], simulations, max_events], sort_keys=True)
    key = hashlib.sha256(key.encode()).hexdigest()[:12]
    return {
        "gnn_model_card": gnn_model_card,
        "gnn_model_card_path": os.path.abspath(gnn_model_card_path),
        "settings_path": settings_path,
        "simulations": simulations,
        "compute": compute,
        "paths": paths,
        "max_events": max_events,
        "dataset_dir": os.path.join(paths["datasets_dir"], "".join(gnn_model_card["inputs"]["channels"]) + "_" + key),
        "run_dir": os.path.join(paths["runs_dir"], gnn_model_card["name"] + "_" + gnn_model_card["version"]),
    }


def git_commit(directory):
    """Git commit of the repository holding `directory` ("-dirty" if modified), or None."""
    try:
        commit = subprocess.run(["git", "-C", directory, "rev-parse", "--short", "HEAD"],
                                capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", directory, "status", "--porcelain", "--untracked-files=no"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return commit + ("-dirty" if dirty else "")
    except (OSError, subprocess.SubprocessError):
        return None


def provenance(run):
    """What a saved network records about how it was made."""
    import grand

    return {"gnn_model_card_file": os.path.basename(run["gnn_model_card_path"]),
            "settings_file": run["settings_path"] and os.path.basename(run["settings_path"]),
            "simulations": run["simulations"],
            "grand_gnn_train_commit": git_commit(os.path.dirname(os.path.abspath(__file__))),
            "grand_commit": git_commit(os.path.dirname(os.path.abspath(grand.__file__)))}
