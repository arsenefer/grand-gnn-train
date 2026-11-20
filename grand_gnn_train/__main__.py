"""Makes the network of a GNN model card: its dataset (or reuses it), its seeds, its export, its evaluation.

    python -m grand_gnn_train --gnn-model-card gnn_model_cards/pgnn_direction_xy.json [--settings settings/ccin2p3.json]
        [--steps dataset train export evaluate test] [--seeds 0 1 2] [--max-events N] [--tarball DIR]
        [--test-settings settings/ccin2p3_test.json]

Steps (all but "test" by default): dataset, train, export, evaluate (plots on
the validation graphs), test (grand's inference on the simulations of
--test-settings, which adds this step).  On SLURM: slurm/submit_all.sh.
"""

import argparse

from . import config, evaluate, export, formatting, training

STEPS = ["dataset", "train", "export", "evaluate", "test"]


def run(gnn_model_card, settings=None, steps=("dataset", "train", "export", "evaluate"), seeds=None,
        max_events=None, tarball=None, test_settings=None, log=print):
    """Runs the steps for a GNN model card file; returns the run (with "model_dir" once exported)."""
    the_run = config.load(gnn_model_card, settings, max_events)
    log("%s/%s: dataset %s, seeds %s" % (the_run["gnn_model_card"]["name"], the_run["gnn_model_card"]["version"],
                                         the_run["dataset_dir"], the_run["run_dir"]))
    if "dataset" in steps:
        formatting.dataset(the_run, log)
    if "train" in steps:
        training.train(the_run, seeds, log)
    if "export" in steps:
        the_run["model_dir"] = export.export(the_run, tarball, log)
    if "evaluate" in steps:
        the_run["evaluation"] = evaluate.validation(the_run, log)
    if "test" in steps or test_settings:
        the_run["test"] = evaluate.test(the_run, config.load(gnn_model_card, test_settings, max_events), log)
    return the_run


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Makes the network of a GNN model card.")
    parser.add_argument("--gnn-model-card", required=True, help="GNN model card (gnn_model_cards/*.json)")
    parser.add_argument("--settings", help="site settings file (settings/*.json)")
    parser.add_argument("--steps", nargs="+", default=["dataset", "train", "export", "evaluate"], choices=STEPS)
    parser.add_argument("--seeds", type=int, nargs="+", help="default: the GNN model card's seeds")
    parser.add_argument("--max-events", type=int, help="format (and test on) at most this many events: a trial")
    parser.add_argument("--tarball", help="also write the forge archive in this directory")
    parser.add_argument("--test-settings", help="settings of other simulations: also test grand's inference on them")
    args = parser.parse_args()
    run(args.gnn_model_card, args.settings, args.steps, args.seeds, args.max_events, args.tarball, args.test_settings)
