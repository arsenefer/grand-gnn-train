"""Performance of a trained network: predictions, plots and a summary.

The network is loaded as grand loads it (grand.analysis.gnn.load_model on
its exported directory), so what is evaluated is what grand will run.

* validation(run): on the validation graphs of the run's dataset, right
  after training (step "evaluate").  Writes <run_dir>/evaluation/validation/.
* test(run, test_run): an inference test on other simulations.  Each event
  goes through grand's inference as a measured event would: peaks with
  grand.analysis.signals, then gnn.recons_GNN / recons_energy_GNN.  Writes
  <run_dir>/evaluation/test_<test settings file name>/.

Each writes predictions.csv (one row per event: truth, GNN, plane-wave fit),
summary.json and the plots:

* direction: angular error of the GNN and of the plane-wave fit; its 68 %
  quantile vs zenith, energy and number of antennas; pulls of zenith and
  azimuth and the coverage of the predicted uncertainty; reconstructed vs
  true zenith;
* energy: log10(E_reco / E_true), its bias and spread vs energy and zenith;
  reconstructed vs true energy; pull and coverage;
* training: loss curves of every seed (validation only).

    python -m grand_gnn_train.evaluate --gnn-model-card C --settings S [--test-settings T] [--max-events N]
"""

import argparse
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.special import erf  # noqa: E402

from grand.analysis import gnn  # noqa: E402
from grand.analysis.fitting import PWF_semianalytical  # noqa: E402
from grand.analysis.gnn import features  # noqa: E402

from . import augment, config, formatting, simulations, training  # noqa: E402

DEG = 180 / np.pi


def load_network(run):
    """The run's exported network, loaded by grand."""
    card = run["gnn_model_card"]
    return gnn.load_model(card["task"], os.path.join(run["paths"]["models_dir"], card["name"], card["version"]))


# ---------------------------------------------------------------- predictions

def _source(zenith, azimuth):
    """Unit vector towards where the shower comes from."""
    return np.stack([np.sin(zenith) * np.cos(azimuth), np.sin(zenith) * np.sin(azimuth), np.cos(zenith)], axis=-1)


def _row(model, value, variance, truth, du_count, k_pwf=None):
    """One row of predictions.csv."""
    row = {"zenith_true": truth["zenith"], "azimuth_true": truth["azimuth"],
           "energy_em_true": truth["energy_em"] * 1e9, "du_count": du_count}
    if model.task == "direction":
        row.update(zip(["zenith", "azimuth", "std_zenith", "std_azimuth"], gnn.direction_from_k(value, variance)))
    else:
        row.update(zip(["energy", "std_energy"],
                       gnn.energy_from_log(value[:, 0], variance[:, 0], model.gnn_model_card["output"]["scale_eV"])))
    if k_pwf is not None:
        source = -np.asarray(k_pwf) / np.linalg.norm(k_pwf)
        row["zenith_pwf"] = np.arccos(source[2])
        row["azimuth_pwf"] = np.arctan2(source[1], source[0]) % (2 * np.pi)
    return row


def predict_graphs(model, graphs, batch=512):
    """Predictions on formatted graphs (Data objects of the dataset)."""
    rows = []
    with_pwf = model.inputs["graph_features"][:3] == ["k_pwf_x", "k_pwf_y", "k_pwf_z"]
    for start in range(0, len(graphs), batch):
        chunk = graphs[start:start + batch]
        values, variances = model.predict([{"x": d.x.numpy(), "edge_index": d.edge_index.numpy(),
                                            "x_graph": d.x_graph.numpy()} for d in chunk])
        for i, d in enumerate(chunk):
            truth = {key: float(d[key]) for key in ("zenith", "azimuth", "energy_em")}
            row = _row(model, values[:, i], variances[:, i], truth, d.x.shape[0],
                       d.x_graph[0, :3].numpy() if with_pwf else None)
            rows.append(dict(row, event_number=int(d.event_number)))
    return pd.DataFrame(rows)


def infer_simulations(model, test_run, log=print):
    """Grand's inference on every event of the test simulations (as on measured data)."""
    sim = test_run["simulations"]
    rng = np.random.default_rng(sim["augment"]["seed"])
    noise = augment.noise_provider(sim["augment"]["noise"])
    recons = gnn.recons_GNN if model.task == "direction" else gnn.recons_energy_GNN
    rows = []
    for directory in simulations.directories(sim["directories"]):
        for event in simulations.iter_events(directory, sim):
            keep = np.ones(len(event["du_id"]), dtype=bool)
            if sim["max_du_id"] is not None:
                keep = event["du_id"] < sim["max_du_id"]
            traces, nanoseconds = augment.apply(event["traces"][keep], event["du_nanoseconds"][keep],
                                                sim["augment"], rng, noise)
            amps, times = formatting.peaks(traces, event["du_seconds"][keep], nanoseconds,
                                           model.gnn_model_card, sim)
            Xants = event["Xants"][keep]
            result = recons(Xants, times, amps, model=model)
            if result.du_count < model.inputs["trigger"]["min_antennas"]:
                continue
            row = {"event_number": event["event_number"], "directory": os.path.basename(directory),
                   "zenith_true": event["truth"]["zenith"], "azimuth_true": event["truth"]["azimuth"],
                   "energy_em_true": event["truth"]["energy_em"] * 1e9, **result._asdict()}
            # The plane-wave fit on the same antennas, for comparison
            used = features.trigger(amps, model.inputs)
            try:
                row["zenith_pwf"], row["azimuth_pwf"] = PWF_semianalytical(Xants[used], times[used])
            except ValueError:  # e.g. antennas on one line
                row["zenith_pwf"], row["azimuth_pwf"] = np.nan, np.nan
            rows.append(row)
            if test_run["max_events"] and len(rows) >= test_run["max_events"]:
                return pd.DataFrame(rows)
        log("%s: %d events so far" % (directory, len(rows)))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------- metrics

def angular_error(df, suffix=""):
    """Angle (deg) between the true and the reconstructed directions."""
    true = _source(df["zenith_true"].to_numpy(), df["azimuth_true"].to_numpy())
    reco = _source(df["zenith" + suffix].to_numpy(), df["azimuth" + suffix].to_numpy())
    return np.arccos(np.clip(np.sum(true * reco, axis=1), -1, 1)) * DEG


def _wrap(angle):
    return (angle + np.pi) % (2 * np.pi) - np.pi


def pulls(df):
    """Residual / predicted standard deviation, per event."""
    if "zenith" in df:
        return {"zenith": (df["zenith"] - df["zenith_true"]) / df["std_zenith"],
                "azimuth": _wrap(df["azimuth"] - df["azimuth_true"]) / df["std_azimuth"]}
    return {"log_energy": np.log(df["energy"] / df["energy_em_true"]) / (df["std_energy"] / df["energy"])}


def _quantile_in_bins(x, y, bins, q=68):
    centers, values, counts = [], [], []
    for low, high in zip(bins[:-1], bins[1:]):
        inside = (x >= low) & (x < high)
        if inside.sum() >= 10:
            centers.append(0.5 * (low + high))
            values.append(np.percentile(y[inside], q))
            counts.append(int(inside.sum()))
    return np.array(centers), np.array(values), counts


def summary(df):
    """Numbers to compare networks: resolution, bias, calibration of the uncertainty."""
    out = {"n_events": len(df)}
    for name, pull in pulls(df).items():
        pull = pull[np.isfinite(pull)]
        out["pull_%s_mean" % name] = float(np.mean(pull))
        out["pull_%s_std" % name] = float(np.std(pull))
        out["coverage_%s_1sigma" % name] = float(np.mean(np.abs(pull) < 1))
    if "zenith" in df:
        out["angular_error_68_deg"] = float(np.percentile(angular_error(df), 68))
        out["angular_error_50_deg"] = float(np.percentile(angular_error(df), 50))
        if "zenith_pwf" in df:
            out["angular_error_68_deg_pwf"] = float(np.percentile(angular_error(df, "_pwf"), 68))
    else:
        r = np.log10(df["energy"] / df["energy_em_true"])
        out["log10_energy_bias"] = float(np.median(r))
        out["log10_energy_resolution_68"] = float(0.5 * (np.percentile(r, 84) - np.percentile(r, 16)))
        out["energy_relative_resolution_68"] = float(0.5 * (np.percentile(10**r, 84) - np.percentile(10**r, 16)))
    return out


# ---------------------------------------------------------------- plots

def _save(fig, out_dir, name):
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, name), dpi=130)
    plt.close(fig)


def plot_pulls(df, out_dir):
    """Pull histograms against N(0, 1), and the coverage of the predicted uncertainty."""
    all_pulls = pulls(df)
    fig, axes = plt.subplots(1, len(all_pulls) + 1, figsize=(5 * (len(all_pulls) + 1), 4))
    x = np.linspace(-5, 5, 200)
    for ax, (name, pull) in zip(axes, all_pulls.items()):
        pull = pull[np.isfinite(pull)]
        ax.hist(np.clip(pull, -5, 5), bins=60, range=(-5, 5), density=True, alpha=0.6,
                label="mean %.2f, std %.2f" % (np.mean(pull), np.std(pull)))
        ax.plot(x, np.exp(-x**2 / 2) / np.sqrt(2 * np.pi), "k", label="N(0, 1)")
        ax.set_xlabel("pull of %s (reco - true) / sigma" % name)
        ax.legend()
    k = np.linspace(0, 3, 61)
    axes[-1].plot(k, erf(k / np.sqrt(2)), "k", label="expected (Gaussian)")
    for name, pull in all_pulls.items():
        pull = np.abs(pull[np.isfinite(pull)])
        axes[-1].plot(k, [np.mean(pull < kk) for kk in k], label=name)
    axes[-1].set_xlabel("k (number of predicted sigma)")
    axes[-1].set_ylabel("fraction of events within k sigma")
    axes[-1].legend()
    _save(fig, out_dir, "uncertainty_pulls.png")


def plot_direction(df, out_dir):
    error = angular_error(df)
    has_pwf = "zenith_pwf" in df
    error_pwf = angular_error(df, "_pwf") if has_pwf else None

    fig, ax = plt.subplots(figsize=(6, 4))
    bins = np.logspace(-2, 1.5, 60)
    ax.hist(error, bins=bins, histtype="step", lw=2, label="GNN: 68%% < %.3f deg" % np.percentile(error, 68))
    if has_pwf:
        ax.hist(error_pwf, bins=bins, histtype="step", lw=2,
                label="plane wave: 68%% < %.3f deg" % np.percentile(error_pwf, 68))
    ax.set_xscale("log")
    ax.set_xlabel("angular error (deg)")
    ax.set_ylabel("events")
    ax.legend()
    _save(fig, out_dir, "direction_angular_error.png")

    zenith = df["zenith_true"].to_numpy() * DEG
    log_energy = np.log10(df["energy_em_true"].to_numpy())
    du_count = df["du_count"].to_numpy()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, x, bins, label in ((axes[0], zenith, np.arange(0, 91, 5), "true zenith (deg)"),
                               (axes[1], log_energy, np.arange(16, 20.01, 0.25), "log10(true E_em / eV)"),
                               (axes[2], du_count, np.unique(np.percentile(du_count, np.linspace(0, 100, 11))),
                                "antennas used")):
        centers, values, _ = _quantile_in_bins(x, error, bins)
        ax.plot(centers, values, "o-", label="GNN")
        if has_pwf:
            centers, values, _ = _quantile_in_bins(x, error_pwf, bins)
            ax.plot(centers, values, "s--", label="plane wave")
        ax.set_xlabel(label)
        ax.set_ylabel("68% angular error (deg)")
        ax.set_yscale("log")
        ax.legend()
    _save(fig, out_dir, "direction_resolution_vs.png")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].hist2d(zenith, df["zenith"] * DEG, bins=90, range=[[0, 90], [0, 90]], cmin=1)
    axes[0].plot([0, 90], [0, 90], "r--", lw=1)
    axes[0].set_xlabel("true zenith (deg)")
    axes[0].set_ylabel("GNN zenith (deg)")
    axes[1].hist(_wrap(df["azimuth"] - df["azimuth_true"]) * DEG * np.sin(df["zenith_true"]), bins=100,
                 range=(-1, 1))
    axes[1].set_xlabel("(azimuth reco - true) x sin(zenith) (deg)")
    _save(fig, out_dir, "direction_reco_vs_true.png")
    plot_pulls(df, out_dir)


def plot_energy(df, out_dir):
    r = np.log10(df["energy"] / df["energy_em_true"]).to_numpy()
    log_true = np.log10(df["energy_em_true"].to_numpy())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(r, bins=100, range=(-0.5, 0.5))
    axes[0].set_xlabel("log10(E_reco / E_true)")
    axes[0].set_title("median %.3f, 68%% half-width %.3f"
                      % (np.median(r), 0.5 * (np.percentile(r, 84) - np.percentile(r, 16))))
    axes[1].hist2d(log_true, np.log10(df["energy"]), bins=80, cmin=1)
    lims = [log_true.min(), log_true.max()]
    axes[1].plot(lims, lims, "r--", lw=1)
    axes[1].set_xlabel("log10(true E_em / eV)")
    axes[1].set_ylabel("log10(GNN E_em / eV)")
    _save(fig, out_dir, "energy_reco_vs_true.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for ax, x, bins, label in ((axes[0], log_true, np.arange(16, 20.01, 0.25), "log10(true E_em / eV)"),
                               (axes[1], df["zenith_true"].to_numpy() * DEG, np.arange(0, 91, 5),
                                "true zenith (deg)")):
        centers, median, _ = _quantile_in_bins(x, r, bins, 50)
        _, high, _ = _quantile_in_bins(x, r, bins, 84)
        _, low, _ = _quantile_in_bins(x, r, bins, 16)
        ax.plot(centers, median, "o-", label="bias (median)")
        ax.plot(centers, 0.5 * (high - low), "s--", label="resolution (68% half-width)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel(label)
        ax.set_ylabel("log10(E_reco / E_true)")
        ax.legend()
    _save(fig, out_dir, "energy_resolution_vs.png")
    plot_pulls(df, out_dir)


def plot_losses(run_dir, out_dir):
    """Training and validation loss of every seed."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for path in sorted(glob.glob(os.path.join(run_dir, "seed_*", "logs.json"))):
        with open(path) as f:
            logs = json.load(f)
        axes[0].plot(logs["train_loss"], lw=1, label="seed %d" % logs["seed"])
        axes[1].plot(logs["val_loss"], lw=1)
    axes[0].set_title("training loss")
    axes[1].set_title("validation loss")
    for ax in axes:
        ax.set_xlabel("epoch")
    axes[0].set_ylabel("Gaussian NLL")
    axes[0].legend(fontsize=6, ncol=2)
    _save(fig, out_dir, "training_losses.png")


def report(df, model, out_dir, log=print):
    """Writes predictions.csv, summary.json and the plots of `df`."""
    os.makedirs(out_dir, exist_ok=True)
    df.to_csv(os.path.join(out_dir, "predictions.csv"), index=False)
    numbers = dict(summary(df), network=model.tag)
    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(numbers, f, indent=2)
    if model.task == "direction":
        plot_direction(df, out_dir)
    else:
        plot_energy(df, out_dir)
    log("Evaluation of %s in %s: %s" % (model.tag, out_dir, json.dumps(numbers)))
    return numbers


# ---------------------------------------------------------------- entry points

def validation(run, log=print):
    """Evaluates the exported network on the validation graphs of its dataset."""
    model = load_network(run)
    graphs = training.load_split(run["dataset_dir"], "validation", run["gnn_model_card"])
    out_dir = os.path.join(run["run_dir"], "evaluation", "validation")
    numbers = report(predict_graphs(model, graphs), model, out_dir, log)
    plot_losses(run["run_dir"], out_dir)
    return numbers


def test(run, test_run, log=print):
    """Evaluates the exported network, through grand's inference, on the simulations of `test_run`."""
    model = load_network(run)
    name = os.path.splitext(os.path.basename(test_run["settings_path"] or "default"))[0]
    df = infer_simulations(model, test_run, log)
    return report(df, model, os.path.join(run["run_dir"], "evaluation", "test_" + name), log)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluates a trained network and plots its performance.")
    parser.add_argument("--gnn-model-card", required=True)
    parser.add_argument("--settings", help="site settings of the training (where the network is)")
    parser.add_argument("--test-settings", help="settings naming other simulations: run the inference test on them")
    parser.add_argument("--max-events", type=int, help="test on at most this many events")
    args = parser.parse_args()
    the_run = config.load(args.gnn_model_card, args.settings)
    if args.test_settings:
        test(the_run, config.load(args.gnn_model_card, args.test_settings, args.max_events))
    else:
        validation(the_run)
