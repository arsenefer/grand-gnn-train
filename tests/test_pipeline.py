"""Formatting, training, export and conversion on synthetic plane waves, with tiny networks.

This checks that the pieces fit -- above all that a network is trained on the
graph grand builds at inference -- not that it learns well.
"""
import glob
import json
import os
import types

import numpy as np
import pytest
import torch

import grand.analysis.constants as cons
import grand.analysis.signals as sig
from grand.analysis import gnn
from grand.analysis.gnn import features
from grand.analysis.gnn.architecture import GNNScaleMDN

import grand_gnn_train.__main__ as cli
from grand_gnn_train import config, convert_legacy, formatting, training

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _gnn_model_card(tmp_path, task="direction", name="toy", version="v1", **inputs):
    """Writes a GNN model card file and a site settings file; returns their paths."""
    gnn_model_card = {"name": name, "version": version, "task": task,
            "inputs": {"trigger": {"noise_rms_per_channel": 7.0}, **inputs},     # 5 sigma = 49.5 ADC
            "architecture": {"hidden": 8, "n_graph_layers": 2, "graph_out_channels": 8},
            "normalisation": {"pos_x": {"scale": 1e-4}, "pos_y": {"scale": 1e-4}, "pos_z": {"scale": 1e-2},
                              "time": {"scale": 2e4}, "amplitude": {"power": 0.2, "scale": 0.125}},
            "training": {"seeds": [0, 1], "batch_size": 8, "phases": [[2, 1e-3]], "patience": 5}}
    gnn_model_card_path = tmp_path / ("%s_%s.json" % (name, version))
    gnn_model_card_path.write_text(json.dumps(gnn_model_card))
    settings = tmp_path / "site.json"
    settings.write_text(json.dumps({
        "simulations": {"directories": ["fake_a", "fake_b"]},
        "compute": {"device": "cpu"},
        "paths": {key: str(tmp_path / "work" / key) for key in ("datasets_dir", "runs_dir", "models_dir")}}))
    return str(gnn_model_card_path), str(settings)


def _event(rng, n=20):
    """Traces (n, 3, 512) in ADC with a pulse on x and y at the plane-wave arrival time."""
    theta, phi = np.arccos(rng.uniform(0.1, 0.6)), rng.uniform(0, 2 * np.pi)
    Xants = np.column_stack([rng.uniform(-3000, 3000, (n, 2)), 1231 + rng.uniform(0, 30, n)])
    arrival = Xants @ features.propagation_vector(theta, phi) / (cons.c_light / cons.n_atm)
    start = arrival - 400e-9 + rng.uniform(-20e-9, 20e-9, n)          # traces start ~400 ns before
    seconds = np.full(n, 1_700_000_000)
    nanoseconds = (start - start.min()) * 1e9 + 1000.0
    traces = rng.normal(0, 5, (n, 3, 512))
    index = np.rint((arrival - start) / 2e-9).astype(int)
    for i in range(n):
        traces[i, :2, index[i]:index[i] + 6] += rng.uniform(80, 800)
    truth = {"zenith": theta, "azimuth": phi, "energy_em": 10 ** rng.uniform(8, 9.5)}
    return Xants, traces, seconds, nanoseconds, truth


@pytest.fixture
def fake_simulations(monkeypatch):
    """Replaces the DC2 reader by 15 synthetic plane waves per directory."""
    def iter_events(directory, sim):
        rng = np.random.default_rng(len(directory) + directory.endswith("b"))
        for number in range(15):
            Xants, traces, seconds, nanoseconds, truth = _event(rng)
            yield {"event_number": number, "du_id": np.arange(len(Xants)), "traces": traces,
                   "du_seconds": seconds, "du_nanoseconds": nanoseconds, "Xants": Xants, "truth": truth}

    monkeypatch.setattr(formatting.simulations, "directories", list)
    monkeypatch.setattr(formatting.simulations, "iter_events", iter_events)


def test_the_training_graph_is_the_inference_graph(tmp_path):
    run = config.load(*_gnn_model_card(tmp_path))
    Xants, traces, s, ns, truth = _event(np.random.default_rng(0))
    amps, times = formatting.peaks(traces, s, ns, run["gnn_model_card"], run["simulations"])
    train_graph = formatting.to_data(Xants, times, amps, run["gnn_model_card"])
    # What grand does on the same traces
    t0 = sig.compute_t0(types.SimpleNamespace(du_seconds=s, du_nanoseconds=ns))
    model = types.SimpleNamespace(channel_indices=lambda source: features.channel_indices(["x", "y"], source))
    amps_g, times_g = gnn.peaks(traces, t0, model, source="sim")
    graph = features.make_graph(Xants, times_g, amps_g, run["gnn_model_card"]["inputs"])
    np.testing.assert_array_equal(train_graph.x.numpy(), graph["x"].astype(np.float32))
    np.testing.assert_array_equal(train_graph.edge_index.numpy(), graph["edge_index"])
    np.testing.assert_array_equal(train_graph.x_graph.numpy(), graph["x_graph"].astype(np.float32))


def test_one_command_makes_a_network_grand_uses(tmp_path, fake_simulations):
    gnn_model_card, settings = _gnn_model_card(tmp_path)
    messages = []
    run = cli.run(gnn_model_card, settings, tarball=str(tmp_path / "forge"), log=messages.append)
    assert any(m.startswith("Formatting the dataset") for m in messages)
    assert (tmp_path / "forge" / "gnn_toy_v1.tar.gz").is_file()
    # The evaluation after training: predictions, summary and plots
    evaluation = os.path.join(run["run_dir"], "evaluation", "validation")
    for name in ("predictions.csv", "summary.json", "direction_angular_error.png", "direction_resolution_vs.png",
                 "direction_reco_vs_true.png", "uncertainty_pulls.png", "training_losses.png"):
        assert os.path.isfile(os.path.join(evaluation, name)), name
    assert run["evaluation"]["n_events"] > 0 and run["evaluation"]["angular_error_68_deg"] > 0

    gnn.clear_cache()
    model = gnn.load_model("direction", run["model_dir"])
    assert model.tag == "toy/v1" and len(model.models) == 2
    assert model.gnn_model_card["inputs"] == run["gnn_model_card"]["inputs"]
    assert model.gnn_model_card["training"]["phases"] == [[2, 1e-3]]   # the hyperparameters are stored
    assert model.gnn_model_card["provenance"]["members"][0]["seed"] == 0
    Xants, traces, s, ns, _ = _event(np.random.default_rng(5))
    amps, times = formatting.peaks(traces, s, ns, run["gnn_model_card"], run["simulations"])
    result = gnn.recons_GNN(Xants, times, amps, model=model)
    assert 0 <= result.zenith <= np.pi and result.du_count == 20

    # Run again: the dataset and the seeds are reused
    messages.clear()
    cli.run(gnn_model_card, settings, steps=["dataset", "train"], log=messages.append)
    assert any(m.startswith("Reusing the dataset") for m in messages)
    assert sum("already trained" in m for m in messages) == 2

    # An energy GNN model card with the same inputs uses the same graphs
    energy = config.load(_gnn_model_card(tmp_path, task="energy", name="toy_e")[0], settings)
    assert energy["dataset_dir"] == run["dataset_dir"]
    assert training.load_split(energy["dataset_dir"], "training", energy["gnn_model_card"])[0].y.shape == (1, 1)
    assert config.load(_gnn_model_card(tmp_path, name="k4", knn_k=4)[0], settings)["dataset_dir"] != run["dataset_dir"]
    gnn.clear_cache()


def test_a_changed_gnn_model_card_needs_another_version(tmp_path, fake_simulations):
    gnn_model_card, settings = _gnn_model_card(tmp_path)
    cli.run(gnn_model_card, settings, steps=["dataset", "train"], seeds=[0], log=lambda *a: None)
    content = json.loads(open(gnn_model_card).read())
    content["training"]["batch_size"] = 2
    open(gnn_model_card, "w").write(json.dumps(content))
    with pytest.raises(ValueError, match="another GNN model card"):
        cli.run(gnn_model_card, settings, steps=["train"], seeds=[1], log=lambda *a: None)


def test_legacy_dicts_convert(tmp_path):
    gnn_model_card, settings = _gnn_model_card(tmp_path, name="legacy", version="v0",
                           node_features=["pos_x", "pos_y", "pos_z", "time", "amplitude", "delta_t"])
    run = config.load(gnn_model_card, settings)
    for seed in range(3):
        torch.manual_seed(seed)
        (tmp_path / ("seed_%02d" % seed)).mkdir()
        state = dict(GNNScaleMDN(run["gnn_model_card"]).state_dict(), dummy_param=torch.empty(0))
        torch.save(state, tmp_path / ("seed_%02d" % seed) / "model.dict")
    directory = convert_legacy.convert(run, str(tmp_path / "seed_*" / "model.dict"), log=lambda *a: None)
    model = gnn.load_model("direction", directory)
    assert len(model.models) == 3 and model.inputs["node_features"][-1] == "delta_t"
    assert "training" not in model.gnn_model_card
    gnn.clear_cache()


def test_the_shipped_gnn_model_cards(tmp_path):
    settings = os.path.join(ROOT, "settings", "ccin2p3.json")
    direction = config.load(os.path.join(ROOT, "gnn_model_cards", "pgnn_direction_xy.json"), settings)
    energy = config.load(os.path.join(ROOT, "gnn_model_cards", "pgnn_energy_xy.json"), settings)
    # The networks of the GNN paper: GNN_forPaper after_writing_selection, configs 33 and 28
    card = direction["gnn_model_card"]
    assert card["inputs"]["node_features"] == ["pos_x", "pos_y", "pos_z", "time", "amplitude", "delta_t"]
    assert card["inputs"]["knn_k"] == 32 and energy["gnn_model_card"]["inputs"]["knn_k"] == 8
    assert card["architecture"]["jk"] == "last" and energy["gnn_model_card"]["architecture"]["jk"] == "cat"
    assert card["training"]["seeds"] == list(range(20))
    assert [n for n, _ in card["training"]["phases"]] == [15, 44, 44, 15]          # 110 epochs
    assert direction["dataset_dir"] != energy["dataset_dir"]           # other neighbours, other graphs
    assert direction["simulations"]["trace_files"] == "adc_*_L1_*.root"
    for run in (direction, energy):
        assert run["gnn_model_card"]["normalisation_method"] == "gnn_lib"   # GNN_forPaper's
        assert run["gnn_model_card"]["normalisation"]["delta_t"]["scale"] == 1 / 2.6e-8
        assert run["gnn_model_card"]["training"]["keep"] == "last"          # its final weights
        assert run["gnn_model_card"]["training"]["patience"] is None
        GNNScaleMDN(run["gnn_model_card"])                             # it builds the network
    # The cat5 cards: one network for both tasks, and the graphs of the energy v1 card (8 neighbours)
    cat5 = [config.load(os.path.join(ROOT, "gnn_model_cards", name + ".json"), settings)
            for name in ("pgnn_direction_xy_cat5", "pgnn_energy_xy_cat5")]
    assert cat5[0]["gnn_model_card"]["architecture"] == cat5[1]["gnn_model_card"]["architecture"]
    assert cat5[0]["gnn_model_card"]["architecture"]["jk"] == "cat"
    assert cat5[0]["gnn_model_card"]["architecture"]["n_graph_layers"] == 5
    assert cat5[0]["dataset_dir"] == cat5[1]["dataset_dir"] == energy["dataset_dir"]
    for run in cat5:
        GNNScaleMDN(run["gnn_model_card"])
    for name in ("pgnn_direction_xy_v0", "pgnn_energy_xy_v0"):
        legacy = config.load(os.path.join(ROOT, "gnn_model_cards", "legacy", name + ".json"), settings)
        GNNScaleMDN(legacy["gnn_model_card"])


def test_the_energy_target(tmp_path):
    run = config.load(*_gnn_model_card(tmp_path, task="energy", name="toy_e"))
    assert formatting.target(run["gnn_model_card"], {"energy_em": 1e8})[0, 0] == pytest.approx(0.0)   # 1e17 eV


def test_inference_test_on_other_simulations(tmp_path, fake_simulations):
    gnn_model_card, settings = _gnn_model_card(tmp_path, task="energy", name="toy_e")
    test_settings = tmp_path / "test_site.json"
    site = json.loads(open(settings).read())
    site["simulations"] = {"directories": ["other_c"], "augment": {"seed": 1}}
    test_settings.write_text(json.dumps(site))
    run = cli.run(gnn_model_card, settings, test_settings=str(test_settings), log=lambda *a: None)
    assert run["evaluation"]["log10_energy_resolution_68"] > 0
    assert run["test"]["n_events"] == 15                                  # every event of the other directory
    tested = glob.glob(os.path.join(run["run_dir"], "evaluation", "test_*", "energy_reco_vs_true.png"))
    assert len(tested) == 1
    gnn.clear_cache()


def test_a_fitted_normalisation_is_fitted_once_and_saved(tmp_path, fake_simulations):
    gnn_model_card, settings = _gnn_model_card(tmp_path)
    content = json.loads(open(gnn_model_card).read())
    content["normalisation"] = "fit"
    open(gnn_model_card, "w").write(json.dumps(content))
    run = cli.run(gnn_model_card, settings, steps=["dataset", "train", "export"], log=lambda *a: None)
    fitted = json.loads(open(os.path.join(run["run_dir"], "normalisation.json")).read())
    gnn.clear_cache()
    model = gnn.load_model("direction", run["model_dir"])
    assert model.gnn_model_card["normalisation_method"] == "fit"
    assert model.gnn_model_card["normalisation"] == fitted                # the constants the seeds were trained with
    assert set(fitted) == set(model.inputs["node_features"])
    gnn.clear_cache()


def test_inspect_a_dataset_and_refuse_to_train_without_one(tmp_path, fake_simulations, capsys):
    from grand_gnn_train import inspect_dataset

    gnn_model_card, settings = _gnn_model_card(tmp_path)
    run = config.load(gnn_model_card, settings)
    with pytest.raises(FileNotFoundError, match="run --steps dataset first"):
        training.train(run, log=lambda message: None)
    cli.run(gnn_model_card, settings, steps=["dataset"], log=lambda message: None)
    inspect_dataset.inspect(gnn_model_card, settings)
    printed = capsys.readouterr().out
    assert "amplitude" in printed and "target" in printed and "antennas per graph" in printed


def test_graphs_with_nan_features_are_not_trained_on(tmp_path, fake_simulations):
    gnn_model_card, settings = _gnn_model_card(tmp_path)
    run = config.load(gnn_model_card, settings)
    cli.run(gnn_model_card, settings, steps=["dataset"], log=lambda message: None)
    graphs = training.load_split(run["dataset_dir"], "training", run["gnn_model_card"])
    bad = graphs[0].clone()
    bad.x_graph[:] = float("nan")                                       # a failed plane-wave fit
    torch.save([bad], os.path.join(run["dataset_dir"], "training", "shard_99999.pt"))
    assert len(training.load_split(run["dataset_dir"], "training", run["gnn_model_card"])) == len(graphs)
