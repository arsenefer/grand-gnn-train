"""Reads GRANDroot simulation directories (DC2 layout), one event at a time.

A directory holds run_*_L0_*.root (trun: antenna positions du_xyz),
shower_*_L0_*.root (tshower: the true shower) and the trace files.  Which
trace files, and which files give the trace start times, are glob patterns
of the settings (simulations.trace_files / time_files, with their trees
trace_tree / time_tree), e.g. adc_*_L1_*.root and tree tadc: naming the
level avoids reading the same events from both L0 and L1 files.  Events are
matched across files by event_number; traces are read chunk_size events at
a time.

Adapted from make_dataset_root.py of the GNN codes.
"""

import glob
import os

import numpy as np
import uproot


def directories(patterns):
    """Simulation directories matching the glob patterns, sorted."""
    found = sorted({d.rstrip("/") for p in patterns for d in glob.glob(p) if os.path.isdir(d)})
    if not found:
        raise FileNotFoundError("no simulation directory matches %s" % (patterns,))
    return found


def _files(directory, pattern):
    files = sorted(glob.glob(os.path.join(directory, pattern)))
    if not files:
        raise FileNotFoundError("%s: no file matches %s" % (directory, pattern))
    return files


def _concat(files, tree, branches):
    out = {b: [] for b in branches}
    for path in files:
        with uproot.open(path) as f:
            arrays = f[tree].arrays(branches, library="np")
        for b in branches:
            out[b].extend(arrays[b])
    return out


def read_truth(directory, angle_unit="deg"):
    """{event_number: {zenith, azimuth (rad, "coming from"), energy_em, energy_primary (GeV), xmax_grams, primary_type}}."""
    branches = ["event_number", "zenith", "azimuth", "energy_em", "energy_primary", "xmax_grams", "primary_type"]
    data = _concat(_files(directory, "shower_*_L0_*.root"), "tshower", branches)
    factor = np.pi / 180 if angle_unit == "deg" else 1.0
    return {int(e): {"zenith": float(z) * factor, "azimuth": float(a) * factor, "energy_em": float(em),
                     "energy_primary": float(ep), "xmax_grams": float(x), "primary_type": str(p)}
            for e, z, a, em, ep, x, p in zip(*(data[b] for b in branches))}


def read_positions(directory):
    """Antenna positions du_xyz of the run, indexed by du_id, in meters."""
    with uproot.open(_files(directory, "run_*_L0_*.root")[0]) as f:
        return np.array(f["trun"]["du_xyz"].array(library="ak")[0].to_list(), dtype=float)


def read_times(directory, pattern, tree):
    """{event_number: (du_id, du_seconds, du_nanoseconds)}."""
    data = _concat(_files(directory, pattern), tree, ["event_number", "du_id", "du_seconds", "du_nanoseconds"])
    return {int(e): (np.asarray(i), np.asarray(s), np.asarray(ns))
            for e, i, s, ns in zip(data["event_number"], data["du_id"], data["du_seconds"], data["du_nanoseconds"])}


def iter_events(directory, sim):
    """Yields one dict per event: event_number, du_id, traces (N, channels, samples),
    du_seconds, du_nanoseconds, Xants (N, 3) and truth.

    `sim` is the simulations block of the settings.
    """
    truth = read_truth(directory, sim["truth_angle_unit"])
    positions = read_positions(directory)
    times = read_times(directory, sim["time_files"], sim["time_tree"])
    tree = sim["trace_tree"]
    trace_branch = "trace_ch" if tree in ("tadc", "trawvoltage") else "trace"
    for path in _files(directory, sim["trace_files"]):
        with uproot.open(path) as f:
            t = f[tree]
            for start in range(0, t.num_entries, sim["chunk_size"]):
                chunk = t.arrays(["event_number", "du_id", trace_branch], entry_start=start,
                                 entry_stop=min(start + sim["chunk_size"], t.num_entries), library="ak")
                for row in chunk:
                    event_number = int(row["event_number"])
                    if event_number not in truth or event_number not in times:
                        continue
                    du_id = np.asarray(row["du_id"], dtype=int)
                    t_ids, seconds, nanoseconds = times[event_number]
                    order = {int(i): k for k, i in enumerate(t_ids)}
                    if any(int(i) not in order for i in du_id):
                        continue
                    pick = [order[int(i)] for i in du_id]
                    yield {"event_number": event_number, "du_id": du_id,
                           "traces": np.asarray(row[trace_branch].to_list(), dtype=np.float64),
                           "du_seconds": np.asarray(seconds)[pick], "du_nanoseconds": np.asarray(nanoseconds)[pick],
                           "Xants": positions[du_id], "truth": truth[event_number]}


def check(settings_path, n_events=20):
    """Prints what the settings select, to check the simulation paths and files before formatting."""
    from grand.analysis.gnn.features import trigger
    from grand.analysis.gnn.model_store import complete_gnn_model_card

    from . import config, formatting

    defaults = config.read_json(config.DEFAULTS_FILE)
    sim = {**defaults["simulations"], **config.read_json(settings_path).get("simulations", {})}
    found = directories(sim["directories"])
    print("%d directories match %s; first: %s" % (len(found), sim["directories"], found[0]))
    for pattern in ("run_*_L0_*.root", "shower_*_L0_*.root", sim["trace_files"], sim["time_files"]):
        print("  %-22s %s" % (pattern, [os.path.basename(f) for f in _files(found[0], pattern)][:3]))
    inputs = complete_gnn_model_card({"task": "direction"})["inputs"]
    gnn_model_card = {"inputs": inputs}
    noise = []
    for i, event in enumerate(iter_events(found[0], sim)):
        if i == n_events:
            break
        amps, times = formatting.peaks(event["traces"], event["du_seconds"], event["du_nanoseconds"],
                                       gnn_model_card, sim)
        # Noise per channel: RMS of the first 100 samples of x and y, before the pulse
        xy = formatting.features.channel_indices(["x", "y"], "sim")
        noise.append(np.median(np.std(event["traces"][:, xy, :100], axis=-1)))
        print("  event %d: %d antennas, traces %s, du_seconds %d..%d, peak amplitude median %.0f ADC, "
              "%d above the 5 sigma trigger, zenith %.1f deg, E_em %.2g GeV"
              % (event["event_number"], len(event["du_id"]), event["traces"].shape[1:], event["du_seconds"].min(),
                 event["du_seconds"].max(), np.median(amps), trigger(amps, inputs).sum(),
                 np.degrees(event["truth"]["zenith"]), event["truth"]["energy_em"]))
    print("noise RMS per channel (first 100 samples of x, y): median %.1f ADC over %d events; "
          "the cards' trigger assumes %.1f ADC (inputs.trigger.noise_rms_per_channel)"
          % (np.median(noise), len(noise), inputs["trigger"]["noise_rms_per_channel"]))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Checks the simulations a settings file selects.")
    parser.add_argument("--settings", required=True)
    parser.add_argument("--events", type=int, default=20)
    args = parser.parse_args()
    check(args.settings, args.events)
