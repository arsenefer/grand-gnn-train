"""Detector effects added to simulated traces before the peaks are taken.

* gain smearing: each antenna's trace times ``N(1, gain_smearing)``;
* time jitter: each trace start time plus ``N(0, time_jitter_ns)`` ns;
* noise: an optional provider, ``"module:function"``, called as
  ``function(n_antennas, n_channels, n_samples, rng, **kwargs)`` and
  returning noise in the unit of the trace tree (µV for ``tvoltage``, ADC
  counts for ``tadc``).  The GNN codes used the galactic noise of
  ``new_rfchain`` (``noise.compute_noise``); wrap it in such a function and
  put its directory on ``PYTHONPATH``.

As in the GNN codes (``make_dataset_h5.py``): noise, then gain, then
conversion to ADC and peak extraction.
"""

import importlib

import numpy as np


def noise_provider(spec):
    r"""Returns the noise function of ``{"provider": "module:function", "kwargs": {...}}``, or None."""
    if not spec:
        return None
    module, _, name = spec["provider"].partition(":")
    function = getattr(importlib.import_module(module), name)
    kwargs = dict(spec.get("kwargs", {}))
    return lambda n, c, s, rng: function(n, c, s, rng, **kwargs)


def apply(traces, du_nanoseconds, augment, rng, noise=None):
    r"""Returns augmented copies of `traces` (N, channels, samples) and `du_nanoseconds` (N,)."""
    traces = np.array(traces, dtype=np.float64)
    if noise is not None:
        traces = traces + noise(*traces.shape, rng)
    if augment.get("gain_smearing"):
        traces *= rng.normal(1.0, augment["gain_smearing"], len(traces))[:, None, None]
    nanoseconds = np.asarray(du_nanoseconds, dtype=np.float64)
    if augment.get("time_jitter_ns"):
        nanoseconds = nanoseconds + rng.normal(0.0, augment["time_jitter_ns"], len(nanoseconds))
    return traces, nanoseconds
