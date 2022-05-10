from graphnet.plots.width_plot import width_plot
import numpy as np


predictions_path = "/groups/hep/pcs557/phd/results/dev_lvl7_robustness_muon_neutrino_0000/dynedge_zenith_9_test_set/results.csv"
database = "/groups/hep/pcs557/GNNReco/data/databases/dev_lvl7_robustness_muon_neutrino_0000/data/dev_lvl7_robustness_muon_neutrino_0000.db"

key_limits = {
    "bias": {
        "energy": {"x": [0, 3], "y": [-100, 100]},
        "zenith": {"x": [0, 3], "y": [-100, 100]},
    },
    "width": {
        "energy": {"x": [0, 3], "y": [-0.5, 1.5]},
        "zenith": {"x": [0, 3], "y": [-100, 100]},
    },
    "rel_imp": {"energy": {"x": [0, 3], "y": [-0.75, 0.75]}},
    "osc": {"energy": {"x": [0, 3], "y": [-0.75, 0.75]}},
    "distributions": {"energy": {"x": [0, 4], "y": [-0.75, 0.75]}},
}
keys = ["zenith"]
key_bins = {
    "energy": np.arange(0, 3.25, 0.25),
    "zenith": np.arange(0, 180, 10),
}

performance_figure = width_plot(
    key_limits,
    keys,
    key_bins,
    database,
    predictions_path,
    figsize=(10, 8),
    include_retro=True,
    track_cascade=True,
)

performance_figure.savefig("test_performance_figure.png")
