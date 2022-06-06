from graphnet.pisa.utils import ContourFitter

# This configuration dictionary overwrites our pisa standard with your preferences.
# note: num_bins should not be higer than 25 for reconstructions.
config_dict = {
    "reco_energy": {"num_bins": 10},
    "reco_coszen": {"num_bins": 10},
    "pid": {"bin_edges": [0, 0.30, 0.90, 1]},
    "true_energy": {"num_bins": 200},
    "true_coszen": {"num_bins": 200},
}

outdir = "/home/iwsatlas1/oersoe/phd/oscillations/sensitivities"  # where you want the .csv-file with the results
run_name = "test_this_Andreas"  # what you call your run
pipeline_path = "/mnt/scratch/rasmus_orsoe/databases/oscillations/dev_lvl7_robustness_muon_neutrino_0000/pipelines/pipeline_oscillation_final/pipeline_oscillation_final.db"

fitter = ContourFitter(
    outdir=outdir,
    pipeline_path=pipeline_path,
    post_fix="_pred",
    model_name="dynedge",
    include_retro=True,
    statistical_fit=False,
)

fitter.fit_1d_contour(
    run_name=run_name + "_1d",
    config_dict=config_dict,
    grid_size=5,
    n_workers=30,
)

fitter.fit_2d_contour(
    run_name=run_name + "_2d",
    config_dict=config_dict,
    grid_size=5,
    n_workers=30,
)
