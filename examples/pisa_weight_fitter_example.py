from graphnet.pisa.fitting import WeightFitter

outdir = "/mnt/scratch/rasmus_orsoe/weight_test"
database_path = "/mnt/scratch/rasmus_orsoe/databases/dev_lvl3_genie_burnsample_v5/data/dev_lvl3_genie_burnsample_v5.db"
fitter = WeightFitter(database_path=database_path)

pisa_config_dict = {
    "reco_energy": {"num_bins": 8},
    "reco_coszen": {"num_bins": 8},
    "pid": {"bin_edges": [0, 0.5, 1]},
    "true_energy": {"num_bins": 200},
    "true_coszen": {"num_bins": 200},
    "livetime": 10
    * 0.01,  # set to 1% of 10 years - correspond to the size of the oscNext burn sample
}
# by calling fitter.fit_weights we get the weights returned pr. event. if add_to_database = True, a table will be added to the database
weights = fitter.fit_weights(
    outdir,
    add_to_database=True,
    weight_name="weight_livetime10_1percent",
    pisa_config_dict=pisa_config_dict,
)

print(weights)
