# Training examples

This subfolder contains two main training scripts:

**`01_train_model.py`** Shows how to train a GNN on neutrino telescope data **using configuration files** to construct the dataset that loads the data and the model that is trained. This is the recommended way to configure standard dataset and models, as it is easier to ready and share that doing so in pure code. This example can be run using a few different models targeting different physics use cases. For instance, you can try running:

```bash
# Show the CLI
(graphnet) $ python examples/04_training/01_train_model.py --help

# Train energy regression model
(graphnet) $ python examples/04_training/01_train_model.py

# Same as above, as this is the default model config.
(graphnet) $ python examples/04_training/01_train_model.py \
    --model-config configs/models/example_energy_reconstruction_model.yml

# Train using a single GPU
(graphnet) $ python examples/04_training/01_train_model.py --gpus 0

# Train using multiple GPUs
(graphnet) $ python examples/04_training/01_train_model.py --gpus 0 1

# Train a vertex position reconstruction model
(graphnet) $ python examples/04_training/01_train_model.py \
    --model-config configs/models/example_vertex_position_reconstruction_model.yml

# Trains a direction (zenith, azimuth) reconstruction model. Note that the
# chosen `Task` in the model config file also returns estemated "kappa" values,
# i.e. inverse variance, for each predicted feature, meaning that we need to
# manually specify the names of these.
(graphnet) $ python examples/04_training/01_train_model.py --gpus 0 \
    --model-config configs/models/example_direction_reconstruction_model.yml  \
    --prediction-names zenith_pred zenith_kappa_pred azimuth_pred azimuth_kappa_pred
```

**`02_train_model_without_configs.py`** Shows how to train a GNN on neutrino telescope data **without configuration files,** i.e., by programatically construct the dataset and model used trained. This is good for debugging and experimenting with different dataset settings and models configurations, as it is easier to build the model using the API than by writing configuration files from scratch. For instance, try running:

```bash
# Show the CLI
(graphnet) $ python examples/04_training/02_train_model_without_configs.py --help

# Train energy regression model
(graphnet) $ python examples/04_training/02_train_model_without_configs.py
```