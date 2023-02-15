#!/bin/bash

#### This script enables the user to run multiple trainings in sequence on the same database but for different model configs.
# To execute this file, copy the file path and write in the terminal; $ bash <filepath>


# execution of bash file in same directory as the script
bash_directory=$(dirname -- "$(readlink -f "${BASH_SOURCE}")")

## Global; applies to all models
# path to dataset configuration file in the GraphNeT directory
dataset_config=$(realpath "$bash_directory/../../configs/datasets/training_example_data_sqlite.yml")
# what GPU to use; more information can be gained with the module nvitop
gpus=0
# the maximum number of epochs; if used, this greatly affect learning rate scheduling
max_epochs=5
# early stopping threshold
early_stopping_patience=5
# events in a batch
batch_size=16
# number of CPUs to use
num_workers=2

## Model dependent; applies to each model in sequence
# path to model files in the GraphNeT directory
model_directory=$(realpath "$bash_directory/../../configs/models")
# list of model configurations to train
declare -a model_configs=(
    "${model_directory}/example_direction_reconstruction_model.yml"
    "${model_directory}/example_energy_reconstruction_model.yml"
    "${model_directory}/example_vertex_position_reconstruction_model.yml"
)

# suffix ending on the created directory
declare -a suffixs=(
    "direction"
    "energy"
    "position"
)

# prediction name outputs per model
declare -a prediction_names=(
    "zenith_pred zenith_kappa_pred azimuth_pred azimuth_kappa_pred"
    "energy_pred"
    "position_x_pred position_y_pred position_z_pred"
)

for i in "${!model_configs[@]}"; do
    echo "training iteration ${i} on ${model_configs[$i]} with output variables ${prediction_names[i][@]}"
    python ${bash_directory}/01_train_model.py \
        --dataset-config ${dataset_config} \
        --model-config ${model_configs[$i]} \
        --gpus ${gpus} \
        --max-epochs ${max_epochs} \
        --early-stopping-patience ${early_stopping_patience} \
        --batch-size ${batch_size} \
        --num-workers ${num_workers} \
        --prediction-names ${prediction_names[i][@]} \
        --suffix ${suffixs[i]}
    wait
done
echo "all trainings are done."