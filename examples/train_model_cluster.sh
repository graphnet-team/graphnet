#!/bin/bash

# conda work environment activated
source /groups/icecube/${USER}/.bashrc
cd /groups/icecube/${USER}/graphnet_user/env/
conda activate gnn_py38_leon

cd /groups/icecube/${USER}/graphnet/examples/
python train_model.py

conda deactivate
cd /groups/icecube/${USER}/graphnet_user/