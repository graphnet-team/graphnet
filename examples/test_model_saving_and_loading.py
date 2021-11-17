# Import(s)
import dill
import torch
from torch.utils.data import DataLoader
from torch_geometric.data import Batch

from gnn_reco.components.loss_functions import  VonMisesFisher2DLoss
from gnn_reco.data.sqlite_dataset import SQLiteDataset
from gnn_reco.data.constants import FEATURES, TRUTH
from gnn_reco.models import Model
from gnn_reco.models.detector.icecube86 import IceCube86
from gnn_reco.models.gnn import DynEdge
from gnn_reco.models.graph_builders import KNNGraphBuilder
from gnn_reco.models.task.reconstruction import AngularReconstructionWithKappa

# Load data
db = "/groups/icecube/leonbozi/datafromrasmus/GNNReco/data/databases/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3/data/dev_level7_noise_muon_nu_classification_pass2_fixedRetro_v3.db"
dataset = SQLiteDataset(db, "SRTTWOfflinePulsesDC", FEATURES.ICECUBE86, TRUTH.ICECUBE86)
dataloader = DataLoader(
    dataset,
    batch_size=4, 
    shuffle=False,
    num_workers=1, 
    collate_fn=Batch.from_data_list,
    persistent_workers=True,
    prefetch_factor=2,
)
batch = next(iter(dataloader))

# Wrap code in functions to make it clear that these two operations are wholly independent
model_path = "my_test_dir/test_model.pth"
state_dict_path = "my_test_dir/test_model_state_dict.pth"

def build_model():
    detector = IceCube86(
        graph_builder=KNNGraphBuilder(nb_nearest_neighbours=8),
    )
    gnn = DynEdge(
        nb_inputs=detector.nb_outputs,
    )
    task = AngularReconstructionWithKappa(
        hidden_size=gnn.nb_outputs, 
        target_label='zenith', 
        loss_function=VonMisesFisher2DLoss(),
    )
    model = Model(
        detector=detector,
        gnn=gnn,
        tasks=[task],
        device='cpu'
    )
    return model

def build_save_model():
    # Build model
    model = build_model()
    model.eval()
    
    # Print example
    print(model(batch))

    # Save, delete model
    model.save(model_path)
    del model

def load_model():
    # Load model
    model = Model.load(model_path)
    model.eval()

    # Print example -- should be identical to the output of `build_save_model` 
    print(model(batch))

def build_save_state_dict():
    # Build model
    model = build_model()
    model.eval()

    # Print example
    print(model(batch))

    # Save, delete model
    model.save_state_dict(state_dict_path)
    del model

def load_state_dict():
    # Build model
    model = build_model()
    model.eval()

    # Print example -- will be random
    print(model(batch))

    # Load state dict
    model.load_state_dict(state_dict_path)
    
    # Print example -- should be identical to the output of `build_save_state_dict` 
    print(model(batch))

# Run utility functions
build_save_model()
load_model()
print("-" * 40)
build_save_state_dict()
load_state_dict()
