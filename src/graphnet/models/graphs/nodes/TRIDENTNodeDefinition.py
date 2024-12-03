import numpy as np
from typing import List
import torch
from torch_geometric.data import Data
from graphnet.models.graphs.nodes import NodeDefinition

class TRIDENTNodeDefinition(NodeDefinition):

    def __init__(
        self,
        output_feature_names: List[str] = [
            "nx","ny","nz","t1st","nhits","norm_xyz",
        ],
        keys: List[str] = [
            "sensor_pos_x",
            "sensor_pos_y",
            "sensor_pos_z",
            "t",
        ],
        id_columns: List[str] = ["sensor_pos_x", "sensor_pos_y", "sensor_pos_z"],
        time_column: str = "t",
    ) -> None:
        
        self._keys = keys
        self._input_feature_names = self._keys
        self.output_feature_names = output_feature_names
        super().__init__(input_feature_names=self._input_feature_names)

        self._id_columns = [self._keys.index(key) for key in id_columns]
        self._time_index = self._keys.index(time_column)

    def _define_output_feature_names(
        self, input_feature_names: List[str]
    ) -> List[str]:
        return self.output_feature_names

    def _construct_nodes(self, x: torch.Tensor) -> Data:
        print(f'Start. x shape: {x.shape}')
        x = x.numpy()

        if x.shape[0] == 0:
            num_features = 6
            zero_array = np.zeros((1, num_features))
            return Data(x=torch.tensor(zero_array, dtype=torch.float))
        
        charge_index: int = len(self._keys)
        norm_index: int = charge_index + 1
        # Add charge and norm columns
        x = np.insert(x, charge_index, np.zeros(x.shape[0]), axis=1)
        x = np.insert(x, norm_index, np.zeros(x.shape[0]), axis=1)

        # Sort by time, select origin as first hit dom
        x = x[x[:, self._time_index].argsort()]
        x[:, self._time_index] -= x[0, self._time_index]
        x[:,self._id_columns] -= x[0, self._id_columns]

        # Fill norm column
        x[:, norm_index] = np.linalg.norm(x[:, self._id_columns], axis=1)
        
        x[:, self._id_columns] /= x[:, norm_index].reshape(-1, 1).clip(min=1e-6)
        x[:, self._time_index] *= 0.2998

        # Fill charge column
        doms = x[:, self._id_columns]
        unique_values, inverse, dom_counts = np.unique(doms, return_inverse=True, return_counts=True, axis=0)
    
        x[:, charge_index] = dom_counts[inverse] #/ len(doms)
        
        # group doms and set time to median time
        x_= []
        for ids in unique_values:
            mask = np.where((x[:, self._id_columns] == ids).all(axis=1))[0]
            t_median = np.median(x[mask, self._time_index])
            x_.append([*ids, t_median, *x[mask[0], charge_index:]])
        
        x = np.array(x_)        
        #node: [nx, ny, nz, t-t0, nhits, norm]
        print(f'End. x shape: {x.shape}')
        print(f'Num hits: {x[:, 4].sum()}')       
        return Data(x=torch.tensor(x,dtype=torch.float))