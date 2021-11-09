from torch.nn import Module

class GNN(Module):         
    """Base class for all core GNN models in gnn_reco."""
    def __init__(self, nb_inputs, nb_outputs):
        # Base class constructor
        super().__init__()

        # Member variables
        self._nb_inputs = nb_inputs
        self._nb_outputs = nb_outputs

    @property
    def nb_inputs(self) -> int:
        """Number of inputs to GNN model."""
        return self._nb_inputs

    @property
    def nb_outputs(self) -> int:
        """Number of outputs from GNN model."""
        return self._nb_outputs
