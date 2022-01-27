import torch
from torch import Tensor
from torch_cluster import radius_graph
from torch_geometric.data import Data


def cluster_identical(tensor: Tensor, eps: float = 1e-03) -> Tensor:
    """Cluster identical rows in `tensor`.

    Args:
        tensor (Tensor): 2D tensor.
        eps (float, optional): Radius-style threshold for clustering rows in
            `tensor`. Defaults to 1e-03.

    Returns:
        Tensor: 1D array of integers assigning each row in `tensor` to a cluter.
    """
    # Check(s)
    assert tensor.ndim == 2

    # By default assign each pulse to a separate DOM
    cluster_index = torch.arange(tensor.size(dim=0), device=tensor.device)

    # Connect all pulses within a radius of < eps, and assign them to the same DOM
    for pair in radius_graph(tensor, eps).T:
        cluster_index[pair[0]] = cluster_index[pair[1]]

    # Ensure that DOMs are squentially numbered from 0 to nb_clusters - 1
    cluster_index = torch.cumsum(torch.cat([torch.tensor([0], device=tensor.device), torch.diff(cluster_index).clip(max=1)]), dim=0)

    return cluster_index

def cluster_pulses_to_dom(data: Data, eps: float = 1e-03) -> Data:
    # Could also use `dom_number` and `string` -- but these only exist for
    # IceCubeUpgrade, not for <= DeepCore

    # Extract spatial coordinates for each pulse
    xyz = torch.stack((data['dom_x'], data['dom_y'], data['dom_z']), dim=1)

    # Add DOM clustering as an attribute to the `Data` object
    data.dom_index = cluster_identical(xyz)

    return data

def cluster_pulses_to_pmt(data: Data, eps: float = 1e-03) -> Data:
    # Could also use `pmt_number`, `dom_number`, and `string` -- but these only
    # exist for IceCubeUpgrade, not for <= DeepCore

    # Extract spatial coordinates for each pulse
    xyzdir = torch.stack((
            data['dom_x'],
            data['dom_y'],
            data['dom_z'],
            data['pmt_dir_x'],
            data['pmt_dir_y'],
            data['pmt_dir_z'],
        ), dim=1)

    # Add pmt clustering as an attribute to the `Data` object
    data.pmt_index = cluster_identical(xyzdir)

    return data