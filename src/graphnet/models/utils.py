from torch_geometric.utils.homophily import homophily

def calculate_xyzt_homophily(x, edge_index, batch):
    """Calculates xyzt homophily from a batch of graphs.

    Homophily is a graph scalar quantity that measures the likeness of variables
    in nodes. Notice that this calculator assumes a special order of input
    features in x.

    Returns:
        tuple : tuple of torch.tensor each with shape [batch_size,1]
    """
    hx = homophily(edge_index, x[:,0], batch).reshape(-1,1)
    hy = homophily(edge_index, x[:,1], batch).reshape(-1,1)
    hz = homophily(edge_index, x[:,2], batch).reshape(-1,1)
    ht = homophily(edge_index, x[:,3], batch).reshape(-1,1)
    return hx, hy, hz, ht
