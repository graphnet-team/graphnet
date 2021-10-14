from torch_geometric.utils.homophily import homophily
def calculate_xyzt_homophily(x,edge_index, batch):
    hx = homophily(edge_index,x[:,0],batch).reshape(-1,1)
    hy = homophily(edge_index,x[:,1],batch).reshape(-1,1)
    hz = homophily(edge_index,x[:,2],batch).reshape(-1,1)
    ht = homophily(edge_index,x[:,3],batch).reshape(-1,1)
    return hx,hy,hz,ht