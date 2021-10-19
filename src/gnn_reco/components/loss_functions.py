import torch
import numpy as np
from torch.nn import CrossEntropyLoss


def log_cosh(prediction, graph, target):
    return torch.sum(torch.log(torch.cosh(((prediction[:,0]-graph[target].squeeze(1))))))

def custom_crossentropy_loss(prediction, graph, target):
    f = CrossEntropyLoss()
    return f(prediction,graph[target].squeeze(1).long())


def vonmises_sinecosine_loss(prediction, graph, target):
    """Repesents a single angle (graph[target]) as a 3D vector (sine(angle), cosine(angle), 1) and calculates 
    the 3D VonMisesFisher loss of the angular difference between the 3D vector representations.

    Args:
        prediction (torch.tensor): Output of the model. Must have shape [batch_size, 3] where 0th column is a prediction of sine(angle) and 1st column is prediction of cosine(angle) and 2nd column is an estimate of Kappa.
        graph (Data-Object): Data-object with target stored in graph[target]
        target (str): name of the target. E.g. 'zenith' or 'azimuth'

    Returns:
        loss (torch.tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
    """
    k            = torch.abs(prediction[:,2])
    angle  = graph[target].squeeze(1)
    u_1 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.sin(angle)
    u_2 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.cos(angle)
    u_3 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*(1)
    
    norm_x  = torch.sqrt(1 + prediction[:,0]**2 + prediction[:,1]**2)
    
    x_1 = (1/norm_x)*prediction[:,0]
    x_2 = (1/norm_x)*prediction[:,1]
    x_3 = (1/norm_x)*(1)
    
    dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
    logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))    
    loss = torch.mean(-k*dotprod + logc_3)
    return loss
