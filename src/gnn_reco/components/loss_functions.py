import torch
import numpy as np
from torch.nn import CrossEntropyLoss


def LogCosh(prediction, graph, target):
    return torch.sum(torch.log(torch.cosh(((prediction[:,0]-graph[target])))))

def CustomCrossEntropyLoss(prediction, graph, target):
    f = CrossEntropyLoss()
    return f(prediction,graph[target].long())


def VonMisesSineCosineLoss(prediction, graph, target):
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
