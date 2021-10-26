import numpy as np

import torch
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

class LossFunction(_WeightedLoss):
    """Base class for loss functions in gnn_reco.
    """
    pass

class LogCosh(LossFunction):
    def _log_cosh(self, x: Tensor) -> Tensor:
        """Used to avoid `inf` for even moderately large differences.
        
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)

    def forward(self, input: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        diff = input[:,0] - target
        elements = self._log_cosh(diff)
        if return_elements:
            return elements
        return torch.mean(elements)

class LogCoshOfLogTransformed(LogCosh):
    def forward(self, input: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        input = torch.log10(input)
        target = torch.log10(target)
        return super().forward(input, target, return_elements)

class VonMisesFisherLoss(LossFunction):
    def forward(self, input: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        """Repesents a single angle (graph[target]) as a 3D vector (sine(angle), cosine(angle), 1) and calculates 
        the 3D VonMisesFisher loss of the angular difference between the 3D vector representations.

        Args:
            prediction (torch.Tensor): Output of the model. Must have shape [batch_size, 2] where 0th column is a prediction of `angle` and 1st column is an estimate of `log(var(angle))`.
            target (torch.Tensor): Target tensor, extracted from graph object.
            return_elements (bool): Whether to return the loss individual for each element/example.

        Returns:
            loss (torch.Tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
        """
        # Check(s)
        assert input.dim() == 2 and input.size()[1] == 2
        assert target.dim() == 1
        assert input.size()[0] == target.size()[0]

        # Formatting target
        angle_true = target
        norm_u = 1. / torch.sqrt(torch.tensor(2, dtype=torch.float))
        u_1 = norm_u * torch.sin(angle_true)
        u_2 = norm_u * torch.cos(angle_true)
        u_3 = norm_u
        
        # Formatting prediction
        angle_pred = input[:,0]
        log_var = input[:,1]
        kappa = 1/torch.exp(log_var)
        cos = torch.cos(angle_pred)
        sin = torch.sin(angle_pred)
        norm_x  = 1. / torch.sqrt(1 + cos**2 + sin**2)
        
        x_1 = norm_x * cos
        x_2 = norm_x * sin
        x_3 = norm_x
        
        # Computing loss
        dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
        logc_3 = - torch.log(kappa) + kappa + torch.log(1 - torch.exp(-2*kappa))    
        elements = -kappa*dotprod + logc_3

        if return_elements:
            return elements
        return torch.mean(elements)

#def log_cosh(prediction, graph, target):
#    return torch.sum(torch.log(torch.cosh(((prediction[:,0]-graph[target].squeeze(1))))))
#
#def custom_crossentropy_loss(prediction, graph, target):
#    f = CrossEntropyLoss()
#    return f(prediction,graph[target].squeeze(1).long())
#
#def vonmises_sinecosine_loss(prediction, graph, target):
#    """Repesents a single angle (graph[target]) as a 3D vector (sine(angle), cosine(angle), 1) and calculates 
#    the 3D VonMisesFisher loss of the angular difference between the 3D vector representations.
#
#    Args:
#        prediction (torch.tensor): Output of the model. Must have shape [batch_size, 3] where 0th column is a prediction of sine(angle) and 1st column is prediction of cosine(angle) and 2nd column is an estimate of Kappa.
#        graph (Data-Object): Data-object with target stored in graph[target]
#        target (str): name of the target. E.g. 'zenith' or 'azimuth'
#
#    Returns:
#        loss (torch.tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
#    """
#    k = torch.abs(prediction[:,2])
#    angle  = graph[target].squeeze(1)
#    u_1 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.sin(angle)
#    u_2 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.cos(angle)
#    u_3 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*(1)
#    
#    norm_x  = torch.sqrt(1 + prediction[:,0]**2 + prediction[:,1]**2)
#    
#    x_1 = (1/norm_x)*prediction[:,0]
#    x_2 = (1/norm_x)*prediction[:,1]
#    x_3 = (1/norm_x)*(1)
#    
#    dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
#    logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))    
#    loss = torch.mean(-k*dotprod + logc_3)
#    return loss
