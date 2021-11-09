
from typing import Callable, Optional
import numpy as np
import torch
from torch.functional import Tensor

from gnn_reco.models.task.task import Task
from gnn_reco.components.loss_functions import LossFunction

class LegacyAngularReconstruction(Task):
    # Requires three features: untransformed points in (x,y)-space, and "kappa", respectively.
    nb_inputs = 3
    def __init__(
        self, 
        hidden_size: int, 
        target_label: str, 
        loss_function: LossFunction, 
        target_scaler: Optional[Callable] = None,
    ):
        super().__init__(hidden_size, target_label, loss_function)
        self.inference = False
        self.target_scaler = target_scaler

    def _forward(self, x):
        if self.inference:
            pred = torch.atan2(x[:,0], x[:,1]).unsqueeze(1)
            if self.target_scaler is not None:
                pred = torch.tensor(self.target_scaler.inverse_transform(pred.detach().cpu().numpy()), dtype=torch.float32).to(pred.device) 
            sigma = torch.sqrt(torch.abs(1 / x[:,2])).unsqueeze(1)
            return torch.cat((pred, sigma), dim=1)
        
        x[:,0] = torch.tanh(x[:,0])
        x[:,1] = torch.tanh(x[:,1])
        return x


class LegacyVonMisesFisherLoss(LossFunction):
    def __init__(self,
        transform_output: Optional[Callable] = None,
        target_scaler: Optional[Callable] = None,
        **kwargs):
        super().__init__(transform_output, **kwargs)
        self.target_scaler = target_scaler

    def _forward(
        self, 
        prediction: Tensor,
        target: Tensor,
    ) -> Tensor:
        """Original function for computing von Mises-Fisher loss.

        Repesents a single angle (graph[target]) as a 3D vector (sine(angle),
        cosine(angle), 1) and calculates the 3D VonMisesFisher loss of the
        angular difference between the 3D vector representations.

        Args:
            prediction (Tensor): Output of the model. Must have shape [N, 3]
                where 0th column is a prediction of sine(angle) and 1st column
                is prediction of cosine(angle) and 2nd column is an estimate of
                kappa.
            target (Tensor): Target tensor, extracted from graph object.

        Returns:
            loss (Tensor): Elementwise von Mises-Fisher loss terms. Shape [N,]
        """
        k = torch.abs(prediction[:,2])
        angle  = target  # Original: graph[target].squeeze(1)
        if self.target_scaler is not None:
            angle = torch.tensor(self.target_scaler.transform(angle.unsqueeze(1).detach().cpu().numpy())).squeeze(1).to(angle.device)

        u_1 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.sin(angle)
        u_2 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*torch.cos(angle)
        u_3 = (1/torch.sqrt(torch.tensor(2,dtype = torch.float)))*(1)

        norm_x  = torch.sqrt(1 + prediction[:,0]**2 + prediction[:,1]**2)

        x_1 = (1/norm_x)*prediction[:,0]
        x_2 = (1/norm_x)*prediction[:,1]
        x_3 = (1/norm_x)*(1)

        dotprod = u_1*x_1 + u_2*x_2 + u_3*x_3
        logc_3 = - torch.log(k) + k + torch.log(1 - torch.exp(-2*k))
        elements = -k*dotprod + logc_3

        return elements
        