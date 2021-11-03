from abc import abstractmethod
from typing import Callable, Optional, final

import numpy as np
import torch
from torch.nn.modules.loss import _WeightedLoss
from torch import Tensor

class LossFunction(_WeightedLoss):
    """Base class for loss functions in gnn_reco.
    """
    def __init__(self, transform_output: Optional[Callable] = None, weight: Optional[Tensor] = None, size_average=None, reduce=None, reduction: str = 'mean'):
        super().__init__(weight, size_average, reduce, reduction)
        self._transform_output = transform_output if transform_output else lambda x: x
    
    @final
    def forward(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        prediction = self._transform_output(prediction)
        target = self._transform_output(target)
        return self._forward(prediction, target, return_elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        """Same syntax as `.forward` for implentation in inheriting classes."""
        pass
    

class LogCoshLoss(LossFunction):
    def _log_cosh(self, x: Tensor) -> Tensor:
        """Numerically stble version on log(cosh(x)).
        
        Used to avoid `inf` for even moderately large differences.        
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        assert prediction.dim() == target.dim() + 1
        diff = prediction[:,0] - target
        elements = self._log_cosh(diff)
        if return_elements:
            return elements
        return torch.mean(elements)

import scipy.special

class LogCMK(torch.autograd.Function):
    """MIT License

    Copyright (c) 2019 Max Ryabinin

    Permission is hereby granted, free of charge, to any person obtaining a copy
    of this software and associated documentation files (the "Software"), to deal
    in the Software without restriction, including without limitation the rights
    to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
    copies of the Software, and to permit persons to whom the Software is
    furnished to do so, subject to the following conditions:

    The above copyright notice and this permission notice shall be included in all
    copies or substantial portions of the Software.

    THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
    IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
    FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
    AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
    LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
    OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
    SOFTWARE.
    _____________________

    From [https://github.com/mryab/vmf_loss/blob/master/losses.py]
    Modified to use modified Bessel function instead of exponentially scaled ditto
    (i.e. `.ive` -> `.iv`) as indiciated in [1812.04616] in spite of suggestion in
    Sec. 8.2 of this paper. The change has been validated through comparison with
    exact calculations for `m=2` and `m=3` and found to yield the correct results.
    """
    @staticmethod
    def forward(ctx, m, k):
        dtype = k.dtype
        ctx.save_for_backward(k)
        ctx.m = m
        ctx.dtype = dtype
        k = k.double()
        iv = torch.from_numpy(scipy.special.iv(m / 2.0 - 1, k.cpu().numpy())).to(
            k.device
        )
        return (
            (m / 2.0 - 1) * torch.log(k) - torch.log(iv) - (m / 2) * np.log(2 * np.pi)
        ).type(dtype)

    @staticmethod
    def backward(ctx, grad_output):
        k = ctx.saved_tensors[0]
        m = ctx.m
        dtype = ctx.dtype
        k = k.double().cpu().numpy()
        grads = -((scipy.special.iv(m / 2.0, k)) / (scipy.special.iv(m / 2.0 - 1, k)))
        return (
            None,
            grad_output * torch.from_numpy(grads).to(grad_output.device).type(dtype),
        )

class VonMisesFisherLoss(LossFunction):
    @classmethod
    def log_cmk_exact(cls, m, k):
        return LogCMK.apply(m, k)

    @classmethod
    def log_cmk_approx(cls, m, k):
        # [https://arxiv.org/abs/1812.04616] Sec. 8.2 with additionaal minus signn
        v = m / 2. - 0.5
        a = torch.sqrt((v + 1)**2 + k**2)
        b = (v - 1)
        return -a + b * torch.log(b + a)

    def _evaluate(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        """Calculates the von Mises-Fisher loss for a vector in D-dimensonal space.

        This loss utilises the von Mises-Fisher distribution, which is a probability
        distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction (torch.Tensor): Predicted vector, of shape [batch_size, D].
            target (torch.Tensor): Target unit vector, of shape [batch_size, D].
            return_elements (bool): Whether to return the loss individual for each element/example.

        Returns:
            loss (torch.Tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
        """
        eps = torch.finfo(prediction.dtype).eps

        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()
   
        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk_exact(m, k) - dotprod

        if return_elements:
            return elements
        return torch.mean(elements)

class VonMisesFisher2DLoss(VonMisesFisherLoss):
    def _forward(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        """Calculates the von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction (torch.Tensor): Output of the model. Must have shape [batch_size, 2] where 0th column is a prediction of `angle` and 1st column is an estimate of `log(var(angle))`.
            target (torch.Tensor): Target tensor, extracted from graph object.
            return_elements (bool): Whether to return the loss individual for each element/example.

        Returns:
            loss (torch.Tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
        """
        # Check(s)
        assert prediction.dim() == 2 and prediction.size()[1] == 2
        assert target.dim() == 1
        assert prediction.size()[0] == target.size()[0]
        
        # Formatting target
        angle_true = target
        t = torch.stack([
            torch.cos(angle_true), 
            torch.sin(angle_true),
        ], dim=1)
        
        # Formatting prediction
        angle_pred = prediction[:,0]
        kappa = prediction[:,1]
        p = kappa.unsqueeze(1) * torch.stack([
            torch.cos(angle_pred),  
            torch.sin(angle_pred),
        ], dim=1)

        return self._evaluate(p, t, return_elements=return_elements)


class LegacyVonMisesFisherLoss(LossFunction):
    def _forward(self, prediction: Tensor, target: Tensor, return_elements: bool = False) -> Tensor:
        """Repesents a single angle (graph[target]) as a 3D vector (sine(angle), cosine(angle), 1) and calculates 
        the 3D VonMisesFisher loss of the angular difference between the 3D vector representations.

        Args:
            prediction (torch.tensor): Output of the model. Must have shape [batch_size, 3] where 0th column is a prediction of sine(angle) and 1st column is prediction of cosine(angle) and 2nd column is an estimate of Kappa.
            graph (Data-Object): Data-object with target stored in graph[target]
            target (str): name of the target. E.g. 'zenith' or 'azimuth'

        Returns:
            loss (torch.tensor): a batch level scalar quantity describing the VonMisesFischer loss. Shape [1,]
        """
        k = torch.abs(prediction[:,2])
        angle  = target  # graph[target].squeeze(1)
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
        
        if return_elements:
            return elements
        return torch.mean(elements)
