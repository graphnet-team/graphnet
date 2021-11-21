"""Collection of loss functions.

All loss functions inherit from `LossFunction` which (...)
"""

from abc import abstractmethod
from typing import Callable, Optional
try:
    from typing import final
except ImportError:  # Python version < 3.8
    final = lambda f: f  # Identity decorator

import numpy as np
import scipy.special
import torch
from torch import Tensor
from torch.nn.modules.loss import _WeightedLoss

class LossFunction(_WeightedLoss):
    """Base class for loss functions in gnn_reco.
    """
    def __init__(
        self,
        transform_output: Optional[Callable] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self._transform_output = transform_output if transform_output else lambda x: x

    @final
    def forward(
        self,
        prediction: Tensor,
        target: Tensor,
        return_elements: bool = False,
    ) -> Tensor:
        """Forward pass for all loss functions.
        Args:
            prediction (Tensor): Tensor containing predictions. Shape [N,P]
            target (Tensor): Tensor containing targets. Shape [N,T]
            return_elements (bool, optional): Whether elementwise loss terms
                should be returned. The alternative is to return the averaged
                loss across examples. Defaults to False.

        Returns:
            Tensor: Loss, either averaged to a scalar (if `return_elements = False`)
                or elementwise terms with shape [N,] (if `return_elements = True`).
        """
        prediction = self._transform_output(prediction)
        target = self._transform_output(target)

        elements = self._forward(prediction, target)
        assert elements.size(dim=0) == target.size(dim=0), \
            "`_forward` should return elementwise loss terms."

        return elements if return_elements else torch.mean(elements)

    @abstractmethod
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Syntax similar to `.forward` for implentation in inheriting classes."""


class LogCoshLoss(LossFunction):
    """Log-cosh loss function. 
    
    Acts like x^2 for small x; and like |x| for large x.
    """
    def _log_cosh(self, x: Tensor) -> Tensor:
        """Numerically stble version on log(cosh(x)).

        Used to avoid `inf` for even moderately large differences.
        See [https://github.com/keras-team/keras/blob/v2.6.0/keras/losses.py#L1580-L1617]
        """
        return x + torch.nn.functional.softplus(-2. * x) - np.log(2.0)

    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Implementation of loss calculation."""
        assert prediction.dim() == target.dim() + 1
        diff = prediction[:,0] - target
        elements = self._log_cosh(diff)
        return elements


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
        iv = torch.from_numpy(
            scipy.special.iv(m / 2.0 - 1, k.cpu().numpy())
        ).to(k.device)
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
    """General class for calculating von Mises-Fisher loss.

    Requires implementation for specific dimension `m` in which the target and
    prediction vectors need to be prepared.
    """
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

    def _evaluate(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculates the von Mises-Fisher loss for a vector in D-dimensonal space.

        This loss utilises the von Mises-Fisher distribution, which is a
        probability distribution on the (D - 1) sphere in D-dimensional space.

        Args:
            prediction (Tensor): Predicted vector, of shape [batch_size, D].
            target (Tensor): Target unit vector, of shape [batch_size, D].

        Returns:
            loss (Tensor): Elementwise von Mises-Fisher loss terms.
        """
        # Check(s)
        assert prediction.dim() == 2
        assert target.dim() == 2
        assert prediction.size() == target.size()

        # Computing loss
        m = target.size()[1]
        k = torch.norm(prediction, dim=1)
        dotprod = torch.sum(prediction * target, dim=1)
        elements = -self.log_cmk_exact(m, k) - dotprod
        return elements

class VonMisesFisher2DLoss(VonMisesFisherLoss):
    """von Mises-Fisher loss function vectors in the 2D plane."""
    def _forward(self, prediction: Tensor, target: Tensor) -> Tensor:
        """Calculates the von Mises-Fisher loss for an angle in the 2D plane.

        Args:
            prediction (Tensor): Output of the model. Must have shape [N, 2]
                where 0th column is a prediction of `angle` and 1st column is an
                estimate of `kappa`.
            target (Tensor): Target tensor, extracted from graph object.

        Returns:
            loss (Tensor): Elementwise von Mises-Fisher loss terms. Shape [N,]
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
        kappa = abs(prediction[:,1])
        p = kappa.unsqueeze(1) * torch.stack([
            torch.cos(angle_pred),
            torch.sin(angle_pred),
        ], dim=1)

        return self._evaluate(p, t)
