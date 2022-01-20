"""Unit tests for loss functions."""

import numpy as np
import pytest
import torch
from torch import Tensor
from torch.autograd import grad

from graphnet.components.loss_functions import LogCoshLoss, VonMisesFisherLoss, BinaryCrossEntropyLoss
from graphnet.utils import eps_like

# Utility method(s)
def _compute_elementwise_gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Computes  gradient of each element in `outptus` wrt. `inputs`.

    It is assumed that each element in `inputs` only affects the corresponding
    element in `outputs`. This should be the result of any vectorised
    calculation (as used in tests).
    """
    # Check(s)
    assert inputs.dim() == 1
    assert outputs.dim() == 1
    assert inputs.size(dim=0) == outputs.size(dim=0)

    # Compute list of elementwise gradients
    nb_elements = inputs.size(dim=0)
    elementwise_gradients = torch.stack([
        grad(
            outputs=outputs[ix],
            inputs=inputs,
            retain_graph=True,
        )[0][ix] for ix in range(nb_elements)
    ])
    return elementwise_gradients


# Unit test(s)
def test_log_cosh(dtype=torch.float32):
    # Prepare test data
    x = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=dtype).unsqueeze(1)  # Shape [N, 1]
    y = 0. * x.clone().squeeze()  # Shape [N,]

    # Calculate losses using loss function, and manually
    log_cosh_loss = LogCoshLoss()
    losses = log_cosh_loss(x, y, return_elements=True)
    losses_reference = torch.log(torch.cosh(x[:,0] - y))

    # (1) Loss functions should not return  `inf` losses, even for large
    #     differences between prediction and target. This is not necessarily
    #     true for the directly calculated loss (reference) where cosh(x) may go
    #     to `inf` for x >~ 100.
    assert torch.all(torch.isfinite(losses))

    # (2) For the inputs where the reference loss _is_ valid, the two
    #     calculations should agree exactly.
    reference_is_valid = torch.isfinite(losses_reference)
    assert torch.allclose(losses_reference[reference_is_valid], losses[reference_is_valid])


def test_log_cosh_of_log_transformed(dtype=torch.float32):
    # Prepare test data
    x = torch.tensor([1, 10, 100, 1000, 10000], dtype=dtype).unsqueeze(1)  # Shape [N, 1]
    y = 0.5 * x.clone().squeeze()  # Shape [N,]

    log_cosh_loss = LogCoshLoss()
    log_cosh_of_log_transformed_loss = LogCoshLoss(transform_prediction_and_target=lambda x: torch.log10(x))
    assert torch.allclose(
        log_cosh_loss(torch.log10(x), torch.log10(y), return_elements=True),
        log_cosh_of_log_transformed_loss(x, y, return_elements=True),
    )


def test_von_mises_fisher_exact_m3(dtype=torch.float64):
    """
    See https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    for exact, simplified reference.
    """
    # Define test parameters
    m = 3
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1., 3., 10., 30., 100.],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_reference = torch.log(k) - k - torch.log(2 * np.pi * (1 - torch.exp(-2 * k)))
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    # Compute gradients
    grads_reference = _compute_elementwise_gradient(res_reference, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test that values agree
    assert torch.allclose(res_exact, res_reference)

    # Test that gradients agree
    assert torch.allclose(grads_reference, grads_exact)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation(m, dtype=torch.float64):
    """See [1812.04616] Sec. 8.2 for approximation"""
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1., 3., 10., 30., 100.],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = res_exact[0] - res_approx[0] # Normalisation constant from integrating gradient
    res_approx += C - eps_like(C)

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test inequality in [1812.04616] Sec. 8.2
    assert torch.all(res_exact >= res_approx), (m, res_exact, res_approx)

    # Test value approximation
    assert torch.allclose(res_approx, res_exact, rtol=1e+0, atol=1e-01)

    # Test gradient approximation
    assert torch.allclose(grads_approx, grads_exact, rtol=1e+0)

@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation_large_kappa(m, dtype=torch.float64):
    """See [1812.04616] Sec. 8.2 for approximation"""
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[100., 200., 300., 500., 1000.],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = res_exact[0] - res_approx[0] # Normalisation constant
    res_approx += C

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    exact_is_valid = torch.isfinite(res_exact)

    # Test value approximation
    assert torch.allclose(res_approx[exact_is_valid], res_exact[exact_is_valid], rtol=1e-2)

    # Test gradient approximation
    assert torch.allclose(grads_approx[exact_is_valid], grads_exact[exact_is_valid], rtol=1e-2)

def test_binary_cross_entropy(dtype=torch.float32):
    # Prepare test data
    n=10
    x = torch.rand(n, dtype=dtype).unsqueeze(1)  # Shape [N, 1]
    y = torch.rand(n, dtype=dtype)  # Shape [N,]

    # Calculate losses using loss function, and manually
    binary_cross_entropy_loss = BinaryCrossEntropyLoss()
    losses = binary_cross_entropy_loss(x, y, return_elements=True)
    losses_reference = torch.nn.functional.binary_cross_entropy(x[:,0], y, reduction='none')

    assert torch.allclose(losses_reference, losses)
