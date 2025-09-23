"""Unit tests for LossFunction classes."""

import numpy as np
import pytest
import torch
import warnings
from torch import Tensor
from torch.autograd import grad

from graphnet.training.loss_functions import LogCoshLoss, VonMisesFisherLoss
from graphnet.utilities.maths import eps_like


# Utility method(s)
def _compute_elementwise_gradient(outputs: Tensor, inputs: Tensor) -> Tensor:
    """Compute gradient of each element in `outptus` wrt. `inputs`.

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
    elementwise_gradients = torch.stack(
        [
            grad(
                outputs=outputs[ix],
                inputs=inputs,
                retain_graph=True,
            )[
                0
            ][ix]
            for ix in range(nb_elements)
        ]
    )
    return elementwise_gradients


# Unit test(s)
def test_log_cosh(dtype: torch.dtype = torch.float32) -> None:
    """Test agreement of the two ways to calculate this loss."""
    # Prepare test data
    x = torch.tensor([-100, -10, -1, 0, 1, 10, 100], dtype=dtype).unsqueeze(
        1
    )  # Shape [N, 1]
    y = 0.0 * x.clone()  # Shape [N,1]
    # Calculate losses using loss function, and manually
    log_cosh_loss = LogCoshLoss()
    losses = log_cosh_loss(x, y, return_elements=True)
    losses_reference = torch.log(torch.cosh(x - y))

    # (1) Loss functions should not return  `inf` losses, even for large
    #     differences between prediction and target. This is not necessarily
    #     true for the directly calculated loss (reference) where cosh(x)
    #     may go to `inf` for x >~ 100.
    assert torch.all(torch.isfinite(losses))

    # (2) For the inputs where the reference loss _is_ valid, the two
    #     calculations should agree exactly.
    reference_is_valid = torch.isfinite(losses_reference)
    assert torch.allclose(
        losses_reference[reference_is_valid], losses[reference_is_valid]
    )


def test_von_mises_fisher_exact_m3(dtype: torch.dtype = torch.float64) -> None:
    """Test implementaion of exact von-Mises Fisher loss with m=3.

    See
    https://en.wikipedia.org/wiki/Von_Mises%E2%80%93Fisher_distribution
    for exact, simplified reference.
    """
    # Define test parameters
    m = 3
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_reference = (
        torch.log(k) - k - torch.log(2 * np.pi * (1 - torch.exp(-2 * k)))
    )
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    # Compute gradients
    grads_reference = _compute_elementwise_gradient(res_reference, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test that values agree
    assert torch.allclose(res_exact, res_reference)

    # Test that gradients agree
    assert torch.allclose(grads_reference, grads_exact)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[0.0001, 0.001, 0.01, 0.1, 1.0, 3.0, 10.0, 30.0, 100.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = (
        res_exact[0] - res_approx[0]
    )  # Normalisation constant from integrating gradient
    res_approx += C - eps_like(C)

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    # Test inequality in [1812.04616] Sec. 8.2
    assert torch.all(res_exact >= res_approx), (m, res_exact, res_approx)

    # Test value approximation
    assert torch.allclose(res_approx, res_exact, rtol=1e0, atol=1e-01)

    # Test gradient approximation
    assert torch.allclose(grads_approx, grads_exact, rtol=1e0)


@pytest.mark.parametrize("m", [2, 3])
def test_von_mises_fisher_approximation_large_kappa(
    m: int, dtype: torch.dtype = torch.float64
) -> None:
    """Test approximate calculation of $log C_{m}(k)$ for large kappa values.

    See [1812.04616], Section 8.2 for approximation.
    """
    # Check(s)
    assert isinstance(m, int)
    assert m > 1

    # Define test parameters
    k = torch.tensor(
        data=[100.0, 200.0, 300.0, 500.0, 1000.0],
        requires_grad=True,
        dtype=dtype,
    )

    # Compute values
    res_approx = VonMisesFisherLoss.log_cmk_approx(m, k)
    res_exact = VonMisesFisherLoss.log_cmk_exact(m, k)

    C = res_exact[0] - res_approx[0]  # Normalisation constant
    res_approx += C

    # Compute gradients
    grads_approx = _compute_elementwise_gradient(res_approx, k)
    grads_exact = _compute_elementwise_gradient(res_exact, k)

    exact_is_valid = torch.isfinite(res_exact)

    # Test value approximation
    assert torch.allclose(
        res_approx[exact_is_valid], res_exact[exact_is_valid], rtol=1e-2
    )

    # Test gradient approximation
    assert torch.allclose(
        grads_approx[exact_is_valid], grads_exact[exact_is_valid], rtol=1e-2
    )


def test_logcmk_backward_zero_handling(dtype: torch.dtype = torch.float64) -> None:
    """Test LogCMK backward pass handles arrays with zero values correctly.
    
    This test ensures that the LogCMK.backward method correctly handles cases
    where the kappa tensor contains zero values without raising division by zero
    errors or warnings. The implementation uses boolean masking to conditionally
    apply different formulas for small (including zero) and large kappa values,
    avoiding double evaluation that would cause RuntimeWarnings.
    
    Args:
        dtype: PyTorch data type for the test tensors.
    """
    # Test parameters
    m = 3  # Dimension for which we have the exact formula
    
    # Create kappa tensor with zeros and other values, including edge cases
    kappa_values = [0.0, 1e-7, 1e-6, 1e-5, 0.1, 1.0, 10.0]
    kappa = torch.tensor(kappa_values, dtype=dtype, requires_grad=True)
    
    # Forward pass using VonMisesFisherLoss.log_cmk_exact which internally uses LogCMK
    result = VonMisesFisherLoss.log_cmk_exact(m, kappa)
    
    # Test that backward pass doesn't raise any errors or warnings
    # Capture warnings to ensure no RuntimeWarnings are generated
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        try:
            grads = torch.autograd.grad(
                outputs=result.sum(),
                inputs=kappa,
                grad_outputs=None,
                retain_graph=False,
                create_graph=False,
            )[0]
            backward_success = True
            error_msg = ""
        except (ZeroDivisionError, RuntimeWarning) as e:
            backward_success = False
            error_msg = str(e)
    
    # Verify no errors occurred
    assert backward_success, f"Backward pass failed with error: {error_msg}"
    
    # Verify no RuntimeWarnings were generated
    runtime_warnings = [w for w in caught_warnings if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings) == 0, f"RuntimeWarnings were generated: {[str(w.message) for w in runtime_warnings]}"
    
    # Verify gradients are finite
    assert torch.all(torch.isfinite(grads)), "Gradients should be finite for all kappa values"
    
    # Test specific values for correctness
    # For kappa=0, the gradient should be -kappa/3 = 0
    zero_idx = 0  # Index where kappa=0
    assert torch.isclose(grads[zero_idx], torch.tensor(0.0, dtype=dtype)), \
        f"Gradient at kappa=0 should be 0, got {grads[zero_idx]}"
    
    # For very small kappa (1e-7), should use -kappa/3 approximation
    small_kappa_idx = 1  # Index where kappa=1e-7
    expected_small_grad = -kappa_values[small_kappa_idx] / 3
    assert torch.isclose(grads[small_kappa_idx], torch.tensor(expected_small_grad, dtype=dtype), atol=1e-10), \
        "Gradient for small kappa should use -kappa/3 approximation"
    
    # Test with array containing multiple zeros
    kappa_multi_zero = torch.tensor([0.0, 0.0, 1.0, 0.0, 10.0], dtype=dtype, requires_grad=True)
    result_multi = VonMisesFisherLoss.log_cmk_exact(m, kappa_multi_zero)
    
    with warnings.catch_warnings(record=True) as caught_warnings_multi:
        warnings.simplefilter("always")
        try:
            grads_multi = torch.autograd.grad(
                outputs=result_multi.sum(),
                inputs=kappa_multi_zero,
                grad_outputs=None,
            )[0]
            multi_zero_success = True
        except (ZeroDivisionError, RuntimeWarning):
            multi_zero_success = False
    
    assert multi_zero_success, "Should handle arrays with multiple zero values"
    assert torch.all(torch.isfinite(grads_multi)), "All gradients should be finite with multiple zeros"
    
    # Verify no RuntimeWarnings for multiple zeros case
    runtime_warnings_multi = [w for w in caught_warnings_multi if issubclass(w.category, RuntimeWarning)]
    assert len(runtime_warnings_multi) == 0, f"RuntimeWarnings were generated with multiple zeros: {[str(w.message) for w in runtime_warnings_multi]}"
    
    # Verify that zero elements have zero gradients
    zero_mask = kappa_multi_zero == 0.0
    assert torch.all(grads_multi[zero_mask] == 0.0), "Zero kappa values should have zero gradients"
