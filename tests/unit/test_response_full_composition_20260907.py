import pytest
import torch

from experiments.response_full_composition_20260907 import two_pass


def test_affine_tangent_normal_gain_and_offset():
    matrix = torch.diag(torch.tensor([1., .2, .4], dtype=torch.float64))
    offset = torch.tensor([.1, .2, -.3], dtype=torch.float64)
    x = torch.zeros((2, 3), dtype=torch.float64)
    result = two_pass(lambda q: q@matrix.T+offset, x, .5)
    expected = (torch.eye(3, dtype=torch.float64)+matrix)@offset
    torch.testing.assert_close(result["posterior"], expected.expand_as(x))
    torch.testing.assert_close(result["response"], torch.full((2,), 1+.2**2+.4**2, dtype=x.dtype))
    torch.testing.assert_close(result["correction"], expected.square().sum().expand(2)/.5**2)
    assert result["jacobian_local_prediction_error_squared"].max() < 1e-25
    assert result["offset_local_prediction_error_over_lambda_squared"].max() < 1e-25


def test_nonlinear_second_jacobian_is_at_first_prediction():
    x = torch.tensor([[.3], [.7]], dtype=torch.float64)
    result = two_pass(lambda q: q.square(), x, .2)
    torch.testing.assert_close(result["posterior"], x**4)
    torch.testing.assert_close(result["jacobian"][:, 0, 0], 4*x[:, 0]**3)
    assert (result["jacobian"]-result["local_squared_jacobian"]).abs().min() > .01
    torch.testing.assert_close(result["correction"], ((x**4-x)/.2).square().sum(-1))


def test_projection_idempotence():
    x = torch.tensor([[1., 2.], [3., 4.]], dtype=torch.float64)
    result = two_pass(lambda q: torch.stack((q[:, 0], q[:, 1]*0), dim=-1), x, .5)
    torch.testing.assert_close(result["base_posterior"], result["posterior"])
    torch.testing.assert_close(result["base_response"], result["response"])
    torch.testing.assert_close(result["base_correction"], result["correction"])


def test_nonpositive_scale_rejected():
    with pytest.raises(ValueError):
        two_pass(lambda q: q, torch.zeros((2, 1), dtype=torch.float64), 0.)
