import pytest
import torch

from experiments.fm_mechanism_cross_residual import residual_gradient, relative_difference, radial_gain_gradient
from experiments.fm_mechanism_pilot import canonical_posterior


class LinearField(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor([[.5, -.1], [.2, .7]], dtype=torch.float64))
        self.bias = torch.nn.Parameter(torch.tensor([.1, -.2], dtype=torch.float64))

    def forward(self, x, log_scale):
        return x @ self.weight.T + self.bias


def inputs():
    x = torch.tensor([[1., 2.], [3., -1.], [-2., .5]], dtype=torch.float64)
    e = torch.tensor([[.1, -.2], [-.3, .4], [.5, .6]], dtype=torch.float64)
    return x, e, torch.tensor([True, False, True])


def test_donor_is_detached_and_vjp_matches_manual_native_scaling():
    field = LinearField()
    x, e, selected = inputs()
    e.requires_grad_(True)
    g = residual_gradient(field, x, e, .2, "rectified_linear", selected, batch_size=1)
    alpha = 1/1.2
    expected_weight = 2*e.detach()[selected].T @ (alpha*x[selected])/(x.numel()*.2**2)
    expected_bias = 2*e.detach()[selected].sum(0)/(x.numel()*.2**2)
    torch.testing.assert_close(g[0], expected_weight)
    torch.testing.assert_close(g[1], expected_bias)
    assert e.grad is None


def test_own_residual_matches_square_loss_gradient_and_partition_adds():
    field = LinearField()
    x, target, selected = inputs()
    prediction = canonical_posterior(field, x, .2, "vp_trigonometric")
    e = (prediction-target).detach()
    own = residual_gradient(field, x, e, .2, "vp_trigonometric", selected)
    field.zero_grad(set_to_none=True)
    prediction = canonical_posterior(field, x, .2, "vp_trigonometric")
    ((prediction-target).square().mean(-1).mul(selected).sum()/(len(x)*.2**2)).backward()
    assert relative_difference(own, [p.grad for p in field.parameters()]) < 1e-14
    rest = residual_gradient(field, x, e, .2, "vp_trigonometric", ~selected)
    all_g = residual_gradient(field, x, e, .2, "vp_trigonometric", torch.ones_like(selected))
    assert relative_difference([a+b for a,b in zip(own,rest)], all_g) < 1e-14


def test_zero_and_opposite_residual_controls():
    field = LinearField()
    x, e, selected = inputs()
    pos = residual_gradient(field, x, e, .2, "rectified_linear", selected)
    neg = residual_gradient(field, x, -e, .2, "rectified_linear", selected)
    zero = residual_gradient(field, x, torch.zeros_like(e), .2, "rectified_linear", selected)
    assert all(torch.equal(a, -b) for a,b in zip(pos,neg))
    assert all(torch.count_nonzero(a) == 0 for a in zero)


def test_invalid_shapes_and_mask_rejected():
    field = LinearField()
    x, e, selected = inputs()
    with pytest.raises(ValueError, match="shapes"):
        residual_gradient(field, x, e[:1], .2, "rectified_linear", selected)
    with pytest.raises(ValueError, match="boolean"):
        residual_gradient(field, x, e, .2, "rectified_linear", selected.double())


def test_projected_radial_gradient_ignores_normal_output_component():
    field = LinearField()
    x, _, _ = inputs()
    radial = x*torch.tensor([1., 0.], dtype=x.dtype)
    value, grad = radial_gain_gradient(field, x, .2, "rectified_linear", radial)
    torch.testing.assert_close(grad[0][1], torch.zeros(2, dtype=x.dtype))
    assert grad[1][1] == 0
    expected = float((canonical_posterior(field,x,.2,"rectified_linear")[:,0]*x[:,0]).sum()/x[:,0].square().sum())
    assert value == expected
